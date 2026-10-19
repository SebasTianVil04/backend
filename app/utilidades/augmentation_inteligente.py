from collections import Counter
from typing import Dict, Any, List

from torch.utils.data import WeightedRandomSampler

from app.servicios.config_tipo_senas import (
    obtener_confusiones_comunes,
    obtener_grupo_confusion,
    es_sena_emergencia,
)


def _severidad_confusion(sena: str) -> float:
    confusiones = obtener_confusiones_comunes(sena)
    grupo = obtener_grupo_confusion(sena)
    severidad = len(confusiones)
    if grupo:
        severidad += len(grupo) - 1
    return severidad


def obtener_configuracion_sampler(clases: List[str]) -> Dict[str, Any]:
    numeros = [c for c in clases if c.isdigit()]
    letras = [c for c in clases if len(c) == 1 and c.isalpha()]
    palabras = [c for c in clases if c not in numeros and c not in letras]
    severidades = {c: _severidad_confusion(c) for c in clases}
    severidad_maxima = max(severidades.values()) if severidades else 0
    return {
        'total_numeros': len(numeros),
        'total_letras': len(letras),
        'total_palabras': len(palabras),
        'severidades': severidades,
        'severidad_maxima': severidad_maxima,
    }


def _multiplicador_oversampling(sena: str, config_sampler: Dict[str, Any], total_clases: int) -> float:
    severidad = config_sampler['severidades'].get(sena, 0)
    severidad_maxima = config_sampler['severidad_maxima']

    if es_sena_emergencia(sena):
        base = 4.0
    elif sena.isdigit():
        base = 3.0
    elif len(sena) == 1 and sena.isalpha():
        base = 1.5
    else:
        base = 1.1

    if severidad_maxima > 0:
        factor = 1.0 + (severidad / severidad_maxima) * 1.3
    else:
        factor = 1.0

    if total_clases <= 6 and severidad > 0:
        factor *= 1.3

    return base * factor


def crear_sampler_balanceado(train_labels: List[int], train_senas: List[str], clases: List[str]):
    class_counts = Counter(train_labels)
    config_sampler = obtener_configuracion_sampler(clases)

    sample_weights = []
    for label in train_labels:
        if label >= len(clases):
            sample_weights.append(1.0)
            continue
        sena = clases[label]
        peso = (1.0 / class_counts[label]) * _multiplicador_oversampling(sena, config_sampler, len(clases))
        sample_weights.append(peso)

    total_samples = int(len(sample_weights) * 1.5)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True,
    )


def obtener_parametros_entrenamiento_optimizados(clases: List[str]) -> Dict[str, Any]:
    numeros = [c for c in clases if c.isdigit()]
    letras = [c for c in clases if len(c) == 1 and c.isalpha()]
    palabras = [c for c in clases if c not in numeros and c not in letras]

    total = len(clases)
    prop_numeros = len(numeros) / total if total > 0 else 0
    prop_letras = len(letras) / total if total > 0 else 0
    prop_palabras = len(palabras) / total if total > 0 else 0

    severidades = [_severidad_confusion(c) for c in clases]
    severidad_promedio = sum(severidades) / len(severidades) if severidades else 0

    parametros = {
        'batch_size': 8 if total <= 6 else 16,
        'num_frames': 20,
        'learning_rate': 0.0006,
        'epochs': 280,
        'weight_decay': 1e-4,
        'patience': 50,
        'confusion_penalty': 2.5,
        'focal_gamma': 2.0,
        'label_smoothing': 0.02,
    }

    if prop_numeros > 0.3:
        parametros.update({'learning_rate': 0.0005, 'epochs': 320, 'patience': 60})
    elif prop_letras > 0.6:
        parametros.update({'learning_rate': 0.0006, 'epochs': 280, 'patience': 55})
    elif prop_palabras > 0.4:
        parametros.update({'learning_rate': 0.0008, 'epochs': 230, 'patience': 45})

    if severidad_promedio > 2.0:
        parametros['epochs'] = min(parametros['epochs'] + 40, 350)
        parametros['patience'] = min(parametros['patience'] + 10, 70)
        parametros['focal_gamma'] = 2.2

    if total <= 6:
        parametros['label_smoothing'] = max(parametros['label_smoothing'], 0.03)
        parametros['learning_rate'] = min(parametros['learning_rate'], 0.0006)

    parametros['distribucion'] = {
        'numeros': len(numeros),
        'letras': len(letras),
        'palabras': len(palabras),
        'severidad_confusion_promedio': severidad_promedio,
    }

    return parametros


def validar_configuracion_dataset(clases: List[str], videos_por_clase: Dict[str, int]) -> Dict[str, Any]:
    problemas, advertencias = [], []
    for clase in clases:
        cantidad = videos_por_clase.get(clase, 0)
        if cantidad < 20:
            advertencias.append(f"'{clase}': {cantidad} videos (recomendado 20+)")
        elif cantidad < 40:
            advertencias.append(
                f"'{clase}': {cantidad} videos (con pocas clases, 40+ por clase ayuda a separar confusiones)"
            )
    return {
        'valido': True,
        'problemas': problemas,
        'advertencias': advertencias,
        'total_problemas': len(problemas),
        'total_advertencias': len(advertencias),
    }