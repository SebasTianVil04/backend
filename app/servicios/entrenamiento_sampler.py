import numpy as np
import random
from typing import Dict, Any, List

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
        'severidad_maxima': severidad_maxima
    }


def _multiplicador_oversampling(sena: str, config_sampler: Dict[str, Any]) -> float:
    severidad = config_sampler['severidades'].get(sena, 0)
    severidad_maxima = config_sampler['severidad_maxima']

    if es_sena_emergencia(sena):
        base = 4.0
    elif sena.isdigit():
        base = 3.0
    elif len(sena) == 1 and sena.isalpha():
        base = 1.3
    else:
        base = 1.1

    if severidad_maxima > 0:
        factor = 1.0 + (severidad / severidad_maxima) * 1.0
    else:
        factor = 1.0

    return base * factor


def crear_sampler_balanceado(train_labels: List[int], train_senas: List[str], clases: List[str]):
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    class_counts = Counter(train_labels)
    config_sampler = obtener_configuracion_sampler(clases)

    sample_weights = []
    for label in train_labels:
        if label >= len(clases):
            sample_weights.append(1.0)
            continue
        sena = clases[label]
        peso = (1.0 / class_counts[label]) * _multiplicador_oversampling(sena, config_sampler)
        sample_weights.append(peso)

    total_samples = int(len(sample_weights) * 1.5)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
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
        'batch_size': 8,
        'num_frames': 20,
        'learning_rate': 0.0008,
        'epochs': 300,
        'weight_decay': 1e-4,
        'patience': 60,
        'confusion_penalty': 2.0,
        'focal_gamma': 1.5,
        'label_smoothing': 0.02,
    }

    if prop_numeros > 0.5:
        parametros['learning_rate'] = 0.0006
        parametros['epochs'] = 350
        parametros['patience'] = 70
    elif prop_letras > 0.7:
        parametros['learning_rate'] = 0.0008
        parametros['epochs'] = 300
        parametros['patience'] = 60
    elif prop_palabras > 0.5:
        parametros['learning_rate'] = 0.0008
        parametros['epochs'] = 250
        parametros['patience'] = 50

    if severidad_promedio > 3:
        parametros['epochs'] = max(parametros['epochs'], 300)
        parametros['patience'] = max(parametros['patience'], 60)

    parametros['distribucion'] = {
        'numeros': len(numeros),
        'letras': len(letras),
        'palabras': len(palabras),
        'severidad_confusion_promedio': severidad_promedio
    }

    return parametros


def validar_configuracion_dataset(clases: List[str], videos_por_clase: Dict[str, int]) -> Dict[str, Any]:
    problemas, advertencias = [], []
    for clase in clases:
        cantidad = videos_por_clase.get(clase, 0)
        if cantidad < 20:
            advertencias.append(f"'{clase}': {cantidad} videos (recomendado 20+)")
    return {
        'valido': True,
        'problemas': problemas,
        'advertencias': advertencias,
        'total_problemas': len(problemas),
        'total_advertencias': len(advertencias)
    }
