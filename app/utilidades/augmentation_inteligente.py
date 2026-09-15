import cv2
import numpy as np
import random
from typing import Dict, Any, List

from app.servicios.config_tipo_senas import (
    obtener_confusiones_comunes,
    obtener_grupo_confusion,
    es_sena_emergencia,
    SENAS_NO_FLIP
)

AUGMENTATION_CONFIG = {
    'NUMERO': {
        'rotation_range': 10,
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.85, 1.15),
        'zoom_range': (0.9, 1.1),
        'shift_range': 25,
        'noise_prob': 0.3,
        'noise_level': (3, 8),
        'flip_prob': 0.0,
        'blur_prob': 0.2,
        'blur_kernel': (3, 5),
        'intensidad': 'ALTA'
    },
    'LETRA': {
        'rotation_range': 15,
        'brightness_range': (0.7, 1.3),
        'contrast_range': (0.8, 1.2),
        'zoom_range': (0.85, 1.15),
        'shift_range': 30,
        'noise_prob': 0.4,
        'noise_level': (5, 15),
        'flip_prob': 0.5,
        'blur_prob': 0.25,
        'blur_kernel': (3, 7),
        'intensidad': 'MEDIA'
    },
    'LETRA_CONFUSA': {
        'rotation_range': 8,
        'brightness_range': (0.85, 1.15),
        'contrast_range': (0.9, 1.1),
        'zoom_range': (0.92, 1.08),
        'shift_range': 20,
        'noise_prob': 0.2,
        'noise_level': (3, 10),
        'flip_prob': 0.0,
        'blur_prob': 0.15,
        'blur_kernel': (3, 5),
        'intensidad': 'MEDIA_BAJA'
    },
    'NUMERO_CONFUSO': {
        'rotation_range': 8,
        'brightness_range': (0.85, 1.15),
        'contrast_range': (0.9, 1.1),
        'zoom_range': (0.92, 1.08),
        'shift_range': 20,
        'noise_prob': 0.25,
        'noise_level': (3, 8),
        'flip_prob': 0.0,
        'blur_prob': 0.15,
        'blur_kernel': (3, 5),
        'intensidad': 'MEDIA'
    },
    'PALABRA': {
        'rotation_range': 12,
        'brightness_range': (0.75, 1.25),
        'contrast_range': (0.85, 1.15),
        'zoom_range': (0.88, 1.12),
        'shift_range': 25,
        'noise_prob': 0.35,
        'noise_level': (5, 12),
        'flip_prob': 0.3,
        'blur_prob': 0.2,
        'blur_kernel': (3, 7),
        'intensidad': 'MEDIA'
    },
    'PALABRA_CONFUSA': {
        'rotation_range': 7,
        'brightness_range': (0.88, 1.12),
        'contrast_range': (0.92, 1.08),
        'zoom_range': (0.95, 1.05),
        'shift_range': 15,
        'noise_prob': 0.15,
        'noise_level': (2, 6),
        'flip_prob': 0.0,
        'blur_prob': 0.1,
        'blur_kernel': (3, 5),
        'intensidad': 'BAJA'
    },
    'EMERGENCIA': {
        'rotation_range': 20,
        'brightness_range': (0.5, 1.5),
        'contrast_range': (0.85, 1.15),
        'zoom_range': (0.9, 1.1),
        'shift_range': 20,
        'noise_prob': 0.1,
        'noise_level': (2, 6),
        'flip_prob': 0.0,
        'blur_prob': 0.1,
        'blur_kernel': (3, 5),
        'intensidad': 'BAJA'
    }
}


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
        base = 6.0
    elif sena.isdigit():
        base = 4.0
    elif len(sena) == 1 and sena.isalpha():
        base = 1.5
    else:
        base = 1.2

    if severidad_maxima > 0:
        factor_severidad = 1.0 + (severidad / severidad_maxima) * 1.5
    else:
        factor_severidad = 1.0

    return base * factor_severidad


def crear_sampler_balanceado(train_labels: List[int], train_senas: List[str],
                             clases: List[str]):
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    class_counts = Counter(train_labels)
    config_sampler = obtener_configuracion_sampler(clases)

    sample_weights = []

    for i, label in enumerate(train_labels):
        clase_idx = label
        if clase_idx >= len(clases):
            sample_weights.append(1.0)
            continue

        sena = clases[clase_idx]
        base_weight = 1.0 / class_counts[label]
        base_weight *= _multiplicador_oversampling(sena, config_sampler)

        sample_weights.append(base_weight)

    total_samples = len(sample_weights)
    if config_sampler['total_numeros'] > 0:
        total_samples = int(total_samples * 1.5)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )

    return sampler


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
        'batch_size': 6,
        'num_frames': 16,
        'learning_rate': 0.001,
        'epochs': 150,
        'weight_decay': 1e-4,
        'patience': 30,
        'confusion_penalty': 3.0,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1
    }

    if prop_numeros > 0.5:
        parametros['learning_rate'] = 0.0003
        parametros['epochs'] = 250
        parametros['patience'] = 50
        parametros['confusion_penalty'] = 4.5
        parametros['focal_gamma'] = 2.5
        parametros['label_smoothing'] = 0.15

    elif prop_letras > 0.7:
        parametros['learning_rate'] = 0.0005
        parametros['epochs'] = 200
        parametros['patience'] = 40
        parametros['confusion_penalty'] = 4.0
        parametros['focal_gamma'] = 2.5
        parametros['label_smoothing'] = 0.12

    elif prop_palabras > 0.5:
        parametros['learning_rate'] = 0.0007
        parametros['epochs'] = 175
        parametros['patience'] = 35
        parametros['confusion_penalty'] = 3.5
        parametros['focal_gamma'] = 2.2
        parametros['label_smoothing'] = 0.1

    else:
        if len(numeros) > 0:
            parametros['learning_rate'] = min(parametros['learning_rate'], 0.0005)
            parametros['epochs'] = max(parametros['epochs'], 200)

    if severidad_promedio > 3:
        parametros['epochs'] = max(parametros['epochs'], 250)
        parametros['patience'] = max(parametros['patience'], 45)
        parametros['confusion_penalty'] = max(parametros['confusion_penalty'], 4.0)

    parametros['distribucion'] = {
        'numeros': len(numeros),
        'letras': len(letras),
        'palabras': len(palabras),
        'proporcion_numeros': prop_numeros,
        'proporcion_letras': prop_letras,
        'proporcion_palabras': prop_palabras,
        'severidad_confusion_promedio': severidad_promedio
    }

    return parametros


def validar_configuracion_dataset(clases: List[str], videos_por_clase: Dict[str, int]) -> Dict[str, Any]:
    min_videos_numero = 15
    min_videos_letra = 12
    min_videos_palabra = 10

    problemas = []
    advertencias = []

    for clase in clases:
        cantidad = videos_por_clase.get(clase, 0)
        severidad = _severidad_confusion(clase)

        if clase.isdigit():
            if cantidad < min_videos_numero:
                problemas.append(f"Número '{clase}': {cantidad} videos (mínimo {min_videos_numero})")

        elif len(clase) == 1 and clase.isalpha():
            if cantidad < min_videos_letra:
                if severidad > 0:
                    problemas.append(f"Letra confusa '{clase}': {cantidad} videos (mínimo {min_videos_letra})")
                else:
                    advertencias.append(f"Letra '{clase}': {cantidad} videos (recomendado {min_videos_letra})")

        else:
            if cantidad < min_videos_palabra:
                advertencias.append(f"Palabra '{clase}': {cantidad} videos (recomendado {min_videos_palabra})")

    return {
        'valido': len(problemas) == 0,
        'problemas': problemas,
        'advertencias': advertencias,
        'total_problemas': len(problemas),
        'total_advertencias': len(advertencias)
    }
