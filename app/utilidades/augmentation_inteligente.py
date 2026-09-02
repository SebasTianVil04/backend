import cv2
import numpy as np
import random
from typing import Dict, Any, List

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
    }
}

LETRAS_CONFUSAS_CRITICAS = {
    'A', 'C', 'S', 'E', 'M', 'N', 'T', 'O', 'Q', 'D', 'B', 'P', 'F', 
    'R', 'K', 'U', 'V', 'W', 'H', 'I', 'L', 'J', 'X', 'Y', 'Z', 'G'
}

NUMEROS_CONFUSOS_CRITICOS = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
}

PALABRAS_CONFUSAS = {
    'HOLA', 'GRACIAS', 'AYUDA', 'SI', 'NO', 'BIEN', 'MAL', 
    'POR FAVOR', 'DE NADA', 'LO SIENTO', 'ADIOS'
}

SENAS_SIN_FLIP = {
    'A', 'C', 'D', 'G', 'J', 'P', 'Q', 'Z', 'B', 'F', 'R', 'T',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'ADIOS', 'VENIR', 'IR', 'DERECHA', 'IZQUIERDA', 'ARRIBA', 'ABAJO'
}

def detectar_categoria_augmentation(sena: str, es_confusa: bool = False, 
                                    es_confusion_numerica: bool = False) -> str:
    sena_upper = sena.upper().strip()
    
    if sena_upper.isdigit():
        if es_confusion_numerica or sena_upper in NUMEROS_CONFUSOS_CRITICOS:
            return 'NUMERO_CONFUSO'
        return 'NUMERO'
    
    if len(sena_upper) == 1 and sena_upper.isalpha():
        if es_confusa or sena_upper in LETRAS_CONFUSAS_CRITICAS:
            return 'LETRA_CONFUSA'
        return 'LETRA'
    
    if sena_upper in PALABRAS_CONFUSAS or es_confusa:
        return 'PALABRA_CONFUSA'
    
    if len(sena_upper.split()) > 1:
        return 'PALABRA_CONFUSA'
    
    if len(sena_upper) <= 15:
        return 'PALABRA'
    
    return 'PALABRA'

def aplicar_augmentation_inteligente(frames: np.ndarray, config: Dict[str, Any], 
                                     sena: str, es_training: bool = True) -> np.ndarray:
    if not es_training:
        return frames
    
    es_confusa = config.get('es_confusa', False)
    es_confusion_numerica = config.get('es_confusion_numerica', False)
    
    categoria = detectar_categoria_augmentation(sena, es_confusa, es_confusion_numerica)
    aug_config = AUGMENTATION_CONFIG.get(categoria, AUGMENTATION_CONFIG['LETRA'])
    
    frames = frames.copy()
    
    sena_upper = sena.upper()
    permite_flip = sena_upper not in SENAS_SIN_FLIP and aug_config['flip_prob'] > 0
    
    if permite_flip and random.random() < aug_config['flip_prob']:
        frames = np.flip(frames, axis=2).copy()
    
    if random.random() > 0.25:
        brightness = random.uniform(*aug_config['brightness_range'])
        frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
    
    if random.random() > 0.35:
        max_angle = aug_config['rotation_range']
        angle = random.uniform(-max_angle, max_angle)
        h, w = frames.shape[1:3]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        frames = np.array([
            cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT) 
            for frame in frames
        ])
    
    if random.random() > 0.4:
        alpha = random.uniform(*aug_config['contrast_range'])
        frames = np.clip(alpha * frames, 0, 255).astype(np.uint8)
    
    if random.random() > 0.5:
        zoom_min, zoom_max = aug_config['zoom_range']
        zoom_factor = random.uniform(zoom_min, zoom_max)
        h, w = frames.shape[1:3]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        frames_zoomed = []
        for frame in frames:
            if zoom_factor > 1:
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                y_start = (new_h - h) // 2
                x_start = (new_w - w) // 2
                cropped = resized[y_start:y_start+h, x_start:x_start+w]
                frames_zoomed.append(cropped)
            else:
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                y_offset = (h - new_h) // 2
                x_offset = (w - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                frames_zoomed.append(canvas)
        
        frames = np.array(frames_zoomed)
    
    if random.random() > 0.55:
        shift_range = aug_config['shift_range']
        shift_x = random.randint(-shift_range, shift_range)
        shift_y = random.randint(-shift_range, shift_range)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frames = np.array([
            cv2.warpAffine(frame, M, (224, 224), borderMode=cv2.BORDER_REFLECT) 
            for frame in frames
        ])
    
    if random.random() < aug_config['noise_prob']:
        noise_min, noise_max = aug_config['noise_level']
        noise_level = random.uniform(noise_min, noise_max)
        noise = np.random.normal(0, noise_level, frames.shape).astype(np.int16)
        frames = np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    if random.random() < aug_config['blur_prob']:
        kernel_min, kernel_max = aug_config['blur_kernel']
        kernel_size = random.choice([3, 5, 7])
        if kernel_size <= kernel_max:
            frames = np.array([
                cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0) 
                for frame in frames
            ])
    
    if categoria in ['NUMERO', 'NUMERO_CONFUSO'] and random.random() > 0.6:
        shadow_intensity = random.uniform(0.6, 0.9)
        shadow_x = random.randint(0, 224)
        shadow_y = random.randint(0, 224)
        shadow_radius = random.randint(60, 120)
        
        for i in range(len(frames)):
            mask = np.zeros((224, 224), dtype=np.float32)
            cv2.circle(mask, (shadow_x, shadow_y), shadow_radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
            for c in range(3):
                frames[i, :, :, c] = np.clip(
                    frames[i, :, :, c] * (shadow_intensity + (1 - shadow_intensity) * mask),
                    0, 255
                ).astype(np.uint8)
    
    if categoria in ['LETRA', 'PALABRA'] and random.random() > 0.7:
        gamma = random.uniform(0.7, 1.3)
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in range(256)
        ]).astype(np.uint8)
        frames = np.array([cv2.LUT(frame, table) for frame in frames])
    
    return frames

def obtener_configuracion_sampler(clases: List[str]) -> Dict[str, Any]:
    from collections import Counter
    
    numeros = [c for c in clases if c.isdigit()]
    letras = [c for c in clases if len(c) == 1 and c.isalpha()]
    palabras = [c for c in clases if len(c) > 1 or c not in numeros + letras]
    
    letras_confusas = [c for c in letras if c.upper() in LETRAS_CONFUSAS_CRITICAS]
    numeros_confusos = [c for c in numeros if c in NUMEROS_CONFUSOS_CRITICOS]
    palabras_confusas = [c for c in palabras if c.upper() in PALABRAS_CONFUSAS]
    
    config = {
        'oversampling_numeros': 4.0 if len(numeros) > 0 else 1.0,
        'oversampling_numeros_confusos': 5.0 if len(numeros_confusos) > 0 else 1.0,
        'oversampling_letras': 1.5 if len(letras) > 0 else 1.0,
        'oversampling_letras_confusas': 2.5 if len(letras_confusas) > 0 else 1.0,
        'oversampling_palabras': 1.2 if len(palabras) > 0 else 1.0,
        'oversampling_palabras_confusas': 2.0 if len(palabras_confusas) > 0 else 1.0,
        'total_numeros': len(numeros),
        'total_letras': len(letras),
        'total_palabras': len(palabras),
        'numeros_confusos': numeros_confusos,
        'letras_confusas': letras_confusas,
        'palabras_confusas': palabras_confusas
    }
    
    return config

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
        sena_upper = sena.upper()
        
        base_weight = 1.0 / class_counts[label]
        
        if sena.isdigit():
            if sena in config_sampler['numeros_confusos']:
                base_weight *= config_sampler['oversampling_numeros_confusos']
            else:
                base_weight *= config_sampler['oversampling_numeros']
        
        elif len(sena) == 1 and sena.isalpha():
            if sena_upper in config_sampler['letras_confusas']:
                base_weight *= config_sampler['oversampling_letras_confusas']
            else:
                base_weight *= config_sampler['oversampling_letras']
        
        else:
            if sena_upper in config_sampler['palabras_confusas']:
                base_weight *= config_sampler['oversampling_palabras_confusas']
            else:
                base_weight *= config_sampler['oversampling_palabras']
        
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
    palabras = [c for c in clases if len(c) > 1 or c not in numeros + letras]
    
    total = len(clases)
    prop_numeros = len(numeros) / total if total > 0 else 0
    prop_letras = len(letras) / total if total > 0 else 0
    prop_palabras = len(palabras) / total if total > 0 else 0
    
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
    
    parametros['distribucion'] = {
        'numeros': len(numeros),
        'letras': len(letras),
        'palabras': len(palabras),
        'proporcion_numeros': prop_numeros,
        'proporcion_letras': prop_letras,
        'proporcion_palabras': prop_palabras
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
        
        if clase.isdigit():
            if cantidad < min_videos_numero:
                problemas.append(f"Número '{clase}': {cantidad} videos (mínimo {min_videos_numero})")
        
        elif len(clase) == 1 and clase.isalpha():
            if cantidad < min_videos_letra:
                if clase.upper() in LETRAS_CONFUSAS_CRITICAS:
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

def generar_reporte_augmentation(clases: List[str]) -> Dict[str, Any]:
    categorias_count = {}
    
    for clase in clases:
        categoria = detectar_categoria_augmentation(clase)
        categorias_count[categoria] = categorias_count.get(categoria, 0) + 1
    
    reporte = {
        'total_clases': len(clases),
        'distribucion_categorias': categorias_count,
        'configuraciones_aplicadas': {}
    }
    
    for categoria, count in categorias_count.items():
        config = AUGMENTATION_CONFIG.get(categoria, {})
        reporte['configuraciones_aplicadas'][categoria] = {
            'cantidad_clases': count,
            'intensidad': config.get('intensidad', 'DESCONOCIDA'),
            'permite_flip': config.get('flip_prob', 0) > 0,
            'rotation_max': config.get('rotation_range', 0),
            'brightness_range': config.get('brightness_range', (1.0, 1.0))
        }
    
    return reporte