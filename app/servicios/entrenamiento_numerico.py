import logging
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter

from app.servicios.config_tipo_senas import (
    es_confusion_numerica_critica,
    generar_pares_confusion_numerica,
    CONFUSIONES_NUMERICAS_CRITICAS
)

logger = logging.getLogger(__name__)

class NumericConfusionAwareLoss(nn.Module):
    
    def __init__(self, clases: List[str], alpha=1.0, gamma=2.5, base_penalty=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.base_penalty = base_penalty
        self.clases = clases
        self.clase_a_idx = {clase: i for i, clase in enumerate(clases)}
        
        self.penalty_matrix = self._construir_matriz_penalizaciones()
        
    def _construir_matriz_penalizaciones(self) -> torch.Tensor:
        n_clases = len(self.clases)
        penalty_matrix = torch.ones(n_clases, n_clases)
        
        for i, clase_i in enumerate(self.clases):
            for j, clase_j in enumerate(self.clases):
                if i != j:
                    if (clase_i == '2' and clase_j == '4') or (clase_i == '4' and clase_j == '2'):
                        penalty_matrix[i, j] = self.base_penalty * 1.4
                    elif es_confusion_numerica_critica(clase_i, clase_j):
                        penalty_matrix[i, j] = self.base_penalty
                    
        return penalty_matrix
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        probs = torch.softmax(inputs, dim=1)
        batch_penalties = torch.ones_like(focal_loss)
        
        for i in range(len(targets)):
            target_idx = targets[i].item()
            for j, prob in enumerate(probs[i]):
                if j != target_idx and self.penalty_matrix[target_idx, j] > 1.0:
                    batch_penalties[i] += prob * self.penalty_matrix[target_idx, j]
        
        return (focal_loss * batch_penalties).mean()

class NumericDataAugmentation:
    
    def __init__(self):
        self.augmentation_config = {
            'rotation_range': 2.0,
            'brightness_range': (0.9, 1.1),
            'contrast_range': (0.95, 1.05),
            'zoom_range': (0.95, 1.05),
            'shift_range': 10,
            'no_flip': True,
            'no_rotation': False,
            'preserve_shape': True
        }
        
        self.augmentation_config_2_4 = {
            'rotation_range': 0.3,
            'brightness_range': (0.98, 1.02),
            'contrast_range': (0.99, 1.01),
            'zoom_range': (0.98, 1.02),
            'shift_range': 5,
            'no_flip': True,
            'no_rotation': True,
            'preserve_shape': True,
            'preserve_angle': True
        }
    
    def augment_for_numeric(self, frames: np.ndarray, sena: str) -> np.ndarray:
        if not sena.isdigit():
            return frames
        
        config = self.augmentation_config_2_4 if sena in ['2', '4'] else self.augmentation_config
        
        frames = frames.copy()
        h, w = frames.shape[1:3]
        
        brightness = np.random.uniform(*config['brightness_range'])
        frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
        
        alpha = np.random.uniform(*config['contrast_range'])
        frames = np.clip(alpha * frames, 0, 255).astype(np.uint8)
        
        if not config.get('no_rotation', False):
            max_angle = config['rotation_range']
            angle = np.random.uniform(-max_angle, max_angle)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            frames = np.array([
                cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT) 
                for frame in frames
            ])
        
        shift_range = config['shift_range']
        shift_x = np.random.randint(-shift_range, shift_range)
        shift_y = np.random.randint(-shift_range, shift_range)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frames = np.array([
            cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT) 
            for frame in frames
        ])
        
        return frames

class NumericTrainingOptimizer:
    
    def __init__(self, clases: List[str]):
        self.clases = clases
        self.has_numeric_confusions = self._detectar_confusiones_numericas()
        self.has_2_and_4 = '2' in clases and '4' in clases
        
    def _detectar_confusiones_numericas(self) -> bool:
        pares_confusion = generar_pares_confusion_numerica(self.clases)
        return len(pares_confusion) > 0
    
    def get_training_parameters(self) -> Dict[str, Any]:
        # NOTA: Las épocas son definidas por el frontend, estos valores son sugerencias
        base_params = {
            'learning_rate': 0.0003,
            'batch_size': 4,
            'epochs': 300,
            'patience': 50,
            'weight_decay': 1e-4
        }
        
        if self.has_2_and_4:
            base_params.update({
                'learning_rate': 0.00015,
                'batch_size': 2,
                'epochs': 350,
                'patience': 60,
                'use_contrastive_loss': True,
                'augmentation_conservative': True,
                'early_stopping_min_delta': 0.001,
                'focal_gamma': 3.0,
                'confusion_penalty': 7.0
            })
        
        elif self.has_numeric_confusions:
            base_params.update({
                'learning_rate': 0.0002,
                'batch_size': 2,
                'epochs': 400,
                'patience': 60,
                'use_contrastive_loss': True,
                'augmentation_conservative': True,
                'early_stopping_min_delta': 0.001
            })
        
        return base_params
    
    def get_loss_function(self):
        if self.has_2_and_4:
            return NumericConfusionAwareLoss(self.clases, alpha=1.0, gamma=3.0, base_penalty=7.0)
        elif self.has_numeric_confusions:
            return NumericConfusionAwareLoss(self.clases, alpha=1.0, gamma=2.5, base_penalty=5.0)
        else:
            return nn.CrossEntropyLoss()
    
    def generate_training_recommendations(self) -> List[str]:
        recomendaciones = []
        
        if self.has_2_and_4:
            recomendaciones.append("ENTRENAMIENTO ULTRA-CRITICO - Confusión 2↔4 detectada:")
            recomendaciones.append("   → Par de confusión MAS DIFICIL del sistema")
            recomendaciones.append("   → 2: DOS dedos separados en V")
            recomendaciones.append("   → 4: CUATRO dedos juntos verticales")
            recomendaciones.extend([
                "   → Usando learning rate ultra-bajo (0.00015)",
                "   → Epocas extendidas (350)",
                "   → Batch size minimo (2)",
                "   → Penalizacion MAXIMA 7.0x por confusiones 2↔4",
                "   → Augmentation MINIMO (±2% brightness, ±0.3° rotation)",
                "   → NO flip horizontal NUNCA",
                "   → NO rotacion (preservar angulo de dedos)",
                "   → Oversampling 8x para numeros 2 y 4"
            ])
        
        elif self.has_numeric_confusions:
            pares = generar_pares_confusion_numerica(self.clases)
            recomendaciones.append("ENTRENAMIENTO CRITICO - Confusiones numericas detectadas:")
            recomendaciones.append(f"   → {len(pares)} pares de confusion numerica")
            
            for par in pares[:5]:
                recomendaciones.append(f"   • {par['sena1']} ↔ {par['sena2']}")
            
            recomendaciones.extend([
                "   → Usando learning rate ultra-bajo (0.0002)",
                "   → Epocas extendidas (400)",
                "   → Batch size pequeño (2)",
                "   → Penalizacion maxima por confusiones numericas",
                "   → Augmentation ultra-conservador",
                "   → NO flip horizontal",
                "   → Rotacion minima (±2°)",
                "   → Oversampling 5x para numeros confusos"
            ])
        else:
            recomendaciones.append("Sin confusiones numericas criticas - Usando parametros estandar")
        
        return recomendaciones

class NumericConfusionService:
    
    def __init__(self):
        self.augmenter = NumericDataAugmentation()
    
    def analyze_dataset_for_numeric_confusions(self, clases: List[str], videos_por_clase: Dict[str, int]) -> Dict[str, Any]:
        numeros = [c for c in clases if c.isdigit() and len(c) == 1]
        letras_confusas = [c for c in clases if c.isalpha() and len(c) == 1 and any(c in CONFUSIONES_NUMERICAS_CRITICAS.get(n, []) for n in numeros)]
        
        pares_confusion = generar_pares_confusion_numerica(clases)
        
        balance_issues = []
        for par in pares_confusion:
            sena1, sena2 = par['sena1'], par['sena2']
            count1 = videos_por_clase.get(sena1, 0)
            count2 = videos_por_clase.get(sena2, 0)
            
            if count1 > 0 and count2 > 0:
                ratio = max(count1, count2) / min(count1, count2)
                if ratio > 3.0:
                    balance_issues.append({
                        'par': f"{sena1}-{sena2}",
                        'counts': {sena1: count1, sena2: count2},
                        'ratio': round(ratio, 2)
                    })
        
        tiene_2_y_4 = '2' in clases and '4' in clases
        
        return {
            'tiene_numeros': len(numeros) > 0,
            'tiene_letras_confusas': len(letras_confusas) > 0,
            'tiene_2_y_4': tiene_2_y_4,
            'total_numeros': len(numeros),
            'total_letras_confusas': len(letras_confusas),
            'pares_confusion_numerica': pares_confusion,
            'total_pares_confusion': len(pares_confusion),
            'balance_issues': balance_issues,
            'recomendaciones_especificas': self._generar_recomendaciones_especificas(numeros, letras_confusas, pares_confusion, tiene_2_y_4)
        }
    
    def _generar_recomendaciones_especificas(self, numeros: List[str], letras_confusas: List[str], pares_confusion: List[Dict], tiene_2_y_4: bool) -> List[str]:
        recomendaciones = []
        
        if not numeros:
            return ["No hay numeros en el dataset - Sin acciones especiales necesarias"]
        
        recomendaciones.append(f"Dataset contiene {len(numeros)} numeros: {', '.join(numeros)}")
        
        if letras_confusas:
            recomendaciones.append(f"Y {len(letras_confusas)} letras confusas: {', '.join(letras_confusas)}")
        
        if tiene_2_y_4:
            recomendaciones.append("ALERTA MAXIMA: Numeros 2 y 4 presentes")
            recomendaciones.append("   → Este es el par de confusion MAS DIFICIL del sistema")
            recomendaciones.append("   → Requiere parametros ultra-conservadores")
            recomendaciones.append("   → 2: Dos dedos en V (indice y medio separados)")
            recomendaciones.append("   → 4: Cuatro dedos juntos (todos hacia arriba)")
            recomendaciones.append("   → Diferencia clave: CANTIDAD DE DEDOS y su DISPOSICION")
        
        if pares_confusion:
            recomendaciones.append(f"{len(pares_confusion)} pares de confusion numerica detectados:")
            
            por_numero = {}
            for par in pares_confusion:
                num = par['sena1'] if par['sena1'].isdigit() else par['sena2']
                letra = par['sena2'] if par['sena1'].isdigit() else par['sena1']
                if num not in por_numero:
                    por_numero[num] = []
                por_numero[num].append(letra)
            
            for num, letras in por_numero.items():
                recomendaciones.append(f"   • {num} se confunde con: {', '.join(letras)}")
        
        pares_hyper_criticos = [('0', 'O'), ('1', 'I'), ('1', 'D'), ('2', '4'), ('4', '2')]
        for num, otra in pares_hyper_criticos:
            if num in numeros and otra in (numeros + letras_confusas):
                if num in ['2', '4'] and otra in ['2', '4']:
                    recomendaciones.append(f"   ATENCION MAXIMA: {num} ↔ {otra} - Par hipercritico EXTREMO")
                else:
                    recomendaciones.append(f"   ATENCION MAXIMA: {num} ↔ {otra} - Par hipercritico")
        
        return recomendaciones

numeric_confusion_service = NumericConfusionService()