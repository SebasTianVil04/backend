import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging
from sqlalchemy.orm import Session

from ..modelos.entrenamiento import ModeloIA
from ..modelos.lstm_video import SignLanguageLSTM, SimpleLSTM

logger = logging.getLogger(__name__)


class LSTMEnsemblePredictor:
    """
    Sistema de ensemble para múltiples modelos LSTM[web:1][web:2][web:3].
    Combina predicciones de varios modelos para mejor accuracy.
    """
    
    def __init__(self, db: Session, device: str = None):
        self.db = db
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelos_cargados: Dict[str, Dict] = {}
        self.cargar_modelos_activos()
    
    def cargar_modelos_activos(self):
        """Carga todos los modelos LSTM activos en memoria."""
        modelos_db = self.db.query(ModeloIA).filter(
            ModeloIA.activo == True,
            ModeloIA.tipo_modelo == "LSTM"
        ).all()
        
        if not modelos_db:
            logger.warning("⚠️ No hay modelos LSTM activos")
            return
        
        logger.info(f"📦 Cargando {len(modelos_db)} modelos LSTM...")
        
        for modelo_db in modelos_db:
            try:
                self._cargar_modelo(modelo_db)
            except Exception as e:
                logger.error(f" Error cargando modelo {modelo_db.nombre}: {e}")
    
    def _cargar_modelo(self, modelo_db: ModeloIA):
        """Carga un modelo LSTM individual."""
        ruta_modelo = Path(modelo_db.ruta_archivo)
        
        if not ruta_modelo.exists():
            logger.error(f" Archivo no encontrado: {ruta_modelo}")
            return
        
        # Cargar checkpoint
        checkpoint = torch.load(ruta_modelo, map_location=self.device)
        
        # Recrear arquitectura
        num_classes = checkpoint.get('num_classes', modelo_db.num_clases)
        hidden_size = checkpoint.get('hidden_size', modelo_db.hidden_size or 512)
        num_layers = checkpoint.get('num_layers', modelo_db.num_layers or 3)
        bidirectional = checkpoint.get('bidirectional', modelo_db.bidirectional)
        
        # Crear modelo
        if checkpoint.get('architecture') == 'SignLanguageLSTM':
            modelo = SignLanguageLSTM(
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional
            )
        else:
            modelo = SimpleLSTM(
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
        
        # Cargar pesos
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.to(self.device)
        modelo.eval()
        
        # Cargar metadatos
        clases = checkpoint.get('clases', [])
        
        # Guardar en diccionario
        self.modelos_cargados[modelo_db.nombre] = {
            'modelo': modelo,
            'peso': modelo_db.peso_ensemble,
            'accuracy': modelo_db.accuracy or 1.0,
            'clases': clases,
            'num_clases': num_classes,
            'num_frames': modelo_db.num_frames or 24
        }
        
        logger.info(f" Modelo cargado: {modelo_db.nombre} "
                   f"(peso: {modelo_db.peso_ensemble}, acc: {modelo_db.accuracy:.4f})")
    
    @torch.no_grad()
    def predecir_ensemble(
        self,
        video_frames: torch.Tensor,
        metodo: str = "weighted_average",
        top_k: int = 3
    ) -> Dict:
        """
        Predice usando ensemble de modelos LSTM[web:2][web:3].
        
        Args:
            video_frames: Tensor (1, num_frames, 3, 224, 224)
            metodo: 'weighted_average', 'voting', 'stacking'
            top_k: Número de predicciones top a retornar
        
        Returns:
            Diccionario con predicción y detalles
        """
        if not self.modelos_cargados:
            raise ValueError("No hay modelos activos para predicción")
        
        video_frames = video_frames.to(self.device)
        predicciones_individuales = {}
        
        # Obtener predicciones de cada modelo
        for nombre, info in self.modelos_cargados.items():
            try:
                modelo = info['modelo']
                peso = info['peso']
                accuracy = info['accuracy']
                clases = info['clases']
                
                # Predicción
                logits = modelo(video_frames)
                probs = torch.softmax(logits, dim=1)[0]  # (num_classes,)
                
                predicciones_individuales[nombre] = {
                    'probabilidades': probs.cpu().numpy(),
                    'peso': peso,
                    'accuracy': accuracy,
                    'clase_idx': int(torch.argmax(probs)),
                    'confianza': float(torch.max(probs)),
                    'clases': clases
                }
                
            except Exception as e:
                logger.error(f" Error en predicción de {nombre}: {e}")
                continue
        
        if not predicciones_individuales:
            raise ValueError("Ningún modelo pudo realizar predicción")
        
        # Combinar predicciones
        if metodo == "weighted_average":
            resultado = self._weighted_average(predicciones_individuales, top_k)
        elif metodo == "voting":
            resultado = self._majority_voting(predicciones_individuales, top_k)
        elif metodo == "stacking":
            resultado = self._stacking(predicciones_individuales, top_k)
        else:
            resultado = self._weighted_average(predicciones_individuales, top_k)
        
        # Agregar detalles de modelos individuales
        resultado['modelos_usados'] = {
            nombre: {
                'clase_predicha': info['clases'][info['clase_idx']] if info['clases'] else str(info['clase_idx']),
                'confianza': info['confianza'],
                'peso': info['peso']
            }
            for nombre, info in predicciones_individuales.items()
        }
        
        return resultado
    
    def _weighted_average(self, predicciones: Dict, top_k: int) -> Dict:
        """Promedio ponderado de probabilidades[web:2]."""
        suma_ponderada = None
        suma_pesos = 0
        clases_referencia = None
        
        for nombre, info in predicciones.items():
            peso_final = info['peso'] * info['accuracy']
            probs = info['probabilidades']
            
            if clases_referencia is None:
                clases_referencia = info['clases']
            
            if suma_ponderada is None:
                suma_ponderada = probs * peso_final
            else:
                suma_ponderada += probs * peso_final
            
            suma_pesos += peso_final
        
        probabilidades_finales = suma_ponderada / suma_pesos
        
        # Top-k predicciones
        top_indices = np.argsort(probabilidades_finales)[-top_k:][::-1]
        
        return {
            'clase_idx': int(top_indices[0]),
            'clase': clases_referencia[top_indices[0]] if clases_referencia else str(top_indices[0]),
            'confianza': float(probabilidades_finales[top_indices[0]]),
            'top_predicciones': [
                {
                    'clase': clases_referencia[idx] if clases_referencia else str(idx),
                    'probabilidad': float(probabilidades_finales[idx])
                }
                for idx in top_indices
            ],
            'metodo': 'weighted_average',
            'num_modelos': len(predicciones)
        }
    
    def _majority_voting(self, predicciones: Dict, top_k: int) -> Dict:
        """Votación por mayoría ponderada[web:3]."""
        votos = {}
        clases_referencia = None
        
        for nombre, info in predicciones.items():
            clase_idx = info['clase_idx']
            peso = info['peso'] * info['accuracy']
            
            if clases_referencia is None:
                clases_referencia = info['clases']
            
            votos[clase_idx] = votos.get(clase_idx, 0) + peso
        
        # Ordenar por votos
        clases_ordenadas = sorted(votos.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        clase_ganadora_idx = clases_ordenadas[0][0]
        
        # Calcular confianza promedio
        confianzas = [
            info['confianza'] 
            for info in predicciones.values() 
            if info['clase_idx'] == clase_ganadora_idx
        ]
        confianza_promedio = np.mean(confianzas) if confianzas else 0.0
        
        return {
            'clase_idx': int(clase_ganadora_idx),
            'clase': clases_referencia[clase_ganadora_idx] if clases_referencia else str(clase_ganadora_idx),
            'confianza': float(confianza_promedio),
            'top_predicciones': [
                {
                    'clase': clases_referencia[idx] if clases_referencia else str(idx),
                    'votos': float(votos_val),
                    'probabilidad': float(votos_val / sum(votos.values()))
                }
                for idx, votos_val in clases_ordenadas
            ],
            'metodo': 'majority_voting',
            'num_modelos': len(predicciones)
        }
    
    def _stacking(self, predicciones: Dict, top_k: int) -> Dict:
        """Stacking: usa modelo con mejor accuracy[web:3]."""
        mejor_nombre, mejor_info = max(
            predicciones.items(),
            key=lambda x: x[1]['accuracy']
        )
        
        clase_idx = mejor_info['clase_idx']
        clases = mejor_info['clases']
        
        # Top-k del mejor modelo
        top_indices = np.argsort(mejor_info['probabilidades'])[-top_k:][::-1]
        
        return {
            'clase_idx': clase_idx,
            'clase': clases[clase_idx] if clases else str(clase_idx),
            'confianza': mejor_info['confianza'],
            'top_predicciones': [
                {
                    'clase': clases[idx] if clases else str(idx),
                    'probabilidad': float(mejor_info['probabilidades'][idx])
                }
                for idx in top_indices
            ],
            'metodo': 'stacking',
            'mejor_modelo': mejor_nombre,
            'num_modelos': len(predicciones)
        }
    
    def recargar_modelos(self):
        """Recarga los modelos activos."""
        self.modelos_cargados.clear()
        self.cargar_modelos_activos()
