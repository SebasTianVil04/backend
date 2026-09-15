import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging
from sqlalchemy.orm import Session

from ..modelos.entrenamiento import ModeloIA
from ..modelos.modelo_adaptativo import ModeloAdaptativoSenas, cargar_modelo_adaptativo

logger = logging.getLogger(__name__)


def _ajustar_num_frames(keypoints: torch.Tensor, num_frames_objetivo: int) -> torch.Tensor:
    num_frames_actual = keypoints.shape[1]

    if num_frames_actual == num_frames_objetivo:
        return keypoints

    if num_frames_actual > num_frames_objetivo:
        indices = torch.linspace(0, num_frames_actual - 1, num_frames_objetivo).long()
        return keypoints[:, indices, :]

    faltantes = num_frames_objetivo - num_frames_actual
    ultimo = keypoints[:, -1:, :].repeat(1, faltantes, 1)
    return torch.cat([keypoints, ultimo], dim=1)


class LSTMEnsemblePredictor:

    def __init__(self, db: Session, device: str = None):
        self.db = db
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelos_cargados: Dict[str, Dict] = {}
        self.clases_referencia: Optional[List[str]] = None
        self.cargar_modelos_activos()

    def cargar_modelos_activos(self):
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
        ruta_modelo = Path(modelo_db.ruta_archivo)

        if not ruta_modelo.exists():
            logger.error(f" Archivo no encontrado: {ruta_modelo}")
            return

        modelo, metadata = cargar_modelo_adaptativo(str(ruta_modelo), self.device)

        clases = metadata['clases']

        if self.clases_referencia is None:
            self.clases_referencia = clases
        elif set(clases) != set(self.clases_referencia):
            logger.error(
                f" Modelo {modelo_db.nombre} tiene clases distintas al resto del ensemble "
                f"({clases} vs {self.clases_referencia}); se omite para evitar mezclar predicciones incompatibles"
            )
            return

        accuracy = modelo_db.accuracy if modelo_db.accuracy is not None else metadata.get('accuracy', 0.5)
        peso = modelo_db.peso_ensemble if modelo_db.peso_ensemble is not None else 1.0

        self.modelos_cargados[modelo_db.nombre] = {
            'modelo': modelo,
            'peso': peso,
            'accuracy': accuracy,
            'clases': clases,
            'tipos_senas': metadata.get('tipos_senas', {}),
            'num_frames': metadata.get('num_frames', 20)
        }

        logger.info(f" Modelo cargado: {modelo_db.nombre} (peso: {peso}, acc: {accuracy:.4f})")

    @torch.no_grad()
    def predecir_ensemble(
        self,
        keypoints: torch.Tensor,
        tipo_sena: Optional[str] = None,
        metodo: str = "weighted_average",
        top_k: int = 3
    ) -> Dict:
        if not self.modelos_cargados:
            raise ValueError("No hay modelos activos para predicción")

        keypoints = keypoints.to(self.device)
        batch_size = keypoints.shape[0]
        predicciones_individuales = {}

        for nombre, info in self.modelos_cargados.items():
            try:
                modelo = info['modelo']
                peso = info['peso']
                accuracy = info['accuracy']
                clases = info['clases']
                num_frames_modelo = info['num_frames']

                keypoints_ajustados = _ajustar_num_frames(keypoints, num_frames_modelo)

                if tipo_sena is not None:
                    tipos_batch = [tipo_sena] * batch_size
                else:
                    tipos_batch = ['DINAMICA'] * batch_size

                logits = modelo.forward_batch_mixto(keypoints_ajustados, tipos_batch)
                probs = torch.softmax(logits, dim=1)[0]

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

        if metodo == "weighted_average":
            resultado = self._weighted_average(predicciones_individuales, top_k)
        elif metodo == "voting":
            resultado = self._majority_voting(predicciones_individuales, top_k)
        elif metodo == "stacking":
            resultado = self._stacking(predicciones_individuales, top_k)
        else:
            resultado = self._weighted_average(predicciones_individuales, top_k)

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
        suma_ponderada = None
        suma_pesos = 0.0
        clases_referencia = None

        for nombre, info in predicciones.items():
            peso_final = max(info['peso'] * info['accuracy'], 1e-6)
            probs = info['probabilidades']

            if clases_referencia is None:
                clases_referencia = info['clases']

            if suma_ponderada is None:
                suma_ponderada = probs * peso_final
            else:
                suma_ponderada += probs * peso_final

            suma_pesos += peso_final

        probabilidades_finales = suma_ponderada / suma_pesos

        top_k = min(top_k, len(probabilidades_finales))
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
        votos = {}
        clases_referencia = None

        for nombre, info in predicciones.items():
            clase_idx = info['clase_idx']
            peso = max(info['peso'] * info['accuracy'], 1e-6)

            if clases_referencia is None:
                clases_referencia = info['clases']

            votos[clase_idx] = votos.get(clase_idx, 0) + peso

        top_k = min(top_k, len(votos))
        clases_ordenadas = sorted(votos.items(), key=lambda x: x[1], reverse=True)[:top_k]

        clase_ganadora_idx = clases_ordenadas[0][0]

        confianzas = [
            info['confianza']
            for info in predicciones.values()
            if info['clase_idx'] == clase_ganadora_idx
        ]
        confianza_promedio = np.mean(confianzas) if confianzas else 0.0

        suma_votos = sum(votos.values()) or 1e-6

        return {
            'clase_idx': int(clase_ganadora_idx),
            'clase': clases_referencia[clase_ganadora_idx] if clases_referencia else str(clase_ganadora_idx),
            'confianza': float(confianza_promedio),
            'top_predicciones': [
                {
                    'clase': clases_referencia[idx] if clases_referencia else str(idx),
                    'votos': float(votos_val),
                    'probabilidad': float(votos_val / suma_votos)
                }
                for idx, votos_val in clases_ordenadas
            ],
            'metodo': 'majority_voting',
            'num_modelos': len(predicciones)
        }

    def _stacking(self, predicciones: Dict, top_k: int) -> Dict:
        mejor_nombre, mejor_info = max(
            predicciones.items(),
            key=lambda x: x[1]['accuracy']
        )

        clase_idx = mejor_info['clase_idx']
        clases = mejor_info['clases']

        top_k = min(top_k, len(mejor_info['probabilidades']))
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
        self.modelos_cargados.clear()
        self.clases_referencia = None
        self.cargar_modelos_activos()