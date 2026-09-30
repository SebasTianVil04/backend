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

        logger.info(f" Modelo cargado: {modelo_db.nombre} (peso: {peso}, acc: {accuracy:.4f}, clases: {len(clases)})")

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
        # Unión de todas las clases de todos los modelos
        clases_union = set()
        for info in predicciones.values():
            clases_union.update(info['clases'])
        clases_union = sorted(clases_union)
        idx_union = {c: i for i, c in enumerate(clases_union)}

        suma_ponderada = np.zeros(len(clases_union), dtype=np.float64)
        suma_pesos = 0.0

        for nombre, info in predicciones.items():
            peso_final = max(info['peso'] * info['accuracy'], 1e-6)
            probs = info['probabilidades']
            clases_modelo = info['clases']

            probs_union = np.zeros(len(clases_union), dtype=np.float64)
            for j, clase in enumerate(clases_modelo):
                destino = idx_union.get(clase)
                if destino is not None:
                    probs_union[destino] = float(probs[j])

            suma_ponderada += probs_union * peso_final
            suma_pesos += peso_final

        probabilidades_finales = suma_ponderada / suma_pesos

        top_k = min(top_k, len(probabilidades_finales))
        top_indices = np.argsort(probabilidades_finales)[-top_k:][::-1]

        return {
            'clase_idx': int(top_indices[0]),
            'clase': clases_union[top_indices[0]],
            'confianza': float(probabilidades_finales[top_indices[0]]),
            'top_predicciones': [
                {
                    'clase': clases_union[idx],
                    'probabilidad': float(probabilidades_finales[idx])
                }
                for idx in top_indices
            ],
            'clases_union': clases_union,
            'metodo': 'weighted_average',
            'num_modelos': len(predicciones)
        }

    def _majority_voting(self, predicciones: Dict, top_k: int) -> Dict:
        votos: Dict[str, float] = {}
        confianzas_por_clase: Dict[str, List[float]] = {}

        for nombre, info in predicciones.items():
            clase = info['clases'][info['clase_idx']]
            peso = max(info['peso'] * info['accuracy'], 1e-6)
            votos[clase] = votos.get(clase, 0.0) + peso
            confianzas_por_clase.setdefault(clase, []).append(info['confianza'])

        top_k = min(top_k, len(votos))
        clases_ordenadas = sorted(votos.items(), key=lambda x: x[1], reverse=True)[:top_k]

        clase_ganadora = clases_ordenadas[0][0]
        confianza_promedio = float(np.mean(confianzas_por_clase[clase_ganadora]))

        suma_votos = sum(votos.values()) or 1e-6

        return {
            'clase_idx': 0,
            'clase': clase_ganadora,
            'confianza': confianza_promedio,
            'top_predicciones': [
                {
                    'clase': clase,
                    'votos': float(v),
                    'probabilidad': float(v / suma_votos)
                }
                for clase, v in clases_ordenadas
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
        self.cargar_modelos_activos()