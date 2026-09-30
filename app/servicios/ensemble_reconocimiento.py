import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .config_tipo_senas import calcular_factor_confusion_mejorado

logger = logging.getLogger(__name__)


@dataclass
class MiembroEnsemble:
    modelo_id: int
    nombre: str
    peso: float
    reconocedor: Any  # ServicioReconocimiento


def _ajustar_frames(tensor: torch.Tensor, n: int) -> torch.Tensor:
    actual = tensor.shape[1]
    if actual == n:
        return tensor
    idx = torch.linspace(0, actual - 1, n).round().long().to(tensor.device)
    return tensor[:, idx, :]


class ReconocedorEnsemble:
    """
    Misma interfaz que ServicioReconocimiento, pero combina las probabilidades
    de varios modelos.

    Los modelos PUEDEN tener conjuntos de clases distintos (p. ej. uno entrenado
    con el alfabeto A-D y otro con señas como SOS). Las probabilidades de cada
    modelo se alinean por NOMBRE de clase sobre la UNIÓN de todas las clases
    presentes; las clases que un modelo no conoce reciben probabilidad 0.
    """

    es_ensemble = True

    def __init__(self, miembros: List[MiembroEnsemble]):
        if len(miembros) < 2:
            raise ValueError("Un ensemble necesita al menos 2 modelos")

        self.miembros = miembros
        self._ref = miembros[0].reconocedor
        self.modelo_ids = [m.modelo_id for m in miembros]

        # Unión ordenada de todas las clases de todos los modelos
        clases_union = set()
        for m in miembros:
            clases_union.update(m.reconocedor.clases)
        self.clases = sorted(clases_union)
        self.num_clases = len(self.clases)

        # Para cada modelo, un vector de mapeo union_idx -> model_idx (-1 si no la tiene)
        self._mapeos: List[np.ndarray] = []
        for m in miembros:
            mapa = np.full(self.num_clases, -1, dtype=np.int64)
            clases_modelo = list(m.reconocedor.clases)
            for union_idx, clase in enumerate(self.clases):
                if clase in clases_modelo:
                    mapa[union_idx] = clases_modelo.index(clase)
            self._mapeos.append(mapa)

        self.num_frames = self._ref.num_frames
        self.device = self._ref.device
        self.enable_amp = self._ref.enable_amp
        self.extractor_version = self._ref.extractor_version

        self.tipos_senas: Dict[str, str] = {}
        for m in miembros:
            self.tipos_senas.update(m.reconocedor.tipos_senas)

        pesos = np.array(
            [max(m.peso * (m.reconocedor.accuracy or 0.5), 1e-6) for m in miembros],
            dtype=np.float64,
        )
        self._pesos = pesos / pesos.sum()
        self.accuracy = float(
            sum(p * (m.reconocedor.accuracy or 0.0) for p, m in zip(self._pesos, miembros))
        )
        self.arquitectura = f"Ensemble({len(miembros)}): " + ", ".join(m.nombre for m in miembros)

    # --- Preprocesamiento: delega en el modelo de referencia ---
    def _resultado_desde_keypoints(self, keypoints_list):
        return self._ref._resultado_desde_keypoints(keypoints_list)

    def _resultado_desde_frames(self, frames_list):
        return self._ref._resultado_desde_frames(frames_list)

    def _tensor_desde_resultado(self, resultado):
        return self._ref._tensor_desde_resultado(resultado)

    # --- Predicción ---
    def predecir(self, tensor_keypoints: torch.Tensor, tipo_sena: str = None,
                 sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tipo_sena = tipo_sena or 'DINAMICA'
        resultados = []  # (miembro, peso, probs en espacio unión)

        for miembro, peso, mapa in zip(self.miembros, self._pesos, self._mapeos):
            rec = miembro.reconocedor
            try:
                t = _ajustar_frames(tensor_keypoints.to(rec.device), rec.num_frames)
                probs_raw = rec.probabilidades(t, tipo_sena)

                # Proyectar probabilidades del modelo a la unión de clases
                probs_union = np.zeros(self.num_clases, dtype=np.float64)
                for union_idx in range(self.num_clases):
                    model_idx = int(mapa[union_idx])
                    if model_idx >= 0:
                        probs_union[union_idx] = float(probs_raw[model_idx])

                resultados.append((miembro, float(peso), probs_union))
            except Exception as e:
                logger.error(f"Ensemble: falló {miembro.nombre}: {e}")

        if not resultados:
            return "desconocido", 0.0, {"error": "Ningún modelo pudo predecir", "alternativas": []}

        suma = sum(p for _, p, _ in resultados)
        combinada = sum(p * probs for _, p, probs in resultados) / suma

        idx = int(np.argmax(combinada))
        sena = self.clases[idx]
        confianza = float(combinada[idx])

        top = np.argsort(combinada)[-min(5, len(self.clases)):][::-1]
        alternativas = [{
            "sena": self.clases[int(i)],
            "confianza": float(combinada[int(i)]),
            "tipo": self.tipos_senas.get(self.clases[int(i)], 'DINAMICA'),
        } for i in top]

        factor = 1.0
        if sena_esperada and sena != sena_esperada:
            factor = calcular_factor_confusion_mejorado(sena_esperada.upper(), sena.upper())

        detalles = {
            "tipo_sena": tipo_sena,
            "tipo_real": self.tipos_senas.get(sena, 'DINAMICA'),
            "alternativas": alternativas,
            "modelo": self.arquitectura,
            "num_frames_usado": self.num_frames,
            "factor_confusion": factor,
            "es_confusion_conocida": factor > 1.0,
            "modo": "ensemble_keypoints_lstm",
            "modelos_ensemble": [{
                "modelo": m.nombre,
                "sena": self.clases[int(np.argmax(probs))],
                "confianza": round(float(np.max(probs)), 4),
                "peso": round(p / suma, 4),
                "clases_conocidas": list(m.reconocedor.clases),
            } for m, p, probs in resultados],
        }
        return sena, confianza, detalles