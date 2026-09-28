import os
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas
from app.servicios.config_tipo_senas import (
    detectar_tipo_sena,
    calcular_factor_confusion_mejorado
)
from app.servicios.extractor_secuencias import (
    ResultadoSecuencia,
    crear_detector_manos,
    landmarks_de_frame,
    extraer_secuencia_landmarks,
    secuencia_desde_frames,
    secuencia_desde_keypoints,
    normalizar_fila,
)

logger = logging.getLogger(__name__)


def _variantes_tta(secuencia: np.ndarray) -> List[np.ndarray]:
    variantes = [secuencia]

    n = len(secuencia)
    if n >= 4:
        shift = max(1, n // 10)
        variantes.append(np.roll(secuencia, shift=shift, axis=0))
        variantes.append(np.roll(secuencia, shift=-shift, axis=0))

    angulos = [-3.0, 3.0]
    for grados in angulos:
        a = np.deg2rad(grados)
        c, s = float(np.cos(a)), float(np.sin(a))
        v = secuencia.copy()
        for i in range(len(v)):
            frame = v[i]
            for slot in range(2):
                ini, fin = slot * 63, slot * 63 + 63
                mano = frame[ini:fin].reshape(21, 3).copy()
                x = mano[:, 0].copy()
                y = mano[:, 1].copy()
                mano[:, 0] = x * c - y * s
                mano[:, 1] = x * s + y * c
                frame[ini:fin] = mano.flatten()
            v[i] = frame
        variantes.append(v)

    return variantes


class ServicioReconocimiento:

    def __init__(self, ruta_modelo: str = None):
        self.model = None
        self.clases = []
        self.num_clases = 0
        self.num_frames = 20
        self.frames_estatica = 8
        self.accuracy = 0.0
        self.arquitectura = ""
        self.extractor_version = None
        self.tipos_senas = {}
        self.modelo_dir = Path('modelos_entrenados')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_amp = torch.cuda.is_available()
        self.usar_tta = True
        self.temperatura = 0.4

        self._hands_frame = crear_detector_manos(
            estatico=True,
            confianza_deteccion=0.4,
            confianza_seguimiento=0.4,
        )
        self._lock_hands = threading.Lock()

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()
            try:
                prueba = torch.randn(1, 20, 126).cuda()
                del prueba
            except Exception as e:
                logger.warning(f"Problema con CUDA: {e}")
                self.device = torch.device('cpu')
                self.enable_amp = False

        if ruta_modelo is None:
            modelos = list(self.modelo_dir.glob('*.pth'))
            if not modelos:
                raise ValueError("No se encontraron modelos .pth")
            ruta_modelo = max(modelos, key=os.path.getctime)
            logger.info(f"Usando modelo más reciente: {ruta_modelo}")

        self.cargar_modelo(ruta_modelo)

    def cargar_modelo(self, ruta_modelo: str):
        try:
            if not Path(ruta_modelo).exists():
                raise FileNotFoundError(f"Modelo no encontrado: {ruta_modelo}")

            checkpoint = torch.load(ruta_modelo, map_location=self.device, weights_only=False)

            self.num_clases = checkpoint['num_clases']
            self.clases = checkpoint['clases']
            self.num_frames = checkpoint.get('num_frames', 20)
            self.frames_estatica = checkpoint.get('frames_estaticos', 8)
            self.accuracy = checkpoint.get('accuracy', 0.0)
            self.arquitectura = checkpoint.get('arquitectura', 'ModeloAdaptativoSenas')
            self.extractor_version = checkpoint.get('extractor_version')
            self.tipos_senas = checkpoint.get('tipos_senas', {})

            if not self.tipos_senas:
                self.tipos_senas = {clase: detectar_tipo_sena(clase) for clase in self.clases}

            self.model = ModeloAdaptativoSenas(
                self.num_clases, self.tipos_senas, self.frames_estatica
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Modelo cargado: {ruta_modelo}")
            logger.info(f"Clases: {self.clases} | Accuracy: {self.accuracy:.4f} | Device: {self.device}")
            logger.info(f"Temperatura softmax: {self.temperatura} | TTA: {self.usar_tta}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.model = None
            self.clases = []
            raise

    def _extraer_keypoints_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            with self._lock_hands:
                fila, presencia = landmarks_de_frame(frame, self._hands_frame)
            if not presencia.any():
                return None
            return normalizar_fila(fila)
        except Exception as e:
            logger.warning(f"Error extrayendo keypoints: {e}")
            return None

    def _tensor_desde_resultado(self, resultado: ResultadoSecuencia) -> torch.Tensor:
        return torch.from_numpy(resultado.secuencia).float().unsqueeze(0).to(self.device)

    def _resultado_desde_keypoints(self, keypoints_list: List[np.ndarray]) -> ResultadoSecuencia:
        return secuencia_desde_keypoints(keypoints_list, self.num_frames)

    def _resultado_desde_frames(self, frames_list: List[np.ndarray]) -> ResultadoSecuencia:
        return secuencia_desde_frames(frames_list, self.num_frames)

    def _analizar_video(self, ruta_video: str, sena_esperada: str = None) -> Tuple[ResultadoSecuencia, str]:
        if not os.path.exists(ruta_video):
            raise FileNotFoundError(f"Video no encontrado: {ruta_video}")

        resultado = extraer_secuencia_landmarks(ruta_video, self.num_frames)
        if resultado is None:
            raise ValueError(f"No se puede abrir video: {ruta_video}")

        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        return resultado, tipo_sena

    def _respuesta_sin_manos(self) -> Tuple[str, float, Dict]:
        return "desconocido", 0.0, {"error": "No se detectaron manos", "alternativas": []}

    def procesar_video(self, ruta_video: str, sena_esperada: str = None) -> Tuple[torch.Tensor, str]:
        resultado, tipo_sena = self._analizar_video(ruta_video, sena_esperada)
        return self._tensor_desde_resultado(resultado), tipo_sena

    def procesar_frame_individual(self, frame: np.ndarray, sena_esperada: str = None) -> Tuple[torch.Tensor, str]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'

        keypoints = self._extraer_keypoints_frame(frame)
        if keypoints is None:
            keypoints = np.zeros(126, dtype=np.float32)

        secuencia = np.tile(keypoints, (self.num_frames, 1)).astype(np.float32)
        tensor = torch.from_numpy(secuencia).float().unsqueeze(0).to(self.device)
        return tensor, tipo_sena

    def procesar_frames_secuencia(self, frames_list: List[np.ndarray], tipo_sena: str = 'DINAMICA') -> torch.Tensor:
        return self._tensor_desde_resultado(self._resultado_desde_frames(frames_list))

    def _ejecutar_forward(self, tensor_keypoints: torch.Tensor, tipos_batch: List[str]):
        try:
            if self.enable_amp:
                with torch.cuda.amp.autocast():
                    return self.model(tensor_keypoints, tipos_batch=tipos_batch)
            return self.model(tensor_keypoints, tipos_batch=tipos_batch)
        except RuntimeError as e:
            if "cuDNN" not in str(e):
                raise
            logger.warning("Error cuDNN, reintentando...")
            torch.backends.cudnn.enabled = False
            try:
                if self.enable_amp:
                    with torch.cuda.amp.autocast():
                        return self.model(tensor_keypoints, tipos_batch=tipos_batch)
                return self.model(tensor_keypoints, tipos_batch=tipos_batch)
            finally:
                torch.backends.cudnn.enabled = True

    def _predecir_tensor(self, tensor: torch.Tensor, tipo_sena: str) -> np.ndarray:
        tipos_batch = [tipo_sena] * tensor.shape[0]
        with torch.no_grad(), torch.inference_mode():
            outputs = self._ejecutar_forward(tensor, tipos_batch)

            temperatura = max(0.05, min(1.0, float(self.temperatura)))
            scaled = outputs / temperatura

            probs = F.softmax(scaled, dim=1)

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / len(self.clases)

            return probs.cpu().numpy()

    def predecir(self, tensor_keypoints: torch.Tensor, tipo_sena: str = None,
                 sena_esperada: str = None) -> Tuple[str, float, Dict]:
        if self.model is None or len(self.clases) == 0:
            return "desconocido", 0.0, {}

        try:
            if tensor_keypoints.device != self.device:
                tensor_keypoints = tensor_keypoints.to(self.device)

            tipo_sena = tipo_sena or 'DINAMICA'

            if self.usar_tta and tensor_keypoints.shape[0] == 1:
                secuencia_np = tensor_keypoints[0].cpu().numpy()
                variantes = _variantes_tta(secuencia_np)

                acumulado = np.zeros(len(self.clases), dtype=np.float64)
                for v in variantes:
                    t = torch.from_numpy(v).float().unsqueeze(0).to(self.device)
                    probs = self._predecir_tensor(t, tipo_sena)[0]
                    acumulado += probs

                probabilities_np = (acumulado / len(variantes)).astype(np.float32)
            else:
                probabilities_np = self._predecir_tensor(tensor_keypoints, tipo_sena)[0]

            indice_predicho = int(np.argmax(probabilities_np))
            confianza = float(probabilities_np[indice_predicho])
            sena_detectada = self.clases[indice_predicho]

            top_k = min(5, len(self.clases))
            top_indices = np.argsort(probabilities_np)[-top_k:][::-1]
            alternativas = [{
                "sena": self.clases[int(idx)],
                "confianza": float(probabilities_np[int(idx)]),
                "tipo": self.tipos_senas.get(self.clases[int(idx)], 'DINAMICA')
            } for idx in top_indices]

            factor_confusion = 1.0
            if sena_esperada and sena_detectada != sena_esperada:
                factor_confusion = calcular_factor_confusion_mejorado(
                    sena_esperada.upper(), sena_detectada.upper()
                )

            detalles = {
                "tipo_sena": tipo_sena,
                "tipo_real": self.tipos_senas.get(sena_detectada, 'DINAMICA'),
                "alternativas": alternativas,
                "modelo": self.arquitectura,
                "num_frames_usado": self.num_frames,
                "factor_confusion": factor_confusion,
                "es_confusion_conocida": factor_confusion > 1.0,
                "modo": "keypoints_lstm",
                "tta_aplicado": self.usar_tta and tensor_keypoints.shape[0] == 1,
                "temperatura": self.temperatura
            }

            return sena_detectada, confianza, detalles
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return "desconocido", 0.0, {"error": str(e), "alternativas": []}

    def predecir_adaptativo(self, keypoints_secuencia: List[np.ndarray],
                             tipo_sena_inicial: str = None) -> Tuple[str, float, Dict]:
        tipo_sena_inicial = tipo_sena_inicial or 'DINAMICA'

        resultado = self._resultado_desde_keypoints(keypoints_secuencia)
        if resultado.frames_detectados == 0:
            return self._respuesta_sin_manos()

        tensor = self._tensor_desde_resultado(resultado)
        sena_detectada, confianza, detalles = self.predecir(tensor, tipo_sena_inicial, None)

        tipo_real = self.tipos_senas.get(sena_detectada, tipo_sena_inicial)

        if tipo_real != tipo_sena_inicial and sena_detectada != "desconocido" and len(self.clases) > 0:
            sena_detectada_2, confianza_2, detalles_2 = self.predecir(tensor, tipo_real, None)
            if confianza_2 >= confianza:
                detalles_2["tipo_inicial"] = tipo_sena_inicial
                detalles_2["reevaluado"] = True
                return sena_detectada_2, confianza_2, detalles_2
            detalles["tipo_inicial"] = tipo_sena_inicial
            detalles["reevaluado"] = True
            return sena_detectada, confianza, detalles

        detalles["tipo_inicial"] = tipo_sena_inicial
        detalles["reevaluado"] = False
        return sena_detectada, confianza, detalles

    def predecir_desde_video(self, ruta_video: str, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        resultado, tipo_sena = self._analizar_video(ruta_video, sena_esperada)
        if resultado.frames_detectados == 0:
            return self._respuesta_sin_manos()
        return self.predecir(self._tensor_desde_resultado(resultado), tipo_sena, sena_esperada)

    def predecir_desde_frame(self, frame: np.ndarray, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        keypoints = self._extraer_keypoints_frame(frame)
        if keypoints is None:
            return self._respuesta_sin_manos()

        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        secuencia = np.tile(keypoints, (self.num_frames, 1)).astype(np.float32)
        tensor = torch.from_numpy(secuencia).float().unsqueeze(0).to(self.device)
        return self.predecir(tensor, tipo_sena, sena_esperada)

    def predecir_desde_secuencia(self, frames_list: List[np.ndarray], sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        resultado = self._resultado_desde_frames(frames_list)
        if resultado.frames_detectados == 0:
            return self._respuesta_sin_manos()
        return self.predecir(self._tensor_desde_resultado(resultado), tipo_sena, sena_esperada)

    def predecir_desde_keypoints(self, keypoints_list: List[np.ndarray],
                                  tipo_sena: str = None) -> Tuple[str, float, Dict]:
        try:
            tipo_sena = tipo_sena or 'DINAMICA'
            resultado = self._resultado_desde_keypoints(keypoints_list)
            if resultado.frames_detectados == 0:
                return self._respuesta_sin_manos()
            return self.predecir(self._tensor_desde_resultado(resultado), tipo_sena, None)
        except Exception as e:
            logger.error(f"Error prediciendo desde keypoints: {e}")
            return "desconocido", 0.0, {"error": str(e), "alternativas": []}

    def ajustar_temperatura(self, temperatura: float) -> float:
        self.temperatura = max(0.05, min(1.0, float(temperatura)))
        logger.info(f"Temperatura ajustada a {self.temperatura}")
        return self.temperatura

    def obtener_estadisticas(self) -> Dict:
        return {
            "total_senas": len(self.clases),
            "senas_disponibles": self.clases,
            "tipos_senas": self.tipos_senas,
            "precision_modelo": self.accuracy,
            "num_frames": self.num_frames,
            "dispositivo": str(self.device),
            "arquitectura": self.arquitectura,
            "extractor_version": self.extractor_version,
            "amp_enabled": self.enable_amp,
            "tta_habilitado": self.usar_tta,
            "temperatura": self.temperatura,
            "ultima_actualizacion": datetime.now().isoformat()
        }

    def validar_video(self, ruta_video: str) -> Dict:
        cap = None
        try:
            cap = cv2.VideoCapture(ruta_video)
            if not cap.isOpened():
                return {"valido": False, "razon": "No se puede abrir video"}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duracion = total_frames / fps if fps > 0 else 0

            if duracion < 0.5:
                return {"valido": False, "razon": "Video muy corto (< 0.5s)"}

            if total_frames < self.num_frames:
                return {
                    "valido": True,
                    "advertencia": f"Video tiene {total_frames} frames, modelo usa {self.num_frames}",
                    "duracion": duracion
                }

            return {"valido": True, "duracion": duracion, "fps": fps, "total_frames": total_frames}
        except Exception as e:
            return {"valido": False, "razon": str(e)}
        finally:
            if cap is not None:
                cap.release()

    def benchmark(self, num_iteraciones: int = 10) -> Dict:
        if self.model is None:
            return {"error": "Modelo no cargado"}

        try:
            dummy_input = torch.randn(1, self.num_frames, 126).to(self.device)
            tipos_batch = ['DINAMICA']

            with torch.no_grad():
                for _ in range(3):
                    self._ejecutar_forward(dummy_input, tipos_batch)

            tiempos = []
            for _ in range(num_iteraciones):
                inicio = time.perf_counter()
                with torch.no_grad():
                    self._ejecutar_forward(dummy_input, tipos_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                tiempos.append(time.perf_counter() - inicio)

            return {
                "tiempo_promedio_ms": round(np.mean(tiempos) * 1000, 2),
                "tiempo_min_ms": round(np.min(tiempos) * 1000, 2),
                "tiempo_max_ms": round(np.max(tiempos) * 1000, 2),
                "fps_teorico": round(1.0 / np.mean(tiempos), 2),
                "iteraciones": num_iteraciones,
                "dispositivo": str(self.device),
                "amp_enabled": self.enable_amp,
                "arquitectura": self.arquitectura
            }
        except Exception as e:
            logger.error(f"Error en benchmark: {e}")
            return {"error": str(e)}