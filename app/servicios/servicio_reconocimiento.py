import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List

import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn.functional as F

from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas
from app.servicios.config_tipo_senas import (
    detectar_tipo_sena,
    obtener_config_sena,
    calcular_factor_confusion_mejorado,
    redimensionar_manteniendo_aspecto
)

logger = logging.getLogger(__name__)


class ServicioReconocimiento:

    def __init__(self, ruta_modelo: str = None):
        self.model = None
        self.clases = []
        self.num_clases = 0
        self.num_frames = 20
        self.frames_estatica = 8
        self.accuracy = 0.0
        self.arquitectura = ""
        self.tipos_senas = {}
        self.modelo_dir = Path('modelos_entrenados')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_amp = torch.cuda.is_available()

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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
            self.accuracy = checkpoint.get('accuracy', 0.0)
            self.arquitectura = checkpoint.get('arquitectura', 'ModeloAdaptativoSenas')
            self.tipos_senas = checkpoint.get('tipos_senas', {})

            if not self.tipos_senas:
                self.tipos_senas = {clase: detectar_tipo_sena(clase) for clase in self.clases}

            self.model = ModeloAdaptativoSenas(self.num_clases, self.tipos_senas).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Modelo cargado: {ruta_modelo}")
            logger.info(f"Arquitectura: {self.arquitectura}")
            logger.info(f"Clases: {self.clases}")
            logger.info(f"Accuracy: {self.accuracy:.4f}")
            logger.info(f"Device: {self.device}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.model = None
            self.clases = []
            raise

    def _extraer_keypoints_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            resultado = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not resultado.multi_hand_landmarks:
                return None

            keypoints = []
            for mano in resultado.multi_hand_landmarks[:2]:
                for punto in mano.landmark:
                    keypoints.extend([punto.x, punto.y, punto.z])

            if len(keypoints) == 63:
                keypoints.extend([0.0] * 63)
            elif len(keypoints) != 126:
                return None

            return np.array(keypoints, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error extrayendo keypoints: {e}")
            return None

    def _indices_estaticos(self, total_frames: int, num_frames_extraer: int) -> np.ndarray:
        centro = total_frames // 2
        ventana = max(1, total_frames // 4)
        inicio = max(0, centro - ventana)
        fin = min(total_frames - 1, centro + ventana)
        if fin <= inicio:
            fin = min(total_frames - 1, inicio + 1)
        indices = np.linspace(inicio, fin, num_frames_extraer, dtype=int)
        return indices

    def _completar_keypoints(self, keypoints_list: List[np.ndarray]) -> List[np.ndarray]:
        while len(keypoints_list) < self.num_frames:
            keypoints_list.append(keypoints_list[-1].copy() if keypoints_list else np.zeros(126, dtype=np.float32))
        return keypoints_list[:self.num_frames]

    def _preparar_keypoints_por_tipo(self, keypoints_list: List[np.ndarray], tipo_sena: str) -> torch.Tensor:
        keypoints_list = list(keypoints_list)
        total = len(keypoints_list)

        if tipo_sena == 'ESTATICA':
            max_frames = min(self.num_frames, self.frames_estatica)
            if total > max_frames:
                centro = total // 2
                inicio = max(0, centro - max_frames // 2)
                fin = min(total, inicio + max_frames)
                keypoints_list = keypoints_list[inicio:fin]
        else:
            if total > self.num_frames:
                indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
                keypoints_list = [keypoints_list[i] for i in indices]

        keypoints_list = self._completar_keypoints(keypoints_list)
        return torch.from_numpy(np.stack(keypoints_list)).float().unsqueeze(0).to(self.device)

    def procesar_video(self, ruta_video: str, sena_esperada: str = None) -> Tuple[torch.Tensor, str]:
        cap = None
        try:
            if not os.path.exists(ruta_video):
                raise FileNotFoundError(f"Video no encontrado: {ruta_video}")

            cap = cv2.VideoCapture(ruta_video)
            if not cap.isOpened():
                raise ValueError(f"No se puede abrir video: {ruta_video}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Video sin frames válidos")

            tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'

            if tipo_sena == 'ESTATICA':
                frame_indices = self._indices_estaticos(total_frames, self.num_frames)
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

            keypoints_list = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()

                if ret and frame is not None:
                    frame = redimensionar_manteniendo_aspecto(frame, (224, 224))
                    keypoints = self._extraer_keypoints_frame(frame)
                    keypoints_list.append(keypoints if keypoints is not None else
                                            (keypoints_list[-1].copy() if keypoints_list else np.zeros(126, dtype=np.float32)))
                else:
                    keypoints_list.append(keypoints_list[-1].copy() if keypoints_list else np.zeros(126, dtype=np.float32))

            keypoints_list = self._completar_keypoints(keypoints_list)
            tensor = torch.from_numpy(np.stack(keypoints_list)).float().unsqueeze(0).to(self.device)

            return tensor, tipo_sena
        except Exception as e:
            logger.error(f"Error procesando video: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()

    def procesar_frame_individual(self, frame: np.ndarray, sena_esperada: str = None) -> Tuple[torch.Tensor, str]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'

        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        keypoints = self._extraer_keypoints_frame(frame)
        if keypoints is None:
            keypoints = np.zeros(126, dtype=np.float32)

        repeticiones = 5 if tipo_sena == 'ESTATICA' else self.num_frames
        keypoints_list = self._completar_keypoints([keypoints.copy() for _ in range(repeticiones)])

        tensor = torch.from_numpy(np.stack(keypoints_list)).float().unsqueeze(0).to(self.device)
        return tensor, tipo_sena

    def procesar_frames_secuencia(self, frames_list: List[np.ndarray], tipo_sena: str = 'DINAMICA') -> torch.Tensor:
        keypoints_list = []

        for frame in frames_list:
            frame = redimensionar_manteniendo_aspecto(frame, (224, 224))
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            keypoints = self._extraer_keypoints_frame(frame)
            keypoints_list.append(keypoints if keypoints is not None else
                                    (keypoints_list[-1].copy() if keypoints_list else np.zeros(126, dtype=np.float32)))

        return self._preparar_keypoints_por_tipo(keypoints_list, tipo_sena)

    def _ejecutar_forward(self, tensor_keypoints: torch.Tensor, tipos_batch: List[str]):
        try:
            if self.enable_amp:
                with torch.cuda.amp.autocast():
                    return self.model(tensor_keypoints, tipos_batch=tipos_batch)
            return self.model(tensor_keypoints, tipos_batch=tipos_batch)
        except RuntimeError as e:
            if "cuDNN" not in str(e):
                raise
            logger.warning("Error cuDNN detectado, reintentando...")
            torch.backends.cudnn.enabled = False
            try:
                if self.enable_amp:
                    with torch.cuda.amp.autocast():
                        return self.model(tensor_keypoints, tipos_batch=tipos_batch)
                return self.model(tensor_keypoints, tipos_batch=tipos_batch)
            finally:
                torch.backends.cudnn.enabled = True

    def predecir(self, tensor_keypoints: torch.Tensor, tipo_sena: str = None, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        if self.model is None or len(self.clases) == 0:
            return "desconocido", 0.0, {}

        try:
            if tensor_keypoints.device != self.device:
                tensor_keypoints = tensor_keypoints.to(self.device)

            tipo_sena = tipo_sena or 'DINAMICA'
            tipos_batch = [tipo_sena] * tensor_keypoints.shape[0]

            with torch.no_grad(), torch.inference_mode():
                outputs = self._ejecutar_forward(tensor_keypoints, tipos_batch)
                probabilities = F.softmax(outputs, dim=1)[0]

                if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                    logger.warning("NaN/Inf detectado en probabilidades")
                    probabilities = torch.ones_like(probabilities) / len(self.clases)

                probabilities_np = probabilities.cpu().numpy()

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
                factor_confusion = calcular_factor_confusion_mejorado(sena_esperada.upper(), sena_detectada.upper())

            detalles = {
                "tipo_sena": tipo_sena,
                "tipo_real": self.tipos_senas.get(sena_detectada, 'DINAMICA'),
                "alternativas": alternativas,
                "modelo": self.arquitectura,
                "num_frames_usado": self.num_frames,
                "factor_confusion": factor_confusion,
                "es_confusion_conocida": factor_confusion > 1.0,
                "modo": "keypoints_lstm"
            }

            return sena_detectada, confianza, detalles
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return "desconocido", 0.0, {"error": str(e), "alternativas": []}

    def predecir_adaptativo(self, keypoints_secuencia: List[np.ndarray], tipo_sena_inicial: str = None) -> Tuple[str, float, Dict]:
        tipo_sena_inicial = tipo_sena_inicial or 'DINAMICA'

        tensor_inicial = self._preparar_keypoints_por_tipo(keypoints_secuencia, tipo_sena_inicial)
        sena_detectada, confianza, detalles = self.predecir(tensor_inicial, tipo_sena_inicial, None)

        tipo_real = self.tipos_senas.get(sena_detectada, tipo_sena_inicial)

        if tipo_real != tipo_sena_inicial and sena_detectada != "desconocido" and len(self.clases) > 0:
            tensor_final = self._preparar_keypoints_por_tipo(keypoints_secuencia, tipo_real)
            sena_detectada_2, confianza_2, detalles_2 = self.predecir(tensor_final, tipo_real, None)

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
        tensor, tipo_sena = self.procesar_video(ruta_video, sena_esperada)
        return self.predecir(tensor, tipo_sena, sena_esperada)

    def predecir_desde_frame(self, frame: np.ndarray, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tensor, tipo_sena = self.procesar_frame_individual(frame, sena_esperada)
        return self.predecir(tensor, tipo_sena, sena_esperada)

    def predecir_desde_secuencia(self, frames_list: List[np.ndarray], sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        tensor = self.procesar_frames_secuencia(frames_list, tipo_sena)
        return self.predecir(tensor, tipo_sena, sena_esperada)

    def predecir_desde_keypoints(self, keypoints_list: List[np.ndarray], tipo_sena: str = None) -> Tuple[str, float, Dict]:
        try:
            tipo_sena = tipo_sena or 'DINAMICA'
            tensor = self._preparar_keypoints_por_tipo(keypoints_list, tipo_sena)
            return self.predecir(tensor, tipo_sena, None)
        except Exception as e:
            logger.error(f"Error prediciendo desde keypoints: {e}")
            return "desconocido", 0.0, {"error": str(e), "alternativas": []}

    def obtener_estadisticas(self) -> Dict:
        return {
            "total_senas": len(self.clases),
            "senas_disponibles": self.clases,
            "tipos_senas": self.tipos_senas,
            "precision_modelo": self.accuracy,
            "num_frames": self.num_frames,
            "dispositivo": str(self.device),
            "arquitectura": self.arquitectura,
            "amp_enabled": self.enable_amp,
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