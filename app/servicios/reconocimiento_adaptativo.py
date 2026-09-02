import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import cv2
import time
import logging
from typing import Tuple, Dict, Optional, List
import torch.nn.functional as F
import mediapipe as mp

from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas
from app.servicios.config_tipo_senas import detectar_tipo_sena, obtener_config_sena, es_confusion_numerica

logger = logging.getLogger(__name__)

def es_numero(sena: str) -> bool:
    if not sena:
        return False
    sena_clean = sena.strip().upper()
    numeros = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return sena_clean in numeros

class ReconocimientoAdaptativoIA:
    def __init__(self, ruta_modelo: str = None):
        self.model = None
        self.clases = []
        self.num_clases = 0
        self.num_frames = 20
        self.accuracy = 0.0
        self.arquitectura = ""
        self.tipos_senas = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo_dir = Path('modelos_entrenados')
        
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
                test_tensor = torch.randn(1, 20, 126).cuda()
                del test_tensor
                logger.info("CUDA inicializada correctamente")
            except Exception as e:
                logger.warning(f"Problema con CUDA: {e}")
                self.device = torch.device('cpu')
                self.enable_amp = False
        
        if ruta_modelo is None:
            pth_files = list(self.modelo_dir.glob('*.pth'))
            if not pth_files:
                raise ValueError("No se encontraron modelos .pth")
            ruta_modelo = max(pth_files, key=os.path.getctime)
            logger.info(f"Usando modelo más reciente: {ruta_modelo}")
        
        self.cargar_modelo(ruta_modelo)
    
    def cargar_modelo(self, ruta_modelo: str):
        try:
            if not Path(ruta_modelo).exists():
                raise FileNotFoundError(f"Modelo no encontrado: {ruta_modelo}")
            
            checkpoint = torch.load(
                ruta_modelo, 
                map_location=self.device,
                weights_only=False
            )
            
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
            
            logger.info(f"Modelo adaptativo cargado: {ruta_modelo}")
            logger.info(f"Arquitectura: {self.arquitectura}")
            logger.info(f"Clases: {self.clases}")
            logger.info(f"Tipos: {self.tipos_senas}")
            logger.info(f"Frames: {self.num_frames}")
            logger.info(f"Accuracy: {self.accuracy:.4f}")
            logger.info(f"Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self.model = None
            self.clases = []
            raise
    
    def _extraer_keypoints_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(frame_rgb)
            
            if not results.multi_hand_landmarks:
                return None
            
            keypoints = []
            
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
            
            if len(keypoints) == 63:
                keypoints.extend([0.0] * 63)
            elif len(keypoints) != 126:
                return None
            
            return np.array(keypoints, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extrayendo keypoints: {e}")
            return None
    
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
            config = obtener_config_sena(sena_esperada) if sena_esperada else None
            es_num = es_numero(sena_esperada) if sena_esperada else False
            
            if tipo_sena == 'ESTATICA':
                if config:
                    num_frames_extraer = config['num_frames_recomendado']
                else:
                    num_frames_extraer = 8 if es_num else 5
                
                frame_central = total_frames // 2
                offset = min(4, total_frames // 4)
                
                if num_frames_extraer <= 5:
                    indices_cercanos = [
                        max(0, frame_central - 2),
                        max(0, frame_central - 1),
                        frame_central,
                        min(total_frames - 1, frame_central + 1),
                        min(total_frames - 1, frame_central + 2)
                    ]
                else:
                    indices_cercanos = [
                        max(0, frame_central - offset*2),
                        max(0, frame_central - offset),
                        max(0, frame_central - 2),
                        max(0, frame_central - 1),
                        frame_central,
                        min(total_frames - 1, frame_central + 1),
                        min(total_frames - 1, frame_central + 2),
                        min(total_frames - 1, frame_central + offset)
                    ]
                
                frame_indices = np.array(indices_cercanos[:num_frames_extraer])
            else:
                num_frames_extraer = self.num_frames
                frame_indices = np.linspace(0, total_frames - 1, num_frames_extraer, dtype=int)
            
            keypoints_list = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
                    keypoints = self._extraer_keypoints_frame(frame)
                    
                    if keypoints is not None:
                        keypoints_list.append(keypoints)
                    else:
                        if len(keypoints_list) > 0:
                            keypoints_list.append(keypoints_list[-1].copy())
                        else:
                            keypoints_list.append(np.zeros(126, dtype=np.float32))
                else:
                    if len(keypoints_list) > 0:
                        keypoints_list.append(keypoints_list[-1].copy())
                    else:
                        keypoints_list.append(np.zeros(126, dtype=np.float32))
            
            while len(keypoints_list) < self.num_frames:
                if len(keypoints_list) > 0:
                    keypoints_list.append(keypoints_list[-1].copy())
                else:
                    keypoints_list.append(np.zeros(126, dtype=np.float32))
            
            keypoints_list = keypoints_list[:self.num_frames]
            
            keypoints_array = np.stack(keypoints_list)
            tensor = torch.from_numpy(keypoints_array).float()
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor, tipo_sena
            
        except Exception as e:
            logger.error(f"Error procesando video: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()
    
    def procesar_frame_individual(self, frame: np.ndarray, es_estatica: bool = True) -> Tuple[torch.Tensor, str]:
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        keypoints = self._extraer_keypoints_frame(frame)
        
        if keypoints is None:
            keypoints = np.zeros(126, dtype=np.float32)
        
        if es_estatica:
            keypoints_array = np.tile(keypoints, (5, 1))
            tipo_sena = 'ESTATICA'
        else:
            keypoints_array = np.tile(keypoints, (self.num_frames, 1))
            tipo_sena = 'DINAMICA'
        
        while keypoints_array.shape[0] < self.num_frames:
            keypoints_array = np.vstack([keypoints_array, keypoints])
        
        keypoints_array = keypoints_array[:self.num_frames]
        
        tensor = torch.from_numpy(keypoints_array).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor, tipo_sena
    
    def procesar_frames_secuencia(self, frames_list: List[np.ndarray], tipo_sena: str = 'DINAMICA') -> torch.Tensor:
        keypoints_list = []
        
        for frame in frames_list:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            keypoints = self._extraer_keypoints_frame(frame)
            
            if keypoints is not None:
                keypoints_list.append(keypoints)
            elif len(keypoints_list) > 0:
                keypoints_list.append(keypoints_list[-1].copy())
            else:
                keypoints_list.append(np.zeros(126, dtype=np.float32))
        
        if tipo_sena == 'ESTATICA':
            if len(keypoints_list) > 5:
                centro = len(keypoints_list) // 2
                indices = [
                    max(0, centro - 2),
                    max(0, centro - 1),
                    centro,
                    min(len(keypoints_list)-1, centro + 1),
                    min(len(keypoints_list)-1, centro + 2)
                ]
                keypoints_list = [keypoints_list[i] for i in indices]
        
        while len(keypoints_list) < self.num_frames:
            if len(keypoints_list) > 0:
                keypoints_list.append(keypoints_list[-1].copy())
            else:
                keypoints_list.append(np.zeros(126, dtype=np.float32))
        
        if len(keypoints_list) > self.num_frames:
            if tipo_sena == 'ESTATICA':
                centro = len(keypoints_list) // 2
                inicio = max(0, centro - self.num_frames//2)
                keypoints_list = keypoints_list[inicio:inicio+self.num_frames]
            else:
                indices = np.linspace(0, len(keypoints_list) - 1, self.num_frames, dtype=int)
                keypoints_list = [keypoints_list[i] for i in indices]
        
        keypoints_array = np.stack(keypoints_list[:self.num_frames])
        tensor = torch.from_numpy(keypoints_array).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predecir(self, tensor_keypoints: torch.Tensor, tipo_sena: str = None, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        if self.model is None or len(self.clases) == 0:
            return "desconocido", 0.0, {}
        
        try:
            if tensor_keypoints.device != self.device:
                tensor_keypoints = tensor_keypoints.to(self.device)
            
            if tipo_sena is None:
                tipo_sena = 'DINAMICA'
            
            batch_size = tensor_keypoints.shape[0]
            tipos_batch = [tipo_sena] * batch_size
            
            with torch.no_grad(), torch.inference_mode():
                try:
                    if self.enable_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(tensor_keypoints, tipos_batch=tipos_batch)
                    else:
                        outputs = self.model(tensor_keypoints, tipos_batch=tipos_batch)
                        
                except RuntimeError as e:
                    if "cuDNN" in str(e):
                        logger.warning("Error cuDNN detectado, reintentando...")
                        torch.backends.cudnn.enabled = False
                        try:
                            if self.enable_amp:
                                with torch.cuda.amp.autocast():
                                    outputs = self.model(tensor_keypoints, tipos_batch=tipos_batch)
                            else:
                                outputs = self.model(tensor_keypoints, tipos_batch=tipos_batch)
                        finally:
                            torch.backends.cudnn.enabled = True
                    else:
                        raise e
                
                probabilities = F.softmax(outputs, dim=1)[0]
                
                if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                    logger.warning("NaN/Inf detectado en probabilidades")
                    probabilities = torch.ones_like(probabilities) / len(self.clases)
                
                probabilities_np = probabilities.cpu().numpy()
            
            if sena_esperada and not es_numero(sena_esperada):
                indices_numeros = [i for i, clase in enumerate(self.clases) if es_numero(clase)]
                if indices_numeros and len(indices_numeros) < len(self.clases):
                    mask = np.ones_like(probabilities_np)
                    for idx in indices_numeros:
                        mask[idx] = 0.4
                    probabilities_np = probabilities_np * mask
                    suma = probabilities_np.sum()
                    if suma > 0:
                        probabilities_np = probabilities_np / suma
            
            elif sena_esperada and es_numero(sena_esperada):
                indices_numeros = [i for i, clase in enumerate(self.clases) if es_numero(clase)]
                if indices_numeros:
                    mask = np.ones_like(probabilities_np) * 0.5
                    for idx in indices_numeros:
                        mask[idx] = 1.2
                    probabilities_np = probabilities_np * mask
                    suma = probabilities_np.sum()
                    if suma > 0:
                        probabilities_np = probabilities_np / suma
            
            indice_predicho = int(np.argmax(probabilities_np))
            confianza = float(probabilities_np[indice_predicho])
            sena_detectada = self.clases[indice_predicho]
            
            top_k = min(5, len(self.clases))
            top_indices = np.argsort(probabilities_np)[-top_k:][::-1]
            
            alternativas = []
            for idx in top_indices:
                alternativas.append({
                    "sena": self.clases[int(idx)],
                    "confianza": float(probabilities_np[int(idx)]),
                    "tipo": self.tipos_senas.get(self.clases[int(idx)], 'DINAMICA'),
                    "es_numero": es_numero(self.clases[int(idx)])
                })
            
            es_confusion = False
            if sena_esperada and sena_detectada != sena_esperada:
                es_confusion = es_confusion_numerica(sena_esperada, sena_detectada)
            
            detalles = {
                "tipo_sena": tipo_sena,
                "tipo_real": self.tipos_senas.get(sena_detectada, 'DINAMICA'),
                "es_numero": es_numero(sena_detectada),
                "alternativas": alternativas,
                "modelo": self.arquitectura,
                "num_frames_usado": self.num_frames,
                "es_confusion_numerica": es_confusion,
                "modo": "keypoints_lstm"
            }
            
            return sena_detectada, confianza, detalles
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return "desconocido", 0.0, {"error": str(e), "alternativas": []}
    
    def predecir_desde_video(self, ruta_video: str, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tensor, tipo_sena = self.procesar_video(ruta_video, sena_esperada)
        return self.predecir(tensor, tipo_sena, sena_esperada)
    
    def predecir_desde_frame(self, frame: np.ndarray, sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        es_estatica = (tipo_sena == 'ESTATICA')
        tensor, tipo_sena = self.procesar_frame_individual(frame, es_estatica)
        return self.predecir(tensor, tipo_sena, sena_esperada)
    
    def predecir_desde_secuencia(self, frames_list: List[np.ndarray], sena_esperada: str = None) -> Tuple[str, float, Dict]:
        tipo_sena = detectar_tipo_sena(sena_esperada) if sena_esperada else 'DINAMICA'
        tensor = self.procesar_frames_secuencia(frames_list, tipo_sena)
        return self.predecir(tensor, tipo_sena, sena_esperada)
    
    def predecir_desde_keypoints(self, keypoints_list: List[np.ndarray], tipo_sena: str = None) -> Tuple[str, float, Dict]:
        try:
            if tipo_sena is None:
                tipo_sena = 'DINAMICA'
            
            if tipo_sena == 'ESTATICA':
                if len(keypoints_list) > 5:
                    centro = len(keypoints_list) // 2
                    indices = [
                        max(0, centro - 2),
                        max(0, centro - 1),
                        centro,
                        min(len(keypoints_list)-1, centro + 1),
                        min(len(keypoints_list)-1, centro + 2)
                    ]
                    keypoints_list = [keypoints_list[i] for i in indices]
            
            while len(keypoints_list) < self.num_frames:
                if len(keypoints_list) > 0:
                    keypoints_list.append(keypoints_list[-1].copy())
                else:
                    keypoints_list.append(np.zeros(126, dtype=np.float32))
            
            if len(keypoints_list) > self.num_frames:
                if tipo_sena == 'ESTATICA':
                    centro = len(keypoints_list) // 2
                    inicio = max(0, centro - self.num_frames//2)
                    keypoints_list = keypoints_list[inicio:inicio+self.num_frames]
                else:
                    indices = np.linspace(0, len(keypoints_list) - 1, self.num_frames, dtype=int)
                    keypoints_list = [keypoints_list[i] for i in indices]
            
            keypoints_array = np.stack(keypoints_list[:self.num_frames])
            tensor = torch.from_numpy(keypoints_array).float()
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return self.predecir(tensor, tipo_sena, None)
            
        except Exception as e:
            logger.error(f"Error prediciendo desde keypoints: {str(e)}")
            return "desconocido", 0.0, {"error": str(e), "alternativas": []}