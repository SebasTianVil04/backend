import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sqlalchemy.orm import Session
from sklearn.model_selection import StratifiedShuffleSplit
import gc
import time
import cv2
import random
from collections import Counter
from app.servicios.drive_service import (
    respaldar_modelo_en_drive,
    descargar_videos_temporalmente,
    limpiar_carpeta_temporal
)
from app.modelos.entrenamiento import ModeloIA
from app.modelos.dataset import VideoDataset
from app.utilidades.base_datos import SessionLocal

logger = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

progresos_entrenamiento: Dict[str, Any] = {}

def normalizar_imagenet(frame: np.ndarray) -> np.ndarray:
    frame_float = frame.astype(np.float32) / 255.0
    frame_float = np.clip(frame_float, 0.0, 1.0)
    frame_normalized = (frame_float - MEAN) / STD
    return frame_normalized

class AdvancedVideoAugmentation:
    def __init__(self, is_training=True, num_frames=20):
        self.is_training = is_training
        self.num_frames = num_frames
    
    def apply(self, frames: np.ndarray) -> np.ndarray:
        if not self.is_training:
            return frames
        
        frames = self._random_brightness_contrast(frames)
        frames = self._random_flip(frames)
        frames = self._random_rotation(frames)
        frames = self._random_scale(frames)
        frames = self._random_translation(frames)
        frames = self._random_noise(frames)
        frames = self._random_blur(frames)
        frames = self._random_color_jitter(frames)
        
        return frames
    
    def _random_brightness_contrast(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.3:
            brightness_factor = random.uniform(0.5, 1.5)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
            
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.6, 1.4)
                mean = frames.mean(axis=(1, 2), keepdims=True)
                frames = np.clip((frames - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        return frames
    
    def _random_flip(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        return frames
    
    def _random_rotation(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.3:
            angle = random.uniform(-20, 20)
            h, w = frames.shape[1:3]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            frames = np.array([cv2.warpAffine(frame, M, (w, h), 
                                             borderMode=cv2.BORDER_REFLECT) 
                              for frame in frames])
        return frames
    
    def _random_scale(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.3:
            scale = random.uniform(0.8, 1.2)
            h, w = frames.shape[1:3]
            new_h, new_w = int(h * scale), int(w * scale)
            
            scaled_frames = []
            for frame in frames:
                scaled = cv2.resize(frame, (new_w, new_h))
                
                if scale < 1.0:
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    padded = cv2.copyMakeBorder(scaled, pad_h, h-new_h-pad_h, 
                                               pad_w, w-new_w-pad_w, 
                                               cv2.BORDER_REFLECT)
                else:
                    crop_h = (new_h - h) // 2
                    crop_w = (new_w - w) // 2
                    padded = scaled[crop_h:crop_h+h, crop_w:crop_w+w]
                
                scaled_frames.append(padded)
            
            frames = np.array(scaled_frames)
        return frames
    
    def _random_translation(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.4:
            h, w = frames.shape[1:3]
            tx = random.randint(-int(w*0.1), int(w*0.1))
            ty = random.randint(-int(h*0.1), int(h*0.1))
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            frames = np.array([cv2.warpAffine(frame, M, (w, h), 
                                             borderMode=cv2.BORDER_REFLECT) 
                              for frame in frames])
        return frames
    
    def _random_noise(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            noise = np.random.normal(0, 10, frames.shape).astype(np.int16)
            frames = np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frames
    
    def _random_blur(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.6:
            kernel_size = random.choice([3, 5])
            frames = np.array([cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0) 
                              for frame in frames])
        return frames
    
    def _random_color_jitter(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.4:
            hsv_frames = []
            for frame in frames:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
                hsv = hsv.astype(np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                hsv_frames.append(rgb)
            
            frames = np.array(hsv_frames)
        return frames

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = SpatialAttention()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ImprovedCNNVideo(nn.Module):
    """Arquitectura mejorada con LSTM"""
    def __init__(self, num_classes: int):
        super(ImprovedCNNVideo, self).__init__()
        
        if num_classes < 2:
            raise ValueError(f"Se requieren al menos 2 clases, recibidas {num_classes}")
        
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.spatial_features(x)
        x = x.view(batch_size, num_frames, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class VideoFrameDataset(Dataset):
    def __init__(self, video_paths: List[str], labels: List[int], 
                 num_frames: int = 20, is_training: bool = True):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.is_training = is_training
        self.augmentation = AdvancedVideoAugmentation(is_training, num_frames)
        
        if len(video_paths) != len(labels):
            raise ValueError(f"Mismatch: {len(video_paths)} paths vs {len(labels)} labels")
        
        logger.info(f"Dataset: {len(video_paths)} videos, {num_frames} frames/video, training={is_training}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            frames = self._load_video_frames(video_path)
            
            if frames.shape[0] != self.num_frames:
                if frames.shape[0] < self.num_frames:
                    frames = np.pad(frames, 
                                  ((0, self.num_frames - frames.shape[0]), (0,0), (0,0), (0,0)),
                                  mode='edge')
                else:
                    frames = frames[:self.num_frames]
            
            frames = self.augmentation.apply(frames)
            frames_normalized = normalizar_imagenet(frames)
            
            frames_tensor = torch.from_numpy(
                frames_normalized.transpose(0, 3, 1, 2).copy()
            ).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            expected_shape = (self.num_frames, 3, 224, 224)
            if frames_tensor.shape != expected_shape:
                raise ValueError(f"Shape invalido: {frames_tensor.shape}")
            
            return frames_tensor, label_tensor
            
        except Exception as e:
            logger.error(f"Error procesando video {video_path}: {e}")
            dummy_frames = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            dummy_normalized = normalizar_imagenet(dummy_frames)
            dummy_tensor = torch.from_numpy(
                dummy_normalized.transpose(0, 3, 1, 2).copy()
            ).float()
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_tensor, dummy_label

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        cap = None
        try:
            if not os.path.exists(video_path):
                return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            
            if total_frames <= self.num_frames:
                frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    if len(frames) > 0:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            while len(frames) < self.num_frames:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            frames_array = np.array(frames[:self.num_frames], dtype=np.uint8)
            
            return frames_array
            
        except Exception as e:
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        finally:
            if cap is not None:
                cap.release()

class EntrenamientoModeloService:
    def __init__(self):
        self.pytorch_disponible = torch is not None
        self.modelo_dir = Path('modelos_entrenados')
        self.modelo_dir.mkdir(exist_ok=True)
        
        self.device, self.gpu_info = self.detectar_y_configurar_gpu()
        
        if torch.cuda.is_available():
            memoria_gpu = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if memoria_gpu >= 8:
                self.batch_size = 12
                self.num_frames = 24
            elif memoria_gpu >= 6:
                self.batch_size = 8
                self.num_frames = 20
            elif memoria_gpu >= 4:
                self.batch_size = 6
                self.num_frames = 16
            else:
                self.batch_size = 4
                self.num_frames = 16
            
            import platform
            self.num_workers = 0 if platform.system() == 'Windows' else 4
        else:
            self.batch_size = 2
            self.num_frames = 16
            self.num_workers = 0
        
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.min_videos_por_sena = 8
        self.min_accuracy = 0.50
        
        logger.info(f"Servicio inicializado en {self.device}")
        logger.info(f"Configuración: batch_size={self.batch_size}, frames={self.num_frames}")

    def detectar_y_configurar_gpu(self) -> Tuple[torch.device, Optional[Dict[str, Any]]]:
        logger.info("Detectando GPU...")
        
        if not torch.cuda.is_available():
            return torch.device('cpu'), None
        
        gpu_idx = 0
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f'cuda:{gpu_idx}')
        
        nombre = torch.cuda.get_device_name(gpu_idx)
        props = torch.cuda.get_device_properties(gpu_idx)
        memoria_total = props.total_memory / (1024**3)
        
        logger.info(f"GPU: {nombre} - {memoria_total:.2f} GB")
        
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        return device, {"nombre": nombre, "memoria_total_gb": round(memoria_total, 2)}

    def generar_nombre_modelo(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"modelo_video_{timestamp}"

    def validar_dataset_entrenamiento(self, categoria_ids: List[int], db: Session) -> Dict[str, Any]:
        try:
            videos_db = db.query(VideoDataset).filter(
                VideoDataset.aprobado == True,
                VideoDataset.categoria_id.in_(categoria_ids)
            ).all()
            
            videos_validos = []
            videos_no_encontrados = []
            
            for video in videos_db:
                tiene_local = video.ruta_video and os.path.exists(video.ruta_video)
                tiene_drive = bool(video.drive_file_id)
                if tiene_local or tiene_drive:
                    videos_validos.append(video)
                else:
                    videos_no_encontrados.append(video.ruta_video or f"id={video.id} (sin ruta ni drive)")
            
            videos_por_sena = {}
            for video in videos_validos:
                sena = video.sena or "sin_clasificar"
                if sena not in videos_por_sena:
                    videos_por_sena[sena] = []
                videos_por_sena[sena].append(video)
            
            total_videos = len(videos_validos)
            total_senas = len(videos_por_sena)
            
            senas_con_minimo = []
            senas_sin_minimo = []
            
            for sena, videos in videos_por_sena.items():
                if len(videos) >= self.min_videos_por_sena:
                    senas_con_minimo.append(sena)
                else:
                    senas_sin_minimo.append(sena)
            
            valido = total_videos >= 20 and len(senas_con_minimo) >= 2
            
            resultado = {
                "valido": valido,
                "total_videos": total_videos,
                "videos_validos": len(videos_validos),
                "videos_no_encontrados": len(videos_no_encontrados),
                "total_senas": total_senas,
                "senas_con_minimo": senas_con_minimo,
                "senas_sin_minimo": senas_sin_minimo,
                "minimo_videos_requerido": 20,
                "minimo_senas_requerido": 2,
                "minimo_videos_por_sena": self.min_videos_por_sena,
                "cumple_requisitos": {
                    "videos_totales": total_videos >= 20,
                    "senas_suficientes": len(senas_con_minimo) >= 2,
                    "videos_por_sena": len(senas_sin_minimo) == 0
                }
            }
            
            return resultado
            
        except Exception as e:
            return {
                "valido": False,
                "error": str(e),
                "total_videos": 0,
                "videos_validos": 0,
                "total_senas": 0,
                "senas_con_minimo": [],
                "senas_sin_minimo": []
            }

    def obtener_progreso_entrenamiento(self, nombre_modelo: str) -> Dict[str, Any]:
        global progresos_entrenamiento
        
        if nombre_modelo in progresos_entrenamiento:
            progress = progresos_entrenamiento[nombre_modelo]
            resultado = {
                "nombre_modelo": progress.get("nombre_modelo", nombre_modelo),
                "estado": progress.get("estado", "desconocido"),
                "progreso": progress.get("progreso", 0.0),
                "epoch_actual": progress.get("epoch_actual", 0),
                "total_epochs": progress.get("total_epochs", 0),
                "accuracy": progress.get("accuracy", 0.0),
                "loss": progress.get("loss", 0.0),
                "train_loss": progress.get("train_loss", 0.0),
                "train_accuracy": progress.get("train_accuracy", 0.0),
                "num_clases": progress.get("num_clases", 0),
                "clases": progress.get("clases", []),
                "total_videos": progress.get("total_videos", 0),
                "frames_procesados": progress.get("frames_procesados", 0),
                "total_frames": progress.get("total_frames", 0),
                "mensaje": progress.get("mensaje", ""),
                "fecha_inicio": progress.get("fecha_inicio", datetime.now().isoformat()),
                "entrenando": progress.get("entrenando", False)
            }
            return resultado
        
        db = SessionLocal()
        try:
            modelo_db = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()
            
            if modelo_db:
                clases = []
                if modelo_db.clases_json:
                    try:
                        clases = json.loads(modelo_db.clases_json)
                    except:
                        clases = []
                
                resultado = {
                    "nombre_modelo": modelo_db.nombre,
                    "estado": "completado" if modelo_db.activo else "inactivo",
                    "progreso": 100.0 if modelo_db.activo else 0.0,
                    "epoch_actual": modelo_db.epocas_entrenamiento or 0,
                    "total_epochs": modelo_db.epocas_entrenamiento or 0,
                    "accuracy": float(modelo_db.accuracy) if modelo_db.accuracy else 0.0,
                    "loss": 0.0,
                    "train_loss": 0.0,
                    "train_accuracy": 0.0,
                    "num_clases": modelo_db.num_clases or 0,
                    "clases": clases,
                    "total_videos": modelo_db.total_imagenes or 0,
                    "frames_procesados": (modelo_db.total_imagenes or 0) * self.num_frames,
                    "total_frames": (modelo_db.total_imagenes or 0) * self.num_frames,
                    "fecha_entrenamiento": modelo_db.fecha_entrenamiento.isoformat() if modelo_db.fecha_entrenamiento else None,
                    "mensaje": "Completado" if modelo_db.activo else "Inactivo",
                    "entrenando": False
                }
                return resultado
            
            else:
                resultado = {
                    "nombre_modelo": nombre_modelo,
                    "estado": "no_encontrado",
                    "progreso": 0.0,
                    "epoch_actual": 0,
                    "total_epochs": 0,
                    "accuracy": 0.0,
                    "loss": 0.0,
                    "train_loss": 0.0,
                    "train_accuracy": 0.0,
                    "num_clases": 0,
                    "clases": [],
                    "total_videos": 0,
                    "frames_procesados": 0,
                    "total_frames": 0,
                    "mensaje": f"Modelo '{nombre_modelo}' no encontrado en sistema",
                    "entrenando": False
                }
                return resultado
        
        except Exception as e:
            return {
                "nombre_modelo": nombre_modelo,
                "estado": "error",
                "progreso": 0.0,
                "accuracy": 0.0,
                "loss": 0.0,
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "num_clases": 0,
                "clases": [],
                "total_videos": 0,
                "frames_procesados": 0,
                "total_frames": 0,
                "mensaje": f"Error consultando: {str(e)}",
                "entrenando": False
            }
        
        finally:
            db.close()

    def entrenar(self, nombre_modelo: str, categoria_ids: List[int], epochs: int):
        global progresos_entrenamiento
        
        db = None
        model = None
        train_loader = None
        val_loader = None
        
        try:
            if not epochs or epochs <= 0:
                epochs = 150
            elif epochs > 500:
                epochs = 500
            
            progresos_entrenamiento[nombre_modelo] = {
                "nombre_modelo": nombre_modelo,
                "estado": "iniciando",
                "progreso": 0.0,
                "epoch_actual": 0,
                "total_epochs": epochs,
                "accuracy": 0.0,
                "loss": 0.0,
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "num_clases": 0,
                "clases": [],
                "total_videos": 0,
                "frames_procesados": 0,
                "total_frames": 0,
                "mensaje": "Inicializando...",
                "fecha_inicio": datetime.now().isoformat(),
                "entrenando": True
            }
            
            db = SessionLocal()
            
            videos_db = db.query(VideoDataset).filter(
                VideoDataset.aprobado == True,
                VideoDataset.categoria_id.in_(categoria_ids)
            ).all()
            
            videos_locales = [v for v in videos_db if v.ruta_video and os.path.exists(v.ruta_video)]
            videos_solo_drive = [
                v for v in videos_db
                if v.drive_file_id and not (v.ruta_video and os.path.exists(v.ruta_video))
            ]
            
            progresos_entrenamiento[nombre_modelo]["mensaje"] = (
                f"Descargando {len(videos_solo_drive)} videos desde Drive..."
            )
            
            rutas_temporales = {}
            if videos_solo_drive:
                rutas_temporales = descargar_videos_temporalmente(
                    videos_solo_drive, carpeta_temporal=f"temp_entrenamiento_{nombre_modelo}"
                )
            
            videos_validos = list(videos_locales)
            for v in videos_solo_drive:
                if v.id in rutas_temporales:
                    v.ruta_video = rutas_temporales[v.id]
                    videos_validos.append(v)
            
            if len(videos_validos) < 20:
                raise Exception(f"Insuficientes videos: {len(videos_validos)} (mínimo 20)")
            
            videos_por_sena = {}
            for video in videos_validos:
                sena = video.sena or "sin_clasificar"
                if sena not in videos_por_sena:
                    videos_por_sena[sena] = []
                videos_por_sena[sena].append(video)
            
            senas_validas = {k: v for k, v in videos_por_sena.items() 
                        if len(v) >= self.min_videos_por_sena}
            
            if len(senas_validas) < 2:
                raise Exception(
                    f"Se necesitan al menos 2 señas con {self.min_videos_por_sena}+ videos. "
                    f"Señas válidas: {list(senas_validas.keys())}"
                )
            
            video_paths = []
            labels = []
            clases = sorted(senas_validas.keys())
            clase_a_idx = {clase: i for i, clase in enumerate(clases)}
            
            for clase, videos_clase in senas_validas.items():
                for video in videos_clase:
                    video_paths.append(video.ruta_video)
                    labels.append(clase_a_idx[clase])
            
            total_videos = len(video_paths)
            total_frames = total_videos * self.num_frames
            
            progresos_entrenamiento[nombre_modelo]["num_clases"] = len(clases)
            progresos_entrenamiento[nombre_modelo]["clases"] = clases
            progresos_entrenamiento[nombre_modelo]["total_videos"] = total_videos
            progresos_entrenamiento[nombre_modelo]["total_frames"] = total_frames
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(video_paths, labels))
            
            train_paths = [video_paths[i] for i in train_idx]
            val_paths = [video_paths[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            class_counts = Counter(train_labels)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            train_dataset = VideoFrameDataset(train_paths, train_labels, 
                                            self.num_frames, is_training=True)
            val_dataset = VideoFrameDataset(val_paths, val_labels, 
                                        self.num_frames, is_training=False)
            
            dataloader_config = {
                'batch_size': self.batch_size,
                'sampler': sampler,
                'num_workers': self.num_workers,
                'drop_last': True
            }
            
            if self.num_workers > 0:
                dataloader_config['pin_memory'] = True if self.device.type == 'cuda' else False
            
            train_loader = DataLoader(train_dataset, **dataloader_config)
            
            val_dataloader_config = {
                'batch_size': self.batch_size,
                'shuffle': False,
                'num_workers': self.num_workers,
                'drop_last': False
            }
            
            if self.num_workers > 0:
                val_dataloader_config['pin_memory'] = True if self.device.type == 'cuda' else False
            
            val_loader = DataLoader(val_dataset, **val_dataloader_config)
            
            model = ImprovedCNNVideo(len(clases)).to(self.device)
            
            criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
            
            warmup_epochs = min(15, epochs // 10)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=warmup_epochs/epochs,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
            
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            progresos_entrenamiento[nombre_modelo]["estado"] = "entrenando"
            progresos_entrenamiento[nombre_modelo]["mensaje"] = "Procesando..."
            
            best_accuracy = 0.0
            best_model_state = None
            patience = 30
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (videos, labels_batch) in enumerate(train_loader):
                    videos = videos.to(self.device, non_blocking=True)
                    labels_batch = labels_batch.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(videos)
                            loss = criterion(outputs, labels_batch)
                        
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(videos)
                        loss = criterion(outputs, labels_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels_batch.size(0)
                    train_correct += (predicted == labels_batch).sum().item()
                    
                    frames_procesados = (epoch * len(train_loader) + batch_idx + 1) * self.batch_size * self.num_frames
                    progresos_entrenamiento[nombre_modelo]["frames_procesados"] = min(frames_procesados, total_frames)
                    
                    del videos, labels_batch, outputs, loss, predicted
                    
                    if (batch_idx + 1) % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for videos, labels_batch in val_loader:
                        videos = videos.to(self.device, non_blocking=True)
                        labels_batch = labels_batch.to(self.device, non_blocking=True)
                        
                        if scaler:
                            with torch.cuda.amp.autocast():
                                outputs = model(videos)
                                loss = criterion(outputs, labels_batch)
                        else:
                            outputs = model(videos)
                            loss = criterion(outputs, labels_batch)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels_batch.size(0)
                        val_correct += (predicted == labels_batch).sum().item()
                        
                        del videos, labels_batch, outputs, loss, predicted
                
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                
                progresos_entrenamiento[nombre_modelo]["epoch_actual"] = epoch + 1
                progresos_entrenamiento[nombre_modelo]["total_epochs"] = epochs
                progresos_entrenamiento[nombre_modelo]["accuracy"] = float(val_acc)
                progresos_entrenamiento[nombre_modelo]["loss"] = float(avg_val_loss)
                progresos_entrenamiento[nombre_modelo]["train_accuracy"] = float(train_acc)
                progresos_entrenamiento[nombre_modelo]["train_loss"] = float(avg_train_loss)
                progresos_entrenamiento[nombre_modelo]["progreso"] = ((epoch + 1) / epochs) * 100.0
                progresos_entrenamiento[nombre_modelo]["mensaje"] = (
                    f"Época {epoch+1}/{epochs} | "
                    f"Train: {train_acc:.4f} | Val: {val_acc:.4f}"
                )
                
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                    logger.info(
                        f"Época {epoch+1:3d}/{epochs} | "
                        f"Train: Loss {avg_train_loss:.4f}, Acc {train_acc:.4f} | "
                        f"Val: Loss {avg_val_loss:.4f}, Acc {val_acc:.4f} | "
                        f"Best: {best_accuracy:.4f}"
                    )
                
                if patience_counter >= patience:
                    break
                
                if (epoch + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if best_accuracy >= self.min_accuracy:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                
                ruta_modelo = self._guardar_modelo(model, nombre_modelo, clases, best_accuracy)
                
                self._actualizar_bd(db, nombre_modelo, str(ruta_modelo), clases, 
                                best_accuracy, epochs, len(video_paths))
                
                progresos_entrenamiento[nombre_modelo]["estado"] = "completado"
                progresos_entrenamiento[nombre_modelo]["entrenando"] = False
                progresos_entrenamiento[nombre_modelo]["mensaje"] = "Entrenamiento completado exitosamente"
                progresos_entrenamiento[nombre_modelo]["frames_procesados"] = total_frames
            else:
                progresos_entrenamiento[nombre_modelo]["estado"] = "error"
                progresos_entrenamiento[nombre_modelo]["entrenando"] = False
                progresos_entrenamiento[nombre_modelo]["mensaje"] = f"Accuracy insuficiente: {best_accuracy*100:.2f}%"
                raise Exception(f"Accuracy insuficiente: {best_accuracy*100:.2f}%")
            
        except Exception as e:
            progresos_entrenamiento[nombre_modelo]["estado"] = "error"
            progresos_entrenamiento[nombre_modelo]["entrenando"] = False
            progresos_entrenamiento[nombre_modelo]["mensaje"] = f"Error: {str(e)}"
            raise
        
        finally:
            if train_loader:
                del train_loader
            if val_loader:
                del val_loader
            if model:
                del model
            if db:
                try:
                    db.close()
                except:
                    pass
            
            try:
                limpiar_carpeta_temporal(f"temp_entrenamiento_{nombre_modelo}")
            except Exception as e_clean:
                logger.warning(f"No se pudo limpiar carpeta temporal: {e_clean}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _guardar_modelo(self, model, nombre_modelo: str, clases: List[str], 
                       accuracy: float) -> Path:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            ruta_modelo = self.modelo_dir / f"{nombre_modelo}.pth"
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'clases': clases,
                'num_clases': len(clases),
                'accuracy': float(accuracy),
                'num_frames': self.num_frames,
                'arquitectura': 'ImprovedCNNVideo',
                'framework': 'pytorch_resnet_attention_lstm',
                'fecha_entrenamiento': datetime.now().isoformat(),
                'normalizacion': 'imagenet',
                'mean': MEAN.tolist(),
                'std': STD.tolist()
            }, str(ruta_modelo))
            
            try:
                respaldar_modelo_en_drive(ruta_modelo)
            except Exception as e_drive:
                logger.warning(f"No se pudo respaldar en Drive (modelo local OK): {e_drive}")
            
            return ruta_modelo
            
        except Exception as e:
            raise

    def _actualizar_bd(self, db: Session, nombre_modelo: str, ruta_modelo: str, 
                      clases: List[str], accuracy: float, epochs: int, total_videos: int):
        try:
            modelo_db = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()
            
            if not modelo_db:
                modelo_db = ModeloIA(nombre=nombre_modelo)
                db.add(modelo_db)
            
            modelo_db.ruta_archivo = ruta_modelo
            modelo_db.accuracy = float(accuracy)
            modelo_db.num_clases = len(clases)
            modelo_db.clases_json = json.dumps(clases, ensure_ascii=False)
            modelo_db.total_imagenes = total_videos
            modelo_db.epocas_entrenamiento = epochs
            modelo_db.arquitectura = "ImprovedCNNVideo"
            modelo_db.entrenando = False
            modelo_db.activo = True
            modelo_db.fecha_entrenamiento = datetime.now()
            modelo_db.version = "2.0"
            modelo_db.descripcion = (
                f"Modelo avanzado con ResNet + Attention + LSTM. "
                f"{len(clases)} señas, {total_videos} videos, "
                f"accuracy: {accuracy*100:.2f}%"
            )
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            raise


entrenamiento_service = EntrenamientoModeloService()