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
from typing import Tuple, Dict, Optional
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalizar_imagenet(frame: np.ndarray) -> np.ndarray:
    frame_float = frame.astype(np.float32) / 255.0
    frame_float = np.clip(frame_float, 0.0, 1.0)
    frame_normalized = (frame_float - MEAN) / STD
    return frame_normalized


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


class AdvancedSignLanguageModel(nn.Module):
    def __init__(self, num_classes: int):
        super(AdvancedSignLanguageModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=384,
            num_layers=3,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        
        x = x.view(batch_size * num_frames, c, h, w)
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(batch_size, num_frames, -1)
        
        lstm_out, _ = self.lstm(x)
        
        attention_weights = self.temporal_attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        x = torch.sum(lstm_out * attention_weights, dim=1)
        
        x = self.classifier(x)
        
        return x


class ReconocimientoIA:
    def __init__(self, ruta_modelo: str = None):
        self.model = None
        self.clases = []
        self.num_clases = 0
        self.num_frames = 16
        self.accuracy = 0.0
        self.arquitectura = ""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo_dir = Path('modelos_entrenados')
        
        self.enable_amp = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
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
            self.num_frames = checkpoint.get('num_frames', 16)
            self.accuracy = checkpoint.get('accuracy', 0.0)
            self.arquitectura = checkpoint.get('arquitectura', 'SimpleCNNVideo')
            
            # Cargar arquitectura correcta
            if self.arquitectura == 'AdvancedSignLanguageModel':
                self.model = AdvancedSignLanguageModel(self.num_clases).to(self.device)
            else:
                # Fallback para modelos antiguos
                from app.servicios.entrenamiento_modelo import SimpleCNNVideo
                self.model = SimpleCNNVideo(self.num_clases).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✓ Modelo cargado: {ruta_modelo}")
            logger.info(f"✓ Arquitectura: {self.arquitectura}")
            logger.info(f"✓ Clases: {self.clases}")
            logger.info(f"✓ Frames: {self.num_frames}")
            logger.info(f"✓ Accuracy: {self.accuracy:.4f}")
            logger.info(f"✓ Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self.model = None
            self.clases = []
            raise
    
    def procesar_video(self, ruta_video: str) -> torch.Tensor:
        """Procesar video completo"""
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
            
            # Muestreo uniforme
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # Usar último frame válido o negro
                    if len(frames) > 0:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            frames_array = np.stack(frames)
            frames_normalized = normalizar_imagenet(frames_array)
            tensor = torch.from_numpy(frames_normalized.transpose(0, 3, 1, 2)).float()
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error procesando video: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()
    
    def procesar_frame_individual(self, frame: np.ndarray) -> torch.Tensor:
        """Procesar un frame individual (replicado para secuencia)"""
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_normalized = normalizar_imagenet(frame)
        frame_array = frame_normalized.transpose(2, 0, 1)
        frame_array = np.expand_dims(frame_array, axis=0)
        frames_array = np.tile(frame_array, (self.num_frames, 1, 1, 1))
        tensor = torch.from_numpy(frames_array).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor
    
    def predecir(self, tensor_frames: torch.Tensor) -> Tuple[str, float, Dict]:
        """Realizar predicción con tensor de frames"""
        if self.model is None or len(self.clases) == 0:
            return "desconocido", 0.0, {}
        
        try:
            if tensor_frames.dim() != 5:
                raise ValueError(f"Tensor inválido: esperado 5 dims, recibido {tensor_frames.shape}")
            
            with torch.no_grad():
                if self.enable_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tensor_frames)
                else:
                    outputs = self.model(tensor_frames)
                
                probs = F.softmax(outputs, dim=1)[0]
                
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    logger.warning("NaN/Inf detectado en predicción")
                    probs = torch.ones_like(probs) / len(self.clases)
                
                conf, pred_idx = torch.max(probs, 0)
                sena_predicha = self.clases[pred_idx.item()]
                confianza = conf.item()
                
                # Top-k alternativas
                top_k = min(5, len(self.clases))
                top_values, top_indices = torch.topk(probs, top_k)
                
                alternativas = []
                for val, idx in zip(top_values, top_indices):
                    if val.item() > 0.05:
                        alternativas.append({
                            "sena": self.clases[idx.item()],
                            "confianza": round(val.item(), 4)
                        })
                
                # Calcular consistencia
                consistencia = 1.0
                if len(alternativas) > 1:
                    diff = alternativas[0]["confianza"] - alternativas[1]["confianza"]
                    consistencia = min(1.0, diff * 2)
                
                confianza_ajustada = confianza * (0.7 + 0.3 * consistencia)
                
                metadata = {
                    "alternativas": alternativas,
                    "consistencia": round(consistencia, 4),
                    "confianza_ajustada": round(confianza_ajustada, 4),
                    "confianza_raw": round(confianza, 4),
                    "arquitectura": self.arquitectura,
                    "num_frames": self.num_frames
                }
                
                return sena_predicha, confianza_ajustada, metadata
                
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return "error", 0.0, {}
    
    def predecir_batch(self, tensor_frames_list: list) -> list:
        """Predicción en batch para múltiples videos"""
        if self.model is None or len(self.clases) == 0:
            return [("desconocido", 0.0, {}) for _ in tensor_frames_list]
        
        try:
            batch_tensor = torch.cat(tensor_frames_list, dim=0)
            
            with torch.no_grad():
                if self.enable_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)
                
                probs = F.softmax(outputs, dim=1)
            
            resultados = []
            for i in range(probs.size(0)):
                prob_single = probs[i]
                
                if torch.isnan(prob_single).any() or torch.isinf(prob_single).any():
                    resultados.append(("desconocido", 0.0, {}))
                    continue
                
                conf, pred_idx = torch.max(prob_single, 0)
                sena = self.clases[pred_idx.item()]
                confianza = conf.item()
                
                top_k = min(5, len(self.clases))
                top_values, top_indices = torch.topk(prob_single, top_k)
                
                alternativas = []
                for val, idx in zip(top_values, top_indices):
                    if val.item() > 0.05:
                        alternativas.append({
                            "sena": self.clases[idx.item()],
                            "confianza": round(val.item(), 4)
                        })
                
                metadata = {"alternativas": alternativas}
                resultados.append((sena, confianza, metadata))
            
            return resultados
            
        except Exception as e:
            logger.error(f"Error en predicción batch: {str(e)}")
            return [("error", 0.0, {}) for _ in tensor_frames_list]

    def obtener_estadisticas(self) -> Dict:
        """Obtener estadísticas del modelo"""
        return {
            "total_senas": len(self.clases),
            "senas_disponibles": self.clases,
            "precision_modelo": self.accuracy,
            "num_frames": self.num_frames,
            "dispositivo": str(self.device),
            "arquitectura": self.arquitectura,
            "amp_enabled": self.enable_amp,
            "ultima_actualizacion": datetime.now().isoformat()
        }
    
    def validar_video(self, ruta_video: str) -> Dict:
        """Validar que un video sea procesable"""
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
            
            return {
                "valido": True,
                "duracion": duracion,
                "fps": fps,
                "total_frames": total_frames
            }
            
        except Exception as e:
            return {"valido": False, "razon": str(e)}
        finally:
            if cap is not None:
                cap.release()
    
    def benchmark(self, num_iteraciones: int = 10) -> Dict:
        """Benchmark de rendimiento"""
        if self.model is None:
            return {"error": "Modelo no cargado"}
        
        try:
            dummy_input = torch.randn(1, self.num_frames, 3, 224, 224).to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy_input)
            
            tiempos = []
            
            for _ in range(num_iteraciones):
                start = time.perf_counter()
                
                with torch.no_grad():
                    if self.enable_amp:
                        with torch.cuda.amp.autocast():
                            _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                tiempos.append(end - start)
            
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
            logger.error(f"Error en benchmark: {str(e)}")
            return {"error": str(e)}