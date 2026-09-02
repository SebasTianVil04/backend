# src/app/modelos/lstm_video.py

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.optim as optim
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, freeze_backbone: bool = True):
        super(CNNFeatureExtractor, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_backbone:
            for idx, (name, param) in enumerate(self.features.named_parameters()):
                if idx < 100:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        self.output_dim = 2048
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        features = self.features(x)
        features = self.global_avg_pool(features)
        features = features.view(features.size(0), -1)
        return features


class SignLanguageLSTM(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(SignLanguageLSTM, self).__init__()
        
        self.cnn = CNNFeatureExtractor(freeze_backbone=True)
        feature_dim = self.cnn.output_dim
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.batch_norm = nn.BatchNorm1d(lstm_output_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.es_modelo_secuencial = True
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        n = param.size(0)
                        param.data[(n // 4):(n // 2)].fill_(1.0)
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, (hidden, cell) = self.lstm(features)
        
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
        weighted_output = self.batch_norm(weighted_output)
        
        output = self.classifier(weighted_output)
        
        return output


class VideoDataset(Dataset):
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        num_frames: int = 24,
        transform=None,
        augment: bool = True
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        self.augment = augment
        
        if augment:
            self.augmentation_transforms = [
                self._random_rotation,
                self._random_brightness,
                self._random_contrast,
                self._random_flip
            ]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        video_path = str(Path(video_path).resolve())
        
        try:
            frames = self._load_video_frames(video_path)
        except Exception as e:
            logger.warning(f"Error cargando video {video_path}: {e}")
            logger.warning(f"Ruta existe: {Path(video_path).exists()}")
            frames = self._get_frames_vacios()
        
        if self.augment and np.random.random() > 0.5:
            frames = self._apply_augmentation(frames)
        
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            frames = torch.stack([self._preprocess_frame(frame) for frame in frames])
        
        return frames, label
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        
        return frame_tensor
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        try:
            ruta_path = Path(video_path)
            
            if not ruta_path.exists():
                logger.error(f"Video no existe: {video_path}")
                logger.error(f"Ruta absoluta: {ruta_path.absolute()}")
                return self._get_frames_vacios()
            
            if ruta_path.stat().st_size == 0:
                logger.error(f"Archivo vacío: {video_path}")
                return self._get_frames_vacios()
            
            video_path_str = str(ruta_path)
            
            logger.info(f"Cargando video: {video_path_str}")
            
            cap = cv2.VideoCapture(video_path_str)
            
            if not cap.isOpened():
                logger.error(f"No se puede abrir video con OpenCV: {video_path_str}")
                logger.error(f"Verificar codec y formato del video")
                cap.release()
                return self._get_frames_vacios()
            
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"Frames totales: {total_frames}, FPS: {fps}")
                
                if total_frames == 0:
                    logger.error(f"Archivo corrupto o codec no soportado: {video_path_str}")
                    cap.release()
                    return self._get_frames_vacios()
                
                frames = []
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                
                for frame_idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (224, 224))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    else:
                        logger.warning(f"No se pudo leer frame {frame_idx} del video")
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                cap.release()
                
                if len(frames) < self.num_frames:
                    logger.warning(f"Solo se obtuvieron {len(frames)} frames de {self.num_frames}")
                    frames = self._completar_frames(frames)
                
                logger.info(f"Video cargado exitosamente: {len(frames)} frames")
                return frames
                
            except Exception as e:
                logger.error(f"Error durante lectura de frames: {e}")
                cap.release()
                return self._get_frames_vacios()
                
        except Exception as e:
            logger.error(f"Error crítico cargando video: {e}")
            import traceback
            traceback.print_exc()
            return self._get_frames_vacios()
    
    def _apply_augmentation(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        augmented_frames = []
        
        for frame in frames:
            if np.random.random() > 0.7:
                aug_func = np.random.choice(self.augmentation_transforms)
                frame = aug_func(frame)
            augmented_frames.append(frame)
        
        return augmented_frames
    
    def _random_rotation(self, frame: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-15, 15)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h))
    
    def _random_brightness(self, frame: np.ndarray) -> np.ndarray:
        brightness = np.random.uniform(0.8, 1.2)
        frame = frame.astype(np.float32) * brightness
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _random_contrast(self, frame: np.ndarray) -> np.ndarray:
        contrast = np.random.uniform(0.8, 1.2)
        frame = frame.astype(np.float32) * contrast
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _random_flip(self, frame: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            return cv2.flip(frame, 1)
        return frame
    
    def _get_frames_vacios(self):
        return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
    
    def _completar_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        return frames[:self.num_frames]


def entrenar_modelo_video(
    model: SignLanguageLSTM,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Tuple[SignLanguageLSTM, dict]:
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    logger.info("Iniciando entrenamiento LSTM para video")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        logger.info(f"  LR    : {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"  Nuevo mejor modelo: {val_acc:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.info(f"Entrenamiento completado. Mejor accuracy: {best_val_acc:.4f}")
    
    return model, history


class SimpleLSTM(nn.Module):
    
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(SimpleLSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        self.es_modelo_secuencial = True
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        
        output = self.classifier(lstm_out[:, -1, :])
        
        return output


if __name__ == "__main__":
    logger.info("Probando modelo LSTM")
    
    modelo = SignLanguageLSTM(num_classes=5)
    
    batch_size = 2
    seq_len = 24
    x = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    output = modelo(x)
    logger.info(f"Modelo funcionando. Output shape: {output.shape}")
    logger.info(f"Parametros del modelo: {sum(p.numel() for p in modelo.parameters()):,}")
