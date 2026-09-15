import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDNN_DETERMINISTIC'] = '1'

import gc
import json
import random
import logging
from pathlib import Path
from threading import Lock
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import Session
from sklearn.model_selection import StratifiedShuffleSplit

from app.servicios.drive_service import (
    respaldar_modelo_en_drive,
    descargar_videos_temporalmente,
    limpiar_carpeta_temporal
)
from app.modelos.entrenamiento import ModeloIA
from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas
from app.modelos.dataset import VideoDataset
from app.utilidades.base_datos import SessionLocal
from app.servicios.config_tipo_senas import (
    detectar_tipo_sena,
    obtener_config_sena,
    validar_consistencia_categoria,
    es_categoria_alfabeto,
    calcular_factor_confusion_mejorado,
    redimensionar_manteniendo_aspecto
)
from app.utilidades.augmentation_inteligente import (
    crear_sampler_balanceado,
    obtener_parametros_entrenamiento_optimizados
)

logger = logging.getLogger(__name__)

progresos_entrenamiento: Dict[str, Any] = {}
_lock_progresos = Lock()


class AumentadorKeypoints:

    @staticmethod
    def ruido(keypoints: np.ndarray, factor: float = 0.01) -> np.ndarray:
        return np.clip(keypoints + np.random.normal(0, factor, keypoints.shape), 0, 1)

    @staticmethod
    def escala(keypoints: np.ndarray, rango: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        factor = np.random.uniform(*rango)
        resultado = keypoints.copy()
        for i in range(2):
            inicio, fin = i * 63, i * 63 + 63
            mano = resultado[inicio:fin].reshape(21, 3)
            centroide = mano[:, :2].mean(axis=0)
            mano[:, :2] = (mano[:, :2] - centroide) * factor + centroide
            resultado[inicio:fin] = mano.flatten()
        return np.clip(resultado, 0, 1)

    @staticmethod
    def traslacion(keypoints: np.ndarray, rango: float = 0.03) -> np.ndarray:
        offset_x = np.random.uniform(-rango, rango)
        offset_y = np.random.uniform(-rango, rango)
        resultado = keypoints.copy()
        for i in range(2):
            inicio, fin = i * 63, i * 63 + 63
            mano = resultado[inicio:fin].reshape(21, 3)
            mano[:, 0] += offset_x
            mano[:, 1] += offset_y
            resultado[inicio:fin] = mano.flatten()
        return np.clip(resultado, 0, 1)


def calcular_intensidad_augmentation(config: Dict[str, Any]) -> float:
    if config.get('es_emergencia', False):
        base = 0.2
    elif config.get('es_confusion_numerica', False):
        base = 0.25
    elif config.get('es_confusa', False):
        base = 0.35
    else:
        base = 0.5

    rotacion = config.get('augmentation_rotation', 10)
    brillo_min, brillo_max = config.get('augmentation_brightness', (0.8, 1.2))
    amplitud_brillo = max(0.0, brillo_max - brillo_min)

    factor_rotacion = min(1.0, rotacion / 20.0)
    factor_brillo = min(1.0, amplitud_brillo / 0.6)
    variacion = (factor_rotacion + factor_brillo) / 2.0

    intensidad = base * (0.6 + 0.4 * variacion)
    return max(0.15, min(0.75, intensidad))


def aplicar_augmentation_keypoints(secuencia: np.ndarray, config: Dict, es_entrenamiento: bool) -> np.ndarray:
    if not es_entrenamiento:
        return secuencia

    prob_base = calcular_intensidad_augmentation(config)

    resultado = secuencia.copy()
    for i in range(len(resultado)):
        if random.random() < prob_base:
            resultado[i] = AumentadorKeypoints.ruido(resultado[i])
        if random.random() < prob_base * 0.7:
            resultado[i] = AumentadorKeypoints.escala(resultado[i])
        if random.random() < prob_base * 0.5:
            resultado[i] = AumentadorKeypoints.traslacion(resultado[i])
    return resultado


class DatasetKeypointsAdaptativo(Dataset):

    def __init__(self, video_paths: List[str], labels: List[int], senas: List[str],
                 num_frames: int = 20, es_entrenamiento: bool = True):
        if not (len(video_paths) == len(labels) == len(senas)):
            raise ValueError("video_paths, labels y senas deben tener el mismo tamaño")

        self.video_paths = video_paths
        self.labels = labels
        self.senas = senas
        self.num_frames = num_frames
        self.es_entrenamiento = es_entrenamiento
        self.tipos = [detectar_tipo_sena(s) for s in senas]
        self.configs = [obtener_config_sena(s) for s in senas]
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        logger.info(f"Precomputando keypoints de {len(video_paths)} videos (una sola vez)...")
        self.cache_keypoints: List[np.ndarray] = []
        for i, video_path in enumerate(self.video_paths):
            secuencia = self._cargar_keypoints(video_path, self.num_frames, self.tipos[i])
            self.cache_keypoints.append(secuencia)
            if (i + 1) % 10 == 0:
                logger.info(f"  Precomputados {i + 1}/{len(video_paths)} videos")
        logger.info("Precomputo de keypoints completado")

        self.mp_hands.close()

    def __len__(self):
        return len(self.video_paths)

    def _extraer_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            resultado = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not resultado.multi_hand_landmarks or not resultado.multi_handedness:
                return None

            manos = {'Right': None, 'Left': None}
            for landmarks, handedness in zip(resultado.multi_hand_landmarks, resultado.multi_handedness):
                etiqueta = handedness.classification[0].label
                valores = []
                for punto in landmarks.landmark:
                    valores.extend([punto.x, punto.y, punto.z])
                if len(valores) == 63:
                    manos[etiqueta] = valores

            mano_derecha = manos['Right'] if manos['Right'] is not None else [0.0] * 63
            mano_izquierda = manos['Left'] if manos['Left'] is not None else [0.0] * 63
            keypoints = mano_derecha + mano_izquierda

            if len(keypoints) != 126:
                return None

            return np.array(keypoints, dtype=np.float32)
        except Exception:
            return None

    def _cargar_keypoints(self, video_path: str, num_frames: int, tipo: str) -> np.ndarray:
        cap = None
        try:
            if not os.path.exists(video_path):
                return np.zeros((num_frames, 126), dtype=np.float32)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return np.zeros((num_frames, 126), dtype=np.float32)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return np.zeros((num_frames, 126), dtype=np.float32)

            if tipo == 'ESTATICA':
                centro = total_frames // 2
                ventana = num_frames // 2
                inicio = max(0, centro - ventana)
                fin = min(total_frames, centro + ventana)
                indices = np.linspace(inicio, max(fin - 1, inicio), num_frames, dtype=int)
            else:
                if total_frames <= num_frames:
                    indices = np.arange(total_frames)
                else:
                    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            indices = np.unique(indices)

            secuencia = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame = redimensionar_manteniendo_aspecto(frame, (224, 224))
                    keypoints = self._extraer_keypoints(frame)
                    secuencia.append(keypoints if keypoints is not None else
                                      (secuencia[-1].copy() if secuencia else np.zeros(126, dtype=np.float32)))
                else:
                    secuencia.append(secuencia[-1].copy() if secuencia else np.zeros(126, dtype=np.float32))

            while len(secuencia) < num_frames:
                secuencia.append(secuencia[-1].copy() if secuencia else np.zeros(126, dtype=np.float32))

            return np.array(secuencia[:num_frames], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error cargando keypoints de {video_path}: {e}")
            return np.zeros((num_frames, 126), dtype=np.float32)
        finally:
            if cap is not None:
                cap.release()

    def __getitem__(self, idx):
        label = self.labels[idx]
        tipo = self.tipos[idx]
        config = self.configs[idx]

        try:
            secuencia = self.cache_keypoints[idx].copy()

            if secuencia.shape[0] != self.num_frames:
                if secuencia.shape[0] < self.num_frames:
                    ultimo = secuencia[-1] if len(secuencia) > 0 else np.zeros(126, dtype=np.float32)
                    relleno = np.tile(ultimo, (self.num_frames - secuencia.shape[0], 1))
                    secuencia = np.vstack([secuencia, relleno])
                else:
                    secuencia = secuencia[:self.num_frames]

            secuencia = aplicar_augmentation_keypoints(secuencia, config, self.es_entrenamiento)

            tensor_keypoints = torch.from_numpy(secuencia.copy()).float()
            tensor_label = torch.tensor(label, dtype=torch.long)

            if tensor_keypoints.shape != (self.num_frames, 126):
                raise ValueError(f"Shape inválido: {tensor_keypoints.shape}")

            return tensor_keypoints, tensor_label, tipo
        except Exception as e:
            logger.error(f"Error procesando índice {idx}: {e}")
            return (torch.zeros((self.num_frames, 126), dtype=torch.float32),
                    torch.tensor(0, dtype=torch.long), 'DINAMICA')

class PerdidaContrastivaFocal(nn.Module):

    def __init__(self, clases: List[str], alpha: float = 1.0, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.clases = clases
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        _, predicted = torch.max(inputs, 1)
        penalizaciones = torch.ones_like(focal_loss)

        for i in range(len(targets)):
            objetivo_idx = targets[i].item()
            predicho_idx = predicted[i].item()
            if objetivo_idx != predicho_idx and objetivo_idx < len(self.clases) and predicho_idx < len(self.clases):
                factor = calcular_factor_confusion_mejorado(
                    self.clases[objetivo_idx].upper(), self.clases[predicho_idx].upper()
                )
                penalizaciones[i] = factor

        return (focal_loss * penalizaciones).mean()


class ServicioEntrenamientoUnificado:

    def __init__(self):
        self.modelo_dir = Path('modelos_entrenados')
        self.modelo_dir.mkdir(exist_ok=True)
        self.device, self.gpu_info = self._detectar_y_configurar_gpu()
        self.batch_size, self.num_frames, self.num_workers = self._configurar_recursos()
        self.weight_decay = 1e-4
        self.min_videos_por_sena = 8
        self.min_accuracy = 0.50
        self.frames_estaticos_modelo = 8
        logger.info(f"Servicio inicializado en {self.device}, batch_size={self.batch_size}, frames={self.num_frames}")

    def _configurar_recursos(self) -> Tuple[int, int, int]:
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2

        if not torch.cuda.is_available():
            return 4, 20, 0

        memoria_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if memoria_gpu >= 8:
            return 8, 20, num_workers
        if memoria_gpu >= 6:
            return 6, 20, num_workers
        return 4, 20, num_workers

    def _detectar_y_configurar_gpu(self) -> Tuple[torch.device, Optional[Dict[str, Any]]]:
        if not torch.cuda.is_available():
            logger.info("GPU no disponible, usando CPU")
            return torch.device('cpu'), None

        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        nombre = torch.cuda.get_device_name(0)
        memoria_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

        logger.info(f"GPU detectada: {nombre} - {memoria_total:.2f} GB")
        return device, {"nombre": nombre, "memoria_total_gb": round(memoria_total, 2)}

    def generar_nombre_modelo(self) -> str:
        return f"modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _obtener_videos_utilizables(self, categoria_ids: List[int], db: Session) -> List[VideoDataset]:
        return db.query(VideoDataset).filter(
            VideoDataset.aprobado == True,
            VideoDataset.categoria_id.in_(categoria_ids)
        ).all()

    def validar_dataset_entrenamiento(self, categoria_ids: List[int], db: Session) -> Dict[str, Any]:
        try:
            videos_db = self._obtener_videos_utilizables(categoria_ids, db)

            videos_validos, videos_no_encontrados = [], []
            for video in videos_db:
                tiene_local = bool(video.ruta_video and os.path.exists(video.ruta_video))
                tiene_drive = bool(video.drive_file_id)
                (videos_validos if tiene_local or tiene_drive else videos_no_encontrados).append(video)

            videos_por_sena: Dict[str, List] = {}
            for video in videos_validos:
                videos_por_sena.setdefault(video.sena or "sin_clasificar", []).append(video)

            senas_con_minimo = [s for s, v in videos_por_sena.items() if len(v) >= self.min_videos_por_sena]
            senas_sin_minimo = [s for s in videos_por_sena if s not in senas_con_minimo]
            senas_estaticas = [s for s in videos_por_sena if detectar_tipo_sena(s) == 'ESTATICA']
            senas_dinamicas = [s for s in videos_por_sena if s not in senas_estaticas]

            consistencia = validar_consistencia_categoria(list(videos_por_sena.keys()))
            es_alfabeto = es_categoria_alfabeto(list(videos_por_sena.keys()))
            total_videos = len(videos_validos)
            valido = total_videos >= 20 and len(senas_con_minimo) >= 2

            return {
                "valido": valido,
                "total_videos": total_videos,
                "videos_validos": len(videos_validos),
                "videos_no_encontrados": len(videos_no_encontrados),
                "total_senas": len(videos_por_sena),
                "senas_con_minimo": senas_con_minimo,
                "senas_sin_minimo": senas_sin_minimo,
                "senas_estaticas": senas_estaticas,
                "senas_dinamicas": senas_dinamicas,
                "distribucion": {
                    "estaticas": len(senas_estaticas),
                    "dinamicas": len(senas_dinamicas),
                    "tipo_predominante": consistencia.get('tipo_mayoritario')
                },
                "consistencia": consistencia,
                "es_categoria_alfabeto": es_alfabeto,
                "minimo_videos_requerido": 20,
                "minimo_senas_requerido": 2,
                "minimo_videos_por_sena": self.min_videos_por_sena,
                "cumple_requisitos": {
                    "videos_totales": total_videos >= 20,
                    "senas_suficientes": len(senas_con_minimo) >= 2,
                    "videos_por_sena": len(senas_sin_minimo) == 0
                }
            }
        except Exception as e:
            logger.error(f"Error validando dataset: {e}")
            return {
                "valido": False, "error": str(e), "total_videos": 0, "videos_validos": 0,
                "total_senas": 0, "senas_con_minimo": [], "senas_sin_minimo": []
            }

    def _progreso_por_defecto(self, nombre_modelo: str, estado: str, mensaje: str, entrenando: bool) -> Dict[str, Any]:
        return {
            "nombre_modelo": nombre_modelo, "estado": estado, "progreso": 0.0,
            "epoch_actual": 0, "total_epochs": 0, "accuracy": 0.0, "loss": 0.0,
            "train_loss": 0.0, "train_accuracy": 0.0, "num_clases": 0, "clases": [],
            "total_videos": 0, "frames_procesados": 0, "total_frames": 0,
            "mensaje": mensaje, "entrenando": entrenando
        }

    def obtener_progreso_entrenamiento(self, nombre_modelo: str) -> Dict[str, Any]:
        with _lock_progresos:
            if nombre_modelo in progresos_entrenamiento:
                progreso = progresos_entrenamiento[nombre_modelo]
                return {
                    "nombre_modelo": progreso.get("nombre_modelo", nombre_modelo),
                    "estado": progreso.get("estado", "desconocido"),
                    "progreso": progreso.get("progreso", 0.0),
                    "epoch_actual": progreso.get("epoch_actual", 0),
                    "total_epochs": progreso.get("total_epochs", 0),
                    "accuracy": progreso.get("accuracy", 0.0),
                    "loss": progreso.get("loss", 0.0),
                    "train_loss": progreso.get("train_loss", 0.0),
                    "train_accuracy": progreso.get("train_accuracy", 0.0),
                    "num_clases": progreso.get("num_clases", 0),
                    "clases": progreso.get("clases", []),
                    "total_videos": progreso.get("total_videos", 0),
                    "frames_procesados": progreso.get("frames_procesados", 0),
                    "total_frames": progreso.get("total_frames", 0),
                    "mensaje": progreso.get("mensaje", ""),
                    "fecha_inicio": progreso.get("fecha_inicio", datetime.now().isoformat()),
                    "entrenando": progreso.get("entrenando", False),
                    "accuracy_por_clase": progreso.get("accuracy_por_clase", {})
                }

        db = SessionLocal()
        try:
            modelo_db = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()
            if not modelo_db:
                return self._progreso_por_defecto(nombre_modelo, "preparando", "Inicializando entrenamiento...", True)

            clases = json.loads(modelo_db.clases_json) if modelo_db.clases_json else []
            return {
                "nombre_modelo": modelo_db.nombre,
                "estado": "completado" if modelo_db.activo else "inactivo",
                "progreso": 100.0 if modelo_db.activo else 0.0,
                "epoch_actual": modelo_db.epocas_entrenamiento or 0,
                "total_epochs": modelo_db.epocas_entrenamiento or 0,
                "accuracy": float(modelo_db.accuracy) if modelo_db.accuracy else 0.0,
                "loss": 0.0, "train_loss": 0.0, "train_accuracy": 0.0,
                "num_clases": modelo_db.num_clases or 0,
                "clases": clases,
                "total_videos": modelo_db.total_imagenes or 0,
                "frames_procesados": (modelo_db.total_imagenes or 0) * self.num_frames,
                "total_frames": (modelo_db.total_imagenes or 0) * self.num_frames,
                "fecha_entrenamiento": modelo_db.fecha_entrenamiento.isoformat() if modelo_db.fecha_entrenamiento else None,
                "mensaje": "Modelo completado previamente" if modelo_db.activo else "Modelo inactivo",
                "entrenando": False
            }
        except Exception as e:
            logger.error(f"Error consultando progreso: {e}")
            return self._progreso_por_defecto(nombre_modelo, "error", f"Error consultando: {e}", False)
        finally:
            db.close()

    def _actualizar_progreso(self, nombre_modelo: str, **campos):
        with _lock_progresos:
            if nombre_modelo in progresos_entrenamiento:
                progresos_entrenamiento[nombre_modelo].update(campos)

    def _preparar_videos(self, categoria_ids: List[int], db: Session, nombre_modelo: str) -> List[VideoDataset]:
        videos_db = self._obtener_videos_utilizables(categoria_ids, db)
        videos_locales = [v for v in videos_db if v.ruta_video and os.path.exists(v.ruta_video)]
        videos_solo_drive = [v for v in videos_db if v.drive_file_id and v not in videos_locales]

        self._actualizar_progreso(nombre_modelo, mensaje=f"Descargando {len(videos_solo_drive)} videos desde Drive...")

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

        return videos_validos

    def entrenar(self, nombre_modelo: str, categoria_ids: List[int], epochs: int):
        epochs = 150 if not epochs or epochs <= 0 else min(epochs, 500)

        db = model = train_loader = val_loader = None
        try:
            with _lock_progresos:
                progresos_entrenamiento[nombre_modelo] = {
                    **self._progreso_por_defecto(nombre_modelo, "iniciando", "Inicializando entrenamiento...", True),
                    "total_epochs": epochs,
                    "fecha_inicio": datetime.now().isoformat()
                }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            db = SessionLocal()
            videos_validos = self._preparar_videos(categoria_ids, db, nombre_modelo)

            self._actualizar_progreso(
                nombre_modelo, estado="preparando_datos",
                mensaje="Preparando dataset...", total_videos=len(videos_validos)
            )

            videos_por_sena: Dict[str, List] = {}
            for video in videos_validos:
                videos_por_sena.setdefault(video.sena or "sin_clasificar", []).append(video)

            senas_validas = {k: v for k, v in videos_por_sena.items() if len(v) >= self.min_videos_por_sena}
            if len(senas_validas) < 2:
                raise Exception(
                    f"Se necesitan al menos 2 señas con {self.min_videos_por_sena}+ videos. "
                    f"Señas válidas: {list(senas_validas.keys())}"
                )

            clases = sorted(senas_validas.keys())
            clase_a_idx = {clase: i for i, clase in enumerate(clases)}
            tipos_senas = {clase: detectar_tipo_sena(clase) for clase in clases}

            video_paths, labels, senas_list = [], [], []
            for clase, videos_clase in senas_validas.items():
                for video in videos_clase:
                    video_paths.append(video.ruta_video)
                    labels.append(clase_a_idx[clase])
                    senas_list.append(clase)

            total_videos = len(video_paths)
            total_frames = total_videos * self.num_frames

            self._actualizar_progreso(
                nombre_modelo, num_clases=len(clases), clases=clases,
                total_videos=total_videos, total_frames=total_frames,
                estado="cargando_datos", mensaje=f"Extrayendo keypoints de {total_videos} videos..."
            )

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(video_paths, labels))

            train_paths = [video_paths[i] for i in train_idx]
            val_paths = [video_paths[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            train_senas = [senas_list[i] for i in train_idx]
            val_senas = [senas_list[i] for i in val_idx]

            sampler = crear_sampler_balanceado(train_labels, train_senas, clases)

            train_dataset = DatasetKeypointsAdaptativo(train_paths, train_labels, train_senas, self.num_frames, True)
            val_dataset = DatasetKeypointsAdaptativo(val_paths, val_labels, val_senas, self.num_frames, False)

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=sampler,
                num_workers=0, drop_last=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=0, drop_last=False
            )

            parametros = obtener_parametros_entrenamiento_optimizados(clases)
            lr = parametros.get('learning_rate', 0.001)
            patience = parametros.get('patience', 30)

            self._actualizar_progreso(nombre_modelo, estado="creando_modelo", mensaje="Inicializando modelo...")

            model = ModeloAdaptativoSenas(len(clases), tipos_senas, self.frames_estaticos_modelo).to(self.device)
            criterion = PerdidaContrastivaFocal(clases, alpha=1.0, gamma=parametros.get('focal_gamma', 2.0))
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

            warmup_epochs = min(15, epochs // 10)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
                pct_start=warmup_epochs / epochs, anneal_strategy='cos'
            )
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

            self._actualizar_progreso(nombre_modelo, estado="entrenando", mensaje="Entrenamiento en progreso...")

            best_accuracy = 0.0
            best_model_state = None
            patience_counter = 0
            matriz_confusion_val = np.zeros((len(clases), len(clases)), dtype=int)

            for epoch in range(epochs):
                model.train()
                train_loss = train_correct = train_total = 0

                for batch_idx, (keypoints, labels_batch, tipos_batch) in enumerate(train_loader):
                    try:
                        keypoints = keypoints.to(self.device, non_blocking=True)
                        labels_batch = labels_batch.to(self.device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)

                        if scaler:
                            with torch.cuda.amp.autocast():
                                outputs = model.forward_batch_mixto(keypoints, tipos_batch)
                                loss = criterion(outputs, labels_batch)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model.forward_batch_mixto(keypoints, tipos_batch)
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
                        self._actualizar_progreso(nombre_modelo, frames_procesados=min(frames_procesados, total_frames))

                        del keypoints, labels_batch, outputs, loss, predicted
                        if (batch_idx + 1) % 2 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "out of memory" not in str(e):
                            raise
                        logger.error(f"Error de memoria en batch {batch_idx}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                model.eval()
                val_loss = val_correct = val_total = 0
                matriz_confusion_epoch = np.zeros((len(clases), len(clases)), dtype=int)

                with torch.no_grad():
                    for keypoints, labels_batch, tipos_batch in val_loader:
                        try:
                            keypoints = keypoints.to(self.device, non_blocking=True)
                            labels_batch = labels_batch.to(self.device, non_blocking=True)

                            if scaler:
                                with torch.cuda.amp.autocast():
                                    outputs = model.forward_batch_mixto(keypoints, tipos_batch)
                                    loss = criterion(outputs, labels_batch)
                            else:
                                outputs = model.forward_batch_mixto(keypoints, tipos_batch)
                                loss = criterion(outputs, labels_batch)

                            val_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            val_total += labels_batch.size(0)
                            val_correct += (predicted == labels_batch).sum().item()

                            for real, pred in zip(labels_batch.tolist(), predicted.tolist()):
                                matriz_confusion_epoch[real][pred] += 1

                            del keypoints, labels_batch, outputs, loss, predicted
                        except RuntimeError as e:
                            if "out of memory" not in str(e):
                                raise
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()

                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

                self._actualizar_progreso(
                    nombre_modelo, epoch_actual=epoch + 1, accuracy=float(val_acc),
                    loss=float(avg_val_loss), train_accuracy=float(train_acc),
                    train_loss=float(avg_train_loss), progreso=((epoch + 1) / epochs) * 100.0,
                    mensaje=f"Época {epoch + 1}/{epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}"
                )

                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_state = model.state_dict().copy()
                    matriz_confusion_val = matriz_confusion_epoch.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                    logger.info(
                        f"Época {epoch + 1:3d}/{epochs} | Train: {train_acc:.4f} | "
                        f"Val: {val_acc:.4f} | Best: {best_accuracy:.4f}"
                    )

                if patience_counter >= patience:
                    logger.info(f"Early stopping en época {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if best_accuracy < self.min_accuracy:
                raise Exception(f"Accuracy insuficiente: {best_accuracy * 100:.2f}%")

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            accuracy_por_clase = {}
            for i, clase in enumerate(clases):
                total_clase = matriz_confusion_val[i].sum()
                aciertos_clase = matriz_confusion_val[i][i]
                accuracy_por_clase[clase] = float(aciertos_clase / total_clase) if total_clase > 0 else 0.0

            logger.info(f"Accuracy por clase (mejor época): {accuracy_por_clase}")

            ruta_modelo = self._guardar_modelo(
                model, nombre_modelo, clases, best_accuracy, tipos_senas,
                accuracy_por_clase, matriz_confusion_val.tolist()
            )
            self._actualizar_bd(db, nombre_modelo, str(ruta_modelo), clases, best_accuracy, epochs, total_videos)

            self._actualizar_progreso(
                nombre_modelo, estado="completado", entrenando=False,
                progreso=100.0, accuracy=float(best_accuracy),
                frames_procesados=total_frames, mensaje="Entrenamiento completado exitosamente",
                accuracy_por_clase=accuracy_por_clase
            )
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            self._actualizar_progreso(nombre_modelo, estado="error", entrenando=False, mensaje=f"Error: {e}")
            raise
        finally:
            del train_loader, val_loader, model
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass
            try:
                limpiar_carpeta_temporal(f"temp_entrenamiento_{nombre_modelo}")
            except Exception as e:
                logger.warning(f"No se pudo limpiar carpeta temporal: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _guardar_modelo(self, model, nombre_modelo: str, clases: List[str],
                         accuracy: float, tipos_senas: dict,
                         accuracy_por_clase: Dict[str, float], matriz_confusion: list) -> Path:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        ruta_modelo = self.modelo_dir / f"{nombre_modelo}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'clases': clases,
            'num_clases': len(clases),
            'accuracy': float(accuracy),
            'accuracy_por_clase': accuracy_por_clase,
            'matriz_confusion': matriz_confusion,
            'num_frames': self.num_frames,
            'frames_estaticos': self.frames_estaticos_modelo,
            'arquitectura': 'ModeloAdaptativoSenas_LSTM',
            'framework': 'pytorch_lstm_keypoints',
            'tipos_senas': tipos_senas,
            'fecha_entrenamiento': datetime.now().isoformat(),
            'version': '6.0_unificado'
        }, str(ruta_modelo))

        try:
            respaldar_modelo_en_drive(ruta_modelo)
        except Exception as e:
            logger.warning(f"No se pudo respaldar en Drive (modelo local OK): {e}")

        return ruta_modelo

    def _actualizar_bd(self, db: Session, nombre_modelo: str, ruta_modelo: str,
                        clases: List[str], accuracy: float, epochs: int, total_videos: int):
        try:
            modelo_db = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()
            if not modelo_db:
                modelo_db = ModeloIA(nombre=nombre_modelo)
                db.add(modelo_db)

            modelo_db.ruta_archivo = ruta_modelo
            modelo_db.accuracy = round(float(accuracy), 6)
            modelo_db.num_clases = len(clases)
            modelo_db.clases_json = json.dumps(clases, ensure_ascii=False)
            modelo_db.total_imagenes = total_videos
            modelo_db.epocas_entrenamiento = epochs
            modelo_db.arquitectura = "LSTM_Keypoints_Unificado"
            modelo_db.entrenando = False
            modelo_db.activo = True
            modelo_db.fecha_entrenamiento = datetime.now()
            modelo_db.version = "6.0"
            modelo_db.descripcion = (
                f"Modelo LSTM sobre keypoints, {len(clases)} señas, "
                f"{total_videos} videos, accuracy: {accuracy * 100:.2f}%"
            )

            db.commit()
            db.refresh(modelo_db)
        except Exception as e:
            db.rollback()
            logger.error(f"Error actualizando BD: {e}")
            raise


servicio_entrenamiento = ServicioEntrenamientoUnificado()