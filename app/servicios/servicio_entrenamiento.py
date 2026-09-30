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
from contextlib import ExitStack
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import Session
from sklearn.model_selection import StratifiedShuffleSplit

from app.servicios.drive_service import (
    respaldar_modelo_en_drive,
    descargar_videos_temporalmente,
    limpiar_carpeta_temporal
)
from app.servicios.extractor_secuencias import (
    extraer_secuencia_landmarks,
    secuencia_desde_keypoints,
    detector_para_lote,
    ResultadoSecuencia,
    VERSION_SECUENCIAS,
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
)
from app.utilidades.augmentation_inteligente import (
    crear_sampler_balanceado,
    obtener_parametros_entrenamiento_optimizados,
    validar_configuracion_dataset,
)

logger = logging.getLogger(__name__)

progresos_entrenamiento: Dict[str, Any] = {}
_lock_progresos = Lock()

SEED_ENTRENAMIENTO = 42


def _fijar_semillas(seed: int = SEED_ENTRENAMIENTO) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AumentadorKeypoints:

    @staticmethod
    def _slots_presentes(frame: np.ndarray) -> List[int]:
        return [i for i in range(2) if np.any(frame[i * 63:(i + 1) * 63] != 0)]

    @staticmethod
    def aplicar_espacial(secuencia: np.ndarray,
                         cos_a: float, sin_a: float,
                         factor_esc: float,
                         off_x: float, off_y: float,
                         factor_ruido: float) -> np.ndarray:
        resultado = secuencia.copy()
        for i in range(len(resultado)):
            frame = resultado[i]
            for slot in AumentadorKeypoints._slots_presentes(frame):
                inicio, fin = slot * 63, slot * 63 + 63
                mano = frame[inicio:fin].reshape(21, 3).copy()

                x = mano[:, 0].copy()
                y = mano[:, 1].copy()
                mano[:, 0] = x * cos_a - y * sin_a
                mano[:, 1] = x * sin_a + y * cos_a

                if factor_esc != 1.0:
                    centroide = mano[:, :2].mean(axis=0)
                    mano[:, :2] = (mano[:, :2] - centroide) * factor_esc + centroide

                mano[:, 0] += off_x
                mano[:, 1] += off_y

                mano += np.random.normal(0, factor_ruido, mano.shape).astype(np.float32)

                frame[inicio:fin] = mano.flatten()
            resultado[i] = frame
        return resultado

    @staticmethod
    def aplicar_escala_z(secuencia: np.ndarray, factor_z: float) -> np.ndarray:
        resultado = secuencia.copy()
        for i in range(len(resultado)):
            frame = resultado[i]
            for slot in AumentadorKeypoints._slots_presentes(frame):
                inicio, fin = slot * 63, slot * 63 + 63
                mano = frame[inicio:fin].reshape(21, 3).copy()
                mano[:, 2] *= factor_z
                frame[inicio:fin] = mano.flatten()
            resultado[i] = frame
        return resultado

    @staticmethod
    def simular_deteccion_imperfecta(secuencia: np.ndarray, max_landmarks_afectados: int = 2) -> np.ndarray:
        resultado = secuencia.copy()
        num_frames = len(resultado)
        n_frames_afectados = max(1, num_frames // 4)
        frames_afectados = np.random.choice(num_frames, size=n_frames_afectados, replace=False)

        for i in frames_afectados:
            frame = resultado[i]
            for slot in AumentadorKeypoints._slots_presentes(frame):
                inicio, fin = slot * 63, slot * 63 + 63
                mano = frame[inicio:fin].reshape(21, 3).copy()
                n_afectados = random.randint(1, max_landmarks_afectados)
                indices = np.random.choice(21, size=n_afectados, replace=False)
                mano[indices] += np.random.normal(0, 0.05, (n_afectados, 3)).astype(np.float32)
                frame[inicio:fin] = mano.flatten()
            resultado[i] = frame
        return resultado


def aplicar_augmentation_keypoints(secuencia: np.ndarray, config: Dict, es_entrenamiento: bool) -> np.ndarray:
    if not es_entrenamiento:
        return secuencia

    resultado = secuencia.copy()

    rotacion_max = float(config.get('augmentation_rotation', 8.0))
    permite_zoom = config.get('permite_zoom', True)
    permite_flip = config.get('permite_flip', True)

    if random.random() < 0.9:
        grados = np.random.uniform(-rotacion_max, rotacion_max)
        angulo = np.deg2rad(grados)
        cos_a, sin_a = float(np.cos(angulo)), float(np.sin(angulo))

        factor_esc = float(np.random.uniform(0.85, 1.15)) if (permite_zoom and random.random() < 0.8) else 1.0
        off_x = float(np.random.uniform(-0.08, 0.08)) if random.random() < 0.7 else 0.0
        off_y = float(np.random.uniform(-0.08, 0.08)) if random.random() < 0.7 else 0.0
        factor_ruido = 0.015

        resultado = AumentadorKeypoints.aplicar_espacial(
            resultado, cos_a, sin_a, factor_esc, off_x, off_y, factor_ruido
        )

    if random.random() < 0.5:
        factor_z = float(np.random.uniform(0.8, 1.25))
        resultado = AumentadorKeypoints.aplicar_escala_z(resultado, factor_z)

    if random.random() < 0.3:
        resultado = AumentadorKeypoints.simular_deteccion_imperfecta(resultado)

    return resultado


class DatasetKeypointsAdaptativo(Dataset):

    def __init__(self, secuencias: List[np.ndarray], labels: List[int], senas: List[str],
                 num_frames: int = 20, es_entrenamiento: bool = True):
        if not (len(secuencias) == len(labels) == len(senas)):
            raise ValueError("secuencias, labels y senas deben tener el mismo tamaño")

        self.secuencias = secuencias
        self.labels = labels
        self.senas = senas
        self.num_frames = num_frames
        self.es_entrenamiento = es_entrenamiento
        self.tipos = [detectar_tipo_sena(s) for s in senas]
        self.configs = [obtener_config_sena(s) for s in senas]

    def __len__(self):
        return len(self.secuencias)

    def __getitem__(self, idx):
        label = self.labels[idx]
        tipo = self.tipos[idx]
        config = self.configs[idx]

        try:
            secuencia = self.secuencias[idx].copy()

            if secuencia.shape[0] != self.num_frames:
                if secuencia.shape[0] < self.num_frames:
                    ultimo = secuencia[-1] if len(secuencia) > 0 else np.zeros(126, dtype=np.float32)
                    relleno = np.tile(ultimo, (self.num_frames - secuencia.shape[0], 1))
                    secuencia = np.vstack([secuencia, relleno])
                else:
                    indices = np.linspace(0, secuencia.shape[0] - 1, self.num_frames).round().astype(int)
                    secuencia = secuencia[indices]

            secuencia = aplicar_augmentation_keypoints(secuencia, config, self.es_entrenamiento)

            tensor_keypoints = torch.from_numpy(secuencia.copy()).float()
            tensor_label = torch.tensor(label, dtype=torch.long)
            return tensor_keypoints, tensor_label, tipo
        except Exception as e:
            logger.error(f"Error procesando índice {idx}: {e}")
            return (torch.zeros((self.num_frames, 126), dtype=torch.float32),
                    torch.tensor(0, dtype=torch.long), 'DINAMICA')


class PerdidaContrastivaFocal(nn.Module):

    def __init__(self, clases: List[str], alpha: float = 1.0, gamma: float = 1.5,
                 label_smoothing: float = 0.02):
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
            if (objetivo_idx != predicho_idx
                    and objetivo_idx < len(self.clases)
                    and predicho_idx < len(self.clases)):
                factor = calcular_factor_confusion_mejorado(
                    self.clases[objetivo_idx].upper(),
                    self.clases[predicho_idx].upper()
                )
                penalizaciones[i] = factor

        return (focal_loss * penalizaciones).mean()


class ServicioEntrenamientoUnificado:

    def __init__(self):
        self.modelo_dir = Path('modelos_entrenados')
        self.modelo_dir.mkdir(exist_ok=True)
        self.secuencias_dir = Path("archivos_subidos") / "secuencias_dataset"
        self.secuencias_dir.mkdir(parents=True, exist_ok=True)
        self.device, self.gpu_info = self._detectar_y_configurar_gpu()
        self.batch_size, self.num_frames, self.num_workers = self._configurar_recursos()
        self.weight_decay = 1e-4
        self.min_videos_por_sena = 25
        self.min_accuracy = 0.65
        self.min_accuracy_por_clase = 0.55
        self.frames_estaticos_modelo = 12
        self.learning_rate_base = 0.0008
        self.epochs_base = 250
        logger.info(f"Servicio optimizado para 50 videos/clase: batch_size={self.batch_size}, min_sena_videos={self.min_videos_por_sena}, epochs={self.epochs_base}")

    def _configurar_recursos(self) -> Tuple[int, int, int]:
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        if not torch.cuda.is_available():
            return 8, 20, 0
        memoria_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if memoria_gpu >= 8:
            return 8, 20, num_workers
        return 6, 20, num_workers

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

    def _obtener_videos_utilizables(self, categoria_ids: List[int], db: Session) -> List[Dict[str, Any]]:
        filas = db.query(
            VideoDataset.id, VideoDataset.sena,
            VideoDataset.ruta_video, VideoDataset.drive_file_id,
        ).filter(
            VideoDataset.aprobado == True,
            VideoDataset.categoria_id.in_(categoria_ids)
        ).all()

        return [{"id": int(f.id), "sena": f.sena or "sin_clasificar",
                 "ruta_video": f.ruta_video, "drive_file_id": f.drive_file_id} for f in filas]

    def _cargar_videos_orm(self, ids: List[int]) -> List[VideoDataset]:
        if not ids:
            return []
        db = SessionLocal()
        try:
            videos = db.query(VideoDataset).filter(VideoDataset.id.in_(ids)).all()
            db.expunge_all()
            return videos
        finally:
            db.close()

    def validar_dataset_entrenamiento(self, categoria_ids: List[int], db: Session) -> Dict[str, Any]:
        try:
            videos_db = self._obtener_videos_utilizables(categoria_ids, db)

            videos_validos, videos_no_encontrados = [], []
            for video in videos_db:
                tiene_local = bool(video["ruta_video"] and os.path.exists(video["ruta_video"]))
                tiene_drive = bool(video["drive_file_id"])
                tiene_kp_cliente = self._cargar_keypoints_cliente(video["id"]) is not None
                tiene_secuencia = (self.secuencias_dir / f"{video['id']}.npz").exists()
                if tiene_local or tiene_drive or tiene_kp_cliente or tiene_secuencia:
                    videos_validos.append(video)
                else:
                    videos_no_encontrados.append(video)

            videos_por_sena: Dict[str, List] = {}
            for video in videos_validos:
                videos_por_sena.setdefault(video["sena"], []).append(video)

            senas_con_minimo = [s for s, v in videos_por_sena.items() if len(v) >= self.min_videos_por_sena]
            senas_sin_minimo = [s for s in videos_por_sena if s not in senas_con_minimo]
            senas_estaticas = [s for s in videos_por_sena if detectar_tipo_sena(s) == 'ESTATICA']
            senas_dinamicas = [s for s in videos_por_sena if s not in senas_estaticas]

            consistencia = validar_consistencia_categoria(list(videos_por_sena.keys()))
            es_alfabeto = es_categoria_alfabeto(list(videos_por_sena.keys()))
            total_videos = len(videos_validos)

            valido = (
                total_videos >= 40 and
                len(senas_con_minimo) >= 2 and
                len(senas_sin_minimo) == 0
            )

            return {
                "valido": valido, "total_videos": total_videos,
                "videos_validos": len(videos_validos),
                "videos_no_encontrados": len(videos_no_encontrados),
                "total_senas": len(videos_por_sena),
                "senas_con_minimo": senas_con_minimo,
                "senas_sin_minimo": senas_sin_minimo,
                "senas_estaticas": senas_estaticas,
                "senas_dinamicas": senas_dinamicas,
                "distribucion": {"estaticas": len(senas_estaticas), "dinamicas": len(senas_dinamicas),
                                 "tipo_predominante": consistencia.get('tipo_mayoritario')},
                "consistencia": consistencia, "es_categoria_alfabeto": es_alfabeto,
                "minimo_videos_requerido": 40, "minimo_senas_requerido": 2,
                "minimo_videos_por_sena": self.min_videos_por_sena,
                "cumple_requisitos": {
                    "videos_totales": total_videos >= 40,
                    "senas_suficientes": len(senas_con_minimo) >= 2,
                    "videos_por_sena": len(senas_sin_minimo) == 0
                }
            }
        except Exception as e:
            logger.error(f"Error validando dataset: {e}")
            return {"valido": False, "error": str(e), "total_videos": 0, "videos_validos": 0,
                    "total_senas": 0, "senas_con_minimo": [], "senas_sin_minimo": []}

    def _progreso_por_defecto(self, nombre_modelo: str, estado: str, mensaje: str, entrenando: bool) -> Dict[str, Any]:
        return {
            "nombre_modelo": nombre_modelo, "estado": estado, "progreso": 0.0,
            "epoch_actual": 0, "total_epochs": 0, "accuracy": 0.0, "loss": 0.0,
            "train_loss": 0.0, "train_accuracy": 0.0, "num_clases": 0, "clases": [],
            "total_videos": 0, "frames_procesados": 0, "total_frames": 0,
            "mensaje": mensaje, "entrenando": entrenando, "advertencias_dataset": [],
            "accuracy_por_clase": {},
        }

    def obtener_progreso_entrenamiento(self, nombre_modelo: str) -> Dict[str, Any]:
        with _lock_progresos:
            if nombre_modelo in progresos_entrenamiento:
                p = progresos_entrenamiento[nombre_modelo]
                return {
                    "nombre_modelo": p.get("nombre_modelo", nombre_modelo),
                    "estado": p.get("estado", "desconocido"),
                    "progreso": p.get("progreso", 0.0),
                    "epoch_actual": p.get("epoch_actual", 0),
                    "total_epochs": p.get("total_epochs", 0),
                    "accuracy": p.get("accuracy", 0.0),
                    "loss": p.get("loss", 0.0),
                    "train_loss": p.get("train_loss", 0.0),
                    "train_accuracy": p.get("train_accuracy", 0.0),
                    "num_clases": p.get("num_clases", 0),
                    "clases": p.get("clases", []),
                    "total_videos": p.get("total_videos", 0),
                    "frames_procesados": p.get("frames_procesados", 0),
                    "total_frames": p.get("total_frames", 0),
                    "mensaje": p.get("mensaje", ""),
                    "fecha_inicio": p.get("fecha_inicio", datetime.now().isoformat()),
                    "entrenando": p.get("entrenando", False),
                    "accuracy_por_clase": p.get("accuracy_por_clase", {}),
                    "advertencias_dataset": p.get("advertencias_dataset", [])
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

    def _ajustar_longitud(self, secuencia: np.ndarray) -> np.ndarray:
        if secuencia.shape[0] == self.num_frames:
            return secuencia
        indices = np.linspace(0, secuencia.shape[0] - 1, self.num_frames).round().astype(int)
        return secuencia[indices]

    def _leer_secuencia_guardada(self, video_id: int) -> Tuple[bool, Optional[np.ndarray]]:
        ruta = self.secuencias_dir / f"{video_id}.npz"
        if not ruta.exists():
            return False, None
        try:
            datos = np.load(ruta)
            secuencia = datos["secuencia"]
            estado = datos["estado"]
            version = int(datos["version"]) if "version" in datos else 0
        except Exception as e:
            logger.warning(f"Secuencia ilegible {ruta.name}: {e}")
            return False, None

        if version != VERSION_SECUENCIAS:
            return False, None
        if secuencia.ndim != 2 or secuencia.shape[1] != 126 or secuencia.shape[0] == 0:
            return False, None
        if not (estado > 0).any():
            return True, None
        return True, self._ajustar_longitud(secuencia.astype(np.float32))

    def _guardar_secuencia(self, video_id: int, resultado: ResultadoSecuencia) -> None:
        try:
            np.savez_compressed(
                self.secuencias_dir / f"{video_id}.npz",
                secuencia=resultado.secuencia, estado=resultado.estado,
                version=np.array(VERSION_SECUENCIAS),
            )
        except Exception as e:
            logger.warning(f"No se pudo guardar secuencia {video_id}: {e}")

    def _cargar_keypoints_cliente(self, video_id: int) -> Optional[list]:
        ruta = self.secuencias_dir / f"{video_id}_kp_cliente.npz"
        if not ruta.exists():
            return None
        try:
            return list(np.load(ruta)["keypoints"])
        except Exception as e:
            logger.warning(f"No se pudo leer kp cliente {video_id}: {e}")
            return None

    def _preparar_secuencias(self, categoria_ids: List[int], nombre_modelo: str) -> Tuple[List[np.ndarray], List[str]]:
        db = SessionLocal()
        try:
            videos = self._obtener_videos_utilizables(categoria_ids, db)
        finally:
            db.close()

        logger.info(f"Videos aprobados a procesar: {len(videos)}")

        secuencias: List[np.ndarray] = []
        senas: List[str] = []
        pendientes: List[Dict[str, Any]] = []

        for v in videos:
            existe, secuencia = self._leer_secuencia_guardada(v["id"])
            if existe:
                if secuencia is not None:
                    secuencias.append(secuencia)
                    senas.append(v["sena"])
                continue
            pendientes.append(v)

        logger.info(f"Secuencias en disco: {len(secuencias)} | por generar: {len(pendientes)}")

        if not pendientes:
            return secuencias, senas

        videos_por_sena = {}
        for v in videos:
            videos_por_sena[v["sena"]] = videos_por_sena.get(v["sena"], 0) + 1

        senas_validas = [s for s, count in videos_por_sena.items() if count >= self.min_videos_por_sena]
        if len(senas_validas) < 2:
            logger.warning(f"Advertencia: Solo {len(senas_validas)} señas cumplen mínimo ({self.min_videos_por_sena} videos/clase)")
            logger.warning(f"Distribución: {videos_por_sena}")

        con_kp_cliente: Dict[int, list] = {}
        sin_kp: List[Dict[str, Any]] = []
        for v in pendientes:
            kp = self._cargar_keypoints_cliente(v["id"])
            if kp is not None:
                con_kp_cliente[v["id"]] = kp
            else:
                sin_kp.append(v)

        locales = {v["id"]: v["ruta_video"] for v in sin_kp
                   if v["ruta_video"] and os.path.exists(v["ruta_video"])}
        ids_sin_kp = {v["id"] for v in sin_kp}
        solo_drive = [v for v in videos if v["id"] in ids_sin_kp
                      and v["id"] not in locales and v["drive_file_id"]]

        rutas_temporales: Dict[int, str] = {}
        if solo_drive:
            self._actualizar_progreso(nombre_modelo, mensaje=f"Descargando {len(solo_drive)} videos desde Drive...")
            videos_drive = self._cargar_videos_orm([v["id"] for v in solo_drive])
            rutas_temporales = descargar_videos_temporalmente(videos_drive, carpeta_temporal=f"temp_entrenamiento_{nombre_modelo}")

        with ExitStack() as pila:
            hands = None

            for i, v in enumerate(pendientes):
                video_id = v["id"]
                sena = v["sena"]
                self._actualizar_progreso(nombre_modelo, mensaje=f"Extrayendo secuencias {i + 1}/{len(pendientes)}...")

                if video_id in con_kp_cliente:
                    arrs = [np.asarray(k, dtype=np.float32) for k in con_kp_cliente[video_id]]
                    resultado = secuencia_desde_keypoints(arrs, self.num_frames)
                else:
                    ruta = locales.get(video_id) or rutas_temporales.get(video_id)
                    if not ruta:
                        continue
                    if hands is None:
                        hands = pila.enter_context(detector_para_lote())
                    resultado = extraer_secuencia_landmarks(ruta, self.num_frames, hands=hands)
                    if resultado is None:
                        logger.warning(f"Video {video_id} ilegible, se omite")
                        continue

                self._guardar_secuencia(video_id, resultado)

                if (resultado.estado > 0).any():
                    secuencias.append(resultado.secuencia)
                    senas.append(sena)

        if len(secuencias) < 20:
            raise Exception(f"Insuficientes videos: {len(secuencias)} (mínimo 20)")

        logger.info(f"Dataset preparado: {len(secuencias)} secuencias, {len(set(senas))} señas únicas")
        return secuencias, senas

    def entrenar(self, nombre_modelo: str, categoria_ids: List[int], epochs: int,
                 desactivar_otros: bool = True):
        """
        desactivar_otros=True  -> al terminar, el modelo nuevo queda como único activo (comportamiento anterior).
        desactivar_otros=False -> el modelo nuevo se activa SIN apagar los demás (modo ensemble).
        """
        _fijar_semillas()

        model = train_loader = val_loader = None
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

            secuencias_todas, senas_todas = self._preparar_secuencias(categoria_ids, nombre_modelo)

            self._actualizar_progreso(nombre_modelo, estado="preparando_datos",
                                       mensaje="Preparando dataset...", total_videos=len(secuencias_todas))

            indices_por_sena: Dict[str, List[int]] = {}
            for i, sena in enumerate(senas_todas):
                indices_por_sena.setdefault(sena, []).append(i)

            senas_validas = {k: v for k, v in indices_por_sena.items() if len(v) >= self.min_videos_por_sena}
            if len(senas_validas) < 2:
                raise Exception(f"Se necesitan al menos 2 señas con {self.min_videos_por_sena}+ videos. "
                                f"Señas válidas: {list(senas_validas.keys())}")

            clases = sorted(senas_validas.keys())
            clase_a_idx = {clase: i for i, clase in enumerate(clases)}
            tipos_senas = {clase: detectar_tipo_sena(clase) for clase in clases}

            videos_por_clase = {clase: len(indices) for clase, indices in senas_validas.items()}
            validacion = validar_configuracion_dataset(clases, videos_por_clase)
            if validacion['advertencias']:
                logger.info(f"Advertencias: {validacion['advertencias']}")

            self._actualizar_progreso(nombre_modelo,
                                       advertencias_dataset=validacion['problemas'] + validacion['advertencias'])

            secuencias, labels, senas_list = [], [], []
            for clase, indices in senas_validas.items():
                for i in indices:
                    secuencias.append(secuencias_todas[i])
                    labels.append(clase_a_idx[clase])
                    senas_list.append(clase)

            total_videos = len(secuencias)
            total_frames = total_videos * self.num_frames

            parametros = obtener_parametros_entrenamiento_optimizados(clases)

            if not epochs or epochs <= 0:
                epochs = parametros.get('epochs', self.epochs_base)
                logger.info(f"No llegó un número de épocas válido desde la solicitud; "
                            f"se usa el valor calculado automáticamente: {epochs}")
            else:
                epochs = int(epochs)

            epochs = max(5, min(epochs, 500))

            lr = float(parametros.get('learning_rate', self.learning_rate_base))
            patience = int(parametros.get('patience', 40))
            focal_gamma = float(parametros.get('focal_gamma', 1.8))
            label_smoothing = float(parametros.get('label_smoothing', 0.015))

            batch_size_entrenamiento = min(self.batch_size, int(parametros.get('batch_size', self.batch_size)))

            logger.info(f"Hiperparámetros: lr={lr}, epochs={epochs}, patience={patience}, "
                        f"batch_size={batch_size_entrenamiento}, focal_gamma={focal_gamma}, "
                        f"label_smoothing={label_smoothing}")

            self._actualizar_progreso(
                nombre_modelo, num_clases=len(clases), clases=clases,
                total_videos=total_videos, total_frames=total_frames,
                total_epochs=epochs,
                estado="cargando_datos", mensaje=f"Preparando {total_videos} secuencias..."
            )

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(np.arange(total_videos), labels))

            train_secuencias = [secuencias[i] for i in train_idx]
            val_secuencias = [secuencias[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            train_senas = [senas_list[i] for i in train_idx]
            val_senas = [senas_list[i] for i in val_idx]

            sampler = crear_sampler_balanceado(train_labels, train_senas, clases)

            train_dataset = DatasetKeypointsAdaptativo(train_secuencias, train_labels, train_senas, self.num_frames, True)
            val_dataset = DatasetKeypointsAdaptativo(val_secuencias, val_labels, val_senas, self.num_frames, False)

            train_loader = DataLoader(train_dataset, batch_size=batch_size_entrenamiento, sampler=sampler,
                                       num_workers=0, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_entrenamiento, shuffle=False,
                                     num_workers=0, drop_last=False)

            self._actualizar_progreso(nombre_modelo, estado="creando_modelo", mensaje="Inicializando modelo...")

            model = ModeloAdaptativoSenas(len(clases), tipos_senas, self.frames_estaticos_modelo).to(self.device)
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Modelo con {n_params:,} parámetros")

            criterion = PerdidaContrastivaFocal(clases, alpha=1.0, gamma=focal_gamma,
                                                 label_smoothing=label_smoothing)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay,
                                     betas=(0.9, 0.999))

            warmup_epochs = max(5, min(20, epochs // 10))
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=max(1, len(train_loader)),
                pct_start=warmup_epochs / epochs, anneal_strategy='cos'
            )
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

            self._actualizar_progreso(nombre_modelo, estado="entrenando", mensaje="Entrenamiento en progreso...")

            best_accuracy = 0.0
            mejor_score = -1.0
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

                        frames_procesados = (epoch * len(train_loader) + batch_idx + 1) * batch_size_entrenamiento * self.num_frames
                        self._actualizar_progreso(nombre_modelo, frames_procesados=min(frames_procesados, total_frames))

                        del keypoints, labels_batch, outputs, loss, predicted
                    except RuntimeError as e:
                        if "out of memory" not in str(e):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                model.eval()
                val_loss = val_correct = val_total = 0
                matriz_confusion_epoch = np.zeros((len(clases), len(clases)), dtype=int)

                with torch.no_grad():
                    for keypoints, labels_batch, tipos_batch in val_loader:
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

                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

                accuracy_por_clase_epoch = {}
                for i, clase in enumerate(clases):
                    total_clase_epoch = matriz_confusion_epoch[i].sum()
                    accuracy_por_clase_epoch[clase] = (
                        float(matriz_confusion_epoch[i][i] / total_clase_epoch) if total_clase_epoch > 0 else 0.0
                    )
                min_class_acc = min(accuracy_por_clase_epoch.values()) if accuracy_por_clase_epoch else 0.0

                score_compuesto = 0.6 * val_acc + 0.4 * min_class_acc

                self._actualizar_progreso(
                    nombre_modelo, epoch_actual=epoch + 1, accuracy=float(val_acc),
                    loss=float(avg_val_loss), train_accuracy=float(train_acc),
                    train_loss=float(avg_train_loss), progreso=((epoch + 1) / epochs) * 100.0,
                    accuracy_por_clase=accuracy_por_clase_epoch,
                    mensaje=f"Época {epoch + 1}/{epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                            f"Peor clase: {min_class_acc:.4f}"
                )

                if score_compuesto > mejor_score:
                    mejor_score = score_compuesto
                    best_accuracy = val_acc
                    best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    matriz_confusion_val = matriz_confusion_epoch.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                    logger.info(f"Época {epoch + 1:3d}/{epochs} | Train: {train_acc:.4f} | "
                                f"Val: {val_acc:.4f} | Best: {best_accuracy:.4f} | "
                                f"Peor clase: {min_class_acc:.4f}")

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

            logger.info(f"Accuracy por clase: {accuracy_por_clase}")

            clases_debiles = {c: a for c, a in accuracy_por_clase.items() if a < self.min_accuracy_por_clase}
            if clases_debiles:
                logger.warning(
                    f"Estas clases quedaron por debajo de {self.min_accuracy_por_clase * 100:.0f}% de accuracy: "
                    f"{clases_debiles}. Si alguna es una letra que ya regrabaste (ej. B), revisa la calidad e "
                    f"iluminación de esos videos, o agrega más ejemplos de esa seña."
                )

            ruta_modelo = self._guardar_modelo(model, nombre_modelo, clases, best_accuracy, tipos_senas,
                                                accuracy_por_clase, matriz_confusion_val.tolist())

            db_final = SessionLocal()
            try:
                self._actualizar_bd(db_final, nombre_modelo, str(ruta_modelo), clases,
                                     best_accuracy, epochs, total_videos,
                                     desactivar_otros=desactivar_otros)
            finally:
                db_final.close()

            try:
                from app.servicios.gestor_reconocimiento import limpiar_cache
                limpiar_cache()
            except Exception as e:
                logger.warning(f"No se pudo limpiar cache: {e}")

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
            try:
                limpiar_carpeta_temporal(f"temp_entrenamiento_{nombre_modelo}")
            except Exception as e:
                logger.warning(f"No se pudo limpiar temp: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _guardar_modelo(self, model, nombre_modelo: str, clases: List[str], accuracy: float,
                        tipos_senas: dict, accuracy_por_clase: Dict[str, float],
                        matriz_confusion: list) -> Path:
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
            'extractor_version': 'secuencias_v3',
            'tipos_senas': tipos_senas,
            'fecha_entrenamiento': datetime.now().isoformat(),
            'version': '11.0'
        }, str(ruta_modelo))

        try:
            respaldar_modelo_en_drive(ruta_modelo)
        except Exception as e:
            logger.warning(f"No se pudo respaldar en Drive: {e}")

        return ruta_modelo

    def _actualizar_bd(self, db: Session, nombre_modelo: str, ruta_modelo: str,
                        clases: List[str], accuracy: float, epochs: int, total_videos: int,
                        desactivar_otros: bool = True):
        ultimo_error = None

        for intento in range(3):
            try:
                modelo_db = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

                if not modelo_db:
                    modelo_db = ModeloIA(
                        nombre=nombre_modelo,
                        version="11.0",
                        descripcion=f"Modelo LSTM keypoints 256x2, {len(clases)} señas, {total_videos} videos, "
                                    f"accuracy: {accuracy * 100:.2f}%",
                        ruta_archivo=ruta_modelo,
                        tipo_modelo="LSTM",
                        accuracy=round(float(accuracy), 6),
                        num_clases=len(clases),
                        clases_json=json.dumps(clases, ensure_ascii=False),
                        total_imagenes=total_videos,
                        epocas_entrenamiento=epochs,
                        arquitectura="LSTM_Keypoints_Unificado",
                        entrenando=False,
                        activo=True,
                        fecha_entrenamiento=datetime.now(),
                        num_frames=self.num_frames,
                        origen="entrenado_local",
                    )
                    db.add(modelo_db)
                    db.flush()
                else:
                    modelo_db.ruta_archivo = ruta_modelo
                    modelo_db.tipo_modelo = "LSTM"
                    modelo_db.accuracy = round(float(accuracy), 6)
                    modelo_db.num_clases = len(clases)
                    modelo_db.clases_json = json.dumps(clases, ensure_ascii=False)
                    modelo_db.total_imagenes = total_videos
                    modelo_db.epocas_entrenamiento = epochs
                    modelo_db.arquitectura = "LSTM_Keypoints_Unificado"
                    modelo_db.entrenando = False
                    modelo_db.activo = True
                    modelo_db.fecha_entrenamiento = datetime.now()
                    modelo_db.version = "11.0"
                    modelo_db.num_frames = self.num_frames

                if desactivar_otros:
                    db.query(ModeloIA).filter(
                        ModeloIA.activo == True,
                        ModeloIA.id != modelo_db.id
                    ).update({"activo": False}, synchronize_session=False)

                db.commit()
                db.refresh(modelo_db)
                logger.info(f"Modelo {nombre_modelo} activado (id={modelo_db.id}, desactivar_otros={desactivar_otros})")
                return

            except Exception as e:
                ultimo_error = e
                db.rollback()
                logger.warning(f"Intento {intento + 1}/3 falló: {e}")
                try:
                    db.close()
                except Exception:
                    pass
                from app.utilidades.base_datos import SessionLocal as _SL
                db = _SL()

        logger.error(f"No se pudo actualizar BD tras 3 intentos: {ultimo_error}")
        raise ultimo_error


servicio_entrenamiento = ServicioEntrenamientoUnificado()