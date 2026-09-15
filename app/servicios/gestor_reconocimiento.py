import base64
import logging
from threading import Lock
from datetime import datetime
from typing import Optional, Tuple, List

import cv2
import torch
import numpy as np
import mediapipe as mp
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..modelos.entrenamiento import ModeloIA
from ..modelos.categoria import Categoria
from ..servicios.servicio_reconocimiento import ServicioReconocimiento
from ..servicios.config_tipo_senas import (
    detectar_tipo_sena,
    detectar_tipo_por_movimiento,
    redimensionar_manteniendo_aspecto,
)

logger = logging.getLogger(__name__)

_cache: dict = {
    'reconocedores': {},
    'ultimo_acceso': {},
    'warmup_hecho': set(),
}
_lock_cache = Lock()

_mediapipe_hands = None
_lock_mediapipe = Lock()


def obtener_mediapipe_hands() -> "mp.solutions.hands.Hands":
    global _mediapipe_hands
    with _lock_mediapipe:
        if _mediapipe_hands is None:
            _mediapipe_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return _mediapipe_hands


def cerrar_mediapipe_hands():
    global _mediapipe_hands
    with _lock_mediapipe:
        if _mediapipe_hands is not None:
            _mediapipe_hands.close()
            _mediapipe_hands = None


def procesar_frame_base64(frame_base64: str) -> Optional[np.ndarray]:
    try:
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',')[1]
        frame_bytes = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = redimensionar_manteniendo_aspecto(frame, (224, 224))
        return frame
    except Exception as e:
        logger.warning(f"Error procesando frame: {e}")
        return None


def extraer_keypoints_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    try:
        hands = obtener_mediapipe_hands()
        resultado = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


def precalentar_modelo(reconocedor: ServicioReconocimiento, modelo_id: int):
    with _lock_cache:
        if modelo_id in _cache['warmup_hecho']:
            return

    try:
        logger.info(f"Precalentando modelo {modelo_id}...")
        dummy_keypoints = np.random.randn(reconocedor.num_frames, 126).astype(np.float32)
        dummy_tensor = torch.from_numpy(dummy_keypoints).float().unsqueeze(0).to(reconocedor.device)

        with torch.no_grad():
            for _ in range(3):
                reconocedor.predecir(dummy_tensor, 'DINAMICA', None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with _lock_cache:
            _cache['warmup_hecho'].add(modelo_id)
        logger.info(f"Modelo {modelo_id} precalentado exitosamente")
    except Exception as e:
        logger.warning(f"Error en precalentamiento del modelo {modelo_id}: {e}")


def obtener_modelo_por_categoria(db: Session, categoria_id: Optional[int] = None) -> ModeloIA:
    if categoria_id:
        categoria = db.query(Categoria).filter(Categoria.id == categoria_id).first()
        if categoria and categoria.modelo_ia_id:
            modelo_categoria = db.query(ModeloIA).filter(
                ModeloIA.id == categoria.modelo_ia_id,
                ModeloIA.activo == True
            ).first()
            if modelo_categoria:
                logger.info(f"Modelo categoría: {categoria.nombre} -> {modelo_categoria.nombre}")
                return modelo_categoria

    modelo_global = db.query(ModeloIA).filter(ModeloIA.activo == True).first()
    if modelo_global:
        logger.info(f"Modelo global: {modelo_global.nombre}")
        return modelo_global

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No hay modelo disponible para reconocimiento"
    )


def obtener_reconocedor(db: Session, categoria_id: Optional[int] = None) -> Tuple[ServicioReconocimiento, int]:
    try:
        modelo_a_usar = obtener_modelo_por_categoria(db, categoria_id)

        with _lock_cache:
            if modelo_a_usar.id in _cache['reconocedores']:
                _cache['ultimo_acceso'][modelo_a_usar.id] = datetime.now()
                return _cache['reconocedores'][modelo_a_usar.id], modelo_a_usar.id

        logger.info(f"Cargando modelo: {modelo_a_usar.nombre}")
        reconocedor = ServicioReconocimiento(ruta_modelo=modelo_a_usar.ruta_archivo)

        precalentar_modelo(reconocedor, modelo_a_usar.id)

        with _lock_cache:
            _cache['reconocedores'][modelo_a_usar.id] = reconocedor
            _cache['ultimo_acceso'][modelo_a_usar.id] = datetime.now()

        return reconocedor, modelo_a_usar.id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo reconocedor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cargando modelo: {e}"
        )


def determinar_tipo_sena(
    db: Session,
    sena_esperada: Optional[str],
    categoria_id: Optional[int],
    reconocedor: Optional[ServicioReconocimiento] = None,
    keypoints_secuencia: Optional[List[np.ndarray]] = None,
) -> str:
    if sena_esperada:
        tipo = detectar_tipo_sena(sena_esperada)
        if tipo:
            return tipo

    if keypoints_secuencia:
        tipo_detectado, detalles = detectar_tipo_por_movimiento(keypoints_secuencia)
        logger.info(f"Tipo (hint) detectado por movimiento: {tipo_detectado} | detalles={detalles}")
        return tipo_detectado

    if reconocedor is not None and reconocedor.clases:
        estaticas = sum(1 for t in reconocedor.tipos_senas.values() if t == 'ESTATICA')
        if estaticas >= len(reconocedor.clases) / 2:
            logger.warning(
                "determinar_tipo_sena: usando heurístico de mayoría global "
                "(sin sena_esperada ni keypoints)"
            )
            return 'ESTATICA'
        return 'DINAMICA'

    return 'DINAMICA'


def limpiar_cache(modelo_id: Optional[int] = None) -> str:
    with _lock_cache:
        if modelo_id:
            _cache['reconocedores'].pop(modelo_id, None)
            _cache['ultimo_acceso'].pop(modelo_id, None)
            _cache['warmup_hecho'].discard(modelo_id)
            mensaje = f"Cache del modelo {modelo_id} limpiado"
        else:
            _cache['reconocedores'].clear()
            _cache['ultimo_acceso'].clear()
            _cache['warmup_hecho'].clear()
            mensaje = "Cache de todos los modelos limpiado"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mensaje


def obtener_reconocedor_cacheado(modelo_id: int) -> Optional[ServicioReconocimiento]:
    with _lock_cache:
        return _cache['reconocedores'].get(modelo_id)


def info_cache() -> dict:
    with _lock_cache:
        return {
            'modelos_cargados': list(_cache['reconocedores'].keys()),
            'modelos_precalentados': list(_cache['warmup_hecho']),
            'ultimo_acceso': {k: v.isoformat() for k, v in _cache['ultimo_acceso'].items()},
        }