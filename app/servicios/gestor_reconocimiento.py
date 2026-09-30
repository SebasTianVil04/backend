import base64
import json
import logging
import os
from threading import Lock
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, TYPE_CHECKING

import cv2
import torch
import numpy as np
import mediapipe as mp
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..modelos.entrenamiento import ModeloIA
from ..modelos.categoria import Categoria
from ..modelos.dataset import CategoriaDataset
from ..servicios.config_tipo_senas import (
    detectar_tipo_sena,
    detectar_tipo_por_movimiento,
    redimensionar_manteniendo_aspecto,
)
from ..servicios.extractor_secuencias import normalizar_fila

if TYPE_CHECKING:
    from ..servicios.servicio_reconocimiento import ServicioReconocimiento

logger = logging.getLogger(__name__)

FEATURES_POR_MANO = 63
FEATURES_TOTALES = 126
MODELOS_DIR = Path("modelos_entrenados")

_cache: dict = {
    'reconocedores': {},
    'ultimo_acceso': {},
    'warmup_hecho': set(),
}
_lock_cache = Lock()

_mediapipe_hands = None
_lock_mediapipe = Lock()


def _importar_servicio_reconocimiento():
    from ..servicios.servicio_reconocimiento import ServicioReconocimiento
    return ServicioReconocimiento


def obtener_mediapipe_hands() -> "mp.solutions.hands.Hands":
    global _mediapipe_hands
    with _lock_mediapipe:
        if _mediapipe_hands is None:
            _mediapipe_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
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
        if frame.shape[0] < 224 or frame.shape[1] < 224:
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

        keypoints = np.zeros(FEATURES_TOTALES, dtype=np.float32)
        hay_mano = False

        for landmarks, lateralidad in zip(resultado.multi_hand_landmarks, resultado.multi_handedness):
            etiqueta = lateralidad.classification[0].label
            slot = 0 if etiqueta == 'Right' else 1
            base = slot * FEATURES_POR_MANO

            for j, punto in enumerate(landmarks.landmark):
                keypoints[base + j * 3] = punto.x
                keypoints[base + j * 3 + 1] = punto.y
                keypoints[base + j * 3 + 2] = punto.z

            hay_mano = True

        if not hay_mano:
            return None

        return normalizar_fila(keypoints)
    except Exception as e:
        logger.warning(f"Error extrayendo keypoints: {e}")
        return None


def precalentar_modelo(reconocedor: "ServicioReconocimiento", modelo_id: int):
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


def _registrar_modelo_desde_disco(db: Session) -> Optional[ModeloIA]:
    if not MODELOS_DIR.exists():
        return None

    archivos = sorted(
        MODELOS_DIR.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not archivos:
        return None

    rutas_bd = {
        str(Path(r[0]).resolve())
        for r in db.query(ModeloIA.ruta_archivo).filter(ModeloIA.ruta_archivo.isnot(None)).all()
    }

    for archivo in archivos:
        ruta_abs = str(archivo.resolve())
        if ruta_abs in rutas_bd:
            continue

        try:
            checkpoint = torch.load(ruta_abs, map_location='cpu', weights_only=True)
            clases = checkpoint.get('clases', [])
            if not clases:
                continue

            modelo = ModeloIA(
                nombre=archivo.stem,
                descripcion=f"Modelo recuperado desde disco ({archivo.name})",
                ruta_archivo=ruta_abs,
                tipo_modelo="LSTM",
                num_clases=len(clases),
                clases_json=json.dumps(clases, ensure_ascii=False),
                accuracy=float(checkpoint.get('accuracy', 0.0)),
                activo=True,
                fecha_entrenamiento=datetime.now(),
                epocas_entrenamiento=0,
                total_imagenes=0,
                arquitectura=checkpoint.get('arquitectura', 'ModeloAdaptativoSenas_LSTM'),
                version=checkpoint.get('version', '8.0'),
            )
            db.add(modelo)
            db.commit()
            db.refresh(modelo)
            logger.info(f"Modelo recuperado desde disco y registrado: {modelo.nombre} (id={modelo.id})")
            return modelo
        except Exception as e:
            db.rollback()
            logger.warning(f"No se pudo registrar {archivo.name}: {e}")

    return None


def _resolver_categoria(db: Session, categoria_id: int) -> Optional[Categoria]:
    categoria = db.query(Categoria).filter(Categoria.id == categoria_id).first()
    if categoria:
        logger.info(
            f"[SELECCION] categoria_id={categoria_id} resuelto como Categoria.id='{categoria.nombre}' "
            f"(modelo_ia_id={categoria.modelo_ia_id})"
        )
        return categoria

    cat_dataset = db.query(CategoriaDataset).filter(CategoriaDataset.id == categoria_id).first()
    if cat_dataset and cat_dataset.categoria_id:
        categoria = db.query(Categoria).filter(Categoria.id == cat_dataset.categoria_id).first()
        if categoria:
            logger.info(
                f"[SELECCION] categoria_id={categoria_id} resuelto como CategoriaDataset.id='{cat_dataset.nombre}' "
                f"-> Categoria.id='{categoria.nombre}' (modelo_ia_id={categoria.modelo_ia_id})"
            )
            return categoria

    logger.warning(
        f"[SELECCION] No se pudo resolver categoria_id={categoria_id} ni en Categoria ni en CategoriaDataset"
    )
    return None


def obtener_modelo_por_categoria(db: Session, categoria_id: Optional[int] = None) -> ModeloIA:
    if categoria_id:
        categoria = _resolver_categoria(db, categoria_id)
        if categoria is not None:
            if categoria.modelo_ia_id:
                modelo_categoria = db.query(ModeloIA).filter(
                    ModeloIA.id == categoria.modelo_ia_id
                ).first()
                if modelo_categoria is not None:
                    if not modelo_categoria.activo:
                        logger.warning(
                            f"[SELECCION] El modelo '{modelo_categoria.nombre}' asignado a "
                            f"la categoría '{categoria.nombre}' está INACTIVO. Se usará de todos modos "
                            f"(asignación explícita). Actívalo o reasigna si no es lo que quieres."
                        )
                    else:
                        logger.info(
                            f"[SELECCION] Categoría '{categoria.nombre}' -> modelo "
                            f"'{modelo_categoria.nombre}' (id={modelo_categoria.id}, activo=True)"
                        )
                    return modelo_categoria
                logger.warning(
                    f"[SELECCION] Categoría '{categoria.nombre}' tiene modelo_ia_id={categoria.modelo_ia_id} "
                    f"pero ese modelo no existe en la BD"
                )
            else:
                logger.warning(
                    f"[SELECCION] Categoría '{categoria.nombre}' (id={categoria.id}) no tiene modelo asignado "
                    f"(modelo_ia_id=None); usando fallback al modelo activo más reciente"
                )

    modelo_global = (
        db.query(ModeloIA)
        .filter(ModeloIA.activo == True)
        .order_by(ModeloIA.fecha_entrenamiento.desc().nullslast(), ModeloIA.id.desc())
        .first()
    )
    if modelo_global:
        logger.info(
            f"[SELECCION] Fallback: modelo activo más reciente -> '{modelo_global.nombre}' "
            f"(id={modelo_global.id})"
        )
        return modelo_global

    recuperado = _registrar_modelo_desde_disco(db)
    if recuperado:
        logger.info(f"[SELECCION] Fallback desde disco: '{recuperado.nombre}' (id={recuperado.id})")
        return recuperado

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No hay modelo disponible para reconocimiento"
    )


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

    logger.info(f"[CACHE] {mensaje}")
    return mensaje


def sincronizar_cache_con_bd(db: Session) -> None:
    with _lock_cache:
        ids_cacheados = set(_cache['reconocedores'].keys())

    if not ids_cacheados:
        return

    activos_bd = {
        r[0] for r in db.query(ModeloIA.id).filter(ModeloIA.activo == True).all()
    }

    modelos_obsoletos = ids_cacheados - activos_bd

    if modelos_obsoletos:
        logger.info(
            f"[CACHE] Detectados modelos obsoletos en cache: {modelos_obsoletos}. "
            "Limpiando cache de reconocimiento."
        )
        limpiar_cache()


def _cargar_reconocedor(
    db: Session,
    modelo: ModeloIA,
    permitir_fallback_disco: bool = True,
) -> "ServicioReconocimiento":
    """
    Devuelve el reconocedor del modelo indicado, usando la caché si sigue vigente
    (mismas clases que en BD) o cargándolo desde disco.

    permitir_fallback_disco=True mantiene el comportamiento original: si el modelo no
    tiene archivo válido, se toma el .pth más reciente de la carpeta. Para una selección
    manual se pasa False, porque cargar otro archivo distinto al elegido sería engañoso.
    """
    with _lock_cache:
        if modelo.id in _cache['reconocedores']:
            reconocedor_cacheado = _cache['reconocedores'][modelo.id]
            cache_clases = set(reconocedor_cacheado.clases)
            bd_clases = set()
            if modelo.clases_json:
                try:
                    bd_clases = set(json.loads(modelo.clases_json))
                except Exception:
                    pass

            if cache_clases == bd_clases:
                _cache['ultimo_acceso'][modelo.id] = datetime.now()
                return reconocedor_cacheado

            logger.warning(
                f"[CACHE] Modelo {modelo.id} cambió sus clases en BD "
                f"(cache={cache_clases}, bd={bd_clases}). Recargando."
            )
            _cache['reconocedores'].pop(modelo.id, None)
            _cache['warmup_hecho'].discard(modelo.id)

    ServicioReconocimiento = _importar_servicio_reconocimiento()

    ruta = modelo.ruta_archivo
    if not ruta or not os.path.exists(ruta):
        if not permitir_fallback_disco:
            raise FileNotFoundError(f"El archivo del modelo no existe en disco: {ruta}")

        logger.warning(
            f"[CACHE] Modelo {modelo.nombre} sin archivo válido ({ruta}). "
            "Buscando el .pth más reciente en disco..."
        )
        if not MODELOS_DIR.exists():
            raise FileNotFoundError("No hay carpeta modelos_entrenados")
        candidatos = sorted(
            MODELOS_DIR.glob("*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not candidatos:
            raise FileNotFoundError("No hay archivos .pth en disco")
        ruta = str(candidatos[0])
        modelo.ruta_archivo = ruta
        try:
            db.commit()
        except Exception:
            db.rollback()

    logger.info(f"[CACHE] Cargando modelo: {modelo.nombre} desde {ruta}")
    reconocedor = ServicioReconocimiento(ruta_modelo=ruta)
    logger.info(
        f"[CACHE] Modelo {modelo.nombre} cargado con éxito "
        f"({len(reconocedor.clases)} clases, num_frames={reconocedor.num_frames})"
    )

    precalentar_modelo(reconocedor, modelo.id)

    with _lock_cache:
        _cache['reconocedores'][modelo.id] = reconocedor
        _cache['ultimo_acceso'][modelo.id] = datetime.now()

    return reconocedor


def obtener_reconocedor(db: Session, categoria_id: Optional[int] = None) -> Tuple["ServicioReconocimiento", int]:
    try:
        sincronizar_cache_con_bd(db)

        modelo_a_usar = obtener_modelo_por_categoria(db, categoria_id)
        reconocedor = _cargar_reconocedor(db, modelo_a_usar, permitir_fallback_disco=True)

        return reconocedor, modelo_a_usar.id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CACHE] Error obteniendo reconocedor: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cargando modelo: {e}"
        )


def obtener_reconocedor_por_modelo(db: Session, modelo_id: int) -> Tuple["ServicioReconocimiento", int]:
    """
    Reconocedor de un modelo elegido explícitamente por el usuario.
    Lanza ValueError (con mensaje apto para mostrar) si el modelo no existe,
    no está activo, no tiene archivo en disco o no se puede cargar.
    """
    sincronizar_cache_con_bd(db)

    modelo = db.query(ModeloIA).filter(ModeloIA.id == modelo_id).first()

    if modelo is None:
        raise ValueError("El modelo seleccionado no existe")
    if not modelo.activo:
        raise ValueError(f"El modelo '{modelo.nombre}' ya no está activo")
    if not modelo.ruta_archivo or not os.path.exists(modelo.ruta_archivo):
        raise ValueError(f"El archivo del modelo '{modelo.nombre}' no existe en disco")

    try:
        reconocedor = _cargar_reconocedor(db, modelo, permitir_fallback_disco=False)
    except Exception as e:
        logger.error(f"[CACHE] Error cargando modelo {modelo_id}: {e}", exc_info=True)
        raise ValueError(f"No se pudo cargar el modelo '{modelo.nombre}': {e}")

    logger.info(f"[SELECCION] Selección manual -> modelo '{modelo.nombre}' (id={modelo.id})")
    return reconocedor, modelo.id


def determinar_tipo_sena(
    db: Session,
    sena_esperada: Optional[str],
    categoria_id: Optional[int],
    reconocedor: Optional["ServicioReconocimiento"] = None,
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


def obtener_reconocedor_cacheado(modelo_id: int) -> Optional["ServicioReconocimiento"]:
    with _lock_cache:
        return _cache['reconocedores'].get(modelo_id)


def info_cache() -> dict:
    with _lock_cache:
        return {
            'modelos_cargados': list(_cache['reconocedores'].keys()),
            'modelos_precalentados': list(_cache['warmup_hecho']),
            'ultimo_acceso': {k: v.isoformat() for k, v in _cache['ultimo_acceso'].items()},
        }