import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.entrenamiento import ModeloIA
from ..servicios.gestor_reconocimiento import (
    obtener_reconocedor,
    obtener_reconocedor_cacheado,
    determinar_tipo_sena,
    procesar_frame_base64,
    extraer_keypoints_frame,
    limpiar_cache,
    cerrar_mediapipe_hands,
    info_cache,
)

router = APIRouter(prefix="/reconocimiento-video", tags=["Reconocimiento Video"])
logger = logging.getLogger(__name__)


class FrameData(BaseModel):
    frame_base64: str
    timestamp: int
    landmarks: Optional[List[Any]] = None


class SolicitudVideoSecuencia(BaseModel):
    frames: List[FrameData]
    configuracion: Dict[str, Any] = {}
    sena_esperada: Optional[str] = None
    categoria_id: Optional[int] = None


CATEGORIAS_CALIDAD = (
    (0.80, "excelente", "Alta confianza"),
    (0.70, "buena", "Buena confianza"),
    (0.60, "moderada", "Confianza moderada"),
    (0.50, "baja", "Baja confianza"),
)


MAX_FRAMES_EXTRACCION = 15


def _evaluar_calidad(confianza: float, sena_detectada: str) -> Dict[str, str]:
    for umbral, calidad, etiqueta in CATEGORIAS_CALIDAD:
        if confianza >= umbral:
            return {"calidad": calidad, "mensaje": f"{etiqueta}: {sena_detectada}"}
    return {"calidad": "muy_baja", "mensaje": "Seña no reconocida claramente"}


@router.post("/secuencia", response_model=Dict[str, Any])
async def reconocer_secuencia_video(
    solicitud: SolicitudVideoSecuencia,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    start_time = time.time()

    try:
        num_frames_recibidos = len(solicitud.frames)
        reconocedor, modelo_id = obtener_reconocedor(db, solicitud.categoria_id)

        frames_a_procesar = (
            solicitud.frames[-MAX_FRAMES_EXTRACCION:]
            if num_frames_recibidos > MAX_FRAMES_EXTRACCION
            else solicitud.frames
        )
        if num_frames_recibidos > MAX_FRAMES_EXTRACCION:
            logger.info(f"Optimizando: {num_frames_recibidos} -> {MAX_FRAMES_EXTRACCION} frames")

        keypoints_secuencia = []
        for frame_data in frames_a_procesar:
            frame = procesar_frame_base64(frame_data.frame_base64)
            if frame is None:
                continue
            keypoints = extraer_keypoints_frame(frame)
            if keypoints is not None:
                keypoints_secuencia.append(keypoints)

        frames_exitosos = len(keypoints_secuencia)

        if frames_exitosos < 4:
            return {
                "exito": False,
                "mensaje": f"Frames insuficientes ({frames_exitosos}/4)",
                "datos": {
                    "confianza": 0.0,
                    "sena_detectada": "",
                    "num_frames_procesados": frames_exitosos,
                    "categoria_id": solicitud.categoria_id
                }
            }

        tipo_sena_inicial = determinar_tipo_sena(
            db, solicitud.sena_esperada, solicitud.categoria_id, reconocedor,
            keypoints_secuencia=keypoints_secuencia
        )
        logger.info(f"Tipo inicial (hint): {tipo_sena_inicial}")

        sena_detectada, confianza, detalles = reconocedor.predecir_adaptativo(keypoints_secuencia, tipo_sena_inicial)
        tipo_sena = detalles.get("tipo_sena", tipo_sena_inicial)

        confianza_validada = max(0.0, min(1.0, confianza))
        processing_time = time.time() - start_time
        evaluacion = _evaluar_calidad(confianza_validada, sena_detectada)

        alternativas_filtradas = sorted(
            [alt for alt in detalles.get("alternativas", []) if alt.get("confianza", 0) >= 0.2],
            key=lambda x: x.get("confianza", 0),
            reverse=True
        )[:3]

        resultado = {
            "sena_detectada": sena_detectada,
            "texto_traducido": sena_detectada,
            "confianza": round(confianza_validada, 4),
            "confianza_raw": round(confianza_validada, 4),
            "porcentaje": round(confianza_validada * 100, 2),
            "calidad": evaluacion["calidad"],
            "mensaje": evaluacion["mensaje"],
            "modo": tipo_sena,
            "modo_inicial": tipo_sena_inicial,
            "reevaluado": detalles.get("reevaluado", False),
            "num_frames_procesados": frames_exitosos,
            "num_frames_usados_prediccion": detalles.get("num_frames_usado", reconocedor.num_frames),
            "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
            "fps": round(frames_exitosos / processing_time, 2) if processing_time > 0 else 0,
            "alternativas": alternativas_filtradas,
            "modelo_usado": {
                "modelo_id": modelo_id,
                "categoria_id": solicitud.categoria_id,
                "modelo_nombre": reconocedor.arquitectura,
                "modo_usado": tipo_sena
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            f"Reconocimiento [Cat:{solicitud.categoria_id}]: {sena_detectada} "
            f"({confianza_validada * 100:.1f}%) - Modo: {tipo_sena} (inicial: {tipo_sena_inicial})"
        )

        return {"exito": True, "mensaje": "Reconocimiento completado", "datos": resultado}

    except Exception as e:
        logger.error(f"Error en reconocimiento [Cat:{solicitud.categoria_id}]: {e}", exc_info=True)
        return {
            "exito": False,
            "mensaje": "Error en el reconocimiento",
            "datos": {
                "confianza": 0.0,
                "sena_detectada": "",
                "num_frames_procesados": 0,
                "categoria_id": solicitud.categoria_id
            },
            "errores": [f"Error interno: {e}"]
        }


@router.get("/info-modelo")
async def obtener_info_modelo(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        reconocedor, modelo_id = obtener_reconocedor(db)
        modelo_db = db.query(ModeloIA).filter(ModeloIA.id == modelo_id).first()

        return {
            "exito": True,
            "modelo": {
                "id": modelo_db.id if modelo_db else None,
                "nombre": modelo_db.nombre if modelo_db else "desconocido",
                "arquitectura": reconocedor.arquitectura,
                "num_clases": reconocedor.num_clases,
                "clases": reconocedor.clases,
                "num_frames": reconocedor.num_frames,
                "accuracy_entrenamiento": round(reconocedor.accuracy * 100, 2),
                "fecha_entrenamiento": modelo_db.fecha_entrenamiento.isoformat() if modelo_db and modelo_db.fecha_entrenamiento else None,
                "tipos_senas": reconocedor.tipos_senas
            },
            "sistema": {
                "device": str(reconocedor.device),
                "cuda_disponible": str(reconocedor.device).startswith('cuda'),
                "amp_enabled": reconocedor.enable_amp,
                "modo": "keypoints_lstm"
            }
        }

    except Exception as e:
        logger.error(f"Error obteniendo info: {e}")
        return {"exito": False, "mensaje": f"Error: {e}"}


@router.get("/estado")
async def obtener_estado_reconocimiento(db: Session = Depends(obtener_bd)):
    try:
        cache_info = info_cache()
        reconocedor = None
        modelo_id = None

        if cache_info['modelos_cargados']:
            modelo_id = cache_info['modelos_cargados'][0]
            reconocedor = obtener_reconocedor_cacheado(modelo_id)
        else:
            try:
                reconocedor, modelo_id = obtener_reconocedor(db)
            except Exception as e:
                logger.warning(f"No se pudo cargar reconocedor: {e}")

        return {
            "estado": "activo" if reconocedor is not None else "inactivo",
            "modelo_cargado": reconocedor is not None,
            "arquitectura": reconocedor.arquitectura if reconocedor else "ninguna",
            "clases_cargadas": len(reconocedor.clases) if reconocedor else 0,
            "clases": reconocedor.clases if reconocedor else [],
            "num_frames": reconocedor.num_frames if reconocedor else 0,
            "device": str(reconocedor.device) if reconocedor else "ninguno",
            "ultimo_uso": cache_info['ultimo_acceso'].get(modelo_id) if modelo_id else None,
            "modo": "keypoints_lstm",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        return {"estado": "error", "error": str(e), "timestamp": datetime.now().isoformat()}


@router.post("/limpiar-cache")
async def limpiar_cache_modelo():
    try:
        mensaje = limpiar_cache()
        cerrar_mediapipe_hands()

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Cache del modelo limpiado")
        return {"exito": True, "mensaje": mensaje}
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return {"exito": False, "mensaje": f"Error: {e}"}