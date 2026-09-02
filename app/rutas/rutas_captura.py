from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import tempfile
import os

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.dataset import CategoriaDataset, VideoDataset
from ..esquemas.respuestas import RespuestaAPI

router = APIRouter(prefix="/captura-entrenamiento", tags=["Captura en Tiempo Real"])
logger = logging.getLogger(__name__)

@router.post("/capturar-video-archivo", response_model=RespuestaAPI)
async def capturar_video_archivo(
    archivo: UploadFile = File(...),
    categoria_id: int = Form(...),
    sena: str = Form(...),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Endpoint mejorado para captura de videos con manejo robusto de formatos.
    """
    try:
        # Validar categoría
        categoria = db.query(CategoriaDataset).filter(
            CategoriaDataset.id == categoria_id,
            CategoriaDataset.activa == True
        ).first()
        
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Categoría no encontrada o inactiva"
            )

        # Validar tipo de archivo
        if not archivo.content_type or not archivo.content_type.startswith('video/'):
            logger.warning(f"Tipo de archivo no válido: {archivo.content_type}")
            # Pero continuamos por si el content-type es incorrecto

        # Usar el servicio mejorado
        from ..servicios.dataset_service import dataset_service
        
        # Subir el video usando el servicio
        video = await dataset_service.subir_video_dataset(
            db=db,
            archivo=archivo,
            categoria_id=categoria_id,
            sena=sena,
            usuario_id=usuario_actual.id
        )
        
        logger.info(f"Video procesado exitosamente: {video.id} con {video.frames_extraidos} frames")
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Video procesado exitosamente. {video.frames_extraidos} frames extraídos. {'✓ Aprobado' if video.aprobado else '⏳ Pendiente de revisión'}.",
            datos={
                "id": video.id,
                "sena": video.sena,
                "duracion": video.duracion_segundos,
                "frames_extraidos": video.frames_extraidos,
                "calidad_promedio": round(video.calidad_promedio, 2),
                "aprobado": video.aprobado,
                "fps": video.fps,
                "estado": "aprobado" if video.aprobado else "pendiente"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error capturando video: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando video: {str(e)}"
        )

@router.get("/estadisticas-captura", response_model=RespuestaAPI)
async def obtener_estadisticas_captura(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """
    Obtiene estadísticas de captura del usuario actual.
    """
    try:
        from sqlalchemy import func
        
        # Estadísticas del usuario
        total_videos = db.query(VideoDataset).filter(
            VideoDataset.usuario_id == usuario_actual.id
        ).count()
        
        videos_aprobados = db.query(VideoDataset).filter(
            VideoDataset.usuario_id == usuario_actual.id,
            VideoDataset.aprobado == True
        ).count()
        
        videos_pendientes = db.query(VideoDataset).filter(
            VideoDataset.usuario_id == usuario_actual.id,
            VideoDataset.aprobado == False
        ).count()
        
        # Promedio de frames extraídos
        avg_frames = db.query(func.avg(VideoDataset.frames_extraidos)).filter(
            VideoDataset.usuario_id == usuario_actual.id
        ).scalar() or 0
        
        # Señas únicas capturadas
        senas_unicas = db.query(VideoDataset.sena).filter(
            VideoDataset.usuario_id == usuario_actual.id
        ).distinct().count()
        
        return RespuestaAPI(
            exito=True,
            mensaje="Estadísticas obtenidas correctamente",
            datos={
                "total_videos": total_videos,
                "videos_aprobados": videos_aprobados,
                "videos_pendientes": videos_pendientes,
                "tasa_aprobacion": round((videos_aprobados / total_videos * 100) if total_videos > 0 else 0, 1),
                "promedio_frames": round(avg_frames, 1),
                "senas_unicas": senas_unicas,
                "usuario": {
                    "id": usuario_actual.id,
                    "nombre": usuario_actual.nombre
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo estadísticas"
        )