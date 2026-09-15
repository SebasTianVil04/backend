import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ..esquemas.respuestas import RespuestaAPI
from ..modelos.dataset import CategoriaDataset, VideoDataset
from ..modelos.usuario import Usuario
from ..servicios.dataset_service import dataset_service
from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual


router = APIRouter(prefix="/captura-entrenamiento", tags=["Captura en Tiempo Real"])
logger = logging.getLogger(__name__)

EXTENSIONES_PERMITIDAS = {"webm", "mp4"}
TAMANO_MAXIMO_BYTES = 50 * 1024 * 1024


@router.post("/capturar-video-archivo", response_model=RespuestaAPI)
async def capturar_video_archivo(
    archivo: UploadFile = File(...),
    categoria_id: int = Form(...),
    sena: str = Form(...),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    try:
        if not sena or not sena.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre de la seña es requerido",
            )

        extension = (archivo.filename or "").rsplit(".", 1)[-1].lower()
        if extension not in EXTENSIONES_PERMITIDAS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Formato de video no permitido: .{extension}",
            )

        categoria = (
            db.query(CategoriaDataset)
            .filter(
                CategoriaDataset.id == categoria_id,
                CategoriaDataset.activa.is_(True),
            )
            .first()
        )
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Categoría no encontrada o inactiva",
            )

        video = await dataset_service.subir_video_dataset(
            db=db,
            archivo=archivo,
            categoria_id=categoria_id,
            sena=sena.strip().upper(),
            usuario_id=usuario_actual.id,
        )

        estado = "aprobado" if video.aprobado else "pendiente"
        etiqueta = "Aprobado" if video.aprobado else "Pendiente de revisión"

        logger.info(
            "Video procesado: id=%s frames=%s aprobado=%s usuario=%s",
            video.id,
            video.frames_extraidos,
            video.aprobado,
            usuario_actual.id,
        )

        return RespuestaAPI(
            exito=True,
            mensaje=(
                f"Video procesado exitosamente. "
                f"{video.frames_extraidos} frames extraídos. {etiqueta}."
            ),
            datos={
                "id": video.id,
                "sena": video.sena,
                "duracion": video.duracion_segundos,
                "frames_extraidos": video.frames_extraidos,
                "calidad_promedio": round(video.calidad_promedio or 0, 2),
                "aprobado": video.aprobado,
                "fps": video.fps,
                "estado": estado,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Error capturando video: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando video: {str(e)}",
        )


@router.get("/estadisticas-captura", response_model=RespuestaAPI)
async def obtener_estadisticas_captura(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    try:
        stats = (
            db.query(
                func.count(VideoDataset.id).label("total"),
                func.sum(
                    case((VideoDataset.aprobado.is_(True), 1), else_=0)
                ).label("aprobados"),
                func.avg(VideoDataset.frames_extraidos).label("avg_frames"),
                func.count(func.distinct(VideoDataset.sena)).label("senas_unicas"),
            )
            .filter(VideoDataset.usuario_id == usuario_actual.id)
            .first()
        )

        total_videos = int(stats.total or 0)
        videos_aprobados = int(stats.aprobados or 0)
        videos_pendientes = total_videos - videos_aprobados
        avg_frames = float(stats.avg_frames or 0)
        senas_unicas = int(stats.senas_unicas or 0)

        tasa_aprobacion = (
            round(videos_aprobados / total_videos * 100, 1) if total_videos > 0 else 0
        )

        return RespuestaAPI(
            exito=True,
            mensaje="Estadísticas obtenidas correctamente",
            datos={
                "total_videos": total_videos,
                "videos_aprobados": videos_aprobados,
                "videos_pendientes": videos_pendientes,
                "tasa_aprobacion": tasa_aprobacion,
                "promedio_frames": round(avg_frames, 1),
                "senas_unicas": senas_unicas,
                "usuario": {
                    "id": usuario_actual.id,
                    "nombre": f"{usuario_actual.nombres} {usuario_actual.apellido_paterno}",
                },
            },
        )

    except Exception as e:
        logger.error("Error obteniendo estadísticas: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo estadísticas",
        )