import os
import json
import tempfile
import logging
from pathlib import Path as PathLib
from datetime import datetime, timezone
from typing import List, Optional

import torch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, Field

from app.modelos.entrenamiento import ModeloIA
from app.servicios.servicio_entrenamiento import servicio_entrenamiento
from app.servicios.dataset_service import dataset_service
from app.servicios.drive_service import eliminar_archivo_de_drive
from ..utilidades.base_datos import obtener_bd
from ..dependencias.permisos import requiere_permiso
from ..modelos.usuario import Usuario
from ..modelos.dataset import CategoriaDataset, VideoDataset
from ..modelos.categoria import Categoria
from ..modelos.tipo_categoria import TipoCategoria
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dataset", tags=["Dataset de Entrenamiento"])


class CategoriaDatasetCrear(BaseModel):
    nombre: str
    descripcion: Optional[str] = None
    tipo_id: Optional[int] = 1
    nivel_requerido: Optional[int] = 1


class VideoDatasetAprobar(BaseModel):
    aprobar: bool
    notas: Optional[str] = None


class ConfiguracionEntrenamiento(BaseModel):
    nombre_modelo: Optional[str] = None
    categoria_ids: List[int] = Field(..., min_items=1)
    epochs: int = Field(default=150, ge=10, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "nombre_modelo": "modelo_senas_v1",
                "categoria_ids": [1, 2],
                "epochs": 150
            }
        }


class AprobacionMasivaRequest(BaseModel):
    video_ids: List[int]
    aprobar: bool = True
    notas: Optional[str] = None


def _eliminar_archivo_video(video: VideoDataset):
    if video.drive_file_id:
        try:
            eliminar_archivo_de_drive(video.drive_file_id)
        except Exception as e:
            logger.warning(f"[Drive] Error eliminando {video.drive_file_id}: {e}")

    if video.ruta_video and os.path.exists(video.ruta_video):
        try:
            os.remove(video.ruta_video)
        except Exception as e:
            logger.warning(f"No se pudo eliminar archivo físico: {e}")


@router.post("/categorias", response_model=RespuestaAPI)
async def crear_categoria_dataset(
    categoria: CategoriaDatasetCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.categorias.gestionar"))
):
    try:
        nombre_normalizado = categoria.nombre.strip().lower()

        categoria_dataset_existente = db.query(CategoriaDataset).filter(
            func.lower(CategoriaDataset.nombre) == nombre_normalizado
        ).first()

        if categoria_dataset_existente:
            total_videos = db.query(VideoDataset).filter(
                VideoDataset.categoria_id == categoria_dataset_existente.id
            ).count()
            total_frames = db.query(func.sum(VideoDataset.frames_extraidos)).filter(
                VideoDataset.categoria_id == categoria_dataset_existente.id
            ).scalar() or 0

            return RespuestaAPI(
                exito=True,
                mensaje=f"La categoría '{categoria.nombre}' ya existe",
                datos={
                    "id": categoria_dataset_existente.id,
                    "categoria_id": categoria_dataset_existente.categoria_id,
                    "nombre": categoria_dataset_existente.nombre,
                    "descripcion": categoria_dataset_existente.descripcion,
                    "total_videos": total_videos,
                    "total_frames": int(total_frames),
                    "activa": categoria_dataset_existente.activa
                }
            )

        categoria_principal = db.query(Categoria).filter(
            func.lower(Categoria.nombre) == nombre_normalizado,
            Categoria.activa == True
        ).first()

        if not categoria_principal:
            tipo_id = categoria.tipo_id or 1
            tipo_categoria = db.query(TipoCategoria).filter(TipoCategoria.id == tipo_id).first()

            if not tipo_categoria:
                tipo_categoria = db.query(TipoCategoria).first()
                if not tipo_categoria:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No existe ningún tipo de categoría en el sistema. Configure los tipos primero."
                    )
                tipo_id = tipo_categoria.id

            max_orden = db.query(func.max(Categoria.orden)).scalar() or 0

            categoria_principal = Categoria(
                nombre=categoria.nombre.strip().title(),
                tipo_id=tipo_id,
                descripcion=categoria.descripcion.strip() if categoria.descripcion else f"Categoría {categoria.nombre}",
                nivel_requerido=categoria.nivel_requerido or 1,
                orden=max_orden + 1,
                activa=True
            )
            db.add(categoria_principal)
            db.flush()

        max_orden_dataset = db.query(func.max(CategoriaDataset.orden)).scalar() or 0

        nueva_categoria_dataset = CategoriaDataset(
            categoria_id=categoria_principal.id,
            nombre=nombre_normalizado,
            descripcion=categoria.descripcion.strip() if categoria.descripcion else f"Dataset para {categoria.nombre}",
            activa=True,
            orden=max_orden_dataset + 1
        )
        db.add(nueva_categoria_dataset)
        db.commit()
        db.refresh(nueva_categoria_dataset)

        return RespuestaAPI(
            exito=True,
            mensaje=f"Categoría '{categoria.nombre}' creada exitosamente",
            datos={
                "id": nueva_categoria_dataset.id,
                "categoria_id": nueva_categoria_dataset.categoria_id,
                "nombre": nueva_categoria_dataset.nombre,
                "descripcion": nueva_categoria_dataset.descripcion,
                "total_videos": 0,
                "total_frames": 0,
                "activa": nueva_categoria_dataset.activa
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creando categoría: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al crear categoría: {e}")


@router.get("/categorias", response_model=RespuestaLista)
async def listar_categorias_dataset(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.ver"))
):
    try:
        categorias = db.query(CategoriaDataset).filter(
            CategoriaDataset.activa == True
        ).order_by(CategoriaDataset.orden).all()

        categorias_data = []
        for cat in categorias:
            if not cat.categoria_rel:
                continue

            total_videos = db.query(VideoDataset).filter(VideoDataset.categoria_id == cat.id).count()
            videos_aprobados = db.query(VideoDataset).filter(
                VideoDataset.categoria_id == cat.id, VideoDataset.aprobado == True
            ).count()
            total_frames = db.query(func.sum(VideoDataset.frames_extraidos)).filter(
                VideoDataset.categoria_id == cat.id
            ).scalar() or 0

            categorias_data.append({
                "id": cat.id,
                "categoria_id": cat.categoria_id,
                "nombre": cat.nombre,
                "descripcion": cat.descripcion,
                "total_videos": total_videos,
                "videos_aprobados": videos_aprobados,
                "total_frames": int(total_frames),
                "activa": cat.activa,
                "fecha_creacion": cat.fecha_creacion.isoformat() if cat.fecha_creacion else None
            })

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(categorias_data)} categorías",
            datos=categorias_data,
            total=len(categorias_data),
            pagina=1,
            por_pagina=len(categorias_data)
        )
    except Exception as e:
        logger.error(f"Error al listar categorías: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al listar categorías: {e}")


@router.post("/videos/subir", response_model=RespuestaAPI)
async def subir_video_dataset(
    archivo: UploadFile = File(...),
    categoria_id: int = Form(...),
    sena: str = Form(...),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar"))
):
    try:
        categoria = db.query(CategoriaDataset).filter(
            CategoriaDataset.id == categoria_id, CategoriaDataset.activa == True
        ).first()

        if not categoria:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Categoría no encontrada o inactiva")

        video = await dataset_service.subir_video_dataset(
            db=db, archivo=archivo, categoria_id=categoria_id, sena=sena, usuario_id=usuario_actual.id
        )

        return RespuestaAPI(
            exito=True,
            mensaje="Video subido exitosamente",
            datos={
                "id": video.id,
                "sena": video.sena,
                "categoria_id": video.categoria_id,
                "ruta": video.ruta_video,
                "duracion": video.duracion_segundos,
                "frames_extraidos": video.frames_extraidos,
                "requiere_aprobacion": not video.aprobado
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al subir video: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al subir video: {e}")


@router.get("/videos/pendientes", response_model=RespuestaLista)
async def listar_videos_pendientes(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar")),
    categoria_id: Optional[int] = None
):
    try:
        query = db.query(VideoDataset).filter(VideoDataset.aprobado == False)
        if categoria_id:
            query = query.filter(VideoDataset.categoria_id == categoria_id)

        videos = query.order_by(VideoDataset.fecha_subida.desc()).all()

        videos_data = [{
            "id": v.id,
            "sena": v.sena,
            "categoria": v.categoria.nombre if v.categoria else "Sin categoría",
            "categoria_id": v.categoria_id,
            "ruta_video": v.ruta_video,
            "drive_url": v.drive_url,
            "drive_file_id": v.drive_file_id,
            "duracion": v.duracion_segundos,
            "frames_extraidos": v.frames_extraidos,
            "fecha_subida": v.fecha_subida.isoformat() if v.fecha_subida else None,
            "subido_por": v.usuario.nombre_completo if v.usuario else "Usuario desconocido"
        } for v in videos]

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(videos_data)} videos pendientes",
            datos=videos_data,
            total=len(videos_data),
            pagina=1,
            por_pagina=len(videos_data)
        )
    except Exception as e:
        logger.error(f"Error al listar videos: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al listar videos: {e}")


@router.put("/videos/{video_id}/aprobar", response_model=RespuestaAPI)
async def aprobar_video_dataset(
    video_id: int,
    datos: VideoDatasetAprobar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar"))
):
    try:
        video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
        if not video:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video no encontrado")

        if datos.aprobar:
            video.aprobado = True
            video.rechazado = False
            video.fecha_aprobado = datetime.now(timezone.utc)
        else:
            video.aprobado = False

        if datos.notas:
            video.notas = datos.notas

        db.commit()
        db.refresh(video)

        estado = "aprobado" if datos.aprobar else "desaprobado"
        logger.info(f"Video {video_id} {estado} por usuario {usuario_actual.id}")

        return RespuestaAPI(
            exito=True,
            mensaje=f"Video {estado} exitosamente",
            datos={
                "id": video.id,
                "sena": video.sena,
                "estado": estado,
                "aprobado": video.aprobado,
                "rechazado": video.rechazado,
                "fecha_aprobado": video.fecha_aprobado.isoformat() if video.fecha_aprobado else None
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al aprobar video: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al aprobar video: {e}")


@router.put("/videos/{video_id}/rechazar", response_model=RespuestaAPI)
async def rechazar_video_dataset(
    video_id: int,
    notas: Optional[str] = None,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar"))
):
    try:
        video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
        if not video:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video no encontrado")

        video.rechazado = True
        video.aprobado = False
        video.fecha_rechazado = datetime.now(timezone.utc)

        if notas:
            video.notas = notas

        db.commit()
        db.refresh(video)

        logger.info(f"Video {video_id} rechazado por usuario {usuario_actual.id}")

        return RespuestaAPI(
            exito=True,
            mensaje="Video rechazado exitosamente",
            datos={
                "id": video.id,
                "sena": video.sena,
                "estado": "rechazado",
                "rechazado": video.rechazado,
                "aprobado": video.aprobado,
                "fecha_rechazado": video.fecha_rechazado.isoformat() if video.fecha_rechazado else None,
                "notas": video.notas
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al rechazar video: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al rechazar video: {e}")


@router.delete("/videos/{video_id}", response_model=RespuestaAPI)
async def eliminar_video_dataset(
    video_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar"))
):
    try:
        video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
        if not video:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video no encontrado")

        sena = video.sena
        _eliminar_archivo_video(video)

        db.delete(video)
        db.commit()

        return RespuestaAPI(exito=True, mensaje="Video eliminado exitosamente", datos={"id": video_id, "sena": sena})
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar video: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al eliminar video: {e}")


@router.delete("/videos/eliminar-todos", response_model=RespuestaAPI)
async def eliminar_todos_videos_pendientes(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar"))
):
    try:
        videos = db.query(VideoDataset).filter(VideoDataset.aprobado == False).all()
        if not videos:
            return RespuestaAPI(exito=True, mensaje="No hay videos pendientes", datos={"total_eliminados": 0})

        videos_eliminados = []
        for video in videos:
            try:
                _eliminar_archivo_video(video)
                videos_eliminados.append({"id": video.id, "sena": video.sena, "ruta_video": video.ruta_video})
                db.delete(video)
            except Exception as e:
                logger.error(f"Error eliminando video {video.id}: {e}")

        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje=f"Se eliminaron {len(videos_eliminados)} videos",
            datos={"total_eliminados": len(videos_eliminados), "videos_eliminados": videos_eliminados}
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar videos: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al eliminar videos: {e}")


@router.get("/videos/categoria/{categoria_id}", response_model=RespuestaLista)
async def listar_videos_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.ver")),
    solo_aprobados: bool = True
):
    try:
        categoria = db.query(CategoriaDataset).filter(CategoriaDataset.id == categoria_id).first()
        if not categoria:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Categoría no encontrada")

        query = db.query(VideoDataset).filter(VideoDataset.categoria_id == categoria_id)
        if solo_aprobados:
            query = query.filter(VideoDataset.aprobado == True)

        videos = query.order_by(VideoDataset.sena, VideoDataset.fecha_subida).all()

        videos_data = [{
            "id": v.id,
            "sena": v.sena,
            "ruta_video": v.ruta_video,
            "drive_url": v.drive_url,
            "drive_file_id": v.drive_file_id,
            "duracion": v.duracion_segundos,
            "frames_extraidos": v.frames_extraidos,
            "aprobado": v.aprobado,
            "fecha_subida": v.fecha_subida.isoformat() if v.fecha_subida else None
        } for v in videos]

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(videos_data)} videos",
            datos=videos_data,
            total=len(videos_data),
            pagina=1,
            por_pagina=len(videos_data)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al listar videos: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al listar videos: {e}")


@router.get("/videos/todos", response_model=RespuestaLista)
async def listar_todos_los_videos(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.ver")),
    categoria_id: Optional[int] = None,
    estado: Optional[str] = None
):
    try:
        query = db.query(VideoDataset).join(CategoriaDataset)

        if categoria_id:
            query = query.filter(VideoDataset.categoria_id == categoria_id)

        if estado:
            if estado == 'aprobado':
                query = query.filter(VideoDataset.aprobado == True)
            elif estado == 'pendiente':
                query = query.filter(VideoDataset.aprobado == False, VideoDataset.rechazado == False)
            elif estado == 'rechazado':
                query = query.filter(VideoDataset.rechazado == True)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Estado inválido: {estado}. Use: 'aprobado', 'pendiente' o 'rechazado'"
                )

        videos = query.order_by(VideoDataset.fecha_subida.desc()).all()

        videos_data = []
        for video in videos:
            if video.aprobado:
                estado_video = 'aprobado'
            elif video.rechazado:
                estado_video = 'rechazado'
            else:
                estado_video = 'pendiente'

            videos_data.append({
                "id": video.id,
                "sena": video.sena,
                "categoria_id": video.categoria_id,
                "categoria_nombre": video.categoria.nombre if video.categoria else None,
                "ruta_video": video.ruta_video,
                "drive_url": video.drive_url,
                "drive_file_id": video.drive_file_id,
                "duracion_segundos": video.duracion_segundos,
                "fps": video.fps,
                "resolucion": video.resolucion,
                "formato": video.formato,
                "tamaño_bytes": video.tamaño_bytes,
                "frames_extraidos": video.frames_extraidos,
                "calidad_promedio": video.calidad_promedio,
                "aprobado": video.aprobado,
                "rechazado": video.rechazado,
                "estado": estado_video,
                "usado_entrenamiento": video.usado_entrenamiento,
                "procesado": video.procesado,
                "fecha_subida": video.fecha_subida.isoformat() if video.fecha_subida else None,
                "fecha_procesado": video.fecha_procesado.isoformat() if video.fecha_procesado else None,
                "fecha_aprobado": video.fecha_aprobado.isoformat() if video.fecha_aprobado else None,
                "fecha_rechazado": video.fecha_rechazado.isoformat() if video.fecha_rechazado else None,
                "usuario_id": video.usuario_id,
                "subido_por": video.usuario.nombre_completo if video.usuario else None,
                "notas": video.notas
            })

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(videos_data)} videos",
            datos=videos_data,
            total=len(videos_data),
            pagina=1,
            por_pagina=len(videos_data)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al listar todos los videos: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al listar videos: {e}")


@router.post("/aprobacion-masiva", response_model=RespuestaAPI)
async def aprobar_videos_masivamente(
    datos: AprobacionMasivaRequest,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar"))
):
    try:
        if not datos.video_ids:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se proporcionaron IDs de videos")

        videos_procesados = 0
        videos_error = []

        for video_id in datos.video_ids:
            try:
                dataset_service.aprobar_video(db=db, video_id=video_id, aprobar=datos.aprobar, notas=datos.notas)
                videos_procesados += 1
            except Exception as e:
                logger.error(f"Error aprobando video {video_id}: {e}")
                videos_error.append({"id": video_id, "error": str(e)})

        estado = "aprobados" if datos.aprobar else "rechazados"

        return RespuestaAPI(
            exito=True,
            mensaje=f"{videos_procesados} videos {estado} correctamente",
            datos={
                "videos_procesados": videos_procesados,
                "videos_error": videos_error,
                "total_solicitados": len(datos.video_ids)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en aprobación masiva: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error en aprobación masiva: {e}")


@router.get("/estadisticas", response_model=RespuestaAPI)
async def obtener_estadisticas_dataset(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.ver"))
):
    try:
        estadisticas = dataset_service.obtener_estadisticas_dataset(db)
        return RespuestaAPI(exito=True, mensaje="Estadísticas obtenidas exitosamente", datos=estadisticas)
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al obtener estadísticas: {e}")


@router.post("/entrenar-modelo", status_code=status.HTTP_202_ACCEPTED, response_model=RespuestaAPI)
async def entrenar_modelo_endpoint(
    configuracion: ConfiguracionEntrenamiento,
    background_tasks: BackgroundTasks,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar"))
):
    try:
        categoria_ids = configuracion.categoria_ids
        epocas = configuracion.epochs
        nombre_modelo = configuracion.nombre_modelo or servicio_entrenamiento.generar_nombre_modelo()

        modelo_existente = db.query(ModeloIA).filter(
            ModeloIA.nombre == nombre_modelo, ModeloIA.activo == True
        ).first()
        if modelo_existente:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ya existe un modelo activo con el nombre '{nombre_modelo}'"
            )

        for cat_id in categoria_ids:
            categoria = db.query(CategoriaDataset).filter(
                CategoriaDataset.id == cat_id, CategoriaDataset.activa == True
            ).first()
            if not categoria:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Categoría {cat_id} no encontrada o inactiva"
                )

        validacion = servicio_entrenamiento.validar_dataset_entrenamiento(categoria_ids, db)

        if not validacion["valido"]:
            error_msg = "Dataset no válido: "
            if not validacion["cumple_requisitos"]["videos_totales"]:
                error_msg += f"{validacion['total_videos']} videos (mín. 20). "
            if not validacion["cumple_requisitos"]["senas_suficientes"]:
                error_msg += f"{len(validacion['senas_con_minimo'])} señas (mín. 2). "
            if validacion.get("senas_sin_minimo"):
                error_msg += f"Señas insuficientes: [{', '.join(validacion['senas_sin_minimo'])}]. "
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg.strip())

        logger.info(f"Entrenamiento iniciado: {nombre_modelo}, {validacion['total_videos']} videos, {epocas} épocas")

        def tarea_entrenamiento():
            try:
                servicio_entrenamiento.entrenar(nombre_modelo, categoria_ids, epocas)
            except Exception as e:
                logger.error(f"Error en entrenamiento {nombre_modelo}: {e}", exc_info=True)

        background_tasks.add_task(tarea_entrenamiento)

        return RespuestaAPI(
            exito=True,
            mensaje=f"Entrenamiento iniciado: {nombre_modelo}",
            datos={
                "nombre_modelo": nombre_modelo,
                "videos_a_procesar": validacion["total_videos"],
                "estado": "en_procesamiento",
                "epocas": epocas,
                "categorias": categoria_ids,
                "senas": validacion["senas_con_minimo"],
                "endpoint_progreso": f"/api/v1/dataset/entrenamiento/progreso/{nombre_modelo}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al iniciar entrenamiento: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al iniciar entrenamiento: {e}")


@router.get("/entrenamiento/progreso/{nombre_modelo}", response_model=RespuestaAPI)
async def obtener_progreso_entrenamiento_endpoint(
    nombre_modelo: str,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar"))
):
    try:
        if not nombre_modelo or not nombre_modelo.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nombre de modelo no válido")

        progreso = servicio_entrenamiento.obtener_progreso_entrenamiento(nombre_modelo)
        return RespuestaAPI(exito=True, mensaje="Progreso obtenido exitosamente", datos=progreso)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo progreso de {nombre_modelo}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error obteniendo progreso: {e}")


def _detectar_clases_y_metricas(estado, clases_manual: Optional[str]):
    accuracy = 0.0
    loss = 0.0

    if isinstance(estado, dict):
        for clave in ('accuracy', 'val_accuracy', 'test_accuracy', 'best_accuracy', 'top1_acc'):
            if clave in estado:
                accuracy = round(float(estado[clave]), 4)
                break
        for clave in ('loss', 'val_loss', 'test_loss'):
            if clave in estado:
                loss = round(float(estado[clave]), 4)
                break

    clases_detectadas = None
    num_clases_detectadas = None

    if isinstance(estado, dict):
        if 'clases' in estado:
            clases_detectadas = estado['clases']
        elif 'class_names' in estado:
            clases_detectadas = estado['class_names']
        elif 'idx_to_class' in estado:
            clases_detectadas = list(estado['idx_to_class'].values())
        elif 'num_classes' in estado:
            num_clases_detectadas = estado['num_classes']

        if clases_detectadas is None and num_clases_detectadas is None:
            model_state = estado.get('model_state_dict', estado)
            for key in reversed(list(model_state.keys())):
                if any(term in key.lower() for term in ('fc.weight', 'classifier.weight', 'head.weight', 'out.weight')):
                    num_clases_detectadas = model_state[key].shape[0]
                    break

    if clases_manual and clases_manual.strip():
        try:
            lista_manual = json.loads(clases_manual)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='El formato de clases debe ser un JSON válido. Ejemplo: ["hola", "gracias", "adios"]'
            )

        esperadas = len(clases_detectadas) if clases_detectadas else num_clases_detectadas
        if esperadas and len(lista_manual) != esperadas:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El modelo tiene {esperadas} clases, pero proporcionaste {len(lista_manual)}"
            )
        lista_clases = lista_manual
    elif clases_detectadas:
        lista_clases = clases_detectadas
    elif num_clases_detectadas:
        lista_clases = [f"clase_{i}" for i in range(num_clases_detectadas)]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='No se pudieron detectar las clases automáticamente. Proporciónalas en formato JSON. '
                   'Ejemplo: ["hola", "gracias", "adios"]'
        )

    if not isinstance(lista_clases, list) or len(lista_clases) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Debe haber al menos 2 clases")

    return lista_clases, accuracy, loss


@router.post("/cargar-modelo", response_model=RespuestaAPI)
async def cargar_modelo_preentrenado(
    archivo: UploadFile = File(...),
    nombre_modelo: str = Form(...),
    descripcion: Optional[str] = Form(None),
    clases: Optional[str] = Form(None),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar"))
):
    ruta_modelo = None
    try:
        extension = archivo.filename.split('.')[-1].lower()
        if extension not in ('pth', 'pt'):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo debe ser .pth o .pt")

        if not nombre_modelo or len(nombre_modelo.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre del modelo debe tener al menos 3 caracteres"
            )

        if db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ya existe un modelo con el nombre '{nombre_modelo}'"
            )

        modelos_dir = PathLib("modelos_entrenados")
        modelos_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta_modelo = modelos_dir / f"{nombre_modelo}_{timestamp}.{extension}"

        contenido = await archivo.read()
        if len(contenido) < 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo del modelo es demasiado pequeño o está corrupto"
            )

        with open(ruta_modelo, 'wb') as f:
            f.write(contenido)

        try:
            estado = torch.load(str(ruta_modelo), map_location='cpu', weights_only=False)
            lista_clases, accuracy_detectado, loss_detectado = _detectar_clases_y_metricas(estado, clases)
        except HTTPException:
            os.remove(ruta_modelo)
            raise
        except json.JSONDecodeError:
            os.remove(ruta_modelo)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='El formato de clases debe ser un JSON válido. Ejemplo: ["hola", "gracias", "adios"]'
            )
        except Exception as e:
            os.remove(ruta_modelo)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error al procesar modelo: {e}")

        nuevo_modelo = ModeloIA(
            nombre=nombre_modelo,
            descripcion=descripcion or f"Modelo pre-entrenado cargado el {datetime.now().strftime('%Y-%m-%d')}",
            ruta_archivo=str(ruta_modelo),
            tipo_modelo="video_cnn",
            num_clases=len(lista_clases),
            clases_json=json.dumps(lista_clases, ensure_ascii=False),
            accuracy=accuracy_detectado,
            loss=loss_detectado,
            activo=False,
            fecha_creacion=datetime.now(),
            fecha_entrenamiento=datetime.now(),
            epocas_entrenamiento=0,
            total_imagenes=0,
            tamaño_mb=round(len(contenido) / (1024 * 1024), 2),
            origen="cargado_manualmente"
        )
        db.add(nuevo_modelo)
        db.commit()
        db.refresh(nuevo_modelo)

        mensaje = f"Modelo '{nombre_modelo}' cargado exitosamente con {len(lista_clases)} clases"
        if accuracy_detectado > 0:
            mensaje += f" y accuracy de {accuracy_detectado:.2%}"

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos={
                "id": nuevo_modelo.id,
                "nombre": nuevo_modelo.nombre,
                "ruta": str(ruta_modelo),
                "num_clases": nuevo_modelo.num_clases,
                "clases": lista_clases,
                "accuracy": nuevo_modelo.accuracy,
                "loss": nuevo_modelo.loss,
                "tamaño_mb": nuevo_modelo.tamaño_mb,
                "activo": nuevo_modelo.activo,
                "fecha_carga": nuevo_modelo.fecha_creacion.isoformat(),
                "origen": nuevo_modelo.origen
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error cargando modelo: {e}", exc_info=True)
        if ruta_modelo and os.path.exists(ruta_modelo):
            try:
                os.remove(ruta_modelo)
            except Exception as cleanup_error:
                logger.error(f"Error al limpiar archivo: {cleanup_error}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al cargar modelo: {e}")


@router.post("/validar-modelo", response_model=RespuestaAPI)
async def validar_modelo_preentrenado(
    archivo: UploadFile = File(...),
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar"))
):
    temp_path = None
    try:
        extension = archivo.filename.split('.')[-1].lower()
        if extension not in ('pth', 'pt'):
            return RespuestaAPI(
                exito=False, mensaje="Extensión inválida. Use .pth o .pt",
                datos={"valido": False, "error": "extension_invalida"}
            )

        contenido = await archivo.read()
        if len(contenido) < 1024:
            return RespuestaAPI(
                exito=False, mensaje="Archivo demasiado pequeño (menos de 1KB)",
                datos={"valido": False, "error": "archivo_pequeno"}
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as temp_file:
            temp_file.write(contenido)
            temp_path = temp_file.name

        try:
            estado = torch.load(temp_path, map_location='cpu', weights_only=False)

            info = {
                "valido": True,
                "tipo": type(estado).__name__,
                "tamaño_mb": round(len(contenido) / (1024 * 1024), 2),
                "nombre_archivo": archivo.filename
            }

            if isinstance(estado, dict):
                info["keys"] = list(estado.keys())[:20]

                if 'model_state_dict' in estado:
                    info["estructura"] = "checkpoint_completo"
                    state_dict = estado['model_state_dict']
                elif any(k.startswith(('conv', 'fc', 'layer')) for k in estado.keys()):
                    info["estructura"] = "state_dict"
                    state_dict = estado
                else:
                    info["estructura"] = "formato_personalizado"
                    state_dict = estado

                info["num_parametros"] = len(state_dict)

                if 'clases' in estado:
                    info["clases_detectadas"] = estado['clases']
                    info["num_clases"] = len(estado['clases'])
                elif 'class_names' in estado:
                    info["clases_detectadas"] = estado['class_names']
                    info["num_clases"] = len(estado['class_names'])
                elif 'idx_to_class' in estado:
                    info["clases_detectadas"] = list(estado['idx_to_class'].values())
                    info["num_clases"] = len(info["clases_detectadas"])
                elif 'num_classes' in estado:
                    info["num_clases"] = estado['num_classes']
                    info["clases_detectadas"] = None
                else:
                    info["clases_detectadas"] = None
                    for key in reversed(list(state_dict.keys())):
                        if any(term in key.lower() for term in ('fc.weight', 'classifier.weight', 'head.weight', 'out.weight')):
                            info["num_clases"] = state_dict[key].shape[0]
                            info["capa_salida"] = key
                            break

                if 'epoch' in estado:
                    info["epochs_entrenadas"] = estado['epoch']
                if 'accuracy' in estado:
                    info["accuracy"] = round(float(estado['accuracy']), 4)
                if 'loss' in estado:
                    info["loss"] = round(float(estado['loss']), 4)

                for key in list(state_dict.keys())[:10]:
                    key_lower = key.lower()
                    if 'lstm' in key_lower:
                        info["arquitectura_detectada"] = "LSTM"
                        break
                    if 'gru' in key_lower:
                        info["arquitectura_detectada"] = "GRU"
                        break
                    if 'conv3d' in key_lower:
                        info["arquitectura_detectada"] = "Conv3D (Video)"
                        break
                    if 'conv' in key_lower:
                        info["arquitectura_detectada"] = "CNN"
                        break
            else:
                info["estructura"] = "modelo_directo"
                info["arquitectura_detectada"] = type(estado).__name__

            mensaje = "Modelo válido y compatible"
            if info.get("num_clases"):
                mensaje += f" - {info['num_clases']} clases detectadas"

            return RespuestaAPI(exito=True, mensaje=mensaje, datos=info)
        except Exception as e:
            logger.error(f"Error al validar modelo: {e}", exc_info=True)
            return RespuestaAPI(
                exito=False, mensaje=f"Modelo inválido o corrupto: {e}",
                datos={"valido": False, "error": str(e), "tipo_error": type(e).__name__}
            )
    except Exception as e:
        logger.error(f"Error en validación: {e}", exc_info=True)
        return RespuestaAPI(exito=False, mensaje=f"Error en validación: {e}", datos={"valido": False, "error": str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logger.error(f"Error al limpiar archivo temporal: {cleanup_error}")