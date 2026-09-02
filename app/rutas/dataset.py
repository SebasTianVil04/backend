import asyncio
from ..servicios.dataset_service import dataset_service
from ..servicios.drive_service import eliminar_archivo_de_drive
from datetime import datetime, timezone
import logging
import tempfile
import threading
import traceback
import torch
from pathlib import Path as PathLib
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel, Field
import os
import json

from app.modelos.entrenamiento import ModeloIA
from ..utilidades.base_datos import obtener_bd, SessionLocal
from ..utilidades.seguridad import verificar_admin, obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.dataset import CategoriaDataset, VideoDataset
from ..modelos.categoria import Categoria
from ..modelos.tipo_categoria import TipoCategoria
from ..servicios.dataset_service import dataset_service
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista
from app.servicios.entrenamiento_modelo import (
    entrenamiento_service, 
    progresos_entrenamiento
)
from app.servicios.entrenamiento_adaptativo import entrenamiento_adaptativo_service, progresos_entrenamiento


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
    categoria_ids: Optional[List[int]] = None
    epochs: int = 50

class ConfiguracionEntrenamientoAdaptativo(BaseModel):
    """Modelo para configuración de entrenamiento adaptativo"""
    nombre_modelo: Optional[str] = None
    categoria_ids: List[int] = Field(..., min_items=1)
    epochs: int = Field(default=50, ge=10, le=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "nombre_modelo": "modelo_senas_v1",
                "categoria_ids": [1, 2],
                "epochs": 100
            }
        }


class RespuestaEntrenamiento(BaseModel):
    """Respuesta al iniciar entrenamiento"""
    nombre_modelo: str
    epocas: int
    total_videos: int
    mensaje: str



class AprobacionMasivaRequest(BaseModel):
    video_ids: List[int]
    aprobar: bool = True
    notas: Optional[str] = None


@router.post("/categorias", response_model=RespuestaAPI)
async def crear_categoria_dataset(
    categoria: CategoriaDatasetCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        nombre_normalizado = categoria.nombre.strip().lower()
        
        categoria_dataset_existente = db.query(CategoriaDataset).filter(
            func.lower(CategoriaDataset.nombre) == nombre_normalizado
        ).first()
        
        if categoria_dataset_existente:
            logger.info(f"CategoriaDataset '{categoria.nombre}' ya existe con ID {categoria_dataset_existente.id}")
            
            total_videos = db.query(VideoDataset).filter(
                VideoDataset.categoria_id == categoria_dataset_existente.id
            ).count()
            
            total_frames = db.query(
                func.sum(VideoDataset.frames_extraidos)
            ).filter(
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
            logger.info(f"Creando nueva Categoria principal: {categoria.nombre}")
            
            tipo_id = categoria.tipo_id or 1
            tipo_categoria = db.query(TipoCategoria).filter(
                TipoCategoria.id == tipo_id
            ).first()
            
            if not tipo_categoria:
                logger.warning(f"TipoCategoria {tipo_id} no encontrado, buscando tipo por defecto")
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
            
            logger.info(f"Categoria principal creada con ID {categoria_principal.id}")
        
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
        
        logger.info(f"CategoriaDataset creada: ID={nueva_categoria_dataset.id}, categoria_id={categoria_principal.id}")
        
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
        logger.error(f"Error creando categoría: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear categoría: {str(e)}"
        )


@router.get("/categorias", response_model=RespuestaLista)
async def listar_categorias_dataset(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        categorias = db.query(CategoriaDataset).filter(
            CategoriaDataset.activa == True
        ).order_by(CategoriaDataset.orden).all()
        
        categorias_data = []
        for cat in categorias:
            if not cat.categoria_rel:
                logger.warning(f"CategoriaDataset {cat.id} sin categoría principal vinculada")
                continue
            
            total_videos = db.query(VideoDataset).filter(
                VideoDataset.categoria_id == cat.id
            ).count()
            
            videos_aprobados = db.query(VideoDataset).filter(
                VideoDataset.categoria_id == cat.id,
                VideoDataset.aprobado == True
            ).count()
            
            total_frames = db.query(
                func.sum(VideoDataset.frames_extraidos)
            ).filter(
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
        logger.error(f"Error al listar categorías: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar categorías: {str(e)}"
        )


@router.post("/videos/subir", response_model=RespuestaAPI)
async def subir_video_dataset(
    archivo: UploadFile = File(...),
    categoria_id: int = Form(...),
    sena: str = Form(...),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        categoria = db.query(CategoriaDataset).filter(
            CategoriaDataset.id == categoria_id,
            CategoriaDataset.activa == True
        ).first()
        
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Categoría no encontrada o inactiva"
            )
        
        video = await dataset_service.subir_video_dataset(
            db=db,
            archivo=archivo,
            categoria_id=categoria_id,
            sena=sena,
            usuario_id=usuario_actual.id
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
        logger.error(f"Error al subir video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al subir video: {str(e)}"
        )


@router.get("/videos/pendientes", response_model=RespuestaLista)
async def listar_videos_pendientes(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin),
    categoria_id: Optional[int] = None
):
    try:
        query = db.query(VideoDataset).filter(VideoDataset.aprobado == False)
        
        if categoria_id:
            query = query.filter(VideoDataset.categoria_id == categoria_id)
        
        videos = query.order_by(VideoDataset.fecha_subida.desc()).all()
        
        videos_data = []
        for video in videos:
            videos_data.append({
                "id": video.id,
                "sena": video.sena,
                "categoria": video.categoria.nombre if video.categoria else "Sin categoría",
                "categoria_id": video.categoria_id,
                "ruta_video": video.ruta_video,
                "drive_url": video.drive_url,
                "drive_file_id": video.drive_file_id,
                "duracion": video.duracion_segundos,
                "frames_extraidos": video.frames_extraidos,
                "fecha_subida": video.fecha_subida.isoformat() if video.fecha_subida else None,
                "subido_por": video.usuario.nombre_completo if video.usuario else "Usuario desconocido"
            })
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(videos_data)} videos pendientes",
            datos=videos_data,
            total=len(videos_data),
            pagina=1,
            por_pagina=len(videos_data)
        )
        
    except Exception as e:
        logger.error(f"Error al listar videos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar videos: {str(e)}"
        )


@router.put("/videos/{video_id}/aprobar", response_model=RespuestaAPI)
async def aprobar_video_dataset(
    video_id: int,
    datos: VideoDatasetAprobar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        # Buscar el video directamente
        video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
        
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video no encontrado"
            )
        
        # SOLUCIÓN: Actualizar ambos campos correctamente
        if datos.aprobar:
            video.aprobado = True
            # Usar setattr para evitar errores si el campo no existe
            if hasattr(video, 'rechazado'):
                video.rechazado = False  # ← LIMPIA EL RECHAZO
            if hasattr(video, 'fecha_aprobado'):
                video.fecha_aprobado = datetime.now(timezone.utc)
        else:
            video.aprobado = False
            # No cambiar rechazado aquí, solo desaprobar
        
        # Actualizar notas si se proporcionaron
        if datos.notas:
            video.notas = datos.notas
        
        # IMPORTANTE: Guardar cambios en la base de datos
        db.commit()
        db.refresh(video)
        
        logger.info(f"Video {video_id} actualizado: aprobado={video.aprobado}, rechazado={video.rechazado}")
        
        # Verificar que se guardó correctamente
        video_verificado = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
        logger.info(f"Verificación post-commit: aprobado={video_verificado.aprobado}, rechazado={video_verificado.rechazado}")
        
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
                "rechazado": getattr(video, 'rechazado', False),
                "tiene_campo_rechazado": hasattr(video, 'rechazado'),
                "fecha_aprobado": video.fecha_aprobado.isoformat() if hasattr(video, 'fecha_aprobado') and video.fecha_aprobado else None
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al aprobar video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al aprobar video: {str(e)}"
        )

@router.delete("/videos/{video_id}", response_model=RespuestaAPI)
async def eliminar_video_dataset(
    video_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
        
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video no encontrado"
            )
        
        ruta_video = video.ruta_video
        drive_file_id = video.drive_file_id
        sena = video.sena

        # Eliminar de Drive si existe ahí
        if drive_file_id:
            try:
                eliminado_drive = eliminar_archivo_de_drive(drive_file_id)
                if eliminado_drive:
                    logger.info(f"[Drive] Archivo eliminado: {drive_file_id}")
                else:
                    logger.warning(f"[Drive] No se pudo confirmar eliminación: {drive_file_id}")
            except Exception as e:
                logger.warning(f"[Drive] Error eliminando archivo {drive_file_id}: {str(e)}")

        # Eliminar copia local si aún existe (caso de respaldo por fallo de subida)
        if ruta_video and os.path.exists(ruta_video):
            try:
                os.remove(ruta_video)
                logger.info(f"Archivo físico eliminado: {ruta_video}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo físico: {str(e)}")
        
        db.delete(video)
        db.commit()
        
        return RespuestaAPI(
            exito=True,
            mensaje="Video eliminado exitosamente",
            datos={
                "id": video_id,
                "sena": sena
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar video: {str(e)}"
        )

@router.get("/estadisticas", response_model=RespuestaAPI)
async def obtener_estadisticas_dataset(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        estadisticas = dataset_service.obtener_estadisticas_dataset(db)
        
        return RespuestaAPI(
            exito=True,
            mensaje="Estadísticas obtenidas exitosamente",
            datos=estadisticas
        )
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )


@router.get("/videos/categoria/{categoria_id}", response_model=RespuestaLista)
async def listar_videos_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    solo_aprobados: bool = True
):
    try:
        categoria = db.query(CategoriaDataset).filter(
            CategoriaDataset.id == categoria_id
        ).first()
        
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Categoría no encontrada"
            )
        
        query = db.query(VideoDataset).filter(
            VideoDataset.categoria_id == categoria_id
        )
        
        if solo_aprobados:
            query = query.filter(VideoDataset.aprobado == True)
        
        videos = query.order_by(VideoDataset.sena, VideoDataset.fecha_subida).all()
        
        videos_data = []
        for video in videos:
            videos_data.append({
                "id": video.id,
                "sena": video.sena,
                "ruta_video": video.ruta_video,
                "drive_url": video.drive_url,
                "drive_file_id": video.drive_file_id,
                "duracion": video.duracion_segundos,
                "frames_extraidos": video.frames_extraidos,
                "aprobado": video.aprobado,
                "fecha_subida": video.fecha_subida.isoformat() if video.fecha_subida else None
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
        logger.error(f"Error al listar videos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar videos: {str(e)}"
        )


@router.post("/entrenar-modelo-video", response_model=RespuestaAPI)
async def entrenar_modelo_video_endpoint(
    configuracion: ConfiguracionEntrenamiento,
    background_tasks: BackgroundTasks,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    db_local = None
    
    try:
        categoria_ids = configuracion.categoria_ids or []
        epocas = configuracion.epochs or 50
        
        if epocas < 10:
            epocas = 10
        elif epocas > 500:
            epocas = 500
        
        nombre_modelo = (
            configuracion.nombre_modelo 
            or f"modelo_video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info(f"[POST /entrenar-modelo-video] Solicitud:")
        logger.info(f"  Nombre: {nombre_modelo}")
        logger.info(f"  Categorías: {categoria_ids}")
        logger.info(f"  Épocas: {epocas}")
        
        if not categoria_ids or len(categoria_ids) == 0:
            logger.warning("[POST] No hay categorías seleccionadas")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Debe seleccionar al menos una categoría"
            )
        
        for cat_id in categoria_ids:
            categoria = db.query(CategoriaDataset).filter(
                CategoriaDataset.id == cat_id,
                CategoriaDataset.activa == True
            ).first()
            
            if not categoria:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Categoría {cat_id} no encontrada o inactiva"
                )
        
        logger.info(f"[POST] Validando dataset...")
        validacion = entrenamiento_service.validar_dataset_entrenamiento(categoria_ids, db)
        
        logger.info(f"[POST] Resultado validación: {json.dumps(validacion, indent=2, default=str)}")
        
        if not validacion["valido"]:
            error_msg = "Dataset no válido: "
            
            if not validacion["cumple_requisitos"]["videos_totales"]:
                error_msg += f"{validacion['total_videos']} videos (mín. 10). "
            
            if not validacion["cumple_requisitos"]["senas_suficientes"]:
                error_msg += f"{len(validacion['senas_con_minimo'])} señas (mín. 2). "
            
            if validacion.get("senas_sin_minimo"):
                senas_insuficientes = ", ".join(validacion["senas_sin_minimo"])
                error_msg += f"Señas insuficientes: [{senas_insuficientes}]. "
            
            logger.warning(f"[POST] {error_msg}")
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg.strip()
            )
        
        logger.info(f"[POST] Validación exitosa")
        logger.info(f"[POST] Total videos: {validacion['total_videos']}")
        logger.info(f"[POST] Señas válidas: {validacion['senas_con_minimo']}")
        
        logger.info(f"[POST] Inicializando progreso en diccionario global...")
        
        progresos_entrenamiento[nombre_modelo] = {
            "nombre_modelo": nombre_modelo,
            "estado": "iniciando",
            "progreso": 0.0,
            "epoch_actual": 0,
            "total_epochs": epocas,
            "accuracy": 0.0,
            "loss": 0.0,
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "num_clases": 0,
            "clases": [],
            "total_videos": 0,
            "frames_procesados": 0,
            "total_frames": 0,
            "mensaje": "Preparando entrenamiento...",
            "fecha_inicio": datetime.datetime.now().isoformat(),
            "entrenando": True,
            "categoria_ids": categoria_ids
        }
        
        logger.info(f"[POST] Progreso inicializado: {progresos_entrenamiento[nombre_modelo]}")
        
        def tarea_entrenamiento_background():
            db_tarea = None
            
            try:
                logger.info(f"[BACKGROUND] Iniciando tarea para: {nombre_modelo}")
                
                db_tarea = SessionLocal()
                
                if torch.cuda.is_available():
                    logger.info("[BACKGROUND] CUDA disponible, configurando...")
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                else:
                    logger.info("[BACKGROUND] CUDA no disponible, usando CPU")
                
                progresos_entrenamiento[nombre_modelo]["estado"] = "cargando_datos"
                progresos_entrenamiento[nombre_modelo]["mensaje"] = "Cargando videos..."
                logger.info(f"[BACKGROUND] Estado: cargando_datos")
                
                logger.info(f"[BACKGROUND] Llamando a entrenamiento_service.entrenar()...")
                
                entrenamiento_service.entrenar(nombre_modelo, categoria_ids, epocas)
                
                logger.info(f"[BACKGROUND] Entrenamiento completado exitosamente")
                
                progresos_entrenamiento[nombre_modelo]["estado"] = "completado"
                progresos_entrenamiento[nombre_modelo]["progreso"] = 100.0
                progresos_entrenamiento[nombre_modelo]["entrenando"] = False
                progresos_entrenamiento[nombre_modelo]["mensaje"] = "Entrenamiento completado exitosamente"
                
                logger.info(f"[BACKGROUND] Estado final: completado")
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"[BACKGROUND] Error durante entrenamiento: {error_str}")
                logger.error(f"[BACKGROUND] Traceback: {traceback.format_exc()}")
                
                progresos_entrenamiento[nombre_modelo]["estado"] = "error"
                progresos_entrenamiento[nombre_modelo]["entrenando"] = False
                progresos_entrenamiento[nombre_modelo]["mensaje"] = f"Error: {error_str}"
                progresos_entrenamiento[nombre_modelo]["progreso"] = 0.0
                
                logger.error(f"[BACKGROUND] Estado actualizado a: error")
            
            finally:
                try:
                    if db_tarea:
                        db_tarea.close()
                        logger.info("[BACKGROUND] Conexión BD cerrada")
                except Exception as e:
                    logger.error(f"[BACKGROUND] Error cerrando BD: {str(e)}")
        
        logger.info(f"[POST] Agregando tarea a background_tasks...")
        background_tasks.add_task(tarea_entrenamiento_background)
        logger.info(f"[POST] Tarea agregada a queue")
        
        respuesta = {
            "exito": True,
            "mensaje": f"Entrenamiento iniciado: {nombre_modelo}",
            "datos": {
                "nombre_modelo": nombre_modelo,
                "videos_a_procesar": validacion["total_videos"],
                "estado": "en_procesamiento",
                "epocas": epocas,
                "categorias": categoria_ids,
                "senas": validacion["senas_con_minimo"],
                "endpoint_progreso": f"/api/v1/dataset/entrenamiento/progreso/{nombre_modelo}"
            }
        }
        
        logger.info(f"[POST] Respuesta: {json.dumps(respuesta, indent=2, default=str)}")
        logger.info(f"[POST] Endpoint completado exitosamente")
        
        return RespuestaAPI(**respuesta)
        
    except HTTPException:
        logger.error(f"[POST] HTTPException levantada")
        raise
    
    except Exception as e:
        error_str = str(e)
        logger.error(f"[POST] Excepción no manejada: {error_str}")
        logger.error(f"[POST] Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al iniciar entrenamiento: {error_str}"
        )
    
    finally:
        try:
            if db_local:
                db_local.close()
        except:
            pass


@router.post("/entrenar-modelo-adaptativo", status_code=status.HTTP_202_ACCEPTED)
async def entrenar_modelo_adaptativo(
    config: ConfiguracionEntrenamientoAdaptativo,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        logger.info(f"Solicitud de entrenamiento adaptativo de usuario {usuario_actual.id}")
        logger.info(f"Usuario: {usuario_actual.email} (Admin: {usuario_actual.es_admin})")
        logger.info(f"Configuracion: {config.dict()}")
        
        if not entrenamiento_adaptativo_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servicio de entrenamiento no disponible"
            )
        
        if not config.categoria_ids or len(config.categoria_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Debe seleccionar al menos una categoria"
            )
        
        if config.epochs < 10 or config.epochs > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Las epocas deben estar entre 10 y 500"
            )
        
        validacion = entrenamiento_adaptativo_service.validar_dataset_entrenamiento(
            config.categoria_ids, 
            db
        )
        
        if not validacion.get("valido", False):
            error_msg = "Dataset insuficiente para entrenamiento:\n"
            
            if validacion.get("total_videos", 0) < 20:
                error_msg += f"- Videos totales: {validacion.get('total_videos', 0)} (minimo 20)\n"
            
            senas_insuficientes = validacion.get("senas_sin_minimo", [])
            if senas_insuficientes:
                error_msg += f"- Senas con pocos videos: {', '.join(senas_insuficientes)}\n"
            
            senas_validas = validacion.get("senas_con_minimo", [])
            if len(senas_validas) < 2:
                error_msg += f"- Senas validas: {len(senas_validas)} (minimo 2)\n"
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg.strip()
            )
        
        nombre_modelo = config.nombre_modelo or entrenamiento_adaptativo_service.generar_nombre_modelo()
        
        modelo_existente = db.query(ModeloIA).filter(
            ModeloIA.nombre == nombre_modelo,
            ModeloIA.activo == True
        ).first()
        
        if modelo_existente:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ya existe un modelo activo con el nombre '{nombre_modelo}'"
            )
        
        total_videos = validacion.get("total_videos", 0)
        senas_validas = validacion.get("senas_con_minimo", [])
        
        logger.info(f"Iniciando entrenamiento adaptativo: {nombre_modelo}")
        logger.info(f"Dataset: {total_videos} videos, {len(senas_validas)} senas validas")
        logger.info(f"Epocas configuradas: {config.epochs}")
        
        from app.utilidades.base_datos import SessionLocal
        
        def entrenar_async():
            db_thread = SessionLocal()
            try:
                logger.info(f"[Thread] Iniciando entrenamiento: {nombre_modelo}")
                logger.info(f"[Thread] Usuario: {usuario_actual.id} - {usuario_actual.email}")
                
                entrenamiento_adaptativo_service.entrenar(
                    nombre_modelo=nombre_modelo,
                    categoria_ids=config.categoria_ids,
                    epochs=config.epochs
                )
                
                logger.info(f"[Thread] Entrenamiento completado: {nombre_modelo}")
                
            except Exception as e:
                logger.error(f"[Thread] Error en entrenamiento: {str(e)}", exc_info=True)
                
            finally:
                db_thread.close()
        
        thread = threading.Thread(target=entrenar_async, daemon=True)
        thread.start()
        
        logger.info(f"Thread de entrenamiento iniciado para: {nombre_modelo}")
        
        await asyncio.sleep(0.5)
        
        return {
            "exito": True,
            "mensaje": f"Entrenamiento iniciado exitosamente: {nombre_modelo}",
            "datos": {
                "nombre_modelo": nombre_modelo,
                "epocas": config.epochs,
                "total_videos": total_videos,
                "mensaje": f"Entrenamiento en progreso con {total_videos} videos de {len(senas_validas)} senas"
            }
        }
        
    except HTTPException as he:
        logger.warning(f"Error HTTP en entrenamiento: {he.detail}")
        raise
    
    except Exception as e:
        logger.error(f"Error inesperado en entrenamiento adaptativo: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error iniciando entrenamiento: {str(e)}"
        )


@router.get("/entrenamiento/progreso/{nombre_modelo}")
async def obtener_progreso_entrenamiento_endpoint(
    nombre_modelo: str,
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        if not nombre_modelo or nombre_modelo.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nombre de modelo no valido"
            )
        
        logger.debug(f"Consultando progreso de {nombre_modelo} (Usuario: {usuario_actual.id})")
        
        progreso = entrenamiento_adaptativo_service.obtener_progreso_entrenamiento(nombre_modelo)
        
        return {
            "exito": True,
            "mensaje": "Progreso obtenido exitosamente",
            "datos": progreso
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo progreso de {nombre_modelo}: {str(e)}")
        
        return {
            "exito": False,
            "mensaje": f"Error obteniendo progreso: {str(e)}",
            "datos": {
                "nombre_modelo": nombre_modelo,
                "estado": "error",
                "progreso": 0.0,
                "accuracy": 0.0,
                "loss": 0.0,
                "train_accuracy": 0.0,
                "train_loss": 0.0,
                "epoch_actual": 0,
                "total_epochs": 0,
                "num_clases": 0,
                "clases": [],
                "total_videos": 0,
                "frames_procesados": 0,
                "total_frames": 0,
                "mensaje": str(e),
                "entrenando": False
            }
        }
    
@router.get("/entrenamiento/progreso/{modelo_nombre}", response_model=RespuestaAPI)
async def obtener_progreso_entrenamiento(
    modelo_nombre: str,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        logger.info(f"[GET /progreso] Consultando: {modelo_nombre}")
        logger.info(f"[GET /progreso] Modelos activos: {list(progresos_entrenamiento.keys())}")
        
        progreso = entrenamiento_service.obtener_progreso_entrenamiento(modelo_nombre)
        
        logger.info(f"[GET /progreso] Progreso encontrado: estado={progreso.get('estado')}")
        
        return RespuestaAPI(
            exito=True,
            mensaje="Progreso obtenido correctamente",
            datos=progreso
        )
    
    except Exception as e:
        error_str = str(e)
        logger.error(f"[GET /progreso] Error: {error_str}")
        logger.error(f"[GET /progreso] Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener progreso: {error_str}"
        )


@router.post("/aprobacion-masiva", response_model=RespuestaAPI)
async def aprobar_videos_masivamente(
    datos: AprobacionMasivaRequest,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        if not datos.video_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se proporcionaron IDs de videos"
            )
        
        videos_procesados = 0
        videos_error = []
        
        for video_id in datos.video_ids:
            try:
                dataset_service.aprobar_video(
                    db=db,
                    video_id=video_id,
                    aprobar=datos.aprobar,
                    notas=datos.notas
                )
                videos_procesados += 1
            except Exception as e:
                logger.error(f"Error aprobando video {video_id}: {str(e)}")
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
        logger.error(f"Error en aprobación masiva: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en aprobación masiva: {str(e)}"
        ) 

@router.delete("/videos/eliminar-todos", response_model=RespuestaAPI)
async def eliminar_todos_videos_pendientes(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        videos = db.query(VideoDataset).filter(VideoDataset.aprobado == False).all()
        
        if not videos:
            return RespuestaAPI(
                exito=True,
                mensaje="No hay videos pendientes",
                datos={"total_eliminados": 0}
            )
        
        total_eliminados = 0
        videos_eliminados = []
        
        for video in videos:
            try:
                # Eliminar de Drive si existe ahí
                if video.drive_file_id:
                    try:
                        eliminar_archivo_de_drive(video.drive_file_id)
                    except Exception as e_drive:
                        logger.warning(f"[Drive] Error eliminando {video.drive_file_id}: {str(e_drive)}")

                # Eliminar copia local si aún existe
                if video.ruta_video and os.path.exists(video.ruta_video):
                    os.remove(video.ruta_video)
                
                videos_eliminados.append({
                    "id": video.id,
                    "sena": video.sena,
                    "ruta_video": video.ruta_video
                })
                
                db.delete(video)
                total_eliminados += 1
                
            except Exception as e:
                logger.error(f"Error eliminando video {video.id}: {str(e)}")
                continue
        
        db.commit()
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Se eliminaron {total_eliminados} videos",
            datos={
                "total_eliminados": total_eliminados,
                "videos_eliminados": videos_eliminados
            }
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar videos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar videos: {str(e)}"
        )
@router.get("/videos/todos", response_model=RespuestaLista)
async def listar_todos_los_videos(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    categoria_id: Optional[int] = None,
    estado: Optional[str] = None  # 'aprobado', 'pendiente', 'rechazado'
):
    """
    Listar TODOS los videos del dataset con filtros opcionales
    
    - **categoria_id**: Filtrar por categoría específica (opcional)
    - **estado**: Filtrar por estado: 'aprobado', 'pendiente', 'rechazado' (opcional)
    """
    try:
        # Query base
        query = db.query(VideoDataset).join(CategoriaDataset)
        
        # Filtro por categoría
        if categoria_id:
            query = query.filter(VideoDataset.categoria_id == categoria_id)
        
        # Filtro por estado
        if estado:
            if estado == 'aprobado':
                query = query.filter(VideoDataset.aprobado == True)
            elif estado == 'pendiente':
                # Pendiente: no aprobado y no rechazado
                query = query.filter(
                    VideoDataset.aprobado == False,
                    VideoDataset.rechazado == False
                )
            elif estado == 'rechazado':
                query = query.filter(VideoDataset.rechazado == True)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Estado inválido: {estado}. Use: 'aprobado', 'pendiente' o 'rechazado'"
                )
        
        # Ordenar por fecha de subida descendente
        videos = query.order_by(VideoDataset.fecha_subida.desc()).all()
        
        videos_data = []
        for video in videos:
            # Determinar estado del video
            if video.aprobado:
                estado_video = 'aprobado'
            elif getattr(video, 'rechazado', False):
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
                
                # Estados
                "aprobado": video.aprobado,
                "rechazado": getattr(video, 'rechazado', False),
                "estado": estado_video,
                "usado_entrenamiento": video.usado_entrenamiento,
                "procesado": video.procesado,
                
                # Fechas
                "fecha_subida": video.fecha_subida.isoformat() if video.fecha_subida else None,
                "fecha_procesado": video.fecha_procesado.isoformat() if video.fecha_procesado else None,
                "fecha_aprobado": video.fecha_aprobado.isoformat() if video.fecha_aprobado else None,
                "fecha_rechazado": getattr(video, 'fecha_rechazado', None).isoformat() if getattr(video, 'fecha_rechazado', None) else None,
                
                # Usuario
                "usuario_id": video.usuario_id,
                "subido_por": video.usuario.nombre_completo if video.usuario else None,
                
                # Notas
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
        logger.error(f"Error al listar todos los videos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar videos: {str(e)}"
        )


@router.put("/videos/{video_id}/rechazar", response_model=RespuestaAPI)
async def rechazar_video_dataset(
    video_id: int,
    notas: Optional[str] = None,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    """
    Rechazar un video del dataset
    
    - **video_id**: ID del video a rechazar
    - **notas**: Notas opcionales sobre el rechazo
    """
    try:
        # Buscar el video
        video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
        
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video no encontrado"
            )
        
        # Verificar que existe la columna 'rechazado'
        if not hasattr(video, 'rechazado'):
            logger.error("La columna 'rechazado' no existe en el modelo VideoDataset")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="La columna 'rechazado' no existe. Ejecute la migración de base de datos."
            )
        
        # Actualizar estados
        video.rechazado = True
        video.aprobado = False
        
        # Actualizar fecha de rechazo si existe la columna
        if hasattr(video, 'fecha_rechazado'):
            video.fecha_rechazado = datetime.now(timezone.utc)

        # Actualizar notas si se proporcionaron
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
                "fecha_rechazado": video.fecha_rechazado.isoformat() if hasattr(video, 'fecha_rechazado') and video.fecha_rechazado else None,
                "notas": video.notas
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error al rechazar video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al rechazar video: {str(e)}"
        )

@router.post("/cargar-modelo", response_model=RespuestaAPI)
async def cargar_modelo_preentrenado(
    archivo: UploadFile = File(...),
    nombre_modelo: str = Form(...),
    descripcion: Optional[str] = Form(None),
    clases: Optional[str] = Form(None),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    """
    Carga un modelo ya entrenado (.pth o .pt) y lo registra en la base de datos
    
    - **archivo**: Archivo del modelo (.pth o .pt)
    - **nombre_modelo**: Nombre único para identificar el modelo
    - **descripcion**: Descripción opcional del modelo
    - **clases**: JSON string con la lista de clases/señas (opcional, se detecta automáticamente)
    """
    ruta_modelo = None
    accuracy_detectado = 0.0  # Valor por defecto
    loss_detectado = 0.0  # Valor por defecto
    
    try:
        # Validar extensión del archivo
        extension = archivo.filename.split('.')[-1].lower()
        if extension not in ['pth', 'pt']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo debe ser .pth o .pt"
            )
        
        # Validar nombre del modelo
        if not nombre_modelo or len(nombre_modelo.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre del modelo debe tener al menos 3 caracteres"
            )
        
        # Verificar que no exista un modelo con ese nombre
        modelo_existente = db.query(ModeloIA).filter(
            ModeloIA.nombre == nombre_modelo
        ).first()
        
        if modelo_existente:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ya existe un modelo con el nombre '{nombre_modelo}'"
            )
        
        # Crear directorio para modelos si no existe
        modelos_dir = PathLib("modelos_entrenados")
        modelos_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo del modelo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nombre_archivo = f"{nombre_modelo}_{timestamp}.{extension}"
        ruta_modelo = modelos_dir / nombre_archivo
        
        # Leer y guardar el archivo
        contenido = await archivo.read()
        
        if len(contenido) < 1024:  # Menos de 1KB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo del modelo es demasiado pequeño o está corrupto"
            )
        
        with open(ruta_modelo, 'wb') as f:
            f.write(contenido)
        
        logger.info(f"Modelo guardado en: {ruta_modelo}")
        
        # Validar y detectar clases del modelo
        try:
            estado = torch.load(str(ruta_modelo), map_location='cpu')
            
            # DETECTAR ACCURACY Y LOSS DEL MODELO PRECARGADO
            if isinstance(estado, dict):
                # Buscar accuracy en diferentes formatos
                if 'accuracy' in estado:
                    accuracy_detectado = round(float(estado['accuracy']), 4)
                elif 'val_accuracy' in estado:
                    accuracy_detectado = round(float(estado['val_accuracy']), 4)
                elif 'test_accuracy' in estado:
                    accuracy_detectado = round(float(estado['test_accuracy']), 4)
                elif 'best_accuracy' in estado:
                    accuracy_detectado = round(float(estado['best_accuracy']), 4)
                elif 'top1_acc' in estado:
                    accuracy_detectado = round(float(estado['top1_acc']), 4)
                
                # Buscar loss en diferentes formatos
                if 'loss' in estado:
                    loss_detectado = round(float(estado['loss']), 4)
                elif 'val_loss' in estado:
                    loss_detectado = round(float(estado['val_loss']), 4)
                elif 'test_loss' in estado:
                    loss_detectado = round(float(estado['test_loss']), 4)
                
                logger.info(f"Accuracy detectado en modelo: {accuracy_detectado}")
                logger.info(f"Loss detectado en modelo: {loss_detectado}")
            
            # Detectar clases automáticamente
            lista_clases_detectadas = None
            num_clases_detectadas = None
            
            # Opción 1: Si el modelo tiene metadata de clases
            if isinstance(estado, dict):
                # Buscar en metadata común
                if 'clases' in estado:
                    lista_clases_detectadas = estado['clases']
                    logger.info(f"Clases encontradas en 'clases': {lista_clases_detectadas}")
                elif 'class_names' in estado:
                    lista_clases_detectadas = estado['class_names']
                    logger.info(f"Clases encontradas en 'class_names': {lista_clases_detectadas}")
                elif 'idx_to_class' in estado:
                    lista_clases_detectadas = list(estado['idx_to_class'].values())
                    logger.info(f"Clases encontradas en 'idx_to_class': {lista_clases_detectadas}")
                elif 'num_classes' in estado:
                    num_clases_detectadas = estado['num_classes']
                    logger.info(f"Número de clases encontrado: {num_clases_detectadas}")
                
                # Opción 2: Detectar desde la última capa del modelo
                if lista_clases_detectadas is None and num_clases_detectadas is None:
                    model_state = estado.get('model_state_dict', estado)
                    
                    # Buscar la última capa fully connected o de salida
                    for key in reversed(list(model_state.keys())):
                        if any(term in key.lower() for term in ['fc.weight', 'classifier.weight', 'head.weight', 'out.weight']):
                            # El número de clases es la primera dimensión del peso de salida
                            num_clases_detectadas = model_state[key].shape[0]
                            logger.info(f"Número de clases detectado desde capa '{key}': {num_clases_detectadas}")
                            break
            
            # Determinar lista final de clases
            lista_clases = None
            
            if lista_clases_detectadas:
                # Se detectaron clases con nombres
                if clases and clases.strip():
                    # Validar que coincidan en número
                    try:
                        lista_clases_manual = json.loads(clases)
                        if len(lista_clases_manual) != len(lista_clases_detectadas):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"El modelo tiene {len(lista_clases_detectadas)} clases, pero proporcionaste {len(lista_clases_manual)}"
                            )
                        # Usar las clases manuales (nombres personalizados)
                        lista_clases = lista_clases_manual
                        logger.info("Usando clases proporcionadas manualmente")
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="El formato de clases debe ser un JSON válido. Ejemplo: [\"hola\", \"gracias\", \"adios\"]"
                        )
                else:
                    # Usar las clases detectadas
                    lista_clases = lista_clases_detectadas
                    logger.info("Usando clases detectadas del modelo")
            
            elif num_clases_detectadas:
                # Solo se detectó el número de clases
                if clases and clases.strip():
                    try:
                        lista_clases_manual = json.loads(clases)
                        if len(lista_clases_manual) != num_clases_detectadas:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"El modelo tiene {num_clases_detectadas} clases, pero proporcionaste {len(lista_clases_manual)}"
                            )
                        lista_clases = lista_clases_manual
                        logger.info("Usando clases proporcionadas manualmente")
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="El formato de clases debe ser un JSON válido. Ejemplo: [\"hola\", \"gracias\", \"adios\"]"
                        )
                else:
                    # Generar nombres genéricos
                    lista_clases = [f"clase_{i}" for i in range(num_clases_detectadas)]
                    logger.warning(f"Generando nombres genéricos para {num_clases_detectadas} clases")
            
            else:
                # No se pudo detectar nada
                if clases and clases.strip():
                    try:
                        lista_clases = json.loads(clases)
                        logger.info(f"Usando clases proporcionadas manualmente: {len(lista_clases)} clases")
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="El formato de clases debe ser un JSON válido. Ejemplo: [\"hola\", \"gracias\", \"adios\"]"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No se pudieron detectar las clases automáticamente. Por favor proporciónalas manualmente en formato JSON. Ejemplo: [\"hola\", \"gracias\", \"adios\"]"
                    )
            
            # Validar lista final
            if not isinstance(lista_clases, list) or len(lista_clases) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Debe haber al menos 2 clases"
                )
            
            logger.info(f"Clases finales configuradas: {lista_clases}")
            
        except HTTPException:
            raise
        except json.JSONDecodeError:
            if os.path.exists(ruta_modelo):
                os.remove(ruta_modelo)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El formato de clases debe ser un JSON válido. Ejemplo: [\"hola\", \"gracias\", \"adios\"]"
            )
        except Exception as e:
            if os.path.exists(ruta_modelo):
                os.remove(ruta_modelo)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error al procesar modelo: {str(e)}"
            )
        
        # Crear registro en la base de datos USANDO EL ACCURACY Y LOSS DETECTADOS
        nuevo_modelo = ModeloIA(
            nombre=nombre_modelo,
            descripcion=descripcion or f"Modelo pre-entrenado cargado el {datetime.now().strftime('%Y-%m-%d')}",
            ruta_archivo=str(ruta_modelo),
            tipo_modelo="video_cnn",
            num_clases=len(lista_clases),
            clases_json=json.dumps(lista_clases, ensure_ascii=False),
            accuracy=accuracy_detectado,  # ← USAR ACCURACY DETECTADO
            loss=loss_detectado,  # ← USAR LOSS DETECTADO
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
        
        logger.info(f"Modelo registrado en BD: ID={nuevo_modelo.id}, Nombre={nuevo_modelo.nombre}, Accuracy={accuracy_detectado}")
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Modelo '{nombre_modelo}' cargado exitosamente con {len(lista_clases)} clases" + 
                   (f" y accuracy de {accuracy_detectado:.2%}" if accuracy_detectado > 0 else ""),
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
        logger.error(f"Error cargando modelo: {str(e)}", exc_info=True)
        
        # Limpiar archivo si se creó
        if ruta_modelo and os.path.exists(ruta_modelo):
            try:
                os.remove(ruta_modelo)
                logger.info(f"Archivo temporal eliminado: {ruta_modelo}")
            except Exception as cleanup_error:
                logger.error(f"Error al limpiar archivo: {cleanup_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cargar modelo: {str(e)}"
        )
    
@router.post("/validar-modelo", response_model=RespuestaAPI)
async def validar_modelo_preentrenado(
    archivo: UploadFile = File(...),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    """
    Valida un archivo de modelo sin guardarlo.
    Útil para verificar la compatibilidad antes de cargarlo.
    
    Retorna información sobre:
    - Validez del archivo
    - Estructura del modelo
    - Número de clases detectadas
    - Clases disponibles (si existen)
    - Tamaño del archivo
    """
    temp_path = None
    
    try:
        # Validar extensión
        extension = archivo.filename.split('.')[-1].lower()
        if extension not in ['pth', 'pt']:
            return RespuestaAPI(
                exito=False,
                mensaje="Extensión inválida. Use .pth o .pt",
                datos={"valido": False, "error": "extension_invalida"}
            )
        
        # Leer archivo a memoria
        contenido = await archivo.read()
        
        if len(contenido) < 1024:
            return RespuestaAPI(
                exito=False,
                mensaje="Archivo demasiado pequeño (menos de 1KB)",
                datos={"valido": False, "error": "archivo_pequeno"}
            )
        
        # Crear archivo temporal para validación
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as temp_file:
            temp_file.write(contenido)
            temp_path = temp_file.name
        
        try:
            # Intentar cargar el modelo
            estado = torch.load(temp_path, map_location='cpu')
            
            # Extraer información del modelo
            info = {
                "valido": True,
                "tipo": type(estado).__name__,
                "tamaño_mb": round(len(contenido) / (1024 * 1024), 2),
                "nombre_archivo": archivo.filename
            }
            
            # Analizar estructura del modelo
            if isinstance(estado, dict):
                info["keys"] = list(estado.keys())[:20]  # Limitar a 20 keys
                
                # Detectar tipo de checkpoint
                if 'model_state_dict' in estado:
                    info["estructura"] = "checkpoint_completo"
                    info["num_parametros"] = len(estado['model_state_dict'])
                    state_dict = estado['model_state_dict']
                elif any(k.startswith('conv') or k.startswith('fc') or k.startswith('layer') for k in estado.keys()):
                    info["estructura"] = "state_dict"
                    info["num_parametros"] = len(estado)
                    state_dict = estado
                else:
                    info["estructura"] = "formato_personalizado"
                    state_dict = estado
                
                # Detectar clases
                clases_detectadas = None
                num_clases_detectadas = None
                
                if 'clases' in estado:
                    clases_detectadas = estado['clases']
                    info["clases_detectadas"] = clases_detectadas
                    info["num_clases"] = len(clases_detectadas)
                elif 'class_names' in estado:
                    clases_detectadas = estado['class_names']
                    info["clases_detectadas"] = clases_detectadas
                    info["num_clases"] = len(clases_detectadas)
                elif 'idx_to_class' in estado:
                    clases_detectadas = list(estado['idx_to_class'].values())
                    info["clases_detectadas"] = clases_detectadas
                    info["num_clases"] = len(clases_detectadas)
                elif 'num_classes' in estado:
                    num_clases_detectadas = estado['num_classes']
                    info["num_clases"] = num_clases_detectadas
                    info["clases_detectadas"] = None
                else:
                    # Buscar en las capas del modelo
                    for key in reversed(list(state_dict.keys())):
                        if any(term in key.lower() for term in ['fc.weight', 'classifier.weight', 'head.weight', 'out.weight']):
                            num_clases_detectadas = state_dict[key].shape[0]
                            info["num_clases"] = num_clases_detectadas
                            info["capa_salida"] = key
                            info["clases_detectadas"] = None
                            break
                
                # Información adicional del modelo
                if 'epoch' in estado:
                    info["epochs_entrenadas"] = estado['epoch']
                if 'accuracy' in estado:
                    info["accuracy"] = round(float(estado['accuracy']), 4)
                if 'loss' in estado:
                    info["loss"] = round(float(estado['loss']), 4)
                
                # Detectar arquitectura
                capas_encontradas = []
                for key in list(state_dict.keys())[:10]:
                    if 'lstm' in key.lower():
                        info["arquitectura_detectada"] = "LSTM"
                        break
                    elif 'gru' in key.lower():
                        info["arquitectura_detectada"] = "GRU"
                        break
                    elif 'conv3d' in key.lower():
                        info["arquitectura_detectada"] = "Conv3D (Video)"
                        break
                    elif 'conv' in key.lower():
                        info["arquitectura_detectada"] = "CNN"
                        break
            else:
                info["estructura"] = "modelo_directo"
                info["arquitectura_detectada"] = type(estado).__name__
            
            mensaje_validacion = "Modelo válido y compatible"
            if info.get("num_clases"):
                mensaje_validacion += f" - {info['num_clases']} clases detectadas"
            
            return RespuestaAPI(
                exito=True,
                mensaje=mensaje_validacion,
                datos=info
            )
            
        except Exception as e:
            logger.error(f"Error al validar modelo: {str(e)}", exc_info=True)
            return RespuestaAPI(
                exito=False,
                mensaje=f"Modelo inválido o corrupto: {str(e)}",
                datos={
                    "valido": False, 
                    "error": str(e),
                    "tipo_error": type(e).__name__
                }
            )
        
    except Exception as e:
        logger.error(f"Error en validación: {str(e)}", exc_info=True)
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error en validación: {str(e)}",
            datos={"valido": False, "error": str(e)}
        )
    
    finally:
        # Limpiar archivo temporal
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Archivo temporal de validación eliminado: {temp_path}")
            except Exception as cleanup_error:
                logger.error(f"Error al limpiar archivo temporal: {cleanup_error}")