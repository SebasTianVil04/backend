import os
import re
import gc
import time
import json
import logging
import threading
from datetime import datetime, date
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from app.modelos.dataset import VideoDataset
from app.utilidades.base_datos import obtener_bd
from app.dependencias.permisos import requiere_permiso
from app.modelos.usuario import Usuario
from app.modelos.rol import Rol
from app.modelos.leccion import Leccion
from app.modelos.examen import Examen, ResultadoExamen
from app.modelos.progreso import ProgresoLeccion as Progreso
from app.modelos.entrenamiento import ModeloIA
from app.modelos.estudio import SesionEstudio
from app.servicios.servicio_entrenamiento import servicio_entrenamiento, progresos_entrenamiento
from app.servicios.dataset_service import dataset_service
from app.servicios.archivos import archivo_service
from app.esquemas.usuario_schemas import UsuarioRespuesta, EstadisticasUsuario
from app.esquemas.respuesta_schemas import RespuestaAPI
from app.utilidades.validaciones import validar_longitud_telefono_por_pais, validar_telefono_unico

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["Administracion"],
    dependencies=[Depends(HTTPBearer())]
)

_lock_inicio_entrenamiento = threading.Lock()


class ActivarModeloRequest(BaseModel):
    modelo_id: Optional[int] = None
    nombre_modelo: Optional[str] = None


class ActivarMultiplesRequest(BaseModel):
    nombres_modelos: List[str] = []
    pesos: Optional[List[float]] = None


class IniciarEntrenamientoRequest(BaseModel):
    categoria_ids: Optional[List[int]] = None
    epochs: int = 150


def _liberar_memoria() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _eliminar_archivo(ruta: str, intentos: int = 5, espera: float = 2.0) -> bool:
    for intento in range(intentos):
        try:
            os.remove(ruta)
            return True
        except FileNotFoundError:
            return True
        except OSError as e:
            logger.warning(f"[ELIMINAR] Intento {intento + 1}/{intentos} fallido para {ruta}: {e}")
            if intento < intentos - 1:
                _liberar_memoria()
                time.sleep(espera)
    return False


def _evaluar_calidad_modelo(accuracy: float) -> str:
    if accuracy >= 0.95:
        return "Excelente"
    elif accuracy >= 0.90:
        return "Muy Bueno"
    elif accuracy >= 0.85:
        return "Bueno"
    elif accuracy >= 0.75:
        return "Regular"
    else:
        return "Necesita Mejora"


def _iso(valor) -> Optional[str]:
    return valor.isoformat() if valor else None


def _datos_usuario(usuario: Usuario) -> Dict[str, Any]:
    return {
        "id": usuario.id,
        "nombre_usuario": usuario.nombre_usuario,
        "email": usuario.email,
        "activo": usuario.activo,
        "verificado": usuario.verificado,
        "rol_id": usuario.rol_id,
    }


def _categorias_con_videos(bd: Session) -> List[int]:
    filas = bd.query(VideoDataset.categoria_id).filter(VideoDataset.aprobado == True).distinct().all()
    return [fila[0] for fila in filas if fila[0] is not None]


def _hay_entrenamiento_activo() -> bool:
    return any(p.get("entrenando") for p in list(progresos_entrenamiento.values()))


def _buscar_modelo(bd: Session, solicitud: ActivarModeloRequest) -> Optional[ModeloIA]:
    if solicitud.modelo_id:
        return bd.query(ModeloIA).filter(ModeloIA.id == solicitud.modelo_id).first()
    if solicitud.nombre_modelo:
        return bd.query(ModeloIA).filter(ModeloIA.nombre == solicitud.nombre_modelo).first()
    return None


def _resumen_modelo(modelo: ModeloIA) -> Dict[str, Any]:
    accuracy = modelo.accuracy or 0
    return {
        "id": modelo.id,
        "nombre": modelo.nombre,
        "version": modelo.version or "1.0",
        "descripcion": modelo.descripcion,
        "accuracy": modelo.accuracy,
        "precision": getattr(modelo, "precision", None),
        "recall": getattr(modelo, "recall", None),
        "f1_score": getattr(modelo, "f1_score", None),
        "num_clases": modelo.num_clases,
        "total_imagenes": modelo.total_imagenes,
        "epocas_entrenamiento": modelo.epocas_entrenamiento,
        "activo": modelo.activo,
        "entrenando": getattr(modelo, "entrenando", False),
        "estado_texto": "Activo" if modelo.activo else "Inactivo",
        "calidad": _evaluar_calidad_modelo(accuracy),
        "accuracy_porcentaje": f"{accuracy * 100:.2f}%",
        "fecha_creacion": _iso(getattr(modelo, "fecha_creacion", None)),
        "fecha_entrenamiento": _iso(modelo.fecha_entrenamiento),
        "fecha_activacion": _iso(getattr(modelo, "fecha_activacion", None)),
        "arquitectura": getattr(modelo, "arquitectura", None),
        "tamaño_mb": getattr(modelo, "tamaño_mb", None),
        "ruta_modelo": getattr(modelo, "ruta_archivo", None)
    }


def _detalle_modelo(modelo: ModeloIA) -> Dict[str, Any]:
    clases: List[str] = []
    if modelo.clases_json:
        try:
            clases = json.loads(modelo.clases_json)
        except (ValueError, TypeError):
            clases = []

    accuracy = modelo.accuracy or 0
    return {
        "id": modelo.id,
        "nombre": modelo.nombre,
        "version": modelo.version or "1.0",
        "descripcion": modelo.descripcion or "Modelo de reconocimiento de senas",
        "accuracy": modelo.accuracy,
        "loss": getattr(modelo, "loss", None),
        "precision": getattr(modelo, "precision", None),
        "recall": getattr(modelo, "recall", None),
        "f1_score": getattr(modelo, "f1_score", None),
        "num_clases": modelo.num_clases,
        "total_imagenes": modelo.total_imagenes,
        "epocas_entrenamiento": modelo.epocas_entrenamiento,
        "activo": modelo.activo,
        "fecha_creacion": _iso(getattr(modelo, "fecha_creacion", None)),
        "fecha_entrenamiento": _iso(modelo.fecha_entrenamiento),
        "fecha_activacion": _iso(getattr(modelo, "fecha_activacion", None)),
        "ruta_archivo": getattr(modelo, "ruta_archivo", None),
        "arquitectura": modelo.arquitectura or "LSTM_Keypoints_Unificado",
        "tamaño_mb": getattr(modelo, "tamaño_mb", None) or 0,
        "parametros": 0,
        "clases": clases,
        "estado": "activo" if modelo.activo else "inactivo",
        "calidad": _evaluar_calidad_modelo(accuracy),
        "disponible": True
    }


@router.get("/dashboard", response_model=Dict[str, Any])
def obtener_dashboard(
    usuario_actual: Usuario = Depends(requiere_permiso("admin.dashboard.ver")),
    bd: Session = Depends(obtener_bd)
):
    try:
        fecha_inicio_mes = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        total_usuarios = bd.query(Usuario).count()
        usuarios_activos = bd.query(Usuario).filter(Usuario.activo == True).count()

        usuarios_nuevos_mes = 0

        total_lecciones = bd.query(Leccion).count()
        lecciones_activas = bd.query(Leccion).filter(Leccion.activa == True).count()

        total_examenes = bd.query(Examen).count()

        examenes_completados = bd.query(
            ResultadoExamen.examen_id, ResultadoExamen.usuario_id
        ).distinct().count()

        progreso_promedio_query = bd.query(func.avg(Progreso.mejor_precision)).filter(
            Progreso.mejor_precision != None
        ).scalar()

        progreso_promedio = float(progreso_promedio_query * 100) if progreso_promedio_query else 0.0

        return {
            "usuarios": {
                "total": total_usuarios,
                "activos": usuarios_activos,
                "nuevos_mes": usuarios_nuevos_mes
            },
            "lecciones": {
                "total": total_lecciones,
                "activas": lecciones_activas
            },
            "examenes": {
                "total": total_examenes,
                "completados": examenes_completados
            },
            "rendimiento": {
                "progreso_promedio": round(progreso_promedio, 2)
            }
        }

    except Exception as e:
        logger.error(f"Error en dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadisticas: {str(e)}"
        )


@router.delete("/modelos/{nombre_modelo}", response_model=RespuestaAPI, status_code=200)
def eliminar_modelo(
    nombre_modelo: str,
    eliminar_videos: bool = Query(False),
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    try:
        modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

        if not modelo:
            logger.warning(f"[ELIMINAR] Modelo '{nombre_modelo}' no encontrado en BD")
            raise HTTPException(status_code=404, detail=f"Modelo '{nombre_modelo}' no encontrado")

        logger.info(f"[ELIMINAR] Iniciando eliminacion de modelo: {nombre_modelo}")

        archivos_eliminados: List[str] = []
        ruta_archivo = getattr(modelo, "ruta_archivo", None)

        try:
            from app.servicios.gestor_reconocimiento import limpiar_cache
            limpiar_cache()
        except Exception as e:
            logger.warning(f"[ELIMINAR] No se pudo limpiar la cache de reconocimiento: {e}")

        if ruta_archivo and os.path.exists(ruta_archivo):
            _liberar_memoria()
            if _eliminar_archivo(ruta_archivo):
                archivos_eliminados.append(ruta_archivo)
                logger.info(f"[ELIMINAR] Archivo eliminado: {ruta_archivo}")
            else:
                logger.error(f"[ELIMINAR] No se pudo eliminar el archivo: {ruta_archivo}")

        videos_eliminados = 0
        errores_videos: List[int] = []

        if eliminar_videos:
            videos = list(getattr(modelo, "videos_entrenamiento", None) or [])
            logger.info(f"[ELIMINAR] Eliminando {len(videos)} videos asociados")

            for video in videos:
                try:
                    if video.ruta_video and os.path.exists(video.ruta_video):
                        if _eliminar_archivo(video.ruta_video, intentos=3, espera=1.0):
                            archivos_eliminados.append(video.ruta_video)
                            videos_eliminados += 1
                        else:
                            errores_videos.append(video.id)

                    secuencia = dataset_service.secuencias_dir / f"{video.id}.npz"
                    if secuencia.exists():
                        secuencia.unlink()

                    db.delete(video)
                except Exception as e:
                    logger.warning(f"[ELIMINAR] Error eliminando video {video.id}: {str(e)}")
                    errores_videos.append(video.id)

            db.flush()

        db.delete(modelo)
        db.commit()
        logger.info("[ELIMINAR] Modelo eliminado exitosamente de BD")

        progresos_entrenamiento.pop(nombre_modelo, None)

        _liberar_memoria()

        mensaje = f"Modelo '{nombre_modelo}' eliminado correctamente"
        if videos_eliminados > 0:
            mensaje += f" junto con {videos_eliminados} videos"

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos={
                "modelo_eliminado": True,
                "nombre_modelo": nombre_modelo,
                "archivos_eliminados": len(archivos_eliminados),
                "videos_eliminados": videos_eliminados,
                "errores_videos": errores_videos if errores_videos else None,
                "rutas_eliminadas": archivos_eliminados[:5] if len(archivos_eliminados) <= 5 else f"{len(archivos_eliminados)} archivos"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ELIMINAR] Error crítico eliminando modelo '{nombre_modelo}': {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al eliminar modelo: {str(e)}")


@router.get("/usuarios", response_model=List[UsuarioRespuesta])
def listar_usuarios(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.listar")),
    bd: Session = Depends(obtener_bd)
):
    usuarios = (
        bd.query(Usuario)
        .join(Rol, Usuario.rol_id == Rol.id)
        .filter(Rol.codigo != "admin")
        .offset(skip)
        .limit(limit)
        .all()
    )

    resultado = []
    for usuario in usuarios:
        datos = _datos_usuario(usuario)
        datos["rol"] = usuario.rol
        datos["permisos"] = usuario.permisos
        resultado.append(datos)

    return resultado


@router.get("/usuarios/{usuario_id}", response_model=UsuarioRespuesta)
def obtener_usuario_por_id(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.ver")),
    db: Session = Depends(obtener_bd)
):
    usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

    if not usuario:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

    datos = _datos_usuario(usuario)
    datos["rol"] = usuario.rol
    datos["permisos"] = usuario.permisos
    return datos


@router.put("/usuarios/{usuario_id}", response_model=RespuestaAPI)
def actualizar_usuario(
    usuario_id: int,
    nombre_usuario: Optional[str] = None,
    email: Optional[str] = None,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.editar")),
    db: Session = Depends(obtener_bd)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

        campos_actualizados = []

        if nombre_usuario is not None and nombre_usuario.strip():
            existente = db.query(Usuario).filter(
                Usuario.nombre_usuario == nombre_usuario.strip(),
                Usuario.id != usuario_id
            ).first()
            if existente:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El nombre de usuario ya está en uso"
                )
            usuario.nombre_usuario = nombre_usuario.strip()
            campos_actualizados.append("nombre_usuario")

        if email is not None and email.strip():
            email_lower = email.strip().lower()
            existente = db.query(Usuario).filter(
                Usuario.email == email_lower,
                Usuario.id != usuario_id
            ).first()
            if existente:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El email ya está en uso por otro usuario"
                )
            usuario.email = email_lower
            campos_actualizados.append("email")

        db.commit()
        db.refresh(usuario)

        datos = _datos_usuario(usuario)
        datos["rol"] = usuario.rol.codigo if usuario.rol else None

        mensaje = "Usuario actualizado exitosamente"
        if campos_actualizados:
            mensaje += f". Campos actualizados: {', '.join(campos_actualizados)}"

        return RespuestaAPI(exito=True, mensaje=mensaje, datos=datos)

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar usuario: {str(e)}"
        )


@router.put("/usuarios/{usuario_id}/rol", response_model=RespuestaAPI)
def asignar_rol(
    usuario_id: int,
    rol_id: int = Query(...),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.asignar_rol")),
    db: Session = Depends(obtener_bd)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

        nuevo_rol = db.query(Rol).filter(Rol.id == rol_id, Rol.activo == True).first()

        if not nuevo_rol:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El rol con id '{rol_id}' no existe o está inactivo"
            )

        if usuario.id == usuario_actual.id and nuevo_rol.codigo != "admin":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No puedes quitarte el rol de administrador a ti mismo"
            )

        usuario.rol_id = nuevo_rol.id
        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje=f"Rol actualizado a {nuevo_rol.codigo} exitosamente",
            datos={"usuario_id": usuario_id, "rol_id": nuevo_rol.id, "rol": nuevo_rol.codigo}
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al asignar rol: {str(e)}"
        )


@router.delete("/usuarios/{usuario_id}", response_model=RespuestaAPI)
def eliminar_usuario(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.eliminar")),
    db: Session = Depends(obtener_bd)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

        if usuario.id == usuario_actual.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No puedes eliminar tu propia cuenta"
            )

        nombre_usuario = usuario.nombre_usuario
        email_usuario = usuario.email

        db.delete(usuario)
        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje=f"Usuario '{nombre_usuario}' eliminado exitosamente",
            datos={
                "usuario_eliminado": email_usuario,
                "fecha_eliminacion": datetime.utcnow().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar usuario: {str(e)}"
        )


@router.get("/usuarios/{usuario_id}/estadisticas", response_model=EstadisticasUsuario)
def obtener_estadisticas_usuario(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.ver_estadisticas")),
    bd: Session = Depends(obtener_bd)
):
    usuario = bd.query(Usuario).filter(Usuario.id == usuario_id).first()
    if not usuario:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

    progresos = bd.query(Progreso).filter(Progreso.usuario_id == usuario_id).all()
    resultados_examenes = bd.query(ResultadoExamen).filter(ResultadoExamen.usuario_id == usuario_id).all()

    total_lecciones_completadas = sum(1 for p in progresos if p.completada)
    total_lecciones_disponibles = bd.query(Leccion).filter(Leccion.activa == True).count()

    precisiones = [p.mejor_precision for p in progresos if p.mejor_precision]
    precision_promedio = (sum(precisiones) / len(precisiones) * 100) if precisiones else 0
    precision_maxima = (max(precisiones) * 100) if precisiones else 0
    precision_minima = (min(precisiones) * 100) if precisiones else 0

    tiempo_total = 0
    for p in progresos:
        if p.fecha_inicio and p.fecha_completada:
            tiempo_total += (p.fecha_completada - p.fecha_inicio).total_seconds() / 60

    examenes_completados = len(resultados_examenes)
    examenes_aprobados = sum(1 for r in resultados_examenes if r.aprobado)
    tasa_aprobacion = (examenes_aprobados / examenes_completados * 100) if examenes_completados > 0 else 0

    experiencia_total = sum(p.total_puntos or 0 for p in progresos)

    ultima_actividad = None
    ultimas_practicas = [p.ultima_practica for p in progresos if p.ultima_practica]
    if ultimas_practicas:
        ultima_actividad = max(ultimas_practicas).isoformat()

    ultima_leccion_completada = None
    lecciones_completadas = [p for p in progresos if p.completada and p.fecha_completada]
    if lecciones_completadas:
        ultima_leccion = max(lecciones_completadas, key=lambda x: x.fecha_completada)
        ultima_leccion_completada = ultima_leccion.leccion.titulo if ultima_leccion.leccion else None

    return {
        "usuario_id": usuario.id,
        "nombre_completo": usuario.nombre_usuario,
        "email": usuario.email,
        "total_lecciones_completadas": total_lecciones_completadas,
        "total_lecciones_disponibles": total_lecciones_disponibles,
        "porcentaje_progreso": (total_lecciones_completadas / total_lecciones_disponibles * 100) if total_lecciones_disponibles > 0 else 0,
        "puntuacion_promedio": round(precision_promedio, 2),
        "puntuacion_maxima": round(precision_maxima, 2),
        "puntuacion_minima": round(precision_minima, 2),
        "tiempo_total_estudio": round(tiempo_total, 2),
        "tiempo_promedio_sesion": round(tiempo_total / len(progresos), 2) if progresos else 0,
        "sesiones_totales": len(progresos),
        "racha_actual": 0,
        "racha_maxima": 0,
        "nivel_actual": 1,
        "experiencia_total": experiencia_total,
        "examenes_completados": examenes_completados,
        "examenes_aprobados": examenes_aprobados,
        "tasa_aprobacion": round(tasa_aprobacion, 2),
        "fecha_registro": None,
        "ultima_actividad": ultima_actividad,
        "ultima_leccion_completada": ultima_leccion_completada,
        "categorias_dominadas": 0,
        "categoria_favorita": None,
        "dias_consecutivos_activo": 0,
        "logros_desbloqueados": 0,
        "posicion_ranking": None,
        "percentil_rendimiento": None
    }


@router.patch("/usuarios/{usuario_id}/estado", response_model=RespuestaAPI)
def cambiar_estado_usuario(
    usuario_id: int,
    activo: bool,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.cambiar_estado")),
    bd: Session = Depends(obtener_bd)
):
    usuario = bd.query(Usuario).filter(Usuario.id == usuario_id).first()
    if not usuario:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

    if usuario.id == usuario_actual.id and not activo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No puedes desactivar tu propia cuenta"
        )

    usuario.activo = activo
    bd.commit()

    return RespuestaAPI(
        exito=True,
        mensaje=f"Usuario {'activado' if activo else 'desactivado'} exitosamente"
    )


@router.post("/lecciones/{leccion_id}/contenido", response_model=RespuestaAPI)
async def subir_contenido_leccion(
    leccion_id: int,
    archivo: UploadFile = File(...),
    usuario_actual: Usuario = Depends(requiere_permiso("lecciones.editar")),
    bd: Session = Depends(obtener_bd)
):
    leccion = bd.query(Leccion).filter(Leccion.id == leccion_id).first()
    if not leccion:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Leccion no encontrada")

    try:
        if not (archivo.content_type or "").startswith(('image/', 'video/')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se permiten archivos de imagen o video"
            )

        ruta_relativa, ruta_completa = await archivo_service.subir_video_leccion(archivo)

        leccion.video_url = ruta_relativa
        bd.commit()

        return RespuestaAPI(
            exito=True,
            mensaje="Contenido subido exitosamente",
            datos={"ruta": ruta_relativa}
        )

    except HTTPException:
        raise
    except Exception as e:
        bd.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al subir contenido: {str(e)}"
        )


@router.post("/modelo/entrenar", response_model=RespuestaAPI)
def iniciar_entrenamiento_modelo(
    background_tasks: BackgroundTasks,
    solicitud: Optional[IniciarEntrenamientoRequest] = None,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        solicitud = solicitud or IniciarEntrenamientoRequest()
        categoria_ids = solicitud.categoria_ids or _categorias_con_videos(bd)

        if not categoria_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay videos aprobados para entrenar"
            )

        validacion = servicio_entrenamiento.validar_dataset_entrenamiento(categoria_ids, bd)
        if not validacion.get("valido"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Dataset insuficiente: {validacion.get('total_videos', 0)} videos válidos y "
                    f"{len(validacion.get('senas_con_minimo', []))} señas con al menos "
                    f"{servicio_entrenamiento.min_videos_por_sena} videos "
                    f"(se requieren 20 videos y 2 señas)"
                )
            )

        with _lock_inicio_entrenamiento:
            if _hay_entrenamiento_activo():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Ya hay un entrenamiento en curso"
                )

            nombre_modelo = servicio_entrenamiento.generar_nombre_modelo()
            progresos_entrenamiento[nombre_modelo] = {
                "nombre_modelo": nombre_modelo,
                "estado": "iniciando",
                "entrenando": True,
                "progreso": 0.0,
                "mensaje": "Entrenamiento en cola..."
            }

        background_tasks.add_task(
            servicio_entrenamiento.entrenar, nombre_modelo, categoria_ids, solicitud.epochs
        )

        return RespuestaAPI(
            exito=True,
            mensaje="Entrenamiento iniciado exitosamente",
            datos={
                "nombre_modelo": nombre_modelo,
                "categoria_ids": categoria_ids,
                "epochs": solicitud.epochs,
                "total_videos": validacion.get("total_videos", 0)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al iniciar entrenamiento: {str(e)}"
        )


@router.get("/modelo/estado", response_model=RespuestaAPI)
def obtener_estado_modelo(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        estadisticas = dataset_service.obtener_estadisticas_dataset(bd)
        modelos_serializados = [_resumen_modelo(modelo) for modelo in bd.query(ModeloIA).all()]

        return RespuestaAPI(
            exito=True,
            mensaje="Estado del modelo obtenido",
            datos={
                "dataset": estadisticas,
                "modelos_disponibles": modelos_serializados
            }
        )
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estado del modelo: {str(e)}"
        )


@router.post("/modelo/toggle", response_model=RespuestaAPI)
def toggle_modelo(
    request: ActivarModeloRequest,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        modelo = _buscar_modelo(bd, request)

        if not modelo:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")

        if modelo.activo:
            modelo.activo = False
            mensaje = f"Modelo '{modelo.nombre}' desactivado"
        else:
            bd.query(ModeloIA).update({"activo": False}, synchronize_session=False)
            bd.flush()
            modelo.activo = True
            modelo.fecha_activacion = datetime.now()
            mensaje = f"Modelo '{modelo.nombre}' activado"

        bd.commit()

        try:
            from app.servicios.gestor_reconocimiento import limpiar_cache
            limpiar_cache()
            logger.info(f"[TOGGLE] Cache de reconocimiento limpiada tras {mensaje}")
        except Exception as e:
            logger.warning(f"[TOGGLE] No se pudo limpiar cache: {e}")

        return RespuestaAPI(exito=True, mensaje=mensaje, datos={"activo": modelo.activo})

    except HTTPException:
        bd.rollback()
        raise
    except Exception as e:
        bd.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modelos", response_model=RespuestaAPI)
def listar_modelos(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        modelos_completos = [_detalle_modelo(modelo) for modelo in bd.query(ModeloIA).all()]

        modelos_completos.sort(key=lambda x: x.get("fecha_entrenamiento") or "", reverse=True)

        logger.info(f"Respuesta final con {len(modelos_completos)} modelos")

        return RespuestaAPI(
            exito=True,
            mensaje=f"Se encontraron {len(modelos_completos)} modelos",
            datos=modelos_completos
        )

    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        return RespuestaAPI(exito=False, mensaje=f"Error listando modelos: {str(e)}")


@router.post("/modelo/activar", response_model=RespuestaAPI)
def activar_modelo(
    request: ActivarModeloRequest,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        modelo = _buscar_modelo(bd, request)

        if not modelo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modelo '{request.nombre_modelo or request.modelo_id}' no encontrado en la base de datos"
            )

        if not modelo.ruta_archivo or not os.path.exists(modelo.ruta_archivo):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El archivo del modelo no existe en disco: {modelo.ruta_archivo}"
            )

        bd.query(ModeloIA).update({"activo": False}, synchronize_session=False)
        bd.flush()

        modelo.activo = True
        modelo.fecha_activacion = datetime.now()
        bd.commit()
        bd.refresh(modelo)

        try:
            from app.servicios.gestor_reconocimiento import limpiar_cache
            limpiar_cache()
            logger.info(f"[ACTIVAR] Cache de reconocimiento limpiada tras activar '{modelo.nombre}'")
        except Exception as e:
            logger.warning(f"[ACTIVAR] No se pudo limpiar cache: {e}")

        return RespuestaAPI(
            exito=True,
            mensaje=f"Modelo '{modelo.nombre}' activado exitosamente",
            datos={
                "modelo_id": modelo.id,
                "nombre": modelo.nombre,
                "accuracy": modelo.accuracy,
                "num_clases": modelo.num_clases,
                "activo": modelo.activo,
                "estado_texto": "Activo" if modelo.activo else "Inactivo"
            }
        )

    except HTTPException:
        bd.rollback()
        raise
    except Exception as e:
        bd.rollback()
        logger.error(f"Error activando modelo: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activando modelo: {str(e)}"
        )


@router.post("/datos-entrenamiento/subir", response_model=RespuestaAPI)
async def subir_datos_entrenamiento(
    archivos: List[UploadFile] = File(...),
    categoria: str = "general",
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.gestionar"))
):
    if not re.fullmatch(r"[\w\-]{1,50}", categoria):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Categoria invalida: use solo letras, numeros, guion y guion bajo"
        )

    try:
        rutas_guardadas = []

        for archivo in archivos:
            if not (archivo.content_type or "").startswith('image/'):
                continue

            ruta_relativa, ruta_completa = await archivo_service.subir_imagen_entrenamiento(
                archivo, categoria
            )
            rutas_guardadas.append(ruta_relativa)

        return RespuestaAPI(
            exito=True,
            mensaje=f"Se subieron {len(rutas_guardadas)} archivos para entrenamiento",
            datos={"archivos_subidos": rutas_guardadas}
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al subir datos de entrenamiento: {str(e)}"
        )


@router.get("/reportes/uso", response_model=RespuestaAPI)
def generar_reporte_uso(
    fecha_inicio: str = None,
    fecha_fin: str = None,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.reportes.ver")),
    bd: Session = Depends(obtener_bd)
):
    try:
        from datetime import timedelta

        condiciones_fecha = []

        if fecha_inicio:
            try:
                fecha_inicio_dt = datetime.fromisoformat(fecha_inicio.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Formato de fecha inicio invalido"
                )
            condiciones_fecha.append(SesionEstudio.fecha_inicio >= fecha_inicio_dt)

        if fecha_fin:
            try:
                fecha_fin_dt = datetime.fromisoformat(fecha_fin.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Formato de fecha fin invalido"
                )
            fecha_fin_dt = fecha_fin_dt.replace(hour=23, minute=59, second=59)
            condiciones_fecha.append(SesionEstudio.fecha_inicio <= fecha_fin_dt)

        base = bd.query(SesionEstudio)
        if condiciones_fecha:
            base = base.filter(and_(*condiciones_fecha))

        total_sesiones = base.count()
        sesiones_completadas = base.filter(SesionEstudio.duracion_segundos > 0).count()
        tiempo_total_segundos = base.with_entities(
            func.coalesce(func.sum(SesionEstudio.duracion_segundos), 0)
        ).scalar() or 0
        usuarios_unicos = base.with_entities(
            func.count(func.distinct(SesionEstudio.usuario_id))
        ).scalar() or 0
        tiempo_promedio_segundos = tiempo_total_segundos / total_sesiones if total_sesiones > 0 else 0

        lecciones_populares = base.filter(SesionEstudio.leccion_id.isnot(None)).with_entities(
            SesionEstudio.leccion_id,
            func.count(SesionEstudio.id).label('total_sesiones'),
            func.avg(SesionEstudio.duracion_segundos).label('tiempo_promedio')
        ).group_by(SesionEstudio.leccion_id).order_by(func.count(SesionEstudio.id).desc()).limit(10).all()

        sesiones_por_tipo = base.with_entities(
            SesionEstudio.tipo_sesion,
            func.count(SesionEstudio.id).label('cantidad'),
            func.sum(SesionEstudio.duracion_segundos).label('tiempo_total')
        ).group_by(SesionEstudio.tipo_sesion).all()

        fecha_limite = datetime.now() - timedelta(days=7)
        actividad_por_dia = bd.query(
            func.date(SesionEstudio.fecha_inicio).label('fecha'),
            func.count(SesionEstudio.id).label('sesiones'),
            func.sum(SesionEstudio.duracion_segundos).label('tiempo_total')
        ).filter(
            SesionEstudio.fecha_inicio >= fecha_limite
        ).group_by(
            func.date(SesionEstudio.fecha_inicio)
        ).order_by(
            func.date(SesionEstudio.fecha_inicio).desc()
        ).all()

        consulta_precision = bd.query(func.avg(Progreso.mejor_precision)).filter(
            Progreso.mejor_precision.isnot(None)
        )
        if condiciones_fecha:
            usuarios_periodo = base.with_entities(SesionEstudio.usuario_id).distinct().subquery()
            consulta_precision = consulta_precision.filter(
                Progreso.usuario_id.in_(bd.query(usuarios_periodo.c.usuario_id))
            )
        precision_promedio = consulta_precision.scalar()
        puntuacion_promedio = float(precision_promedio) * 100 if precision_promedio else 0

        reporte = {
            "total_sesiones": total_sesiones,
            "sesiones_completadas": sesiones_completadas,
            "usuarios_activos": usuarios_unicos,
            "tiempo_promedio_sesion": round(tiempo_promedio_segundos, 2),
            "tiempo_total_estudio": tiempo_total_segundos,
            "puntuacion_promedio": round(puntuacion_promedio, 2),
            "lecciones_mas_populares": [
                {
                    "leccion_id": lp.leccion_id,
                    "completadas": lp.total_sesiones,
                    "tiempo_promedio": lp.tiempo_promedio or 0
                }
                for lp in lecciones_populares
            ],
            "sesiones_por_tipo": [
                {
                    "tipo": st.tipo_sesion,
                    "cantidad": st.cantidad,
                    "tiempo_total": st.tiempo_total or 0
                }
                for st in sesiones_por_tipo
            ],
            "actividad_por_dia": [
                {
                    "fecha": str(ad.fecha),
                    "sesiones": ad.sesiones,
                    "tiempo_total": ad.tiempo_total or 0
                }
                for ad in actividad_por_dia
            ],
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "periodo_analizado": f"{fecha_inicio} a {fecha_fin}" if fecha_inicio and fecha_fin else "Todo el periodo"
        }

        logger.info(f"Reporte generado: {total_sesiones} sesiones, {usuarios_unicos} usuarios")

        return RespuestaAPI(exito=True, mensaje="Reporte generado exitosamente", datos=reporte)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al generar reporte: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar reporte: {str(e)}"
        )


@router.delete("/limpiar-archivos-temp", response_model=RespuestaAPI)
def limpiar_archivos_temporales(
    usuario_actual: Usuario = Depends(requiere_permiso("admin.archivos.limpiar"))
):
    try:
        temp_dir = "archivos_subidos/temp"
        if not os.path.exists(temp_dir):
            return RespuestaAPI(exito=True, mensaje="No hay archivos temporales para eliminar")

        archivos_eliminados = 0
        for archivo in os.listdir(temp_dir):
            ruta_archivo = os.path.join(temp_dir, archivo)
            if os.path.isfile(ruta_archivo):
                os.remove(ruta_archivo)
                archivos_eliminados += 1

        return RespuestaAPI(
            exito=True,
            mensaje=f"Se eliminaron {archivos_eliminados} archivos temporales"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al limpiar archivos temporales: {str(e)}"
        )


@router.get("/usuarios/{usuario_id}/progreso-detallado", response_model=RespuestaAPI)
def obtener_progreso_detallado_usuario(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.ver_estadisticas")),
    bd: Session = Depends(obtener_bd)
):
    try:
        usuario = bd.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

        progresos = bd.query(Progreso).join(Leccion).filter(
            Progreso.usuario_id == usuario_id
        ).all()

        progreso_detallado = []
        for progreso in progresos:
            tiempo_empleado = 0
            if progreso.fecha_inicio and progreso.fecha_completada:
                tiempo_empleado = (progreso.fecha_completada - progreso.fecha_inicio).total_seconds() / 60

            progreso_detallado.append({
                "leccion_id": progreso.leccion_id,
                "titulo_leccion": progreso.leccion.titulo if progreso.leccion else "Sin titulo",
                "completado": progreso.completada,
                "puntuacion": round(progreso.mejor_precision * 100, 2) if progreso.mejor_precision else 0,
                "total_puntos": progreso.total_puntos or 0,
                "total_intentos": progreso.total_intentos or 0,
                "tiempo_completado": round(tiempo_empleado, 2),
                "fecha_inicio": _iso(progreso.fecha_inicio),
                "fecha_completado": _iso(progreso.fecha_completada),
                "ultima_practica": _iso(progreso.ultima_practica),
                "estrella_dorada": progreso.tiene_estrella_dorada
            })

        return RespuestaAPI(
            exito=True,
            mensaje="Progreso detallado obtenido exitosamente",
            datos={
                "usuario": {
                    "id": usuario.id,
                    "nombre_completo": usuario.nombre_usuario,
                    "email": usuario.email
                },
                "progreso": progreso_detallado,
                "resumen": {
                    "total_lecciones": len(progreso_detallado),
                    "completadas": sum(1 for p in progreso_detallado if p["completado"]),
                    "puntos_totales": sum(p["total_puntos"] for p in progreso_detallado),
                    "precision_promedio": round(
                        sum(p["puntuacion"] for p in progreso_detallado) / len(progreso_detallado), 2
                    ) if progreso_detallado else 0
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en progreso detallado: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo progreso detallado: {str(e)}"
        )


@router.get("/estadisticas-entrenamiento", response_model=RespuestaAPI)
def obtener_estadisticas_entrenamiento(
    categoria_ids: Optional[List[int]] = Query(None),
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        ids = categoria_ids or _categorias_con_videos(bd)
        estadisticas = dataset_service.obtener_estadisticas_dataset(bd)
        validacion = servicio_entrenamiento.validar_dataset_entrenamiento(ids, bd)

        return RespuestaAPI(
            exito=True,
            mensaje="Estadisticas obtenidas exitosamente",
            datos={
                "estadisticas": estadisticas,
                "validacion": validacion
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estadisticas: {str(e)}"
        )


@router.post("/modelos/activar-multiples")
def activar_multiples_modelos(
    solicitud: ActivarMultiplesRequest,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    try:
        nombres = solicitud.nombres_modelos
        pesos = solicitud.pesos or [1.0] * len(nombres)

        logger.info(f"[ENSEMBLE] Activando: {nombres} con pesos: {pesos}")

        if len(nombres) < 2:
            raise HTTPException(status_code=400, detail="Se requieren al menos 2 modelos para ensemble")

        if len(pesos) != len(nombres):
            raise HTTPException(status_code=400, detail="Cantidad de pesos debe coincidir con modelos")

        modelos = {
            m.nombre: m for m in db.query(ModeloIA).filter(ModeloIA.nombre.in_(nombres)).all()
        }
        no_encontrados = [n for n in nombres if n not in modelos]

        if len(modelos) < 2:
            raise HTTPException(
                status_code=404,
                detail=f"Solo se encontraron {len(modelos)} modelos de {len(nombres)}; no se cambió nada"
            )

        db.query(ModeloIA).update({"activo": False}, synchronize_session=False)

        modelos_activados = []
        for nombre, peso in zip(nombres, pesos):
            modelo = modelos.get(nombre)
            if not modelo:
                continue

            modelo.activo = True
            if hasattr(modelo, 'peso_ensemble'):
                modelo.peso_ensemble = peso
            modelos_activados.append(nombre)

        db.commit()

        try:
            from app.servicios.gestor_reconocimiento import limpiar_cache
            limpiar_cache()
        except Exception as e:
            logger.warning(f"[ENSEMBLE] No se pudo limpiar cache: {e}")

        return {
            "exito": True,
            "mensaje": f"Ensemble activado con {len(modelos_activados)} modelos",
            "datos": {
                "modelos_activados": modelos_activados,
                "modelos_no_encontrados": no_encontrados,
                "total_modelos": len(modelos_activados)
            }
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error activando ensemble: {str(e)}")


@router.post("/modelos/desactivar-todos")
def desactivar_todos_modelos(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    try:
        total = db.query(ModeloIA).filter(ModeloIA.activo == True).count()

        db.query(ModeloIA).update({"activo": False}, synchronize_session=False)
        db.commit()

        try:
            from app.servicios.gestor_reconocimiento import limpiar_cache
            limpiar_cache()
        except Exception as e:
            logger.warning(f"[DESACTIVAR-TODOS] No se pudo limpiar cache: {e}")

        return {
            "exito": True,
            "mensaje": f"Se desactivaron {total} modelos",
            "datos": {"modelos_desactivados": total}
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/modelos/activos/ensemble")
def obtener_config_ensemble(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    try:
        modelos = db.query(ModeloIA).filter(ModeloIA.activo == True).all()

        config = [
            {
                "nombre": modelo.nombre,
                "peso": getattr(modelo, 'peso_ensemble', 1.0),
                "accuracy": modelo.accuracy or 0.0,
                "num_clases": modelo.num_clases or 0,
                "tipo": getattr(modelo, 'tipo_modelo', 'LSTM'),
                "fecha_activacion": _iso(getattr(modelo, "fecha_activacion", None))
            }
            for modelo in modelos
        ]

        return {
            "exito": True,
            "mensaje": f"Ensemble con {len(modelos)} modelos",
            "datos": {
                "total_modelos": len(modelos),
                "modelos": config
            }
        }
    except Exception as e:
        return {
            "exito": False,
            "mensaje": f"Error: {str(e)}",
            "datos": {"total_modelos": 0, "modelos": []}
        }


@router.get("/modelos/{nombre_modelo}/videos")
def obtener_videos_modelo(
    nombre_modelo: str,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

    if not modelo:
        raise HTTPException(status_code=404, detail=f"Modelo '{nombre_modelo}' no encontrado")

    videos = [
        {
            "id": v.id,
            "sena": v.sena,
            "duracion_segundos": v.duracion_segundos,
            "frames_extraidos": v.frames_extraidos,
            "fecha_subida": _iso(v.fecha_subida)
        }
        for v in modelo.videos_entrenamiento
    ]

    return {
        "exito": True,
        "mensaje": f"Videos del modelo '{nombre_modelo}'",
        "datos": {
            "total": len(videos),
            "videos": videos
        }
    }