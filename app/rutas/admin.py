from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import shutil
from datetime import datetime
import logging
import gc
import time

from app.modelos.dataset import VideoDataset
from app.utilidades.base_datos import obtener_bd
from app.dependencias.permisos import requiere_permiso
from app.modelos.usuario import Usuario
from app.modelos.rol import Rol
from app.modelos.leccion import Leccion
from app.modelos.examen import Examen
from app.modelos.progreso import ProgresoLeccion as Progreso
from app.modelos.entrenamiento import ModeloIA
from app.esquemas.modelo_ia_schemas import ModeloIASchema
from app.servicios.servicio_entrenamiento import servicio_entrenamiento
from app.servicios.archivos import archivo_service
from app.esquemas.usuario_schemas import UsuarioRespuesta, EstadisticasUsuario
from app.esquemas.respuesta_schemas import RespuestaAPI
from app.modelos.estudio import SesionEstudio
from app.utilidades.validaciones import validar_longitud_telefono_por_pais, validar_telefono_unico

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["Administracion"],
    dependencies=[Depends(HTTPBearer())]
)


class ActivarModeloRequest(BaseModel):
    modelo_id: int = None
    nombre_modelo: str = None


@router.get("/dashboard", response_model=Dict[str, Any])
async def obtener_dashboard(
    usuario_actual: Usuario = Depends(requiere_permiso("admin.dashboard.ver")),
    bd: Session = Depends(obtener_bd)
):
    try:
        from sqlalchemy import func
        from datetime import datetime
        from dateutil.relativedelta import relativedelta

        fecha_inicio_mes = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        total_usuarios = bd.query(Usuario).count()
        usuarios_activos = bd.query(Usuario).filter(Usuario.activo == True).count()

        usuarios_nuevos_mes = bd.query(Usuario).filter(
            Usuario.fecha_creacion != None,
            Usuario.fecha_creacion >= fecha_inicio_mes
        ).count()

        total_lecciones = bd.query(Leccion).count()
        lecciones_activas = bd.query(Leccion).filter(Leccion.activa == True).count()

        total_examenes = bd.query(Examen).count()

        from ..modelos.examen import ResultadoExamen
        examenes_completados = bd.query(ResultadoExamen).distinct(
            ResultadoExamen.examen_id,
            ResultadoExamen.usuario_id
        ).count()

        progreso_promedio_query = bd.query(func.avg(Progreso.mejor_precision)).filter(
            Progreso.mejor_precision != None
        ).scalar()

        progreso_promedio = float(progreso_promedio_query * 100) if progreso_promedio_query else 0.0

        resultado = {
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

        return resultado

    except Exception as e:
        logger.error(f"Error en dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadisticas: {str(e)}"
        )


@router.delete("/modelos/{nombre_modelo}", response_model=RespuestaAPI, status_code=200)
async def eliminar_modelo(
    nombre_modelo: str,
    eliminar_videos: bool = Query(False),
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    modelo = None
    try:
        modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

        if not modelo:
            logger.warning(f"[ELIMINAR] Modelo '{nombre_modelo}' no encontrado en BD")
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{nombre_modelo}' no encontrado"
            )

        logger.info(f"[ELIMINAR] Iniciando eliminacion de modelo: {nombre_modelo}")

        ruta_archivo = modelo.ruta_archivo
        archivos_eliminados = []

        if ruta_archivo and os.path.exists(ruta_archivo):
            logger.info(f"[ELIMINAR] Intentando eliminar archivo: {ruta_archivo}")

            db.close()

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass

            max_intentos = 5
            eliminado = False

            for intento in range(max_intentos):
                try:
                    os.remove(ruta_archivo)
                    eliminado = True
                    archivos_eliminados.append(ruta_archivo)
                    logger.info(f"[ELIMINAR] Archivo eliminado exitosamente: {ruta_archivo}")
                    break
                except PermissionError as e:
                    logger.warning(f"[ELIMINAR] PermissionError en intento {intento + 1}/{max_intentos}: {str(e)}")
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except:
                        pass
                    time.sleep(2)
                except OSError as e:
                    if intento < max_intentos - 1:
                        logger.warning(f"[ELIMINAR] OSError en intento {intento + 1}/{max_intentos}: {str(e)}")
                        gc.collect()
                        time.sleep(2)
                    else:
                        logger.error(f"[ELIMINAR] No se pudo eliminar archivo despues de {max_intentos} intentos")
                        break

            db = next(obtener_bd())
            modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

            if not modelo:
                logger.error(f"[ELIMINAR] Modelo desaparecio durante operacion")
                raise HTTPException(
                    status_code=404,
                    detail="Modelo no encontrado tras reabrir sesion"
                )

        videos_eliminados = 0
        errores_videos = []

        if eliminar_videos:
            logger.info(f"[ELIMINAR] Eliminando videos asociados al modelo")

            try:
                if hasattr(modelo, 'videos_entrenamiento') and modelo.videos_entrenamiento:
                    videos = modelo.videos_entrenamiento
                    total_videos = len(videos)
                    logger.info(f"[ELIMINAR] Encontrados {total_videos} videos para eliminar")

                    db.close()
                    gc.collect()

                    for video in videos:
                        try:
                            if video.ruta_video and os.path.exists(video.ruta_video):
                                for intento in range(3):
                                    try:
                                        os.remove(video.ruta_video)
                                        archivos_eliminados.append(video.ruta_video)
                                        videos_eliminados += 1
                                        break
                                    except (PermissionError, OSError):
                                        if intento < 2:
                                            gc.collect()
                                            time.sleep(1)
                                        else:
                                            errores_videos.append(video.id)
                        except Exception as e:
                            logger.warning(f"[ELIMINAR] Error eliminando video {video.id}: {str(e)}")
                            errores_videos.append(video.id)

                    db = next(obtener_bd())

                    videos_bd = db.query(VideoDataset).filter(
                        VideoDataset.id.in_([v.id for v in videos])
                    ).all()

                    for video_bd in videos_bd:
                        try:
                            db.delete(video_bd)
                        except Exception as e:
                            logger.warning(f"[ELIMINAR] Error eliminando registro de video: {str(e)}")
                            errores_videos.append(video_bd.id)

                    db.flush()

                    modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

            except Exception as e:
                logger.error(f"[ELIMINAR] Error en proceso de videos: {str(e)}")
                db = next(obtener_bd())
                modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

        if modelo:
            logger.info(f"[ELIMINAR] Eliminando registro de BD: {nombre_modelo}")
            db.delete(modelo)
            db.commit()
            logger.info(f"[ELIMINAR] Modelo eliminado exitosamente de BD")

        try:
            if hasattr(servicio_entrenamiento, 'limpiar_progreso_entrenamiento'):
                servicio_entrenamiento.limpiar_progreso_entrenamiento(nombre_modelo)
        except Exception as e:
            logger.warning(f"[ELIMINAR] Error limpiando progreso: {str(e)}")

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        mensaje = f"Modelo '{nombre_modelo}' eliminado correctamente"
        if videos_eliminados > 0:
            mensaje += f" junto con {videos_eliminados} videos"

        logger.info(f"[ELIMINAR] {mensaje}")

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

        try:
            if db:
                db.rollback()
        except Exception as rollback_error:
            logger.error(f"[ELIMINAR] Error en rollback: {str(rollback_error)}")

        raise HTTPException(
            status_code=500,
            detail=f"Error al eliminar modelo: {str(e)}"
        )
    finally:
        try:
            if db:
                db.close()
        except:
            pass
        gc.collect()


@router.get("/usuarios", response_model=List[UsuarioRespuesta])
async def listar_usuarios(
    skip: int = 0,
    limit: int = 100,
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

    usuarios_serializados = []
    for usuario in usuarios:
        usuario_dict = {
            "id": usuario.id,
            "tipo_usuario": usuario.tipo_usuario,
            "email": usuario.email,
            "dni": usuario.dni,
            "pasaporte": usuario.pasaporte,
            "nombres": usuario.nombres,
            "apellido_paterno": usuario.apellido_paterno,
            "apellido_materno": usuario.apellido_materno,
            "apellidos": f"{usuario.apellido_paterno} {usuario.apellido_materno}".strip(),
            "telefono": usuario.telefono,
            "fecha_nacimiento": usuario.fecha_nacimiento.isoformat() if usuario.fecha_nacimiento else None,
            "direccion": usuario.direccion,
            "rol": usuario.rol,
            "permisos": usuario.permisos,
            "activo": usuario.activo,
            "verificado": usuario.verificado,
            "fecha_registro": usuario.fecha_creacion.isoformat() if usuario.fecha_creacion else None,
            "fecha_creacion": usuario.fecha_creacion,
            "nombre_completo": usuario.nombre_completo
        }
        usuarios_serializados.append(usuario_dict)

    return usuarios_serializados


@router.get("/usuarios/{usuario_id}", response_model=UsuarioRespuesta)
async def obtener_usuario_por_id(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.ver")),
    db: Session = Depends(obtener_bd)
):
    usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
        )

    apellidos_completos = f"{usuario.apellido_paterno or ''} {usuario.apellido_materno or ''}".strip()

    return {
        "id": usuario.id,
        "tipo_usuario": usuario.tipo_usuario,
        "email": usuario.email,
        "dni": usuario.dni,
        "pasaporte": usuario.pasaporte,
        "nombres": usuario.nombres,
        "apellido_paterno": usuario.apellido_paterno,
        "apellido_materno": usuario.apellido_materno,
        "apellidos": apellidos_completos,
        "telefono": usuario.telefono,
        "fecha_nacimiento": usuario.fecha_nacimiento.isoformat() if usuario.fecha_nacimiento else None,
        "direccion": usuario.direccion,
        "rol": usuario.rol,
        "permisos": usuario.permisos,
        "activo": usuario.activo,
        "verificado": usuario.verificado,
        "fecha_creacion": usuario.fecha_creacion,
        "nombre_completo": usuario.nombre_completo
    }


@router.put("/usuarios/{usuario_id}", response_model=RespuestaAPI)
async def actualizar_usuario(
    usuario_id: int,
    nombres: Optional[str] = None,
    apellido_paterno: Optional[str] = None,
    apellido_materno: Optional[str] = None,
    email: Optional[str] = None,
    telefono: Optional[str] = None,
    direccion: Optional[str] = None,
    fecha_nacimiento: Optional[str] = None,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.editar")),
    db: Session = Depends(obtener_bd)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )

        campos_actualizados = []

        if nombres is not None and nombres.strip():
            usuario.nombres = nombres.strip()
            campos_actualizados.append("nombres")

        if apellido_paterno is not None and apellido_paterno.strip():
            usuario.apellido_paterno = apellido_paterno.strip()
            campos_actualizados.append("apellido_paterno")

        if apellido_materno is not None and apellido_materno.strip():
            usuario.apellido_materno = apellido_materno.strip()
            campos_actualizados.append("apellido_materno")

        if email is not None and email.strip():
            email_lower = email.strip().lower()
            usuario_existente = db.query(Usuario).filter(
                Usuario.email == email_lower,
                Usuario.id != usuario_id
            ).first()

            if usuario_existente:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El email ya está en uso por otro usuario"
                )

            usuario.email = email_lower
            campos_actualizados.append("email")

        if telefono is not None:
            if telefono.strip():
                if not telefono.startswith('+'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="El teléfono debe incluir el código de país (ej: +51987654321)"
                    )

                es_valido, mensaje_error = validar_longitud_telefono_por_pais(telefono)
                if not es_valido:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=mensaje_error
                    )

                validar_telefono_unico(db, telefono, usuario_id=usuario_id, permitir_duplicados=False)

                usuario.telefono = telefono
            else:
                usuario.telefono = None

            campos_actualizados.append("telefono")

        if direccion is not None:
            usuario.direccion = direccion.strip() if direccion.strip() else None
            campos_actualizados.append("direccion")

        if fecha_nacimiento is not None and fecha_nacimiento.strip():
            from datetime import date, datetime as dt
            try:
                fecha_nac = dt.strptime(fecha_nacimiento, "%Y-%m-%d").date()
                hoy = date.today()
                edad = hoy.year - fecha_nac.year

                if hoy.month < fecha_nac.month or (hoy.month == fecha_nac.month and hoy.day < fecha_nac.day):
                    edad -= 1

                if usuario.tipo_usuario == 'peruano_menor' and edad >= 18:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="La fecha de nacimiento no coincide con el tipo de usuario (menor de edad)"
                    )
                elif usuario.tipo_usuario in ['peruano_mayor', 'extranjero'] and edad < 18:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="La fecha de nacimiento no coincide con el tipo de usuario (mayor de edad)"
                    )

                usuario.fecha_nacimiento = fecha_nac
                campos_actualizados.append("fecha_nacimiento")
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Formato de fecha inválido. Use YYYY-MM-DD"
                )

        db.commit()
        db.refresh(usuario)

        apellidos_completos = f"{usuario.apellido_paterno or ''} {usuario.apellido_materno or ''}".strip()

        usuario_actualizado = {
            "id": usuario.id,
            "tipo_usuario": usuario.tipo_usuario,
            "email": usuario.email,
            "dni": usuario.dni,
            "pasaporte": usuario.pasaporte,
            "nombres": usuario.nombres,
            "apellido_paterno": usuario.apellido_paterno,
            "apellido_materno": usuario.apellido_materno,
            "apellidos": apellidos_completos,
            "telefono": usuario.telefono,
            "fecha_nacimiento": usuario.fecha_nacimiento.isoformat() if usuario.fecha_nacimiento else None,
            "direccion": usuario.direccion,
            "rol": usuario.rol.codigo,
            "activo": usuario.activo,
            "verificado": usuario.verificado,
            "fecha_creacion": usuario.fecha_creacion,
            "nombre_completo": usuario.nombre_completo
        }

        mensaje = "Usuario actualizado exitosamente"
        if campos_actualizados:
            mensaje += f". Campos actualizados: {', '.join(campos_actualizados)}"

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos=usuario_actualizado
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar usuario: {str(e)}"
        )


@router.put("/usuarios/{usuario_id}/rol", response_model=RespuestaAPI)
async def asignar_rol(
    usuario_id: int,
    rol_id: int = Query(...),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.asignar_rol")),
    db: Session = Depends(obtener_bd)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )

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
async def eliminar_usuario(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.eliminar")),
    db: Session = Depends(obtener_bd)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )

        if usuario.id == usuario_actual.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No puedes eliminar tu propia cuenta"
            )

        nombre_usuario = usuario.nombre_completo
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
async def obtener_estadisticas_usuario(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.ver_estadisticas")),
    bd: Session = Depends(obtener_bd)
):
    usuario = bd.query(Usuario).filter(Usuario.id == usuario_id).first()
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
        )

    progresos = bd.query(Progreso).filter(Progreso.usuario_id == usuario_id).all()

    from ..modelos.examen import ResultadoExamen

    resultados_examenes = bd.query(ResultadoExamen).filter(
        ResultadoExamen.usuario_id == usuario_id
    ).all()

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
    if progresos:
        ultimas_practicas = [p.ultima_practica for p in progresos if p.ultima_practica]
        if ultimas_practicas:
            ultima_actividad = max(ultimas_practicas).isoformat()

    ultima_leccion_completada = None
    lecciones_completadas = [p for p in progresos if p.completada and p.fecha_completada]
    if lecciones_completadas:
        ultima_leccion = max(lecciones_completadas, key=lambda x: x.fecha_completada)
        ultima_leccion_completada = ultima_leccion.leccion.titulo if ultima_leccion.leccion else None

    estadisticas = {
        "usuario_id": usuario.id,
        "nombre_completo": usuario.nombre_completo,
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
        "fecha_registro": usuario.fecha_creacion.isoformat() if usuario.fecha_creacion else None,
        "ultima_actividad": ultima_actividad,
        "ultima_leccion_completada": ultima_leccion_completada,
        "categorias_dominadas": 0,
        "categoria_favorita": None,
        "dias_consecutivos_activo": 0,
        "logros_desbloqueados": 0,
        "posicion_ranking": None,
        "percentil_rendimiento": None
    }

    return estadisticas


@router.patch("/usuarios/{usuario_id}/estado", response_model=RespuestaAPI)
async def cambiar_estado_usuario(
    usuario_id: int,
    activo: bool,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.cambiar_estado")),
    bd: Session = Depends(obtener_bd)
):
    usuario = bd.query(Usuario).filter(Usuario.id == usuario_id).first()
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Leccion no encontrada"
        )

    try:
        if not archivo.content_type.startswith(('image/', 'video/')):
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al subir contenido: {str(e)}"
        )


@router.post("/modelo/entrenar", response_model=RespuestaAPI)
async def iniciar_entrenamiento_modelo(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        resultado = servicio_entrenamiento.entrenar_modelo(db=bd)
        return RespuestaAPI(
            exito=True,
            mensaje="Entrenamiento iniciado exitosamente",
            datos=resultado
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al iniciar entrenamiento: {str(e)}"
        )


@router.get("/modelo/estado", response_model=RespuestaAPI)
async def obtener_estado_modelo(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        estadisticas = servicio_entrenamiento.obtener_estadisticas_dataset()

        modelos_bd = bd.query(ModeloIA).all()

        modelos_serializados = []
        for modelo in modelos_bd:
            modelo_dict = {
                "id": modelo.id,
                "nombre": modelo.nombre,
                "version": modelo.version or "1.0",
                "descripcion": modelo.descripcion,
                "accuracy": modelo.accuracy,
                "precision": modelo.precision,
                "recall": modelo.recall,
                "f1_score": modelo.f1_score,
                "num_clases": modelo.num_clases,
                "total_imagenes": modelo.total_imagenes,
                "epocas_entrenamiento": modelo.epocas_entrenamiento,
                "activo": modelo.activo,
                "entrenando": modelo.entrenando if hasattr(modelo, 'entrenando') else False,
                "estado_texto": "Activo" if modelo.activo else "Inactivo",
                "calidad": _evaluar_calidad_modelo(modelo.accuracy or 0),
                "accuracy_porcentaje": f"{(modelo.accuracy or 0) * 100:.2f}%",
                "fecha_creacion": modelo.fecha_creacion.isoformat() if modelo.fecha_creacion else None,
                "fecha_entrenamiento": modelo.fecha_entrenamiento.isoformat() if modelo.fecha_entrenamiento else None,
                "fecha_activacion": modelo.fecha_activacion.isoformat() if modelo.fecha_activacion else None,
                "arquitectura": modelo.arquitectura if hasattr(modelo, 'arquitectura') else None,
                "tamaño_mb": modelo.tamaño_mb if hasattr(modelo, 'tamaño_mb') else None,
                "ruta_modelo": modelo.ruta_modelo
            }
            modelos_serializados.append(modelo_dict)

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
async def toggle_modelo(
    request: ActivarModeloRequest,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        modelo = None

        if request.modelo_id:
            modelo = bd.query(ModeloIA).filter(ModeloIA.id == request.modelo_id).first()
        elif request.nombre_modelo:
            modelo = bd.query(ModeloIA).filter(ModeloIA.nombre == request.nombre_modelo).first()

        if not modelo:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")

        if modelo.activo:
            modelo.activo = False
            mensaje = f"Modelo '{modelo.nombre}' desactivado"
        else:
            bd.query(ModeloIA).update({"activo": False})
            bd.flush()
            modelo.activo = True
            modelo.fecha_activacion = datetime.now()
            mensaje = f"Modelo '{modelo.nombre}' activado"

        bd.commit()

        return RespuestaAPI(exito=True, mensaje=mensaje, datos={"activo": modelo.activo})

    except Exception as e:
        bd.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modelos", response_model=RespuestaAPI)
async def listar_modelos(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        modelos_bd = bd.query(ModeloIA).all()

        for m in modelos_bd:
            logger.info(f"Modelo: {m.nombre}, num_clases: {m.num_clases}, clases_json: {m.clases_json}")

        modelos_completos = []

        for modelo_bd in modelos_bd:
            clases = []
            if modelo_bd.clases_json:
                try:
                    import json
                    clases = json.loads(modelo_bd.clases_json)
                except:
                    clases = []

            modelo_completo = {
                "id": modelo_bd.id,
                "nombre": modelo_bd.nombre,
                "version": modelo_bd.version or "1.0",
                "descripcion": modelo_bd.descripcion or "Modelo de reconocimiento de senas",
                "accuracy": modelo_bd.accuracy,
                "loss": modelo_bd.loss,
                "precision": modelo_bd.precision,
                "recall": modelo_bd.recall,
                "f1_score": modelo_bd.f1_score,
                "num_clases": modelo_bd.num_clases,
                "total_imagenes": modelo_bd.total_imagenes,
                "epocas_entrenamiento": modelo_bd.epocas_entrenamiento,
                "activo": modelo_bd.activo,
                "fecha_creacion": modelo_bd.fecha_creacion.isoformat() if modelo_bd.fecha_creacion else None,
                "fecha_entrenamiento": modelo_bd.fecha_entrenamiento.isoformat() if modelo_bd.fecha_entrenamiento else None,
                "fecha_activacion": modelo_bd.fecha_activacion.isoformat() if modelo_bd.fecha_activacion else None,
                "ruta_archivo": modelo_bd.ruta_archivo,
                "arquitectura": modelo_bd.arquitectura or "CNN-Custom",
                "tamaño_mb": modelo_bd.tamaño_mb or 0,
                "parametros": 0,
                "clases": clases,
                "estado": "activo" if modelo_bd.activo else "inactivo",
                "calidad": _evaluar_calidad_modelo(modelo_bd.accuracy or 0),
                "disponible": True
            }

            logger.info(f"Enviando al frontend - num_clases: {modelo_completo['num_clases']}")

            modelos_completos.append(modelo_completo)

        modelos_completos.sort(
            key=lambda x: x.get("fecha_entrenamiento") or "",
            reverse=True
        )

        logger.info(f"Respuesta final con {len(modelos_completos)} modelos")

        return RespuestaAPI(
            exito=True,
            mensaje=f"Se encontraron {len(modelos_completos)} modelos",
            datos=modelos_completos
        )

    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error listando modelos: {str(e)}"
        )


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


@router.post("/modelo/activar", response_model=RespuestaAPI)
async def activar_modelo(
    request: ActivarModeloRequest,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    bd: Session = Depends(obtener_bd)
):
    try:
        modelo = None

        if request.modelo_id:
            modelo = bd.query(ModeloIA).filter(ModeloIA.id == request.modelo_id).first()
        elif request.nombre_modelo:
            modelo = bd.query(ModeloIA).filter(ModeloIA.nombre == request.nombre_modelo).first()

        if not modelo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modelo '{request.nombre_modelo or request.modelo_id}' no encontrado en la base de datos"
            )

        bd.query(ModeloIA).update({"activo": False})
        bd.flush()

        modelo.activo = True
        modelo.fecha_activacion = datetime.now()
        bd.commit()
        bd.refresh(modelo)

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
        raise
    except Exception as e:
        bd.rollback()
        logger.error(f"Error activando modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activando modelo: {str(e)}"
        )


def _crear_modelo_desde_archivo(nombre_modelo: str, bd: Session) -> ModeloIA:
    try:
        info_modelo = servicio_entrenamiento.obtener_metricas_detalladas(nombre_modelo)

        if "error" in info_modelo:
            return None

        modelo = ModeloIA(
            nombre=nombre_modelo,
            version=info_modelo.get("version", "1.0"),
            descripcion=info_modelo.get("descripcion", f"Modelo de reconocimiento de senas"),
            ruta_modelo=info_modelo.get("ruta", ""),
            accuracy=info_modelo.get("accuracy", 0.95),
            precision=info_modelo.get("accuracy", 0.95),
            recall=info_modelo.get("accuracy", 0.95),
            f1_score=info_modelo.get("accuracy", 0.95),
            num_clases=info_modelo.get("num_clases", 26),
            total_imagenes=info_modelo.get("total_imagenes", 0),
            epocas_entrenamiento=info_modelo.get("epochs_entrenadas", 0),
            activo=False,
            fecha_entrenamiento=datetime.now()
        )

        bd.add(modelo)
        bd.flush()

        logger.info(f"Modelo creado desde archivo: {nombre_modelo}")
        return modelo

    except Exception as e:
        logger.error(f"Error creando modelo desde archivo: {e}")
        return None


@router.post("/datos-entrenamiento/subir", response_model=RespuestaAPI)
async def subir_datos_entrenamiento(
    archivos: List[UploadFile] = File(...),
    categoria: str = "general",
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.gestionar"))
):
    try:
        rutas_guardadas = []

        for archivo in archivos:
            if not archivo.content_type.startswith('image/'):
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
async def generar_reporte_uso(
    fecha_inicio: str = None,
    fecha_fin: str = None,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.reportes.ver")),
    bd: Session = Depends(obtener_bd)
):
    try:
        from sqlalchemy import func, and_
        from datetime import datetime, timedelta

        from ..modelos.estudio import SesionEstudio

        query = bd.query(SesionEstudio)

        condiciones_fecha = []
        if fecha_inicio:
            try:
                fecha_inicio_dt = datetime.fromisoformat(fecha_inicio.replace('Z', '+00:00'))
                condiciones_fecha.append(SesionEstudio.fecha_inicio >= fecha_inicio_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Formato de fecha inicio invalido"
                )

        if fecha_fin:
            try:
                fecha_fin_dt = datetime.fromisoformat(fecha_fin.replace('Z', '+00:00'))
                fecha_fin_dt = fecha_fin_dt.replace(hour=23, minute=59, second=59)
                condiciones_fecha.append(SesionEstudio.fecha_inicio <= fecha_fin_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Formato de fecha fin invalido"
                )

        if condiciones_fecha:
            query = query.filter(and_(*condiciones_fecha))

        sesiones = query.all()

        logger.info(f"Sesiones encontradas: {len(sesiones)}")

        total_sesiones = len(sesiones)
        sesiones_completadas = len([s for s in sesiones if s.duracion_segundos > 0])

        tiempo_total_segundos = sum(s.duracion_segundos for s in sesiones if s.duracion_segundos)
        tiempo_promedio_segundos = tiempo_total_segundos / total_sesiones if total_sesiones > 0 else 0

        usuarios_unicos = bd.query(SesionEstudio.usuario_id).filter(
            and_(*condiciones_fecha) if condiciones_fecha else True
        ).distinct().count()

        lecciones_populares = bd.query(
            SesionEstudio.leccion_id,
            func.count(SesionEstudio.id).label('total_sesiones'),
            func.avg(SesionEstudio.duracion_segundos).label('tiempo_promedio')
        ).filter(
            SesionEstudio.leccion_id.isnot(None)
        )

        if condiciones_fecha:
            lecciones_populares = lecciones_populares.filter(and_(*condiciones_fecha))

        lecciones_populares = lecciones_populares.group_by(
            SesionEstudio.leccion_id
        ).order_by(
            func.count(SesionEstudio.id).desc()
        ).limit(10).all()

        sesiones_por_tipo = bd.query(
            SesionEstudio.tipo_sesion,
            func.count(SesionEstudio.id).label('cantidad'),
            func.sum(SesionEstudio.duracion_segundos).label('tiempo_total')
        )

        if condiciones_fecha:
            sesiones_por_tipo = sesiones_por_tipo.filter(and_(*condiciones_fecha))

        sesiones_por_tipo = sesiones_por_tipo.group_by(
            SesionEstudio.tipo_sesion
        ).all()

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

        progresos = bd.query(Progreso).filter(
            Progreso.mejor_precision.isnot(None)
        )

        if condiciones_fecha and sesiones:
            usuarios_con_sesiones = list(set(s.usuario_id for s in sesiones))
            progresos = progresos.filter(Progreso.usuario_id.in_(usuarios_con_sesiones))

        progresos_list = progresos.all()
        precisiones = [p.mejor_precision for p in progresos_list if p.mejor_precision]
        puntuacion_promedio = (sum(precisiones) / len(precisiones) * 100) if precisiones else 0

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
                    "fecha": ad.fecha.isoformat(),
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

        return RespuestaAPI(
            exito=True,
            mensaje="Reporte generado exitosamente",
            datos=reporte
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al generar reporte: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar reporte: {str(e)}"
        )


@router.delete("/limpiar-archivos-temp", response_model=RespuestaAPI)
async def limpiar_archivos_temporales(
    usuario_actual: Usuario = Depends(requiere_permiso("admin.archivos.limpiar"))
):
    try:
        temp_dir = "archivos_subidos/temp"
        if os.path.exists(temp_dir):
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
        else:
            return RespuestaAPI(
                exito=True,
                mensaje="No hay archivos temporales para eliminar"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al limpiar archivos temporales: {str(e)}"
        )


@router.get("/usuarios/{usuario_id}/progreso-detallado", response_model=RespuestaAPI)
async def obtener_progreso_detallado_usuario(
    usuario_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.usuarios.ver_estadisticas")),
    bd: Session = Depends(obtener_bd)
):
    try:
        usuario = bd.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )

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
                "fecha_inicio": progreso.fecha_inicio.isoformat() if progreso.fecha_inicio else None,
                "fecha_completado": progreso.fecha_completada.isoformat() if progreso.fecha_completada else None,
                "ultima_practica": progreso.ultima_practica.isoformat() if progreso.ultima_practica else None,
                "estrella_dorada": progreso.tiene_estrella_dorada
            })

        return RespuestaAPI(
            exito=True,
            mensaje="Progreso detallado obtenido exitosamente",
            datos={
                "usuario": {
                    "id": usuario.id,
                    "nombre_completo": usuario.nombre_completo,
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

    except Exception as e:
        logger.error(f"Error en progreso detallado: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo progreso detallado: {str(e)}"
        )


@router.get("/estadisticas-entrenamiento", response_model=RespuestaAPI)
async def obtener_estadisticas_entrenamiento(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar"))
):
    try:
        estadisticas = servicio_entrenamiento.obtener_estadisticas_dataset()
        validacion = servicio_entrenamiento.validar_dataset_entrenamiento()

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
async def activar_multiples_modelos(
    request: dict,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    try:
        nombres = request.get('nombres_modelos', [])
        pesos = request.get('pesos', [1.0] * len(nombres))

        print(f"[ENSEMBLE] Activando: {nombres} con pesos: {pesos}")

        if len(nombres) < 2:
            raise HTTPException(400, "Se requieren al menos 2 modelos para ensemble")

        if len(pesos) != len(nombres):
            raise HTTPException(400, "Cantidad de pesos debe coincidir con modelos")

        db.query(ModeloIA).update({"activo": False})

        modelos_activados = []
        for nombre, peso in zip(nombres, pesos):
            modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre).first()

            if not modelo:
                continue

            modelo.activo = True
            if hasattr(modelo, 'peso_ensemble'):
                modelo.peso_ensemble = peso
            modelos_activados.append(nombre)

        db.commit()

        return {
            "exito": True,
            "mensaje": f"Ensemble activado con {len(modelos_activados)} modelos",
            "datos": {
                "modelos_activados": modelos_activados,
                "total_modelos": len(modelos_activados)
            }
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Error activando ensemble: {str(e)}")


@router.post("/modelos/desactivar-todos")
async def desactivar_todos_modelos(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    try:
        modelos_activos = db.query(ModeloIA).filter(ModeloIA.activo == True).all()
        total = len(modelos_activos)

        db.query(ModeloIA).update({"activo": False})
        db.commit()

        return {
            "exito": True,
            "mensaje": f"Se desactivaron {total} modelos",
            "datos": {"modelos_desactivados": total}
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Error: {str(e)}")


@router.get("/modelos/activos/ensemble")
async def obtener_config_ensemble(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    try:
        modelos = db.query(ModeloIA).filter(ModeloIA.activo == True).all()

        config = []
        for modelo in modelos:
            config.append({
                "nombre": modelo.nombre,
                "peso": getattr(modelo, 'peso_ensemble', 1.0),
                "accuracy": modelo.accuracy or 0.0,
                "num_clases": modelo.num_clases or 0,
                "tipo": getattr(modelo, 'tipo_modelo', 'CNN'),
                "fecha_activacion": modelo.fecha_activacion.isoformat() if modelo.fecha_activacion else None
            })

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
async def obtener_videos_modelo(
    nombre_modelo: str,
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar")),
    db: Session = Depends(obtener_bd)
):
    modelo = db.query(ModeloIA).filter(ModeloIA.nombre == nombre_modelo).first()

    if not modelo:
        raise HTTPException(404, f"Modelo '{nombre_modelo}' no encontrado")

    videos = [
        {
            "id": v.id,
            "sena": v.sena,
            "duracion_segundos": v.duracion_segundos,
            "frames_extraidos": v.frames_extraidos,
            "fecha_subida": v.fecha_subida.isoformat()
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