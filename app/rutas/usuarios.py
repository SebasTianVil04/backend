from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual, obtener_hash_password, verificar_password
from ..modelos.usuario import Usuario
from ..modelos.progreso import ProgresoClase, ProgresoLeccion
from ..modelos.leccion import Leccion
from ..esquemas.usuario_schemas import (
    UsuarioRespuesta,
    UsuarioActualizacion,
    CambiarPasswordRequest
)
from ..esquemas.respuesta_schemas import (
    RespuestaAPI,
    RespuestaProgresoUsuario,
    RespuestaLista
)


router = APIRouter(prefix="/usuarios", tags=["Usuarios"])


@router.get("/perfil", response_model=UsuarioRespuesta)
async def obtener_perfil(
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    return usuario_actual


@router.put("/perfil", response_model=RespuestaAPI)
async def actualizar_mi_perfil(
    datos_actualizacion: UsuarioActualizacion,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd)
):
    try:
        campos_actualizados = []

        if datos_actualizacion.nombre_usuario is not None:
            existente = db.query(Usuario).filter(
                Usuario.nombre_usuario == datos_actualizacion.nombre_usuario,
                Usuario.id != usuario_actual.id
            ).first()
            if existente:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Ese nombre de usuario ya está en uso"
                )
            usuario_actual.nombre_usuario = datos_actualizacion.nombre_usuario
            campos_actualizados.append("nombre_usuario")

        db.commit()
        db.refresh(usuario_actual)

        mensaje = "Perfil actualizado exitosamente"
        if campos_actualizados:
            mensaje += f". Campos actualizados: {', '.join(campos_actualizados)}"

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos=UsuarioRespuesta.model_validate(usuario_actual).model_dump()
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar perfil: {str(e)}"
        )


@router.put("/cambiar-password", response_model=RespuestaAPI)
async def cambiar_password(
    datos: CambiarPasswordRequest,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd)
):
    try:
        if datos.password_actual == datos.password_nueva:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña no puede ser igual a la actual. Por favor, ingresa una contraseña diferente."
            )

        if not verificar_password(datos.password_actual, usuario_actual.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La contraseña actual es incorrecta"
            )

        usuario_actual.password_hash = obtener_hash_password(datos.password_nueva)
        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje="Contraseña actualizada exitosamente",
            datos={"usuario_id": usuario_actual.id}
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cambiar contraseña: {str(e)}"
        )


@router.get("/mi-progreso", response_model=RespuestaProgresoUsuario)
async def obtener_mi_progreso(
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd)
):
    try:
        total_lecciones = db.query(Leccion).filter(Leccion.activa == True).count()

        lecciones_completadas = db.query(ProgresoLeccion).filter(
            and_(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada == True
            )
        ).count()

        puntos_totales = db.query(func.sum(ProgresoLeccion.total_puntos)).filter(
            and_(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada == True
            )
        ).scalar() or 0

        porcentaje_progreso = (lecciones_completadas / total_lecciones * 100) if total_lecciones > 0 else 0

        nivel_actual = 1
        if lecciones_completadas >= 15:
            nivel_actual = 3
        elif lecciones_completadas >= 8:
            nivel_actual = 2

        ultimo_progreso = db.query(ProgresoLeccion).join(Leccion).filter(
            and_(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada == True
            )
        ).order_by(ProgresoLeccion.fecha_completada.desc()).first()

        ultima_leccion = ultimo_progreso.leccion.titulo if ultimo_progreso else None

        proxima_leccion_obj = db.query(Leccion).outerjoin(
            ProgresoLeccion,
            and_(
                ProgresoLeccion.leccion_id == Leccion.id,
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada == True
            )
        ).filter(
            and_(
                Leccion.activa == True,
                ProgresoLeccion.id.is_(None)
            )
        ).order_by(Leccion.orden).first()

        proxima_leccion = proxima_leccion_obj.titulo if proxima_leccion_obj else None

        return RespuestaProgresoUsuario(
            nivel_actual=nivel_actual,
            lecciones_completadas=lecciones_completadas,
            total_lecciones=total_lecciones,
            puntos_totales=int(puntos_totales),
            porcentaje_progreso=round(porcentaje_progreso, 1),
            ultima_leccion=ultima_leccion,
            proxima_leccion=proxima_leccion
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener progreso: {str(e)}"
        )


@router.get("/mis-lecciones", response_model=RespuestaLista)
async def obtener_mis_lecciones(
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd),
    solo_completadas: bool = Query(False, description="Solo mostrar lecciones completadas"),
    pagina: int = Query(1, ge=1),
    por_pagina: int = Query(10, ge=1, le=50)
):
    try:
        query = db.query(Leccion, ProgresoLeccion).outerjoin(
            ProgresoLeccion,
            and_(
                ProgresoLeccion.leccion_id == Leccion.id,
                ProgresoLeccion.usuario_id == usuario_actual.id
            )
        ).filter(Leccion.activa == True)

        if solo_completadas:
            query = query.filter(ProgresoLeccion.completada == True)

        query = query.order_by(Leccion.orden)

        total = query.count()

        offset = (pagina - 1) * por_pagina
        resultados = query.offset(offset).limit(por_pagina).all()

        lecciones_data = []
        for leccion, progreso in resultados:
            leccion_info = {
                "id": leccion.id,
                "titulo": leccion.titulo,
                "descripcion": leccion.descripcion,
                "sena": leccion.sena,
                "orden": leccion.orden,
                "nivel_dificultad": leccion.nivel_dificultad,
                "puntos_base": leccion.puntos_base,
                "puntos_perfecto": leccion.puntos_perfecto,
                "imagen_miniatura": leccion.imagen_miniatura,
                "color_tema": leccion.color_tema,
                "progreso": {
                    "completada": progreso.completada if progreso else False,
                    "total_puntos": progreso.total_puntos if progreso else 0,
                    "mejor_precision": progreso.mejor_precision if progreso else 0.0,
                    "estrellas": progreso.estrellas if progreso else 0,
                    "fecha_completada": progreso.fecha_completada if progreso else None,
                    "total_intentos": progreso.total_intentos if progreso else 0
                }
            }
            lecciones_data.append(leccion_info)

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(lecciones_data)} lecciones",
            datos=lecciones_data,
            total=total,
            pagina=pagina,
            por_pagina=por_pagina
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener lecciones: {str(e)}"
        )