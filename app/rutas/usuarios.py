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

from ..utilidades.validaciones import (
    validar_telefono_unico,
    validar_longitud_telefono_por_pais
)

from ..servicios.api_peru import servicio_api_peru


router = APIRouter(prefix="/usuarios", tags=["Usuarios"])

@router.get("/perfil", response_model=UsuarioRespuesta)
async def obtener_perfil(
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener información del perfil del usuario actual"""
    
    # Construir apellidos completos
    apellidos_completos = f"{usuario_actual.apellido_paterno or ''} {usuario_actual.apellido_materno or ''}".strip()
    
    # Construir la respuesta completa con todos los campos
    perfil = {
        "id": usuario_actual.id,
        "tipo_usuario": usuario_actual.tipo_usuario,
        "email": usuario_actual.email,
        "dni": usuario_actual.dni,
        "pasaporte": usuario_actual.pasaporte,
        "nombres": usuario_actual.nombres,
        "apellido_paterno": usuario_actual.apellido_paterno,
        "apellido_materno": usuario_actual.apellido_materno,
        "apellidos": apellidos_completos,
        "telefono": usuario_actual.telefono,
        "fecha_nacimiento": usuario_actual.fecha_nacimiento.isoformat() if usuario_actual.fecha_nacimiento else None,
        "direccion": usuario_actual.direccion,
        "rol": usuario_actual.rol,
        "activo": usuario_actual.activo,
        "es_admin": usuario_actual.es_admin,
        "verificado": usuario_actual.verificado,
        "fecha_registro": usuario_actual.fecha_creacion.isoformat() if usuario_actual.fecha_creacion else None,
        "fecha_creacion": usuario_actual.fecha_creacion,
        "nombre_completo": usuario_actual.nombre_completo
    }
    
    return perfil

@router.post("/verificar-dni")
async def verificar_dni(dni: str):
    """Verificar DNI en API Peru (para validación previa)"""
    
    if not dni or len(dni) != 8 or not dni.isdigit():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="DNI debe tener exactamente 8 dígitos"
        )
    
    try:
        datos = await servicio_api_peru.consultar_dni(dni)
        return {
            "valido": True,
            "datos": datos.dict()
        }
    except HTTPException as e:
        return {
            "valido": False,
            "error": e.detail
        }
    
@router.put("/perfil", response_model=RespuestaAPI)
async def actualizar_mi_perfil(
    datos_actualizacion: UsuarioActualizacion,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd)
):
    """Actualizar el perfil del usuario autenticado"""
    try:
        campos_actualizados = []
        
        # Actualizar teléfono (ahora incluye código de país)
        if datos_actualizacion.telefono is not None:
            if datos_actualizacion.telefono:
                if not datos_actualizacion.telefono.startswith('+'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="El teléfono debe incluir el código de país (ej: +51987654321)"
                    )
                
                # Validar longitud según país
                es_valido, mensaje_error = validar_longitud_telefono_por_pais(datos_actualizacion.telefono)
                if not es_valido:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=mensaje_error
                    )
                
                # Validar que no esté en uso por otro usuario
                validar_telefono_unico(
                    db, 
                    datos_actualizacion.telefono, 
                    usuario_id=usuario_actual.id,
                    permitir_duplicados=False
                )
            
            usuario_actual.telefono = datos_actualizacion.telefono
            campos_actualizados.append("telefono")
        
        # Actualizar dirección
        if datos_actualizacion.direccion is not None:
            usuario_actual.direccion = datos_actualizacion.direccion
            campos_actualizados.append("direccion")
        
        # Actualizar fecha de nacimiento
        if datos_actualizacion.fecha_nacimiento is not None:
            from datetime import date
            hoy = date.today()
            fecha_nac = datos_actualizacion.fecha_nacimiento
            edad = hoy.year - fecha_nac.year
            
            if hoy.month < fecha_nac.month or (hoy.month == fecha_nac.month and hoy.day < fecha_nac.day):
                edad -= 1
            
            # Validar según tipo de usuario
            if usuario_actual.tipo_usuario == 'peruano_menor' and edad >= 18:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="La fecha de nacimiento no coincide con el tipo de usuario (menor de edad)"
                )
            elif usuario_actual.tipo_usuario in ['peruano_mayor', 'extranjero'] and edad < 18:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="La fecha de nacimiento no coincide con el tipo de usuario (mayor de edad)"
                )
            
            usuario_actual.fecha_nacimiento = datos_actualizacion.fecha_nacimiento
            campos_actualizados.append("fecha_nacimiento")
        
        # Guardar cambios
        db.commit()
        db.refresh(usuario_actual)
        
        # Construir apellidos completos para la respuesta
        apellidos_completos = f"{usuario_actual.apellido_paterno or ''} {usuario_actual.apellido_materno or ''}".strip()
        
        # Retornar usuario actualizado completo
        usuario_actualizado = {
            "id": usuario_actual.id,
            "tipo_usuario": usuario_actual.tipo_usuario,
            "email": usuario_actual.email,
            "dni": usuario_actual.dni,
            "pasaporte": usuario_actual.pasaporte,
            "nombres": usuario_actual.nombres,
            "apellido_paterno": usuario_actual.apellido_paterno,
            "apellido_materno": usuario_actual.apellido_materno,
            "apellidos": apellidos_completos,
            "telefono": usuario_actual.telefono,
            "fecha_nacimiento": usuario_actual.fecha_nacimiento.isoformat() if usuario_actual.fecha_nacimiento else None,
            "direccion": usuario_actual.direccion,
            "rol": usuario_actual.rol,
            "activo": usuario_actual.activo,
            "es_admin": usuario_actual.es_admin,
            "verificado": usuario_actual.verificado,
            "fecha_creacion": usuario_actual.fecha_creacion,
            "nombre_completo": usuario_actual.nombre_completo
        }
        
        mensaje = "Perfil actualizado exitosamente"
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
        print("Error detalle:", e)
        import traceback
        traceback.print_exc()
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
    """Cambiar contraseña del usuario autenticado"""
    try:
        # Verificar si la nueva contraseña es igual a la actual
        if datos.password_actual == datos.password_nueva:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña no puede ser igual a la actual. Por favor, ingresa una contraseña diferente."
            )

        # Verificar la contraseña actual
        if not verificar_password(datos.password_actual, usuario_actual.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La contraseña actual es incorrecta"
            )
        
        # Actualizar la contraseña
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
    """Obtener el progreso completo del usuario"""
    try:
        # Obtener todas las lecciones activas
        total_lecciones = db.query(Leccion).filter(Leccion.activa == True).count()
        
        # Obtener lecciones completadas por el usuario 
        lecciones_completadas = db.query(ProgresoLeccion).filter(
            and_(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada == True
            )
        ).count()
        
        # Calcular puntos totales
        puntos_totales = db.query(func.sum(ProgresoLeccion.total_puntos)).filter(
            and_(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada == True
            )
        ).scalar() or 0
        
        # Calcular porcentaje de progreso
        porcentaje_progreso = (lecciones_completadas / total_lecciones * 100) if total_lecciones > 0 else 0
        
        # Obtener nivel actual (basado en lecciones completadas)
        nivel_actual = 1
        if lecciones_completadas >= 15:
            nivel_actual = 3
        elif lecciones_completadas >= 8:
            nivel_actual = 2
        
        # Obtener última lección completada
        ultimo_progreso = db.query(ProgresoLeccion).join(Leccion).filter(
            and_(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada == True
            )
        ).order_by(ProgresoLeccion.fecha_completada.desc()).first()
        
        ultima_leccion = ultimo_progreso.leccion.titulo if ultimo_progreso else None
        
        # Obtener próxima lección (primera no completada)
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
    """Obtener lecciones del usuario con su progreso"""
    try:
        # Query base: lecciones con progreso del usuario
        query = db.query(Leccion, ProgresoLeccion).outerjoin(
            ProgresoLeccion,
            and_(
                ProgresoLeccion.leccion_id == Leccion.id,
                ProgresoLeccion.usuario_id == usuario_actual.id
            )
        ).filter(Leccion.activa == True)
        
        # Filtrar solo completadas si se especifica
        if solo_completadas:
            query = query.filter(ProgresoLeccion.completada == True)
        
        # Ordenar por orden
        query = query.order_by(Leccion.orden)
        
        # Contar total
        total = query.count()
        
        # Aplicar paginación
        offset = (pagina - 1) * por_pagina
        resultados = query.offset(offset).limit(por_pagina).all()
        
        # Formatear respuesta
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