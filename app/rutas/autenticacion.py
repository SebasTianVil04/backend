from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import timedelta, date, datetime
from typing import Optional
from ..utilidades.validaciones import validar_telefono_unico

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import (
    verificar_password, 
    obtener_hash_password, 
    crear_access_token,
    obtener_usuario_actual
)
from ..utilidades.configuracion import configuracion
from ..modelos.usuario import Usuario
from ..modelos.token_recuperacion import TokenRecuperacion
from ..esquemas.usuario_schemas import (
    UsuarioRegistro, 
    UsuarioLogin, 
    TokenRespuesta, 
    UsuarioRespuesta,
    CambiarPassword,
    SolicitudRecuperacion,
    ConfirmarRecuperacion,
    RespuestaRecuperacion,
    VerificarEmailRequest,
    VerificarEmailResponse
)
from ..servicios.api_peru import servicio_api_peru
from ..servicios.email_service import servicio_email
from app.modelos import usuario

router = APIRouter(prefix="/auth", tags=["Autenticación"])

@router.post("/registro", response_model=TokenRespuesta)
async def registrar_usuario(
    datos_usuario: UsuarioRegistro,
    bd: Session = Depends(obtener_bd)
):
    email_normalizado = datos_usuario.email.lower().strip()
    
    usuario_existente = bd.query(Usuario).filter(
        Usuario.email == email_normalizado
    ).first()
    
    if usuario_existente:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El email ya está registrado"
        )
    
    nombres = datos_usuario.nombres
    apellido_paterno = datos_usuario.apellido_paterno
    apellido_materno = datos_usuario.apellido_materno
    
    if datos_usuario.tipo_usuario == "peruano_mayor":
        if datos_usuario.dni:
            dni_existe = bd.query(Usuario).filter(
                Usuario.dni == datos_usuario.dni
            ).first()
            
            if dni_existe:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El DNI ya está registrado"
                )
        
        if datos_usuario.dni:
            try:
                datos_reniec = await servicio_api_peru.consultar_dni(datos_usuario.dni)
                nombres = datos_reniec.nombres
                apellido_paterno = datos_reniec.apellido_paterno
                apellido_materno = datos_reniec.apellido_materno
            except Exception as e:
                pass
    
    elif datos_usuario.tipo_usuario == "extranjero":
        if datos_usuario.pasaporte:
            pasaporte_existe = bd.query(Usuario).filter(
                Usuario.pasaporte == datos_usuario.pasaporte
            ).first()
            
            if pasaporte_existe:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El pasaporte ya está registrado"
                )
    
    if datos_usuario.telefono:
        validar_telefono_unico(
            bd, 
            datos_usuario.telefono, 
            usuario_id=None,
            permitir_duplicados=False
        )
    
    try:
        nuevo_usuario = Usuario(
            tipo_usuario=datos_usuario.tipo_usuario,
            email=email_normalizado,
            password_hash=obtener_hash_password(datos_usuario.password),
            dni=datos_usuario.dni if datos_usuario.tipo_usuario != "extranjero" else None,
            pasaporte=datos_usuario.pasaporte if datos_usuario.tipo_usuario == "extranjero" else None,
            nombres=nombres,
            apellido_paterno=apellido_paterno,
            apellido_materno=apellido_materno,
            telefono=datos_usuario.telefono,
            direccion=datos_usuario.direccion,
            fecha_nacimiento=datos_usuario.fecha_nacimiento,
            activo=True,
            es_admin=False,
            verificado=True
        )
        
        bd.add(nuevo_usuario)
        bd.commit()
        bd.refresh(nuevo_usuario)
        
    except Exception as e:
        bd.rollback()
        
        error_str = str(e).lower()
        if "password" in error_str and ("72 bytes" in error_str or "too long" in error_str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La contraseña es demasiado larga. Debe tener máximo 72 caracteres."
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear usuario: {str(e)}"
        )
    
    access_token_expires = timedelta(minutes=configuracion.access_token_expire_minutes)
    access_token = crear_access_token(
        data={"sub": nuevo_usuario.email},
        expires_delta=access_token_expires
    )
    
    apellidos_completos = f"{nuevo_usuario.apellido_paterno} {nuevo_usuario.apellido_materno}".strip()
    
    usuario_respuesta = {
        "id": nuevo_usuario.id,
        "tipo_usuario": nuevo_usuario.tipo_usuario,
        "email": nuevo_usuario.email,
        "dni": nuevo_usuario.dni,
        "pasaporte": nuevo_usuario.pasaporte,
        "nombres": nuevo_usuario.nombres,
        "apellido_paterno": nuevo_usuario.apellido_paterno,
        "apellido_materno": nuevo_usuario.apellido_materno,
        "apellidos": apellidos_completos,
        "telefono": nuevo_usuario.telefono,
        "fecha_nacimiento": nuevo_usuario.fecha_nacimiento.isoformat() if nuevo_usuario.fecha_nacimiento else None,
        "direccion": nuevo_usuario.direccion,
        "rol": "admin" if nuevo_usuario.es_admin else "usuario",
        "activo": nuevo_usuario.activo,
        "es_admin": nuevo_usuario.es_admin,
        "verificado": nuevo_usuario.verificado,
        "fecha_registro": nuevo_usuario.fecha_creacion.isoformat() if nuevo_usuario.fecha_creacion else None,
        "fecha_creacion": nuevo_usuario.fecha_creacion,
        "nombre_completo": nuevo_usuario.nombre_completo
    }
    
    return TokenRespuesta(
        access_token=access_token,
        token_type="bearer",
        usuario=UsuarioRespuesta(**usuario_respuesta)
    )


@router.post("/login", response_model=TokenRespuesta)
async def login_usuario(
    credenciales: UsuarioLogin,
    bd: Session = Depends(obtener_bd)
):
    email_normalizado = credenciales.email.lower().strip()
    
    usuario = bd.query(Usuario).filter(
        Usuario.email == email_normalizado
    ).first()
    
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos"
        )
    
    if not verificar_password(credenciales.password, usuario.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos"
        )
    
    if not usuario.activo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario desactivado"
        )
    
    access_token_expires = timedelta(minutes=configuracion.access_token_expire_minutes)
    access_token = crear_access_token(
        data={"sub": usuario.email},
        expires_delta=access_token_expires
    )
    
    apellidos_completos = f"{usuario.apellido_paterno} {usuario.apellido_materno}".strip()
    
    usuario_respuesta = {
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
        "rol": "admin" if usuario.es_admin else "usuario",
        "activo": usuario.activo,
        "es_admin": usuario.es_admin,
        "verificado": usuario.verificado,
        "fecha_registro": usuario.fecha_creacion.isoformat() if usuario.fecha_creacion else None,
        "fecha_creacion": usuario.fecha_creacion,
        "nombre_completo": usuario.nombre_completo
    }
    
    return TokenRespuesta(
        access_token=access_token,
        token_type="bearer",
        usuario=UsuarioRespuesta(**usuario_respuesta)
    )


@router.post("/logout")
async def logout_usuario(
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    return {
        "mensaje": "Sesión cerrada exitosamente",
        "usuario": usuario_actual.nombre_completo
    }


@router.put("/cambiar-password")
async def cambiar_password(
    datos: CambiarPassword,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    bd: Session = Depends(obtener_bd)
):
    if not verificar_password(datos.password_actual, usuario_actual.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña actual es incorrecta"
        )
    
    if verificar_password(datos.password_nueva, usuario_actual.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La nueva contraseña debe ser diferente a la contraseña actual"
        )
    
    try:
        usuario_actual.password_hash = obtener_hash_password(datos.password_nueva)
        bd.commit()
        
        return {
            "mensaje": "Contraseña actualizada exitosamente",
            "usuario": usuario_actual.nombre_completo
        }
    except Exception as e:
        bd.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cambiar contraseña: {str(e)}"
        )


@router.post("/solicitar-recuperacion", response_model=RespuestaRecuperacion)
async def solicitar_recuperacion_password(
    datos: SolicitudRecuperacion,
    background_tasks: BackgroundTasks,
    bd: Session = Depends(obtener_bd)
):
    email_normalizado = datos.email.lower().strip()
    
    usuario = bd.query(Usuario).filter(
        Usuario.email == email_normalizado
    ).first()
    
    if not usuario:
        return RespuestaRecuperacion(
            mensaje="Si el email existe, recibirás un enlace de recuperación en unos minutos",
            email=datos.email
        )
    
    if not usuario.activo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario desactivado"
        )
    
    tokens_antiguos = bd.query(TokenRecuperacion).filter(
        TokenRecuperacion.usuario_email == usuario.email,
        TokenRecuperacion.usado == False
    ).all()
    
    for token in tokens_antiguos:
        token.usado = True
    
    nuevo_token = TokenRecuperacion.crear_token(
        usuario_email=usuario.email,
        expire_minutes=configuracion.reset_token_expire_minutes
    )
    
    bd.add(nuevo_token)
    bd.commit()
    bd.refresh(nuevo_token)
    
    background_tasks.add_task(
        servicio_email.enviar_recuperacion_password,
        usuario.email,
        usuario.nombre_completo,
        nuevo_token.token
    )
    
    return RespuestaRecuperacion(
        mensaje="Si el email existe, recibirás un enlace de recuperación en unos minutos",
        email=datos.email
    )


@router.post("/confirmar-recuperacion", response_model=RespuestaRecuperacion)
async def confirmar_recuperacion_password(
    datos: ConfirmarRecuperacion,
    background_tasks: BackgroundTasks,
    bd: Session = Depends(obtener_bd)
):
    try:
        token_recuperacion = bd.query(TokenRecuperacion).filter(
            TokenRecuperacion.token == datos.token,
            TokenRecuperacion.usado == False
        ).first()
        
        if not token_recuperacion:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token inválido o ya usado"
            )
        
        if token_recuperacion.esta_expirado:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El token ha expirado. Solicita uno nuevo"
            )
        
        usuario = bd.query(Usuario).filter(
            Usuario.email == token_recuperacion.usuario_email
        ).first()
        
        if not usuario:
            token_recuperacion.usado = True
            bd.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Usuario no encontrado"
            )
        
        es_misma_contraseña = verificar_password(datos.password_nueva, usuario.password_hash)
        
        if es_misma_contraseña:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña debe ser diferente a la contraseña anterior"
            )
        
        usuario.password_hash = obtener_hash_password(datos.password_nueva)
        token_recuperacion.usado = True
        
        bd.commit()
        
        background_tasks.add_task(
            servicio_email.enviar_notificacion_cambio_password,
            usuario.email,
            usuario.nombre_completo
        )
        
        return RespuestaRecuperacion(
            mensaje="Contraseña actualizada exitosamente. Ya puedes iniciar sesión",
            email=usuario.email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        bd.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado: {str(e)}"
        )

@router.post("/verificar-email", response_model=VerificarEmailResponse)
async def verificar_email_existe(
    datos: VerificarEmailRequest,
    bd: Session = Depends(obtener_bd)
):
    email_normalizado = datos.email.lower().strip()
    
    usuario = bd.query(Usuario).filter(
        Usuario.email == email_normalizado
    ).first()
    
    if not usuario:
        return VerificarEmailResponse(
            existe=False,
            activo=None,
            mensaje="Email no encontrado"
        )
    
    return VerificarEmailResponse(
        existe=True,
        activo=usuario.activo,
        mensaje="Email encontrado y usuario activo" if usuario.activo else "Usuario desactivado"
    )


@router.get("/verificar-token-recuperacion/{token}")
async def verificar_token_recuperacion(
    token: str,
    bd: Session = Depends(obtener_bd)
):
    token_recuperacion = bd.query(TokenRecuperacion).filter(
        TokenRecuperacion.token == token,
        TokenRecuperacion.usado == False
    ).first()
    
    if not token_recuperacion:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token inválido o ya usado"
        )
    
    if token_recuperacion.esta_expirado:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El token ha expirado. Solicita uno nuevo"
        )
    
    usuario = bd.query(Usuario).filter(
        Usuario.email == token_recuperacion.usuario_email
    ).first()
    
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuario no encontrado"
        )
    
    return {
        "valido": True,
        "email": usuario.email,
        "mensaje": "Token válido"
    }