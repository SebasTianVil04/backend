from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import timedelta

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import (
    verificar_password,
    obtener_hash_password,
    crear_access_token,
    obtener_usuario_actual
)
from ..utilidades.configuracion import configuracion
from ..modelos.usuario import Usuario
from ..modelos.rol import Rol
from ..modelos.token_recuperacion import TokenRecuperacion
from ..esquemas.usuario_schemas import (
    UsuarioRegistro,
    UsuarioLogin,
    TokenRespuesta,
    UsuarioRespuesta,
    CambiarPasswordRequest,
    SolicitudRecuperacion,
    ConfirmarRecuperacion,
    RespuestaRecuperacion,
    VerificarEmailRequest,
    VerificarEmailResponse
)
from ..servicios.email_service import servicio_email

router = APIRouter(prefix="/auth", tags=["Autenticación"])


def _construir_usuario_respuesta(usuario: Usuario) -> dict:
    return {
        "id": usuario.id,
        "nombre_usuario": usuario.nombre_usuario,
        "email": usuario.email,
        "rol": usuario.rol,
        "permisos": usuario.permisos,
        "activo": usuario.activo,
        "verificado": usuario.verificado,
        "fecha_creacion": usuario.fecha_creacion,
    }


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

    nombre_usuario_existente = bd.query(Usuario).filter(
        Usuario.nombre_usuario == datos_usuario.nombre_usuario
    ).first()

    if nombre_usuario_existente:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ese nombre de usuario ya está en uso"
        )

    try:
        rol_por_defecto = bd.query(Rol).filter(
            Rol.codigo == "usuario"
        ).first()

        if not rol_por_defecto:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Rol por defecto 'usuario' no está configurado"
            )

        nuevo_usuario = Usuario(
            nombre_usuario=datos_usuario.nombre_usuario,
            email=email_normalizado,
            password_hash=obtener_hash_password(datos_usuario.password),
            rol_id=rol_por_defecto.id,
            activo=True,
            verificado=True
        )

        bd.add(nuevo_usuario)
        bd.commit()
        bd.refresh(nuevo_usuario)

    except HTTPException:
        raise
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

    return TokenRespuesta(
        access_token=access_token,
        token_type="bearer",
        usuario=UsuarioRespuesta(**_construir_usuario_respuesta(nuevo_usuario))
    )


@router.post("/login", response_model=TokenRespuesta)
async def login_usuario(
    credenciales: UsuarioLogin,
    bd: Session = Depends(obtener_bd)
):
    email_normalizado = credenciales.email.lower().strip()

    usuario_db = bd.query(Usuario).filter(
        Usuario.email == email_normalizado
    ).first()

    if not usuario_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos"
        )

    if not verificar_password(credenciales.password, usuario_db.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos"
        )

    if not usuario_db.activo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario desactivado"
        )

    access_token_expires = timedelta(minutes=configuracion.access_token_expire_minutes)
    access_token = crear_access_token(
        data={"sub": usuario_db.email},
        expires_delta=access_token_expires
    )

    return TokenRespuesta(
        access_token=access_token,
        token_type="bearer",
        usuario=UsuarioRespuesta(**_construir_usuario_respuesta(usuario_db))
    )


@router.post("/logout")
async def logout_usuario(
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    return {
        "mensaje": "Sesión cerrada exitosamente",
        "usuario": usuario_actual.nombre_usuario
    }


@router.put("/cambiar-password")
async def cambiar_password(
    datos: CambiarPasswordRequest,
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
            "usuario": usuario_actual.nombre_usuario
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

    usuario_db = bd.query(Usuario).filter(
        Usuario.email == email_normalizado
    ).first()

    if not usuario_db:
        return RespuestaRecuperacion(
            mensaje="Si el email existe, recibirás un enlace de recuperación en unos minutos",
            email=datos.email
        )

    if not usuario_db.activo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario desactivado"
        )

    tokens_antiguos = bd.query(TokenRecuperacion).filter(
        TokenRecuperacion.usuario_email == usuario_db.email,
        TokenRecuperacion.usado == False
    ).all()

    for token in tokens_antiguos:
        token.usado = True

    nuevo_token = TokenRecuperacion.crear_token(
        usuario_email=usuario_db.email,
        expire_minutes=configuracion.reset_token_expire_minutes
    )

    bd.add(nuevo_token)
    bd.commit()
    bd.refresh(nuevo_token)

    background_tasks.add_task(
        servicio_email.enviar_recuperacion_password,
        usuario_db.email,
        usuario_db.nombre_usuario,
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

        usuario_db = bd.query(Usuario).filter(
            Usuario.email == token_recuperacion.usuario_email
        ).first()

        if not usuario_db:
            token_recuperacion.usado = True
            bd.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Usuario no encontrado"
            )

        es_misma_contraseña = verificar_password(datos.password_nueva, usuario_db.password_hash)

        if es_misma_contraseña:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña debe ser diferente a la contraseña anterior"
            )

        usuario_db.password_hash = obtener_hash_password(datos.password_nueva)
        token_recuperacion.usado = True

        bd.commit()

        background_tasks.add_task(
            servicio_email.enviar_notificacion_cambio_password,
            usuario_db.email,
            usuario_db.nombre_usuario
        )

        return RespuestaRecuperacion(
            mensaje="Contraseña actualizada exitosamente. Ya puedes iniciar sesión",
            email=usuario_db.email
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

    usuario_db = bd.query(Usuario).filter(
        Usuario.email == email_normalizado
    ).first()

    if not usuario_db:
        return VerificarEmailResponse(
            existe=False,
            activo=None,
            mensaje="Email no encontrado"
        )

    return VerificarEmailResponse(
        existe=True,
        activo=usuario_db.activo,
        mensaje="Email encontrado y usuario activo" if usuario_db.activo else "Usuario desactivado"
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

    usuario_db = bd.query(Usuario).filter(
        Usuario.email == token_recuperacion.usuario_email
    ).first()

    if not usuario_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuario no encontrado"
        )

    return {
        "valido": True,
        "email": usuario_db.email,
        "mensaje": "Token válido"
    }