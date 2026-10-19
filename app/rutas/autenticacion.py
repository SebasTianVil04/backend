import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload, undefer

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import (
    COOKIE_REFRESH,
    borrar_cookies_sesion,
    crear_access_token,
    establecer_cookies_sesion,
    iniciar_sesion,
    obtener_hash_password,
    obtener_usuario_actual,
    revocar_sesion_por_refresh,
    revocar_todas_las_sesiones,
    rotar_sesion,
    verificar_password,
    verificar_password_falso,
)
from ..utilidades.configuracion import configuracion
from ..modelos.usuario import Usuario
from ..modelos.rol import Rol
from ..modelos.token_recuperacion import TokenRecuperacion
from ..esquemas.usuario_schemas import (
    UsuarioRegistro,
    UsuarioLogin,
    SesionRespuesta,
    UsuarioRespuesta,
    CambiarPasswordRequest,
    SolicitudRecuperacion,
    ConfirmarRecuperacion,
    RespuestaRecuperacion,
)
from ..servicios.email_service import servicio_email

logger = logging.getLogger("signafree.auth")

router = APIRouter(prefix="/auth", tags=["Autenticación"])

ERROR_LOGIN = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Correo o contraseña incorrectos",
)
MENSAJE_RECUPERACION = "Si el email existe, recibirás un enlace de recuperación en unos minutos"


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


def _error_interno(contexto: str, exc: Exception) -> HTTPException:
    logger.exception("%s: %s", contexto, exc)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Ocurrió un error interno. Inténtalo más tarde.",
    )


def _enmascarar_email(email: str) -> str:
    local, _, dominio = email.partition("@")
    return f"{local[:1]}***@{dominio}" if dominio else "***"


@router.post("/registro", response_model=SesionRespuesta)
def registrar_usuario(
    datos_usuario: UsuarioRegistro,
    request: Request,
    response: Response,
    bd: Session = Depends(obtener_bd),
):
    email_normalizado = datos_usuario.email.lower().strip()

    if bd.query(Usuario).filter(Usuario.email == email_normalizado).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El email ya está registrado")

    if bd.query(Usuario).filter(Usuario.nombre_usuario == datos_usuario.nombre_usuario).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ese nombre de usuario ya está en uso")

    try:
        rol_por_defecto = bd.query(Rol).filter(Rol.codigo == "usuario").first()
        if not rol_por_defecto:
            logger.error("Rol por defecto 'usuario' no está configurado")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ocurrió un error interno. Inténtalo más tarde.",
            )

        nuevo_usuario = Usuario(
            nombre_usuario=datos_usuario.nombre_usuario,
            email=email_normalizado,
            password_hash=obtener_hash_password(datos_usuario.password),
            rol_id=rol_por_defecto.id,
            activo=True,
            verificado=True,
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
                detail="La contraseña es demasiado larga. Debe tener máximo 72 caracteres.",
            )
        raise _error_interno("Error al crear usuario", e)

    iniciar_sesion(bd, nuevo_usuario, request, response)
    return SesionRespuesta(usuario=UsuarioRespuesta(**_construir_usuario_respuesta(nuevo_usuario)))


@router.post("/login", response_model=SesionRespuesta)
def login_usuario(
    credenciales: UsuarioLogin,
    request: Request,
    response: Response,
    bd: Session = Depends(obtener_bd),
):
    email_normalizado = credenciales.email.lower().strip()
    # undefer: aquí sí necesitamos password_hash para verificar la contraseña.
    usuario_db = (
        bd.query(Usuario)
        .options(joinedload(Usuario.rol), undefer(Usuario.password_hash))
        .filter(Usuario.email == email_normalizado)
        .first()
    )

    if not usuario_db:
        verificar_password_falso(credenciales.password)
        raise ERROR_LOGIN

    if not verificar_password(credenciales.password, usuario_db.password_hash):
        raise ERROR_LOGIN

    if not usuario_db.activo:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Usuario desactivado")

    iniciar_sesion(bd, usuario_db, request, response)
    return SesionRespuesta(usuario=UsuarioRespuesta(**_construir_usuario_respuesta(usuario_db)))


@router.post("/refresh", status_code=status.HTTP_200_OK, response_class=JSONResponse)
def refrescar_sesion(request: Request, bd: Session = Depends(obtener_bd)):
    refresh = request.cookies.get(COOKIE_REFRESH)
    if not refresh:
        r = JSONResponse(status_code=401, content={"detail": "No autenticado"})
        borrar_cookies_sesion(r)
        return r

    try:
        sesion, nuevo_refresh = rotar_sesion(bd, refresh)
    except HTTPException as exc:
        r = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        borrar_cookies_sesion(r)
        return r

    r = JSONResponse(status_code=status.HTTP_200_OK, content={"ok": True})
    establecer_cookies_sesion(r, crear_access_token(sesion.usuario_id, sesion.id), nuevo_refresh)
    return r


@router.post("/logout", status_code=status.HTTP_200_OK, response_class=JSONResponse)
def logout_usuario(request: Request, bd: Session = Depends(obtener_bd)):
    refresh = request.cookies.get(COOKIE_REFRESH)
    if refresh:
        revocar_sesion_por_refresh(bd, refresh, "logout")

    r = JSONResponse(status_code=status.HTTP_200_OK, content={"ok": True})
    borrar_cookies_sesion(r)
    return r


@router.post("/logout-todas", status_code=status.HTTP_200_OK, response_class=JSONResponse)
def logout_todas_las_sesiones(
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    bd: Session = Depends(obtener_bd),
):
    revocar_todas_las_sesiones(bd, usuario_actual.id, "logout_todas")
    r = JSONResponse(status_code=status.HTTP_200_OK, content={"ok": True})
    borrar_cookies_sesion(r)
    return r


@router.put("/cambiar-password")
def cambiar_password(
    datos: CambiarPasswordRequest,
    request: Request,
    response: Response,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    bd: Session = Depends(obtener_bd),
):
    # Recargamos el usuario con undefer para tener acceso a password_hash.
    # usuario_actual viene de obtener_usuario_actual(), que ya no carga ese campo.
    usuario_db = (
        bd.query(Usuario)
        .options(undefer(Usuario.password_hash))
        .filter(Usuario.id == usuario_actual.id)
        .first()
    )

    if not usuario_db or not verificar_password(datos.password_actual, usuario_db.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La contraseña actual es incorrecta")

    if verificar_password(datos.password_nueva, usuario_db.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La nueva contraseña debe ser diferente a la contraseña actual",
        )

    try:
        usuario_db.password_hash = obtener_hash_password(datos.password_nueva)
        bd.commit()
    except Exception as e:
        bd.rollback()
        raise _error_interno("Error al cambiar contraseña", e)

    revocar_todas_las_sesiones(bd, usuario_db.id, "cambio_password")
    iniciar_sesion(bd, usuario_db, request, response)

    return {"mensaje": "Contraseña actualizada exitosamente", "usuario": usuario_db.nombre_usuario}


@router.post("/solicitar-recuperacion", response_model=RespuestaRecuperacion)
def solicitar_recuperacion_password(
    datos: SolicitudRecuperacion,
    background_tasks: BackgroundTasks,
    bd: Session = Depends(obtener_bd),
):
    email_normalizado = datos.email.lower().strip()
    usuario_db = bd.query(Usuario).filter(Usuario.email == email_normalizado).first()

    if not usuario_db or not usuario_db.activo:
        return RespuestaRecuperacion(mensaje=MENSAJE_RECUPERACION, email=datos.email)

    tokens_antiguos = bd.query(TokenRecuperacion).filter(
        TokenRecuperacion.usuario_email == usuario_db.email,
        TokenRecuperacion.usado == False,  # noqa: E712
    ).all()
    for token in tokens_antiguos:
        token.usado = True

    nuevo_token = TokenRecuperacion.crear_token(
        usuario_email=usuario_db.email,
        expire_minutes=configuracion.reset_token_expire_minutes,
    )
    bd.add(nuevo_token)
    bd.commit()
    bd.refresh(nuevo_token)

    background_tasks.add_task(
        servicio_email.enviar_recuperacion_password,
        usuario_db.email,
        usuario_db.nombre_usuario,
        nuevo_token.token,
    )
    return RespuestaRecuperacion(mensaje=MENSAJE_RECUPERACION, email=datos.email)


@router.post("/confirmar-recuperacion", response_model=RespuestaRecuperacion)
def confirmar_recuperacion_password(
    datos: ConfirmarRecuperacion,
    background_tasks: BackgroundTasks,
    bd: Session = Depends(obtener_bd),
):
    try:
        token_recuperacion = bd.query(TokenRecuperacion).filter(
            TokenRecuperacion.token == datos.token,
            TokenRecuperacion.usado == False,  # noqa: E712
        ).first()

        if not token_recuperacion:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido o ya usado")

        if token_recuperacion.esta_expirado:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El token ha expirado. Solicita uno nuevo",
            )

        # undefer: necesitamos password_hash para comparar con la nueva contraseña.
        usuario_db = (
            bd.query(Usuario)
            .options(undefer(Usuario.password_hash))
            .filter(Usuario.email == token_recuperacion.usuario_email)
            .first()
        )
        if not usuario_db:
            token_recuperacion.usado = True
            bd.commit()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido o ya usado")

        if verificar_password(datos.password_nueva, usuario_db.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña debe ser diferente a la contraseña anterior",
            )

        usuario_db.password_hash = obtener_hash_password(datos.password_nueva)
        token_recuperacion.usado = True
        bd.commit()

        revocar_todas_las_sesiones(bd, usuario_db.id, "recuperacion_password")

        background_tasks.add_task(
            servicio_email.enviar_notificacion_cambio_password,
            usuario_db.email,
            usuario_db.nombre_usuario,
        )
        return RespuestaRecuperacion(
            mensaje="Contraseña actualizada exitosamente. Ya puedes iniciar sesión",
            email=_enmascarar_email(usuario_db.email),
        )

    except HTTPException:
        raise
    except Exception as e:
        bd.rollback()
        raise _error_interno("Error en confirmar-recuperacion", e)


@router.get("/verificar-token-recuperacion/{token}")
def verificar_token_recuperacion(token: str, bd: Session = Depends(obtener_bd)):
    token_recuperacion = bd.query(TokenRecuperacion).filter(
        TokenRecuperacion.token == token,
        TokenRecuperacion.usado == False,  # noqa: E712
    ).first()

    if not token_recuperacion:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido o ya usado")

    if token_recuperacion.esta_expirado:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El token ha expirado. Solicita uno nuevo",
        )

    usuario_db = bd.query(Usuario).filter(Usuario.email == token_recuperacion.usuario_email).first()
    if not usuario_db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido o ya usado")

    return {"valido": True, "email": _enmascarar_email(usuario_db.email), "mensaje": "Token válido"}