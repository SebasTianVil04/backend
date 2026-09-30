# app/utilidades/seguridad.py
import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, Request, Response, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import or_
from sqlalchemy.orm import Session

from .auditoria import marcar_usuario_actual
from .base_datos import obtener_bd
from .configuracion import configuracion
from ..modelos.usuario import Usuario
from ..modelos.sesion_usuario import SesionUsuario

logger = logging.getLogger("signafree.seguridad")

contexto_password = CryptContext(schemes=["bcrypt"], deprecated="auto")

COOKIE_ACCESS = "access_token"
COOKIE_REFRESH = "refresh_token"
COOKIE_CSRF = "csrf_token"
HEADER_CSRF = "X-CSRF-Token"

GRACIA_ROTACION_SEG = 10  # tolera 2 pestañas refrescando a la vez sin marcarlo como robo

CREDENCIALES_INVALIDAS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Credenciales inválidas",
)


# ======================================================================
# Passwords
# ======================================================================
def _prehash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def obtener_hash_password(password: str) -> str:
    return contexto_password.hash(_prehash_password(password))


def verificar_password(password_plano: str, password_hash: str) -> bool:
    try:
        return contexto_password.verify(_prehash_password(password_plano), password_hash)
    except Exception:
        return False


_HASH_FALSO = contexto_password.hash(_prehash_password("hash-falso-para-igualar-tiempos"))


def verificar_password_falso(password_plano: str) -> None:
    """Gasta el mismo tiempo que una verificación real (evita enumerar usuarios por timing)."""
    verificar_password(password_plano, _HASH_FALSO)


# ======================================================================
# Utilidades internas
# ======================================================================
def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ======================================================================
# Access token (JWT corto, viaja en cookie HttpOnly)
# ======================================================================
def crear_access_token(usuario_id: int, sesion_id: int, expires_delta: Optional[timedelta] = None) -> str:
    ahora = _ahora()
    expira = ahora + (expires_delta or timedelta(minutes=configuracion.access_token_expire_minutes))
    payload = {
        "sub": str(usuario_id),
        "sid": sesion_id,
        "exp": expira,
        "iat": ahora,
        "jti": uuid.uuid4().hex,
        "type": "access",
    }
    return jwt.encode(payload, configuracion.secret_key, algorithm=configuracion.algorithm)


# ======================================================================
# Sesiones (refresh token opaco + rotación + revocación)
# ======================================================================
def crear_sesion(bd: Session, usuario: Usuario, request: Request) -> Tuple[SesionUsuario, str]:
    refresh = secrets.token_urlsafe(48)
    ahora = _ahora()
    absoluta = ahora + timedelta(days=configuracion.sesion_max_dias)
    sesion = SesionUsuario(
        usuario_id=usuario.id,
        refresh_hash=_hash_token(refresh),
        creada_en=ahora,
        ultimo_uso=ahora,
        expira_en=min(ahora + timedelta(days=configuracion.refresh_token_expire_days), absoluta),
        expira_absoluta_en=absoluta,
        revocada=False,
        ip=request.client.host if request.client else None,
        user_agent=(request.headers.get("user-agent") or "")[:255],
    )
    bd.add(sesion)
    bd.commit()
    bd.refresh(sesion)
    return sesion, refresh


def _revocar(bd: Session, sesion: SesionUsuario, motivo: str) -> None:
    sesion.revocada = True
    sesion.revocada_en = _ahora()
    sesion.revocada_motivo = motivo
    bd.commit()


def rotar_sesion(bd: Session, refresh_plano: str) -> Tuple[SesionUsuario, str]:
    """Valida el refresh, lo invalida y entrega uno nuevo (rotación)."""
    h = _hash_token(refresh_plano)
    sesion = bd.query(SesionUsuario).filter(
        or_(SesionUsuario.refresh_hash == h, SesionUsuario.refresh_hash_anterior == h)
    ).first()

    if not sesion or sesion.revocada:
        raise CREDENCIALES_INVALIDAS

    ahora = _ahora()

    if sesion.refresh_hash != h:
        # Se presentó un refresh que ya fue rotado.
        rotada_en = _aware(sesion.rotada_en) if sesion.rotada_en else None
        dentro_de_gracia = rotada_en and (ahora - rotada_en).total_seconds() <= GRACIA_ROTACION_SEG
        if not dentro_de_gracia:
            logger.warning("Reutilización de refresh token: sesion_id=%s usuario_id=%s",
                           sesion.id, sesion.usuario_id)
            _revocar(bd, sesion, "reutilizacion_refresh")
            raise CREDENCIALES_INVALIDAS

    if _aware(sesion.expira_en) <= ahora or _aware(sesion.expira_absoluta_en) <= ahora:
        _revocar(bd, sesion, "expirada")
        raise CREDENCIALES_INVALIDAS

    usuario = bd.get(Usuario, sesion.usuario_id)
    if not usuario or not usuario.activo:
        _revocar(bd, sesion, "usuario_inactivo")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Usuario inactivo")

    nuevo = secrets.token_urlsafe(48)
    sesion.refresh_hash_anterior = sesion.refresh_hash
    sesion.refresh_hash = _hash_token(nuevo)
    sesion.rotada_en = ahora
    sesion.ultimo_uso = ahora
    sesion.expira_en = min(
        ahora + timedelta(days=configuracion.refresh_token_expire_days),
        _aware(sesion.expira_absoluta_en),
    )
    bd.commit()
    return sesion, nuevo


def revocar_sesion_por_refresh(bd: Session, refresh_plano: str, motivo: str = "logout") -> None:
    h = _hash_token(refresh_plano)
    sesion = bd.query(SesionUsuario).filter(
        or_(SesionUsuario.refresh_hash == h, SesionUsuario.refresh_hash_anterior == h)
    ).first()
    if sesion and not sesion.revocada:
        _revocar(bd, sesion, motivo)


def revocar_todas_las_sesiones(
    bd: Session, usuario_id: int, motivo: str, excepto_sesion_id: Optional[int] = None
) -> None:
    q = bd.query(SesionUsuario).filter(
        SesionUsuario.usuario_id == usuario_id,
        SesionUsuario.revocada.is_(False),
    )
    if excepto_sesion_id is not None:
        q = q.filter(SesionUsuario.id != excepto_sesion_id)
    q.update(
        {"revocada": True, "revocada_en": _ahora(), "revocada_motivo": motivo},
        synchronize_session=False,
    )
    bd.commit()


# ======================================================================
# Cookies
# ======================================================================
def establecer_cookies_sesion(response: Response, access_token: str, refresh_token: str) -> None:
    seguras = dict(secure=configuracion.cookie_secure, samesite=configuracion.cookie_samesite)
    max_refresh = configuracion.refresh_token_expire_days * 86400

    # Access: HttpOnly, host-only, viaja a toda la API
    response.set_cookie(
        COOKIE_ACCESS, access_token,
        max_age=configuracion.access_token_expire_minutes * 60,
        httponly=True, path="/", **seguras,
    )
    # Refresh: HttpOnly y SOLO viaja a /auth/* (menos exposición)
    response.set_cookie(
        COOKIE_REFRESH, refresh_token,
        max_age=max_refresh,
        httponly=True, path=configuracion.cookie_refresh_path, **seguras,
    )
    # CSRF: legible por JS a propósito (el front la reenvía en el header X-CSRF-Token)
    response.set_cookie(
        COOKIE_CSRF, secrets.token_urlsafe(32),
        max_age=max_refresh,
        httponly=False, path="/", domain=configuracion.cookie_domain_csrf, **seguras,
    )


def borrar_cookies_sesion(response: Response) -> None:
    seguras = dict(secure=configuracion.cookie_secure, samesite=configuracion.cookie_samesite)
    response.delete_cookie(COOKIE_ACCESS, path="/", httponly=True, **seguras)
    response.delete_cookie(COOKIE_REFRESH, path=configuracion.cookie_refresh_path, httponly=True, **seguras)
    response.delete_cookie(COOKIE_CSRF, path="/", domain=configuracion.cookie_domain_csrf, **seguras)


def iniciar_sesion(bd: Session, usuario: Usuario, request: Request, response: Response) -> SesionUsuario:
    """Crea la sesión en BD y deja las 3 cookies en la respuesta."""
    sesion, refresh = crear_sesion(bd, usuario, request)
    establecer_cookies_sesion(response, crear_access_token(usuario.id, sesion.id), refresh)
    return sesion


# ======================================================================
# CSRF (double-submit + validación de Origin). Registrar de forma global en main.py
# ======================================================================
METODOS_SEGUROS = {"GET", "HEAD", "OPTIONS"}
RUTAS_CSRF_EXENTAS = (
    "/auth/login",
    "/auth/registro",
    "/auth/solicitar-recuperacion",
    "/auth/confirmar-recuperacion",
)


def verificar_csrf(request: Request) -> None:
    if request.method in METODOS_SEGUROS:
        return
    if request.url.path.endswith(RUTAS_CSRF_EXENTAS):
        return
    # Sin cookies de sesión no hay credenciales automáticas que proteger
    if not (request.cookies.get(COOKIE_ACCESS) or request.cookies.get(COOKIE_REFRESH)):
        return

    origen = request.headers.get("origin")
    if origen and origen not in configuracion.origenes_permitidos:
        logger.warning("CSRF: origen no permitido %s", origen)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Origen no permitido")

    cookie = request.cookies.get(COOKIE_CSRF, "")
    header = request.headers.get(HEADER_CSRF, "")
    if not cookie or not header or not hmac.compare_digest(cookie, header):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token CSRF inválido")


# ======================================================================
# Autenticación de cada petición
# ======================================================================
def verificar_token(request: Request) -> dict:
    """Lee el access token de la cookie HttpOnly. Devuelve el payload validado."""
    token = request.cookies.get(COOKIE_ACCESS)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")
    try:
        payload = jwt.decode(token, configuracion.secret_key, algorithms=[configuracion.algorithm])
    except JWTError:
        raise CREDENCIALES_INVALIDAS

    if payload.get("type") != "access":
        raise CREDENCIALES_INVALIDAS
    try:
        payload["_usuario_id"] = int(payload["sub"])
        payload["_sesion_id"] = int(payload["sid"])
    except (KeyError, ValueError, TypeError):
        raise CREDENCIALES_INVALIDAS
    return payload


def obtener_usuario_actual(
    request: Request,
    payload: dict = Depends(verificar_token),
    bd: Session = Depends(obtener_bd),
) -> Usuario:
    sesion = bd.get(SesionUsuario, payload["_sesion_id"])
    if (
        not sesion
        or sesion.revocada
        or sesion.usuario_id != payload["_usuario_id"]
        or _aware(sesion.expira_en) <= _ahora()
    ):
        raise CREDENCIALES_INVALIDAS

    usuario = bd.get(Usuario, payload["_usuario_id"])
    if not usuario:
        logger.warning("Token válido sin usuario asociado: usuario_id=%s", payload["_usuario_id"])
        raise CREDENCIALES_INVALIDAS
    if not usuario.activo:
        logger.warning("Intento de acceso con usuario inactivo: usuario_id=%s", usuario.id)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Usuario inactivo")

    request.state.sesion_id = sesion.id
    marcar_usuario_actual(bd, usuario.id)
    return usuario


# ======================================================================
# Autorización por endpoint (ESTA es la barrera real; los guards del front son solo UX)
# ======================================================================
def requiere_roles(*codigos_rol: str):
    def dependencia(usuario_actual: Usuario = Depends(obtener_usuario_actual)) -> Usuario:
        if not usuario_actual.tiene_rol(*codigos_rol):
            logger.warning(
                "Acceso denegado: usuario_id=%s rol=%s roles_requeridos=%s",
                usuario_actual.id,
                usuario_actual.rol.codigo if usuario_actual.rol else None,
                codigos_rol,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos suficientes",
            )
        return usuario_actual
    return dependencia


def requiere_permiso(*codigos_permiso: str):
    def dependencia(usuario_actual: Usuario = Depends(obtener_usuario_actual)) -> Usuario:
        if not any(usuario_actual.tiene_permiso(codigo) for codigo in codigos_permiso):
            logger.warning(
                "Acceso denegado por permiso: usuario_id=%s rol=%s permisos_requeridos=%s",
                usuario_actual.id,
                usuario_actual.rol.codigo if usuario_actual.rol else None,
                codigos_permiso,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos suficientes",
            )
        return usuario_actual
    return dependencia


verificar_admin = requiere_roles("admin")
verificar_token_admin = verificar_admin