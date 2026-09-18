# app/utilidades/seguridad.py
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .auditoria import marcar_usuario_actual
from .base_datos import obtener_bd
from .configuracion import configuracion
from ..modelos.usuario import Usuario

logger = logging.getLogger("signafree.seguridad")

contexto_password = CryptContext(schemes=["bcrypt"], deprecated="auto")


class BearerSeguro(HTTPBearer):
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        try:
            return await super().__call__(request)
        except HTTPException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No autenticado",
                headers={"WWW-Authenticate": "Bearer"},
            )


security = BearerSeguro(auto_error=True)

CREDENCIALES_INVALIDAS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Credenciales inválidas",
    headers={"WWW-Authenticate": "Bearer"},
)


def _prehash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def obtener_hash_password(password: str) -> str:
    return contexto_password.hash(_prehash_password(password))


def verificar_password(password_plano: str, password_hash: str) -> bool:
    try:
        return contexto_password.verify(_prehash_password(password_plano), password_hash)
    except Exception:
        return False


def crear_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    ahora = datetime.now(timezone.utc)
    expire = ahora + (
        expires_delta if expires_delta else timedelta(minutes=configuracion.access_token_expire_minutes)
    )
    to_encode.update({
        "exp": expire,
        "iat": ahora,
        "jti": uuid.uuid4().hex,
        "type": "access",
    })
    return jwt.encode(to_encode, configuracion.secret_key, algorithm=configuracion.algorithm)


def _decodificar_token(credentials: HTTPAuthorizationCredentials) -> str:
    try:
        payload = jwt.decode(
            credentials.credentials,
            configuracion.secret_key,
            algorithms=[configuracion.algorithm],
        )
    except JWTError:
        raise CREDENCIALES_INVALIDAS

    if payload.get("type") != "access":
        raise CREDENCIALES_INVALIDAS

    email = payload.get("sub")
    if not email:
        raise CREDENCIALES_INVALIDAS

    return email


def _obtener_usuario_por_email(email: str, bd: Session) -> Usuario:
    usuario = bd.query(Usuario).filter(Usuario.email == email).first()
    if not usuario:
        logger.warning("Token válido para email sin usuario asociado: %s", email)
        raise CREDENCIALES_INVALIDAS

    if not usuario.activo:
        logger.warning("Intento de acceso con usuario inactivo: usuario_id=%s", usuario.id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario inactivo",
        )

    marcar_usuario_actual(bd, usuario.id)
    return usuario


def verificar_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    return _decodificar_token(credentials)


def obtener_usuario_actual(
    email: str = Depends(verificar_token),
    bd: Session = Depends(obtener_bd)
) -> Usuario:
    return _obtener_usuario_por_email(email, bd)


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