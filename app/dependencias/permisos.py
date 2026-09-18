from typing import Optional
from fastapi import Depends, HTTPException, status

from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario


def _codigos_permisos(usuario: Usuario) -> set:
    if not usuario or not usuario.permisos:
        return set()
    codigos = set()
    for p in usuario.permisos:
        if isinstance(p, str):
            codigos.add(p)
        else:
            codigo = getattr(p, "codigo", None)
            if codigo:
                codigos.add(codigo)
    return codigos


def _codigo_rol(usuario: Usuario) -> Optional[str]:
    if not usuario or not usuario.rol:
        return None
    rol = usuario.rol
    if isinstance(rol, str):
        return rol
    return getattr(rol, "codigo", None)


def _verificar_cuenta_activa(usuario: Usuario) -> None:
    if not usuario or not usuario.activo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="La cuenta está inactiva. Contacta a un administrador.",
        )


def requiere_permiso(codigo_permiso: str):
    def dependencia(usuario_actual: Usuario = Depends(obtener_usuario_actual)) -> Usuario:
        _verificar_cuenta_activa(usuario_actual)

        if usuario_actual.es_admin:
            return usuario_actual

        if codigo_permiso not in _codigos_permisos(usuario_actual):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No tienes permiso para realizar esta acción ({codigo_permiso})",
            )
        return usuario_actual

    return dependencia


def requiere_alguno(*codigos_permiso: str):
    def dependencia(usuario_actual: Usuario = Depends(obtener_usuario_actual)) -> Usuario:
        _verificar_cuenta_activa(usuario_actual)

        if usuario_actual.es_admin:
            return usuario_actual

        if not _codigos_permisos(usuario_actual).intersection(codigos_permiso):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No tienes ninguno de los permisos requeridos ({', '.join(codigos_permiso)})",
            )
        return usuario_actual

    return dependencia


def requiere_todos(*codigos_permiso: str):
    def dependencia(usuario_actual: Usuario = Depends(obtener_usuario_actual)) -> Usuario:
        _verificar_cuenta_activa(usuario_actual)

        if usuario_actual.es_admin:
            return usuario_actual

        faltantes = set(codigos_permiso) - _codigos_permisos(usuario_actual)
        if faltantes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Faltan permisos: {', '.join(sorted(faltantes))}",
            )
        return usuario_actual

    return dependencia


def requiere_rol(*codigos_rol: str):
    def dependencia(usuario_actual: Usuario = Depends(obtener_usuario_actual)) -> Usuario:
        _verificar_cuenta_activa(usuario_actual)

        if usuario_actual.es_admin:
            return usuario_actual

        if _codigo_rol(usuario_actual) not in set(codigos_rol):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes el rol necesario para realizar esta acción",
            )
        return usuario_actual

    return dependencia