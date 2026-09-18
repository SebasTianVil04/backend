from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..dependencias.permisos import requiere_permiso
from ..esquemas.rol_schemas import (
    AsignarPermisosRequest,
    PermisoRespuesta,
    RolActualizar,
    RolCrear,
    RolRespuesta,
)
from ..modelos.rol import Permiso, Rol
from ..utilidades.base_datos import obtener_bd

router = APIRouter(prefix="/api/v1/roles", tags=["Roles"])

ROL_PROTEGIDO = "admin"


@router.get("", response_model=List[RolRespuesta])
async def listar_roles(
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    return db.query(Rol).order_by(Rol.nombre).all()


@router.get("/permisos", response_model=List[PermisoRespuesta])
async def listar_permisos(
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    return db.query(Permiso).order_by(Permiso.modulo, Permiso.codigo).all()


@router.post("", response_model=RolRespuesta, status_code=status.HTTP_201_CREATED)
async def crear_rol(
    datos: RolCrear,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    codigo = datos.codigo.strip().lower()

    existente = db.query(Rol).filter(Rol.codigo == codigo).first()
    if existente:
        raise HTTPException(status_code=400, detail="Ya existe un rol con ese código")

    permisos = []
    if datos.permisos_ids:
        permisos = db.query(Permiso).filter(Permiso.id.in_(datos.permisos_ids)).all()
        if len(permisos) != len(set(datos.permisos_ids)):
            raise HTTPException(status_code=400, detail="Alguno de los permisos_ids no existe")

    nuevo_rol = Rol(
        codigo=codigo,
        nombre=datos.nombre.strip(),
        descripcion=datos.descripcion,
        es_sistema=False,
        activo=True,
        permisos=permisos,
    )
    db.add(nuevo_rol)
    db.commit()
    db.refresh(nuevo_rol)
    return nuevo_rol


@router.put("/{rol_id}", response_model=RolRespuesta)
async def actualizar_rol(
    rol_id: int,
    datos: RolActualizar,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    rol = db.query(Rol).filter(Rol.id == rol_id).first()
    if not rol:
        raise HTTPException(status_code=404, detail="Rol no encontrado")

    if datos.nombre is not None:
        rol.nombre = datos.nombre.strip()
    if datos.descripcion is not None:
        rol.descripcion = datos.descripcion
    if datos.activo is not None:
        if rol.es_sistema:
            raise HTTPException(status_code=400, detail="No se puede desactivar un rol del sistema")
        rol.activo = datos.activo

    db.commit()
    db.refresh(rol)
    return rol


@router.put("/{rol_id}/permisos", response_model=RolRespuesta)
async def asignar_permisos(
    rol_id: int,
    datos: AsignarPermisosRequest,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    rol = db.query(Rol).filter(Rol.id == rol_id).first()
    if not rol:
        raise HTTPException(status_code=404, detail="Rol no encontrado")

    if rol.codigo == ROL_PROTEGIDO:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El rol 'admin' siempre tiene todos los permisos y no se puede editar aquí",
        )

    permisos = db.query(Permiso).filter(Permiso.id.in_(datos.permisos_ids)).all()
    if len(permisos) != len(set(datos.permisos_ids)):
        raise HTTPException(status_code=400, detail="Alguno de los permisos_ids no existe")

    rol.permisos = permisos

    db.commit()
    db.refresh(rol)
    return rol


@router.delete("/{rol_id}", status_code=status.HTTP_204_NO_CONTENT)
async def eliminar_rol(
    rol_id: int,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    rol = db.query(Rol).filter(Rol.id == rol_id).first()
    if not rol:
        raise HTTPException(status_code=404, detail="Rol no encontrado")
    if rol.es_sistema:
        raise HTTPException(status_code=400, detail="No se puede eliminar un rol del sistema")
    if rol.usuarios:
        raise HTTPException(status_code=400, detail="No se puede eliminar un rol que tiene usuarios asignados")

    db.delete(rol)
    db.commit()