from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..dependencias.permisos import requiere_permiso
from ..esquemas.rol_schemas import (
    AsignarPermisosRequest,
    PermisoActualizar,
    PermisoCrear,
    PermisoRespuesta,
    RolActualizar,
    RolCrear,
    RolRespuesta,
)
from ..modelos.rol import Permiso, Rol
from ..utilidades.base_datos import obtener_bd

router = APIRouter(prefix="/api/v1/roles", tags=["Roles"])

ROL_PROTEGIDO = "admin"

MODULOS_RESERVADOS = {"admin_usuarios", "admin_roles", "admin_menu"}


def _sincronizar_rol_bd(db: Session, codigo: str) -> None:
    db.execute(text("SELECT public.sincronizar_rol_bd(:codigo)"), {"codigo": codigo})


def _sincronizar_todos_los_roles_bd(db: Session) -> None:
    db.execute(text("SELECT public.sincronizar_todos_roles_bd()"))


def _eliminar_rol_bd(db: Session, codigo: str) -> None:
    db.execute(text("SELECT public.eliminar_rol_bd(:codigo)"), {"codigo": codigo})


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


@router.post("/permisos", response_model=PermisoRespuesta, status_code=status.HTTP_201_CREATED)
async def crear_permiso(
    datos: PermisoCrear,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    codigo = datos.codigo.strip().lower()
    modulo = datos.modulo.strip().lower()

    if not codigo or not modulo:
        raise HTTPException(status_code=400, detail="El código y el módulo son obligatorios")

    if db.query(Permiso).filter(Permiso.codigo == codigo).first():
        raise HTTPException(status_code=400, detail=f"Ya existe un permiso con código '{codigo}'")

    nuevo = Permiso(
        codigo=codigo,
        modulo=modulo,
        descripcion=datos.descripcion.strip(),
    )
    db.add(nuevo)
    db.commit()
    db.refresh(nuevo)
    return nuevo


@router.put("/permisos/{permiso_id}", response_model=PermisoRespuesta)
async def actualizar_permiso(
    permiso_id: int,
    datos: PermisoActualizar,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    permiso = db.query(Permiso).filter(Permiso.id == permiso_id).first()
    if not permiso:
        raise HTTPException(status_code=404, detail="Permiso no encontrado")

    if datos.modulo is not None:
        permiso.modulo = datos.modulo.strip().lower()
    if datos.descripcion is not None:
        permiso.descripcion = datos.descripcion.strip()

    db.commit()
    db.refresh(permiso)
    return permiso


@router.delete("/permisos/{permiso_id}", status_code=status.HTTP_204_NO_CONTENT)
async def eliminar_permiso(
    permiso_id: int,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    permiso = db.query(Permiso).filter(Permiso.id == permiso_id).first()
    if not permiso:
        raise HTTPException(status_code=404, detail="Permiso no encontrado")

    if permiso.modulo in MODULOS_RESERVADOS:
        raise HTTPException(
            status_code=400,
            detail="No se puede eliminar un permiso de un módulo del sistema",
        )

    roles_afectados = list(getattr(permiso, "roles", []) or [])
    for rol in roles_afectados:
        try:
            rol.permisos.remove(permiso)
        except (ValueError, AttributeError):
            pass

    db.delete(permiso)
    db.flush()
    _sincronizar_todos_los_roles_bd(db)
    db.commit()


@router.post("", response_model=RolRespuesta, status_code=status.HTTP_201_CREATED)
async def crear_rol(
    datos: RolCrear,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.roles.gestionar")),
):
    codigo = datos.codigo.strip().lower()
    nombre = datos.nombre.strip()

    if not codigo or not nombre:
        raise HTTPException(status_code=400, detail="El código y el nombre son obligatorios")

    existente = db.query(Rol).filter(Rol.codigo == codigo).first()
    if existente:
        raise HTTPException(status_code=400, detail="Ya existe un rol con ese código")

    permisos = []
    ids_unicos = set(datos.permisos_ids or [])
    if ids_unicos:
        permisos = db.query(Permiso).filter(Permiso.id.in_(ids_unicos)).all()
        if len(permisos) != len(ids_unicos):
            raise HTTPException(status_code=400, detail="Alguno de los permisos_ids no existe")

    nuevo_rol = Rol(
        codigo=codigo,
        nombre=nombre,
        descripcion=datos.descripcion,
        es_sistema=False,
        activo=True,
        permisos=permisos,
    )
    db.add(nuevo_rol)
    db.flush()
    _sincronizar_rol_bd(db, nuevo_rol.codigo)
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
        nombre = datos.nombre.strip()
        if not nombre:
            raise HTTPException(status_code=400, detail="El nombre no puede estar vacío")
        rol.nombre = nombre
    if datos.descripcion is not None:
        rol.descripcion = datos.descripcion
    if datos.activo is not None and datos.activo != rol.activo:
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

    ids_unicos = set(datos.permisos_ids)
    permisos = []
    if ids_unicos:
        permisos = db.query(Permiso).filter(Permiso.id.in_(ids_unicos)).all()
        if len(permisos) != len(ids_unicos):
            raise HTTPException(status_code=400, detail="Alguno de los permisos_ids no existe")

    rol.permisos = permisos
    db.flush()
    _sincronizar_rol_bd(db, rol.codigo)

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

    codigo = rol.codigo
    db.delete(rol)
    db.flush()
    _eliminar_rol_bd(db, codigo)
    db.commit()