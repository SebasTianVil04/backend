from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..utilidades.base_datos import obtener_bd
from ..dependencias.permisos import requiere_permiso
from ..modelos.menu import MenuItem


router = APIRouter(prefix="/api/v1/menu", tags=["Menú Admin"])


class MenuItemIn(BaseModel):
    label: str
    icon: Optional[str] = None
    ruta: Optional[str] = None
    permiso_codigo: Optional[str] = None
    orden: int = 0
    activo: bool = True
    padre_id: Optional[int] = None


class MenuItemOut(MenuItemIn):
    id: int
    hijos: List["MenuItemOut"] = []

    class Config:
        from_attributes = True


@router.get("", response_model=List[MenuItemOut])
def listar_menu_completo(
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.menu.gestionar")),
):
    return (
        db.query(MenuItem)
        .filter(MenuItem.padre_id.is_(None))
        .order_by(MenuItem.orden)
        .all()
    )


@router.post("", response_model=MenuItemOut, status_code=status.HTTP_201_CREATED)
def crear_item(
    datos: MenuItemIn,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.menu.gestionar")),
):
    item = MenuItem(**datos.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.put("/{item_id}", response_model=MenuItemOut)
def actualizar_item(
    item_id: int,
    datos: MenuItemIn,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.menu.gestionar")),
):
    item = db.query(MenuItem).get(item_id)
    if not item:
        raise HTTPException(404, "Ítem no encontrado")
    for k, v in datos.model_dump().items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_item(
    item_id: int,
    db: Session = Depends(obtener_bd),
    _=Depends(requiere_permiso("admin.menu.gestionar")),
):
    item = db.query(MenuItem).get(item_id)
    if not item:
        raise HTTPException(404, "Ítem no encontrado")
    db.delete(item)
    db.commit()