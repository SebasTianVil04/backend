from typing import List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.menu import MenuItem


router = APIRouter(prefix="/auth", tags=["Menú"])


class MenuItemPublico(BaseModel):
    label: str
    icon: Optional[str] = None
    ruta: Optional[str] = None
    hijos: Optional[List["MenuItemPublico"]] = None


def _filtrar(items, permisos: set[str], es_admin: bool):
    resultado = []
    for item in items:
        if not item.activo:
            continue
        if not es_admin and item.permiso_codigo and item.permiso_codigo not in permisos:
            continue
        hijos = _filtrar(item.hijos, permisos, es_admin) if item.hijos else []
        if item.hijos and not hijos:
            continue
        resultado.append(MenuItemPublico(
            label=item.label, icon=item.icon, ruta=item.ruta, hijos=hijos or None,
        ))
    return resultado


@router.get("/menu", response_model=List[MenuItemPublico])
def obtener_menu(
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd),
):
    raices = (
        db.query(MenuItem)
        .filter(MenuItem.padre_id.is_(None), MenuItem.activo == True)
        .order_by(MenuItem.orden)
        .all()
    )
    permisos = set(usuario_actual.permisos or [])
    return _filtrar(raices, permisos, bool(usuario_actual.es_admin))