from collections import defaultdict
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


MenuItemPublico.model_rebuild()


def _es_ruta_admin(ruta: Optional[str]) -> bool:
    if not ruta:
        return False
    return ruta.startswith("/admin")


def _cargar_hijos_de(db: Session) -> dict:
    items = (
        db.query(MenuItem)
        .filter(MenuItem.activo == True)
        .order_by(MenuItem.orden)
        .all()
    )
    hijos_de = defaultdict(list)
    for item in items:
        hijos_de[item.padre_id].append(item)
    return hijos_de


def _filtrar(items, hijos_de, permisos, es_admin):
    resultado = []

    for item in items:
        hijos_item = hijos_de.get(item.id, [])

        if hijos_item:
            hijos_filtrados = _filtrar(hijos_item, hijos_de, permisos, es_admin)
            if not hijos_filtrados:
                continue
            resultado.append(MenuItemPublico(
                label=item.label,
                icon=item.icon,
                ruta=item.ruta,
                hijos=hijos_filtrados,
            ))
            continue

        if es_admin:
            if item.ruta and not _es_ruta_admin(item.ruta):
                continue
        else:
            if item.permiso_codigo and item.permiso_codigo not in permisos:
                continue

        resultado.append(MenuItemPublico(
            label=item.label,
            icon=item.icon,
            ruta=item.ruta,
            hijos=None,
        ))

    return resultado


@router.get("/menu", response_model=List[MenuItemPublico])
def obtener_menu(
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd),
):
    hijos_de = _cargar_hijos_de(db)
    raices = hijos_de.get(None, [])
    permisos = set(usuario_actual.permisos or [])
    return _filtrar(raices, hijos_de, permisos, bool(usuario_actual.es_admin))