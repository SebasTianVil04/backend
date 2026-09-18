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


MenuItemPublico.update_forward_refs()


def _es_ruta_admin(ruta: Optional[str]) -> bool:
    if not ruta:
        return False
    return ruta.startswith("/admin")


def _filtrar(items, permisos, es_admin):
    resultado = []

    for item in items:
        if not item.activo:
            continue

        tiene_hijos = bool(item.hijos)

        if tiene_hijos:
            hijos_filtrados = _filtrar(item.hijos, permisos, es_admin)
            if not hijos_filtrados:
                continue

            resultado.append(MenuItemPublico(
                label=item.label,
                icon=item.icon,
                ruta=item.ruta,
                hijos=hijos_filtrados,
            ))
            continue

        es_ruta_admin_item = _es_ruta_admin(item.ruta)

        if es_admin:
            if item.ruta and not es_ruta_admin_item:
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
    raices = (
        db.query(MenuItem)
        .filter(MenuItem.padre_id.is_(None), MenuItem.activo == True)
        .order_by(MenuItem.orden)
        .all()
    )
    permisos = set(usuario_actual.permisos or [])
    return _filtrar(raices, permisos, bool(usuario_actual.es_admin))