import logging
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ..modelos.menu import MenuItem

logger = logging.getLogger(__name__)


ITEMS: List[Dict] = [
    {"label": "Inicio", "icon": "🏠", "ruta": "/inicio", "permiso": None, "orden": 1},
    {"label": "Lecciones", "icon": "📚", "ruta": "/lecciones", "permiso": "lecciones.ver", "orden": 2},
    {"label": "Autoevaluación", "icon": "📝", "ruta": "/examenes", "permiso": "examenes.ver", "orden": 3},
    {"label": "Mi Progreso", "icon": "📊", "ruta": "/progreso", "permiso": "usuarios.ver_progreso", "orden": 4},

    {"label": "Panel", "icon": "📊", "ruta": "/admin/panel", "permiso": "admin.dashboard.ver", "orden": 10},
    {
        "label": "Gestión de Contenido", "icon": "📚", "permiso": "admin.categorias.gestionar", "orden": 20,
        "hijos": [
            {"label": "Lecciones", "icon": "📖", "ruta": "/admin/lecciones", "permiso": "admin.lecciones.gestionar", "orden": 21},
            {"label": "Autoevaluación", "icon": "📝", "ruta": "/admin/examenes", "permiso": "admin.examenes.gestionar", "orden": 22},
            {"label": "Categorías", "icon": "🖼️", "ruta": "/admin/categorias", "permiso": "admin.categorias.gestionar", "orden": 23},
            {"label": "Videos", "icon": "🎬", "ruta": "/admin/imagenes", "permiso": "dataset.gestionar", "orden": 24},
        ],
    },

    {"label": "Usuarios", "icon": "👥", "ruta": "/admin/usuarios", "permiso": "admin.usuarios.listar", "orden": 30},
    {"label": "Roles y Permisos", "icon": "🔐", "ruta": "/admin/roles", "permiso": "admin.roles.gestionar", "orden": 31},
    {"label": "Menú", "icon": "🧩", "ruta": "/admin/menu", "permiso": "admin.menu.gestionar", "orden": 32},
    {"label": "Capturar Datos", "icon": "📸", "ruta": "/admin/captura-unificada", "permiso": "captura.gestionar", "orden": 40},

    {
        "label": "Sistema", "icon": "🤖", "orden": 50,
        "hijos": [
            {"label": "Entrenamiento", "icon": "⚡", "ruta": "/admin/entrenamiento", "permiso": "modelos.gestionar", "orden": 51},
            {"label": "Estadísticas", "icon": "📈", "ruta": "/admin/estadisticas", "permiso": "admin.reportes.ver", "orden": 52},
            {"label": "Traductor", "icon": "🌐", "ruta": "/admin/traductor", "permiso": "traductor.usar", "orden": 53},
        ],
    },
]


def _clave(data: Dict, padre_id: Optional[int]) -> tuple:
    return (data["label"], data.get("ruta"), padre_id)


def _sincronizar(db: Session, items: List[Dict], padre_id: Optional[int] = None) -> int:
    creados = 0
    actualizados = 0

    existentes = {
        (mi.label, mi.ruta, mi.padre_id): mi
        for mi in db.query(MenuItem).filter(MenuItem.padre_id == padre_id).all()
    }

    for data in items:
        clave = _clave(data, padre_id)
        item = existentes.get(clave)

        if item is None:
            item = MenuItem(
                label=data["label"],
                icon=data.get("icon"),
                ruta=data.get("ruta"),
                permiso_codigo=data.get("permiso"),
                orden=data.get("orden", 0),
                activo=data.get("activo", True),
                padre_id=padre_id,
            )
            db.add(item)
            db.flush()
            creados += 1
            logger.info("Ítem de menú creado: %s", data["label"])
        else:
            nuevo_permiso = data.get("permiso")
            if item.permiso_codigo != nuevo_permiso:
                logger.info(
                    "Ítem de menú actualizado: %s (permiso: %s -> %s)",
                    data["label"], item.permiso_codigo, nuevo_permiso
                )
                item.permiso_codigo = nuevo_permiso
                actualizados += 1

        creados_hijos, actualizados_hijos = _sincronizar(db, data.get("hijos", []), item.id)
        creados += creados_hijos
        actualizados += actualizados_hijos

    return creados, actualizados


def seed_menu(db: Session) -> None:
    try:
        creados, actualizados = _sincronizar(db, ITEMS)
        db.commit()
        total = db.query(MenuItem).count()
        logger.info(
            "Seed menú completado: %d ítems nuevos, %d actualizados, %d en total",
            creados, actualizados, total
        )

    except Exception as e:
        db.rollback()
        logger.error("Error en seed de menú: %s", e)
        raise