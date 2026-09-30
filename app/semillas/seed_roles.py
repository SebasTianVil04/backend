import logging
from typing import Dict, List, Union

from sqlalchemy.orm import Session

from ..modelos.rol import Permiso, Rol
from ..utilidades.permisos_catalogo import codigos_validos

logger = logging.getLogger(__name__)

PERMISOS_BASE: List[str] = [
    "lecciones.ver",
    "practicas.registrar",
    "examenes.ver",
    "progreso.ver",
    "categorias.ver",
    "clases.ver",
]

ROLES_SISTEMA: List[Dict[str, Union[str, List[str]]]] = [
    {
        "codigo": "usuario",
        "nombre": "Usuario",
        "descripcion": "Usuario estándar con acceso a lecciones, exámenes y su progreso",
        "permisos": [],
    },
    {
        "codigo": "editor",
        "nombre": "Editor",
        "descripcion": "Gestiona contenido del sistema (categorías, lecciones, clases, tipos)",
        "permisos": [
            "categorias.gestionar_senas",
            "admin.categorias.gestionar",
            "admin.lecciones.gestionar",
            "admin.examenes.gestionar",
            "tipos_categoria.gestionar",
            "clases.crear",
            "clases.editar",
        ],
    },
    {
        "codigo": "entrenador",
        "nombre": "Entrenador",
        "descripcion": "Gestiona el dataset y los modelos IA",
        "permisos": [
            "dataset.ver",
            "dataset.categorias.gestionar",
            "dataset.videos.gestionar",
            "modelos.ver",
            "modelos.gestionar",
            "captura.gestionar",
        ],
    },
    {
        "codigo": "admin",
        "nombre": "Administrador",
        "descripcion": "Acceso total al sistema",
        "permisos": "*",
    },
]


def _resolver_permisos(db: Session, codigos: List[str]) -> List[Permiso]:
    if not codigos:
        return []

    invalidos = set(codigos) - codigos_validos()
    if invalidos:
        logger.warning(
            "seed_roles: códigos NO presentes en el catálogo (revisa permisos_catalogo.py): %s",
            sorted(invalidos),
        )

    return db.query(Permiso).filter(Permiso.codigo.in_(codigos)).all()


def seed_roles(db: Session) -> None:
    try:
        for data in ROLES_SISTEMA:
            rol = db.query(Rol).filter(Rol.codigo == data["codigo"]).first()

            if rol is None:
                rol = Rol(
                    codigo=data["codigo"],
                    nombre=data["nombre"],
                    descripcion=data["descripcion"],
                    es_sistema=True,
                    activo=True,
                )
                db.add(rol)
                db.flush()
                logger.info("Rol '%s' creado", data["codigo"])
            else:
                rol.nombre = data["nombre"]
                rol.descripcion = data["descripcion"]

            if data["permisos"] == "*":
                rol.permisos = db.query(Permiso).all()
                continue

            requeridos = list(dict.fromkeys(PERMISOS_BASE + list(data["permisos"])))
            actuales = {permiso.codigo for permiso in rol.permisos}
            faltantes = [p for p in _resolver_permisos(db, requeridos) if p.codigo not in actuales]

            if faltantes:
                rol.permisos.extend(faltantes)
                logger.info(
                    "Rol '%s': permisos agregados %s",
                    data["codigo"],
                    sorted(p.codigo for p in faltantes),
                )

        db.commit()
        logger.info("Seed roles: %d roles sincronizados", len(ROLES_SISTEMA))

    except Exception as e:
        db.rollback()
        logger.error("Error en seed de roles: %s", e)
        raise