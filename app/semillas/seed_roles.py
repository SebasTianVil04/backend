import logging
from typing import Dict, List, Union

from sqlalchemy.orm import Session

from ..modelos.rol import Rol, Permiso
from ..utilidades.permisos_catalogo import codigos_validos

logger = logging.getLogger(__name__)


ROLES_SISTEMA: List[Dict[str, Union[str, List[str]]]] = [
    {
        "codigo": "usuario",
        "nombre": "Usuario",
        "descripcion": "Usuario estándar con acceso a lecciones, exámenes y su progreso",
        "permisos": [
            "usuarios.ver_perfil",
            "usuarios.editar_perfil",
            "usuarios.ver_progreso",
            "lecciones.ver",
            "clases.ver",
            "examenes.ver",
            "categorias.ver",
            "traductor.usar",
            "reconocimiento.usar",
            "practicas.registrar",
            "practicas.ver_historial",
            "progreso.ver",
            "progreso.registrar",
            "estudio.registrar",
            "estudio.ver",
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
            "seed_roles: códigos de permiso no encontrados en el catálogo (revisa "
            "permisos_catalogo.py, probablemente un typo): %s",
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
            else:
                rol.permisos = _resolver_permisos(db, data["permisos"])

        db.commit()
        logger.info("Seed roles: %d roles sincronizados", len(ROLES_SISTEMA))

    except Exception as e:
        db.rollback()
        logger.error("Error en seed de roles: %s", e)
        raise