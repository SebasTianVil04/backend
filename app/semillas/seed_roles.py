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
        "permisos": PERMISOS_BASE,
    },
    {
        "codigo": "admin",
        "nombre": "Administrador",
        "descripcion": "Acceso total al sistema",
        "permisos": "*",
    },
]

ROLES_LEGADOS: List[str] = ["editor", "entrenador"]


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


def _liberar_roles_legados(db: Session) -> None:
    legados = (
        db.query(Rol)
        .filter(Rol.codigo.in_(ROLES_LEGADOS), Rol.es_sistema.is_(True))
        .all()
    )
    for rol in legados:
        rol.es_sistema = False
        logger.info("Rol '%s' pasa a ser editable (ya no es rol del sistema)", rol.codigo)


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

                if data["permisos"] == "*":
                    rol.permisos = db.query(Permiso).all()
                else:
                    rol.permisos = _resolver_permisos(db, list(data["permisos"]))

                logger.info("Rol '%s' creado con %d permisos", data["codigo"], len(rol.permisos))
                continue

            if not rol.es_sistema:
                rol.es_sistema = True

            if data["permisos"] == "*":
                rol.permisos = db.query(Permiso).all()

        _liberar_roles_legados(db)

        db.commit()
        logger.info("Seed roles: %d roles de sistema sincronizados", len(ROLES_SISTEMA))

    except Exception as e:
        db.rollback()
        logger.error("Error en seed de roles: %s", e)
        raise