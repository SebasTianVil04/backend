import logging

from sqlalchemy.orm import Session

from ..modelos.rol import Permiso
from ..utilidades.permisos_catalogo import CATALOGO_PERMISOS

logger = logging.getLogger(__name__)


def seed_permisos(db: Session) -> None:
    try:
        existentes = {p.codigo: p for p in db.query(Permiso).all()}
        creados, actualizados = 0, 0

        for codigo, modulo, descripcion in CATALOGO_PERMISOS:
            permiso = existentes.get(codigo)

            if permiso is None:
                permiso = Permiso(codigo=codigo, modulo=modulo, descripcion=descripcion)
                db.add(permiso)
                creados += 1
            elif permiso.modulo != modulo or permiso.descripcion != descripcion:
                permiso.modulo = modulo
                permiso.descripcion = descripcion
                actualizados += 1

        db.commit()

        codigos_catalogo = {c for c, _, _ in CATALOGO_PERMISOS}
        huerfanos = set(existentes.keys()) - codigos_catalogo
        if huerfanos:
            logger.warning(
                "Permisos en BD que ya NO están en el catálogo (revisar manualmente): %s",
                sorted(huerfanos),
            )

        logger.info(
            "Seed permisos completado: %d creados, %d actualizados, %d sin cambios",
            creados, actualizados, len(existentes) - actualizados,
        )

    except Exception as e:
        db.rollback()
        logger.error("Error en seed de permisos: %s", e)
        raise