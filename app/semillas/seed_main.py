import logging
import sys

from sqlalchemy.orm import Session

from .seed_admin import seed_admin
from .seed_menu import seed_menu
from .seed_permisos import seed_permisos
from .seed_roles import seed_roles

logger = logging.getLogger(__name__)


def ejecutar_seeds(db: Session) -> None:
    logger.info("Iniciando seeds...")

    seed_permisos(db)
    seed_roles(db)
    seed_admin(db)
    seed_menu(db)

    logger.info("Seeds completados correctamente.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ..utilidades.base_datos import SessionLocal

    bd = SessionLocal()
    try:
        ejecutar_seeds(bd)
    finally:
        bd.close()
    sys.exit(0)
