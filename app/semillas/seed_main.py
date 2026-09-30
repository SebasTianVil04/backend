import logging

from sqlalchemy.orm import Session

from .seed_admin import seed_admin
from .seed_menu import seed_menu
from .seed_permisos import seed_permisos
from .seed_roles import seed_roles

logger = logging.getLogger(__name__)


def ejecutar_seeds(db: Session) -> None:
    logger.info("=" * 50)
    logger.info("EJECUTANDO SEEDS")
    logger.info("=" * 50)

    logger.info("1/4 - Permisos...")
    seed_permisos(db)

    logger.info("2/4 - Roles...")
    seed_roles(db)

    logger.info("3/4 - Admin principal...")
    seed_admin(db)

    logger.info("4/4 - Menú...")
    seed_menu(db)

    logger.info("=" * 50)
    logger.info("SEEDS COMPLETADOS CORRECTAMENTE")
    logger.info("=" * 50)


def _run_desde_cli() -> None:
    from ..utilidades.base_datos import SessionLocal

    bd = SessionLocal()
    try:
        ejecutar_seeds(bd)
    except Exception as e:
        logger.error("Error ejecutando seeds: %s", e, exc_info=True)
        raise
    finally:
        bd.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _run_desde_cli()