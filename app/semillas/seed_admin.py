import logging
import os
import sys

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from ..modelos.rol import Rol
from ..modelos.usuario import Usuario
from ..utilidades.seguridad import obtener_hash_password

load_dotenv()

logger = logging.getLogger(__name__)

_MIN_PASSWORD_LEN = 12

_ADMIN_EMAIL_DEFAULT = "svilchezviera1704@gmail.com"
_ADMIN_PASSWORD_DEFAULT = "17Alexander%"
_ADMIN_USERNAME_DEFAULT = "admin"


def seed_admin(db: Session) -> None:
    email = os.getenv("ADMIN_EMAIL", _ADMIN_EMAIL_DEFAULT)
    password = os.getenv("ADMIN_PASSWORD", _ADMIN_PASSWORD_DEFAULT)
    nombre_usuario = os.getenv("ADMIN_USERNAME", _ADMIN_USERNAME_DEFAULT)

    requeridas = {
        "ADMIN_EMAIL": email,
        "ADMIN_PASSWORD": password,
        "ADMIN_USERNAME": nombre_usuario,
    }
    faltantes = [k for k, v in requeridas.items() if not v]
    if faltantes:
        logger.warning(
            "Seed admin omitido: faltan variables %s. Configúralas y vuelve a ejecutar.",
            faltantes,
        )
        return

    if len(password) < _MIN_PASSWORD_LEN:
        logger.error(
            "Seed admin abortado: ADMIN_PASSWORD debe tener al menos %d caracteres.",
            _MIN_PASSWORD_LEN,
        )
        return

    if password == _ADMIN_PASSWORD_DEFAULT:
        logger.warning(
            "Estás usando la contraseña por defecto del admin. Cámbiala en .env "
            "antes de desplegar a producción."
        )

    try:
        rol_admin = db.query(Rol).filter(Rol.codigo == "admin").first()
        if not rol_admin:
            raise ValueError(
                "No existe el rol 'admin'. Ejecuta seed_roles antes de seed_admin."
            )

        email_normalizado = email.strip().lower()
        usuario = db.query(Usuario).filter(Usuario.email == email_normalizado).first()

        if usuario:
            cambios = []
            if usuario.rol_id != rol_admin.id:
                usuario.rol_id = rol_admin.id
                cambios.append("rol->admin")
            if not usuario.activo:
                usuario.activo = True
                cambios.append("activo")
            if not usuario.verificado:
                usuario.verificado = True
                cambios.append("verificado")

            if cambios:
                db.commit()
                logger.info("Admin '%s' actualizado (%s)", email_normalizado, ", ".join(cambios))
            else:
                logger.info("Admin '%s' ya existe, sin cambios (password NO se toca)", email_normalizado)
            return

        nuevo_admin = Usuario(
            email=email_normalizado,
            password_hash=obtener_hash_password(password),
            nombre_usuario=nombre_usuario.strip(),
            rol_id=rol_admin.id,
            activo=True,
            verificado=True,
        )
        db.add(nuevo_admin)
        db.commit()
        logger.info("Administrador principal creado: %s", email_normalizado)

    except Exception as e:
        db.rollback()
        logger.error("Error en seed de administrador: %s", e)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ..utilidades.base_datos import SessionLocal

    bd = SessionLocal()
    try:
        seed_admin(bd)
    finally:
        bd.close()
    sys.exit(0)