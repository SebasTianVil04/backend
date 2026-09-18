import logging
import os
import sys
from datetime import date, datetime

from sqlalchemy.orm import Session

from ..modelos.rol import Rol
from ..modelos.usuario import Usuario
from ..utilidades.seguridad import obtener_hash_password

logger = logging.getLogger(__name__)

_MIN_PASSWORD_LEN = 12


def seed_admin(db: Session) -> None:
    email = os.getenv("ADMIN_EMAIL")
    password = os.getenv("ADMIN_PASSWORD")
    nombres = os.getenv("ADMIN_NOMBRES")
    apellido_paterno = os.getenv("ADMIN_APELLIDO_PATERNO")
    apellido_materno = os.getenv("ADMIN_APELLIDO_MATERNO")
    dni = os.getenv("ADMIN_DNI")
    telefono = os.getenv("ADMIN_TELEFONO")
    direccion = os.getenv("ADMIN_DIRECCION")
    fecha_nacimiento_str = os.getenv("ADMIN_FECHA_NACIMIENTO")

    requeridas = {
        "ADMIN_EMAIL": email,
        "ADMIN_PASSWORD": password,
        "ADMIN_NOMBRES": nombres,
        "ADMIN_APELLIDO_PATERNO": apellido_paterno,
        "ADMIN_APELLIDO_MATERNO": apellido_materno,
        "ADMIN_DNI": dni,
    }
    faltantes = [k for k, v in requeridas.items() if not v]
    if faltantes:
        logger.warning(
            "Seed admin omitido: faltan variables de entorno %s. "
            "Configúralas y vuelve a ejecutar el seed si necesitas crear el admin.",
            faltantes,
        )
        return

    if len(password) < _MIN_PASSWORD_LEN:
        logger.error(
            "Seed admin abortado: ADMIN_PASSWORD debe tener al menos %d caracteres.",
            _MIN_PASSWORD_LEN,
        )
        return

    try:
        rol_admin = db.query(Rol).filter(Rol.codigo == "admin").first()
        if not rol_admin:
            raise ValueError(
                "No existe el rol 'admin' en la tabla roles. "
                "Ejecuta seed_roles antes de seed_admin."
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

        fecha_nacimiento = date(1990, 1, 1)
        if fecha_nacimiento_str:
            try:
                fecha_nacimiento = datetime.strptime(fecha_nacimiento_str, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(
                    "ADMIN_FECHA_NACIMIENTO inválida ('%s'), usando valor por defecto",
                    fecha_nacimiento_str,
                )

        nuevo_admin = Usuario(
            email=email_normalizado,
            password_hash=obtener_hash_password(password),
            nombres=nombres.strip(),
            apellido_paterno=apellido_paterno.strip(),
            apellido_materno=apellido_materno.strip(),
            dni=dni.strip(),
            rol_id=rol_admin.id,
            tipo_usuario="peruano_mayor",
            telefono=telefono,
            direccion=direccion,
            fecha_nacimiento=fecha_nacimiento,
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