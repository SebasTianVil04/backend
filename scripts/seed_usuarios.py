import sys
from datetime import date
from app.utilidades.base_datos import SessionLocal
from app.utilidades.seguridad import obtener_hash_password
from app.modelos.usuario import Usuario, RolUsuario


def crear_usuario_si_no_existe(bd, email, password, nombres, apellido_paterno, apellido_materno, rol, dni,
                                telefono=None, direccion=None, fecha_nacimiento=date(2000, 1, 1),
                                tipo_usuario="peruano_mayor"):
    existente = bd.query(Usuario).filter(Usuario.email == email).first()
    if existente:
        print(f"Ya existe: {email} (rol={existente.rol.value})")
        return existente

    usuario = Usuario(
        email=email,
        password_hash=obtener_hash_password(password),
        nombres=nombres,
        apellido_paterno=apellido_paterno,
        apellido_materno=apellido_materno,
        dni=dni,
        rol=rol,
        tipo_usuario=tipo_usuario,
        telefono=telefono,
        direccion=direccion,
        fecha_nacimiento=fecha_nacimiento,
        activo=True,
        verificado=True,
    )
    bd.add(usuario)
    bd.commit()
    bd.refresh(usuario)
    print(f"Creado: {email} ({rol.value})")
    return usuario


def main():
    bd = SessionLocal()
    try:
        crear_usuario_si_no_existe(
            bd,
            email="svilchezviera1704@gmail.com",
            password="17Alexander%",
            nombres="SEBASTIAN ALEXANDER",
            apellido_paterno="VILCHEZ",
            apellido_materno="VIERA",
            rol=RolUsuario.ADMIN,
            dni="76009799",
            telefono="+51940964458",
            direccion="Ah 12 de Octubre 123, Lima, Peru",
            fecha_nacimiento=date(2001, 4, 17),
        )
        crear_usuario_si_no_existe(
            bd,
            email="usuario@signafree.com",
            password="CambiarEstaPassword123!",
            nombres="Usuario",
            apellido_paterno="De",
            apellido_materno="Prueba",
            rol=RolUsuario.USUARIO,
            dni="00000002",
        )
    finally:
        bd.close()


if __name__ == "__main__":
    sys.exit(main())