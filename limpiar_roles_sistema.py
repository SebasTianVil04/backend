import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utilidades.base_datos import SessionLocal
from app.modelos.rol import Rol
from app.modelos.usuario import Usuario

CODIGOS_A_ELIMINAR = ["entrenador", "editor", "soporte_usuarios"]


def main():
    db = SessionLocal()
    try:
        for codigo in CODIGOS_A_ELIMINAR:
            rol = db.query(Rol).filter(Rol.codigo == codigo).first()
            if rol is None:
                print(f"Rol '{codigo}' no existe en la BD, se omite")
                continue

            usuarios_afectados = db.query(Usuario).filter(Usuario.rol.has(codigo=codigo)).count()
            if usuarios_afectados > 0:
                print(f"Rol '{codigo}' tiene {usuarios_afectados} usuario(s) asignado(s), reasígnalos antes de eliminar. Se omite.")
                continue

            rol.permisos = []
            db.delete(rol)
            print(f"Rol '{codigo}' eliminado")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error eliminando roles: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()