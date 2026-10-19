import re
import traceback
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .configuracion import configuracion

_CONNECT_ARGS = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}

engine_owner = create_engine(
    configuracion.url_owner,
    pool_pre_ping=True,
    pool_recycle=280,
    pool_size=2,
    max_overflow=2,
    connect_args=_CONNECT_ARGS,
)

engine = create_engine(
    configuracion.url_app,
    pool_pre_ping=True,
    pool_recycle=280,
    pool_size=configuracion.db_pool_size,
    max_overflow=configuracion.db_max_overflow,
    pool_timeout=configuracion.db_pool_timeout,
    connect_args=_CONNECT_ARGS,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)

SessionOwner = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine_owner,
)

Base = declarative_base()

ROL_BD_AUTH = "rol_bd_auth"
ROL_BD_USUARIO = "rol_bd_usuario"
ROL_BD_ADMIN = "rol_bd_admin"
PREFIJO_ROL_APP = "rol_app_"

_PATRON_ROL_BD = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def rol_bd_para(codigo_rol_app: str) -> str:
    if codigo_rol_app == "admin":
        return ROL_BD_ADMIN
    if codigo_rol_app == "usuario":
        return ROL_BD_USUARIO
    limpio = re.sub(r"[^a-z0-9_]", "_", codigo_rol_app.lower())[:50]
    nombre = f"{PREFIJO_ROL_APP}{limpio}"
    if not _PATRON_ROL_BD.match(nombre):
        raise ValueError(f"Código de rol no convertible a rol de BD: {codigo_rol_app!r}")
    return nombre


# set_config('role', ..., true) equivale a SET LOCAL ROLE: dura solo la transacción actual.
_SQL_CONTEXTO = text(
    "SELECT set_config('role', :rol, true), set_config('app.usuario_id', :uid, true)"
)


def _aplicar_contexto(connection, rol_bd: str, usuario_id) -> None:
    connection.execute(
        _SQL_CONTEXTO,
        {"rol": rol_bd, "uid": "" if usuario_id is None else str(usuario_id)},
    )


@event.listens_for(SessionLocal, "after_begin")
def _al_iniciar_transaccion(session, transaction, connection):
    """Reaplica el rol en cada transacción nueva (SET LOCAL se pierde en cada commit)."""
    if not configuracion.rls_activo:
        return
    rol = session.info.get("rol_bd", ROL_BD_AUTH)
    _aplicar_contexto(connection, rol, session.info.get("usuario_id"))


def fijar_contexto_bd(bd, codigo_rol_app: str, usuario_id: int) -> None:
    rol_bd = rol_bd_para(codigo_rol_app)
    bd.info["rol_bd"] = rol_bd
    bd.info["usuario_id"] = usuario_id
    if configuracion.rls_activo:
        _aplicar_contexto(bd.connection(), rol_bd, usuario_id)


@contextmanager
def como_auth(bd):
    """Ejecuta el bloque como rol_bd_auth (sesiones_usuario, tokens_recuperacion, usuarios).

    Al salir, restaura el rol anterior de la sesión.
    """
    previo = bd.info.get("rol_bd", ROL_BD_AUTH)
    uid = bd.info.get("usuario_id")
    bd.info["rol_bd"] = ROL_BD_AUTH
    if configuracion.rls_activo:
        _aplicar_contexto(bd.connection(), ROL_BD_AUTH, uid)
    try:
        yield bd
    finally:
        bd.info["rol_bd"] = previo
        if configuracion.rls_activo:
            try:
                _aplicar_contexto(bd.connection(), previo, uid)
            except Exception:
                bd.rollback()


def obtener_bd():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


get_db = obtener_bd


def crear_tablas():
    try:
        print("Importando modelos para crear tablas...")

        from app.modelos.usuario import Usuario
        from app.modelos.rol import Rol
        from app.modelos.sesion_usuario import SesionUsuario
        from app.modelos.categoria import Categoria
        from app.modelos.leccion import Leccion
        from app.modelos.clase import Clase
        from app.modelos.progreso import ProgresoClase, ProgresoLeccion
        from app.modelos.practica import Practica
        from app.modelos.examen import Examen, PreguntaExamen, ResultadoExamen
        from app.modelos.entrenamiento import Entrenamiento, ModeloIA
        from app.modelos.estudio import SesionEstudio
        from app.modelos.dataset import CategoriaDataset, VideoDataset, CalibracionUsuario
        from app.modelos.tipo_categoria import TipoCategoria
        from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas

        print("Todos los modelos importados correctamente")

        # Corre como signafree_owner (dueño de las tablas)
        Base.metadata.create_all(bind=engine_owner)
        print("Tablas creadas/verificadas exitosamente en PostgreSQL")

    except Exception as e:
        print(f"Error creando tablas: {str(e)}")
        traceback.print_exc()
        raise


def verificar_conexion() -> bool:
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        print("Conexión a PostgreSQL verificada")
        return True
    except Exception as e:
        print(f"Error conectando a PostgreSQL: {str(e)}")
        return False
    finally:
        db.close()