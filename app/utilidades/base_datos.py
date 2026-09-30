from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Generator
from .configuracion import configuracion
import time

engine = create_engine(
    configuracion.database_url,
    pool_pre_ping=True,
    pool_recycle=280,
    pool_size=5,
    max_overflow=10,
    connect_args={
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

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

        Base.metadata.create_all(bind=engine)
        print("Tablas creadas/verificadas exitosamente en PostgreSQL")

    except Exception as e:
        print(f"Error creando tablas: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def verificar_conexion() -> bool:
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        print("Conexión a PostgreSQL verificada")
        return True
    except Exception as e:
        print(f"Error conectando a PostgreSQL: {str(e)}")
        return False