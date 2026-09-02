from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Generator
from .configuracion import configuracion
import time

# Configuración optimizada para PostgreSQL
engine = create_engine(
    configuracion.database_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
    connect_args={
        "connect_timeout": 10,
        "application_name": "signafree_api"
    }
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

Base = declarative_base()

def obtener_bd() -> Generator:
    """
    Dependency para obtener sesión de base de datos
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Error en sesión de BD: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

# CORRECCIÓN: get_db debe ser igual a obtener_bd
get_db = obtener_bd

def crear_tablas():
    """
    Crear todas las tablas en la base de datos
    """
    try:
        print("Importando modelos para crear tablas...")
        
        from app.modelos.usuario import Usuario
        from app.modelos.categoria import Categoria
        from app.modelos.leccion import Leccion
        from app.modelos.clase import Clase
        from app.modelos.progreso import ProgresoClase, ProgresoLeccion
        from app.modelos.practica import Practica
        from app.modelos.examen import Examen, PreguntaExamen, ResultadoExamen  
        from app.modelos.entrenamiento import Entrenamiento, ModeloIA
        from app.modelos.estudio import SesionEstudio
        from app.modelos.dataset import CategoriaDataset, VideoDataset, CalibracionUsuario
        from app.modelos.lstm_video import SignLanguageLSTM , CNNFeatureExtractor , VideoDataset , SimpleLSTM
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
    """Verificar conexión a PostgreSQL"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("Conexión a PostgreSQL verificada")
        return True
    except Exception as e:
        print(f"Error conectando a PostgreSQL: {str(e)}")
        return False