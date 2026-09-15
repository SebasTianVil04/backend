from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Table
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin

modelo_video_association = Table(
    'modelo_video_entrenamiento',
    Base.metadata,
    Column('modelo_id', Integer, ForeignKey('modelos_ia.id', ondelete='CASCADE'), primary_key=True),
    Column('video_id', Integer, ForeignKey('videos_dataset.id', ondelete='CASCADE'), primary_key=True),
    Column('fecha_asociacion', DateTime(timezone=True), server_default=func.now()),
    Column('contribucion_accuracy', Float, nullable=True),
    Column('creado_por_id', Integer, ForeignKey('usuarios.id', ondelete='SET NULL'), nullable=True)
)


class CalibracionUsuario(AuditoriaMixin, Base):
    __tablename__ = "calibraciones_usuario"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id", ondelete='CASCADE'), nullable=False, index=True)
    ajustes_json = Column(Text, nullable=False)
    fecha_calibracion = Column(DateTime(timezone=True), server_default=func.now())
    activa = Column(Boolean, default=True, index=True)

    usuario = relationship("Usuario", back_populates="calibraciones", foreign_keys=[usuario_id])


class CategoriaDataset(AuditoriaMixin, Base):
    __tablename__ = "categorias_dataset"

    id = Column(Integer, primary_key=True, index=True)
    categoria_id = Column(Integer, ForeignKey("categorias.id", ondelete='CASCADE'), nullable=False, index=True)
    nombre = Column(String(100), nullable=False, unique=True, index=True)
    descripcion = Column(Text, nullable=True)
    activa = Column(Boolean, default=True, index=True)
    orden = Column(Integer, default=0)

    categoria_rel = relationship("Categoria", back_populates="dataset_categoria")
    videos = relationship("VideoDataset", back_populates="categoria", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CategoriaDataset(id={self.id}, nombre='{self.nombre}', categoria_id={self.categoria_id})>"


class VideoDataset(AuditoriaMixin, Base):
    __tablename__ = "videos_dataset"

    id = Column(Integer, primary_key=True, index=True)
    categoria_id = Column(Integer, ForeignKey("categorias_dataset.id", ondelete='CASCADE'), nullable=False, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id", ondelete='SET NULL'), nullable=True, index=True)

    sena = Column(String(50), nullable=False, index=True)
    ruta_video = Column(String(500), nullable=True)

    drive_file_id = Column(String(200), nullable=True, index=True)
    drive_url = Column(String(500), nullable=True)

    duracion_segundos = Column(Float, default=0.0)
    fps = Column(Float, default=30.0)
    resolucion = Column(String(100))
    tamaño_bytes = Column(Integer, nullable=True)
    formato = Column(String(10), nullable=True)
    notas = Column(Text, nullable=True)

    procesado = Column(Boolean, default=False, index=True)
    aprobado = Column(Boolean, default=False, index=True)
    rechazado = Column(Boolean, default=False, index=True)
    usado_entrenamiento = Column(Boolean, default=False, index=True)

    frames_extraidos = Column(Integer, default=0)
    calidad_promedio = Column(Float, nullable=True)

    fecha_subida = Column(DateTime(timezone=True), server_default=func.now())
    fecha_procesado = Column(DateTime(timezone=True), nullable=True)
    fecha_aprobado = Column(DateTime(timezone=True), nullable=True)
    fecha_rechazado = Column(DateTime(timezone=True), nullable=True)

    categoria = relationship("CategoriaDataset", back_populates="videos")
    usuario = relationship("Usuario", foreign_keys=[usuario_id])
    modelos_entrenados = relationship(
        "ModeloIA",
        secondary=modelo_video_association,
        back_populates="videos_entrenamiento"
    )

    def __repr__(self):
        return f"<VideoDataset(id={self.id}, sena='{self.sena}', categoria_id={self.categoria_id})>"