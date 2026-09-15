from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin
from .dataset import modelo_video_association


class Entrenamiento(AuditoriaMixin, Base):
    __tablename__ = "entrenamientos"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)

    nombre_sena = Column(String(255), nullable=False)
    descripcion = Column(Text, nullable=True)
    categoria = Column(String(100), nullable=True)

    imagen_url = Column(String(500), nullable=False)
    puntos_mano = Column(JSON, nullable=True)

    procesado = Column(Boolean, default=False)
    aprobado = Column(Boolean, default=False)
    usado_entrenamiento = Column(Boolean, default=False)

    nombre_archivo = Column(String(500), nullable=False)
    tamaño_archivo = Column(Integer, nullable=True)

    fecha_procesado = Column(DateTime(timezone=True), nullable=True)
    fecha_aprobado = Column(DateTime(timezone=True), nullable=True)

    usuario = relationship("Usuario", back_populates="entrenamientos", foreign_keys=[usuario_id])


class ModeloIA(AuditoriaMixin, Base):
    __tablename__ = "modelos_ia"

    id = Column(Integer, primary_key=True, index=True)

    nombre = Column(String(255), nullable=False, unique=True)
    version = Column(String(50), nullable=False, default="1.0")
    descripcion = Column(Text, nullable=True)

    ruta_archivo = Column(String(500), nullable=False)
    ruta_metadatos = Column(String(500), nullable=True)
    ruta_etiquetas = Column(String(500), nullable=True)

    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)

    num_clases = Column(Integer, nullable=True)
    clases_json = Column(Text, nullable=True)
    total_imagenes = Column(Integer, nullable=True)
    epocas_entrenamiento = Column(Integer, nullable=True)

    activo = Column(Boolean, default=False)
    entrenando = Column(Boolean, default=False)

    peso_ensemble = Column(Float, default=1.0)
    tipo_modelo = Column(String(50), default="LSTM")

    hidden_size = Column(Integer, nullable=True)
    num_layers = Column(Integer, nullable=True)
    bidirectional = Column(Boolean, default=True)
    num_frames = Column(Integer, default=24)

    arquitectura = Column(String(100), nullable=True)
    tamaño_mb = Column(Float, nullable=True)
    parametros = Column(Integer, nullable=True)

    origen = Column(String(50), default="entrenado_local")

    fecha_entrenamiento = Column(DateTime(timezone=True), nullable=True)
    fecha_activacion = Column(DateTime(timezone=True), nullable=True)

    videos_entrenamiento = relationship(
        "VideoDataset",
        secondary=modelo_video_association,
        back_populates="modelos_entrenados"
    )

    @property
    def clases(self):
        return self.clases_json

    @clases.setter
    def clases(self, value):
        self.clases_json = value

    @property
    def epochs_entrenadas(self):
        return self.epocas_entrenamiento

    @epochs_entrenadas.setter
    def epochs_entrenadas(self, value):
        self.epocas_entrenamiento = value

    @property
    def accuracy_porcentaje(self):
        if self.accuracy:
            return f"{self.accuracy * 100:.2f}%"
        return "N/A"

    @property
    def calidad(self):
        if not self.accuracy:
            return "Sin datos"
        elif self.accuracy >= 0.95:
            return "Excelente"
        elif self.accuracy >= 0.90:
            return "Muy Bueno"
        elif self.accuracy >= 0.85:
            return "Bueno"
        elif self.accuracy >= 0.75:
            return "Regular"
        else:
            return "Necesita Mejora"

    @property
    def estado_texto(self):
        return "Activo" if self.activo else "Inactivo"

    @property
    def categorias_vinculadas(self):
        return [cat.nombre for cat in self.categorias_asignadas]

    @property
    def num_categorias_vinculadas(self):
        return len(self.categorias_asignadas) if hasattr(self, 'categorias_asignadas') else 0