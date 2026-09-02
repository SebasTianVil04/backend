from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .dataset import modelo_video_association 


class Entrenamiento(Base):
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
    
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_procesado = Column(DateTime(timezone=True), nullable=True)
    fecha_aprobado = Column(DateTime(timezone=True), nullable=True)
    
    usuario = relationship("Usuario", back_populates="entrenamientos")


class ModeloIA(Base):
    __tablename__ = "modelos_ia"
    
    id = Column(Integer, primary_key=True, index=True)
    
    nombre = Column(String(255), nullable=False, unique=True)
    version = Column(String(50), nullable=False, default="1.0")
    descripcion = Column(Text, nullable=True)
    
    # Archivos del modelo
    ruta_archivo = Column(String(500), nullable=False)  # .pth para PyTorch
    ruta_metadatos = Column(String(500), nullable=True)
    ruta_etiquetas = Column(String(500), nullable=True)
    
    # Métricas del modelo
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Información del dataset
    num_clases = Column(Integer, nullable=True)
    clases_json = Column(Text, nullable=True)  # JSON string con las clases
    total_imagenes = Column(Integer, nullable=True)
    epocas_entrenamiento = Column(Integer, nullable=True)  # Alias para epochs
    
    # Estados
    activo = Column(Boolean, default=False)
    entrenando = Column(Boolean, default=False)
    
    # Configuración para ensemble
    peso_ensemble = Column(Float, default=1.0)
    tipo_modelo = Column(String(50), default="LSTM")  # LSTM, CNN, video_cnn, etc.
    
    # Información de arquitectura LSTM
    hidden_size = Column(Integer, nullable=True)
    num_layers = Column(Integer, nullable=True)
    bidirectional = Column(Boolean, default=True)
    num_frames = Column(Integer, default=24)  # Frames usados en entrenamiento
    
    # Metadatos adicionales
    arquitectura = Column(String(100), nullable=True)
    tamaño_mb = Column(Float, nullable=True)
    parametros = Column(Integer, nullable=True)
    
    # NUEVO: Origen del modelo
    origen = Column(String(50), default="entrenado_local")  # entrenado_local, cargado_manualmente, transferencia
    
    # Timestamps
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_entrenamiento = Column(DateTime(timezone=True), nullable=True)
    fecha_activacion = Column(DateTime(timezone=True), nullable=True)
    
    # Relación many-to-many con videos
    videos_entrenamiento = relationship(
        "VideoDataset",
        secondary=modelo_video_association,
        back_populates="modelos_entrenados"
    )
    
    # Property para compatibilidad con código antiguo
    @property
    def clases(self):
        """Alias para clases_json para compatibilidad"""
        return self.clases_json
    
    @clases.setter
    def clases(self, value):
        """Setter para clases_json"""
        self.clases_json = value
    
    @property
    def epochs_entrenadas(self):
        """Alias para epocas_entrenamiento"""
        return self.epocas_entrenamiento
    
    @epochs_entrenadas.setter
    def epochs_entrenadas(self, value):
        """Setter para epocas_entrenamiento"""
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
        """Retorna lista de categorías que usan este modelo"""
        return [cat.nombre for cat in self.categorias_asignadas]
    
    @property
    def num_categorias_vinculadas(self):
        """Cuenta cuántas categorías usan este modelo"""
        return len(self.categorias_asignadas) if hasattr(self, 'categorias_asignadas') else 0