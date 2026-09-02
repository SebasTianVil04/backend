from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from ..utilidades.base_datos import Base

class TipoVideo(str, enum.Enum):
    YOUTUBE = "youtube"
    GOOGLE_DRIVE = "google_drive"
    VIMEO = "vimeo"

class Clase(Base):
    __tablename__ = "clases"
    
    id = Column(Integer, primary_key=True, index=True)
    
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=False)
    
    titulo = Column(String(255), nullable=False)
    descripcion = Column(Text, nullable=True)
    contenido_texto = Column(Text, nullable=True) 
    
    sena = Column(String(50), nullable=True)
    
    tipo_video = Column(Enum(TipoVideo), default=TipoVideo.YOUTUBE)
    video_url = Column(String(500), nullable=True)
    video_id = Column(String(255), nullable=True)
    
    imagen_referencia = Column(String(500), nullable=True)
    gif_demostracion = Column(String(500), nullable=True)
    
    orden = Column(Integer, nullable=False)
    duracion_estimada = Column(Integer, nullable=True)
    
    tips = Column(Text, nullable=True) 
    errores_comunes = Column(Text, nullable=True) 
    
    requiere_practica = Column(Boolean, default=True)
    intentos_minimos = Column(Integer, default=3)
    precision_minima = Column(Float, default=0.7)
    
    activa = Column(Boolean, default=True)
    
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    leccion = relationship("Leccion", back_populates="clases")
    progresos_clase = relationship("ProgresoClase", back_populates="clase", cascade="all, delete-orphan")
    sesiones_estudio = relationship("SesionEstudio", back_populates="clase", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Clase(id={self.id}, titulo='{self.titulo}', leccion_id={self.leccion_id})>"
    
    @property
    def url_video_embebida(self):
        if self.tipo_video == TipoVideo.YOUTUBE and self.video_id:
            return f"https://www.youtube.com/embed/{self.video_id}"
        elif self.tipo_video == TipoVideo.GOOGLE_DRIVE and self.video_id:
            return f"https://drive.google.com/file/d/{self.video_id}/preview"
        return None
    
    def calcular_puntos_obtenidos(self, precision: float, completada: bool = False):
        """Calcula los puntos obtenidos basado en la precisión"""
        if not completada:
            return 0
        
        # Puntos base por completar la clase
        puntos_base = 10
        
        # Puntos adicionales por precisión
        puntos_adicionales = int(puntos_base * precision)
        
        return puntos_base + puntos_adicionales
    
    def calcular_xp_obtenido(self, precision: float, completada: bool = False):
        """Calcula el XP obtenido basado en la precisión"""
        if not completada:
            return 0
        
        # XP base por completar la clase
        xp_base = 5
        
        # XP adicional por precisión
        xp_adicional = int(xp_base * precision)
        
        return xp_base + xp_adicional