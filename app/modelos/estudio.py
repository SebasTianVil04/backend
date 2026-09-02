# app/modelos/estudio.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base

class SesionEstudio(Base):
    __tablename__ = "sesiones_estudio"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    clase_id = Column(Integer, ForeignKey("clases.id"), nullable=True)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=True)
    
    # Tipo de sesión
    tipo_sesion = Column(String(50), nullable=False)  # 'clase', 'practica', 'examen', 'video'
    
    # Tiempos
    fecha_inicio = Column(DateTime(timezone=True), nullable=False)
    fecha_fin = Column(DateTime(timezone=True), nullable=True)
    duracion_segundos = Column(Integer, default=0)
    
    # Metadata
    dispositivo = Column(String(255), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Timestamps
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    
    # 🔥 RELACIONES CORREGIDAS
    usuario = relationship("Usuario", back_populates="sesiones_estudio")
    clase = relationship("Clase", back_populates="sesiones_estudio")
    leccion = relationship("Leccion", back_populates="sesiones_estudio")
    
    def __repr__(self):
        return f"<SesionEstudio(usuario_id={self.usuario_id}, duracion={self.duracion_segundos}s)>"