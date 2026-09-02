# app/modelos/practica.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base

class Practica(Base):
    __tablename__ = "practicas"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Relaciones
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=False)
    
    # Resultados de la práctica
    precision = Column(Float, nullable=False)  # 0.0 a 1.0 (0% a 100%)
    puntos_ganados = Column(Integer, nullable=False)
    tiempo_empleado = Column(Integer, nullable=True)  # En segundos
    
    # Detalles de reconocimiento
    sena_esperada = Column(String(100), nullable=False)
    sena_detectada = Column(String(100), nullable=True)
    confianza = Column(Float, nullable=True)  # Confianza del modelo
    
    # Datos técnicos (para análisis)
    puntos_mano_detectados = Column(JSON, nullable=True)
    imagen_capturada = Column(String(500), nullable=True)  # URL opcional
    
    # Feedback
    feedback = Column(String(50), nullable=True)  # "excelente", "bueno", "regular", "intenta_nuevamente"
    
    # Timestamps
    fecha_practica = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relaciones
    usuario = relationship("Usuario", back_populates="practicas")
    leccion = relationship("Leccion", back_populates="practicas")
    
    @property
    def precision_porcentaje(self):
        """Retorna la precisión como porcentaje"""
        return f"{self.precision * 100:.1f}%"
    
    @property
    def es_perfecto(self):
        """Verifica si la práctica fue perfecta (100%)"""
        return self.precision >= 1.0
    
    @property
    def calificacion(self):
        """Retorna calificación basada en precisión"""
        if self.precision >= 0.95:
            return "excelente"
        elif self.precision >= 0.80:
            return "bueno"
        elif self.precision >= 0.60:
            return "regular"
        else:
            return "intenta_nuevamente"
    
    def __repr__(self):
        return f"<Practica(usuario_id={self.usuario_id}, precision={self.precision_porcentaje})>"