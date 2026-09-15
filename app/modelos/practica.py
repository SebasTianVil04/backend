from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin


class Practica(AuditoriaMixin, Base):
    __tablename__ = "practicas"

    id = Column(Integer, primary_key=True, index=True)

    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=False)

    precision = Column(Float, nullable=False)
    puntos_ganados = Column(Integer, nullable=False)
    tiempo_empleado = Column(Integer, nullable=True)

    sena_esperada = Column(String(100), nullable=False)
    sena_detectada = Column(String(100), nullable=True)
    confianza = Column(Float, nullable=True)

    puntos_mano_detectados = Column(JSON, nullable=True)
    imagen_capturada = Column(String(500), nullable=True)

    feedback = Column(String(50), nullable=True)

    fecha_practica = Column(DateTime(timezone=True), server_default=func.now())

    usuario = relationship("Usuario", back_populates="practicas", foreign_keys=[usuario_id])
    leccion = relationship("Leccion", back_populates="practicas")

    @property
    def precision_porcentaje(self):
        return f"{self.precision * 100:.1f}%"

    @property
    def es_perfecto(self):
        return self.precision >= 1.0

    @property
    def calificacion(self):
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