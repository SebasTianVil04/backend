from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin


class SesionEstudio(AuditoriaMixin, Base):
    __tablename__ = "sesiones_estudio"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    clase_id = Column(Integer, ForeignKey("clases.id"), nullable=True)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=True)

    tipo_sesion = Column(String(50), nullable=False)

    fecha_inicio = Column(DateTime(timezone=True), nullable=False)
    fecha_fin = Column(DateTime(timezone=True), nullable=True)
    duracion_segundos = Column(Integer, default=0)

    dispositivo = Column(String(255), nullable=True)
    user_agent = Column(Text, nullable=True)

    usuario = relationship("Usuario", back_populates="sesiones_estudio", foreign_keys=[usuario_id])
    clase = relationship("Clase", back_populates="sesiones_estudio")
    leccion = relationship("Leccion", back_populates="sesiones_estudio")

    def __repr__(self):
        return f"<SesionEstudio(usuario_id={self.usuario_id}, duracion={self.duracion_segundos}s)>"