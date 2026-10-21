from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship

from ..utilidades.base_datos import Base


class TicketSoporte(Base):
    __tablename__ = "tickets_soporte"

    id = Column(Integer, primary_key=True, index=True)
    codigo = Column(String(20), unique=True, index=True, nullable=True)

    usuario_id = Column(Integer, ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True, index=True)
    asignado_a_id = Column(Integer, ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True, index=True)

    nombre = Column(String(80), nullable=False)
    email = Column(String(255), nullable=False, index=True)

    tipo = Column(String(20), nullable=False)
    servicio = Column(String(30), nullable=False, index=True)
    asunto = Column(String(120), nullable=False)
    descripcion = Column(Text, nullable=False)

    prioridad = Column(String(10), nullable=False, default="media")
    estado = Column(String(20), nullable=False, default="nueva", index=True)
    origen = Column(String(15), nullable=False, default="publico")

    puntaje_spam = Column(Integer, nullable=False, default=0)
    ip_hash = Column(String(64), nullable=True, index=True)
    hash_contenido = Column(String(64), nullable=True, index=True)

    fecha_creacion = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    fecha_actualizacion = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    fecha_cierre = Column(DateTime, nullable=True)

    comentarios = relationship(
        "TicketComentario",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="TicketComentario.fecha_creacion",
    )

    __table_args__ = (
        Index("ix_tickets_email_fecha", "email", "fecha_creacion"),
        Index("ix_tickets_ip_fecha", "ip_hash", "fecha_creacion"),
    )

    def __repr__(self):
        return f"<TicketSoporte(codigo={self.codigo}, estado={self.estado})>"


class TicketComentario(Base):
    __tablename__ = "tickets_comentarios"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets_soporte.id", ondelete="CASCADE"), nullable=False, index=True)
    autor_id = Column(Integer, ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True)
    autor_nombre = Column(String(80), nullable=False)
    es_soporte = Column(Boolean, nullable=False, default=False)
    es_interno = Column(Boolean, nullable=False, default=False)
    mensaje = Column(Text, nullable=False)
    fecha_creacion = Column(DateTime, nullable=False, default=datetime.utcnow)

    ticket = relationship("TicketSoporte", back_populates="comentarios")
