from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone, timedelta
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin


class TokenRecuperacion(AuditoriaMixin, Base):
    __tablename__ = "tokens_recuperacion"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    usuario_email = Column(String, ForeignKey("usuarios.email"), nullable=False)
    usado = Column(Boolean, default=False)
    fecha_expiracion = Column(DateTime(timezone=True), nullable=False)

    usuario = relationship("Usuario", back_populates="tokens_recuperacion", foreign_keys=[usuario_email])

    @property
    def esta_expirado(self) -> bool:
        return datetime.now(timezone.utc) > self.fecha_expiracion

    @classmethod
    def crear_token(cls, usuario_email: str, expire_minutes: int = 15):
        import secrets
        token = secrets.token_urlsafe(32)
        fecha_expiracion = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)

        return cls(
            token=token,
            usuario_email=usuario_email,
            fecha_expiracion=fecha_expiracion
        )