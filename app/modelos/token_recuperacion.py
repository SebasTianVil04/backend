from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from ..utilidades.base_datos import Base

class TokenRecuperacion(Base):
    __tablename__ = "tokens_recuperacion"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    usuario_email = Column(String, ForeignKey("usuarios.email"), nullable=False)
    usado = Column(Boolean, default=False)
    fecha_creacion = Column(DateTime, default=datetime.utcnow)
    fecha_expiracion = Column(DateTime, nullable=False)
    
    # Relación
    usuario = relationship("Usuario", back_populates="tokens_recuperacion")
    
    @property
    def esta_expirado(self) -> bool:
        return datetime.utcnow() > self.fecha_expiracion
    
    @classmethod
    def crear_token(cls, usuario_email: str, expire_minutes: int = 15):
        import secrets
        token = secrets.token_urlsafe(32)
        fecha_expiracion = datetime.utcnow() + timedelta(minutes=expire_minutes)
        
        return cls(
            token=token,
            usuario_email=usuario_email,
            fecha_expiracion=fecha_expiracion
        )