# app/modelos/sesion_usuario.py
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String

from ..utilidades.base_datos import Base  # ajusta si tu Base vive en otro módulo


class SesionUsuario(Base):
    """
    Una fila = una sesión (un dispositivo/navegador).
    - El refresh token NUNCA se guarda en claro: solo su SHA-256.
    - refresh_hash_anterior permite detectar reutilización de un refresh ya rotado (posible robo).
    - El access token lleva el id de esta fila (claim "sid"): si la sesión se revoca,
      el access token deja de valer al instante aunque no haya expirado.
    """
    __tablename__ = "sesiones_usuario"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(
        Integer, ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False, index=True
    )

    refresh_hash = Column(String(64), nullable=False, unique=True, index=True)
    refresh_hash_anterior = Column(String(64), nullable=True, index=True)
    rotada_en = Column(DateTime(timezone=True), nullable=True)

    creada_en = Column(DateTime(timezone=True), nullable=False)
    ultimo_uso = Column(DateTime(timezone=True), nullable=False)
    expira_en = Column(DateTime(timezone=True), nullable=False)
    expira_absoluta_en = Column(DateTime(timezone=True), nullable=False)

    revocada = Column(Boolean, nullable=False, default=False, index=True)
    revocada_en = Column(DateTime(timezone=True), nullable=True)
    revocada_motivo = Column(String(50), nullable=True)

    ip = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)