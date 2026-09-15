from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin


class TipoCategoria(AuditoriaMixin, Base):
    __tablename__ = "tipos_categoria"

    id = Column(Integer, primary_key=True, index=True)
    valor = Column(String(50), nullable=False, unique=True, index=True)
    etiqueta = Column(String(100), nullable=False)
    icono = Column(String(20), nullable=False)
    color = Column(String(20), nullable=False)
    activo = Column(Boolean, default=True)

    categorias = relationship("Categoria", back_populates="tipo_rel", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TipoCategoria(valor='{self.valor}', etiqueta='{self.etiqueta}')>"