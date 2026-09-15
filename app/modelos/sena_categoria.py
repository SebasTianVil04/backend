from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin


class SenaCategoria(AuditoriaMixin, Base):
    __tablename__ = "senas_categoria"

    id = Column(Integer, primary_key=True, index=True)
    categoria_id = Column(Integer, ForeignKey("categorias.id", ondelete="CASCADE"), nullable=False, index=True)
    nombre = Column(String(100), nullable=False, index=True)
    orden = Column(Integer, nullable=False, default=1)
    archivo_referencia = Column(String(500), nullable=True)
    tipo_referencia = Column(String(10), nullable=True)
    activa = Column(Boolean, default=True)

    categoria_rel = relationship("Categoria", backref="senas_catalogo")

    def __repr__(self):
        return f"<SenaCategoria(id={self.id}, nombre='{self.nombre}', categoria_id={self.categoria_id})>"