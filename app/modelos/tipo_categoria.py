# app/modelos/tipo_categoria.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base

class TipoCategoria(Base):
    __tablename__ = "tipos_categoria"
    
    id = Column(Integer, primary_key=True, index=True)
    valor = Column(String(50), nullable=False, unique=True, index=True)
    etiqueta = Column(String(100), nullable=False)
    icono = Column(String(20), nullable=False)
    color = Column(String(20), nullable=False)
    activo = Column(Boolean, default=True)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    # NUEVO: Relación con Categoría
    categorias = relationship("Categoria", back_populates="tipo_rel", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TipoCategoria(valor='{self.valor}', etiqueta='{self.etiqueta}')>"
