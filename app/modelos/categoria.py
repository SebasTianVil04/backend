# app/modelos/categoria.py
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from typing import List, Optional


class AsignacionModeloLote(BaseModel):
    categoria_id: int
    modelo_id: Optional[int] = None

class AsignacionesLoteRequest(BaseModel):
    asignaciones: List[AsignacionModeloLote]

class Categoria(Base):
    __tablename__ = "categorias"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), nullable=False, unique=True, index=True)
    
    tipo_id = Column(Integer, ForeignKey("tipos_categoria.id"), nullable=False, index=True)
    
    # ✅ NUEVO: Vincular con ModeloIA
    modelo_ia_id = Column(Integer, ForeignKey("modelos_ia.id", ondelete='SET NULL'), nullable=True, index=True)
    
    descripcion = Column(Text, nullable=True)
    icono = Column(String(255), nullable=True)
    color = Column(String(20), nullable=True)
    orden = Column(Integer, nullable=False)
    nivel_requerido = Column(Integer, default=1)
    activa = Column(Boolean, default=True)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relaciones existentes
    tipo_rel = relationship("TipoCategoria", back_populates="categorias")
    lecciones = relationship("Leccion", back_populates="categoria_rel", cascade="all, delete-orphan", order_by="Leccion.orden")
    dataset_categoria = relationship("CategoriaDataset", back_populates="categoria_rel", uselist=False)
    
    # ✅ NUEVO: Relación con ModeloIA
    modelo_ia = relationship("ModeloIA", foreign_keys=[modelo_ia_id], backref="categorias_asignadas")
    
    def __repr__(self):
        return f"<Categoria(id={self.id}, nombre='{self.nombre}', modelo_ia_id={self.modelo_ia_id})>"
    
    @property
    def total_lecciones(self):
        return len(self.lecciones) if self.lecciones else 0
    
    @property
    def tiene_modelo_asignado(self):
        """Verifica si la categoría tiene un modelo IA asignado"""
        return self.modelo_ia_id is not None
    
    @property
    def nombre_modelo(self):
        """Retorna el nombre del modelo asignado o None"""
        return self.modelo_ia.nombre if self.modelo_ia else None