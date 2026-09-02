# app/esquemas/categoria.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


class CategoriaBase(BaseModel):
    nombre: str = Field(..., max_length=100)
    tipo_id: int = Field(..., description="ID del tipo de categoría")
    descripcion: Optional[str] = Field(None, max_length=500)
    icono: Optional[str] = Field(None, max_length=10)
    color: Optional[str] = Field(None, max_length=20)
    orden: int = Field(..., ge=1)
    nivel_requerido: int = Field(default=1, ge=1, le=10)

class CategoriaCrear(CategoriaBase):
    pass

class CategoriaActualizar(BaseModel):
    nombre: Optional[str] = Field(None, max_length=100)
    tipo_id: Optional[int] = None
    descripcion: Optional[str] = Field(None, max_length=500)
    icono: Optional[str] = Field(None, max_length=10)
    color: Optional[str] = Field(None, max_length=20)
    orden: Optional[int] = Field(None, ge=1)
    nivel_requerido: Optional[int] = Field(None, ge=1, le=10)
    activa: Optional[bool] = None

class CategoriaRespuesta(CategoriaBase):
    id: int
    activa: bool
    total_lecciones: int = 0
    tipo_valor: Optional[str] = None
    tipo_etiqueta: Optional[str] = None
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None
    
    class Config:
        from_attributes = True
