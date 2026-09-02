# app/esquemas/tipo_categoria_schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TipoCategoriaBase(BaseModel):
    valor: str = Field(..., max_length=50, description="Identificador único del tipo")
    etiqueta: str = Field(..., max_length=100, description="Nombre para mostrar")
    icono: str = Field(..., max_length=20, description="Emoji o símbolo")
    color: str = Field(..., max_length=20, description="Color en formato hex")

class TipoCategoriaCrear(TipoCategoriaBase):
    pass

class TipoCategoriaActualizar(BaseModel):
    etiqueta: Optional[str] = Field(None, max_length=100)
    icono: Optional[str] = Field(None, max_length=20)
    color: Optional[str] = Field(None, max_length=20)
    activo: Optional[bool] = None

class TipoCategoriaRespuesta(TipoCategoriaBase):
    id: int
    activo: bool
    fecha_creacion: datetime
    
    class Config:
        from_attributes = True
