from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SenaCategoriaCrear(BaseModel):
    nombre: str = Field(..., max_length=100)
    orden: int = Field(default=1, ge=1)


class SenaCategoriaActualizar(BaseModel):
    nombre: Optional[str] = Field(None, max_length=100)
    orden: Optional[int] = Field(None, ge=1)
    activa: Optional[bool] = None


class SenaCategoriaRespuesta(BaseModel):
    id: int
    categoria_id: int
    nombre: str
    orden: int
    archivo_referencia: Optional[str] = None
    tipo_referencia: Optional[str] = None
    activa: bool
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None

    class Config:
        from_attributes = True