from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import Optional, Union
from datetime import datetime
from enum import Enum

class TipoVideo(str, Enum):
    YOUTUBE = "youtube"
    GOOGLE_DRIVE = "google_drive"
    VIMEO = "vimeo"

class ClaseBase(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=255)
    descripcion: Optional[str] = None
    contenido_texto: Optional[str] = None
    sena: Optional[str] = Field(None, max_length=50)
    tipo_video: TipoVideo = TipoVideo.YOUTUBE
    video_url: Optional[str] = Field(None, max_length=500)
    video_id: Optional[str] = Field(None, max_length=255)
    imagen_referencia: Optional[str] = Field(None, max_length=500)
    gif_demostracion: Optional[str] = Field(None, max_length=500)
    orden: int = Field(..., ge=1)
    duracion_estimada: Optional[int] = Field(None, ge=0)
    tips: Optional[str] = None
    errores_comunes: Optional[str] = None
    requiere_practica: bool = True
    intentos_minimos: int = Field(default=3, ge=1)
    precision_minima: float = Field(default=0.7, ge=0.0, le=1.0)
    activa: bool = True

class ClaseCrear(ClaseBase):
    leccion_id: int = Field(..., ge=1)
    
    @validator('sena')
    def validar_sena(cls, v, values):
        requiere_practica = values.get('requiere_practica', True)
        
        if requiere_practica:
            if v is None:
                raise ValueError('La seña es obligatoria cuando se requiere práctica')
            if isinstance(v, str) and not v.strip():
                raise ValueError('La seña es obligatoria cuando se requiere práctica')
        
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        
        return v

    @validator('descripcion', 'contenido_texto', 'tips', 'errores_comunes', 'video_url', 'video_id', 'imagen_referencia', 'gif_demostracion')
    def convertir_vacios_a_none(cls, v):
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        return v

class ClaseActualizar(BaseModel):
    titulo: Optional[str] = Field(None, min_length=1, max_length=255)
    descripcion: Optional[str] = None
    contenido_texto: Optional[str] = None
    sena: Optional[str] = Field(None, max_length=50)
    tipo_video: Optional[TipoVideo] = None
    video_url: Optional[str] = Field(None, max_length=500)
    video_id: Optional[str] = Field(None, max_length=255)
    imagen_referencia: Optional[str] = Field(None, max_length=500)
    gif_demostracion: Optional[str] = Field(None, max_length=500)
    orden: Optional[int] = Field(None, ge=1)
    duracion_estimada: Optional[int] = Field(None, ge=0)
    tips: Optional[str] = None
    errores_comunes: Optional[str] = None
    requiere_practica: Optional[bool] = None
    intentos_minimos: Optional[int] = Field(None, ge=1)
    precision_minima: Optional[float] = Field(None, ge=0.0, le=1.0)
    activa: Optional[bool] = None

    @validator('sena')
    def validar_sena(cls, v):
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        return v

    @validator('descripcion', 'contenido_texto', 'tips', 'errores_comunes', 'video_url', 'video_id', 'imagen_referencia', 'gif_demostracion')
    def convertir_vacios_a_none(cls, v):
        if v is not None and isinstance(v, str) and not v.strip():
            return None
        return v

class ClaseRespuesta(ClaseBase):
    id: int
    leccion_id: int
    url_video_embebida: Optional[str] = None
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None
    
    class Config:
        from_attributes = True