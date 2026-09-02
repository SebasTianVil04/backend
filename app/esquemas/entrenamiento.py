from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class EntrenamientoBase(BaseModel):
    nombre_sena: str
    descripcion: Optional[str] = None
    categoria: Optional[str] = None

class EntrenamientoCrear(EntrenamientoBase):
    imagen_url: str
    nombre_archivo: str
    tamaño_archivo: Optional[int] = None

class EntrenamientoRespuesta(BaseModel):
    id: int
    usuario_id: int
    nombre_sena: str
    descripcion: Optional[str] = None
    categoria: Optional[str] = None
    imagen_url: str
    procesado: bool
    aprobado: bool
    usado_entrenamiento: bool
    nombre_archivo: str
    fecha_creacion: datetime
    fecha_procesado: Optional[datetime] = None
    fecha_aprobado: Optional[datetime] = None
    usuario_nombre: Optional[str] = None
    
    class Config:
        from_attributes = True

class ModeloIABase(BaseModel):
    nombre: str
    version: str = "1.0"
    descripcion: Optional[str] = None
    arquitectura: Optional[str] = None

class ModeloIACrear(ModeloIABase):
    ruta_modelo: str
    ruta_etiquetas: Optional[str] = None

class ModeloIAActualizar(BaseModel):
    descripcion: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    num_clases: Optional[int] = None
    total_imagenes: Optional[int] = None
    epocas_entrenamiento: Optional[int] = None
    activo: Optional[bool] = None

class ModeloIARespuesta(BaseModel):
    id: int
    nombre: str
    version: str
    descripcion: Optional[str] = None
    ruta_modelo: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    num_clases: Optional[int] = None
    total_imagenes: Optional[int] = None
    epocas_entrenamiento: Optional[int] = None
    activo: bool
    entrenando: bool
    arquitectura: Optional[str] = None
    tamaño_mb: Optional[float] = None
    estado_texto: str
    calidad: str
    accuracy_porcentaje: str
    fecha_creacion: datetime
    fecha_entrenamiento: Optional[datetime] = None
    fecha_activacion: Optional[datetime] = None
    
    class Config:
        from_attributes = True
