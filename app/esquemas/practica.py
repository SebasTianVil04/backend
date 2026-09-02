from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any
from datetime import datetime

class PracticaBase(BaseModel):
    leccion_id: int
    sena_esperada: str
    precision: float
    puntos_ganados: int
    tiempo_empleado: Optional[int] = None
    sena_detectada: Optional[str] = None
    confianza: Optional[float] = None
    puntos_mano_detectados: Optional[Dict[str, Any]] = None
    imagen_capturada: Optional[str] = None

class PracticaCrear(BaseModel):
    leccion_id: int
    sena_detectada: str
    confianza: float
    tiempo_empleado: Optional[int] = None
    puntos_mano_detectados: Optional[Dict[str, Any]] = None
    imagen_capturada: Optional[str] = None

class PracticaRespuesta(BaseModel):
    id: int
    usuario_id: int
    leccion_id: int
    sena_esperada: str
    sena_detectada: Optional[str] = None
    precision: float
    precision_porcentaje: str
    puntos_ganados: int
    confianza: Optional[float] = None
    feedback: Optional[str] = None
    es_perfecto: bool
    calificacion: str
    tiempo_empleado: Optional[int] = None
    fecha_practica: datetime
    leccion_titulo: Optional[str] = None
    
    class Config:
        from_attributes = True