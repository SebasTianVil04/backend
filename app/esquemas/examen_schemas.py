from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime

class PreguntaExamenBase(BaseModel):
    pregunta: str
    tipo_pregunta: str
    sena_esperada: Optional[str] = None
    imagen_sena: Optional[str] = None
    opciones: Optional[Dict[str, str]] = None
    respuesta_correcta: Optional[str] = None
    puntos: int = 10
    orden: int

class PreguntaExamenCrear(PreguntaExamenBase):
    leccion_id: Optional[int] = None
    
    @field_validator('tipo_pregunta')
    @classmethod
    def validar_tipo_pregunta(cls, v):
        tipos_validos = ['reconocimiento', 'multiple', 'verdadero_falso']
        if v not in tipos_validos:
            raise ValueError(f'Tipo de pregunta debe ser uno de: {tipos_validos}')
        return v

class PreguntaExamenRespuesta(PreguntaExamenBase):
    id: int
    examen_id: int
    leccion_id: Optional[int] = None
    fecha_creacion: datetime
    
    class Config:
        from_attributes = True

class ExamenBase(BaseModel):
    titulo: str
    descripcion: Optional[str] = None
    tipo: str
    nivel: Optional[int] = None
    leccion_id: int
    orden: int = 1
    clases_requeridas: int = 0
    requiere_todas_clases: bool = False
    tiempo_limite: Optional[int] = None
    puntuacion_minima: float = 70.0
    activo: bool = True

class ExamenCrear(ExamenBase):
    preguntas: List[PreguntaExamenCrear] = []
    
    @field_validator('tipo')
    @classmethod
    def validar_tipo(cls, v):
        if v not in ['nivel', 'final']:
            raise ValueError('El tipo debe ser "nivel" o "final"')
        return v

class ExamenRespuesta(ExamenBase):
    id: int
    fecha_creacion: datetime
    preguntas: List[PreguntaExamenRespuesta] = []
    total_preguntas: Optional[int] = 0
    
    class Config:
        from_attributes = True

class ResultadoExamenCrear(BaseModel):
    examen_id: int
    respuestas: Dict[str, Any]
    tiempo_empleado: Optional[int] = None

class ResultadoExamenRespuesta(BaseModel):
    id: int
    usuario_id: int
    examen_id: int
    puntuacion_obtenida: float
    puntuacion_maxima: float
    porcentaje: float
    aprobado: bool
    tiempo_empleado: Optional[int] = None
    fecha_inicio: datetime
    fecha_finalizacion: datetime
    examen_titulo: Optional[str] = None
    
    class Config:
        from_attributes = True