from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional

class ModeloIASchema(BaseModel):
    id: int
    nombre: str
    version: str = "1.0"
    descripcion: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    num_clases: Optional[int] = None
    total_imagenes: Optional[int] = None
    epocas_entrenamiento: Optional[int] = None
    activo: bool = False
    entrenando: bool = False
    estado_texto: str = "Inactivo"
    calidad: str = "Sin datos"
    accuracy_porcentaje: str = "N/A"
    fecha_creacion: Optional[datetime] = None
    fecha_entrenamiento: Optional[datetime] = None
    fecha_activacion: Optional[datetime] = None
    arquitectura: Optional[str] = None
    tamaño_mb: Optional[float] = None
    ruta_modelo: Optional[str] = None

    class Config:
        from_attributes = True

    @validator('estado_texto', always=True)
    def calcular_estado_texto(cls, v, values):
        return "Activo" if values.get('activo', False) else "Inactivo"

    @validator('accuracy_porcentaje', always=True)
    def calcular_accuracy_porcentaje(cls, v, values):
        accuracy = values.get('accuracy')
        if accuracy is not None:
            return f"{accuracy * 100:.2f}%"
        return "N/A"

    @validator('calidad', always=True)
    def calcular_calidad(cls, v, values):
        accuracy = values.get('accuracy')
        if accuracy is None:
            return "Sin datos"
        elif accuracy >= 0.95:
            return "Excelente"
        elif accuracy >= 0.90:
            return "Muy Bueno"
        elif accuracy >= 0.85:
            return "Bueno"
        elif accuracy >= 0.75:
            return "Regular"
        else:
            return "Necesita Mejora"
