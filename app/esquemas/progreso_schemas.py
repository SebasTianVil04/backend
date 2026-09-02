from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class NivelDominio(str, Enum):
    PERFECTO = "Perfecto"
    EXCELENTE = "Excelente"
    BUENO = "Bueno"
    ACEPTABLE = "Aceptable"
    NECESITA_MEJORAR = "Necesita Mejorar"

class DominioLeccion(str, Enum):
    DOMINIO_COMPLETO = "Dominio Completo"
    DOMINIO_ALTO = "Dominio Alto"
    DOMINIO_MEDIO = "Dominio Medio"
    DOMINIO_BASICO = "Dominio Básico"
    EN_DESARROLLO = "En Desarrollo"

class ProgresoClaseBase(BaseModel):
    vista: bool = False
    completada: bool = False
    aprobada: bool = False
    intentos_realizados: int = Field(default=0, ge=0)
    intentos_exitosos: int = Field(default=0, ge=0)
    mejor_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    ultima_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_promedio: float = Field(default=0.0, ge=0.0, le=1.0)
    puntos_ganados: int = Field(default=0, ge=0)
    xp_ganado: int = Field(default=0, ge=0)
    tiempo_total_practica: int = Field(default=0, ge=0)

class ProgresoClaseCrear(ProgresoClaseBase):
    usuario_id: int = Field(..., ge=1)
    clase_id: int = Field(..., ge=1)

class ProgresoClaseActualizar(BaseModel):
    vista: Optional[bool] = None
    completada: Optional[bool] = None
    aprobada: Optional[bool] = None
    intentos_realizados: Optional[int] = Field(None, ge=0)
    intentos_exitosos: Optional[int] = Field(None, ge=0)
    mejor_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    ultima_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision_promedio: Optional[float] = Field(None, ge=0.0, le=1.0)
    puntos_ganados: Optional[int] = Field(None, ge=0)
    xp_ganado: Optional[int] = Field(None, ge=0)
    tiempo_total_practica: Optional[int] = Field(None, ge=0)

class ProgresoClaseRespuesta(ProgresoClaseBase):
    id: int
    usuario_id: int
    clase_id: int
    porcentaje_precision: str
    tasa_exito: float
    nivel_dominio: str
    fecha_primera_vista: Optional[datetime] = None
    fecha_completada: Optional[datetime] = None
    fecha_mejor_precision: Optional[datetime] = None
    ultima_practica: Optional[datetime] = None
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class RegistroIntentoRequest(BaseModel):
    precision: float = Field(..., ge=0.0, le=1.0)
    tiempo_practica: int = Field(..., ge=0)
    sena_reconocida: str
    es_exitoso: Optional[bool] = None

class RegistroIntentoRespuesta(BaseModel):
    puntos_ganados: int
    xp_ganado: int
    precision: float
    es_exitoso: bool
    nivel_dominio: str
    clase_aprobada: bool
    clase_completada: bool

class ResumenDesempenoClase(BaseModel):
    clase_id: int
    completada: bool
    aprobada: bool
    intentos_realizados: int
    intentos_exitosos: int
    tasa_exito: float
    mejor_precision: float
    precision_promedio: float
    ultima_precision: float
    nivel_dominio: str
    puntos_ganados: int
    xp_ganado: int
    tiempo_total_practica_minutos: int
    fecha_completada: Optional[str] = None

class ProgresoLeccionBase(BaseModel):
    desbloqueada: bool = False
    bloqueada: bool = False
    iniciada: bool = False
    completada: bool = False
    total_clases: int = Field(default=0, ge=0)
    clases_completadas: int = Field(default=0, ge=0)
    clases_aprobadas: int = Field(default=0, ge=0)
    clases_vistas: int = Field(default=0, ge=0)
    mejor_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_promedio: float = Field(default=0.0, ge=0.0, le=1.0)
    total_intentos: int = Field(default=0, ge=0)
    intentos_exitosos: int = Field(default=0, ge=0)
    total_puntos: int = Field(default=0, ge=0)
    xp_total: int = Field(default=0, ge=0)
    puntos_bonificacion_completa: int = Field(default=0, ge=0)
    xp_bonificacion_completa: int = Field(default=0, ge=0)
    puntos_maximos_posibles: int = Field(default=0, ge=0)
    estrellas: int = Field(default=0, ge=0, le=3)
    estrella_dorada: bool = False
    tiempo_total_minutos: int = Field(default=0, ge=0)
    tiempo_promedio_clase_minutos: int = Field(default=0, ge=0)
    examenes_disponibles: int = Field(default=0, ge=0)
    examenes_completados: int = Field(default=0, ge=0)
    examenes_aprobados: int = Field(default=0, ge=0)
    mejor_calificacion_examen: Optional[float] = Field(None, ge=0.0, le=100.0)
    racha_dias_consecutivos: int = Field(default=0, ge=0)
    dias_activos: int = Field(default=0, ge=0)

class ProgresoLeccionCrear(ProgresoLeccionBase):
    usuario_id: int = Field(..., ge=1)
    leccion_id: int = Field(..., ge=1)

class ProgresoLeccionActualizar(BaseModel):
    desbloqueada: Optional[bool] = None
    bloqueada: Optional[bool] = None
    iniciada: Optional[bool] = None
    completada: Optional[bool] = None
    total_clases: Optional[int] = Field(None, ge=0)
    clases_completadas: Optional[int] = Field(None, ge=0)
    clases_aprobadas: Optional[int] = Field(None, ge=0)
    clases_vistas: Optional[int] = Field(None, ge=0)
    mejor_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision_promedio: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_intentos: Optional[int] = Field(None, ge=0)
    intentos_exitosos: Optional[int] = Field(None, ge=0)
    total_puntos: Optional[int] = Field(None, ge=0)
    xp_total: Optional[int] = Field(None, ge=0)
    puntos_bonificacion_completa: Optional[int] = Field(None, ge=0)
    xp_bonificacion_completa: Optional[int] = Field(None, ge=0)
    puntos_maximos_posibles: Optional[int] = Field(None, ge=0)
    estrellas: Optional[int] = Field(None, ge=0, le=3)
    estrella_dorada: Optional[bool] = None
    tiempo_total_minutos: Optional[int] = Field(None, ge=0)
    tiempo_promedio_clase_minutos: Optional[int] = Field(None, ge=0)
    examenes_disponibles: Optional[int] = Field(None, ge=0)
    examenes_completados: Optional[int] = Field(None, ge=0)
    examenes_aprobados: Optional[int] = Field(None, ge=0)
    mejor_calificacion_examen: Optional[float] = Field(None, ge=0.0, le=100.0)
    racha_dias_consecutivos: Optional[int] = Field(None, ge=0)
    dias_activos: Optional[int] = Field(None, ge=0)

class ProgresoLeccionRespuesta(ProgresoLeccionBase):
    id: int
    usuario_id: int
    leccion_id: int
    porcentaje_completado: float
    porcentaje_precision: str
    tasa_exito_general: float
    eficiencia_puntos: float
    nivel_dominio_leccion: str
    tiene_estrella_dorada: bool
    dias_desde_ultima_actividad: int
    fecha_desbloqueo: Optional[datetime] = None
    fecha_inicio: Optional[datetime] = None
    fecha_completada: Optional[datetime] = None
    ultima_practica: Optional[datetime] = None
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class ResultadoPracticaRequest(BaseModel):
    precision: float
    tiempo_practica: int
    sena_reconocida: str
    es_exitoso: Optional[bool] = None

class RequisitosClase(BaseModel):
    intentos_minimos: int
    precision_minima: float
    cumple_intentos: bool
    cumple_precision: bool
    sena_correcta: bool
    sena_esperada: str
    sena_reconocida: str

class RespuestaGuardarPractica(BaseModel):
    exito: bool
    mensaje: str
    progreso: ProgresoClaseRespuesta
    clase_completada: bool
    puntos_ganados: int
    xp_ganado: int
    racha_actual: int
    nivel_subido: bool
    nuevo_nivel: Optional[int] = None
    requisitos: RequisitosClase