from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class NivelDificultad(int, Enum):
    PRINCIPIANTE = 1
    INTERMEDIO = 2
    AVANZADO = 3

class LeccionBase(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=255)
    descripcion: Optional[str] = None
    sena: str = Field(..., min_length=1, max_length=100)
    orden: int = Field(..., ge=1)
    nivel_dificultad: NivelDificultad = Field(default=NivelDificultad.PRINCIPIANTE)
    activa: bool = True
    bloqueada: bool = False
    leccion_previa_id: Optional[int] = None
    puntos_base: int = Field(default=10, ge=0)
    puntos_perfecto: int = Field(default=20, ge=0)
    requiere_examen_nivel: bool = False
    numero_examen: Optional[int] = Field(None, ge=1)
    imagen_miniatura: Optional[str] = None
    color_tema: str = Field(default="#3B82F6", pattern="^#[0-9A-Fa-f]{6}$")

class LeccionCrear(LeccionBase):
    categoria_id: int = Field(..., ge=1)
    
    @validator('puntos_perfecto')
    def validar_puntos(cls, v, values):
        if 'puntos_base' in values and v < values['puntos_base']:
            raise ValueError('puntos_perfecto debe ser mayor o igual a puntos_base')
        return v

class LeccionActualizar(BaseModel):
    titulo: Optional[str] = Field(None, min_length=1, max_length=255)
    descripcion: Optional[str] = None
    sena: Optional[str] = Field(None, min_length=1, max_length=100)
    orden: Optional[int] = Field(None, ge=1)
    nivel_dificultad: Optional[NivelDificultad] = None
    activa: Optional[bool] = None
    bloqueada: Optional[bool] = None
    leccion_previa_id: Optional[int] = None
    puntos_base: Optional[int] = Field(None, ge=0)
    puntos_perfecto: Optional[int] = Field(None, ge=0)
    requiere_examen_nivel: Optional[bool] = None
    numero_examen: Optional[int] = Field(None, ge=1)
    imagen_miniatura: Optional[str] = None
    color_tema: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")

class LeccionRespuesta(LeccionBase):
    id: int
    categoria_id: int
    categoria_nombre: Optional[str] = None
    total_clases: int = 0
    total_preguntas_examen: int = 0
    total_examenes: int = 0
    nivel_dificultad_texto: str
    fecha_creacion: datetime
    fecha_actualizacion: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class LeccionConProgreso(LeccionRespuesta):
    progreso_usuario: Optional['ProgresoLeccionRespuesta'] = None
    clases_con_progreso: Optional[List['ClaseConProgreso']] = None

class TipoExamen(str, Enum):
    NIVEL = "nivel"
    FINAL = "final"

class ExamenBase(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=255)
    descripcion: Optional[str] = None
    tipo: TipoExamen
    nivel: Optional[int] = Field(None, ge=1)
    orden: int = Field(..., ge=1)
    clases_requeridas: int = Field(default=0, ge=0)
    requiere_todas_clases: bool = False
    tiempo_limite: Optional[int] = Field(None, ge=0)
    puntuacion_minima: int = Field(default=70, ge=0, le=100)
    activo: bool = True

class ExamenCrear(ExamenBase):
    leccion_id: int = Field(..., ge=1)

class ExamenActualizar(BaseModel):
    titulo: Optional[str] = Field(None, min_length=1, max_length=255)
    descripcion: Optional[str] = None
    tipo: Optional[TipoExamen] = None
    nivel: Optional[int] = Field(None, ge=1)
    orden: Optional[int] = Field(None, ge=1)
    clases_requeridas: Optional[int] = Field(None, ge=0)
    requiere_todas_clases: Optional[bool] = None
    tiempo_limite: Optional[int] = Field(None, ge=0)
    puntuacion_minima: Optional[int] = Field(None, ge=0, le=100)
    activo: Optional[bool] = None

class ExamenRespuesta(ExamenBase):
    id: int
    leccion_id: int
    total_preguntas: int = 0
    fecha_creacion: datetime
    disponible: bool = False
    completado: bool = False
    mejor_calificacion: Optional[float] = None
    
    class Config:
        from_attributes = True

class ProgresoExamen(BaseModel):
    examen_id: int
    completado: bool = False
    mejor_calificacion: float = 0.0
    intentos_realizados: int = 0
    ultimo_intento: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class RespuestaAPI(BaseModel):
    exito: bool
    mensaje: str
    datos: Optional[dict] = None

class RespuestaLista(BaseModel):
    exito: bool
    mensaje: str
    datos: list
    total: int
    pagina: int = 1
    por_pagina: int = 10