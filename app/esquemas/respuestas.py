from pydantic import BaseModel
from typing import Any, Optional, List, Dict, Union

class RespuestaAPI(BaseModel):
    """Esquema base flexible para respuestas de la API"""
    exito: bool
    mensaje: str
    datos: Optional[Union[Dict[str, Any], List[Any], Any]] = None
    errores: Optional[List[str]] = None

class RespuestaLista(BaseModel):
    """Esquema para respuestas que contienen listas paginadas"""
    exito: bool
    mensaje: str
    datos: List[Any]
    total: int
    pagina: int = 1
    por_pagina: int = 10
    total_paginas: int = 1

class RespuestaError(BaseModel):
    """Esquema para respuestas de error"""
    exito: bool = False
    mensaje: str
    codigo_error: str
    detalles: Optional[Dict[str, Any]] = None

class RespuestaProgresoUsuario(BaseModel):
    """Respuesta del progreso del usuario"""
    nivel_actual: int
    lecciones_completadas: int
    total_lecciones: int
    puntos_totales: int
    porcentaje_progreso: float
    estrellas_doradas: int
    racha_actual: int
    ultima_leccion: Optional[Dict[str, Any]] = None
    proxima_leccion: Optional[Dict[str, Any]] = None
    categorias_progreso: Optional[List[Dict[str, Any]]] = None

class RespuestaPractica(BaseModel):
    """Respuesta después de una práctica"""
    exito: bool
    precision: float
    puntos_ganados: int
    feedback: str
    mensaje: str
    es_perfecto: bool
    nueva_mejor_marca: bool
    progreso_actualizado: Optional[Dict[str, Any]] = None

class RespuestaReconocimiento(BaseModel):
    """Respuesta del sistema de reconocimiento"""
    sena_detectada: str
    confianza: float
    es_correcta: bool
    precision: float
    mensaje: str
    puntos_mano: Optional[Dict[str, Any]] = None