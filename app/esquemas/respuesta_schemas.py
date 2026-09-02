from pydantic import BaseModel
from typing import Any, Optional, List, Dict

class RespuestaAPI(BaseModel):
    """Esquema base para respuestas de la API"""
    exito: bool
    mensaje: str
    datos: Optional[Any] = None
    errores: Optional[List[str]] = None

class RespuestaLista(BaseModel):
    """Esquema para respuestas que contienen listas"""
    exito: bool
    mensaje: str
    datos: List[Any]
    total: int
    pagina: int = 1
    por_pagina: int = 10

class RespuestaError(BaseModel):
    """Esquema para respuestas de error"""
    exito: bool = False
    mensaje: str
    codigo_error: str
    detalles: Optional[Dict[str, Any]] = None

class RespuestaProgresoUsuario(BaseModel):
    """Respuesta específica para el progreso del usuario"""
    nivel_actual: int
    lecciones_completadas: int
    total_lecciones: int
    puntos_totales: int
    porcentaje_progreso: float
    ultima_leccion: Optional[str] = None
    proxima_leccion: Optional[str] = None

class RespuestaEstadisticas(BaseModel):
    """Respuesta para estadísticas del sistema"""
    total_usuarios: int
    usuarios_activos: int
    total_lecciones: int
    total_examenes: int
    promedio_puntuacion: float
