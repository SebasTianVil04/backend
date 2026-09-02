from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class EstadisticasUsuario(BaseModel):
    """Estadísticas detalladas del usuario"""
    usuario_id: int
    nombre_completo: str
    email: str
    
    # Progreso general
    total_lecciones_completadas: int = 0
    total_lecciones_disponibles: int = 0
    porcentaje_progreso: float = 0.0
    
    # Prácticas
    total_practicas: int = 0
    precision_promedio: float = 0.0
    mejor_precision: float = 0.0
    estrellas_doradas: int = 0
    
    # Puntos y nivel
    puntos_totales: int = 0
    nivel_actual: int = 1
    experiencia_total: int = 0
    
    # Exámenes
    examenes_completados: int = 0
    examenes_aprobados: int = 0
    tasa_aprobacion: float = 0.0
    
    # Actividad
    racha_actual: int = 0
    racha_maxima: int = 0
    dias_consecutivos_activo: int = 0
    tiempo_total_estudio: int = 0
    
    # Categorías
    categorias_dominadas: int = 0
    categoria_favorita: Optional[str] = None
    
    # Fechas
    fecha_registro: datetime
    ultima_actividad: Optional[datetime] = None
    ultima_leccion_completada: Optional[datetime] = None
    
    # Ranking
    posicion_ranking: Optional[int] = None
    percentil_rendimiento: Optional[float] = None
    
    class Config:
        from_attributes = True

class EstadisticasCategoria(BaseModel):
    """Estadísticas por categoría"""
    categoria_id: int
    categoria_nombre: str
    total_lecciones: int
    lecciones_completadas: int
    porcentaje_completado: float
    precision_promedio: float
    puntos_ganados: int
    tiempo_invertido: int
    
class EstadisticasGlobales(BaseModel):
    """Estadísticas del sistema completo"""
    total_usuarios: int
    usuarios_activos: int
    total_lecciones: int
    total_categorias: int
    total_examenes: int
    total_practicas: int
    precision_promedio_global: float
    categoria_mas_popular: Optional[str] = None
    leccion_mas_practicada: Optional[str] = None
