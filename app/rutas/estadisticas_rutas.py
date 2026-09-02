from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel

from app.servicios.estadisticas_servicio import EstadisticasServicio
from app.utilidades.base_datos import get_db

router = APIRouter(prefix="/estadisticas", tags=["estadisticas"])

# Modelos Pydantic para las respuestas
class EstadisticasGeneralesResponse(BaseModel):
    total_sesiones: int
    sesiones_completadas: int
    porcentaje_completadas: float
    usuarios_activos: int
    tiempo_promedio_segundos: float
    tiempo_total_segundos: float
    tiempo_promedio_minutos: float
    tiempo_total_minutos: float
    tiempo_promedio_display: str
    tiempo_total_display: str

class LeccionPopularResponse(BaseModel):
    leccion_id: int
    titulo: str
    categoria: str
    completadas: int
    usuarios_unicos: int
    tiempo_promedio_segundos: float
    tiempo_promedio_minutos: float
    popularidad: str

class EstadisticasResponse(BaseModel):
    success: bool
    fecha_inicio: str
    fecha_fin: str
    estadisticas_generales: EstadisticasGeneralesResponse
    lecciones_populares: List[LeccionPopularResponse]

@router.get("/", response_model=EstadisticasResponse)
async def obtener_estadisticas(
    fecha_inicio: Optional[str] = Query(None, description="Fecha inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha fin (YYYY-MM-DD)"),
    db: Session = Depends(get_db)  # ✅ FastAPI maneja el generador automáticamente
):
    """Endpoint principal para obtener estadísticas en formato JSON para Angular"""
    
    # Obtener parámetros de fecha o usar valores por defecto (últimos 30 días)
    fecha_fin_default = datetime.now()
    fecha_inicio_default = fecha_fin_default - timedelta(days=30)
    
    fecha_inicio_dt = fecha_inicio_default
    fecha_fin_dt = fecha_fin_default
    
    # Convertir parámetros string a datetime
    if fecha_inicio:
        try:
            fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha_inicio inválido. Use YYYY-MM-DD")
    
    if fecha_fin:
        try:
            fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha_fin inválido. Use YYYY-MM-DD")
    
    # ✅ CORRECCIÓN: db ya es una Session, no un generator
    servicio = EstadisticasServicio(db)
    
    try:
        stats_generales = servicio.obtener_estadisticas_generales(fecha_inicio_dt, fecha_fin_dt)
        lecciones_populares = servicio.obtener_lecciones_populares(fecha_inicio_dt, fecha_fin_dt)
        
        # Formatear tiempos para display
        stats_generales['tiempo_promedio_display'] = servicio.formatear_tiempo_para_display(
            stats_generales['tiempo_promedio_minutos']
        )
        stats_generales['tiempo_total_display'] = servicio.formatear_tiempo_para_display(
            stats_generales['tiempo_total_minutos']
        )
        
        return {
            "success": True,
            "fecha_inicio": fecha_inicio_dt.strftime('%Y-%m-%d'),
            "fecha_fin": fecha_fin_dt.strftime('%Y-%m-%d'),
            "estadisticas_generales": stats_generales,
            "lecciones_populares": lecciones_populares
        }
        
    except Exception as e:
        print(f"Error en endpoint de estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

@router.get("/rango-fechas", response_model=EstadisticasResponse)
async def obtener_estadisticas_rango_fechas(
    fecha_inicio: str = Query(..., description="Fecha inicio (YYYY-MM-DD)"),
    fecha_fin: str = Query(..., description="Fecha fin (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Endpoint alternativo con parámetros requeridos"""
    
    try:
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")
    
    servicio = EstadisticasServicio(db)
    
    try:
        stats_generales = servicio.obtener_estadisticas_generales(fecha_inicio_dt, fecha_fin_dt)
        lecciones_populares = servicio.obtener_lecciones_populares(fecha_inicio_dt, fecha_fin_dt)
        
        # Formatear tiempos para display
        stats_generales['tiempo_promedio_display'] = servicio.formatear_tiempo_para_display(
            stats_generales['tiempo_promedio_minutos']
        )
        stats_generales['tiempo_total_display'] = servicio.formatear_tiempo_para_display(
            stats_generales['tiempo_total_minutos']
        )
        
        return {
            "success": True,
            "fecha_inicio": fecha_inicio_dt.strftime('%Y-%m-%d'),
            "fecha_fin": fecha_fin_dt.strftime('%Y-%m-%d'),
            "estadisticas_generales": stats_generales,
            "lecciones_populares": lecciones_populares
        }
        
    except Exception as e:
        print(f"Error en endpoint de estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")