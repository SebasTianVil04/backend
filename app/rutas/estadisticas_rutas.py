import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.servicios.estadisticas_servicio import EstadisticasServicio
from app.utilidades.base_datos import get_db


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/estadisticas", tags=["estadisticas"])

FORMATO_FECHA = "%Y-%m-%d"
DIAS_POR_DEFECTO = 30


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


def _parsear_fecha(valor: str, campo: str) -> datetime:
    try:
        return datetime.strptime(valor, FORMATO_FECHA)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de {campo} inválido. Use YYYY-MM-DD",
        )


def _resolver_rango(
    fecha_inicio: Optional[str],
    fecha_fin: Optional[str],
) -> tuple:
    fecha_fin_dt = datetime.now()
    fecha_inicio_dt = fecha_fin_dt - timedelta(days=DIAS_POR_DEFECTO)

    if fecha_inicio:
        fecha_inicio_dt = _parsear_fecha(fecha_inicio, "fecha_inicio")
    if fecha_fin:
        fecha_fin_dt = _parsear_fecha(fecha_fin, "fecha_fin")

    if fecha_inicio_dt > fecha_fin_dt:
        raise HTTPException(
            status_code=400,
            detail="fecha_inicio no puede ser posterior a fecha_fin",
        )

    return fecha_inicio_dt, fecha_fin_dt


def _construir_respuesta(db: Session, fecha_inicio_dt: datetime, fecha_fin_dt: datetime) -> dict:
    servicio = EstadisticasServicio(db)
    stats_generales = servicio.obtener_estadisticas_generales(fecha_inicio_dt, fecha_fin_dt)
    lecciones_populares = servicio.obtener_lecciones_populares(fecha_inicio_dt, fecha_fin_dt)

    stats_generales["tiempo_promedio_display"] = servicio.formatear_tiempo_para_display(
        stats_generales["tiempo_promedio_minutos"]
    )
    stats_generales["tiempo_total_display"] = servicio.formatear_tiempo_para_display(
        stats_generales["tiempo_total_minutos"]
    )

    return {
        "success": True,
        "fecha_inicio": fecha_inicio_dt.strftime(FORMATO_FECHA),
        "fecha_fin": fecha_fin_dt.strftime(FORMATO_FECHA),
        "estadisticas_generales": stats_generales,
        "lecciones_populares": lecciones_populares,
    }


@router.get("/", response_model=EstadisticasResponse)
async def obtener_estadisticas(
    fecha_inicio: Optional[str] = Query(None, description="Fecha inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha fin (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
):
    inicio_dt, fin_dt = _resolver_rango(fecha_inicio, fecha_fin)

    try:
        return _construir_respuesta(db, inicio_dt, fin_dt)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error en endpoint de estadísticas: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estadísticas: {str(e)}",
        )


@router.get("/rango-fechas", response_model=EstadisticasResponse)
async def obtener_estadisticas_rango_fechas(
    fecha_inicio: str = Query(..., description="Fecha inicio (YYYY-MM-DD)"),
    fecha_fin: str = Query(..., description="Fecha fin (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
):
    inicio_dt, fin_dt = _resolver_rango(fecha_inicio, fecha_fin)

    try:
        return _construir_respuesta(db, inicio_dt, fin_dt)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error en endpoint de estadísticas: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estadísticas: {str(e)}",
        )