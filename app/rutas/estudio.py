# app/rutas/estudio.py - VERSIÓN CORREGIDA CON CÁLCULO AUTOMÁTICO DE DURACIÓN
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field

from app.utilidades.base_datos import obtener_bd
from app.modelos.estudio import SesionEstudio
from app.modelos.usuario import Usuario
from app.modelos.clase import Clase
from app.modelos.leccion import Leccion
from app.utilidades.seguridad import obtener_usuario_actual

router = APIRouter(prefix="/estudio", tags=["Tiempo de Estudio"])

# Modelos Pydantic para validación
class SesionEstudioCrear(BaseModel):
    clase_id: Optional[int] = Field(None, ge=1)
    leccion_id: Optional[int] = Field(None, ge=1)
    tipo_sesion: str = Field(..., pattern="^(clase|practica|examen|video)$")
    fecha_inicio: str
    fecha_fin: Optional[str] = None  # Hacer opcional para sesiones en progreso
    duracion_segundos: Optional[int] = Field(None, ge=0)  # Hacer opcional
    dispositivo: Optional[str] = None
    user_agent: Optional[str] = None

class SesionEstudioResponse(BaseModel):
    exito: bool
    mensaje: str
    sesion_id: int
    duracion_segundos: int

class TiempoTotalResponse(BaseModel):
    tiempo_total_segundos: int
    tiempo_total_formateado: str
    total_sesiones: int

class EstadisticasTiempoResponse(BaseModel):
    rango_dias: int
    tiempo_total_segundos: int
    tiempo_total_formateado: str
    sesiones_por_tipo: List[dict]
    tiempo_por_dia: List[dict]
    promedio_diario_segundos: int
    promedio_diario_formateado: str
    total_sesiones: int

@router.post("/sesiones", response_model=SesionEstudioResponse)
async def crear_sesion_estudio(
    sesion_data: SesionEstudioCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Crear una nueva sesión de estudio con validación de datos"""
    try:
        # Validar que al menos un ID esté presente
        if not sesion_data.clase_id and not sesion_data.leccion_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requiere al menos clase_id o leccion_id"
            )

        # Validar que clase_id existe si está presente
        if sesion_data.clase_id:
            clase = db.query(Clase).filter(Clase.id == sesion_data.clase_id).first()
            if not clase:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"clase_id {sesion_data.clase_id} no existe"
                )

        # Validar que leccion_id existe si está presente
        if sesion_data.leccion_id:
            leccion = db.query(Leccion).filter(Leccion.id == sesion_data.leccion_id).first()
            if not leccion:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"leccion_id {sesion_data.leccion_id} no existe"
                )

        # Validar fecha_inicio
        try:
            fecha_inicio = datetime.fromisoformat(sesion_data.fecha_inicio.replace('Z', '+00:00'))
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Formato de fecha_inicio inválido: {str(e)}"
            )

        # Manejar fecha_fin y calcular duración
        fecha_fin = None
        duracion_segundos = sesion_data.duracion_segundos or 0

        if sesion_data.fecha_fin:
            try:
                fecha_fin = datetime.fromisoformat(sesion_data.fecha_fin.replace('Z', '+00:00'))
                
                # Validar que fecha_fin es posterior a fecha_inicio
                if fecha_fin <= fecha_inicio:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="fecha_fin debe ser posterior a fecha_inicio"
                    )
                
                # 🔥 CALCULAR DURACIÓN AUTOMÁTICAMENTE SI NO SE PROPORCIONA
                if not sesion_data.duracion_segundos:
                    duracion_segundos = int((fecha_fin - fecha_inicio).total_seconds())
                    print(f"✅ Duración calculada automáticamente: {duracion_segundos}s")
                
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Formato de fecha_fin inválido: {str(e)}"
                )
        else:
            # Si no hay fecha_fin, la sesión está en progreso
            print("⚠️ Sesión creada sin fecha_fin (en progreso)")

        # 🔥 VALIDAR DURACIÓN MÍNIMA (evitar sesiones de 0 segundos)
        if duracion_segundos == 0 and fecha_fin:
            print("⚠️ Advertencia: Sesión con duración 0 segundos")
            # Forzar una duración mínima de 1 segundo si las fechas son diferentes
            if fecha_fin > fecha_inicio:
                duracion_segundos = 1
                print("✅ Duración mínima aplicada: 1s")

        # Crear sesión
        sesion = SesionEstudio(
            usuario_id=usuario_actual.id,
            clase_id=sesion_data.clase_id,
            leccion_id=sesion_data.leccion_id,
            tipo_sesion=sesion_data.tipo_sesion,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            duracion_segundos=duracion_segundos,
            dispositivo=sesion_data.dispositivo,
            user_agent=sesion_data.user_agent
        )
        
        db.add(sesion)
        db.commit()
        db.refresh(sesion)
        
        print(f"✅ Sesión guardada - Usuario: {usuario_actual.id}, Duración: {duracion_segundos}s")
        
        return SesionEstudioResponse(
            exito=True,
            mensaje="Sesión guardada correctamente",
            sesion_id=sesion.id,
            duracion_segundos=duracion_segundos
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error guardando sesión: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar sesión: {str(e)}"
        )

@router.put("/sesiones/{sesion_id}/finalizar")
async def finalizar_sesion_estudio(
    sesion_id: int,
    fecha_fin: str,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Finalizar una sesión de estudio en progreso"""
    try:
        # Buscar sesión
        sesion = db.query(SesionEstudio).filter(
            SesionEstudio.id == sesion_id,
            SesionEstudio.usuario_id == usuario_actual.id
        ).first()
        
        if not sesion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sesión no encontrada"
            )
        
        if sesion.fecha_fin is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La sesión ya está finalizada"
            )
        
        # Validar fecha_fin
        try:
            fecha_fin_dt = datetime.fromisoformat(fecha_fin.replace('Z', '+00:00'))
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Formato de fecha_fin inválido: {str(e)}"
            )
        
        # Calcular duración
        duracion_segundos = int((fecha_fin_dt - sesion.fecha_inicio).total_seconds())
        
        # Validar duración mínima
        if duracion_segundos <= 0:
            duracion_segundos = 1  # Duración mínima
        
        # Actualizar sesión
        sesion.fecha_fin = fecha_fin_dt
        sesion.duracion_segundos = duracion_segundos
        
        db.commit()
        
        print(f"✅ Sesión finalizada - ID: {sesion_id}, Duración: {duracion_segundos}s")
        
        return {
            "exito": True,
            "mensaje": "Sesión finalizada correctamente",
            "duracion_segundos": duracion_segundos
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error finalizando sesión: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al finalizar sesión: {str(e)}"
        )

@router.get("/usuario/total", response_model=TiempoTotalResponse)
async def obtener_tiempo_total_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener tiempo total de estudio del usuario"""
    try:
        # 🔥 FILTRAR SOLO SESIONES CON DURACIÓN > 0
        tiempo_total = db.query(func.sum(SesionEstudio.duracion_segundos)).filter(
            SesionEstudio.usuario_id == usuario_actual.id,
            SesionEstudio.duracion_segundos > 0  # Solo sesiones con duración
        ).scalar() or 0
        
        total_sesiones = db.query(SesionEstudio).filter(
            SesionEstudio.usuario_id == usuario_actual.id,
            SesionEstudio.duracion_segundos > 0  # Solo sesiones con duración
        ).count()
        
        return TiempoTotalResponse(
            tiempo_total_segundos=int(tiempo_total),
            tiempo_total_formateado=formatear_tiempo(int(tiempo_total)),
            total_sesiones=total_sesiones
        )
        
    except Exception as e:
        print(f"Error obteniendo tiempo total: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener tiempo total: {str(e)}"
        )

@router.get("/estadisticas", response_model=EstadisticasTiempoResponse)
async def obtener_estadisticas_tiempo(
    rango_dias: Optional[int] = 30,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener estadísticas detalladas de tiempo de estudio"""
    try:
        # Validar rango de días
        if rango_dias < 1 or rango_dias > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="rango_dias debe estar entre 1 y 365"
            )
        
        fecha_inicio = datetime.now() - timedelta(days=rango_dias)
        
        # 🔥 FILTRAR SOLO SESIONES CON DURACIÓN > 0
        tiempo_total = db.query(func.sum(SesionEstudio.duracion_segundos)).filter(
            and_(
                SesionEstudio.usuario_id == usuario_actual.id,
                SesionEstudio.fecha_inicio >= fecha_inicio,
                SesionEstudio.duracion_segundos > 0  # Solo sesiones con duración
            )
        ).scalar() or 0
        
        # Sesiones por tipo
        sesiones_por_tipo_query = db.query(
            SesionEstudio.tipo_sesion,
            func.count(SesionEstudio.id),
            func.sum(SesionEstudio.duracion_segundos)
        ).filter(
            and_(
                SesionEstudio.usuario_id == usuario_actual.id,
                SesionEstudio.fecha_inicio >= fecha_inicio,
                SesionEstudio.duracion_segundos > 0  # Solo sesiones con duración
            )
        ).group_by(SesionEstudio.tipo_sesion).all()
        
        sesiones_por_tipo = []
        for tipo, cantidad, tiempo in sesiones_por_tipo_query:
            sesiones_por_tipo.append({
                "tipo": tipo,
                "cantidad": cantidad,
                "tiempo_total": int(tiempo or 0),
                "tiempo_formateado": formatear_tiempo(int(tiempo or 0))
            })
        
        # Tiempo por día (últimos 7 días)
        tiempo_por_dia = []
        dias_a_mostrar = min(7, rango_dias)
        
        for i in range(dias_a_mostrar):
            fecha = datetime.now() - timedelta(days=i)
            fecha_inicio_dia = fecha.replace(hour=0, minute=0, second=0, microsecond=0)
            fecha_fin_dia = fecha.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            tiempo_dia = db.query(func.sum(SesionEstudio.duracion_segundos)).filter(
                and_(
                    SesionEstudio.usuario_id == usuario_actual.id,
                    SesionEstudio.fecha_inicio >= fecha_inicio_dia,
                    SesionEstudio.fecha_inicio <= fecha_fin_dia,
                    SesionEstudio.duracion_segundos > 0  # Solo sesiones con duración
                )
            ).scalar() or 0
            
            tiempo_por_dia.append({
                "fecha": fecha_inicio_dia.date().isoformat(),
                "tiempo_segundos": int(tiempo_dia),
                "tiempo_formateado": formatear_tiempo(int(tiempo_dia))
            })
        
        # Calcular promedio diario
        promedio_diario = int(tiempo_total / rango_dias) if rango_dias > 0 else 0
        
        return EstadisticasTiempoResponse(
            rango_dias=rango_dias,
            tiempo_total_segundos=int(tiempo_total),
            tiempo_total_formateado=formatear_tiempo(int(tiempo_total)),
            sesiones_por_tipo=sesiones_por_tipo,
            tiempo_por_dia=list(reversed(tiempo_por_dia)),
            promedio_diario_segundos=promedio_diario,
            promedio_diario_formateado=formatear_tiempo(promedio_diario),
            total_sesiones=sum(item["cantidad"] for item in sesiones_por_tipo)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )

def formatear_tiempo(segundos: int) -> str:
    """Formatear segundos a formato legible"""
    if segundos == 0:
        return "0m"
    
    if segundos < 60:
        return f"{segundos}s"
    
    minutos = segundos // 60
    segundos_restantes = segundos % 60
    
    if minutos < 60:
        if segundos_restantes > 0:
            return f"{minutos}m {segundos_restantes}s"
        return f"{minutos}m"
    
    horas = minutos // 60
    minutos_restantes = minutos % 60
    
    if minutos_restantes > 0:
        return f"{horas}h {minutos_restantes}m"
    return f"{horas}h"