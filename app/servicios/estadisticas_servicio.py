from sqlalchemy import func, desc, case
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.modelos import SesionEstudio, Leccion, Usuario
from typing import List, Dict, Any, Optional

class EstadisticasServicio:
    def __init__(self, db: Session):
        self.db = db
    
    def obtener_estadisticas_generales(self, fecha_inicio: datetime, fecha_fin: datetime) -> Dict[str, Any]:
        """Obtiene estadísticas generales de uso del sistema - VERSIÓN SIMPLIFICADA"""
        
        try:
            # 🔥 VERSIÓN SIMPLIFICADA Y ROBUSTA
            # Contar sesiones totales
            total_sesiones = self.db.query(func.count(SesionEstudio.id)).filter(
                SesionEstudio.fecha_inicio >= fecha_inicio,
                SesionEstudio.fecha_inicio <= fecha_fin.replace(hour=23, minute=59, second=59)
            ).scalar() or 0
            
            # Contar sesiones completadas
            sesiones_completadas = self.db.query(func.count(SesionEstudio.id)).filter(
                SesionEstudio.fecha_inicio >= fecha_inicio,
                SesionEstudio.fecha_inicio <= fecha_fin.replace(hour=23, minute=59, second=59),
                SesionEstudio.fecha_fin.isnot(None)
            ).scalar() or 0
            
            # Usuarios activos
            usuarios_activos = self.db.query(func.count(func.distinct(SesionEstudio.usuario_id))).filter(
                SesionEstudio.fecha_inicio >= fecha_inicio,
                SesionEstudio.fecha_inicio <= fecha_fin.replace(hour=23, minute=59, second=59)
            ).scalar() or 0
            
            # Tiempos
            tiempo_result = self.db.query(
                func.coalesce(func.avg(SesionEstudio.duracion_segundos), 0).label('promedio'),
                func.coalesce(func.sum(SesionEstudio.duracion_segundos), 0).label('total')
            ).filter(
                SesionEstudio.fecha_inicio >= fecha_inicio,
                SesionEstudio.fecha_inicio <= fecha_fin.replace(hour=23, minute=59, second=59)
            ).first()
            
            tiempo_promedio_segundos = float(tiempo_result.promedio) if tiempo_result else 0
            tiempo_total_segundos = float(tiempo_result.total) if tiempo_result else 0
            
            # Calcular porcentaje
            porcentaje_completadas = 0
            if total_sesiones > 0:
                porcentaje_completadas = (sesiones_completadas / total_sesiones) * 100
            
            # Convertir a minutos
            tiempo_promedio_minutos = tiempo_promedio_segundos / 60.0
            tiempo_total_minutos = tiempo_total_segundos / 60.0
            
            return {
                'total_sesiones': total_sesiones,
                'sesiones_completadas': sesiones_completadas,
                'porcentaje_completadas': round(porcentaje_completadas, 1),
                'usuarios_activos': usuarios_activos,
                'tiempo_promedio_segundos': tiempo_promedio_segundos,
                'tiempo_total_segundos': tiempo_total_segundos,
                'tiempo_promedio_minutos': round(tiempo_promedio_minutos, 1),
                'tiempo_total_minutos': round(tiempo_total_minutos, 1)
            }
            
        except Exception as e:
            print(f"Error obteniendo estadísticas generales: {e}")
            return {
                'total_sesiones': 0,
                'sesiones_completadas': 0,
                'porcentaje_completadas': 0,
                'usuarios_activos': 0,
                'tiempo_promedio_segundos': 0,
                'tiempo_total_segundos': 0,
                'tiempo_promedio_minutos': 0,
                'tiempo_total_minutos': 0
            }
    
    def obtener_lecciones_populares(self, fecha_inicio: datetime, fecha_fin: datetime) -> List[Dict[str, Any]]:
        """Obtiene las lecciones más populares con estadísticas - VERSIÓN SIMPLIFICADA"""
        
        try:
            # 🔥 VERSIÓN SIMPLIFICADA CON CONSULTAS SEPARADAS
            lecciones_data = self.db.query(
                Leccion.id,
                Leccion.titulo,
                Leccion.categoria,
                func.count(SesionEstudio.id).label('sesiones_count'),
                func.count(func.distinct(SesionEstudio.usuario_id)).label('usuarios_count')
            ).join(SesionEstudio, SesionEstudio.leccion_id == Leccion.id).filter(
                SesionEstudio.fecha_inicio >= fecha_inicio,
                SesionEstudio.fecha_inicio <= fecha_fin.replace(hour=23, minute=59, second=59),
                SesionEstudio.leccion_id.isnot(None)
            ).group_by(Leccion.id, Leccion.titulo, Leccion.categoria).order_by(
                desc(func.count(SesionEstudio.id))
            ).all()
            
            resultado = []
            for leccion in lecciones_data:
                # Obtener tiempo promedio para esta lección
                tiempo_promedio_result = self.db.query(
                    func.coalesce(func.avg(SesionEstudio.duracion_segundos), 0)
                ).filter(
                    SesionEstudio.leccion_id == leccion.id,
                    SesionEstudio.fecha_inicio >= fecha_inicio,
                    SesionEstudio.fecha_inicio <= fecha_fin.replace(hour=23, minute=59, second=59)
                ).scalar()
                
                tiempo_promedio_segundos = float(tiempo_promedio_result) if tiempo_promedio_result else 0
                tiempo_promedio_minutos = tiempo_promedio_segundos / 60.0
                
                # Calcular popularidad
                if leccion.sesiones_count >= 10:
                    popularidad = "Alta"
                elif leccion.sesiones_count >= 5:
                    popularidad = "Media"
                else:
                    popularidad = "Baja"
                
                # Manejar categoría
                categoria_str = str(leccion.categoria)
                if hasattr(leccion.categoria, 'value'):
                    categoria_str = leccion.categoria.value
                elif hasattr(leccion.categoria, 'name'):
                    categoria_str = leccion.categoria.name
                
                resultado.append({
                    'leccion_id': leccion.id,
                    'titulo': leccion.titulo,
                    'categoria': categoria_str,
                    'completadas': leccion.sesiones_count,
                    'usuarios_unicos': leccion.usuarios_count,
                    'tiempo_promedio_segundos': tiempo_promedio_segundos,
                    'tiempo_promedio_minutos': round(tiempo_promedio_minutos, 1),
                    'popularidad': popularidad
                })
            
            return resultado
            
        except Exception as e:
            print(f"Error obteniendo lecciones populares: {e}")
            return []
    
    def formatear_tiempo_para_display(self, minutos: float) -> str:
        """Formatea el tiempo para mostrar en la interfaz"""
        if minutos == 0:
            return "0m"
        
        if minutos < 1:
            segundos = int(minutos * 60)
            return f"{segundos}s"
        elif minutos < 60:
            return f"{int(minutos)}m"
        else:
            horas = int(minutos // 60)
            mins_restantes = int(minutos % 60)
            if mins_restantes == 0:
                return f"{horas}h"
            else:
                return f"{horas}h {mins_restantes}m"