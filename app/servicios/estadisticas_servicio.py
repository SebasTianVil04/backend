import logging
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from app.modelos import Leccion, SesionEstudio
from app.modelos.categoria import Categoria


logger = logging.getLogger(__name__)


class EstadisticasServicio:
    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _fin_del_dia(fecha: datetime) -> datetime:
        return fecha.replace(hour=23, minute=59, second=59, microsecond=999999)

    def obtener_estadisticas_generales(
        self,
        fecha_inicio: datetime,
        fecha_fin: datetime,
    ) -> Dict[str, Any]:
        try:
            fila = (
                self.db.query(
                    func.count(SesionEstudio.id).label("total_sesiones"),
                    func.sum(
                        case((SesionEstudio.fecha_fin.isnot(None), 1), else_=0)
                    ).label("sesiones_completadas"),
                    func.count(
                        func.distinct(SesionEstudio.usuario_id)
                    ).label("usuarios_activos"),
                    func.coalesce(
                        func.avg(SesionEstudio.duracion_segundos), 0
                    ).label("tiempo_promedio_segundos"),
                    func.coalesce(
                        func.sum(SesionEstudio.duracion_segundos), 0
                    ).label("tiempo_total_segundos"),
                )
                .filter(
                    SesionEstudio.fecha_inicio >= fecha_inicio,
                    SesionEstudio.fecha_inicio <= self._fin_del_dia(fecha_fin),
                )
                .first()
            )

            total_sesiones = int(fila.total_sesiones or 0)
            sesiones_completadas = int(fila.sesiones_completadas or 0)
            usuarios_activos = int(fila.usuarios_activos or 0)
            tiempo_promedio_segundos = float(fila.tiempo_promedio_segundos or 0)
            tiempo_total_segundos = float(fila.tiempo_total_segundos or 0)

            porcentaje_completadas = (
                round(sesiones_completadas / total_sesiones * 100, 1)
                if total_sesiones > 0
                else 0.0
            )

            return {
                "total_sesiones": total_sesiones,
                "sesiones_completadas": sesiones_completadas,
                "porcentaje_completadas": porcentaje_completadas,
                "usuarios_activos": usuarios_activos,
                "tiempo_promedio_segundos": tiempo_promedio_segundos,
                "tiempo_total_segundos": tiempo_total_segundos,
                "tiempo_promedio_minutos": round(tiempo_promedio_segundos / 60.0, 1),
                "tiempo_total_minutos": round(tiempo_total_segundos / 60.0, 1),
            }

        except Exception:
            logger.exception("Error obteniendo estadísticas generales")
            return {
                "total_sesiones": 0,
                "sesiones_completadas": 0,
                "porcentaje_completadas": 0,
                "usuarios_activos": 0,
                "tiempo_promedio_segundos": 0,
                "tiempo_total_segundos": 0,
                "tiempo_promedio_minutos": 0,
                "tiempo_total_minutos": 0,
            }

    def obtener_lecciones_populares(
        self,
        fecha_inicio: datetime,
        fecha_fin: datetime,
    ) -> List[Dict[str, Any]]:
        try:
            filas = (
                self.db.query(
                    Leccion.id.label("leccion_id"),
                    Leccion.titulo.label("titulo"),
                    Categoria.nombre.label("categoria"),
                    func.count(SesionEstudio.id).label("completadas"),
                    func.count(
                        func.distinct(SesionEstudio.usuario_id)
                    ).label("usuarios_unicos"),
                    func.coalesce(
                        func.avg(SesionEstudio.duracion_segundos), 0
                    ).label("tiempo_promedio_segundos"),
                )
                .join(SesionEstudio, SesionEstudio.leccion_id == Leccion.id)
                .join(Categoria, Categoria.id == Leccion.categoria_id)
                .filter(
                    SesionEstudio.fecha_inicio >= fecha_inicio,
                    SesionEstudio.fecha_inicio <= self._fin_del_dia(fecha_fin),
                    SesionEstudio.leccion_id.isnot(None),
                )
                .group_by(Leccion.id, Leccion.titulo, Categoria.nombre)
                .order_by(desc(func.count(SesionEstudio.id)))
                .all()
            )

            resultado: List[Dict[str, Any]] = []
            for fila in filas:
                segundos = float(fila.tiempo_promedio_segundos or 0)
                completadas = int(fila.completadas or 0)

                if completadas >= 10:
                    popularidad = "Alta"
                elif completadas >= 5:
                    popularidad = "Media"
                else:
                    popularidad = "Baja"

                resultado.append(
                    {
                        "leccion_id": fila.leccion_id,
                        "titulo": fila.titulo,
                        "categoria": fila.categoria or "Sin categoría",
                        "completadas": completadas,
                        "usuarios_unicos": int(fila.usuarios_unicos or 0),
                        "tiempo_promedio_segundos": segundos,
                        "tiempo_promedio_minutos": round(segundos / 60.0, 1),
                        "popularidad": popularidad,
                    }
                )

            return resultado

        except Exception:
            logger.exception("Error obteniendo lecciones populares")
            return []

    def formatear_tiempo_para_display(self, minutos: float) -> str:
        if minutos == 0:
            return "0m"

        if minutos < 1:
            return f"{int(minutos * 60)}s"

        if minutos < 60:
            return f"{int(minutos)}m"

        horas = int(minutos // 60)
        mins_restantes = int(minutos % 60)
        return f"{horas}h" if mins_restantes == 0 else f"{horas}h {mins_restantes}m"