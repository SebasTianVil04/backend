from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


class ProgresoClase(AuditoriaMixin, Base):
    __tablename__ = "progreso_clases"
    __table_args__ = (UniqueConstraint("usuario_id", "clase_id", name="uq_progreso_clase_usuario_clase"),)

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False, index=True)
    clase_id = Column(Integer, ForeignKey("clases.id"), nullable=False, index=True)

    vista = Column(Boolean, default=False)
    completada = Column(Boolean, default=False)
    aprobada = Column(Boolean, default=False)

    intentos_realizados = Column(Integer, default=0)
    intentos_exitosos = Column(Integer, default=0)
    mejor_precision = Column(Float, default=0.0)
    ultima_precision = Column(Float, default=0.0)
    precision_promedio = Column(Float, default=0.0)

    puntos_ganados = Column(Integer, default=0)
    xp_ganado = Column(Integer, default=0)
    tiempo_total_practica = Column(Integer, default=0)

    fecha_primera_vista = Column(DateTime(timezone=True), nullable=True)
    fecha_completada = Column(DateTime(timezone=True), nullable=True)
    fecha_mejor_precision = Column(DateTime(timezone=True), nullable=True)
    ultima_practica = Column(DateTime(timezone=True), nullable=True)

    usuario = relationship("Usuario", back_populates="progresos_clase", foreign_keys=[usuario_id])
    clase = relationship("Clase", back_populates="progresos_clase")

    @property
    def porcentaje_precision(self):
        return f"{int((self.mejor_precision or 0) * 100)}%"

    @property
    def tasa_exito(self):
        if not self.intentos_realizados:
            return 0.0
        return (self.intentos_exitosos or 0) / self.intentos_realizados

    @property
    def porcentaje_tasa_exito(self):
        return f"{int(self.tasa_exito * 100)}%"

    @property
    def eficiencia_puntos(self):
        if not self.tiempo_total_practica:
            return 0.0
        return (self.puntos_ganados or 0) / max(self.tiempo_total_practica / 60, 1)

    @property
    def nivel_dominio(self):
        precision = self.mejor_precision or 0
        if precision >= 0.9:
            return "Perfecto"
        if precision >= 0.8:
            return "Excelente"
        if precision >= 0.7:
            return "Bueno"
        if precision >= 0.6:
            return "Aceptable"
        return "Necesita Mejorar"

    @property
    def dias_desde_ultima_practica(self):
        ultima = self.ultima_practica
        if not ultima:
            return 999
        if ultima.tzinfo is None:
            ultima = ultima.replace(tzinfo=timezone.utc)
        return (_ahora() - ultima).days

    @property
    def esta_en_racha(self):
        return self.dias_desde_ultima_practica <= 1

    @property
    def tiempo_promedio_intento(self):
        if not self.intentos_realizados:
            return 0
        return (self.tiempo_total_practica or 0) // self.intentos_realizados

    @property
    def racha_dias_consecutivos(self):
        if self.dias_desde_ultima_practica <= 1:
            return min(getattr(self, "_racha_calculada", 1), 30)
        return 0

    @property
    def mejor_racha(self):
        return max(getattr(self, "_mejor_racha_calculada", 0), self.racha_dias_consecutivos)

    def registrar_intento(
        self,
        precision: float,
        duracion_segundos: int,
        clase_obj=None,
        es_exitoso: Optional[bool] = None,
    ):
        ahora = _ahora()

        self.intentos_realizados = (self.intentos_realizados or 0) + 1
        self.ultima_precision = precision
        self.tiempo_total_practica = (self.tiempo_total_practica or 0) + duracion_segundos

        if precision > (self.mejor_precision or 0):
            self.mejor_precision = precision
            self.fecha_mejor_precision = ahora

        n = self.intentos_realizados
        self.precision_promedio = ((self.precision_promedio or 0) * (n - 1) + precision) / n

        if es_exitoso is None:
            precision_minima = getattr(clase_obj, "precision_minima", None) if clase_obj else None
            es_exitoso = precision >= (0.7 if precision_minima is None else precision_minima)

        puntos_base = 10
        xp_base = 5

        if es_exitoso:
            self.intentos_exitosos = (self.intentos_exitosos or 0) + 1
            puntos_ganados = int(puntos_base * precision)
            xp_ganado = int(xp_base * precision)
        else:
            puntos_ganados = int(puntos_base * precision * 0.5)
            xp_ganado = int(xp_base * precision * 0.3)

        self.puntos_ganados = (self.puntos_ganados or 0) + puntos_ganados
        self.xp_ganado = (self.xp_ganado or 0) + xp_ganado
        self.ultima_practica = ahora

        self.verificar_completada(clase_obj)

        return {
            "puntos_ganados": puntos_ganados,
            "xp_ganado": xp_ganado,
            "precision": precision,
            "es_exitoso": es_exitoso,
            "nivel_dominio": self.obtener_nivel_dominio(),
        }

    def verificar_completada(self, clase_obj=None):
        intentos_minimos = getattr(clase_obj, "intentos_minimos", None) if clase_obj else None
        precision_minima = getattr(clase_obj, "precision_minima", None) if clase_obj else None

        intentos_minimos = 3 if intentos_minimos is None else intentos_minimos
        precision_minima = 0.7 if precision_minima is None else precision_minima

        if (self.intentos_realizados or 0) >= intentos_minimos and (self.mejor_precision or 0) >= precision_minima:
            if not self.completada:
                self.completada = True
                self.fecha_completada = _ahora()
            self.aprobada = True
            return True
        return False

    def obtener_nivel_dominio(self):
        precision = self.mejor_precision or 0
        if precision >= 0.9:
            return "Avanzado"
        if precision >= 0.7:
            return "Intermedio"
        return "Principiante"


class ProgresoLeccion(AuditoriaMixin, Base):
    __tablename__ = "progreso_lecciones"
    __table_args__ = (UniqueConstraint("usuario_id", "leccion_id", name="uq_progreso_leccion_usuario_leccion"),)

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False, index=True)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=False, index=True)

    desbloqueada = Column(Boolean, default=False)
    bloqueada = Column(Boolean, default=True)
    iniciada = Column(Boolean, default=False)
    completada = Column(Boolean, default=False)

    total_clases = Column(Integer, default=0)
    clases_completadas = Column(Integer, default=0)
    clases_aprobadas = Column(Integer, default=0)
    clases_vistas = Column(Integer, default=0)

    mejor_precision = Column(Float, default=0.0)
    precision_promedio = Column(Float, default=0.0)
    total_intentos = Column(Integer, default=0)
    intentos_exitosos = Column(Integer, default=0)

    total_puntos = Column(Integer, default=0)
    xp_total = Column(Integer, default=0)
    puntos_bonificacion_completa = Column(Integer, default=0)
    xp_bonificacion_completa = Column(Integer, default=0)
    puntos_maximos_posibles = Column(Integer, default=0)

    estrellas = Column(Integer, default=0)
    estrella_dorada = Column(Boolean, default=False)

    tiempo_total_minutos = Column(Integer, default=0)
    tiempo_promedio_clase_minutos = Column(Integer, default=0)

    examenes_disponibles = Column(Integer, default=0)
    examenes_completados = Column(Integer, default=0)
    examenes_aprobados = Column(Integer, default=0)
    mejor_calificacion_examen = Column(Float, default=0.0)

    racha_dias_consecutivos = Column(Integer, default=0)
    dias_activos = Column(Integer, default=0)

    porcentaje_completado = Column(Float, default=0.0)
    porcentaje_precision = Column(String(10), default="0%")
    tasa_exito_general = Column(Float, default=0.0)
    eficiencia_puntos = Column(Float, default=0.0)
    nivel_dominio_leccion = Column(String(50), default="Principiante")
    tiene_estrella_dorada = Column(Boolean, default=False)
    dias_desde_ultima_actividad = Column(Integer, default=0)

    fecha_desbloqueo = Column(DateTime(timezone=True), nullable=True)
    fecha_inicio = Column(DateTime(timezone=True), nullable=True)
    fecha_completada = Column(DateTime(timezone=True), nullable=True)
    ultima_practica = Column(DateTime(timezone=True), nullable=True)

    usuario = relationship("Usuario", back_populates="progresos_leccion", foreign_keys=[usuario_id])
    leccion = relationship("Leccion", back_populates="progresos_leccion")