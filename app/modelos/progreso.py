from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime
from ..utilidades.base_datos import Base

class ProgresoClase(Base):
    __tablename__ = "progreso_clases"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    clase_id = Column(Integer, ForeignKey("clases.id"), nullable=False)
    
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
    
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    usuario = relationship("Usuario", back_populates="progresos_clase")
    clase = relationship("Clase", back_populates="progresos_clase")
    
    # Propiedades calculadas para compatibilidad con Pydantic
    @property
    def porcentaje_precision(self):
        return f"{int(self.mejor_precision * 100)}%"
    
    @property
    def tasa_exito(self):
        if self.intentos_realizados == 0:
            return 0.0
        return self.intentos_exitosos / self.intentos_realizados
    
    @property
    def porcentaje_tasa_exito(self):
        return f"{int(self.tasa_exito * 100)}%"
    
    @property
    def eficiencia_puntos(self):
        if self.tiempo_total_practica == 0:
            return 0.0
        return self.puntos_ganados / max(self.tiempo_total_practica / 60, 1)  # puntos por minuto
    
    @property
    def nivel_dominio(self):
        if self.mejor_precision >= 0.9:
            return "Perfecto"
        elif self.mejor_precision >= 0.8:
            return "Excelente"
        elif self.mejor_precision >= 0.7:
            return "Bueno"
        elif self.mejor_precision >= 0.6:
            return "Aceptable"
        else:
            return "Necesita Mejorar"
    
    @property
    def dias_desde_ultima_practica(self):
        if not self.ultima_practica:
            return 999  # Un valor alto para indicar que nunca ha practicado
        delta = datetime.now().replace(tzinfo=None) - self.ultima_practica.replace(tzinfo=None)
        return delta.days
    
    @property
    def esta_en_racha(self):
        return self.dias_desde_ultima_practica <= 1
    
    @property
    def tiempo_promedio_intento(self):
        if self.intentos_realizados == 0:
            return 0
        return self.tiempo_total_practica // self.intentos_realizados
    
    @property
    def racha_dias_consecutivos(self):
        # Esta lógica debería venir de un servicio de rachas
        # Por ahora devolvemos un valor basado en la actividad reciente
        if self.dias_desde_ultima_practica <= 1:
            return min(getattr(self, '_racha_calculada', 1), 30)
        return 0
    
    @property
    def mejor_racha(self):
        # Esta lógica debería venir de un historial de rachas
        return max(getattr(self, '_mejor_racha_calculada', 0), self.racha_dias_consecutivos)
    
    def registrar_intento(self, precision: float, duracion_segundos: int, clase_obj=None, es_exitoso: bool = None):
        self.intentos_realizados += 1
        self.ultima_precision = precision
        self.tiempo_total_practica += duracion_segundos
        
        if precision > self.mejor_precision:
            self.mejor_precision = precision
            self.fecha_mejor_precision = datetime.now()
        
        if self.precision_promedio == 0:
            self.precision_promedio = precision
        else:
            self.precision_promedio = (self.precision_promedio + precision) / 2
        
        if es_exitoso is None:
            precision_minima = 0.7
            if clase_obj:
                precision_minima = getattr(clase_obj, 'precision_minima', 0.7)
            es_exitoso = precision >= precision_minima
        
        puntos_base = 10
        xp_base = 5
        
        if es_exitoso:
            self.intentos_exitosos += 1
            puntos_ganados = int(puntos_base * precision)
            xp_ganado = int(xp_base * precision)
        else:
            puntos_ganados = int(puntos_base * precision * 0.5)
            xp_ganado = int(xp_base * precision * 0.3)
        
        self.puntos_ganados += puntos_ganados
        self.xp_ganado += xp_ganado
        self.ultima_practica = datetime.now()
        
        self.verificar_completada(clase_obj)
        
        return {
            'puntos_ganados': puntos_ganados,
            'xp_ganado': xp_ganado,
            'precision': precision,
            'es_exitoso': es_exitoso,
            'nivel_dominio': self.obtener_nivel_dominio()
        }
    
    def verificar_completada(self, clase_obj=None):
        intentos_minimos = 3
        precision_minima = 0.7
        
        if clase_obj:
            intentos_minimos = getattr(clase_obj, 'intentos_minimos', 3)
            precision_minima = getattr(clase_obj, 'precision_minima', 0.7)
        
        if (self.intentos_realizados >= intentos_minimos and 
            self.mejor_precision >= precision_minima):
            self.completada = True
            self.fecha_completada = datetime.now()
            return True
        return False
    
    def obtener_nivel_dominio(self):
        if self.mejor_precision >= 0.9:
            return "Avanzado"
        elif self.mejor_precision >= 0.7:
            return "Intermedio"
        else:
            return "Principiante"
        
class ProgresoLeccion(Base):
    __tablename__ = "progreso_lecciones"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=False)
    
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
    
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    usuario = relationship("Usuario", back_populates="progresos_leccion")
    leccion = relationship("Leccion", back_populates="progresos_leccion")