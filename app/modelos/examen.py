from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base

class Examen(Base):
    __tablename__ = "examenes"
    
    id = Column(Integer, primary_key=True, index=True)
    titulo = Column(String(255), nullable=False)
    descripcion = Column(Text, nullable=True)
    
    tipo = Column(String(50), nullable=False)
    nivel = Column(Integer, nullable=True)
    
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=False)
    orden = Column(Integer, nullable=False, default=1)
    clases_requeridas = Column(Integer, nullable=False, default=0)
    requiere_todas_clases = Column(Boolean, default=False)
    
    tiempo_limite = Column(Integer, nullable=True)
    puntuacion_minima = Column(Float, default=70.0)
    
    activo = Column(Boolean, default=True)
    
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    leccion = relationship("Leccion", back_populates="examenes")
    preguntas = relationship("PreguntaExamen", back_populates="examen", cascade="all, delete-orphan")
    resultados = relationship("ResultadoExamen", back_populates="examen", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Examen(id={self.id}, titulo='{self.titulo}', tipo='{self.tipo}', nivel={self.nivel})>"
    
    @property
    def total_preguntas(self):
        return len(self.preguntas)
    
    @property
    def puntos_totales(self):
        return sum(pregunta.puntos for pregunta in self.preguntas)
    
    @property
    def duracion_estimada(self):
        if self.tiempo_limite:
            return self.tiempo_limite
        return self.total_preguntas * 2

class PreguntaExamen(Base):
    __tablename__ = "preguntas_examen"
    
    id = Column(Integer, primary_key=True, index=True)
    
    examen_id = Column(Integer, ForeignKey("examenes.id"), nullable=False)
    leccion_id = Column(Integer, ForeignKey("lecciones.id"), nullable=True)
    
    pregunta = Column(Text, nullable=False)
    tipo_pregunta = Column(String(50), nullable=False)
    
    sena_esperada = Column(String(255), nullable=True)
    imagen_sena = Column(String(500), nullable=True)
    
    opciones = Column(JSON, nullable=True)
    respuesta_correcta = Column(String(10), nullable=True)
    
    puntos = Column(Integer, default=10)
    orden = Column(Integer, nullable=False)
    
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    
    examen = relationship("Examen", back_populates="preguntas")
    leccion = relationship("Leccion", back_populates="preguntas_examen")
    
    def __repr__(self):
        return f"<PreguntaExamen(id={self.id}, tipo='{self.tipo_pregunta}', examen_id={self.examen_id})>"
    
    def es_respuesta_correcta(self, respuesta_usuario: str) -> bool:
        if self.tipo_pregunta == 'multiple' or self.tipo_pregunta == 'verdadero_falso':
            return respuesta_usuario.strip().lower() == self.respuesta_correcta.strip().lower()
        elif self.tipo_pregunta == 'reconocimiento':
            return respuesta_usuario.strip().lower() == self.sena_esperada.strip().lower()
        return False

class ResultadoExamen(Base):
    __tablename__ = "resultados_examenes"
    
    id = Column(Integer, primary_key=True, index=True)
    
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    examen_id = Column(Integer, ForeignKey("examenes.id"), nullable=False)
    
    puntuacion_obtenida = Column(Float, nullable=False)
    puntuacion_maxima = Column(Float, nullable=False)
    porcentaje = Column(Float, nullable=False)
    aprobado = Column(Boolean, nullable=False)
    
    tiempo_empleado = Column(Integer, nullable=True)
    respuestas = Column(JSON, nullable=True)
    
    fecha_inicio = Column(DateTime(timezone=True), nullable=False)
    fecha_finalizacion = Column(DateTime(timezone=True), nullable=False)
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    
    usuario = relationship("Usuario", back_populates="resultados_examenes")
    examen = relationship("Examen", back_populates="resultados")
    
    def __repr__(self):
        status = "APROBADO" if self.aprobado else "REPROBADO"
        return f"<ResultadoExamen(usuario_id={self.usuario_id}, examen_id={self.examen_id}, {status})>"