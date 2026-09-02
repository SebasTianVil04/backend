from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base

class Leccion(Base):
    __tablename__ = "lecciones"
    
    id = Column(Integer, primary_key=True, index=True)
    categoria_id = Column(Integer, ForeignKey("categorias.id"), nullable=False)
    titulo = Column(String(255), nullable=False)
    descripcion = Column(Text, nullable=True)
    sena = Column(String(100), nullable=False)
    orden = Column(Integer, nullable=False)
    nivel_dificultad = Column(Integer, default=1)
    activa = Column(Boolean, default=True)
    bloqueada = Column(Boolean, default=False)
    leccion_previa_id = Column(Integer, ForeignKey("lecciones.id"), nullable=True)
    puntos_base = Column(Integer, default=10, nullable=False)
    puntos_perfecto = Column(Integer, default=20, nullable=False)
    requiere_examen_nivel = Column(Boolean, default=False)
    numero_examen = Column(Integer, nullable=True)
    imagen_miniatura = Column(String(500), nullable=True)
    color_tema = Column(String(7), default="#3B82F6")
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(DateTime(timezone=True), onupdate=func.now())
    
    categoria_rel = relationship("Categoria", back_populates="lecciones")
    clases = relationship("Clase", back_populates="leccion", cascade="all, delete-orphan", order_by="Clase.orden")
    examenes = relationship("Examen", back_populates="leccion", cascade="all, delete-orphan")
    preguntas_examen = relationship("PreguntaExamen", back_populates="leccion", cascade="all, delete-orphan")
    practicas = relationship("Practica", back_populates="leccion", cascade="all, delete-orphan")
    # CORRECCIÓN: Cambiar nombre para que coincida
    progresos_leccion = relationship("ProgresoLeccion", back_populates="leccion", cascade="all, delete-orphan")
    sesiones_estudio = relationship("SesionEstudio", back_populates="leccion", cascade="all, delete-orphan")
    leccion_previa = relationship("Leccion", remote_side=[id], foreign_keys=[leccion_previa_id], backref="lecciones_siguientes")
    
    def __repr__(self):
        return f"<Leccion(id={self.id}, titulo='{self.titulo}', categoria_id={self.categoria_id}, orden={self.orden})>"
    
    @property
    def total_clases(self):
        return len(self.clases)
    
    @property
    def categoria_nombre(self):
        return self.categoria_rel.nombre if self.categoria_rel else None
    
    @property
    def nivel_dificultad_texto(self):
        niveles = {1: "Principiante", 2: "Intermedio", 3: "Avanzado"}
        return niveles.get(self.nivel_dificultad, "Desconocido")
    
    @property
    def total_preguntas_examen(self):
        return len(self.preguntas_examen)
    
    @property
    def total_examenes(self):
        return len(self.examenes)
    
    def obtener_examen_nivel(self, db):
        from .examen import Examen
        if self.numero_examen:
            return db.query(Examen).filter(
                Examen.tipo == 'nivel',
                Examen.nivel == self.numero_examen,
                Examen.activo == True
            ).first()
        return None
    
    def validar_nivel_dificultad(self):
        if self.nivel_dificultad not in [1, 2, 3]:
            raise ValueError("El nivel de dificultad debe ser 1 (Principiante), 2 (Intermedio) o 3 (Avanzado)")