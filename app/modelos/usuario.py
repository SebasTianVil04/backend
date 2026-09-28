from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin


class Usuario(AuditoriaMixin, Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)

    rol_id = Column(Integer, ForeignKey("roles.id"), nullable=False, index=True)
    rol = relationship("Rol", back_populates="usuarios", lazy="joined")

    nombre_usuario = Column(String(30), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(512), nullable=False)

    activo = Column(Boolean, default=True)
    verificado = Column(Boolean, default=False)

    progresos_clase = relationship("ProgresoClase", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="ProgresoClase.usuario_id")
    progresos_leccion = relationship("ProgresoLeccion", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="ProgresoLeccion.usuario_id")
    resultados_examenes = relationship("ResultadoExamen", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="ResultadoExamen.usuario_id")
    entrenamientos = relationship("Entrenamiento", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="Entrenamiento.usuario_id")
    tokens_recuperacion = relationship("TokenRecuperacion", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="TokenRecuperacion.usuario_email")
    practicas = relationship("Practica", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="Practica.usuario_id")
    sesiones_estudio = relationship("SesionEstudio", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="SesionEstudio.usuario_id")
    calibraciones = relationship("CalibracionUsuario", back_populates="usuario", cascade="all, delete-orphan", foreign_keys="CalibracionUsuario.usuario_id")

    @property
    def es_admin(self) -> bool:
        return self.rol is not None and self.rol.codigo == "admin"

    @property
    def permisos(self):
        if not self.rol:
            return []
        return [p.codigo for p in self.rol.permisos]

    def tiene_rol(self, *codigos: str) -> bool:
        return self.rol is not None and self.rol.codigo in codigos

    def tiene_permiso(self, codigo_permiso: str) -> bool:
        return self.rol is not None and self.rol.tiene_permiso(codigo_permiso)

    def __repr__(self):
        return f"<Usuario(email={self.email}, nombre_usuario={self.nombre_usuario}, rol_id={self.rol_id})>"