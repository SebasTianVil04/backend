from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Date, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base
from .mixins import AuditoriaMixin
import enum


class TipoUsuario(str, enum.Enum):
    PERUANO_MAYOR = "peruano_mayor"
    PERUANO_MENOR = "peruano_menor"
    EXTRANJERO = "extranjero"


class Usuario(AuditoriaMixin, Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)

    tipo_usuario = Column(String(20), nullable=False, default="peruano_mayor")

    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(512), nullable=False)

    dni = Column(String(8), unique=True, index=True, nullable=True)
    pasaporte = Column(String(20), unique=True, index=True, nullable=True)

    nombres = Column(String(255), nullable=False)
    apellido_paterno = Column(String(255), nullable=False)
    apellido_materno = Column(String(255), nullable=False)

    telefono = Column(String(20), nullable=True, index=True)
    direccion = Column(Text, nullable=True)
    fecha_nacimiento = Column(Date, nullable=True)

    activo = Column(Boolean, default=True)
    es_admin = Column(Boolean, default=False)
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
    def nombre_completo(self):
        return f"{self.nombres} {self.apellido_paterno} {self.apellido_materno}"

    @property
    def rol(self):
        return "admin" if self.es_admin else "usuario"

    @property
    def documento_identidad(self):
        if self.tipo_usuario == "extranjero":
            return self.pasaporte
        return self.dni

    @property
    def edad(self):
        if not self.fecha_nacimiento:
            return None

        from datetime import date
        hoy = date.today()
        edad = hoy.year - self.fecha_nacimiento.year

        if hoy.month < self.fecha_nacimiento.month or \
           (hoy.month == self.fecha_nacimiento.month and hoy.day < self.fecha_nacimiento.day):
            edad -= 1

        return edad

    @property
    def racha_actual(self):
        if not self.ultima_practica:
            return 0

        from datetime import datetime, timedelta
        hoy = datetime.now().date()
        ultima_fecha = self.ultima_practica.date()

        if ultima_fecha == hoy:
            return getattr(self, '_racha_cache', 1)
        elif ultima_fecha == hoy - timedelta(days=1):
            return getattr(self, '_racha_cache', 1) + 1
        else:
            return 0

    @property
    def es_mayor_edad(self):
        edad = self.edad
        return edad >= 18 if edad is not None else None

    @property
    def telefono_formateado(self):
        if not self.telefono:
            return None

        if self.telefono.startswith('+'):
            codigo_pais = self.telefono[:3]
            numero = self.telefono[3:]

            if len(numero) == 9:
                return f"{codigo_pais} {numero[:3]} {numero[3:6]} {numero[6:]}"
            else:
                return f"{codigo_pais} {numero}"

        return self.telefono

    def validar_edad_con_tipo(self):
        edad = self.edad
        if edad is None:
            return False

        if self.tipo_usuario == "peruano_menor" and edad >= 18:
            return False

        if self.tipo_usuario == "peruano_mayor" and edad < 18:
            return False

        if self.tipo_usuario == "extranjero" and edad < 18:
            return False

        return True

    def __repr__(self):
        return f"<Usuario(email={self.email}, tipo={self.tipo_usuario}, edad={self.edad})>"