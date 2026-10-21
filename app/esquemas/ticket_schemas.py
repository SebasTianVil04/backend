from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..servicios.antispam_contacto import (
    validar_email,
    validar_nombre,
    validar_texto_libre,
)


class TipoTicket(str, Enum):
    solicitud = "solicitud"
    incidencia = "incidencia"


class ServicioTicket(str, Enum):
    acceso_cuenta = "acceso_cuenta"
    aprendizaje = "aprendizaje"
    practicas_camara = "practicas_camara"
    examenes_progreso = "examenes_progreso"
    contenido_educativo = "contenido_educativo"
    reconocimiento_ia = "reconocimiento_ia"
    plataforma_datos = "plataforma_datos"


class PrioridadTicket(str, Enum):
    baja = "baja"
    media = "media"
    alta = "alta"
    critica = "critica"


class EstadoTicket(str, Enum):
    cuarentena = "cuarentena"
    nueva = "nueva"
    en_revision = "en_revision"
    aprobada = "aprobada"
    rechazada = "rechazada"
    asignada = "asignada"
    en_atencion = "en_atencion"
    esperando_usuario = "esperando_usuario"
    resuelta = "resuelta"
    cerrada = "cerrada"
    descartada = "descartada"


class ContactoPublicoIn(BaseModel):
    nombre: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=6, max_length=255)
    tipo: TipoTicket
    servicio: ServicioTicket
    asunto: str = Field(min_length=5, max_length=120)
    descripcion: str = Field(min_length=20, max_length=2000)
    token_formulario: str = Field(min_length=20, max_length=300)
    sitio_web: Optional[str] = Field(default=None, max_length=200)
    turnstile_token: Optional[str] = Field(default=None, max_length=2048)

    @field_validator("nombre")
    @classmethod
    def _nombre(cls, v: str) -> str:
        return validar_nombre(v)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return validar_email(v)

    @field_validator("asunto")
    @classmethod
    def _asunto(cls, v: str) -> str:
        return validar_texto_libre(v, minimo=5, maximo=120, multilinea=False)

    @field_validator("descripcion")
    @classmethod
    def _descripcion(cls, v: str) -> str:
        return validar_texto_libre(v, minimo=20, maximo=2000, multilinea=True)


class TicketUsuarioIn(BaseModel):
    tipo: TipoTicket
    servicio: ServicioTicket
    asunto: str = Field(min_length=5, max_length=120)
    descripcion: str = Field(min_length=20, max_length=2000)
    prioridad: PrioridadTicket = PrioridadTicket.media

    @field_validator("asunto")
    @classmethod
    def _asunto(cls, v: str) -> str:
        return validar_texto_libre(v, minimo=5, maximo=120, multilinea=False)

    @field_validator("descripcion")
    @classmethod
    def _descripcion(cls, v: str) -> str:
        return validar_texto_libre(v, minimo=20, maximo=2000, multilinea=True)

    @field_validator("prioridad")
    @classmethod
    def _prioridad(cls, v: PrioridadTicket) -> PrioridadTicket:
        if v == PrioridadTicket.critica:
            return PrioridadTicket.alta
        return v


class ComentarioIn(BaseModel):
    mensaje: str = Field(min_length=2, max_length=2000)
    es_interno: bool = False

    @field_validator("mensaje")
    @classmethod
    def _mensaje(cls, v: str) -> str:
        return validar_texto_libre(v, minimo=2, maximo=2000, multilinea=True)


class TicketAdminUpdate(BaseModel):
    estado: Optional[EstadoTicket] = None
    prioridad: Optional[PrioridadTicket] = None
    asignado_a_id: Optional[int] = Field(default=None, ge=1)


class ComentarioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    autor_nombre: str
    es_soporte: bool
    es_interno: bool
    mensaje: str
    fecha_creacion: datetime


class ComentarioPublicoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    autor_nombre: str
    es_soporte: bool
    mensaje: str
    fecha_creacion: datetime


class TicketResumenOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    codigo: Optional[str]
    nombre: str
    email: str
    tipo: str
    servicio: str
    asunto: str
    prioridad: str
    estado: str
    origen: str
    puntaje_spam: int
    asignado_a_id: Optional[int]
    fecha_creacion: datetime
    fecha_actualizacion: datetime


class TicketDetalleOut(TicketResumenOut):
    usuario_id: Optional[int]
    descripcion: str
    fecha_cierre: Optional[datetime]
    comentarios: List[ComentarioOut] = []


class TicketUsuarioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    codigo: Optional[str]
    tipo: str
    servicio: str
    asunto: str
    descripcion: str
    prioridad: str
    estado: str
    fecha_creacion: datetime
    fecha_actualizacion: datetime
    fecha_cierre: Optional[datetime]
    comentarios: List[ComentarioPublicoOut] = []


class TicketCreadoOut(BaseModel):
    codigo: str
    estado: str
    mensaje: str


class SeguimientoOut(BaseModel):
    codigo: str
    tipo: str
    servicio: str
    asunto: str
    estado: str
    fecha_creacion: datetime
    fecha_actualizacion: datetime
    comentarios: List[ComentarioPublicoOut] = []


class TokenFormularioOut(BaseModel):
    token: str
    espera_minima_segundos: int
    turnstile_requerido: bool
