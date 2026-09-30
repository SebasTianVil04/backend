from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, Tuple, List
from datetime import datetime
import re

MAX_PASSWORD_LENGTH = 72
MIN_USERNAME_LENGTH = 3
MAX_USERNAME_LENGTH = 30


def _validar_password(v: str) -> str:
    if len(v) > MAX_PASSWORD_LENGTH:
        raise ValueError(f'La contraseña no puede exceder {MAX_PASSWORD_LENGTH} caracteres')
    if len(v) < 8:
        raise ValueError('La contraseña debe tener al menos 8 caracteres')
    if not re.search(r'[A-Z]', v):
        raise ValueError('Debe contener al menos una mayúscula')
    if not re.search(r'[a-z]', v):
        raise ValueError('Debe contener al menos una minúscula')
    if not re.search(r'\d', v):
        raise ValueError('Debe contener al menos un número')
    return v


def _validar_nombre_usuario(v: str) -> str:
    v = v.strip()
    if len(v) < MIN_USERNAME_LENGTH or len(v) > MAX_USERNAME_LENGTH:
        raise ValueError(
            f'El nombre de usuario debe tener entre {MIN_USERNAME_LENGTH} y {MAX_USERNAME_LENGTH} caracteres'
        )
    if not re.match(r'^[a-zA-Z0-9_]+$', v):
        raise ValueError('El nombre de usuario solo puede contener letras, números y guion bajo (_)')
    return v


class UsuarioRegistro(BaseModel):
    nombre_usuario: str
    email: EmailStr
    password: str

    @field_validator('nombre_usuario')
    @classmethod
    def validar_nombre_usuario(cls, v):
        return _validar_nombre_usuario(v)

    @field_validator('password')
    @classmethod
    def validar_password(cls, v):
        return _validar_password(v)


class UsuarioLogin(BaseModel):
    email: EmailStr
    password: str


class RolResumen(BaseModel):
    id: int
    codigo: str
    nombre: str

    class Config:
        from_attributes = True


class UsuarioRespuesta(BaseModel):
    id: int
    nombre_usuario: str
    email: str
    rol: Optional[RolResumen] = None
    permisos: List[str] = []
    activo: bool
    verificado: bool
    fecha_creacion: Optional[datetime] = None

    class Config:
        from_attributes = True

class SesionRespuesta(BaseModel):
    """Respuesta de login/refresh: ya no expone el access_token (va en cookie HttpOnly)."""
    usuario: UsuarioRespuesta

    class Config:
        from_attributes = True

class UsuarioActualizacion(BaseModel):
    nombre_usuario: Optional[str] = None

    @field_validator('nombre_usuario')
    @classmethod
    def validar_nombre_usuario(cls, v):
        if v is None:
            return v
        return _validar_nombre_usuario(v)

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "nombre_usuario": "sebastian_v"
            }
        }


class CambiarPasswordRequest(BaseModel):
    password_actual: str
    password_nueva: str

    @field_validator('password_nueva')
    @classmethod
    def validar_password_nueva(cls, v):
        return _validar_password(v)


class SolicitudRecuperacion(BaseModel):
    email: EmailStr


class ConfirmarRecuperacion(BaseModel):
    token: str
    password_nueva: str

    @field_validator('password_nueva')
    @classmethod
    def validar_password_nueva(cls, v):
        errores = []

        if len(v) > MAX_PASSWORD_LENGTH:
            raise ValueError(f'La contraseña no puede exceder {MAX_PASSWORD_LENGTH} caracteres')

        if len(v) < 8:
            errores.append('Debe tener al menos 8 caracteres')
        if not re.search(r'[A-Z]', v):
            errores.append('Debe contener al menos una mayúscula')
        if not re.search(r'[a-z]', v):
            errores.append('Debe contener al menos una minúscula')
        if not re.search(r'\d', v):
            errores.append('Debe contener al menos un número')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            errores.append('Debe contener al menos un carácter especial')

        if errores:
            raise ValueError(f'Contraseña insegura: {", ".join(errores)}')

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "token": "abc123def456ghi789",
                "password_nueva": "MiNuevaPassword123!"
            }
        }


class RespuestaRecuperacion(BaseModel):
    mensaje: str
    email: Optional[str] = None


class VerificarEmailRequest(BaseModel):
    email: EmailStr


class VerificarEmailResponse(BaseModel):
    existe: bool
    activo: Optional[bool] = None
    mensaje: str



class DatosApiperu(BaseModel):
    """Obsoleto: ya no se usa (se eliminó la integración con apiperu.dev).
    Se mantiene solo para no romper imports de módulos que aún la referencian;
    puede eliminarse por completo una vez limpiados esos módulos."""
    nombres: Optional[str] = None
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None


def validar_fortaleza_password(password: str) -> Tuple[bool, List[str]]:
    errores = []

    if len(password) > MAX_PASSWORD_LENGTH:
        errores.append(f'No puede exceder {MAX_PASSWORD_LENGTH} caracteres')

    if len(password) < 8:
        errores.append('Debe tener al menos 8 caracteres')
    if not re.search(r'[A-Z]', password):
        errores.append('Debe contener al menos una mayúscula')
    if not re.search(r'[a-z]', password):
        errores.append('Debe contener al menos una minúscula')
    if not re.search(r'\d', password):
        errores.append('Debe contener al menos un número')
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errores.append('Debe contener al menos un carácter especial')

    return len(errores) == 0, errores


class EstadisticasUsuario(BaseModel):
    usuario_id: int
    nombre_usuario: str
    email: str

    total_lecciones_completadas: int = 0
    total_lecciones_disponibles: int = 0
    porcentaje_progreso: float = 0.0

    puntuacion_promedio: float = 0.0
    puntuacion_maxima: float = 0.0
    puntuacion_minima: float = 0.0

    tiempo_total_estudio: int = 0
    tiempo_promedio_sesion: float = 0.0
    sesiones_totales: int = 0

    racha_actual: int = 0
    racha_maxima: int = 0
    nivel_actual: int = 1
    experiencia_total: int = 0

    examenes_completados: int = 0
    examenes_aprobados: int = 0
    tasa_aprobacion: float = 0.0

    fecha_registro: Optional[datetime] = None
    ultima_actividad: Optional[datetime] = None
    ultima_leccion_completada: Optional[datetime] = None

    categorias_dominadas: int = 0
    categoria_favorita: Optional[str] = None

    dias_consecutivos_activo: int = 0
    logros_desbloqueados: int = 0

    posicion_ranking: Optional[int] = None
    percentil_rendimiento: Optional[float] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }