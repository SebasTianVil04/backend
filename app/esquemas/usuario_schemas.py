from pydantic import BaseModel, EmailStr, field_validator, validator
from typing import Optional, Tuple, List, Literal
from datetime import datetime, date
import re

# Constante para longitud máxima de contraseña
MAX_PASSWORD_LENGTH = 72  # Límite de bcrypt


# ============================================
# SCHEMAS DE REGISTRO Y LOGIN
# ============================================


class UsuarioRegistro(BaseModel):
    """Schema para registro de usuario con tipos"""
    tipo_usuario: Literal["peruano_mayor", "peruano_menor", "extranjero"]
    email: EmailStr
    password: str
    
    # Documentos (condicionales según tipo)
    dni: Optional[str] = None
    pasaporte: Optional[str] = None
    
    # Datos personales
    nombres: str
    apellido_paterno: str
    apellido_materno: str
    
    # Datos adicionales (teléfono ahora incluye código de país)
    telefono: Optional[str] = None
    direccion: Optional[str] = None
    fecha_nacimiento: Optional[date] = None
    
    @field_validator('password')
    @classmethod
    def validar_password(cls, v):
        # ✅ Validar longitud máxima primero
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
    
    @field_validator('dni')
    @classmethod
    def validar_dni(cls, v, info):
        if info.data.get('tipo_usuario') == 'peruano_mayor':
            if not v:
                raise ValueError('DNI es requerido para peruanos mayores de edad')
            if not v.isdigit() or len(v) != 8:
                raise ValueError('DNI debe tener 8 dígitos')
        return v
    
    @field_validator('pasaporte')
    @classmethod
    def validar_pasaporte(cls, v, info):
        if info.data.get('tipo_usuario') == 'extranjero':
            if not v:
                raise ValueError('Pasaporte es requerido para extranjeros')
            if len(v) < 6 or len(v) > 20:
                raise ValueError('Pasaporte debe tener entre 6 y 20 caracteres')
            # Validar formato alfanumérico
            if not v.replace('-', '').replace(' ', '').isalnum():
                raise ValueError('Pasaporte debe contener solo letras y números')
        return v


class UsuarioLogin(BaseModel):
    """Schema para login de usuario"""
    email: EmailStr
    password: str


# ============================================
# SCHEMAS DE RESPUESTA DE USUARIO
# ============================================


class UsuarioRespuesta(BaseModel):
    """Schema completo para respuesta de usuario"""
    id: int
    tipo_usuario: Literal['peruano_mayor', 'peruano_menor', 'extranjero']
    email: str
    dni: Optional[str] = None
    pasaporte: Optional[str] = None
    nombres: str
    apellido_paterno: str
    apellido_materno: str
    apellidos: Optional[str] = None
    telefono: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    direccion: Optional[str] = None
    rol: str
    activo: bool
    es_admin: bool
    verificado: bool
    fecha_registro: Optional[str] = None
    fecha_creacion: Optional[datetime] = None
    nombre_completo: Optional[str] = None
    
    class Config:
        from_attributes = True


# ============================================
# SCHEMAS DE ACTUALIZACIÓN
# ============================================


class UsuarioActualizacion(BaseModel):
    """Schema para actualización de datos de usuario"""
    telefono: Optional[str] = None
    direccion: Optional[str] = None
    fecha_nacimiento: Optional[date] = None
    
    @field_validator('telefono')
    @classmethod
    def validar_telefono(cls, v):
        """Validar teléfono con código de país"""
        if v is None or v == '':
            return v
        
        # Debe incluir código de país
        if not v.startswith('+'):
            raise ValueError('El teléfono debe incluir código de país (ej: +51987654321)')
        
        # Extraer solo dígitos
        solo_digitos = ''.join(filter(str.isdigit, v))
        
        # Validar longitud total
        if len(solo_digitos) < 9 or len(solo_digitos) > 15:
            raise ValueError('El teléfono debe tener entre 9 y 15 dígitos')
        
        return v
    
    @field_validator('fecha_nacimiento', mode='before')
    @classmethod
    def validar_fecha_actualizacion(cls, v):
        if v is None or v == '':
            return None
        
        if isinstance(v, str):
            try:
                if 'T' in v:
                    fecha = datetime.fromisoformat(v.replace('Z', '+00:00')).date()
                else:
                    fecha = datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError('Formato de fecha inválido (use YYYY-MM-DD)')
        elif isinstance(v, datetime):
            fecha = v.date()
        elif isinstance(v, date):
            fecha = v
        else:
            return None
        
        if fecha > date.today():
            raise ValueError('La fecha de nacimiento no puede ser futura')
        
        return fecha
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "telefono": "+51987654321",
                "direccion": "Av. Example 123, Lima",
                "fecha_nacimiento": "2001-04-19"
            }
        }


# ============================================
# SCHEMAS DE CONTRASEÑA
# ============================================


class CambiarPassword(BaseModel):
    """Schema para cambiar contraseña (desde perfil autenticado)"""
    password_actual: str
    password_nueva: str
    
    @validator('password_nueva')
    def validar_password_nueva(cls, v):
        # ✅ Validar longitud máxima
        if len(v) > MAX_PASSWORD_LENGTH:
            raise ValueError(f'La contraseña no puede exceder {MAX_PASSWORD_LENGTH} caracteres')
        
        if len(v) < 8:
            raise ValueError('La contraseña debe tener al menos 8 caracteres')
        if not any(c.isupper() for c in v):
            raise ValueError('La contraseña debe contener al menos una mayúscula')
        if not any(c.islower() for c in v):
            raise ValueError('La contraseña debe contener al menos una minúscula')
        if not any(c.isdigit() for c in v):
            raise ValueError('La contraseña debe contener al menos un número')
        return v


class CambiarPasswordRequest(BaseModel):
    """Schema alternativo para cambiar contraseña"""
    password_actual: str
    password_nueva: str
    
    @field_validator('password_nueva')
    @classmethod
    def validar_password(cls, v):
        # ✅ Validar longitud máxima
        if len(v) > MAX_PASSWORD_LENGTH:
            raise ValueError(f'La contraseña no puede exceder {MAX_PASSWORD_LENGTH} caracteres')
        
        if len(v) < 8:
            raise ValueError('La contraseña debe tener al menos 8 caracteres')
        if not any(c.isupper() for c in v):
            raise ValueError('La contraseña debe contener al menos una mayúscula')
        if not any(c.islower() for c in v):
            raise ValueError('La contraseña debe contener al menos una minúscula')
        if not any(c.isdigit() for c in v):
            raise ValueError('La contraseña debe contener al menos un número')
        return v


# ============================================
# SCHEMAS DE RECUPERACIÓN DE CONTRASEÑA
# ============================================


class SolicitudRecuperacion(BaseModel):
    """Schema para solicitar recuperación de contraseña"""
    email: EmailStr


class ConfirmarRecuperacion(BaseModel):
    """Schema para confirmar recuperación"""
    token: str
    password_nueva: str
    
    @field_validator('password_nueva')
    @classmethod
    def validar_password_nueva(cls, v):
        errores = []
        
        # ✅ Validar longitud máxima primero
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
    """Schema para respuesta de recuperación"""
    mensaje: str
    email: Optional[str] = None


# ============================================
# SCHEMAS DE VERIFICACIÓN
# ============================================


class VerificarEmailRequest(BaseModel):
    """Schema para verificar si un email existe"""
    email: EmailStr


class VerificarEmailResponse(BaseModel):
    """Schema para respuesta de verificación de email"""
    existe: bool
    activo: Optional[bool] = None
    mensaje: str


# ============================================
# SCHEMAS DE TOKEN
# ============================================


class TokenRespuesta(BaseModel):
    """Schema para respuesta de token"""
    access_token: str
    token_type: str = "bearer"
    usuario: UsuarioRespuesta


# ============================================
# SCHEMAS DE API PERU
# ============================================


class DatosApiperu(BaseModel):
    """Schema para datos de API Peru"""
    nombres: str
    apellido_paterno: str
    apellido_materno: str
    fecha_nacimiento: Optional[date] = None 
    direccion: Optional[str] = None


# ============================================
# FUNCIONES AUXILIARES
# ============================================


def validar_fortaleza_password(password: str) -> Tuple[bool, List[str]]:
    """
    Validar fortaleza de contraseña y retornar errores
    
    Args:
        password: Contraseña a validar
    
    Returns:
        Tuple[bool, List[str]]: (es_valida, lista_de_errores)
    """
    errores = []
    
    # ✅ Validar longitud máxima
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


# ============================================
# SCHEMAS DE ESTADÍSTICAS
# ============================================


class EstadisticasUsuario(BaseModel):
    """Schema para estadísticas detalladas de usuario"""
    usuario_id: int
    nombre_completo: str
    email: str
    
    # Estadísticas de progreso
    total_lecciones_completadas: int = 0
    total_lecciones_disponibles: int = 0
    porcentaje_progreso: float = 0.0
    
    # Estadísticas de rendimiento
    puntuacion_promedio: float = 0.0
    puntuacion_maxima: float = 0.0
    puntuacion_minima: float = 0.0
    
    # Estadísticas de tiempo
    tiempo_total_estudio: int = 0
    tiempo_promedio_sesion: float = 0.0
    sesiones_totales: int = 0
    
    # Estadísticas de racha y nivel
    racha_actual: int = 0
    racha_maxima: int = 0
    nivel_actual: int = 1
    experiencia_total: int = 0
    
    # Estadísticas de examenes
    examenes_completados: int = 0
    examenes_aprobados: int = 0
    tasa_aprobacion: float = 0.0
    
    # Fechas importantes
    fecha_registro: datetime
    ultima_actividad: Optional[datetime] = None
    ultima_leccion_completada: Optional[datetime] = None
    
    # Estadísticas por categoría
    categorias_dominadas: int = 0
    categoria_favorita: Optional[str] = None
    
    # Metas y logros
    dias_consecutivos_activo: int = 0
    logros_desbloqueados: int = 0
    
    # Ranking y posición
    posicion_ranking: Optional[int] = None
    percentil_rendimiento: Optional[float] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }