import re
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

def validar_email(email: str) -> bool:
    """Validar formato de email"""
    patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(patron, email))

def validar_dni(dni: str) -> bool:
    """Validar formato de DNI peruano"""
    return dni.isdigit() and len(dni) == 8

def validar_pasaporte(pasaporte: str) -> bool:
    """
    Validar formato de pasaporte internacional
    - Entre 6 y 20 caracteres
    - Solo letras, números, guiones y espacios
    """
    if not pasaporte or len(pasaporte) < 6 or len(pasaporte) > 20:
        return False
    
    # Permitir letras, números, guiones y espacios
    patron = r'^[A-Z0-9\-\s]+$'
    pasaporte_upper = pasaporte.upper().strip()
    return bool(re.match(patron, pasaporte_upper))

def validar_telefono(telefono: str) -> bool:
    """
    Validar formato de teléfono internacional con código de país
    Formato esperado: +[código][número]
    Ejemplo: +51987654321
    """
    if not telefono:
        return True  # Opcional
    
    # Debe empezar con +
    if not telefono.startswith('+'):
        return False
    
    # Remover el + inicial y verificar que el resto sean solo dígitos
    numero_sin_mas = telefono[1:]
    solo_digitos = ''.join(filter(str.isdigit, numero_sin_mas))
    
    # Verificar longitud total (código país + número): entre 9 y 15 dígitos
    return 9 <= len(solo_digitos) <= 15

def validar_codigo_pais_telefono(telefono: str) -> Tuple[bool, Optional[str]]:
    """
    Validar y extraer código de país del teléfono
    Returns: (es_valido, codigo_pais)
    """
    if not telefono or not telefono.startswith('+'):
        return False, None
    
    # Códigos de país conocidos (expandible)
    codigos_validos = {
        '+1': 'USA/Canadá',
        '+7': 'Rusia',
        '+33': 'Francia',
        '+34': 'España',
        '+44': 'Reino Unido',
        '+49': 'Alemania',
        '+51': 'Perú',
        '+52': 'México',
        '+53': 'Cuba',
        '+54': 'Argentina',
        '+55': 'Brasil',
        '+56': 'Chile',
        '+57': 'Colombia',
        '+58': 'Venezuela',
        '+593': 'Ecuador',
        '+507': 'Panamá',
    }
    
    for codigo, pais in codigos_validos.items():
        if telefono.startswith(codigo):
            return True, codigo
    
    # Si no está en la lista pero tiene formato válido, aceptar
    # (código de 2-4 dígitos)
    match = re.match(r'^\+(\d{1,4})', telefono)
    if match:
        return True, f'+{match.group(1)}'
    
    return False, None

def validar_longitud_telefono_por_pais(telefono: str) -> Tuple[bool, Optional[str]]:
    """
    Validar longitud del número según el código de país
    Returns: (es_valido, mensaje_error)
    """
    es_valido, codigo_pais = validar_codigo_pais_telefono(telefono)
    
    if not es_valido:
        return False, "Código de país no válido"
    
    # Extraer número sin código de país
    numero = telefono[len(codigo_pais):]
    solo_digitos = ''.join(filter(str.isdigit, numero))
    
    # Longitudes esperadas por país
    longitudes_por_pais = {
        '+51': (9, 9, 'Perú: exactamente 9 dígitos'),
        '+1': (10, 10, 'USA/Canadá: exactamente 10 dígitos'),
        '+34': (9, 9, 'España: exactamente 9 dígitos'),
        '+52': (10, 10, 'México: exactamente 10 dígitos'),
        '+54': (10, 11, 'Argentina: 10-11 dígitos'),
        '+55': (10, 11, 'Brasil: 10-11 dígitos'),
        '+56': (9, 9, 'Chile: exactamente 9 dígitos'),
        '+57': (10, 10, 'Colombia: exactamente 10 dígitos'),
        '+593': (9, 9, 'Ecuador: exactamente 9 dígitos'),
    }
    
    if codigo_pais in longitudes_por_pais:
        min_long, max_long, mensaje = longitudes_por_pais[codigo_pais]
        if len(solo_digitos) < min_long or len(solo_digitos) > max_long:
            return False, mensaje
    else:
        # Validación genérica para otros países
        if len(solo_digitos) < 7 or len(solo_digitos) > 15:
            return False, "El número debe tener entre 7 y 15 dígitos"
    
    return True, None

def validar_telefono_unico(
    bd: Session, 
    telefono: str, 
    usuario_id: Optional[int] = None,
    permitir_duplicados: bool = False
) -> None:
    """
    Valida que un teléfono no esté siendo usado por otro usuario
    
    Args:
        bd: Sesión de base de datos
        telefono: Número de teléfono a validar
        usuario_id: ID del usuario actual (para excluir en actualizaciones)
        permitir_duplicados: Si es True, solo advierte pero no bloquea
    
    Raises:
        HTTPException: Si el teléfono ya está en uso y no se permiten duplicados
    """
    # Importar aquí para evitar dependencias circulares
    from ..modelos.usuario import Usuario
    
    if not telefono:
        return
    
    # Normalizar teléfono (remover espacios)
    telefono_normalizado = telefono.strip()
    
    # Buscar usuarios con el mismo teléfono
    query = bd.query(Usuario).filter(Usuario.telefono == telefono_normalizado)
    
    # Excluir usuario actual si es actualización
    if usuario_id:
        query = query.filter(Usuario.id != usuario_id)
    
    usuarios_con_telefono = query.all()
    
    if usuarios_con_telefono:
        if permitir_duplicados:
            # Solo advertir en logs
            print(f"Advertencia: El teléfono {telefono} ya está en uso por {len(usuarios_con_telefono)} usuario(s)")
            for usuario in usuarios_con_telefono:
                print(f"   - Usuario ID {usuario.id}: {usuario.email}")
        else:
            # Bloquear registro/actualización
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El teléfono {telefono} ya está registrado por otro usuario"
            )

def validar_password_segura(password: str) -> Tuple[bool, List[str]]:
    """
    Validar que la contraseña cumpla con criterios de seguridad
    Returns: (es_valida, lista_errores)
    """
    errores = []
    
    if len(password) < 8:
        errores.append("Debe tener al menos 8 caracteres")
    
    if not re.search(r'[A-Z]', password):
        errores.append("Debe contener al menos una letra mayúscula")
    
    if not re.search(r'[a-z]', password):
        errores.append("Debe contener al menos una letra minúscula")
    
    if not re.search(r'\d', password):
        errores.append("Debe contener al menos un número")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errores.append("Se recomienda incluir al menos un símbolo especial")
    
    return len(errores) == 0, errores

def validar_fecha_nacimiento(fecha_str: str) -> bool:
    """Validar formato de fecha de nacimiento"""
    if not fecha_str:
        return True  # Opcional
    
    try:
        fecha = datetime.strptime(fecha_str, '%Y-%m-%d')
        
        # Verificar que no sea futura
        if fecha > datetime.now():
            return False
        
        # Verificar que no sea muy antigua (más de 120 años)
        if (datetime.now() - fecha).days > 120 * 365:
            return False
        
        return True
    except ValueError:
        return False

def limpiar_texto(texto: str) -> str:
    """Limpiar y normalizar texto"""
    if not texto:
        return ""
    
    # Remover espacios extra y convertir a mayúsculas
    return " ".join(texto.strip().split()).upper()

def validar_nombres(nombres: str) -> bool:
    """Validar que los nombres solo contengan letras y espacios"""
    if not nombres or not nombres.strip():
        return False
    
    # Solo letras, espacios y caracteres acentuados
    patron = r'^[a-zA-ZÁÉÍÓÚáéíóúÑñ\s]+$'
    return bool(re.match(patron, nombres.strip()))

def normalizar_telefono(telefono: str) -> str:
    """
    Normalizar formato de teléfono
    Remueve espacios y guiones, mantiene solo + y dígitos
    """
    if not telefono:
        return ""
    
    # Mantener solo el + inicial y los dígitos
    if telefono.startswith('+'):
        return '+' + ''.join(filter(str.isdigit, telefono[1:]))
    
    return ''.join(filter(str.isdigit, telefono))

class ValidadorDatos:
    """Clase para validaciones complejas"""
    
    @staticmethod
    def validar_datos_usuario(datos: Dict) -> Tuple[bool, Dict]:
        """
        Validar datos completos de usuario
        Returns: (es_valido, errores_por_campo)
        """
        errores = {}
        
        # Validar email
        if 'email' in datos:
            if not validar_email(datos['email']):
                errores['email'] = "Formato de email inválido"
        
        # Validar DNI
        if 'dni' in datos and datos['dni']:
            if not validar_dni(datos['dni']):
                errores['dni'] = "DNI debe tener exactamente 8 dígitos"
        
        # Validar pasaporte
        if 'pasaporte' in datos and datos['pasaporte']:
            if not validar_pasaporte(datos['pasaporte']):
                errores['pasaporte'] = "Pasaporte debe tener entre 6 y 20 caracteres alfanuméricos"
        
        # Validar teléfono
        if 'telefono' in datos and datos['telefono']:
            if not validar_telefono(datos['telefono']):
                errores['telefono'] = "Formato de teléfono inválido. Use formato internacional: +51987654321"
            else:
                # Validar longitud según país
                es_valido, mensaje = validar_longitud_telefono_por_pais(datos['telefono'])
                if not es_valido:
                    errores['telefono'] = mensaje
        
        # Validar nombres
        if 'nombres' in datos:
            if not validar_nombres(datos['nombres']):
                errores['nombres'] = "Nombres solo pueden contener letras y espacios"
        
        # Validar apellidos
        if 'apellido_paterno' in datos:
            if not validar_nombres(datos['apellido_paterno']):
                errores['apellido_paterno'] = "Apellido solo puede contener letras y espacios"
        
        if 'apellido_materno' in datos:
            if not validar_nombres(datos['apellido_materno']):
                errores['apellido_materno'] = "Apellido solo puede contener letras y espacios"
        
        # Validar fecha de nacimiento
        if 'fecha_nacimiento' in datos and datos['fecha_nacimiento']:
            if not validar_fecha_nacimiento(datos['fecha_nacimiento']):
                errores['fecha_nacimiento'] = "Fecha de nacimiento inválida"
        
        return len(errores) == 0, errores
    
    @staticmethod
    def validar_documento_identidad(tipo_usuario: str, dni: Optional[str], pasaporte: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validar que el documento de identidad sea apropiado según el tipo de usuario
        Returns: (es_valido, mensaje_error)
        """
        if tipo_usuario == "extranjero":
            if not pasaporte:
                return False, "Pasaporte es requerido para extranjeros"
            if not validar_pasaporte(pasaporte):
                return False, "Formato de pasaporte inválido"
        else:  # peruano_mayor o peruano_menor
            if not dni:
                return False, f"DNI es requerido para {tipo_usuario}"
            if not validar_dni(dni):
                return False, "DNI debe tener exactamente 8 dígitos"
        
        return True, None