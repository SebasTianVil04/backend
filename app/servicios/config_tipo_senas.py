import logging
import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

CONFIG_SENAS = {
    'ESTATICA': {
        'num_frames_recomendado': 8,
        'fps_muestreo': 10,
        'enfoque': 'centro_secuencia',
        'requiere_estabilidad': True,
        'umbral_confianza': 0.65,
        'procesamiento': 'frames_centrales',
        'augmentation_brightness': (0.7, 1.3),
        'augmentation_rotation': 15
    },
    'DINAMICA': {
        'num_frames_recomendado': 16,
        'fps_muestreo': 15,
        'enfoque': 'secuencia_completa',
        'requiere_estabilidad': False,
        'umbral_confianza': 0.55,
        'procesamiento': 'distribucion_uniforme',
        'augmentation_brightness': (0.6, 1.4),
        'augmentation_rotation': 10
    }
}

SENAS_ESTATICAS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'hola', 'gracias', 'si', 'no', 'ayuda', 'casa', 'agua', 'comida', 'baño',
    'familia', 'amigo', 'trabajo', 'escuela', 'doctor', 'hospital', 'dinero',
    'por favor', 'de nada', 'lo siento', 'bien', 'mal'
}

PATRONES_DINAMICOS = [
    r'.*familia.*', r'.*tiempo.*', r'.*trabajo.*', r'.*escuela.*',
    r'.*amigo.*', r'.*favorito.*', r'.*gustar.*', r'.*querer.*',
    r'.*necesitar.*', r'.*entender.*', r'.*explicar.*', r'.*preguntar.*',
    r'.*contar.*', r'.*enseñar.*', r'.*aprender.*', r'.*trabajar.*',
    r'.*caminar.*', r'.*correr.*', r'.*saltar.*', r'.*bailar.*'
]

CONFUSIONES_COMUNES = {
    'A': ['C', 'S', 'E', 'M', 'N', 'T'],
    'B': ['1', 'D', 'P', 'R', '4'],
    'C': ['A', 'O', 'D', '3', 'G', '0', 'E'],
    'D': ['B', 'C', 'P', 'Q', '1', 'F'],
    'E': ['3', 'M', 'S', 'A', 'C', 'O'],
    'F': ['9', 'P', 'R', 'D', 'K'],
    'G': ['C', 'Q', '6', 'H', 'O'],
    'H': ['8', 'N', 'M', 'U', 'R'],
    'I': ['1', 'L', '7', 'J', 'Y'],
    'J': ['Z', '1', '7', 'I', 'T'],
    'K': ['X', 'R', '4', 'P', 'V'],
    'L': ['1', 'I', '7', 'U', 'G'],
    'M': ['N', '3', 'W', 'A', 'E', 'S'],
    'N': ['M', 'Z', '2', 'A', 'H'],
    'Ñ': ['N', 'M', 'A'],
    'O': ['0', 'C', 'Q', 'E', 'G'],
    'P': ['B', 'F', 'R', 'D', 'K', 'Q'],
    'Q': ['O', 'G', '2', 'P', 'A', 'C'],
    'R': ['K', 'P', 'F', 'U', 'V'],
    'S': ['A', '5', '8', 'E', 'M', 'T'],
    'T': ['7', 'J', '1', 'I', 'A', 'S'],
    'U': ['V', 'W', '2', 'N', 'H', 'R'],
    'V': ['U', 'W', '2', 'K', 'R'],
    'W': ['M', 'U', 'V', '3', 'N'],
    'X': ['K', 'Y', '4', 'R'],
    'Y': ['X', 'V', '4', 'I', 'L'],
    'Z': ['2', 'N', '7', 'J', '1'],
    '0': ['O', 'C', 'Q', 'G', 'E'],
    '1': ['I', 'L', 'J', 'B', 'D', 'T', '7'],
    '2': ['V', 'U', 'N', 'Z', 'K'],
    '3': ['E', 'M', 'C', 'W', 'A'],
    '4': ['Y', 'X', 'B', 'K'],
    '5': ['S', 'A', '8'],
    '6': ['G', 'Q', 'W'],
    '7': ['T', 'J', 'I', '1', 'L'],
    '8': ['H', 'S', 'U', 'B', '5'],
    '9': ['F', 'P', 'R', 'G'],
    'HOLA': ['AYUDA', 'GRACIAS', 'POR FAVOR', 'ADIOS'],
    'GRACIAS': ['HOLA', 'POR FAVOR', 'ADIOS', 'DE NADA'],
    'AYUDA': ['HOLA', 'POR FAVOR', 'NECESITO', 'QUIERO'],
    'SI': ['NO', 'BIEN', 'BUENO'],
    'NO': ['SI', 'MAL', 'NADA'],
    'BIEN': ['BUENO', 'SI', 'GRACIAS'],
    'MAL': ['NO', 'TRISTE', 'MALO'],
    'POR FAVOR': ['GRACIAS', 'AYUDA', 'QUIERO', 'NECESITO'],
    'DE NADA': ['GRACIAS', 'BIEN', 'OK'],
    'LO SIENTO': ['DISCULPA', 'PERDON', 'MAL'],
    'ADIOS': ['HOLA', 'GRACIAS', 'HASTA LUEGO'],
    'FAMILIA': ['CASA', 'PADRES', 'HERMANOS'],
    'AMIGO': ['PERSONA', 'CONOCIDO', 'COMPAÑERO'],
    'CASA': ['HOGAR', 'FAMILIA', 'VIVIR'],
    'AGUA': ['BEBER', 'TOMAR', 'LIQUIDO'],
    'COMIDA': ['COMER', 'ALIMENTO', 'HAMBRE'],
}

GRUPOS_ALTA_CONFUSION = [
    {'A', 'C', 'S', 'E', 'M', 'N', 'T'},
    {'B', 'D', 'P', '1', 'I', 'L'},
    {'O', 'C', 'Q', '0', 'G', 'E'},
    {'U', 'V', 'W', '2', 'N', 'H'},
    {'K', 'R', 'P', 'F', 'X'},
    {'1', '7', 'I', 'J', 'T', 'L'},
    {'3', 'W', 'M', 'E'},
    {'5', 'S', '8', 'A'},
    {'6', 'G', '9', 'F'},
]

SENAS_NO_FLIP = {
    'A', 'C', 'D', 'G', 'J', 'P', 'Q', 'Z', 'B', 'F', 'R', 'T',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'ADIOS', 'VENIR', 'IR', 'DERECHA', 'IZQUIERDA', 'ARRIBA', 'ABAJO'
}

CONFIG_CONFUSIONES_NUMERICAS = {
    '0': {
        'confusiones_principales': ['C', 'O', 'Q', 'G'],
        'caracteristicas_clave': ['forma_circular_completa', 'pulgar_junto_dedos', 'apertura_minima'],
        'penalizacion': 4.0,
        'augmentation_especial': {
            'rotation_range': 1.5,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.98, 1.02),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_C': 'C tiene apertura lateral, 0 es círculo cerrado',
            'vs_O': 'O tiene dedos más juntos, 0 forma perfecta',
            'vs_Q': 'Q tiene índice hacia abajo, 0 no'
        }
    },
    '1': {
        'confusiones_principales': ['D', 'I', 'L', 'J', 'T', 'B'],
        'caracteristicas_clave': ['dedo_indice_solo', 'vertical_estricto', 'otros_dedos_cerrados'],
        'penalizacion': 4.0,
        'augmentation_especial': {
            'rotation_range': 1.0,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.98, 1.02),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_D': 'D forma círculo con índice, 1 solo índice recto',
            'vs_I': 'I meñique solo, 1 es índice',
            'vs_L': 'L tiene pulgar perpendicular, 1 no'
        }
    },
    '2': {
        'confusiones_principales': ['V', 'U', 'N', 'K', 'Z'],
        'caracteristicas_clave': ['dos_dedos_separados', 'forma_V', 'palma_orientacion'],
        'penalizacion': 3.5,
        'augmentation_especial': {
            'rotation_range': 2.0,
            'brightness_range': (0.93, 1.07),
            'contrast_range': (0.96, 1.04),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_V': 'V letras más cerradas, 2 más abierto',
            'vs_U': 'U dedos juntos, 2 separados',
            'vs_K': 'K tiene índice y medio en ángulo diferente'
        }
    },
    '3': {
        'confusiones_principales': ['E', 'M', 'W', 'C'],
        'caracteristicas_clave': ['tres_dedos_extendidos', 'pulgar_meñique_juntos'],
        'penalizacion': 3.5,
        'augmentation_especial': {
            'rotation_range': 2.0,
            'brightness_range': (0.93, 1.07),
            'contrast_range': (0.96, 1.04),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_E': 'E dedos más doblados, 3 más extendidos',
            'vs_W': 'W tiene separación diferente entre dedos',
            'vs_M': 'M mano más cerrada'
        }
    },
    '4': {
        'confusiones_principales': ['B', 'Y', 'X', 'K'],
        'caracteristicas_clave': ['cuatro_dedos_juntos', 'pulgar_cruzado'],
        'penalizacion': 3.0,
        'augmentation_especial': {
            'rotation_range': 2.5,
            'brightness_range': (0.92, 1.08),
            'contrast_range': (0.95, 1.05),
            'no_flip': True,
            'no_zoom': False
        },
        'diferencias_criticas': {
            'vs_B': 'B pulgar delante, 4 pulgar cruzado',
            'vs_Y': 'Y meñique y pulgar extendidos'
        }
    },
    '5': {
        'confusiones_principales': ['S', 'A', '8', 'B'],
        'caracteristicas_clave': ['cinco_dedos_extendidos', 'mano_abierta_completa'],
        'penalizacion': 2.5,
        'augmentation_especial': {
            'rotation_range': 3.0,
            'brightness_range': (0.90, 1.10),
            'contrast_range': (0.94, 1.06),
            'no_flip': True,
            'no_zoom': False
        },
        'diferencias_criticas': {
            'vs_S': 'S mano cerrada, 5 abierta',
            'vs_8': '8 posición y orientación diferente'
        }
    },
    '6': {
        'confusiones_principales': ['G', 'W', 'Q', 'F'],
        'caracteristicas_clave': ['pulgar_meñique_unidos', 'tres_dedos_medio_extendidos'],
        'penalizacion': 3.0,
        'augmentation_especial': {
            'rotation_range': 2.5,
            'brightness_range': (0.92, 1.08),
            'contrast_range': (0.95, 1.05),
            'no_flip': True,
            'no_zoom': False
        },
        'diferencias_criticas': {
            'vs_G': 'G índice extendido lateral, 6 diferente orientación',
            'vs_W': 'W tres dedos separados uniformes'
        }
    },
    '7': {
        'confusiones_principales': ['T', 'J', 'I', '1', 'L'],
        'caracteristicas_clave': ['pulgar_entre_indice_medio', 'orientacion_especifica'],
        'penalizacion': 3.5,
        'augmentation_especial': {
            'rotation_range': 1.5,
            'brightness_range': (0.94, 1.06),
            'contrast_range': (0.96, 1.04),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_T': 'T pulgar bajo puño, 7 entre dedos',
            'vs_1': '1 solo índice, 7 configuración compleja'
        }
    },
    '8': {
        'confusiones_principales': ['H', 'S', 'U', '5', 'B'],
        'caracteristicas_clave': ['indice_medio_extendidos_juntos', 'orientacion_horizontal'],
        'penalizacion': 3.0,
        'augmentation_especial': {
            'rotation_range': 2.0,
            'brightness_range': (0.92, 1.08),
            'contrast_range': (0.95, 1.05),
            'no_flip': True,
            'no_zoom': False
        },
        'diferencias_criticas': {
            'vs_H': 'H similar pero orientación y separación diferente',
            'vs_U': 'U vertical, 8 horizontal'
        }
    },
    '9': {
        'confusiones_principales': ['F', 'P', 'R', 'G', 'OK'],
        'caracteristicas_clave': ['pulgar_indice_circulo', 'otros_dedos_juntos'],
        'penalizacion': 3.0,
        'augmentation_especial': {
            'rotation_range': 2.5,
            'brightness_range': (0.92, 1.08),
            'contrast_range': (0.95, 1.05),
            'no_flip': True,
            'no_zoom': False
        },
        'diferencias_criticas': {
            'vs_F': 'F círculo más pequeño, orientación diferente',
            'vs_OK': 'OK es señal universal, 9 es número específico'
        }
    },
    'C': {
        'confusiones_principales': ['0', 'A', 'O', 'E', 'G'],
        'caracteristicas_clave': ['forma_C', 'apertura_lateral', 'no_circulo_completo'],
        'penalizacion': 4.0,
        'augmentation_especial': {
            'rotation_range': 1.5,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.98, 1.02),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_0': 'C abierto a un lado, 0 círculo cerrado',
            'vs_O': 'C más abierto que O',
            'vs_A': 'A puño cerrado, C semi-abierto'
        }
    },
    'D': {
        'confusiones_principales': ['1', 'B', 'P', 'F'],
        'caracteristicas_clave': ['indice_levantado_circulo', 'pulgar_tocando_medio'],
        'penalizacion': 4.0,
        'augmentation_especial': {
            'rotation_range': 1.5,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.98, 1.02),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_1': 'D forma círculo, 1 solo índice recto',
            'vs_B': 'B cuatro dedos juntos, D solo índice'
        }
    },
    'O': {
        'confusiones_principales': ['0', 'C', 'Q', 'E'],
        'caracteristicas_clave': ['circulo_dedos_juntos', 'sin_apertura'],
        'penalizacion': 4.0,
        'augmentation_especial': {
            'rotation_range': 1.5,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.98, 1.02),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_0': 'O letra más pequeno y cerrado',
            'vs_C': 'O más cerrado que C',
            'vs_Q': 'Q tiene índice apuntando abajo'
        }
    },
}

tipo_sena_config = {
    'ESTATICA': {
        'frames_entrenamiento': 8,
        'fps': 10,
        'augmentation': {
            'brightness_range': (0.7, 1.3),
            'rotation_range': 15,
            'flip_prob': 0.5
        }
    },
    'DINAMICA': {
        'frames_entrenamiento': 16,
        'fps': 15,
        'augmentation': {
            'brightness_range': (0.6, 1.4),
            'rotation_range': 10,
            'flip_prob': 0.5,
            'noise_level': 10,
            'frame_drop_prob': 0.3
        }
    }
}

def detectar_tipo_sena(nombre_sena: str) -> str:
    if not nombre_sena:
        return 'DINAMICA'
    
    nombre_limpio = nombre_sena.lower().strip()
    
    if nombre_limpio in SENAS_ESTATICAS:
        return 'ESTATICA'
    
    if len(nombre_limpio) == 1 and nombre_limpio.isalpha():
        return 'ESTATICA'
    
    if nombre_limpio.isdigit() and len(nombre_limpio) <= 2:
        return 'ESTATICA'
    
    for patron in PATRONES_DINAMICOS:
        if re.match(patron, nombre_limpio):
            return 'DINAMICA'
    
    if len(nombre_limpio.split()) == 1 and len(nombre_limpio) <= 8:
        return 'ESTATICA'
    
    return 'DINAMICA'

def obtener_config_sena(nombre_sena: str) -> Dict[str, Any]:
    tipo_sena = detectar_tipo_sena(nombre_sena)
    config = CONFIG_SENAS[tipo_sena].copy()
    config['tipo_detectado'] = tipo_sena
    config['nombre_sena'] = nombre_sena
    
    sena_upper = nombre_sena.upper()
    config['es_confusa'] = sena_upper in CONFUSIONES_COMUNES
    config['permite_flip'] = sena_upper not in SENAS_NO_FLIP
    
    config_confusion_numerica = obtener_config_confusion_numerica(nombre_sena)
    if config_confusion_numerica:
        config['es_confusion_numerica'] = True
        config['confusiones_principales'] = config_confusion_numerica.get('confusiones_principales', [])
        config['caracteristicas_clave'] = config_confusion_numerica.get('caracteristicas_clave', [])
        config['penalizacion_especial'] = config_confusion_numerica.get('penalizacion', 4.0)
        config['diferencias_criticas'] = config_confusion_numerica.get('diferencias_criticas', {})
        
        augmentation_especial = config_confusion_numerica.get('augmentation_especial', {})
        config['augmentation_rotation'] = augmentation_especial.get('rotation_range', 1.5)
        config['augmentation_brightness'] = augmentation_especial.get('brightness_range', (0.95, 1.05))
        config['augmentation_contrast'] = augmentation_especial.get('contrast_range', (0.98, 1.02))
        config['permite_flip'] = not augmentation_especial.get('no_flip', True)
        config['permite_zoom'] = not augmentation_especial.get('no_zoom', True)
    elif config['es_confusa']:
        config['augmentation_rotation'] = 3.0
        config['augmentation_brightness'] = (0.9, 1.1)
        config['augmentation_contrast'] = (0.95, 1.05)
    
    config['grupo_confusion'] = obtener_grupo_confusion(nombre_sena)
    
    return config

def obtener_confusiones_comunes(sena: str) -> List[str]:
    sena_upper = sena.upper()
    return CONFUSIONES_COMUNES.get(sena_upper, [])

def es_confusion_comun(sena_esperada: str, sena_detectada: str) -> bool:
    if sena_esperada.upper() == sena_detectada.upper():
        return False
    
    confusiones = obtener_confusiones_comunes(sena_esperada)
    return sena_detectada.upper() in [c.upper() for c in confusiones]

def obtener_config_confusion_numerica(sena: str) -> Dict[str, Any]:
    sena_upper = sena.upper()
    return CONFIG_CONFUSIONES_NUMERICAS.get(sena_upper, {})

def es_confusion_numerica(sena_esperada: str, sena_detectada: str) -> bool:
    sena_esperada_upper = sena_esperada.upper()
    sena_detectada_upper = sena_detectada.upper()
    
    config_esperada = CONFIG_CONFUSIONES_NUMERICAS.get(sena_esperada_upper, {})
    if config_esperada:
        confusiones_principales = config_esperada.get('confusiones_principales', [])
        if sena_detectada_upper in confusiones_principales:
            return True
    
    config_detectada = CONFIG_CONFUSIONES_NUMERICAS.get(sena_detectada_upper, {})
    if config_detectada:
        confusiones_principales = config_detectada.get('confusiones_principales', [])
        if sena_esperada_upper in confusiones_principales:
            return True
    
    return False

def calcular_factor_confusion_mejorado(sena_esperada: str, sena_detectada: str) -> float:
    if sena_esperada.upper() == sena_detectada.upper():
        return 1.0
    
    sena_esperada_upper = sena_esperada.upper()
    sena_detectada_upper = sena_detectada.upper()
    
    if es_confusion_numerica(sena_esperada, sena_detectada):
        confusiones_criticas = [
            ('0', 'C'), ('0', 'O'), ('C', '0'), ('O', '0'),
            ('1', 'D'), ('1', 'I'), ('1', 'L'), ('D', '1'),
            ('2', 'V'), ('2', 'U'), ('V', '2'), ('U', '2'),
            ('3', 'E'), ('3', 'M'), ('E', '3'), ('M', '3'),
        ]
        
        if (sena_esperada_upper, sena_detectada_upper) in confusiones_criticas:
            return 4.5
        
        return 4.0
    
    grupo_esperada = obtener_grupo_confusion(sena_esperada)
    if grupo_esperada and sena_detectada_upper in grupo_esperada:
        if grupo_esperada == {'A', 'C', 'S', 'E', 'M', 'N', 'T'}:
            return 4.0
        return 3.5
    
    if es_confusion_comun(sena_esperada, sena_detectada):
        return 2.5
    
    return 1.5

def obtener_grupo_confusion(sena: str) -> Optional[Set[str]]:
    sena_upper = sena.upper()
    for grupo in GRUPOS_ALTA_CONFUSION:
        if sena_upper in grupo:
            return grupo
    return None

def calcular_factor_confusion(sena_esperada: str, sena_detectada: str) -> float:
    return calcular_factor_confusion_mejorado(sena_esperada, sena_detectada)

def validar_consistencia_categoria(senas: List[str]) -> Dict[str, Any]:
    tipos = [detectar_tipo_sena(sena) for sena in senas]
    conteo_tipos = {
        'ESTATICA': tipos.count('ESTATICA'),
        'DINAMICA': tipos.count('DINAMICA')
    }
    
    total = len(tipos)
    proporcion_estaticas = conteo_tipos['ESTATICA'] / total if total > 0 else 0
    proporcion_dinamicas = conteo_tipos['DINAMICA'] / total if total > 0 else 0
    
    tipo_mayoritario = 'ESTATICA' if conteo_tipos['ESTATICA'] > conteo_tipos['DINAMICA'] else 'DINAMICA'
    
    senas_confusas = [s for s in senas if s.upper() in CONFUSIONES_COMUNES]
    senas_confusion_numerica = []
    senas_grupo_critico = []
    
    for sena in senas:
        config = obtener_config_confusion_numerica(sena)
        if config:
            senas_confusion_numerica.append(sena)
        
        grupo = obtener_grupo_confusion(sena)
        if grupo and any(s in ['A', 'C', 'S', 'E', 'M'] for s in grupo):
            senas_grupo_critico.append(sena)
    
    tiene_confusiones = len(senas_confusas) > 0
    
    if len(senas_confusion_numerica) > 0:
        nivel_dificultad = 'MUY_ALTO'
    elif len(senas_grupo_critico) > 0:
        nivel_dificultad = 'ALTO'
    elif tiene_confusiones:
        nivel_dificultad = 'MEDIO'
    else:
        nivel_dificultad = 'BAJO'
    
    return {
        'total_senas': total,
        'conteo_tipos': conteo_tipos,
        'proporcion_estaticas': round(proporcion_estaticas, 3),
        'proporcion_dinamicas': round(proporcion_dinamicas, 3),
        'tipo_mayoritario': tipo_mayoritario,
        'consistente': (proporcion_estaticas > 0.8 or proporcion_dinamicas > 0.8),
        'tiene_confusiones': tiene_confusiones,
        'senas_confusas': senas_confusas,
        'senas_confusion_numerica': senas_confusion_numerica,
        'senas_grupo_critico': senas_grupo_critico,
        'nivel_dificultad': nivel_dificultad
    }

def es_categoria_alfabeto(senas: List[str]) -> bool:
    if not senas:
        return False
    
    letras_alfabeto = set('abcdefghijklmnopqrstuvwxyzñ')
    conteo_letras = sum(1 for sena in senas if sena.lower() in letras_alfabeto)
    
    return conteo_letras / len(senas) > 0.7

def es_categoria_numeros(senas: List[str]) -> bool:
    if not senas:
        return False
    
    numeros = set('0123456789')
    conteo_numeros = sum(1 for sena in senas if sena in numeros)
    
    return conteo_numeros / len(senas) > 0.7

def obtener_parametros_entrenamiento(senas: List[str]) -> Dict[str, Any]:
    analisis_consistencia = validar_consistencia_categoria(senas)
    es_alfabeto = es_categoria_alfabeto(senas)
    es_numeros = es_categoria_numeros(senas)
    
    parametros = {
        'batch_size': 8,
        'num_frames': 16,
        'learning_rate': 0.001,
        'epochs': 150,
        'weight_decay': 1e-4,
        'patience': 30
    }
    
    nivel_dificultad = analisis_consistencia['nivel_dificultad']
    
    if nivel_dificultad == 'MUY_ALTO':
        parametros['learning_rate'] = 0.0003
        parametros['epochs'] = 250
        parametros['patience'] = 40
        parametros['confusion_penalty'] = 4.5
        parametros['focal_gamma'] = 2.5
        parametros['oversampling'] = '4x para numéricas, 3x para críticas'
        
    elif nivel_dificultad == 'ALTO':
        parametros['learning_rate'] = 0.0005
        parametros['epochs'] = 200
        parametros['patience'] = 35
        parametros['confusion_penalty'] = 4.0
        parametros['focal_gamma'] = 2.5
        parametros['oversampling'] = '3x para críticas, 2x para confusas'
        
    elif nivel_dificultad == 'MEDIO':
        parametros['learning_rate'] = 0.0007
        parametros['epochs'] = 175
        parametros['patience'] = 33
        parametros['confusion_penalty'] = 3.0
        parametros['focal_gamma'] = 2.0
        parametros['oversampling'] = '2x para confusas'
        
    else:
        parametros['confusion_penalty'] = 2.0
        parametros['focal_gamma'] = 2.0
        parametros['oversampling'] = '1x proporcional'
    
    if analisis_consistencia['tipo_mayoritario'] == 'ESTATICA':
        parametros['num_frames'] = 8
        parametros['batch_size'] = 12
    else:
        parametros['num_frames'] = 16
        parametros['batch_size'] = 6
    
    if es_alfabeto:
        parametros['epochs'] = max(parametros['epochs'], 250)
        parametros['learning_rate'] = min(parametros['learning_rate'], 0.0003)
    
    if es_numeros:
        parametros['epochs'] = max(parametros['epochs'], 300)
        parametros['learning_rate'] = 0.0003
        parametros['confusion_penalty'] = 4.5
    
    return {
        **parametros,
        'analisis_consistencia': analisis_consistencia,
        'es_categoria_alfabeto': es_alfabeto,
        'es_categoria_numeros': es_numeros,
        'nivel_dificultad': nivel_dificultad,
        'recomendaciones': generar_recomendaciones_entrenamiento(
            analisis_consistencia, es_alfabeto, es_numeros
        )
    }

def generar_recomendaciones_entrenamiento(consistencia: Dict[str, Any], 
                                         es_alfabeto: bool,
                                         es_numeros: bool) -> List[str]:
    recomendaciones = []
    nivel_dificultad = consistencia.get('nivel_dificultad', 'MEDIO')
    
    if consistencia['consistente']:
        if consistencia['tipo_mayoritario'] == 'ESTATICA':
            recomendaciones.append("✅ Categoría estática - usar 8 frames y procesamiento optimizado")
        else:
            recomendaciones.append("✅ Categoría dinámica - usar 16 frames y captura de movimiento")
    else:
        recomendaciones.append("⚠️  Categoría mixta - usar modelo adaptativo con doble backbone")
    
    if es_alfabeto:
        recomendaciones.append("📚 Categoría ALFABETO - entrenar con LR bajo (0.0003) y 250+ épocas")
    
    if es_numeros:
        recomendaciones.append("🔢 Categoría NÚMEROS - máxima precaución con confusiones numéricas")
        recomendaciones.append("   → Usar augmentation ultra-conservador")
        recomendaciones.append("   → NO aplicar flip horizontal")
        recomendaciones.append("   → Oversampling 4x")
    
    if nivel_dificultad == 'MUY_ALTO':
        recomendaciones.append("🔴🔴🔴 DIFICULTAD MUY ALTA - Confusiones numéricas detectadas")
        recomendaciones.append("   → Learning rate: 0.0003")
        recomendaciones.append("   → Epochs: 250+")
        recomendaciones.append("   → Penalización: 4.5x")
        recomendaciones.append("   → Augmentation: ±5% brightness, ±1.5° rotation")
        recomendaciones.append("   → Oversampling: 4x para confusas numéricas")
        
    elif nivel_dificultad == 'ALTO':
        recomendaciones.append("🔴🔴 DIFICULTAD ALTA - Grupo crítico detectado")
        recomendaciones.append("   → Learning rate: 0.0005")
        recomendaciones.append("   → Epochs: 200+")
        recomendaciones.append("   → Penalización: 4.0x")
        recomendaciones.append("   → Oversampling: 3x para grupo crítico")
        
    elif nivel_dificultad == 'MEDIO':
        recomendaciones.append("🟡 DIFICULTAD MEDIA - Señas confusas presentes")
        recomendaciones.append("   → Learning rate: 0.0007")
        recomendaciones.append("   → Epochs: 175")
        recomendaciones.append("   → Penalización: 3.0x")
        
    else:
        recomendaciones.append("🟢 DIFICULTAD BAJA - Sin confusiones significativas")
        recomendaciones.append("   → Parámetros estándar")
    
    if consistencia.get('senas_confusion_numerica'):
        recomendaciones.append(f"\n⚠️⚠️⚠️  CRÍTICO: {len(consistencia['senas_confusion_numerica'])} señas con confusión numérica:")
        for sena in consistencia['senas_confusion_numerica'][:5]:
            config = obtener_config_confusion_numerica(sena)
            confusiones = config.get('confusiones_principales', [])
            recomendaciones.append(f"   • {sena} ↔ {', '.join(confusiones)}")
    
    if consistencia.get('senas_grupo_critico'):
        recomendaciones.append(f"\n⚠️⚠️  ATENCIÓN: {len(consistencia['senas_grupo_critico'])} señas en grupo crítico:")
        recomendaciones.append(f"   {', '.join(consistencia['senas_grupo_critico'][:10])}")
    
    if consistencia['total_senas'] < 10:
        recomendaciones.append("\n💡 SUGERENCIA: Pocas señas para entrenar")
        recomendaciones.append("   → Considerar agregar más clases para mejor generalización")
    
    return recomendaciones

def analizar_pares_confusion(clases: List[str]) -> Dict[str, Any]:
    pares_confusion = []
    pares_criticos = []
    
    for i, clase1 in enumerate(clases):
        for j, clase2 in enumerate(clases):
            if i < j:
                if es_confusion_comun(clase1, clase2):
                    factor = calcular_factor_confusion_mejorado(clase1, clase2)
                    par = {
                        'sena1': clase1,
                        'sena2': clase2,
                        'factor_penalizacion': factor,
                        'es_numerica': es_confusion_numerica(clase1, clase2),
                        'mismo_grupo': obtener_grupo_confusion(clase1) == obtener_grupo_confusion(clase2)
                    }
                    pares_confusion.append(par)
                    
                    if factor >= 4.0:
                        pares_criticos.append(par)
    
    return {
        'total_pares': len(pares_confusion),
        'pares_criticos': len(pares_criticos),
        'pares_confusion': sorted(pares_confusion, key=lambda x: x['factor_penalizacion'], reverse=True),
        'pares_criticos_detalle': sorted(pares_criticos, key=lambda x: x['factor_penalizacion'], reverse=True)
    }

def generar_matriz_confusion_teorica(clases: List[str]) -> Dict[str, List[str]]:
    matriz = {}
    
    for clase in clases:
        confusiones_posibles = []
        confusiones_conocidas = obtener_confusiones_comunes(clase)
        
        for otra_clase in clases:
            if otra_clase != clase and otra_clase.upper() in [c.upper() for c in confusiones_conocidas]:
                confusiones_posibles.append(otra_clase)
        
        if confusiones_posibles:
            matriz[clase] = confusiones_posibles
    
    return matriz

def exportar_configuracion_confusiones(clases: List[str], ruta_salida: str = None):
    analisis = {
        'timestamp': datetime.now().isoformat(),
        'total_clases': len(clases),
        'clases': clases,
        'analisis_consistencia': validar_consistencia_categoria(clases),
        'pares_confusion': analizar_pares_confusion(clases),
        'matriz_confusion_teorica': generar_matriz_confusion_teorica(clases),
        'parametros_recomendados': obtener_parametros_entrenamiento(clases)
    }
    
    if ruta_salida:
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(analisis, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuración de confusiones exportada: {ruta_salida}")
    
    return analisis

def calibrar_modelo_usuario(usuario_id: int, senas_calibracion: Dict[str, str]) -> Dict[str, Any]:
    resultados = {}
    ajustes = {}
    
    for sena_esperada, sena_detectada in senas_calibracion.items():
        if sena_esperada != sena_detectada:
            tipo_esperado = detectar_tipo_sena(sena_esperada)
            tipo_detectado = detectar_tipo_sena(sena_detectada)
            factor_confusion = calcular_factor_confusion_mejorado(sena_esperada, sena_detectada)
            
            if tipo_esperado != tipo_detectado:
                ajustes[sena_esperada] = {
                    'tipo_corregido': tipo_esperado,
                    'confianza_ajustada': 0.7,
                    'factor_compensacion': 1.2,
                    'confusion_comun': es_confusion_comun(sena_esperada, sena_detectada),
                    'factor_penalizacion': factor_confusion
                }
            elif es_confusion_comun(sena_esperada, sena_detectada):
                ajustes[sena_esperada] = {
                    'confusion_comun': True,
                    'sena_confundida': sena_detectada,
                    'factor_penalizacion': factor_confusion,
                    'es_confusion_numerica': es_confusion_numerica(sena_esperada, sena_detectada)
                }
    
    return {
        'usuario_id': usuario_id,
        'ajustes_aplicados': ajustes,
        'total_errores': len(ajustes),
        'fecha_calibracion': datetime.now().isoformat()
    }

def generar_recomendaciones_calibracion(errores: Dict[str, Any]) -> List[str]:
    recomendaciones = []
    confusiones_numericas = 0
    confusiones_comunes = 0
    
    for sena, info_error in errores.items():
        if info_error.get('es_confusion_numerica', False):
            confusiones_numericas += 1
            sena_confundida = info_error.get('sena_confundida', 'desconocida')
            config = obtener_config_confusion_numerica(sena)
            if config:
                diferencias = config.get('diferencias_criticas', {})
                clave_diff = f"vs_{sena_confundida}"
                if clave_diff in diferencias:
                    recomendaciones.append(
                        f"🔴 CRÍTICO '{sena}' ↔ '{sena_confundida}': {diferencias[clave_diff]}"
                    )
                else:
                    recomendaciones.append(
                        f"🔴 Confusión numérica: '{sena}' se confunde con '{sena_confundida}'"
                    )
        elif info_error.get('confusion_comun'):
            confusiones_comunes += 1
            sena_confundida = info_error.get('sena_confundida', 'desconocida')
            recomendaciones.append(
                f"⚠️  Confusión común: '{sena}' ↔ '{sena_confundida}'. "
                f"Practica las diferencias clave."
            )
        elif info_error.get('tipo_corregido'):
            tipo_esperado = info_error['tipo_corregido']
            recomendaciones.append(
                f"ℹ️  Seña '{sena}' es {tipo_esperado.lower()}. "
                f"Ejecuta la seña {'sin movimiento' if tipo_esperado == 'ESTATICA' else 'con movimiento fluido'}."
            )
    
    if confusiones_numericas > 0:
        recomendaciones.append(
            f"\n🔴 {confusiones_numericas} confusiones NUMÉRICAS críticas detectadas. "
            "Estas son las más difíciles. Practica con especial cuidado."
        )
    
    if confusiones_comunes > 3:
        recomendaciones.append(
            f"\n⚠️  {confusiones_comunes} confusiones comunes detectadas. "
            "Considera mejorar iluminación y ángulo de cámara."
        )
    
    if len(errores) > 5:
        recomendaciones.append(
            "\n💡 SUGERENCIA: Muchos errores detectados. Recomendaciones:"
            "\n   • Verifica la iluminación (luz frontal uniforme)"
            "\n   • Ajusta el ángulo de la cámara"
            "\n   • Mantén la mano a distancia adecuada"
            "\n   • Practica las señas básicas primero"
        )
    
    if not recomendaciones:
        recomendaciones.append("✅ Calibración exitosa. No se detectaron errores significativos.")
    
    return recomendaciones

def obtener_estadisticas_confusiones() -> Dict[str, Any]:
    total_senas = len(CONFUSIONES_COMUNES)
    total_confusiones = sum(len(confusiones) for confusiones in CONFUSIONES_COMUNES.values())
    
    confusiones_numericas = len([k for k in CONFIG_CONFUSIONES_NUMERICAS.keys()])
    total_grupos = len(GRUPOS_ALTA_CONFUSION)
    senas_sin_flip = len(SENAS_NO_FLIP)
    
    return {
        'total_senas_confusas': total_senas,
        'total_pares_confusion': total_confusiones,
        'promedio_confusiones_por_sena': total_confusiones / total_senas if total_senas > 0 else 0,
        'confusiones_numericas_config': confusiones_numericas,
        'total_grupos_alta_confusion': total_grupos,
        'senas_sin_flip': senas_sin_flip,
        'top_senas_confusas': sorted(
            [(k, len(v)) for k, v in CONFUSIONES_COMUNES.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }

def validar_integridad_configuracion() -> Dict[str, Any]:
    errores = []
    advertencias = []
    
    for sena, confusiones in CONFUSIONES_COMUNES.items():
        for conf in confusiones:
            if conf in CONFUSIONES_COMUNES:
                if sena not in CONFUSIONES_COMUNES[conf]:
                    advertencias.append(
                        f"Confusión no bidireccional: {sena} → {conf}, pero {conf} no lista a {sena}"
                    )
    
    for sena in CONFIG_CONFUSIONES_NUMERICAS.keys():
        if sena not in CONFUSIONES_COMUNES:
            errores.append(
                f"Seña '{sena}' en CONFIG_CONFUSIONES_NUMERICAS pero no en CONFUSIONES_COMUNES"
            )
    
    for i, grupo in enumerate(GRUPOS_ALTA_CONFUSION):
        for sena in grupo:
            if sena not in CONFUSIONES_COMUNES:
                advertencias.append(
                    f"Seña '{sena}' en GRUPOS_ALTA_CONFUSION[{i}] pero no en CONFUSIONES_COMUNES"
                )
    
    return {
        'valido': len(errores) == 0,
        'errores': errores,
        'advertencias': advertencias,
        'total_errores': len(errores),
        'total_advertencias': len(advertencias)
    }