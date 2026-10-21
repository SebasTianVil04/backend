import logging
import re
import json
import unicodedata
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime


def redimensionar_manteniendo_aspecto(frame: np.ndarray, tamano: Tuple[int, int] = (224, 224)) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("frame vacío")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w = frame.shape[:2]
    ancho_destino, alto_destino = tamano
    escala = min(ancho_destino / w, alto_destino / h)
    nuevo_w, nuevo_h = max(1, int(w * escala)), max(1, int(h * escala))
    frame_redim = cv2.resize(frame, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((alto_destino, ancho_destino, 3), dtype=np.uint8)
    y_offset = (alto_destino - nuevo_h) // 2
    x_offset = (ancho_destino - nuevo_w) // 2
    canvas[y_offset:y_offset + nuevo_h, x_offset:x_offset + nuevo_w] = frame_redim
    return canvas


logger = logging.getLogger(__name__)


def _quitar_tildes(texto: str) -> str:
    texto = texto.replace('ñ', '\x00').replace('Ñ', '\x01')
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto.replace('\x00', 'ñ').replace('\x01', 'Ñ')


def normalizar_nombre(nombre: Optional[str]) -> str:
    if not nombre:
        return ''
    n = _quitar_tildes(str(nombre).lower())
    n = re.sub(r'[_\-]+', ' ', n)
    return re.sub(r'\s+', ' ', n).strip()


def _clave(nombre: Optional[str]) -> str:
    return normalizar_nombre(nombre).upper()


CONFIG_SENAS = {
    'ESTATICA': {
        'num_frames_recomendado': 8,
        'fps_muestreo': 10,
        'enfoque': 'centro_secuencia',
        'requiere_estabilidad': True,
        'umbral_confianza': 0.65,
        'procesamiento': 'frames_centrales',
        'augmentation_brightness': (0.8, 1.2),
        'augmentation_rotation': 10,
        'prioridad': 'NORMAL',
    },
    'DINAMICA': {
        'num_frames_recomendado': 16,
        'fps_muestreo': 15,
        'enfoque': 'secuencia_completa',
        'requiere_estabilidad': False,
        'umbral_confianza': 0.55,
        'procesamiento': 'distribucion_uniforme',
        'augmentation_brightness': (0.8, 1.2),
        'augmentation_rotation': 10,
        'prioridad': 'NORMAL',
    },
    'EMERGENCIA': {
        'num_frames_recomendado': 20,
        'fps_muestreo': 20,
        'enfoque': 'secuencia_completa_prioritaria',
        'requiere_estabilidad': False,
        'umbral_confianza': 0.45,
        'confirmaciones_consecutivas_requeridas': 2,
        'procesamiento': 'prioridad_maxima',
        'augmentation_brightness': (0.7, 1.3),
        'augmentation_rotation': 12,
        'prioridad': 'CRITICA',
        'oversampling_minimo': '4x',
    }
}

LETRAS_CON_MOVIMIENTO = {'j', 'z', 'ñ'}

SENAS_ESTATICAS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
}

SENAS_VERIFICADAS_DINAMICAS = {
    'doctor', 'hospital', 'beber', 'dentista', 'abogado', 'ayuda',
    'hola', 'gracias', 'si', 'no', 'casa', 'agua', 'comida', 'baño',
    'familia', 'amigo', 'trabajo', 'escuela', 'dinero',
    'por favor', 'de nada', 'lo siento', 'bien', 'mal', 'adios',
} | LETRAS_CON_MOVIMIENTO

SENAS_EMERGENCIA = {
    'sos', 'emergencia', 'socorro', 'auxilio', 'peligro', 'fuego',
    'accidente', 'policia', 'ambulancia', 'me siento mal',
    'no puedo respirar', 'estoy atrapado', 'duele mucho',
    'necesito ayuda urgente', 'llamen a alguien',
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
    'G': ['C', 'Q', '6', 'H', 'O', 'L'],
    'H': ['8', 'N', 'M', 'U', 'R', 'G'],
    'I': ['1', 'L', '7', 'J', 'Y', 'T'],
    'J': ['Z', '1', '7', 'I', 'T'],
    'K': ['X', 'R', '4', 'P', 'V', 'F', '2'],
    'L': ['1', 'I', '7', 'U', 'G', 'Y'],
    'M': ['N', '3', 'W', 'A', 'E', 'S', 'H'],
    'N': ['M', 'Z', '2', 'A', 'H', 'Ñ', 'U', 'W'],
    'Ñ': ['N', 'M', 'A'],
    'O': ['0', 'C', 'Q', 'E', 'G'],
    'P': ['B', 'F', 'R', 'D', 'K', 'Q', '9'],
    'Q': ['O', 'G', '2', 'P', 'A', 'C', 'D', '0', '6'],
    'R': ['K', 'P', 'F', 'U', 'V', 'B', 'H', 'X', '9'],
    'S': ['A', '5', '8', 'E', 'M', 'T'],
    'T': ['7', 'J', '1', 'I', 'A', 'S'],
    'U': ['V', 'W', '2', 'N', 'H', 'R', 'L', '8'],
    'V': ['U', 'W', '2', 'K', 'R', 'Y'],
    'W': ['M', 'U', 'V', '3', 'N', '6'],
    'X': ['K', 'Y', '4', 'R'],
    'Y': ['X', 'V', '4', 'I', 'L'],
    'Z': ['2', 'N', '7', 'J', '1'],
    '0': ['O', 'C', 'Q', 'G', 'E'],
    '1': ['I', 'L', 'J', 'B', 'D', 'T', '7', 'Z'],
    '2': ['V', 'U', 'N', 'Z', 'K', 'Q'],
    '3': ['E', 'M', 'C', 'W', 'A'],
    '4': ['Y', 'X', 'B', 'K'],
    '5': ['S', 'A', '8'],
    '6': ['G', 'Q', 'W'],
    '7': ['T', 'J', 'I', '1', 'L', 'Z'],
    '8': ['H', 'S', 'U', 'B', '5'],
    '9': ['F', 'P', 'R', 'G'],
    'HOLA': ['GRACIAS', 'POR FAVOR', 'ADIOS'],
    'GRACIAS': ['HOLA', 'POR FAVOR', 'DE NADA'],
    'AYUDA': ['POR FAVOR', 'SOS', 'SOCORRO', 'AUXILIO'],
    'SI': ['NO', 'BIEN'],
    'NO': ['SI', 'MAL'],
    'BIEN': ['MAL', 'SI', 'GRACIAS'],
    'MAL': ['BIEN', 'NO', 'LO SIENTO'],
    'POR FAVOR': ['GRACIAS', 'AYUDA'],
    'DE NADA': ['GRACIAS', 'BIEN'],
    'LO SIENTO': ['MAL', 'GRACIAS'],
    'ADIOS': ['HOLA', 'GRACIAS'],
    'FAMILIA': ['AMIGO', 'CASA'],
    'AMIGO': ['FAMILIA'],
    'CASA': ['FAMILIA', 'HOSPITAL'],
    'AGUA': ['COMIDA'],
    'COMIDA': ['AGUA'],
    'DOCTOR': ['HOSPITAL', 'AMBULANCIA', 'ACCIDENTE'],
    'HOSPITAL': ['DOCTOR', 'CASA', 'AMBULANCIA'],
    'BEBER': ['AGUA', 'COMIDA'],
    'DENTISTA': ['DOCTOR', 'HOSPITAL'],
    'ABOGADO': ['DOCTOR', 'POLICIA'],
    'BAÑO': ['AGUA', 'CASA'],
    'TRABAJO': ['ESCUELA', 'DINERO'],
    'ESCUELA': ['TRABAJO', 'CASA'],
    'DINERO': ['TRABAJO', 'COMIDA'],
    'SOS': ['AYUDA', 'SOCORRO', 'AUXILIO', 'EMERGENCIA'],
    'EMERGENCIA': ['SOS', 'PELIGRO', 'AYUDA', 'ACCIDENTE'],
    'SOCORRO': ['SOS', 'AYUDA', 'AUXILIO'],
    'AUXILIO': ['SOS', 'AYUDA', 'SOCORRO'],
    'PELIGRO': ['EMERGENCIA', 'FUEGO', 'ACCIDENTE'],
    'FUEGO': ['PELIGRO', 'ACCIDENTE'],
    'ACCIDENTE': ['EMERGENCIA', 'PELIGRO', 'FUEGO', 'DOCTOR'],
    'POLICIA': ['AYUDA', 'EMERGENCIA', 'PELIGRO'],
    'AMBULANCIA': ['DOCTOR', 'HOSPITAL', 'EMERGENCIA'],
    'ME SIENTO MAL': ['DUELE MUCHO', 'NO PUEDO RESPIRAR', 'MAL'],
    'DUELE MUCHO': ['ME SIENTO MAL', 'DOCTOR'],
    'NO PUEDO RESPIRAR': ['ME SIENTO MAL', 'EMERGENCIA'],
    'ESTOY ATRAPADO': ['PELIGRO', 'AYUDA'],
    'NECESITO AYUDA URGENTE': ['AYUDA', 'SOS', 'EMERGENCIA'],
    'LLAMEN A ALGUIEN': ['AYUDA', 'SOS'],
}


def _simetrizar_confusiones(dic: Dict[str, List[str]]) -> Dict[str, List[str]]:
    for clave, confusiones in list(dic.items()):
        for otra in list(confusiones):
            if otra not in dic:
                dic[otra] = []
            if clave not in dic[otra]:
                dic[otra].append(clave)
    return dic


CONFUSIONES_COMUNES = _simetrizar_confusiones(CONFUSIONES_COMUNES)

GRUPO_CRITICO_ALFABETO = {'A', 'C', 'S', 'E', 'M', 'N', 'T'}

GRUPOS_ALTA_CONFUSION = [
    GRUPO_CRITICO_ALFABETO,
    {'B', 'D', 'P', '1', 'I', 'L'},
    {'O', 'C', 'Q', '0', 'G', 'E'},
    {'U', 'V', 'W', '2', 'N', 'H'},
    {'K', 'R', 'P', 'F', 'X'},
    {'1', '7', 'I', 'J', 'T', 'L'},
    {'3', 'W', 'M', 'E'},
    {'5', 'S', '8', 'A'},
    {'6', 'G', '9', 'F'},
    {'SOS', 'SOCORRO', 'AUXILIO', 'EMERGENCIA'},
]

SENAS_NO_FLIP = {
    'A', 'C', 'D', 'G', 'J', 'P', 'Q', 'Z', 'Ñ', 'B', 'F', 'R', 'T',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'ADIOS', 'VENIR', 'IR', 'DERECHA', 'IZQUIERDA', 'ARRIBA', 'ABAJO',
    'SOS', 'EMERGENCIA', 'SOCORRO', 'AUXILIO', 'PELIGRO', 'FUEGO', 'POLICIA',
}

CONFIG_CONFUSIONES_NUMERICAS = {
    '0': {
        'confusiones_principales': ['C', 'O', 'Q', 'G'],
        'caracteristicas_clave': ['forma_circular_completa', 'pulgar_junto_dedos', 'apertura_minima'],
        'penalizacion': 3.0,
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
        'penalizacion': 3.0,
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
            'vs_L': 'L tiene pulgar perpendicular, 1 no',
            'vs_B': '1 extiende solo el índice, B extiende los cuatro dedos juntos'
        }
    },
    '2': {
        'confusiones_principales': ['V', 'U', 'N', 'K', 'Z'],
        'caracteristicas_clave': ['dos_dedos_separados', 'forma_V', 'palma_orientacion'],
        'penalizacion': 2.5,
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
        'penalizacion': 2.5,
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
        'caracteristicas_clave': ['cuatro_dedos_separados', 'pulgar_cruzado'],
        'penalizacion': 2.0,
        'augmentation_especial': {
            'rotation_range': 2.5,
            'brightness_range': (0.92, 1.08),
            'contrast_range': (0.95, 1.05),
            'no_flip': True,
            'no_zoom': False
        },
        'diferencias_criticas': {
            'vs_B': '4 separa los cuatro dedos, B los mantiene juntos',
            'vs_Y': 'Y meñique y pulgar extendidos'
        }
    },
    '5': {
        'confusiones_principales': ['S', 'A', '8', 'B'],
        'caracteristicas_clave': ['cinco_dedos_extendidos', 'mano_abierta_completa'],
        'penalizacion': 2.0,
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
        'penalizacion': 2.0,
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
        'penalizacion': 2.5,
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
        'penalizacion': 2.0,
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
        'confusiones_principales': ['F', 'P', 'R', 'G'],
        'caracteristicas_clave': ['pulgar_indice_circulo', 'otros_dedos_juntos'],
        'penalizacion': 2.0,
        'augmentation_especial': {
            'rotation_range': 2.5,
            'brightness_range': (0.92, 1.08),
            'contrast_range': (0.95, 1.05),
            'no_flip': True,
            'no_zoom': False
        },
        'diferencias_criticas': {
            'vs_F': 'F círculo más pequeño, orientación diferente'
        }
    },
    'B': {
        'confusiones_principales': ['D', 'P', '1', '4', 'R'],
        'caracteristicas_clave': ['cuatro_dedos_juntos_extendidos', 'pulgar_cruzado_sobre_palma', 'mano_plana_vertical'],
        'penalizacion': 3.0,
        'augmentation_especial': {
            'rotation_range': 1.5,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.98, 1.02),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_D': 'B tiene los cuatro dedos juntos y extendidos, D solo el índice extendido en círculo con el pulgar',
            'vs_P': 'P apunta hacia abajo con el pulgar entre índice y medio, B es vertical con dedos juntos',
            'vs_1': '1 extiende solo el índice, B extiende los cuatro dedos',
            'vs_4': '4 separa los cuatro dedos, B los mantiene juntos'
        }
    },
    'C': {
        'confusiones_principales': ['0', 'A', 'O', 'E', 'G'],
        'caracteristicas_clave': ['forma_C', 'apertura_lateral', 'no_circulo_completo'],
        'penalizacion': 3.0,
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
        'penalizacion': 3.0,
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
        'penalizacion': 3.0,
        'augmentation_especial': {
            'rotation_range': 1.5,
            'brightness_range': (0.95, 1.05),
            'contrast_range': (0.98, 1.02),
            'no_flip': True,
            'no_zoom': True
        },
        'diferencias_criticas': {
            'vs_0': 'O letra más pequeño y cerrado',
            'vs_C': 'O más cerrado que C',
            'vs_Q': 'Q tiene índice apuntando abajo'
        }
    },
}

CONFIG_CONFUSIONES_EMERGENCIA = {
    'SOS': {
        'confusiones_principales': ['SOCORRO', 'AUXILIO', 'AYUDA'],
        'caracteristicas_clave': ['patron_repetitivo_marcado', 'expresion_facial_urgente'],
        'penalizacion': 4.0,
        'diferencias_criticas': {
            'vs_AYUDA': 'SOS suele repetirse/enfatizarse; AYUDA es una sola seña sin repetición marcada',
            'vs_SOCORRO': 'Verificar con un intérprete de LSP la forma exacta de cada una en la región del usuario',
        }
    },
    'EMERGENCIA': {
        'confusiones_principales': ['PELIGRO', 'ACCIDENTE'],
        'caracteristicas_clave': ['urgencia', 'movimiento_amplio'],
        'penalizacion': 4.0,
        'diferencias_criticas': {
            'vs_PELIGRO': 'EMERGENCIA implica necesidad de ayuda inmediata; PELIGRO advierte de un riesgo',
        }
    },
}

tipo_sena_config = {
    'ESTATICA': {
        'frames_entrenamiento': 8,
        'fps': 10,
        'augmentation': {
            'brightness_range': (0.8, 1.2),
            'rotation_range': 10,
            'flip_prob': 0.0
        }
    },
    'DINAMICA': {
        'frames_entrenamiento': 16,
        'fps': 15,
        'augmentation': {
            'brightness_range': (0.8, 1.2),
            'rotation_range': 10,
            'flip_prob': 0.0,
            'noise_level': 8,
            'frame_drop_prob': 0.2
        }
    },
    'EMERGENCIA': {
        'frames_entrenamiento': 20,
        'fps': 20,
        'augmentation': {
            'brightness_range': (0.7, 1.3),
            'rotation_range': 12,
            'flip_prob': 0.0,
            'noise_level': 10,
            'frame_drop_prob': 0.1
        }
    }
}

UMBRAL_MOVIMIENTO_PROMEDIO = 0.012
UMBRAL_MOVIMIENTO_PICO = 0.03
MIN_FRAMES_VALIDOS_PARA_ANALISIS = 3
PROPORCION_FRAMES_CON_MOVIMIENTO = 0.4
PROPORCION_MIN_PARA_PICO = 0.2


def _frame_valido(kp: Optional[np.ndarray]) -> bool:
    return kp is not None and not np.all(kp == 0)


def _desplazamiento_por_punto(anterior: np.ndarray, actual: np.ndarray) -> Optional[float]:
    pa = anterior.reshape(-1, 3)[:, :2]
    pb = actual.reshape(-1, 3)[:, :2]
    valido = ~np.all(pa == 0, axis=1) & ~np.all(pb == 0, axis=1)
    if not np.any(valido):
        return None
    return float(np.mean(np.linalg.norm(pb[valido] - pa[valido], axis=1)))


def detectar_tipo_por_movimiento(
    keypoints_secuencia: List[np.ndarray],
    umbral_promedio: float = UMBRAL_MOVIMIENTO_PROMEDIO,
    umbral_pico: float = UMBRAL_MOVIMIENTO_PICO,
) -> Tuple[str, Dict[str, Any]]:
    n_validos = sum(1 for kp in keypoints_secuencia if _frame_valido(kp))

    if n_validos < MIN_FRAMES_VALIDOS_PARA_ANALISIS:
        return 'ESTATICA', {'razon': 'frames_insuficientes', 'frames_validos': n_validos}

    desplazamientos = []
    for i in range(1, len(keypoints_secuencia)):
        a, b = keypoints_secuencia[i - 1], keypoints_secuencia[i]
        if _frame_valido(a) and _frame_valido(b):
            d = _desplazamiento_por_punto(a, b)
            if d is not None:
                desplazamientos.append(d)

    if not desplazamientos:
        return 'ESTATICA', {'razon': 'sin_desplazamientos'}

    movimiento_promedio = float(np.mean(desplazamientos))
    movimiento_pico = float(np.max(desplazamientos))
    proporcion_frames_movidos = sum(1 for d in desplazamientos if d > umbral_promedio) / len(desplazamientos)

    por_media = (movimiento_promedio > umbral_promedio
                 and proporcion_frames_movidos >= PROPORCION_FRAMES_CON_MOVIMIENTO)
    por_pico = (movimiento_pico > umbral_pico
                and proporcion_frames_movidos >= PROPORCION_MIN_PARA_PICO)
    es_dinamica = por_media or por_pico

    detalles = {
        'movimiento_promedio': round(movimiento_promedio, 5),
        'movimiento_pico': round(movimiento_pico, 5),
        'proporcion_frames_movidos': round(proporcion_frames_movidos, 3),
        'frames_analizados': n_validos,
        'pares_comparados': len(desplazamientos),
        'criterio': 'media' if por_media else ('pico' if por_pico else 'ninguno'),
    }

    return ('DINAMICA' if es_dinamica else 'ESTATICA'), detalles


def calcular_tipo_clase_desde_muestras(
    muestras_keypoints: List[List[np.ndarray]],
    umbral_proporcion_dinamica: float = 0.5,
) -> Tuple[str, Dict[str, Any]]:
    if not muestras_keypoints:
        return 'DINAMICA', {'razon': 'sin_muestras'}

    resultados = [detectar_tipo_por_movimiento(m) for m in muestras_keypoints]
    tipos = [r[0] for r in resultados]
    proporcion_dinamica = tipos.count('DINAMICA') / len(tipos)

    tipo_final = 'DINAMICA' if proporcion_dinamica >= umbral_proporcion_dinamica else 'ESTATICA'

    return tipo_final, {
        'total_muestras': len(tipos),
        'proporcion_dinamica': round(proporcion_dinamica, 3),
        'detalle_por_muestra': [r[1] for r in resultados],
    }


def es_sena_emergencia(nombre_sena: str) -> bool:
    return normalizar_nombre(nombre_sena) in SENAS_EMERGENCIA


def detectar_tipo_sena(nombre_sena: str) -> str:
    nombre_limpio = normalizar_nombre(nombre_sena)
    if not nombre_limpio:
        return 'DINAMICA'

    if nombre_limpio in SENAS_EMERGENCIA:
        return 'EMERGENCIA'

    if nombre_limpio in SENAS_VERIFICADAS_DINAMICAS:
        return 'DINAMICA'

    if nombre_limpio in SENAS_ESTATICAS:
        return 'ESTATICA'

    if len(nombre_limpio) == 1 and nombre_limpio.isalpha():
        return 'ESTATICA'

    if nombre_limpio.isdigit():
        return 'ESTATICA' if len(nombre_limpio) == 1 else 'DINAMICA'

    for patron in PATRONES_DINAMICOS:
        if re.match(patron, nombre_limpio):
            return 'DINAMICA'

    return 'DINAMICA'


def obtener_config_sena(nombre_sena: str) -> Dict[str, Any]:
    tipo_sena = detectar_tipo_sena(nombre_sena)
    config = CONFIG_SENAS[tipo_sena].copy()
    config['tipo_detectado'] = tipo_sena
    config['nombre_sena'] = nombre_sena

    sena_upper = _clave(nombre_sena)
    config['es_confusa'] = sena_upper in CONFUSIONES_COMUNES
    config['permite_flip'] = tipo_sena == 'ESTATICA' and sena_upper not in SENAS_NO_FLIP
    config['permite_zoom'] = True
    config['es_emergencia'] = tipo_sena == 'EMERGENCIA'

    config_confusion_numerica = obtener_config_confusion_numerica(nombre_sena)
    config_confusion_emergencia = CONFIG_CONFUSIONES_EMERGENCIA.get(sena_upper, {})

    if config_confusion_numerica:
        config['es_confusion_numerica'] = True
        config['confusiones_principales'] = config_confusion_numerica.get('confusiones_principales', [])
        config['caracteristicas_clave'] = config_confusion_numerica.get('caracteristicas_clave', [])
        config['penalizacion_especial'] = config_confusion_numerica.get('penalizacion', 3.0)
        config['diferencias_criticas'] = config_confusion_numerica.get('diferencias_criticas', {})

        augmentation_especial = config_confusion_numerica.get('augmentation_especial', {})
        config['augmentation_rotation'] = augmentation_especial.get('rotation_range', 2.0)
        config['augmentation_brightness'] = augmentation_especial.get('brightness_range', (0.95, 1.05))
        config['augmentation_contrast'] = augmentation_especial.get('contrast_range', (0.98, 1.02))
        config['permite_flip'] = not augmentation_especial.get('no_flip', True)
        config['permite_zoom'] = not augmentation_especial.get('no_zoom', True)
    elif config_confusion_emergencia:
        config['es_confusion_numerica'] = False
        config['confusiones_principales'] = config_confusion_emergencia.get('confusiones_principales', [])
        config['caracteristicas_clave'] = config_confusion_emergencia.get('caracteristicas_clave', [])
        config['penalizacion_especial'] = config_confusion_emergencia.get('penalizacion', 4.0)
        config['diferencias_criticas'] = config_confusion_emergencia.get('diferencias_criticas', {})
        config['permite_flip'] = False
        config['permite_zoom'] = False
    elif config['es_confusa']:
        config['augmentation_rotation'] = 5.0
        config['augmentation_brightness'] = (0.9, 1.1)
        config['augmentation_contrast'] = (0.95, 1.05)

    config['grupo_confusion'] = obtener_grupo_confusion(nombre_sena)

    return config


def obtener_confusiones_comunes(sena: str) -> List[str]:
    return CONFUSIONES_COMUNES.get(_clave(sena), [])


def es_confusion_comun(sena_esperada: str, sena_detectada: str) -> bool:
    if _clave(sena_esperada) == _clave(sena_detectada):
        return False

    confusiones = obtener_confusiones_comunes(sena_esperada)
    return _clave(sena_detectada) in [c.upper() for c in confusiones]


def obtener_config_confusion_numerica(sena: str) -> Dict[str, Any]:
    return CONFIG_CONFUSIONES_NUMERICAS.get(_clave(sena), {})


def es_confusion_numerica(sena_esperada: str, sena_detectada: str) -> bool:
    sena_esperada_upper = _clave(sena_esperada)
    sena_detectada_upper = _clave(sena_detectada)

    config_esperada = CONFIG_CONFUSIONES_NUMERICAS.get(sena_esperada_upper, {})
    if config_esperada:
        if sena_detectada_upper in config_esperada.get('confusiones_principales', []):
            return True

    config_detectada = CONFIG_CONFUSIONES_NUMERICAS.get(sena_detectada_upper, {})
    if config_detectada:
        if sena_esperada_upper in config_detectada.get('confusiones_principales', []):
            return True

    return False


def obtener_grupos_confusion(sena: str) -> List[Set[str]]:
    sena_upper = _clave(sena)
    return [g for g in GRUPOS_ALTA_CONFUSION if sena_upper in g]


def obtener_grupo_confusion(sena: str) -> Optional[Set[str]]:
    grupos = obtener_grupos_confusion(sena)
    return grupos[0] if grupos else None


def _grupos_compartidos(sena_a: str, sena_b: str) -> List[Set[str]]:
    b = _clave(sena_b)
    return [g for g in obtener_grupos_confusion(sena_a) if b in g]


def calcular_factor_confusion_mejorado(sena_esperada: str, sena_detectada: str) -> float:
    sena_esperada_upper = _clave(sena_esperada)
    sena_detectada_upper = _clave(sena_detectada)

    if sena_esperada_upper == sena_detectada_upper:
        return 1.0

    if es_sena_emergencia(sena_esperada) or es_sena_emergencia(sena_detectada):
        conf_emergencia = CONFIG_CONFUSIONES_EMERGENCIA.get(sena_esperada_upper, {})
        if sena_detectada_upper in conf_emergencia.get('confusiones_principales', []):
            return conf_emergencia.get('penalizacion', 4.0)
        conf_emergencia_inversa = CONFIG_CONFUSIONES_EMERGENCIA.get(sena_detectada_upper, {})
        if sena_esperada_upper in conf_emergencia_inversa.get('confusiones_principales', []):
            return conf_emergencia_inversa.get('penalizacion', 4.0)
        if es_confusion_comun(sena_esperada, sena_detectada):
            return 3.5

    if es_confusion_numerica(sena_esperada, sena_detectada):
        confusiones_criticas = [
            ('0', 'C'), ('0', 'O'), ('C', '0'), ('O', '0'),
            ('1', 'D'), ('1', 'I'), ('1', 'L'), ('D', '1'),
            ('1', 'B'), ('B', '1'),
            ('2', 'V'), ('2', 'U'), ('V', '2'), ('U', '2'),
            ('3', 'E'), ('3', 'M'), ('E', '3'), ('M', '3'),
        ]

        if (sena_esperada_upper, sena_detectada_upper) in confusiones_criticas:
            return 3.5

        return 3.0

    compartidos = _grupos_compartidos(sena_esperada, sena_detectada)
    if compartidos:
        if any(g == GRUPO_CRITICO_ALFABETO for g in compartidos):
            return 3.0
        return 2.5

    if es_confusion_comun(sena_esperada, sena_detectada):
        return 2.0

    return 1.3


def calcular_factor_confusion(sena_esperada: str, sena_detectada: str) -> float:
    return calcular_factor_confusion_mejorado(sena_esperada, sena_detectada)


def generar_alerta_emergencia(sena_detectada: str, confianza: float,
                              detecciones_consecutivas: int = 1) -> Dict[str, Any]:
    if not es_sena_emergencia(sena_detectada):
        return {
            'es_emergencia': False,
            'disparar_alerta': False,
            'motivo': 'La seña detectada no pertenece al vocabulario de emergencia.'
        }

    config = CONFIG_SENAS['EMERGENCIA']
    umbral = config['umbral_confianza']
    confirmaciones_requeridas = config['confirmaciones_consecutivas_requeridas']

    cumple_confianza = confianza >= umbral
    cumple_confirmacion = detecciones_consecutivas >= confirmaciones_requeridas

    return {
        'es_emergencia': True,
        'sena': sena_detectada,
        'confianza': confianza,
        'umbral_requerido': umbral,
        'detecciones_consecutivas': detecciones_consecutivas,
        'confirmaciones_requeridas': confirmaciones_requeridas,
        'disparar_alerta': cumple_confianza and cumple_confirmacion,
        'motivo': (
            'Alerta confirmada.' if (cumple_confianza and cumple_confirmacion) else
            'Confianza insuficiente, seguir observando.' if not cumple_confianza else
            'Confianza suficiente, esperando confirmación adicional para evitar falsa alarma.'
        )
    }


def obtener_prioridad_procesamiento(nombre_sena: str) -> str:
    tipo = detectar_tipo_sena(nombre_sena)
    return CONFIG_SENAS[tipo].get('prioridad', 'NORMAL')


def generar_guia_aprendizaje_emergencia() -> List[Dict[str, Any]]:
    guia = []
    for sena in sorted(SENAS_EMERGENCIA):
        sena_upper = sena.upper()
        confusiones = obtener_confusiones_comunes(sena)
        config_conf = CONFIG_CONFUSIONES_EMERGENCIA.get(sena_upper, {})
        diferencias = config_conf.get('diferencias_criticas', {})
        guia.append({
            'sena': sena,
            'categoria': 'EMERGENCIA',
            'prioridad': 'CRITICA',
            'posibles_confusiones': confusiones,
            'diferencias_clave': diferencias,
            'recomendacion': (
                f"Practica '{sena}' de forma clara y, si puedes, repítela dos veces "
                f"seguidas: el sistema espera confirmación repetida antes de avisar, "
                f"para no generar falsas alarmas. "
                + (f"No la confundas con: {', '.join(confusiones)}." if confusiones
                   else "No tiene confusiones registradas por ahora.")
            )
        })
    return guia


def es_categoria_emergencia(senas: List[str]) -> bool:
    if not senas:
        return False
    return any(es_sena_emergencia(s) for s in senas)


def validar_consistencia_categoria(senas: List[str]) -> Dict[str, Any]:
    tipos = [detectar_tipo_sena(sena) for sena in senas]
    conteo_tipos = {
        'ESTATICA': tipos.count('ESTATICA'),
        'DINAMICA': tipos.count('DINAMICA'),
        'EMERGENCIA': tipos.count('EMERGENCIA'),
    }

    total = len(tipos)
    proporcion_estaticas = conteo_tipos['ESTATICA'] / total if total > 0 else 0
    proporcion_dinamicas = conteo_tipos['DINAMICA'] / total if total > 0 else 0
    proporcion_emergencia = conteo_tipos['EMERGENCIA'] / total if total > 0 else 0

    tipo_mayoritario = max(conteo_tipos, key=conteo_tipos.get) if total > 0 else 'DINAMICA'

    senas_confusas = [s for s in senas if _clave(s) in CONFUSIONES_COMUNES]
    senas_confusion_numerica = []
    senas_grupo_critico = []
    senas_emergencia = [s for s in senas if es_sena_emergencia(s)]

    for sena in senas:
        config = obtener_config_confusion_numerica(sena)
        if config:
            senas_confusion_numerica.append(sena)

        if any(any(s in ('A', 'C', 'S', 'E', 'M') for s in g) for g in obtener_grupos_confusion(sena)):
            senas_grupo_critico.append(sena)

    tiene_confusiones = len(senas_confusas) > 0

    if len(senas_emergencia) > 0:
        nivel_dificultad = 'CRITICA_SEGURIDAD'
    elif len(senas_confusion_numerica) > 0:
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
        'proporcion_emergencia': round(proporcion_emergencia, 3),
        'tipo_mayoritario': tipo_mayoritario,
        'consistente': (proporcion_estaticas > 0.8 or proporcion_dinamicas > 0.8 or proporcion_emergencia > 0.8),
        'tiene_confusiones': tiene_confusiones,
        'senas_confusas': senas_confusas,
        'senas_confusion_numerica': senas_confusion_numerica,
        'senas_grupo_critico': senas_grupo_critico,
        'senas_emergencia': senas_emergencia,
        'nivel_dificultad': nivel_dificultad
    }


def es_categoria_alfabeto(senas: List[str]) -> bool:
    if not senas:
        return False

    letras_alfabeto = set('abcdefghijklmnopqrstuvwxyzñ')
    conteo_letras = sum(1 for sena in senas if normalizar_nombre(sena) in letras_alfabeto)

    return conteo_letras / len(senas) > 0.7


def es_categoria_numeros(senas: List[str]) -> bool:
    if not senas:
        return False

    numeros = set('0123456789')
    conteo_numeros = sum(1 for sena in senas if normalizar_nombre(sena) in numeros)

    return conteo_numeros / len(senas) > 0.7


def obtener_parametros_entrenamiento(senas: List[str]) -> Dict[str, Any]:
    analisis_consistencia = validar_consistencia_categoria(senas)
    es_alfabeto = es_categoria_alfabeto(senas)
    es_numeros = es_categoria_numeros(senas)
    es_emergencia = es_categoria_emergencia(senas)

    parametros = {
        'batch_size': 8,
        'num_frames': 16,
        'learning_rate': 0.0008,
        'epochs': 120,
        'weight_decay': 1e-4,
        'patience': 25
    }

    nivel_dificultad = analisis_consistencia['nivel_dificultad']

    if nivel_dificultad == 'CRITICA_SEGURIDAD':
        parametros['learning_rate'] = 0.0005
        parametros['epochs'] = 200
        parametros['patience'] = 40
        parametros['confusion_penalty'] = 3.5
        parametros['focal_gamma'] = 1.5
        parametros['label_smoothing'] = 0.05
        parametros['oversampling'] = '4x para señas de emergencia'
        parametros['prioridad_seguridad'] = True

    elif nivel_dificultad == 'MUY_ALTO':
        parametros['learning_rate'] = 0.0005
        parametros['epochs'] = 180
        parametros['patience'] = 35
        parametros['confusion_penalty'] = 3.0
        parametros['focal_gamma'] = 1.5
        parametros['label_smoothing'] = 0.05
        parametros['oversampling'] = '2.5x para numéricas'

    elif nivel_dificultad == 'ALTO':
        parametros['learning_rate'] = 0.0007
        parametros['epochs'] = 160
        parametros['patience'] = 30
        parametros['confusion_penalty'] = 2.5
        parametros['focal_gamma'] = 1.5
        parametros['label_smoothing'] = 0.05
        parametros['oversampling'] = '2x para críticas'

    elif nivel_dificultad == 'MEDIO':
        parametros['learning_rate'] = 0.0008
        parametros['epochs'] = 140
        parametros['patience'] = 28
        parametros['confusion_penalty'] = 2.0
        parametros['focal_gamma'] = 1.5
        parametros['label_smoothing'] = 0.05
        parametros['oversampling'] = '1.5x para confusas'

    else:
        parametros['confusion_penalty'] = 1.5
        parametros['focal_gamma'] = 1.5
        parametros['label_smoothing'] = 0.05
        parametros['oversampling'] = '1x proporcional'

    conteo = analisis_consistencia['conteo_tipos']
    frames_por_tipo = {
        t: CONFIG_SENAS[t]['num_frames_recomendado'] for t, n in conteo.items() if n > 0
    }
    parametros['num_frames'] = max(frames_por_tipo.values()) if frames_por_tipo else 16
    solo_estaticas = bool(frames_por_tipo) and set(frames_por_tipo) == {'ESTATICA'}
    parametros['batch_size'] = 16 if solo_estaticas else 8

    if es_alfabeto:
        parametros['epochs'] = max(parametros['epochs'], 150)
        parametros['learning_rate'] = min(parametros['learning_rate'], 0.0008)

    if es_numeros:
        parametros['epochs'] = max(parametros['epochs'], 180)
        parametros['learning_rate'] = min(parametros['learning_rate'], 0.0006)
        parametros['confusion_penalty'] = max(parametros['confusion_penalty'], 3.0)

    if es_emergencia:
        parametros['epochs'] = max(parametros['epochs'], 200)
        parametros['learning_rate'] = min(parametros['learning_rate'], 0.0005)
        parametros['confusion_penalty'] = max(parametros['confusion_penalty'], 3.5)
        parametros['prioridad_seguridad'] = True

    return {
        **parametros,
        'num_frames_por_tipo': frames_por_tipo,
        'analisis_consistencia': analisis_consistencia,
        'es_categoria_alfabeto': es_alfabeto,
        'es_categoria_numeros': es_numeros,
        'es_categoria_emergencia': es_emergencia,
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
            recomendaciones.append("Categoría estática - usar 8 frames y procesamiento optimizado")
        elif consistencia['tipo_mayoritario'] == 'EMERGENCIA':
            recomendaciones.append("Categoría de EMERGENCIA - usar 20 frames y procesamiento prioritario")
        else:
            recomendaciones.append("Categoría dinámica - usar 16 frames y captura de movimiento")
    else:
        recomendaciones.append("Categoría mixta - usar modelo adaptativo con doble backbone")

    conteo = consistencia.get('conteo_tipos', {})
    if conteo.get('ESTATICA', 0) and (conteo.get('DINAMICA', 0) or conteo.get('EMERGENCIA', 0)):
        recomendaciones.append(
            "Hay señas estáticas y con movimiento mezcladas: se usa el nº de frames del tipo más "
            "exigente para todas; las estáticas se pueden muestrear del centro de la secuencia."
        )

    if es_alfabeto:
        recomendaciones.append("Categoría ALFABETO - entrenar con LR bajo (0.0008) y 150+ épocas")

    if es_numeros:
        recomendaciones.append("Categoría NÚMEROS - máxima precaución con confusiones numéricas")
        recomendaciones.append("   -> Usar augmentation ultra-conservador")
        recomendaciones.append("   -> NO aplicar flip horizontal")
        recomendaciones.append("   -> Oversampling 2.5x")

    if consistencia.get('senas_emergencia'):
        recomendaciones.append(
            f"SEÑAS DE EMERGENCIA presentes ({len(consistencia['senas_emergencia'])}): "
            "un falso negativo aquí puede impedir que alguien pida ayuda real."
        )
        recomendaciones.append("   -> Umbral de confianza más bajo (0.45) + confirmación temporal (2 detecciones)")
        recomendaciones.append("   -> NO aplicar flip horizontal, orientación importa")
        recomendaciones.append("   -> Oversampling mínimo 4x")
        recomendaciones.append("   -> Probar el modelo con usuarios reales antes de desplegarlo")
        recomendaciones.append("   -> Ofrecer también un botón físico/manual de SOS como respaldo")

    if nivel_dificultad == 'CRITICA_SEGURIDAD':
        recomendaciones.append("PRIORIDAD DE SEGURIDAD - por encima de cualquier otra dificultad")
        recomendaciones.append("   -> Learning rate: 0.0005")
        recomendaciones.append("   -> Epochs: 200+")
        recomendaciones.append("   -> Penalización: 3.5x")

    elif nivel_dificultad == 'MUY_ALTO':
        recomendaciones.append("DIFICULTAD MUY ALTA - Confusiones numéricas detectadas")
        recomendaciones.append("   -> Learning rate: 0.0005")
        recomendaciones.append("   -> Epochs: 180+")
        recomendaciones.append("   -> Penalización: 3.0x")
        recomendaciones.append("   -> Augmentation: ±5% brightness, ±2° rotation")

    elif nivel_dificultad == 'ALTO':
        recomendaciones.append("DIFICULTAD ALTA - Grupo crítico detectado")
        recomendaciones.append("   -> Learning rate: 0.0007")
        recomendaciones.append("   -> Epochs: 160+")
        recomendaciones.append("   -> Penalización: 2.5x")

    elif nivel_dificultad == 'MEDIO':
        recomendaciones.append("DIFICULTAD MEDIA - Señas confusas presentes")
        recomendaciones.append("   -> Learning rate: 0.0008")
        recomendaciones.append("   -> Epochs: 140")
        recomendaciones.append("   -> Penalización: 2.0x")

    else:
        recomendaciones.append("DIFICULTAD BAJA - Sin confusiones significativas")
        recomendaciones.append("   -> Parámetros estándar")

    if consistencia.get('senas_confusion_numerica'):
        recomendaciones.append(f"CRÍTICO: {len(consistencia['senas_confusion_numerica'])} señas con confusión numérica:")
        for sena in consistencia['senas_confusion_numerica'][:5]:
            config = obtener_config_confusion_numerica(sena)
            confusiones = config.get('confusiones_principales', [])
            recomendaciones.append(f"   - {sena} <-> {', '.join(confusiones)}")

    if consistencia.get('senas_grupo_critico'):
        recomendaciones.append(f"ATENCIÓN: {len(consistencia['senas_grupo_critico'])} señas en grupo crítico:")
        recomendaciones.append(f"   {', '.join(consistencia['senas_grupo_critico'][:10])}")

    if consistencia['total_senas'] < 10:
        recomendaciones.append("SUGERENCIA: Pocas señas para entrenar")
        recomendaciones.append("   -> Considerar agregar más clases para mejor generalización")

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
                        'es_emergencia': es_sena_emergencia(clase1) or es_sena_emergencia(clase2),
                        'mismo_grupo': bool(_grupos_compartidos(clase1, clase2))
                    }
                    pares_confusion.append(par)

                    if factor >= 3.0:
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
        conocidas_upper = [c.upper() for c in confusiones_conocidas]

        for otra_clase in clases:
            if _clave(otra_clase) != _clave(clase) and _clave(otra_clase) in conocidas_upper:
                confusiones_posibles.append(otra_clase)

        if confusiones_posibles:
            matriz[clase] = confusiones_posibles

    return matriz


def exportar_configuracion_confusiones(clases: List[str], ruta_salida: Optional[str] = None):
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
    ajustes = {}

    for sena_esperada, sena_detectada in senas_calibracion.items():
        if _clave(sena_esperada) == _clave(sena_detectada):
            continue

        tipo_esperado = detectar_tipo_sena(sena_esperada)
        tipo_detectado = detectar_tipo_sena(sena_detectada)
        factor_confusion = calcular_factor_confusion_mejorado(sena_esperada, sena_detectada)
        comun = es_confusion_comun(sena_esperada, sena_detectada)

        base = {
            'sena_confundida': sena_detectada,
            'confusion_comun': comun,
            'factor_penalizacion': factor_confusion,
            'es_confusion_numerica': es_confusion_numerica(sena_esperada, sena_detectada),
            'es_emergencia': es_sena_emergencia(sena_esperada),
        }

        if tipo_esperado != tipo_detectado:
            ajustes[sena_esperada] = {
                **base,
                'tipo_corregido': tipo_esperado,
                'confianza_ajustada': 0.7,
                'factor_compensacion': 1.2,
            }
        elif comun:
            ajustes[sena_esperada] = base
        else:
            ajustes[sena_esperada] = {**base, 'error_no_catalogado': True}

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
    confusiones_emergencia = 0

    for sena, info_error in errores.items():
        sena_confundida = info_error.get('sena_confundida', 'desconocida')
        if info_error.get('es_emergencia'):
            confusiones_emergencia += 1
            recomendaciones.append(
                f"CRÍTICO DE SEGURIDAD: '{sena}' se confundió con '{sena_confundida}'. "
                "Practica esta seña de emergencia con especial cuidado y, si es posible, "
                "pide a un intérprete de LSP que revise tu ejecución."
            )
        elif info_error.get('es_confusion_numerica', False):
            confusiones_numericas += 1
            config = obtener_config_confusion_numerica(sena)
            diferencias = config.get('diferencias_criticas', {}) if config else {}
            clave_diff = f"vs_{_clave(sena_confundida)}"
            if clave_diff in diferencias:
                recomendaciones.append(
                    f"CRÍTICO '{sena}' <-> '{sena_confundida}': {diferencias[clave_diff]}"
                )
            else:
                recomendaciones.append(
                    f"Confusión numérica: '{sena}' se confunde con '{sena_confundida}'"
                )
        elif info_error.get('confusion_comun'):
            confusiones_comunes += 1
            recomendaciones.append(
                f"Confusión común: '{sena}' <-> '{sena_confundida}'. "
                f"Practica las diferencias clave."
            )
        elif info_error.get('tipo_corregido'):
            tipo_esperado = info_error['tipo_corregido']
            recomendaciones.append(
                f"Seña '{sena}' es {tipo_esperado.lower()}. "
                f"Ejecuta la seña {'sin movimiento' if tipo_esperado == 'ESTATICA' else 'con movimiento fluido'}."
            )
        else:
            recomendaciones.append(
                f"'{sena}' se confundió con '{sena_confundida}' (par no catalogado). "
                "Revisa si conviene añadirlo a CONFUSIONES_COMUNES."
            )

    if confusiones_emergencia > 0:
        recomendaciones.append(
            f"{confusiones_emergencia} confusiones en señas de EMERGENCIA. "
            "Esto tiene la máxima prioridad de todas: practica estas señas primero."
        )

    if confusiones_numericas > 0:
        recomendaciones.append(
            f"{confusiones_numericas} confusiones NUMÉRICAS críticas detectadas. "
            "Estas son las más difíciles. Practica con especial cuidado."
        )

    if confusiones_comunes > 3:
        recomendaciones.append(
            f"{confusiones_comunes} confusiones comunes detectadas. "
            "Considera mejorar iluminación y ángulo de cámara."
        )

    if len(errores) > 5:
        recomendaciones.append(
            "SUGERENCIA: Muchos errores detectados. Recomendaciones:"
            "\n   - Verifica la iluminación (luz frontal uniforme)"
            "\n   - Ajusta el ángulo de la cámara"
            "\n   - Mantén la mano a distancia adecuada"
            "\n   - Practica las señas básicas primero"
        )

    if not recomendaciones:
        recomendaciones.append("Calibración exitosa. No se detectaron errores significativos.")

    return recomendaciones


def obtener_estadisticas_confusiones() -> Dict[str, Any]:
    total_senas = len(CONFUSIONES_COMUNES)
    total_confusiones = sum(len(confusiones) for confusiones in CONFUSIONES_COMUNES.values())

    confusiones_numericas = len(CONFIG_CONFUSIONES_NUMERICAS)
    confusiones_emergencia = len(CONFIG_CONFUSIONES_EMERGENCIA)
    total_grupos = len(GRUPOS_ALTA_CONFUSION)
    senas_sin_flip = len(SENAS_NO_FLIP)
    total_senas_emergencia = len(SENAS_EMERGENCIA)

    return {
        'total_senas_confusas': total_senas,
        'total_pares_confusion': total_confusiones,
        'promedio_confusiones_por_sena': total_confusiones / total_senas if total_senas > 0 else 0,
        'confusiones_numericas_config': confusiones_numericas,
        'confusiones_emergencia_config': confusiones_emergencia,
        'total_grupos_alta_confusion': total_grupos,
        'senas_sin_flip': senas_sin_flip,
        'total_senas_emergencia': total_senas_emergencia,
        'top_senas_confusas': sorted(
            [(k, len(v)) for k, v in CONFUSIONES_COMUNES.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }


def verificar_clases_referenciadas() -> Dict[str, Any]:
    clases_validas = set()
    clases_validas.update(c.upper() for c in 'abcdefghijklmnopqrstuvwxyzñ')
    clases_validas.update(str(d) for d in range(10))
    clases_validas.update(s.upper() for s in SENAS_ESTATICAS)
    clases_validas.update(s.upper() for s in SENAS_VERIFICADAS_DINAMICAS)
    clases_validas.update(s.upper() for s in SENAS_EMERGENCIA)

    referencias_invalidas = []
    for sena, confusiones in CONFUSIONES_COMUNES.items():
        if sena not in clases_validas:
            referencias_invalidas.append(f"Clave '{sena}' no corresponde a ninguna clase definida")
        for c in confusiones:
            if c not in clases_validas:
                referencias_invalidas.append(f"'{sena}' referencia a '{c}', que no es una clase definida")

    return {
        'valido': len(referencias_invalidas) == 0,
        'referencias_invalidas': referencias_invalidas,
        'total_problemas': len(referencias_invalidas)
    }


def validar_integridad_configuracion() -> Dict[str, Any]:
    errores = []
    advertencias = []

    for sena, confusiones in CONFUSIONES_COMUNES.items():
        for conf in confusiones:
            if conf in CONFUSIONES_COMUNES:
                if sena not in CONFUSIONES_COMUNES[conf]:
                    advertencias.append(
                        f"Confusión no bidireccional: {sena} -> {conf}, pero {conf} no lista a {sena}"
                    )

    for sena in CONFIG_CONFUSIONES_NUMERICAS.keys():
        if sena not in CONFUSIONES_COMUNES:
            errores.append(
                f"Seña '{sena}' en CONFIG_CONFUSIONES_NUMERICAS pero no en CONFUSIONES_COMUNES"
            )

    for sena in CONFIG_CONFUSIONES_EMERGENCIA.keys():
        if sena not in CONFUSIONES_COMUNES:
            errores.append(
                f"Seña '{sena}' en CONFIG_CONFUSIONES_EMERGENCIA pero no en CONFUSIONES_COMUNES"
            )

    for i, grupo in enumerate(GRUPOS_ALTA_CONFUSION):
        for sena in grupo:
            if sena not in CONFUSIONES_COMUNES:
                advertencias.append(
                    f"Seña '{sena}' en GRUPOS_ALTA_CONFUSION[{i}] pero no en CONFUSIONES_COMUNES"
                )

    referencia_check = verificar_clases_referenciadas()
    if not referencia_check['valido']:
        errores.extend(referencia_check['referencias_invalidas'])

    for sena in SENAS_EMERGENCIA:
        if sena.upper() not in CONFUSIONES_COMUNES:
            advertencias.append(
                f"La seña de emergencia '{sena}' no tiene ninguna confusión registrada en CONFUSIONES_COMUNES"
            )

    for sena in SENAS_VERIFICADAS_DINAMICAS:
        if sena.upper() not in CONFUSIONES_COMUNES:
            advertencias.append(
                f"La seña dinámica '{sena}' no tiene ninguna confusión registrada en CONFUSIONES_COMUNES"
            )

    return {
        'valido': len(errores) == 0,
        'errores': errores,
        'advertencias': advertencias,
        'total_errores': len(errores),
        'total_advertencias': len(advertencias)
    }