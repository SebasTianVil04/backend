from .api_peru import ServicioApiPeru, servicio_api_peru
from .archivos import ArchivoService, archivo_service
from .estadisticas_servicio import EstadisticasServicio
from .dataset_service import dataset_service

try:
    from .servicio_entrenamiento import servicio_entrenamiento
except ImportError:
    servicio_entrenamiento = None

try:
    from .servicio_reconocimiento import ServicioReconocimiento
except ImportError:
    ServicioReconocimiento = None

try:
    from .gestor_reconocimiento import (
        obtener_reconocedor,
        obtener_reconocedor_cacheado,
        determinar_tipo_sena,
        procesar_frame_base64,
        extraer_keypoints_frame,
        limpiar_cache,
        cerrar_mediapipe_hands,
        info_cache,
    )
except ImportError:
    obtener_reconocedor = None
    obtener_reconocedor_cacheado = None
    determinar_tipo_sena = None
    procesar_frame_base64 = None
    extraer_keypoints_frame = None
    limpiar_cache = None
    cerrar_mediapipe_hands = None
    info_cache = None

__all__ = [
    "ServicioApiPeru",
    "servicio_api_peru",
    "ArchivoService",
    "archivo_service",
    "EstadisticasServicio",
    "dataset_service",
    "servicio_entrenamiento",
    "Servicio_reconocimiento",
    "obtener_reconocedor",
    "obtener_reconocedor_cacheado",
    "determinar_tipo_sena",
    "procesar_frame_base64",
    "extraer_keypoints_frame",
    "limpiar_cache",
    "cerrar_mediapipe_hands",
    "info_cache",
]