from .api_peru import ServicioApiPeru, servicio_api_peru
from .archivos import ArchivoService, archivo_service
from .estadisticas_servicio import EstadisticasServicio

from .reconocimiento_ia import ReconocimientoIA
try:
    from .entrenamiento_modelo import EntrenamientoModeloService
except ImportError:
    EntrenamientoModeloService = None

__all__ = [
    "ServicioApiPeru",
    "servicio_api_peru",
    "ArchivoService", 
    "archivo_service",
    "ReconocimientoIA",
    "EntrenamientoModeloService",
    "EstadisticasServicio",
    "DatasetVideoService"

]