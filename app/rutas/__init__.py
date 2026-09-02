# app/rutas/__init__.py
from . import autenticacion
from . import rutas_captura
from . import usuarios
from . import categorias
from . import lecciones
from . import clases
from . import practicas
from . import progreso
from . import examenes
from . import dataset
from . import traductor
from . import admin
from . import examenes_admin
from . import estudio
from . import estadisticas_rutas


__all__ = [
    'autenticacion',
    'rutas_captura', 
    'usuarios',
    'categorias',
    'lecciones',
    'clases',
    'practicas',
    'progreso',
    'examenes',
    'dataset',
    'traductor',
    'admin',
    'examenes_admin',
    'estudio',
    'estadisticas_rutas',
    'reconocimiento_video'
]