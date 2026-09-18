from typing import List, Tuple

CATALOGO_PERMISOS: List[Tuple[str, str, str]] = [
    # ---- Usuario: su propio perfil / progreso -----------------------------
    ("usuarios.ver_perfil", "usuarios", "Ver perfil propio"),
    ("usuarios.editar_perfil", "usuarios", "Editar perfil propio"),
    ("usuarios.ver_progreso", "usuarios", "Ver progreso propio (resumen)"),

    # ---- Progreso detallado por lección/clase ------------------------------
    ("progreso.ver", "progreso", "Ver progreso detallado propio"),
    ("progreso.registrar", "progreso", "Registrar avance de progreso propio"),

    # ---- Prácticas ----------------------------------------------------------
    ("practicas.registrar", "practicas", "Registrar una práctica propia"),
    ("practicas.ver_historial", "practicas", "Ver historial propio de prácticas"),

    # ---- Tiempo de estudio ----------------------------------------------------
    ("estudio.registrar", "estudio", "Registrar sesiones de estudio propias"),
    ("estudio.ver", "estudio", "Ver estadísticas propias de tiempo de estudio"),

    # ---- Administración de usuarios (panel admin) --------------------------
    ("admin.usuarios.listar", "admin_usuarios", "Listar todos los usuarios"),
    ("admin.usuarios.ver", "admin_usuarios", "Ver detalle de un usuario"),
    ("admin.usuarios.editar", "admin_usuarios", "Editar datos de un usuario"),
    ("admin.usuarios.eliminar", "admin_usuarios", "Eliminar un usuario"),
    ("admin.usuarios.asignar_rol", "admin_usuarios", "Asignar rol a un usuario"),
    ("admin.usuarios.cambiar_estado", "admin_usuarios", "Activar o desactivar usuario"),
    ("admin.usuarios.ver_estadisticas", "admin_usuarios", "Ver estadísticas/progreso de un usuario"),

    ("admin.menu.gestionar", "admin_menu", "Gestionar el menú y vistas del sistema"),
    ("admin.roles.gestionar", "admin_roles", "Crear, editar y asignar permisos a roles"),

    # ---- Categorías -----------------------------------------------------------
    ("categorias.ver", "categorias", "Ver categorías"),
    ("categorias.crear", "categorias", "Crear categorías"),
    ("categorias.editar", "categorias", "Editar categorías"),
    ("categorias.eliminar", "categorias", "Eliminar categorías"),
    ("categorias.gestionar_senas", "categorias", "Gestionar señas de una categoría"),
    ("admin.categorias.gestionar", "admin_categorias", "Acceder al panel admin de gestión de categorías"),

    # ---- Lecciones / Clases -----------------------------------------------------
    ("lecciones.ver", "lecciones", "Ver lecciones"),
    ("lecciones.crear", "lecciones", "Crear lecciones"),
    ("lecciones.editar", "lecciones", "Editar lecciones (incluye subir contenido)"),
    ("lecciones.eliminar", "lecciones", "Eliminar lecciones"),
    ("admin.lecciones.gestionar", "admin_lecciones", "Acceder al panel admin de gestión de lecciones"),

    ("clases.ver", "clases", "Ver clases"),
    ("clases.crear", "clases", "Crear clases"),
    ("clases.editar", "clases", "Editar clases"),
    ("clases.eliminar", "clases", "Eliminar clases"),

    # ---- Exámenes -----------------------------------------------------------------
    ("examenes.ver", "examenes", "Ver y presentar exámenes"),
    ("admin.examenes.gestionar", "admin_examenes", "Crear, editar y eliminar exámenes y preguntas"),

    # ---- Dataset / Modelos IA / Captura ---------------------------------------------
    ("dataset.ver", "dataset", "Ver dataset y categorías de dataset"),
    ("dataset.gestionar", "dataset", "Gestionar (subir/aprobar/eliminar) dataset y videos"),
    ("dataset.categorias.gestionar", "dataset", "Crear/editar categorías del dataset"),
    ("dataset.videos.gestionar", "dataset", "Aprobar, rechazar o eliminar videos del dataset"),
    ("modelos.gestionar", "modelos", "Entrenar, activar, desactivar y eliminar modelos"),
    ("captura.gestionar", "captura", "Captura de video para entrenamiento"),

    # ---- Panel / Reportes / Sistema ----------------------------------------------------
    ("admin.dashboard.ver", "admin_dashboard", "Ver panel de administración"),
    ("admin.reportes.ver", "admin_reportes", "Ver reportes de uso"),
    ("admin.archivos.limpiar", "admin_sistema", "Limpiar archivos temporales del servidor"),
    ("estadisticas.ver", "estadisticas", "Ver estadísticas generales del sistema"),
    ("estadisticas.rango", "estadisticas", "Ver estadísticas filtradas por rango de fechas"),

    # ---- Tipos de categoría -----------------------------------------------------------
    ("tipos_categoria.gestionar", "tipos_categoria", "Gestionar tipos de categoría"),

    # ---- Traductor / Reconocimiento -----------------------------------------------------
    ("traductor.usar", "traductor", "Usar el traductor de señas"),
    ("reconocimiento.usar", "reconocimiento", "Usar reconocimiento en video"),
]


def codigos_validos() -> set:
    return {codigo for codigo, _, _ in CATALOGO_PERMISOS}