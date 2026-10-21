from typing import List, Tuple

CATALOGO_PERMISOS: List[Tuple[str, str, str]] = [
    ("admin.usuarios.listar", "admin_usuarios", "Listar todos los usuarios"),
    ("admin.usuarios.ver", "admin_usuarios", "Ver detalle de un usuario"),
    ("admin.usuarios.editar", "admin_usuarios", "Editar datos de un usuario"),
    ("admin.usuarios.eliminar", "admin_usuarios", "Eliminar un usuario"),
    ("admin.usuarios.asignar_rol", "admin_usuarios", "Asignar rol a un usuario"),
    ("admin.usuarios.cambiar_estado", "admin_usuarios", "Activar o desactivar usuario"),
    ("admin.usuarios.ver_estadisticas", "admin_usuarios", "Ver estadísticas/progreso de un usuario"),

    ("admin.dashboard.ver", "admin_dashboard", "Ver panel de administración"),
    ("admin.menu.gestionar", "admin_menu", "Gestionar el menú y vistas del sistema"),
    ("admin.roles.gestionar", "admin_roles", "Crear, editar y asignar permisos a roles"),
    ("admin.reportes.ver", "admin_reportes", "Ver reportes de uso"),
    ("admin.archivos.limpiar", "admin_sistema", "Limpiar archivos temporales del servidor"),

    ("admin.categorias.gestionar", "admin_categorias", "Acceder al panel admin de gestión de categorías"),
    ("admin.lecciones.gestionar", "admin_lecciones", "Acceder al panel admin de gestión de lecciones"),
    ("admin.examenes.gestionar", "admin_examenes", "Crear, editar y eliminar exámenes y preguntas"),
    ("admin.tickets.gestionar", "admin_tickets", "Listar, asignar, responder y cambiar el estado de los tickets de soporte"),

    ("lecciones.ver", "lecciones", "Ver lecciones y estudiar"),
    ("practicas.registrar", "lecciones", "Practicar y registrar intentos en las clases"),
    ("examenes.ver", "examenes", "Ver y presentar exámenes"),
    ("progreso.ver", "progreso", "Ver el progreso propio"),

    ("tickets.crear", "tickets", "Crear tickets de soporte y comentar los propios"),
    ("tickets.ver_propios", "tickets", "Ver los tickets de soporte propios"),

    ("categorias.ver", "categorias", "Ver categorías"),
    ("categorias.gestionar_senas", "categorias", "Gestionar señas de una categoría"),

    ("clases.ver", "clases", "Ver clases"),
    ("clases.crear", "clases", "Crear clases"),
    ("clases.editar", "clases", "Editar clases"),
    ("clases.eliminar", "clases", "Eliminar clases"),

    ("dataset.ver", "dataset", "Ver dataset y categorías de dataset"),
    ("dataset.categorias.gestionar", "dataset", "Crear/editar categorías del dataset"),
    ("dataset.videos.gestionar", "dataset", "Aprobar, rechazar o eliminar videos del dataset"),

    ("modelos.ver", "modelos", "Ver información de modelos IA"),
    ("modelos.gestionar", "modelos", "Entrenar, activar, desactivar y eliminar modelos"),
    ("captura.gestionar", "captura", "Captura de video para entrenamiento"),

    ("tipos_categoria.gestionar", "tipos_categoria", "Gestionar tipos de categoría"),

    ("traductor.usar", "traductor", "Usar el traductor de señas"),
]


def codigos_validos() -> set:
    return {codigo for codigo, _, _ in CATALOGO_PERMISOS}
