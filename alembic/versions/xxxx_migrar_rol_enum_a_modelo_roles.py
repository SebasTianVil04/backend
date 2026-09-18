from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6_roles"
down_revision = "ecef25f5af56"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("codigo", sa.String(length=50), nullable=False, unique=True, index=True),
        sa.Column("nombre", sa.String(length=100), nullable=False),
        sa.Column("descripcion", sa.Text(), nullable=True),
        sa.Column("es_sistema", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("activo", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("creado_en", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("actualizado_en", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "permisos",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("codigo", sa.String(length=100), nullable=False, unique=True, index=True),
        sa.Column("modulo", sa.String(length=50), nullable=False, index=True),
        sa.Column("descripcion", sa.String(length=255), nullable=False),
        sa.Column("creado_en", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "rol_permisos",
        sa.Column("rol_id", sa.Integer(), sa.ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("permiso_id", sa.Integer(), sa.ForeignKey("permisos.id", ondelete="CASCADE"), primary_key=True),
    )

    op.add_column("usuarios", sa.Column("rol_id", sa.Integer(), nullable=True))

    conexion = op.get_bind()

    conexion.execute(sa.text(
        "INSERT INTO roles (codigo, nombre, es_sistema, activo) VALUES "
        "('admin', 'Administrador', true, true), "
        "('usuario', 'Usuario', true, true)"
    ))

    conexion.execute(sa.text(
        "UPDATE usuarios SET rol_id = (SELECT id FROM roles WHERE roles.codigo = usuarios.rol)"
    ))

    op.alter_column("usuarios", "rol_id", nullable=False)

    op.create_foreign_key(
        "fk_usuarios_rol_id", "usuarios", "roles", ["rol_id"], ["id"]
    )

    op.create_index("ix_usuarios_rol_id", "usuarios", ["rol_id"])

    op.drop_column("usuarios", "rol")

    catalogo_permisos = [
        ("usuarios.ver_perfil", "usuarios", "Ver perfil propio"),
        ("usuarios.editar_perfil", "usuarios", "Editar perfil propio"),
        ("usuarios.ver_progreso", "usuarios", "Ver progreso propio"),
        ("admin.usuarios.listar", "admin_usuarios", "Listar todos los usuarios"),
        ("admin.usuarios.ver", "admin_usuarios", "Ver detalle de un usuario"),
        ("admin.usuarios.editar", "admin_usuarios", "Editar datos de un usuario"),
        ("admin.usuarios.eliminar", "admin_usuarios", "Eliminar un usuario"),
        ("admin.usuarios.asignar_rol", "admin_usuarios", "Asignar rol a un usuario"),
        ("admin.usuarios.cambiar_estado", "admin_usuarios", "Activar o desactivar usuario"),
        ("admin.usuarios.ver_estadisticas", "admin_usuarios", "Ver estadísticas de un usuario"),
        ("admin.roles.gestionar", "admin_roles", "Crear, editar y asignar permisos a roles"),
        ("categorias.ver", "categorias", "Ver categorías"),
        ("categorias.crear", "categorias", "Crear categorías"),
        ("categorias.editar", "categorias", "Editar categorías"),
        ("categorias.eliminar", "categorias", "Eliminar categorías"),
        ("categorias.gestionar_senas", "categorias", "Gestionar señas de una categoría"),
        ("lecciones.ver", "lecciones", "Ver lecciones"),
        ("lecciones.crear", "lecciones", "Crear lecciones"),
        ("lecciones.editar", "lecciones", "Editar lecciones"),
        ("lecciones.eliminar", "lecciones", "Eliminar lecciones"),
        ("clases.ver", "clases", "Ver clases"),
        ("clases.crear", "clases", "Crear clases"),
        ("clases.editar", "clases", "Editar clases"),
        ("clases.eliminar", "clases", "Eliminar clases"),
        ("examenes.ver", "examenes", "Ver y presentar exámenes"),
        ("admin.examenes.gestionar", "admin_examenes", "Crear, editar y eliminar exámenes y preguntas"),
        ("dataset.gestionar", "dataset", "Gestionar dataset de entrenamiento y videos"),
        ("modelos.gestionar", "modelos", "Entrenar, activar y eliminar modelos"),
        ("captura.gestionar", "captura", "Captura de video para entrenamiento"),
        ("admin.dashboard.ver", "admin_dashboard", "Ver panel de administración"),
        ("admin.reportes.ver", "admin_reportes", "Ver reportes de uso"),
        ("admin.archivos.limpiar", "admin_sistema", "Limpiar archivos temporales del servidor"),
        ("tipos_categoria.gestionar", "tipos_categoria", "Gestionar tipos de categoría"),
        ("traductor.usar", "traductor", "Usar el traductor de señas"),
        ("reconocimiento.usar", "reconocimiento", "Usar reconocimiento en video"),
    ]

    for codigo, modulo, descripcion in catalogo_permisos:
        conexion.execute(
            sa.text(
                "INSERT INTO permisos (codigo, modulo, descripcion) VALUES (:codigo, :modulo, :descripcion)"
            ),
            {"codigo": codigo, "modulo": modulo, "descripcion": descripcion}
        )

    conexion.execute(sa.text(
        "INSERT INTO rol_permisos (rol_id, permiso_id) "
        "SELECT (SELECT id FROM roles WHERE codigo = 'admin'), id FROM permisos"
    ))

    codigos_usuario = [
        "usuarios.ver_perfil", "usuarios.editar_perfil", "usuarios.ver_progreso",
        "categorias.ver", "lecciones.ver", "clases.ver", "examenes.ver",
        "traductor.usar", "reconocimiento.usar",
    ]

    for codigo in codigos_usuario:
        conexion.execute(
            sa.text(
                "INSERT INTO rol_permisos (rol_id, permiso_id) "
                "SELECT (SELECT id FROM roles WHERE codigo = 'usuario'), id FROM permisos WHERE codigo = :codigo"
            ),
            {"codigo": codigo}
        )


def downgrade():
    op.add_column("usuarios", sa.Column("rol", sa.String(length=20), nullable=True))

    conexion = op.get_bind()
    conexion.execute(sa.text(
        "UPDATE usuarios SET rol = (SELECT codigo FROM roles WHERE roles.id = usuarios.rol_id)"
    ))

    op.alter_column("usuarios", "rol", nullable=False)
    op.drop_index("ix_usuarios_rol_id", table_name="usuarios")
    op.drop_constraint("fk_usuarios_rol_id", "usuarios", type_="foreignkey")
    op.drop_column("usuarios", "rol_id")

    op.drop_table("rol_permisos")
    op.drop_table("permisos")
    op.drop_table("roles")