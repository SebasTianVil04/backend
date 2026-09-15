"""agregar campos de auditoria a todos los modelos

Revision ID: 0001_agregar_auditoria
Revises: 
Create Date: 2026-09-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_agregar_auditoria"
down_revision = "bd0c3f199bc9"
branch_labels = None
depends_on = None

TABLAS_SOLO_USUARIOS_AUDITORIA = [
    "categorias",
    "examenes",
    "progreso_clases",
    "progreso_lecciones",
    "clases",
    "lecciones",
    "senas_categoria",
    "tipos_categoria",
]

TABLAS_CON_FECHA_ACTUALIZACION_FALTANTE = [
    "categorias_dataset",
    "entrenamientos",
    "modelos_ia",
    "preguntas_examen",
    "resultados_examenes",
    "sesiones_estudio",
    "tokens_recuperacion",
]

TABLAS_CON_FECHA_CREACION_Y_ACTUALIZACION_FALTANTE = [
    "videos_dataset",
    "calibraciones_usuario",
    "practicas",
]


def _agregar_columnas_usuarios_auditoria(tabla):
    op.add_column(tabla, sa.Column("creado_por_id", sa.Integer(), nullable=True))
    op.add_column(tabla, sa.Column("actualizado_por_id", sa.Integer(), nullable=True))
    op.create_index(f"ix_{tabla}_creado_por_id", tabla, ["creado_por_id"])
    op.create_index(f"ix_{tabla}_actualizado_por_id", tabla, ["actualizado_por_id"])
    op.create_foreign_key(
        f"fk_{tabla}_creado_por_id_usuarios",
        tabla, "usuarios", ["creado_por_id"], ["id"], ondelete="SET NULL"
    )
    op.create_foreign_key(
        f"fk_{tabla}_actualizado_por_id_usuarios",
        tabla, "usuarios", ["actualizado_por_id"], ["id"], ondelete="SET NULL"
    )


def _quitar_columnas_usuarios_auditoria(tabla):
    op.drop_constraint(f"fk_{tabla}_actualizado_por_id_usuarios", tabla, type_="foreignkey")
    op.drop_constraint(f"fk_{tabla}_creado_por_id_usuarios", tabla, type_="foreignkey")
    op.drop_index(f"ix_{tabla}_actualizado_por_id", table_name=tabla)
    op.drop_index(f"ix_{tabla}_creado_por_id", table_name=tabla)
    op.drop_column(tabla, "actualizado_por_id")
    op.drop_column(tabla, "creado_por_id")


def upgrade():
    for tabla in TABLAS_SOLO_USUARIOS_AUDITORIA:
        _agregar_columnas_usuarios_auditoria(tabla)

    for tabla in TABLAS_CON_FECHA_ACTUALIZACION_FALTANTE:
        _agregar_columnas_usuarios_auditoria(tabla)
        op.add_column(tabla, sa.Column("fecha_actualizacion", sa.DateTime(timezone=True), nullable=True))

    for tabla in TABLAS_CON_FECHA_CREACION_Y_ACTUALIZACION_FALTANTE:
        _agregar_columnas_usuarios_auditoria(tabla)
        op.add_column(
            tabla,
            sa.Column("fecha_creacion", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        )
        op.add_column(tabla, sa.Column("fecha_actualizacion", sa.DateTime(timezone=True), nullable=True))

    op.add_column("usuarios", sa.Column("creado_por_id", sa.Integer(), nullable=True))
    op.add_column("usuarios", sa.Column("actualizado_por_id", sa.Integer(), nullable=True))
    op.create_index("ix_usuarios_creado_por_id", "usuarios", ["creado_por_id"])
    op.create_index("ix_usuarios_actualizado_por_id", "usuarios", ["actualizado_por_id"])
    op.create_foreign_key(
        "fk_usuarios_creado_por_id_usuarios",
        "usuarios", "usuarios", ["creado_por_id"], ["id"], ondelete="SET NULL"
    )
    op.create_foreign_key(
        "fk_usuarios_actualizado_por_id_usuarios",
        "usuarios", "usuarios", ["actualizado_por_id"], ["id"], ondelete="SET NULL"
    )

    op.alter_column(
        "tokens_recuperacion",
        "fecha_creacion",
        type_=sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        postgresql_using="fecha_creacion AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "tokens_recuperacion",
        "fecha_expiracion",
        type_=sa.DateTime(timezone=True),
        postgresql_using="fecha_expiracion AT TIME ZONE 'UTC'",
    )

    op.add_column(
        "modelo_video_entrenamiento",
        sa.Column("creado_por_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_modelo_video_entrenamiento_creado_por_id_usuarios",
        "modelo_video_entrenamiento", "usuarios", ["creado_por_id"], ["id"], ondelete="SET NULL"
    )


def downgrade():
    op.drop_constraint(
        "fk_modelo_video_entrenamiento_creado_por_id_usuarios",
        "modelo_video_entrenamiento", type_="foreignkey"
    )
    op.drop_column("modelo_video_entrenamiento", "creado_por_id")

    op.alter_column(
        "tokens_recuperacion",
        "fecha_expiracion",
        type_=sa.DateTime(timezone=False),
    )
    op.alter_column(
        "tokens_recuperacion",
        "fecha_creacion",
        type_=sa.DateTime(timezone=False),
        server_default=None,
    )

    op.drop_constraint("fk_usuarios_actualizado_por_id_usuarios", "usuarios", type_="foreignkey")
    op.drop_constraint("fk_usuarios_creado_por_id_usuarios", "usuarios", type_="foreignkey")
    op.drop_index("ix_usuarios_actualizado_por_id", table_name="usuarios")
    op.drop_index("ix_usuarios_creado_por_id", table_name="usuarios")
    op.drop_column("usuarios", "actualizado_por_id")
    op.drop_column("usuarios", "creado_por_id")

    for tabla in TABLAS_CON_FECHA_CREACION_Y_ACTUALIZACION_FALTANTE:
        op.drop_column(tabla, "fecha_actualizacion")
        op.drop_column(tabla, "fecha_creacion")
        _quitar_columnas_usuarios_auditoria(tabla)

    for tabla in TABLAS_CON_FECHA_ACTUALIZACION_FALTANTE:
        op.drop_column(tabla, "fecha_actualizacion")
        _quitar_columnas_usuarios_auditoria(tabla)

    for tabla in TABLAS_SOLO_USUARIOS_AUDITORIA:
        _quitar_columnas_usuarios_auditoria(tabla)