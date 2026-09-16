"""reemplazar es_admin por rol

Revision ID: ecef25f5af56
Revises: 0001_agregar_auditoria
Create Date: ...
"""
from alembic import op
import sqlalchemy as sa

revision = "ecef25f5af56"
down_revision = "0001_agregar_auditoria"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "usuarios",
        sa.Column("rol", sa.String(20), nullable=False, server_default="usuario"),
    )
    op.create_index("ix_usuarios_rol", "usuarios", ["rol"])
    op.execute("UPDATE usuarios SET rol = 'admin' WHERE es_admin = true")
    op.drop_column("usuarios", "es_admin")


def downgrade():
    op.add_column("usuarios", sa.Column("es_admin", sa.Boolean(), server_default="false"))
    op.execute("UPDATE usuarios SET es_admin = true WHERE rol = 'admin'")
    op.drop_index("ix_usuarios_rol", table_name="usuarios")
    op.drop_column("usuarios", "rol")