"""simplificar_usuario

Revision ID: 5b30616b2093
Revises: 24a958c8e3a7
Create Date: 2026-09-27 15:53:05.890324

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b30616b2093'
down_revision: Union[str, None] = '24a958c8e3a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Agregar la columna nueva, nullable por ahora para poder rellenarla
    op.add_column('usuarios', sa.Column('nombre_usuario', sa.String(length=30), nullable=True))

    # 2. Rellenar nombre_usuario para filas existentes, usando la parte
    #    del email antes del @ (y el id para evitar choques de unicidad)
    op.execute("""
        UPDATE usuarios
        SET nombre_usuario = LOWER(SPLIT_PART(email, '@', 1)) || '_' || id::text
        WHERE nombre_usuario IS NULL
    """)

    # 3. Ahora sí, hacerla obligatoria y única
    op.alter_column('usuarios', 'nombre_usuario', nullable=False)
    op.create_unique_constraint('uq_usuarios_nombre_usuario', 'usuarios', ['nombre_usuario'])
    op.create_index('ix_usuarios_nombre_usuario', 'usuarios', ['nombre_usuario'])

    # 4. Eliminar los índices/constraints únicos de las columnas viejas antes de borrarlas
    op.drop_index('ix_usuarios_dni', table_name='usuarios')
    op.drop_index('ix_usuarios_pasaporte', table_name='usuarios')
    op.drop_index('ix_usuarios_telefono', table_name='usuarios')

    # 5. Eliminar las columnas sensibles
    op.drop_column('usuarios', 'dni')
    op.drop_column('usuarios', 'pasaporte')
    op.drop_column('usuarios', 'tipo_usuario')
    op.drop_column('usuarios', 'nombres')
    op.drop_column('usuarios', 'apellido_paterno')
    op.drop_column('usuarios', 'apellido_materno')
    op.drop_column('usuarios', 'telefono')
    op.drop_column('usuarios', 'direccion')
    op.drop_column('usuarios', 'fecha_nacimiento')


def downgrade() -> None:
    # Recrea las columnas viejas (vacías; los datos originales no se pueden recuperar)
    op.add_column('usuarios', sa.Column('dni', sa.String(length=8), nullable=True))
    op.add_column('usuarios', sa.Column('pasaporte', sa.String(length=20), nullable=True))
    op.add_column('usuarios', sa.Column('tipo_usuario', sa.String(length=20), nullable=False, server_default='peruano_mayor'))
    op.add_column('usuarios', sa.Column('nombres', sa.String(length=255), nullable=False, server_default=''))
    op.add_column('usuarios', sa.Column('apellido_paterno', sa.String(length=255), nullable=False, server_default=''))
    op.add_column('usuarios', sa.Column('apellido_materno', sa.String(length=255), nullable=False, server_default=''))
    op.add_column('usuarios', sa.Column('telefono', sa.String(length=20), nullable=True))
    op.add_column('usuarios', sa.Column('direccion', sa.Text(), nullable=True))
    op.add_column('usuarios', sa.Column('fecha_nacimiento', sa.Date(), nullable=True))

    op.create_index('ix_usuarios_dni', 'usuarios', ['dni'], unique=True)
    op.create_index('ix_usuarios_pasaporte', 'usuarios', ['pasaporte'], unique=True)
    op.create_index('ix_usuarios_telefono', 'usuarios', ['telefono'])

    op.drop_index('ix_usuarios_nombre_usuario', table_name='usuarios')
    op.drop_constraint('uq_usuarios_nombre_usuario', 'usuarios', type_='unique')
    op.drop_column('usuarios', 'nombre_usuario')