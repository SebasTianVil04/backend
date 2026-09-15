from alembic import op
import sqlalchemy as sa
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = 'bd0c3f199bc9'
down_revision: Union[str, None] = '3e86086e658f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'senas_categoria',
        'imagen_referencia',
        new_column_name='archivo_referencia'
    )
    op.add_column(
        'senas_categoria',
        sa.Column('tipo_referencia', sa.String(length=10), nullable=True)
    )
    op.execute(
        "UPDATE senas_categoria SET tipo_referencia = 'imagen' WHERE archivo_referencia IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column('senas_categoria', 'tipo_referencia')
    op.alter_column(
        'senas_categoria',
        'archivo_referencia',
        new_column_name='imagen_referencia'
    )
