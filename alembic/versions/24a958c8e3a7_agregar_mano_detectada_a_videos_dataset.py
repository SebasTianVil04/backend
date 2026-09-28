"""agregar mano_detectada a videos_dataset

Revision ID: 24a958c8e3a7
Revises: a1b2c3d4e5f6_roles
Create Date: 2026-09-23 23:33:19.268402

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '24a958c8e3a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6_roles'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('videos_dataset', sa.Column('mano_detectada', sa.String(length=20), nullable=True))
    op.create_index(op.f('ix_videos_dataset_mano_detectada'), 'videos_dataset', ['mano_detectada'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_videos_dataset_mano_detectada'), table_name='videos_dataset')
    op.drop_column('videos_dataset', 'mano_detectada')