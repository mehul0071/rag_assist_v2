"""add_summary_to_conversations
Revision ID: f7767a30f634
Revises: de797ee82055
Create Date: 2026-07-10 10:03:07.931885
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f7767a30f634'
down_revision: Union[str, Sequence[str], None] = 'de797ee82055'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('conversations', sa.Column('summary', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('conversations', 'summary')
