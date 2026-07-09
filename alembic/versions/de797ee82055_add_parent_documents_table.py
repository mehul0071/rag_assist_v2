"""add_parent_documents_table
Revision ID: de797ee82055
Revises: f30245e19eb3
Create Date: 2026-07-09 11:04:10.093312
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'de797ee82055'
down_revision: Union[str, Sequence[str], None] = 'f30245e19eb3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('parent_documents',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('page_content', sa.Text(), nullable=False),
    sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('parent_documents')
