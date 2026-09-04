"""Persist paper trailing-stop state."""

import sqlalchemy as sa
from alembic import op

import btc_trader.db.types as bt

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("positions", sa.Column("high_watermark_price", bt.DecimalType(), nullable=True))


def downgrade() -> None:
    op.drop_column("positions", "high_watermark_price")
