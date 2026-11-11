"""Drop unused market-data capture.

The strategy trades daily bars built from tickers and external daily candles.
Order-book snapshots, public execution prints and the paper trailing-stop
watermark had no remaining reader, so their tables and columns are removed.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

import btc_trader.db.types as bt

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

BIG_INT_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("executions") as batch_op:
        batch_op.drop_index("ix_executions_product_code_executed_at")
    op.drop_table("executions")
    with op.batch_alter_table("order_book_snapshots") as batch_op:
        batch_op.drop_index("ix_order_book_snapshots_product_code_received_at")
    op.drop_table("order_book_snapshots")
    with op.batch_alter_table("positions") as batch_op:
        batch_op.drop_column("high_watermark_price")
    with op.batch_alter_table("collector_runs") as batch_op:
        batch_op.drop_column("executions_inserted")
        batch_op.drop_column("order_books_inserted")


def downgrade() -> None:
    with op.batch_alter_table("collector_runs") as batch_op:
        batch_op.add_column(
            sa.Column("order_books_inserted", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("executions_inserted", sa.Integer(), nullable=False, server_default="0")
        )
    with op.batch_alter_table("positions") as batch_op:
        batch_op.add_column(sa.Column("high_watermark_price", bt.DecimalType(), nullable=True))
    op.create_table(
        "order_book_snapshots",
        sa.Column("id", BIG_INT_ID, autoincrement=True, nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("product_code", sa.String(length=32), nullable=False),
        sa.Column("received_at", bt.UtcDateTime(), nullable=False),
        sa.Column("mid_price", bt.DecimalType(30, 12), nullable=True),
        sa.Column("depth_levels", sa.Integer(), nullable=False),
        sa.Column("bids", JSON, nullable=False),
        sa.Column("asks", JSON, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_order_book_snapshots")),
        sa.UniqueConstraint(
            "exchange",
            "product_code",
            "received_at",
            name="uq_order_book_snapshots_exchange_product_code_received_at",
        ),
    )
    with op.batch_alter_table("order_book_snapshots") as batch_op:
        batch_op.create_index(
            "ix_order_book_snapshots_product_code_received_at", ["product_code", "received_at"]
        )
    op.create_table(
        "executions",
        sa.Column("id", BIG_INT_ID, autoincrement=True, nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("product_code", sa.String(length=32), nullable=False),
        sa.Column("exchange_execution_id", sa.String(length=64), nullable=False),
        sa.Column(
            "side", sa.Enum("BUY", "SELL", name="side_enum", native_enum=False), nullable=True
        ),
        sa.Column("price", bt.DecimalType(30, 12), nullable=False),
        sa.Column("size", bt.DecimalType(30, 12), nullable=False),
        sa.Column("executed_at", bt.UtcDateTime(), nullable=False),
        sa.Column("received_at", bt.UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_executions")),
        sa.UniqueConstraint(
            "exchange",
            "product_code",
            "exchange_execution_id",
            name="uq_executions_exchange_product_code_exchange_execution_id",
        ),
    )
    with op.batch_alter_table("executions") as batch_op:
        batch_op.create_index(
            "ix_executions_product_code_executed_at", ["product_code", "executed_at"]
        )
