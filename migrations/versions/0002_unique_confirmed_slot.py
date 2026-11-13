"""one confirmed booking per owner slot

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-14
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Partial unique index: two leads can never hold the same owner slot, even under a race.
    op.create_index(
        "uq_bookings_owner_slot_confirmed",
        "bookings",
        ["owner_id", "slot_iso"],
        unique=True,
        postgresql_where=sa.text("status = 'CONFIRMED'"),
        sqlite_where=sa.text("status = 'CONFIRMED'"),
    )


def downgrade() -> None:
    op.drop_index("uq_bookings_owner_slot_confirmed", table_name="bookings")
