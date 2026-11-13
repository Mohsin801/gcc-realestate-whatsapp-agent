"""initial schema: owners, leads, chat_logs, bookings

Revision ID: 0001
Revises:
Create Date: 2026-11-13
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "owners",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255)),
        sa.Column("calendar_id", sa.String(255), nullable=False, server_default="primary"),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="Asia/Dubai"),
        sa.Column("working_hours", sa.JSON()),
        sa.Column("google_refresh_token_enc", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "leads",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("phone", sa.String(32), nullable=False),
        sa.Column("profile_name", sa.String(255)),
        sa.Column("language", sa.String(2), nullable=False, server_default="en"),
        sa.Column("target_country", sa.String(16)),
        sa.Column("intent", sa.String(32)),
        sa.Column("budget_usd", sa.Float()),
        sa.Column("budget_original_amount", sa.Float()),
        sa.Column("budget_original_currency", sa.String(8)),
        sa.Column("timeline", sa.String(16)),
        sa.Column("is_qualified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("booking_status", sa.String(16), nullable=False, server_default="NONE"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_leads_phone", "leads", ["phone"], unique=True)

    op.create_table(
        "chat_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("lead_id", sa.BigInteger(), sa.ForeignKey("leads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("direction", sa.String(3), nullable=False),
        sa.Column("wamid", sa.String(255), unique=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(2)),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chat_logs_lead_id", "chat_logs", ["lead_id"])

    op.create_table(
        "bookings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("lead_id", sa.BigInteger(), sa.ForeignKey("leads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_id", sa.String(64), sa.ForeignKey("owners.id"), nullable=False),
        sa.Column("slot_iso", sa.String(64), nullable=False),
        sa.Column("google_event_id", sa.String(255), unique=True),
        sa.Column("meet_link", sa.String(512)),
        sa.Column("lead_email", sa.String(255)),
        sa.Column("email_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(16), nullable=False, server_default="CONFIRMED"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_bookings_lead_id", "bookings", ["lead_id"])
    op.create_index("ix_bookings_owner_id", "bookings", ["owner_id"])


def downgrade() -> None:
    op.drop_table("bookings")
    op.drop_table("chat_logs")
    op.drop_table("leads")
    op.drop_table("owners")
