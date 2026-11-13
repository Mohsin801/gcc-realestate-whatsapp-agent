"""PostgreSQL schema: owners, leads, chat logs, bookings."""
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# BIGINT on Postgres; plain INTEGER on SQLite (used in tests) so autoincrement works there.
BigId = BigInteger().with_variant(Integer, "sqlite")


class Owner(Base):
    """A property owner/consultant whose Google Calendar receives bookings."""

    __tablename__ = "owners"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    calendar_id: Mapped[str] = mapped_column(String(255), default="primary")
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Dubai")
    working_hours: Mapped[dict | None] = mapped_column(JSON)  # {"days":[0..4],"start":"09:00","end":"18:00"}
    google_refresh_token_enc: Mapped[str | None] = mapped_column(Text)  # Fernet-encrypted
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    profile_name: Mapped[str | None] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(2), default="en")
    target_country: Mapped[str | None] = mapped_column(String(16))
    intent: Mapped[str | None] = mapped_column(String(32))
    budget_usd: Mapped[float | None] = mapped_column(Float)
    budget_original_amount: Mapped[float | None] = mapped_column(Float)
    budget_original_currency: Mapped[str | None] = mapped_column(String(8))
    timeline: Mapped[str | None] = mapped_column(String(16))
    is_qualified: Mapped[bool] = mapped_column(Boolean, default=False)
    booking_status: Mapped[str] = mapped_column(String(16), default="NONE")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chat_logs: Mapped[list["ChatLog"]] = relationship(back_populates="lead")
    bookings: Mapped[list["Booking"]] = relationship(back_populates="lead")


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    lead_id: Mapped[int] = mapped_column(BigId, ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    direction: Mapped[str] = mapped_column(String(3))  # 'in' | 'out'
    wamid: Mapped[str | None] = mapped_column(String(255), unique=True)  # inbound WhatsApp message id
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(2))
    latency_ms: Mapped[int | None] = mapped_column(Integer)  # outbound: webhook receipt -> send
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="chat_logs")


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = (
        # one confirmed booking per owner slot (migration 0002); also the reservation lock when booking
        Index(
            "uq_bookings_owner_slot_confirmed", "owner_id", "slot_iso", unique=True,
            postgresql_where=text("status = 'CONFIRMED'"), sqlite_where=text("status = 'CONFIRMED'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    lead_id: Mapped[int] = mapped_column(BigId, ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("owners.id"), index=True)
    slot_iso: Mapped[str] = mapped_column(String(64))  # UTC ISO, tools.scheduling.to_iso()
    google_event_id: Mapped[str | None] = mapped_column(String(255), unique=True)  # booking reference
    meet_link: Mapped[str | None] = mapped_column(String(512))
    lead_email: Mapped[str | None] = mapped_column(String(255))
    email_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="CONFIRMED")  # CONFIRMED | CANCELLED
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="bookings")
