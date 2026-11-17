"""Dev reset of one lead: services/reset.py, POST /admin/reset/{phone}."""
import operator
from typing import Annotated, TypedDict

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from core.config import Settings, get_settings
from db import repository
from db.models import Booking, Owner
from main import app
from services.redis import load_offered_slots, save_offered_slots
from services.reset import normalize_phone, reset_lead
from tools.demo_calendar import DemoScheduler

PHONE, OTHER = "923001234567", "971501234567"


class S(TypedDict):
    messages: Annotated[list, operator.add]


def tiny_graph(checkpointer):
    g = StateGraph(S)
    g.add_node("echo", lambda s: {"messages": ["ok"]})
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    return g.compile(checkpointer=checkpointer)


async def seed(sessionmaker, fake_redis, graph, scheduler):
    async with sessionmaker() as session:
        session.add(Owner(id="default", name="Owner"))
        await session.commit()
        for hour, phone in enumerate((PHONE, OTHER), start=5):
            lead = await repository.get_or_create_lead(session, phone, profile_name="Lead")
            await repository.update_lead_profile(session, phone, target_country="DUBAI", booking_status="BOOKED")
            await repository.log_message(session, lead.id, "in", "hi", wamid=f"wamid.{phone}")
            session.add(Booking(lead_id=lead.id, owner_id="default", slot_iso=f"2026-12-01T{hour:02d}:00:00Z",
                                google_event_id=f"evt-{phone}"))
        await session.commit()
    for phone in (PHONE, OTHER):
        await graph.ainvoke({"messages": ["hi"]}, {"configurable": {"thread_id": phone}})
        await save_offered_slots(fake_redis, phone, ["2026-12-01T05:00:00+00:00"])
        scheduler.offered[phone] = []
        scheduler.booked[phone] = object()


async def test_reset_lead_wipes_only_that_phone(sessionmaker, fake_redis):
    saver = InMemorySaver()
    graph = tiny_graph(saver)
    scheduler = DemoScheduler(Settings(_env_file=None))
    await seed(sessionmaker, fake_redis, graph, scheduler)

    result = await reset_lead(PHONE, redis=fake_redis, sessionmaker=sessionmaker, checkpointer=saver,
                              scheduler=scheduler)

    assert result.as_dict() == {
        "phone": PHONE, "checkpoint_cleared": True, "demo_calendar_cleared": True,
        "leads": 1, "chat_logs": 1, "bookings": 1, "google_events": 1,
    }
    cfg = lambda p: {"configurable": {"thread_id": p}}  # noqa: E731
    assert (await graph.aget_state(cfg(PHONE))).values == {}
    assert (await graph.aget_state(cfg(OTHER))).values == {"messages": ["hi", "ok"]}
    assert await load_offered_slots(fake_redis, PHONE) == []
    assert await load_offered_slots(fake_redis, OTHER) != []
    assert PHONE not in scheduler.offered and PHONE not in scheduler.booked
    assert OTHER in scheduler.booked
    async with sessionmaker() as session:
        assert await repository.get_lead(session, PHONE) is None
        other = await repository.get_lead(session, OTHER)
        assert other is not None and await repository.latest_booking(session, other.id) is not None

    again = await reset_lead(PHONE, redis=fake_redis, sessionmaker=sessionmaker)  # nothing left: no error
    assert (again.leads, again.chat_logs, again.bookings, again.checkpoint_cleared) == (0, 0, 0, False)


@pytest.mark.parametrize("raw,expected", [
    ("923001234567", "923001234567"), ("+92 300-1234567", "923001234567"), ("(971) 50.123.4567", "971501234567"),
])
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


@pytest.mark.parametrize("raw", ["", "abc", "12345", "1234567890123456", "92300123456x"])
def test_normalize_phone_rejects(raw):
    with pytest.raises(ValueError):
        normalize_phone(raw)


@pytest.fixture
async def client(monkeypatch, fake_redis, sessionmaker):
    monkeypatch.setenv("ADMIN_TOKEN", "admin-secret")
    get_settings.cache_clear()
    saver = InMemorySaver()
    app.state.redis, app.state.sessionmaker = fake_redis, sessionmaker
    app.state.checkpointer, app.state.scheduler = saver, DemoScheduler(Settings(_env_file=None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        c.graph = tiny_graph(saver)
        yield c


async def test_reset_endpoint_requires_admin_token_header(client):
    assert (await client.post(f"/admin/reset/{PHONE}")).status_code == 403
    assert (await client.post(f"/admin/reset/{PHONE}", headers={"X-Admin-Token": "wrong"})).status_code == 403
    # the token in the query string is not accepted
    assert (await client.post(f"/admin/reset/{PHONE}", params={"token": "admin-secret"})).status_code == 403


async def test_reset_endpoint_disabled_outside_development(client, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    r = await client.post(f"/admin/reset/{PHONE}", headers={"X-Admin-Token": "admin-secret"})
    assert r.status_code == 404


async def test_reset_endpoint_disabled_without_admin_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "")
    get_settings.cache_clear()
    assert (await client.post(f"/admin/reset/{PHONE}", headers={"X-Admin-Token": ""})).status_code == 403


async def test_reset_endpoint_resets_the_lead(client, sessionmaker):
    async with sessionmaker() as session:
        await repository.get_or_create_lead(session, PHONE)
    await client.graph.ainvoke({"messages": ["hi"]}, {"configurable": {"thread_id": PHONE}})

    r = await client.post("/admin/reset/+92 300 1234567", headers={"X-Admin-Token": "admin-secret"})

    assert r.status_code == 200
    assert r.json() == {"phone": PHONE, "checkpoint_cleared": True, "demo_calendar_cleared": True,
                             "leads": 1, "chat_logs": 0, "bookings": 0, "google_events": 0}
    assert (await client.graph.aget_state({"configurable": {"thread_id": PHONE}})).values == {}
    bad = await client.post("/admin/reset/not-a-phone", headers={"X-Admin-Token": "admin-secret"})
    assert bad.status_code == 400
