# GCC Real Estate WhatsApp AI Agent

Bilingual (English + Khaleeji/MSA Arabic) 24/7 WhatsApp lead qualifier and auto-scheduler for GCC real estate
(UAE/Dubai, Saudi Arabia, Qatar, Kuwait). Built to *Technical Specification V2.0*.

**Stack:** FastAPI · LangGraph · Google Gemini · Redis · PostgreSQL · Qdrant · Google Calendar (with Google Meet) · Meta WhatsApp Cloud API

```
WhatsApp (EN/AR) ──► POST /whatsapp/webhook ──► FastAPI ──► LangGraph pipeline ◄──► Redis (session + locks)
                                                   │               │        │
                                                   ▼               ▼        ▼
                                             PostgreSQL         Qdrant   Google Calendar
                                        (leads, chat logs)   (GCC legal  (live slots + booking)
                                                             & properties)
```

## Project layout
| Path | Purpose |
|---|---|
| `main.py` | FastAPI app + lifespan (Redis, DB, LLM, LangGraph checkpointer) |
| `core/` | settings (`config.py`), JSON logging |
| `api/` | `whatsapp.py` webhook (spec §5), `health.py`, `auth_google.py` (owner calendar connection) |
| `agent/` | `state.py` (spec §3), LangGraph graph, nodes, prompts |
| `tools/` | `calendar.py` (spec §4) + `scheduling.py` (slots, labels) + `calendar_providers/` (Google); RAG in Phase 4 |
| `services/` | Redis helpers, WhatsApp client, background message processor |
| `scripts/` | `chat_cli.py`: chat with the agent in the terminal |
| `db/`, `migrations/` | SQLAlchemy models + Alembic |

## Quick start
1. Install **Docker Desktop**.
2. `copy .env.example .env` and fill in the values (see below).
3. `docker compose up --build`
4. Check `http://localhost:8000/health`: all of `redis`, `postgres` and `qdrant` should say `ok`.

## How the agent works (Phase 2)
Each WhatsApp message runs one turn of a LangGraph state machine (`agent/graph.py`):

`detect_language → extract → normalize → [answer_question] → qualify | offer_slots | book_slot | capture_email | follow_up → respond → persona_guard`

- **extract**: one Gemini structured-output call. It is the bilingual intent classifier, reading English, Gulf Arabic, MSA or mixed messages, and it pulls out country, purpose, budget, timeline, slot choice and email. Unambiguous replies, such as a slot number or a bare email, skip the LLM.
- **normalize**: deterministic code that converts the budget to USD using the pegged rates in config. It understands Arabic numerals and phrasings such as "مليونين" and "مليون ونص". It also maps synonyms like Abu Dhabi to DUBAI and Doha to QATAR.
- **qualify**: asks one question at a time, in the order country → purpose → budget → timeline. When all four are known, `is_qualified` becomes true and live slots are offered.
- **persona_guard**: checks every reply and blocks:
  - AI/bot wording in English or Arabic (controlled by `PERSONA_CONCEAL_AI`, default on as the spec requires);
  - walls of text, so a reply is at most 2 bubbles of 400 characters;
  - a reply in the wrong language;
  - fees or percentages that don't come from the knowledge base.

  A rejected draft is regenerated once; if it fails again, a vetted template is sent instead.
- **Resilience**: if the LLM is slow or down, rule-based extraction and Khaleeji/English templates take over, so the lead always gets an answer.
- **Sessions**: kept in Redis through the LangGraph checkpointer. The qualification results are also saved to Postgres `leads`, and an expired session is re-seeded from there.

## Calendar engine (Phase 3)
`tools/calendar.py` provides the two spec functions, `get_available_slots` and `create_calendar_event`, with the spec's exact signatures. They use the owner's Google Calendar through its REST API (`tools/calendar_providers/google.py`).
- **Finding slots:**
  - 30-minute slots inside working hours (default Mon–Fri 09:00–18:00 Asia/Dubai);
  - at least 2 hours' notice;
  - busy times removed using Google freeBusy and existing bookings;
  - 3 slots, at most 1 per day.
- **Timezone:** each lead sees times in their own timezone, chosen from the phone country code: +971 → GST; +966, +974, +965 → AST. Any other number gets the owner's timezone.
- **Labels:** every label includes its date, e.g. "Tomorrow, Tue 17 Nov at 8:00 AM AST" or "بكرة الثلاثاء 17 نوفمبر، الساعة 9:00 الصبح بتوقيت الإمارات". Labels are rebuilt from the stored UTC time each time they're shown, so "Today" and "Tomorrow" always match the real date.
- **Booking:**
  1. The slot is reserved in Postgres; a unique index means two leads can't get the same slot.
  2. It's checked again against freeBusy.
  3. The Google Calendar event is created with a Google Meet link.
  4. The event id (the booking reference) and the Meet link are saved in `bookings`.
- **Calendar invite:** when the lead sends their email, they are added as an attendee and Google emails them the invite.

**Connecting the owner's calendar (one time):**
1. In Google Cloud Console, enable the **Google Calendar API**. Then create an OAuth client of type **Web application**, with redirect URI `<PUBLIC_BASE_URL>/auth/google/callback`.
2. In the OAuth consent screen, add the scopes `calendar.events` and `calendar.freebusy`. **Publish the app** (move it out of "Testing"). Apps in Testing get refresh tokens that expire after 7 days.
3. Set these in `.env`: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `TOKEN_ENCRYPTION_KEY` (see `.env.example`), `ADMIN_TOKEN`, `OWNER_NAME`, `OWNER_EMAIL`.
4. The owner opens `<PUBLIC_BASE_URL>/auth/google/start?token=<ADMIN_TOKEN>` and approves.

Until the calendar is connected, a qualified lead is told the consultant will reach out to arrange a time.

Try it locally without WhatsApp:
```
.venv\Scripts\python scripts\chat_cli.py            # needs GEMINI_API_KEY + GEMINI_MODEL in .env
.venv\Scripts\python scripts\chat_cli.py --no-llm --slots   # rules/templates + an empty demo calendar
.venv\Scripts\python scripts\chat_cli.py --slots --phone 966512345678   # see times in Saudi time (AST)
```

### Running tests
```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```
or inside Docker: `docker compose run --rm app pytest`

## Credentials you need
| What | Where |
|---|---|
| Gemini API key | https://aistudio.google.com/apikey |
| WhatsApp token, phone number ID, app secret | Meta developer app → WhatsApp → API Setup; app secret under App settings → Basic |
| Webhook verify token | Any random string; enter the same value in Meta's webhook config |
| Google OAuth client | Google Cloud Console → enable Calendar API → Credentials → OAuth client (Web). Redirect URI: `<PUBLIC_BASE_URL>/auth/google/callback` |
| Public URL for the webhook | `ngrok http 8000` (or cloudflared) during development |
