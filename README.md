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
| `main.py` | FastAPI app + lifespan (Redis, DB) |
| `core/` | settings (`config.py`), JSON logging |
| `api/` | `whatsapp.py` webhook (spec §5), `health.py`, Google OAuth |
| `agent/` | `state.py` (spec §3), LangGraph graph, nodes, prompts |
| `tools/` | `calendar.py` (spec §4), RAG |
| `services/` | Redis helpers, WhatsApp client |
| `db/`, `migrations/` | SQLAlchemy models + Alembic |

## Quick start
1. Install **Docker Desktop**.
2. `copy .env.example .env` and fill in the values (see below).
3. `docker compose up --build`
4. Check `http://localhost:8000/health`: all of `redis`, `postgres` and `qdrant` should say `ok`.

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
