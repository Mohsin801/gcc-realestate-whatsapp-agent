# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
A bilingual (English + Khaleeji/MSA Arabic), always-on WhatsApp agent for GCC real estate (UAE/Dubai, Saudi, Qatar, Kuwait). It qualifies leads on Budget, Country, Intent and Timeline, answers legal/tax FAQs from a Qdrant knowledge base, and books Google Meet calls into the owner's Google Calendar. The source of truth is `GCC Real Estate WhatsApp AI Agent Technical Specification.pdf` (V2.0) in the repo root. Read it before changing any behaviour the spec defines.

The work follows the spec's Sprint 1 priorities, in phases: 0 scaffold → 1 webhook + Redis → 2 LangGraph bilingual agent → 3 Google Calendar engine → 4 Qdrant RAG → 5 latency and evaluation hardening. Some modules described below may not exist yet. Check before assuming.

## Commands
The host has Python 3.14; the Docker app image uses 3.12. Run these from PowerShell in the repo root:
```
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"   # setup
.\.venv\Scripts\python.exe -m pytest -q                                        # all tests
.\.venv\Scripts\python.exe -m pytest tests/test_scaffold.py::test_claim_wamid_is_idempotent   # single test
docker compose up --build                  # postgres:16, redis:8, qdrant, app on :8000 (runs `alembic upgrade head` first)
docker compose run --rm app pytest         # tests inside the container
.\.venv\Scripts\python.exe -m alembic revision -m "msg"   # new migration (write it by hand; keep it in sync with db/models.py)
```
Tests use `fakeredis` and `respx` mocks. pytest is set to `asyncio_mode = "auto"`, so async tests need no decorator. Unit tests must not need live Postgres, Redis, Qdrant, Meta or Google.

## Architecture
Request path: Meta WhatsApp → `POST /whatsapp/webhook` (`api/whatsapp.py`) → FastAPI `BackgroundTasks` → LangGraph pipeline (`agent/`) → reply sent through the Meta Graph API (`services/whatsapp_client.py`).
- Redis does three jobs: the LangGraph checkpointer (session state per phone number; needs the `redis:8` image for its JSON/Search modules), the `wamid` dedup keys, and the per-phone processing locks.
- Postgres holds the durable records: `leads`, `chat_logs`, `bookings`, `owners`.
- Qdrant holds the GCC legal/tax knowledge base and property listings.
- `tools/calendar.py` talks to Google Calendar through a `CalendarProvider` interface.

Packages sit at the top level (`api/`, `agent/`, `tools/`, `services/`, `db/`, `core/`, plus `main.py`). They are not under `app/`, because the spec names `agent/state.py`, `tools/calendar.py` and `api/whatsapp.py`.

### Contracts fixed by the spec (do not change)
- `agent/state.py` `GCCRealEstateState`: keep exactly the spec's fields and enum values:
  - `target_country`: DUBAI/SAUDI/QATAR/KUWAIT
  - `intent`: INVESTMENT/PERSONAL_USE/GOLDEN_VISA
  - `timeline`: IMMEDIATE/3_MONTHS/OFF_PLAN
  - `booking_status`: NONE/SLOTS_OFFERED/BOOKED
  - `messages` uses `operator.add`
  - `is_qualified` is true only when all four of budget, country, intent and timeline are set

  Data the spec doesn't list is kept outside the state:
  - the lead's name: Postgres `leads.profile_name`
  - ISO times of offered slots: Redis `offered:{phone}`
  - booking reference, Meet link, lead email: Postgres `bookings`
- `get_available_slots(owner_id, timezone="Asia/Dubai") -> list[str]` and `create_calendar_event(lead_phone, lead_name, slot_iso) -> bool` keep their spec signatures. `create_calendar_event` returns a bool, and the booking reference (the Google event id) is saved to `bookings.google_event_id`.
- The webhook handler signature is `handle_incoming_msg(payload: dict, background_tasks: BackgroundTasks, redis=Depends(get_redis))`. It must return 200 in under 200 ms (Meta's SLA), so it only parses, dedups with `SET NX` on the wamid, detects language and queues the task. Checking Meta's `X-Hub-Signature-256` signature is a route-level dependency, so the handler signature stays unchanged.

### Product decisions (made by the user)
- **LLM:** Gemini Flash through `langchain-google-genai`, behind a provider abstraction (`agent/llm.py`) so we can switch to Claude or OpenAI. Gemini embeddings feed Qdrant. Model ids come from `GEMINI_MODEL` / `GEMINI_EMBEDDING_MODEL`; never hardcode them.
- **WhatsApp:** Meta Cloud API only.
- **Calendar:** Google Calendar only, with the owner's own OAuth (not a service account). The refresh token is stored Fernet-encrypted in `owners`.
- **Booking flow:** the event is created as soon as the lead picks a slot, with a Google Meet link. Then we ask for the lead's email and add them as an attendee (`sendUpdates="all"`), which sends the invite. The booking is still valid if they never give an email.
- **Slots:** working hours are set in config (default Mon–Fri 09:00–18:00 in the owner's timezone, Asia/Dubai), filtered by Google freebusy, with 3 slots and by default at most 1 per day. Each lead sees times in their own country's timezone: DUBAI is "GST" (UTC+4); SAUDI, QATAR and KUWAIT are "AST" (UTC+3).
- **Persona:** the agent never reveals it is an AI or a bot, in English or Arabic, and a guard checks every reply for this. It speaks as a senior property consultant. Replies are short WhatsApp messages. Arabic replies use Khaleeji dialect.
- **No hallucinated legal or tax figures:** fees, percentages and thresholds may only come from retrieved knowledge-base passages. If retrieval returns nothing above the score threshold, the agent defers to the consultant call.
- **Response time:** under 3 s from webhook to send, in both languages (spec Definition of Done). `chat_logs.latency_ms` records it.

### Config
All settings live in `core/config.py` (pydantic-settings, loaded from `.env`; `.env.example` lists them). List-type values are JSON in the environment, e.g. `WORKING_DAYS=[0,1,2,3,4]` with Mon=0. Inside docker-compose, the `app` service overrides `DATABASE_URL`, `REDIS_URL` and `QDRANT_URL` with container hostnames.

## Git workflow: commit and push as you go (required)
The project's state must never live only in the working tree. While you work, commit to Git and push to GitHub regularly, so no progress is lost and any step can be reverted.
- **Commit at every logical checkpoint:** a feature or module finished, a bug fixed, tests made to pass, docs or config updated. Don't let several unrelated changes pile up, and don't end a session with uncommitted work.
- **Push right after each commit:** `git push` to `origin main` (private repo `Mohsin801/gcc-realestate-whatsapp-agent`). Check that `git status -sb` shows `## main...origin/main` with nothing ahead.
- **Clean commit messages:** a subject line in the imperative mood under ~72 characters (e.g. "Add WhatsApp webhook with wamid dedup"), a blank line, then a short bulleted body saying what changed and why. One concern per commit.
- **Run the tests before committing.** If something is committed in a broken or partial state, say so in the message.
- **Never commit** `.env` or `.venv`. Both are in `.gitignore`.
- **PowerShell:** write the message to a temp file and run `git commit -F <file>`. Piping a here-string to `-F -` does not work.
- `.gitattributes` keeps LF line endings because the code runs in Linux containers.
