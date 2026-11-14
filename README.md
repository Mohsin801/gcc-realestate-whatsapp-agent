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
| `tools/` | `calendar.py` (spec §4) + `scheduling.py` (slots, labels) + `calendar_providers/` (Google); `rag.py` (Qdrant knowledge base) |
| `knowledge_base/facts/` | verified GCC legal/tax facts, one JSON file per country, each with official sources and the date checked |
| `services/` | Redis helpers, WhatsApp client, background message processor |
| `evals/` | conversation evaluation: `conversations.json` (scripted EN/AR leads for all four markets) and `harness.py` (the checks and the report) |
| `scripts/` | `chat_cli.py`: chat with the agent in the terminal; `ingest_kb.py`: validate, ingest and search the knowledge base; `eval_agent.py`: run the evaluation; `latency_report.py`: production reply latency from `chat_logs` |
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
.venv\Scripts\python scripts\chat_cli.py --kb       # answer legal/tax questions from Qdrant (after an ingest)
```

## Knowledge base (Phase 4)
The agent answers legal and tax questions (transfer fees, RETT, foreign ownership, mortgage caps, residency through property) only from facts checked on **official government sources**. When the knowledge base has nothing relevant, it doesn't guess. It says it will confirm the details on the call.

**What's in it:** 24 facts in `knowledge_base/facts/` (Dubai 6, Saudi 8, Qatar 5, Kuwait 5). Each fact is written in English and Arabic, lists its official source URLs, and records the date it was checked (`checked_on`). For example:
```json
{
  "id": "dubai-dld-sale-registration-fee",
  "country": "DUBAI",
  "topic": "fees",
  "title_en": "Dubai Land Department (DLD) property sale registration / transfer fee",
  "title_ar": "رسوم تسجيل بيع العقار (رسوم النقل) لدى دائرة الأراضي والأملاك في دبي",
  "en": "In Dubai, the Dubai Land Department (DLD) charges a sale registration fee of 4% of the sale value. ...",
  "ar": "في دبي، تفرض دائرة الأراضي والأملاك رسوم تسجيل بيع بنسبة 4% من قيمة البيع. ...",
  "sources": [{"name": "Dubai Land Department - Property Sale Registration service",
               "url": "https://dubailand.gov.ae/en/eservices/property-sale-registration/"}],
  "checked_on": "2026-11-14"
}
```

**How a question is answered:**
1. The question is embedded with Gemini and searched in Qdrant. The search is filtered to the market the question names. If it names none, it uses the lead's own target market. Asking about another market ("and in Saudi?") doesn't change the lead's target market.
2. Passages scoring below `RAG_SCORE_THRESHOLD` are ignored. If none are left, or the search fails or takes longer than `RAG_TIMEOUT_SECONDS`, the agent defers to the call.
3. The passage is sent in the lead's language, and the LLM answers using only those passages.
4. The persona guard rejects any reply that contains a number not found in the passages or in the lead's own message. If the LLM is down, the best passage itself is sent as the answer.
5. The sources, scores and `checked_on` dates used for each answer are kept in the turn state (`turn["sources"]`).

**Commands** (`ingest` and `search` need `GEMINI_API_KEY`, `GEMINI_EMBEDDING_MODEL` and a running Qdrant):
```
.venv\Scripts\python scripts\ingest_kb.py validate                  # check the fact files (offline)
.venv\Scripts\python scripts\ingest_kb.py ingest                    # embed + upsert the facts
.venv\Scripts\python scripts\ingest_kb.py ingest --listings my_listings.json   # also load the owner's listings
.venv\Scripts\python scripts\ingest_kb.py ingest --recreate         # after changing the embedding model or EMBEDDING_DIM
.venv\Scripts\python scripts\ingest_kb.py search "DLD fees?" --country DUBAI --lang en --all-scores
```
`ingest` can be re-run safely: a fact removed from the files is also removed from Qdrant.

**Adding or changing a fact:**
1. Find the rule on an official government page. Only the domains in `tools/rag.py` `OFFICIAL_DOMAINS` are accepted (DLD, Dubai government, u.ae, ICP, ZATCA, REGA, SPA, my.gov.sa, Ministry of Housing, Qatar MoJ / Aqarat / Hukoomi / Sharek, KDIPA, Kuwait MoJ / e.gov.kw). Add a domain only if it is a government site.
2. Write the fact in both languages, at most 400 characters each, and make sure the English and Arabic contain **the same numbers**. Set `checked_on` to the day you checked the source.
3. Run `ingest_kb.py validate` and the tests, then `ingest`.

**Keeping facts current:** `validate` and `ingest` warn about any fact not re-checked within `RAG_FACT_MAX_AGE_DAYS` (180). Re-open its sources, update the text if the rule changed, and set a new `checked_on`.

**Calibrating the threshold:** `RAG_SCORE_THRESHOLD` defaults to 0.6, which hasn't been tested with the real Gemini embeddings yet. After the first ingest, run `search --all-scores` with:
- questions the knowledge base should answer;
- questions it shouldn't, such as other markets' rules or off-topic questions.

Set the threshold between the two groups' scores. Repeat this whenever the embedding model changes.

**Owner listings (optional):** a JSON list of objects with `id`, `country` (DUBAI/SAUDI/QATAR/KUWAIT), `title`, `price` and `currency` (AED/SAR/QAR/KWD/USD). The optional fields are `location`, `property_type`, `bedrooms`, `description`, `title_ar`, `description_ar` and `url`. Listings go into the separate `properties` collection.

**Left out on purpose**, because the official sources were unclear or unreachable on 2026-11-14 (the agent defers these to the call):
- How long the UAE Golden Visa lasts: u.ae says 5 years, the ICP page says 10.
- Saudi Premium Residency: sapr.gov.sa could not be reached.
- How many freehold and usufruct areas Qatar has: the Ministry of Justice page shows them only as images.

Without an embedding model configured, the app starts with the knowledge base switched off, and every legal or tax question is deferred to the call.

## Latency and evaluation (Phase 5)
The spec's Definition of Done asks for a reply in **under 3 seconds** in both English and Arabic, and for leads to be qualified and questions answered **without hallucinating**.

**Latency budget.** Each turn has a deadline that starts when the webhook receives the message (`LATENCY_BUDGET_SECONDS`, default 3). Every slow call is capped to the time that is left:
- **Extraction (LLM):** capped so that `LATENCY_REPLY_RESERVE_SECONDS` stays free for writing the reply. If it runs out, the rule-based extractor reads the message instead.
- **Knowledge search:** capped the same way. If it runs out, the agent says it will confirm the details on the call.
- **Reply (LLM):** capped so that `LATENCY_SEND_RESERVE_SECONDS` stays free for sending through Meta. If it runs out, the vetted template is sent.
- **Persona-guard retry:** only attempted if there is still time; otherwise the template is sent.

A call that would get less than `LATENCY_MIN_CALL_SECONDS` is skipped. Skipped steps are logged per turn (`budget_skips`). Calendar calls are **not** cut short, because a booking must never stop half-way; `AGENT_TIMEOUT_SECONDS` (20) is the hard cap on a turn.

**What else keeps the reply fast:**
- The read receipt and typing indicator are sent alongside the work, not before it.
- All reply bubbles are sent before they are written to the database.
- The LangGraph session is saved once per turn instead of after every step.
- Google freeBusy is cached for `CALENDAR_BUSY_CACHE_SECONDS` (120) and fetched in the background as soon as the lead is one answer away from being offered slots. Bookings since the fetch are still excluded (they are read from Postgres), and booking always re-checks freeBusy live.
- At startup one tiny LLM call, knowledge search and calendar fetch open the connections (`WARMUP_ON_STARTUP`).

**Measuring it.** `chat_logs.latency_ms` on the first reply bubble is the time from webhook receipt to Meta accepting the reply. Replies over the budget are logged as warnings. The per-turn agent log includes how long each graph step took (`timings_ms`).
```
.venv\Scripts\python scripts\latency_report.py                # p50 / p95 / p99 / max per language, last 7 days
.venv\Scripts\python scripts\latency_report.py --days 1 --check   # exit 1 if p95 is over the budget
```

**Evaluation.** `evals/conversations.json` holds scripted conversations: English and Gulf Arabic leads in Dubai, Saudi, Qatar and Kuwait, booking flows, legal/tax questions, an off-topic question, "are you a bot?" in both languages, and a language switch. Every turn is checked for:
- the reply language;
- no AI/bot disclosure;
- at most 2 bubbles of 400 characters;
- **no figure** that didn't come from the lead or from a verified knowledge-base passage (stricter than the live guard, which checks question turns only);
- the agent finishing within the budget minus the send reserve;
- the expected qualification state, reply steps and retrieved facts.
```
.venv\Scripts\python scripts\eval_agent.py --no-llm          # offline: rules + templates (also runs in pytest)
.venv\Scripts\python scripts\eval_agent.py                   # live Gemini, no knowledge base
.venv\Scripts\python scripts\eval_agent.py --kb --repeat 5   # live Gemini + Qdrant, latency over 5 runs
```
The report lists every failed check, latency per language, how often the reply came from the LLM or from a template, steps skipped for time, and the p95 of each graph step. Cases marked `"requires": ["llm"]` are skipped offline. Run the live evaluation after changing the model, the prompts or the latency settings. If `extract` or `respond` p95 is close to the budget, try `GEMINI_THINKING_LEVEL=minimal` or a faster model before raising the budget.

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
| Gemini API key | https://aistudio.google.com/apikey (also used for the knowledge-base embeddings) |
| Qdrant | runs in docker-compose; for Qdrant Cloud set `QDRANT_URL` and `QDRANT_API_KEY` |
| WhatsApp token, phone number ID, app secret | Meta developer app → WhatsApp → API Setup; app secret under App settings → Basic |
| Webhook verify token | Any random string; enter the same value in Meta's webhook config |
| Google OAuth client | Google Cloud Console → enable Calendar API → Credentials → OAuth client (Web). Redirect URI: `<PUBLIC_BASE_URL>/auth/google/callback` |
| Public URL for the webhook | `ngrok http 8000` (or cloudflared) during development |
