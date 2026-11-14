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
.\.venv\Scripts\python.exe scripts\eval_agent.py --no-llm   # conversation evaluation offline (live: drop --no-llm, add --kb)
.\.venv\Scripts\python.exe scripts\latency_report.py        # production reply latency from chat_logs
.\.venv\Scripts\python.exe -m alembic revision -m "msg"   # new migration (write it by hand; keep it in sync with db/models.py)
```
Tests use `fakeredis` and `respx` mocks. pytest is set to `asyncio_mode = "auto"`, so async tests need no decorator. Unit tests must not need live Postgres, Redis, Qdrant, Meta or Google.

## Architecture
Request path: Meta WhatsApp → `POST /whatsapp/webhook` (`api/whatsapp.py`) → FastAPI `BackgroundTasks` → LangGraph pipeline (`agent/`) → reply sent through the Meta Graph API (`services/whatsapp_client.py`).
- Redis does three jobs: the LangGraph checkpointer (session state per phone number; needs the `redis:8` image for its JSON/Search modules), the `wamid` dedup keys, and the per-phone processing locks.
- Postgres holds the durable records: `leads`, `chat_logs`, `bookings`, `owners`.
- Qdrant holds the GCC legal/tax knowledge base and property listings.
- `tools/calendar.py` talks to Google Calendar through a `CalendarProvider` interface.

Background tasks reach the shared WhatsApp client, DB session factory and agent function through `services/runtime.py` (`Runtime`, `set_runtime`/`get_runtime`). It works this way because the spec-fixed webhook signature has no `Request`. `main.py` fills the container at startup, and tests swap in fakes (`tests/conftest.py`: `RecordingSender`, fakeredis, in-memory SQLite). `services/processor.py` is the background step:
1. take the per-phone lock
2. mark the message read and show the typing indicator
3. upsert the lead and log the inbound message
4. call `runtime.run_agent(msg, language)` with a timeout
5. send the replies and log each outbound message with `latency_ms`

It never raises. On any failure it sends a persona-safe template from `agent/templates.py`, so the lead always gets an answer.

### Agent (Phase 2)
`agent/runner.py` `AgentService` is the `run_agent` callable. Each turn it:
- calls the compiled graph with `thread_id = phone`;
- when no checkpoint exists, seeds the qualification fields from Postgres `leads`;
- afterwards writes them back with `repository.update_lead_profile`.

The graph is built in `agent/graph.py`; the nodes live in `agent/nodes.py` (`AgentNodes`):
`detect_language → extract → normalize → [answer_question] → qualify|offer_slots|book_slot|capture_email|follow_up → respond → persona_guard`.

- The graph state is `GCCRealEstateState` plus one `turn` dict for per-turn working data: the extraction, the `parts` to say, the draft, and the final `replies`. The runner resets `turn` at the start of each turn.
- Step nodes don't write text. They append `parts` (e.g. `ASK`, `OFFER_SLOTS`, `BOOKED`). `respond._compose` turns the parts into:
  - task lines for the LLM;
  - a template fallback;
  - an optional appendix bubble (the slot list, or the booking time and Meet link), which is written by code so it is always exact.
- `extract` uses the LLM's structured output (`agent/prompts.py` `Extraction`). It skips the LLM for a bare slot number or email. On any LLM error it falls back to the rules in `agent/normalize.py`.
- The guard (`agent/persona_guard.py`) checks for:
  - AI/bot wording (only while `persona_conceal_ai` is on);
  - replies too long for the bubble limit;
  - a reply in the wrong language;
  - ungrounded figures.

  A failing draft is regenerated once, then replaced by the template.
- Calendar and RAG are reached only through the `Scheduler` / `Knowledge` protocols in `agent/services.py`.
  - `main.py` wires in `tools.calendar.CalendarScheduler`.
  - `main.py` wires in `tools.rag.KnowledgeBase` through `open_knowledge`. It falls back to `NullKnowledge`, which defers every question to the call, when no embedding model is configured. `NullKnowledge` is also the default whenever `build_graph` gets no knowledge.
  - `NullScheduler` is the default whenever `build_graph` gets no scheduler (tests, `chat_cli` without `--slots`). It gives qualified leads a "consultant will reach out" handoff.

### Calendar (Phase 3)
- **`tools/calendar.py`:**
  - The spec functions delegate to a process-wide `CalendarEngine`, which `calendar.configure()` sets in the lifespan.
  - `CalendarScheduler` adapts the engine for the agent. It stores offered slots as UTC ISO strings in Redis `offered:{phone}`, and on every call it renders labels from them with `format_slot` in `core.timezones.lead_timezone(phone, owner_tz)`.
- **`tools/scheduling.py`:** pure logic, with no I/O: `candidate_slots`, `pick_slots`, `format_slot`, `to_iso`/`from_iso`. The canonical stored form is UTC, at second precision.
- **Booking:**
  1. `book()` validates the slot against the current candidates: not in the past, outside the notice period, on the grid, within working hours.
  2. It reserves the slot by inserting the `bookings` row; a partial unique index on (owner_id, slot_iso) WHERE status='CONFIRMED' (migration 0002) blocks double-booking.
  3. It re-checks freeBusy.
  4. It creates the event with a Meet link and saves `google_event_id` and `meet_link`.

  Any failure deletes the reservation. Read lead attributes *before* the reserve: a rollback expires loaded objects, and lazy-loading after that fails under asyncio.
- **Provider:** `tools/calendar_providers/google.py` calls the REST API over async httpx; it does not use google-api-python-client.
  - `GoogleOAuth` caches access tokens per refresh token.
  - A 400/401 from the token endpoint raises `CalendarNotConnected`.
  - The provider retries once on a network error, 429 or 5xx, and refreshes the token on a 401.
- **`api/auth_google.py`:**
  - `/auth/google/start` needs `ADMIN_TOKEN`; when it is empty, the endpoint is disabled.
  - The single-use `oauth_state:{state}` lives in Redis.
  - The callback requires a refresh token and every scope in `GOOGLE_OAUTH_SCOPES`, then upserts the `owners` row with the token Fernet-encrypted (`core/crypto.py`).
- **Tests:**
  - Inject `clock=` and `provider_factory=` into `CalendarEngine` (`tests/test_calendar.py` `FakeProvider`).
  - Google HTTP is mocked with respx.
  - `tzdata` is a dependency, because Windows has no system zoneinfo.
- `agent/llm.py` `build_llm(settings)` returns a `LangChainLLM` with `extract` and `generate`. Every call is time-boxed (`LLM_TIMEOUT_SECONDS`), and `max_retries=0`.
- Arabic synonym patterns must be written in `normalize_text()` form: ا for أ/إ/آ, ي for ى, ه for ة. A test enforces this.
- Tests use `tests/agent_fakes.py` (`FakeLLM`, `FakeScheduler`, `FakeKnowledge`, `FakeEmbedder`) together with `InMemorySaver`. `scripts/chat_cli.py` runs the agent in the terminal (`--no-llm`, `--slots`, `--kb`).
- `tests/conftest.py` switches the working directory to `tests/` (so no `.env` is loaded). Build repo paths in tests from `Path(__file__)`, not from the working directory.

Packages sit at the top level (`api/`, `agent/`, `tools/`, `services/`, `db/`, `core/`, plus `main.py`). They are not under `app/`, because the spec names `agent/state.py`, `tools/calendar.py` and `api/whatsapp.py`.

### Knowledge base (Phase 4)
- **Content: `knowledge_base/facts/{dubai,saudi,qatar,kuwait}.json`.**
  - Each fact has `id`, `country`, `topic`, `title_en/ar`, `en/ar`, `sources[{name,url}]` and `checked_on`.
  - Facts come **only from official government sources**. `tools/rag.py` `OFFICIAL_DOMAINS` is the allowlist; don't add a domain that isn't a government site.
  - Never add a figure that you haven't seen on the official page. If the sources disagree or can't be read, leave the fact out; the agent then defers to the call.
  - Kuwait facts come from the scanned Official Gazette (Kuwait Al-Youm No. 1726, Decree-Law 7/2025), via the KDIPA PDF.
  - Left out on purpose (2026-11-14):
    - the UAE Golden Visa duration (u.ae says 5 years, ICP says 10);
    - Saudi Premium Residency (sapr.gov.sa was unreachable);
    - the Qatar freehold/usufruct area counts (only in images on the MoJ page).
- **Validation (`Fact` model):**
  - the English and Arabic texts contain the same figures;
  - the Arabic text really is Arabic;
  - `checked_on` is not in the future;
  - each text is at most 400 characters (one bubble, so it can be sent as-is).

  `stale_facts` flags facts older than `RAG_FACT_MAX_AGE_DAYS` (180). `Listing` and `load_listings` load the owner's listings into the separate `properties` collection.
- **`GeminiEmbedder`:**
  - gemini-embedding-2 rejects `task_type`, so tasks go in as text prefixes (`GEMINI_EMBEDDING_TASK_PREFIXES`).
  - It sends one request per text, because a list input returns a single aggregated vector.
  - Vectors are L2-normalised.
- **`KnowledgeBase` (implements `Knowledge`):**
  - `search` filters by country, applies `RAG_SCORE_THRESHOLD`, dedupes by `doc_id` and returns the passage in the lead's language. The whole search is wrapped in `RAG_TIMEOUT_SECONDS`.
  - `ingest` stores one point per language and deletes documents that are no longer in the files. A vector-dimension mismatch requires `--recreate`.
  - `build_knowledge(settings)` raises `KnowledgeConfigError` when no embedding model is configured.
- **Agent:**
  - `normalize`: a question about another market doesn't overwrite `target_country`; it sets `turn["question_country"]`.
  - `answer_question` searches with that country, then the extraction's, then one matched in the question text, then the lead's target market. It records `turn["sources"]` (source, score, doc_id, checked_on).
    - With passages, it adds an ANSWER part whose template fallback is the top passage text.
    - With none, or when the search fails or times out, it adds DEFER ("I'll confirm that for you on the call").
  - Guard: when there is knowledge or a DEFER, every number in the draft must appear in the system prompt (which holds the passages) or in the lead's message. `numbers_in` drops thousands separators. With no knowledge, any percentage is rejected.
- **`scripts/ingest_kb.py`:** `validate` / `ingest [--listings f] [--recreate]` / `search "q" --country --lang --all-scores`.
- **Tests:** `tests/test_knowledge_facts.py`, `tests/test_rag.py` and the knowledge tests in `tests/test_agent_graph.py`. They use in-memory Qdrant (`AsyncQdrantClient(location=":memory:")`) with `FakeEmbedder`.
- **Open item before production:** run `scripts/ingest_kb.py ingest` against the real Qdrant with the Gemini key, then calibrate `RAG_SCORE_THRESHOLD` with `search --all-scores`. The default of 0.6 hasn't been tested with real embeddings.

### Latency and evaluation (Phase 5)
- **Budget:** `core/latency.py` `Deadline` starts at `IncomingMessage.received_at` (webhook receipt). `AgentService(budget_seconds=LATENCY_BUDGET_SECONDS)` puts it in `config["configurable"]["deadline"]`. A non-primitive object there stays out of checkpoint metadata, but a float would not, so don't pass it as a number.
  - `AgentNodes._allow(config, turn, step, timeout, keep=)` returns the seconds a call may take, keeping `LATENCY_SEND_RESERVE_SECONDS` plus `keep` free. It returns `None` when less than `LATENCY_MIN_CALL_SECONDS` would be left: the node then skips the call, uses its fallback and records the step in `turn["budget_skips"]`.
  - Extraction and the knowledge search keep `LATENCY_REPLY_RESERVE_SECONDS`; the reply and the guard retry keep only the send reserve.
  - `LLMProvider.extract/generate` take an optional `timeout` (capped at `LLM_TIMEOUT_SECONDS`). New fakes or providers must accept it.
  - Calendar calls are never capped by the budget (a booking must not stop half-way). The hard cap is `AGENT_TIMEOUT_SECONDS` in the processor.
  - Without a budget (`budget_seconds=None`, e.g. most tests), calls get their configured timeouts.
- **Timings:** `agent/graph.py` `timed()` wraps every node and writes `turn["timings"][node]` in ms. Don't use `functools.wraps` there: LangGraph would read the inner signature and stop passing `config`. `persona_guard` sets `turn["reply_source"]` (llm / llm_retry / template).
- **Runner:** `ainvoke(..., durability="exit")` saves the checkpoint once per turn.
- **Processor:**
  - `mark_read` runs as a task alongside the work; the first send waits at most `READ_RECEIPT_HEAD_START_SECONDS` for it.
  - All bubbles are sent first, then logged in one session.
  - `latency_ms` (first bubble only) is measured after Meta accepts the send. Replies over the budget log a warning.
- **Calendar:** `CalendarEngine._busy` caches freeBusy per owner for `CALENDAR_BUSY_CACHE_SECONDS`, in process memory, with a 1-day margin on the window. `confirmed_slots` is always read live, and `book()` always re-checks live freeBusy.
  - `Scheduler.prefetch()` (sync, must not raise) is called by `qualify` when exactly one field is missing. `CalendarScheduler` runs `warm()` as a background task. Every `Scheduler` implementation needs `prefetch`.
- **Warm-up:** `services/warmup.py` runs one LLM call, knowledge search and calendar fetch at startup (`WARMUP_ON_STARTUP`).
- **Evaluation (`evals/`):** `conversations.json` holds scripted cases (one phone per case, `requires: ["llm"]` for LLM-only phrasing). `harness.py` has `run_eval`, `check_turn`, `Report` and `DemoScheduler`, which `scripts/chat_cli.py` also uses.
  - Universal checks: reply, language, persona, bubbles, grounded (every number in the text bubbles must come from the lead's messages or the turn's system prompt; the appendix is exempt), latency (budget minus send reserve) and llm (any LLM error except `TimeoutError` fails the turn, so the fallbacks can't hide a broken LLM). Nodes record failures in `turn["llm_errors"]` ("step: ErrorType").
  - `--rpm N` wraps the LLM in `RequestPacer`, which waits *between* turns until 3 more calls fit in the minute (the Gemini free tier allows 5 requests/minute per model).
  - `expect` keys: `state`, `parts`, `parts_any`, `not_parts`, `kb` (`"defer"` or a list of acceptable `doc_id`s), `language`.
  - `tests/test_evals.py` runs every offline case with rules + templates and the real facts in in-memory Qdrant (FakeEmbedder, threshold 0.25). When you change agent behaviour, add or adjust a case.
- **Embedded Qdrant:** `QDRANT_PATH` makes `tools/rag.py` `open_qdrant` use `AsyncQdrantClient(path=...)` (no server, no Docker; one process at a time; payload indexes are ignored there, which only warns). `/health` reports Qdrant ok in that mode. `.qdrant/` is git-ignored.
- **Gemini gotchas (found in the first live run, 2026-11-14):** the API rejects deadlines under 10 s, so `build_llm` passes `max(LLM_TIMEOUT_SECONDS, GEMINI_MIN_DEADLINE_SECONDS)` to the client and `LangChainLLM` enforces the real cap with `asyncio.wait_for`. `gemini-3.8-flash` rejects thinking level `minimal`, so the default is `low`.
- **Open item before production:** the first live run (2026-11-14) produced no usable numbers. Gemini answered with free-tier 429s (5 requests/minute) and then 503/504 "high demand". Re-run `scripts/eval_agent.py --kb --repeat 1` (add `--rpm 5` on the free tier), then `--repeat 5`, and tune `LATENCY_REPLY_RESERVE_SECONDS` / `GEMINI_THINKING_LEVEL` from the node p95s and the template rate.

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
- **Slots:** working hours are set in config (default Mon–Fri 09:00–18:00 in the owner's timezone, Asia/Dubai), filtered by Google freebusy, with 3 slots and by default at most 1 per day.
  - Each lead sees times in their own timezone, chosen from the phone country code: +971 → Asia/Dubai "GST"; +966/+974/+965 → Riyadh/Qatar/Kuwait "AST". Any other code falls back to the owner's timezone. This follows the phone, not the target market.
  - Every label includes the date ("Tomorrow, Tue 17 Nov at 8:00 AM AST"). Labels are rebuilt from the stored UTC ISO each time they're shown, so "Today"/"Tomorrow" always match the stored date.
- **Persona:** the agent never reveals it is an AI or a bot, in English or Arabic, and a guard checks every reply for this. The rule is the config flag `PERSONA_CONCEAL_AI`: it defaults to true, as the spec requires; setting it to false lets the agent disclose honestly when asked. It speaks as a senior property consultant. Replies are short WhatsApp messages. Arabic replies use Khaleeji dialect.
- **No hallucinated legal or tax figures:** fees, percentages and thresholds may only come from retrieved knowledge-base passages. If retrieval returns nothing above the score threshold, the agent defers to the consultant call.
- **Response time:** under 3 s from webhook to send, in both languages (spec Definition of Done). `chat_logs.latency_ms` records it. Enforced by the Phase 5 latency budget: when time runs short, the agent falls back to rules or templates rather than replying late.

### Config
All settings live in `core/config.py` (pydantic-settings, loaded from `.env`; `.env.example` lists them). List-type values are JSON in the environment, e.g. `WORKING_DAYS=[0,1,2,3,4]` with Mon=0. Inside docker-compose, the `app` service overrides `DATABASE_URL`, `REDIS_URL` and `QDRANT_URL` with container hostnames.

## Git workflow: commit and push as you go (required)
The project's state must never live only in the working tree. While you work, commit to Git and push to GitHub regularly, so no progress is lost and any step can be reverted.
- **Commit at every logical checkpoint:** a feature or module finished, a bug fixed, tests made to pass, docs or config updated. Don't let several unrelated changes pile up, and don't end a session with uncommitted work.
- **Push right after each commit:** `git push` to `origin main` (private repo `Mohsin801/gcc-realestate-whatsapp-agent`). Check that `git status -sb` shows `## main...origin/main` with nothing ahead.
- **Clean commit messages:** a subject line in the imperative mood under ~72 characters (e.g. "Add WhatsApp webhook with wamid dedup"), a blank line, then a short bulleted body saying what changed and why. One concern per commit.
- **Run the tests before committing.** If something is committed in a broken or partial state, say so in the message.
- **Never commit** `.env` or `.venv`. Both are in `.gitignore`.
- **PowerShell:** write the message to a temp file and run `git commit -F <file>`. Piping a here-string to `-F -` does not work. Write the file with the editor's Write tool, or as UTF-8 *without* a BOM. Windows PowerShell 5.1's `Set-Content -Encoding utf8` adds a BOM, and the BOM then appears at the start of the commit subject.
- `.gitattributes` keeps LF line endings because the code runs in Linux containers.
