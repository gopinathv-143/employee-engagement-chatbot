# Employee Engagement Chatbot (POC)

A small, modular chatbot over an employee engagement survey dataset. A Groq
model orchestrates tool calls; every tool is a plain, independently-testable
Python function; every tool result is verified in code before the agent is
allowed to answer.

## Architecture

```
Excel (.xlsx) --> pandas clean --> SQLite ---------------\
                                     |                     \
                                     |                      >--- Groq Agent (tool-calling loop)
                                     v                      /            |
                              LlamaIndex (local              |            v
                              HF embeddings, over             |     Verification (rule-based,
                              employee Comments) -------------/      runs after every tool call)
                                                                            |
                                                                            v
                                                                   Final natural-language
                                                                   answer (FastAPI /chat)
```

Tools available to the agent (`backend/app/agent.py`, `TOOL_SCHEMAS`):

| Tool                        | Backed by                                                   | Used for |
|------------------------------|--------------------------------------------------------------|----------|
| `query_database`             | `sql_generator.py` + `sql_executor.py`                       | counts, lookups, group-bys, one-off SQL |
| `run_analytics`               | `analytics.py` (pure pandas)                                  | percentages, averages, distributions, trends |
| `search_employee_comments`     | `retrieval.py` -> `indexing.py` (LlamaIndex)                    | "what do employees say about X" |
| `analyze_sentiment`             | `sentiment.py`                                                    | Positive/Neutral/Negative on retrieved comments |

`verification.py` is **not** an LLM-selectable tool. It runs automatically,
in Python, right after every tool call (see `agent.py::_run_tool`) -
verification is not something the model can choose to skip.

`schema_index.py` is a small semantic resolver: it embeds every "closed
vocabulary" column's real distinct values (Question, Theme, Department,
Role, ...) once, and ranks them by similarity to whatever the user actually
typed. Both `sql_generator.py` (as grounding candidates in the prompt) and
`analytics.py`/`indexing.py` (as a filter-matching fallback) use it so a
paraphrase like "the finance folks" or "pay fairness" resolves to the real
stored value instead of silently matching nothing.

## Project layout

```
employee-engagement-chatbot/
  data/
    employee_engagement_5000.xlsx   # raw source data (shared - not backend- or frontend-owned)

  backend/
    .env                    # real secrets (git-ignored)
    .env.example             # template for a fresh checkout
    requirements.txt
    pytest.ini
    db/engagement.db          # created by scripts/build_db.py
    storage/                   # created by scripts/build_index.py (LlamaIndex persisted vectors)
    app/
      config.py                 # env/config loading
      groq_client.py              # shared Groq client + retry-wrapped chat_complete()
      retry_utils.py               # shared 429 retry/backoff policy
      data_processing.py            # Step 1: ingest + clean + SQLite
      indexing.py                    # Step 2: LlamaIndex build/load + retrieval
      agent.py                        # Step 4: LlamaIndex Workflow tool-calling orchestration
      main.py                          # Step 5: FastAPI app
      tools/
        schema_index.py                 # semantic value resolver (paraphrase -> real DB value)
        sql_generator.py                 # NL -> SQL (Groq, grounded via schema_index)
        sql_executor.py                   # safe SQL execution (sqlite3, no LLM)
        query_database.py                  # composes the two above + 1 auto-retry
        analytics.py                        # deterministic pandas metrics (no LLM)
        retrieval.py                         # thin wrapper over app.indexing
        sentiment.py                          # batched Groq sentiment classifier
        verification.py                        # rule-based checks (no LLM)
    scripts/
      build_db.py                # run Step 1
      build_index.py              # run Step 2
      ask.py                        # talk to the agent from the terminal
    tests/                           # pytest; no real API calls anywhere in the suite

  frontend/
    app.py                    # Streamlit UI - talks to the backend only over HTTP
    requirements.txt
```

`data/`, `backend/`, and `frontend/` are deliberately separate: `data/` is
raw input owned by neither side, `backend/` is the FastAPI + agent service,
`frontend/` is a thin HTTP client with zero import dependency on `backend/`
code. One shared `.venv` at the repo root covers both (see Setup below) -
split into two only if you want to deploy them independently.

## Setup

```powershell
cd employee-engagement-chatbot
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt -r frontend\requirements.txt
copy backend\.env.example backend\.env
# then edit backend\.env and paste in a real GROQ_API_KEY from https://console.groq.com/keys
```

## Step 1 - Data ingestion (`backend/app/data_processing.py`, `backend/scripts/build_db.py`)

**What**: reads the Excel export with pandas, validates/cleans it (parses
dates, coerces Rating to 1-5, drops rows missing required fields, dedupes by
`Response_ID`), and writes the result into `backend/db/engagement.db` as the
single `engagement` table every other tool reads from.

**Why**: every downstream tool (SQL, analytics, indexing) needs one
guaranteed-clean source of truth. Cleaning happens exactly once here rather
than defensively, differently, in every tool.

**Run it** (from `backend/`):
```powershell
cd backend
..\.venv\Scripts\python scripts\build_db.py
```
Expected output on this dataset: `5000 rows in, 5000 rows out, 0 dropped`
(this dataset happens to be already clean), 36 departments, 15 themes.

**Test it** (no API key needed - pure pandas against a small synthetic
dataframe):
```powershell
..\.venv\Scripts\pytest tests\test_data_processing.py -v
```

## Step 2 - LlamaIndex retrieval (`backend/app/indexing.py`, `backend/scripts/build_index.py`)

**What**: builds one `Document` per survey row with a non-empty `Comment`
(all 5000, in this dataset), embeds them locally with a HuggingFace/
sentence-transformers model (`BAAI/bge-small-en-v1.5` by default - no API
key, no external service, no rate limit), and persists a `VectorStoreIndex`
to `backend/storage/`. Metadata (Department, Role, Theme, Question, Rating,
Employee_Feedback, Response_Date) rides along on every node for filtering
and citation. `search_employee_comments()` returns raw matched comments +
metadata - never an LLM-written summary - so Verification can check "did we
actually find something relevant" before the agent is allowed to describe
it.

**Run it** (from `backend/`):
```powershell
..\.venv\Scripts\python scripts\build_index.py --limit 50   # fast smoke test first
..\.venv\Scripts\python scripts\build_index.py --force       # then the full 5000-row build
```

**Test it**: retrieval logic that doesn't need a live index (metadata
filtering, theme-prefix stripping) is covered indirectly through
`verification.py`'s tests. The real end-to-end check is manual, since it
needs the actual embedding index:
```powershell
..\.venv\Scripts\python scripts\ask.py --trace "What do employees say about management support?"
```

## Step 3 - Tools (`backend/app/tools/*.py`)

Each tool is a plain function/dataclass with no dependency on the agent or
FastAPI, so each is unit-tested in isolation. From `backend/`:

```powershell
..\.venv\Scripts\pytest tests\test_sql_executor.py -v      # safety checks + real sqlite execution
..\.venv\Scripts\pytest tests\test_analytics.py -v           # percentage/average/trend math
..\.venv\Scripts\pytest tests\test_verification.py -v          # pass/fail rules for every tool shape
..\.venv\Scripts\pytest tests\test_sentiment.py -v               # JSON parsing, mocked Groq call
..\.venv\Scripts\pytest tests\test_query_database.py -v            # generator+executor compose/retry, mocked
```

Or all of them (this whole suite makes zero network calls - safe to run
anytime, no API key needed):
```powershell
..\.venv\Scripts\pytest
```

Notable design choices:
- **`query_database`** is the single agent-facing tool that composes
  `sql_generator.generate_sql()` (Groq writes SQL, grounded in the real
  schema, few-shot examples, and `schema_index`'s semantic candidates) and
  `sql_executor.run_sql()` (sqlite3, read-only connection, SELECT-only
  allow-list, forbidden-keyword block-list, hard row cap). If execution
  fails, `query_database` feeds the error back into `generate_sql()` once
  for a self-corrected retry before giving up.
- **`run_analytics`** exists *alongside* `query_database`, not instead of
  it, specifically for percentage/average/trend questions - those are
  computed deterministically in pandas rather than trusted to freeform SQL
  every time, which is both more reliable and cheaper. It has no
  correlation/statistical/predictive capability - see Known limitations.
- **`schema_index`** exists because paraphrases ("pay fairness", "the
  finance folks") almost never match a stored Question/Theme/Department/
  Role value verbatim. It embeds each column's real distinct values once
  and is used both as SQL-generation grounding and as an exact-match
  fallback in analytics/retrieval filters - above a calibrated confidence
  threshold it auto-resolves, below it, it reports the mismatch with
  candidates rather than guessing.
- **`verification.py`** is deliberately rule-based, not another LLM call:
  it needs to be fast, free, and 100% consistent, and it must run on every
  tool result unconditionally (see `agent.py::_run_tool`), not only when
  the model decides to check.

## Step 4 - The Groq agent (`backend/app/agent.py`)

**What**: a LlamaIndex `Workflow` (event-driven, not a plain loop - see the
module docstring for the event diagram). Groq picks a tool + arguments ->
Python executes it -> Python verifies it -> the raw result *and* the
verification outcome go back to Groq -> repeat (up to
`AGENT_MAX_TOOL_ITERATIONS`, default 5) until Groq has enough verified data
to answer in plain language, or the loop gives up and says so honestly.

**Run it** (interactive, prints the full tool trace with `--trace`, from `backend/`):
```powershell
..\.venv\Scripts\python scripts\ask.py --trace "How many survey responses are there?"
..\.venv\Scripts\python scripts\ask.py --trace "What percentage of responses are Very Dissatisfied?"
..\.venv\Scripts\python scripts\ask.py --trace "What do employees say about compensation?"
..\.venv\Scripts\python scripts\ask.py                       # interactive loop, no single question
```

Read the `TOOL TRACE` output to see exactly which tool(s) ran, with what
arguments, what came back, and whether verification passed - this is the
main way to debug/extend the agent.

## Step 5 - FastAPI (`backend/app/main.py`)

**What**: `POST /chat {"message": "..."}` -> `{"answer", "tool_trace",
"iterations_used", "gave_up"}`. `GET /health` reports whether the DB/index
exist, which models are configured, and the live response count. `POST
/admin/rebuild` re-runs ingestion/indexing without restarting the process.

**Run it** (from `backend/`):
```powershell
cd backend
..\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```
Then, in another terminal:
```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"How many survey responses are there?\"}"
```
Or open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

## Frontend (`frontend/app.py`)

A Streamlit UI over `/health` and `/chat` - no business logic, just
presentation (SQL/tables/charts per tool call, verification badges, a
headline metric/chart surfaced immediately below each answer). Run it
(from `frontend/`, with the backend already running on port 8000):
```powershell
cd frontend
..\.venv\Scripts\streamlit run app.py
```

## Known limitations (intentional, for a POC)

- Single in-process SQLite file, no concurrent-write handling - fine for a
  read-mostly analytics chatbot, not for a multi-writer production system.
- `/admin/rebuild` has no auth - would need it before this went anywhere
  shared.
- No correlation, statistical, or predictive capability - `run_analytics`
  computes percentages/averages/distributions/trends, nothing that answers
  "is X connected to Y" or "what if we changed Z". Scenario questions that
  need that get an honest "couldn't produce a verified answer" rather than
  a fabricated one.
- The dataset has no separate "employee ID" column - each row is one
  survey-question response, so "employee count" in this POC really means
  "response count" (see the schema note in
  `data_processing.get_schema_description()`).
- Sentiment classification is batched per request, not pre-computed and
  cached - fine at the "classify these 8-20 retrieved comments" scale the
  agent operates at, but would need caching/pre-computation to run over the
  full 5000 rows repeatedly.
- No conversation persistence - `/chat` accepts an optional `history` list
  per request, but nothing is stored server-side between requests.
