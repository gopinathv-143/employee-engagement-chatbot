# Employee Engagement Chatbot (POC)

A small, modular chatbot over an employee engagement survey dataset. Mistral
orchestrates tool calls; every tool is a plain, independently-testable
Python function; every tool result is verified in code before the agent is
allowed to answer.

## Architecture

```
Excel (.xlsx) --> pandas clean --> SQLite ---------------\
                                     |                     \
                                     |                      >--- Mistral Agent (tool-calling loop)
                                     v                      /            |
                              LlamaIndex (Mistral            |            v
                              embeddings, over               |     Verification (rule-based,
                              employee Comments) ------------/      runs after every tool call)
                                                                            |
                                                                            v
                                                                   Final natural-language
                                                                   answer (FastAPI /chat)
```

Tools available to the agent (`app/agent.py`, `TOOL_SCHEMAS`):

| Tool                        | Backed by                                                   | Used for |
|------------------------------|--------------------------------------------------------------|----------|
| `query_database`             | `app/tools/sql_generator.py` + `app/tools/sql_executor.py`   | counts, lookups, group-bys |
| `run_analytics`               | `app/tools/analytics.py` (pure pandas)                        | percentages, averages, distributions, trends |
| `search_employee_comments`     | `app/tools/retrieval.py` -> `app/indexing.py` (LlamaIndex)     | "what do employees say about X" |
| `analyze_sentiment`             | `app/tools/sentiment.py`                                        | Positive/Neutral/Negative on retrieved comments |

`app/tools/verification.py` is **not** an LLM-selectable tool. It runs
automatically, in Python, right after every tool call (see
`app/agent.py::_run_tool`) - verification is not something the model can
choose to skip.

## Project layout

```
employee-engagement-chatbot/
  .env                     # real secrets (git-ignored) - already filled in for this machine
  .env.example             # template for a fresh checkout
  requirements.txt
  pytest.ini
  data/employee_engagement_5000.xlsx
  db/engagement.db          # created by scripts/build_db.py
  storage/                   # created by scripts/build_index.py (LlamaIndex persisted vectors)
  app/
    config.py                 # env/config loading
    mistral_client.py          # shared Mistral client + retry-wrapped chat_complete()
    retry_utils.py              # shared 429 retry/backoff policy
    data_processing.py           # Step 1: ingest + clean + SQLite
    indexing.py                   # Step 2: LlamaIndex build/load + retrieval
    agent.py                       # Step 4: tool-calling orchestration loop
    main.py                         # Step 5: FastAPI app
    tools/
      sql_generator.py               # NL -> SQL (Mistral)
      sql_executor.py                 # safe SQL execution (sqlite3, no LLM)
      query_database.py                # composes the two above + 1 auto-retry
      analytics.py                      # deterministic pandas metrics (no LLM)
      retrieval.py                       # thin wrapper over app.indexing
      sentiment.py                        # batched Mistral sentiment classifier
      verification.py                      # rule-based checks (no LLM)
  scripts/
    build_db.py                # run Step 1
    build_index.py              # run Step 2
    ask.py                        # talk to the agent from the terminal
  tests/                           # pytest; only test_sentiment/test_query_database need mocking
                                      (no real API calls in the test suite)
```

## Setup (already done on this machine, included for a fresh checkout)

```powershell
cd employee-engagement-chatbot
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
# then edit .env and paste in a real MISTRAL_API_KEY from https://console.mistral.ai/
```

**A note on this specific API key's tier**, discovered while building this:
`mistral-large-latest` returned `403 tier_not_allowed`, and both chat and
embedding calls hit `429 rate_limited` under sustained use. `.env` is set to
`mistral-small-latest` (available on this tier), and every Mistral call in
this codebase goes through `app/retry_utils.py`'s exponential backoff
(`app/mistral_client.chat_complete`, and `app/indexing.py`'s batched index
inserts) so transient 429s are absorbed automatically instead of crashing a
request. If you use a different key/tier, you can safely raise
`MISTRAL_CHAT_MODEL` to `mistral-large-latest` or `mistral-medium-latest` in
`.env`.

## Step 1 - Data ingestion (`app/data_processing.py`, `scripts/build_db.py`)

**What**: reads the Excel export with pandas, validates/cleans it (parses
dates, coerces Rating to 1-5, drops rows missing required fields, dedupes by
`Response_ID`), and writes the result into `db/engagement.db` as the single
`engagement` table every other tool reads from.

**Why**: every downstream tool (SQL, analytics, indexing) needs one
guaranteed-clean source of truth. Cleaning happens exactly once here rather
than defensively, differently, in every tool.

**Run it**:
```powershell
.venv\Scripts\python scripts\build_db.py
```
Expected output on this dataset: `5000 rows in, 5000 rows out, 0 dropped`
(this dataset happens to be already clean), 36 departments, 15 themes.

**Test it** (no API key needed - pure pandas against a small synthetic
dataframe):
```powershell
.venv\Scripts\pytest tests\test_data_processing.py -v
```

## Step 2 - LlamaIndex retrieval (`app/indexing.py`, `scripts/build_index.py`)

**What**: builds one `Document` per survey row with a non-empty `Comment`
(all 5000, in this dataset), embeds them with `mistral-embed`, and persists
a `VectorStoreIndex` to `storage/`. Metadata (Department, Role, Theme,
Question, Rating, Employee_Feedback, Response_Date) rides along on every
node for filtering and citation. `search_employee_comments()` returns raw
matched comments + metadata - never an LLM-written summary - so
Verification can check "did we actually find something relevant" before the
agent is allowed to describe it.

**Why it's built the way it is**: this Mistral tier's embeddings rate limit
is low enough that a naive bulk `VectorStoreIndex.from_documents(...)` call
dies partway through with a `429`. Indexing is instead done in small paced
batches (20 documents at a time, 1.5s apart) with exponential-backoff retry
on any 429 (`app/retry_utils.py`), so the full 5000-row build completes
reliably even on this tier - it just takes a few minutes.

**Run it**:
```powershell
.venv\Scripts\python scripts\build_index.py --limit 50   # fast smoke test first
.venv\Scripts\python scripts\build_index.py --force       # then the full 5000-row build
```

**Test it**: retrieval logic that doesn't need a live index (metadata
filtering, theme-prefix stripping) is covered indirectly through
`app/tools/verification.py`'s tests. The real end-to-end check is manual,
since it needs the actual embedding index:
```powershell
.venv\Scripts\python scripts\ask.py --trace "What do employees say about management support?"
```

## Step 3 - Tools (`app/tools/*.py`)

Each tool is a plain function/dataclass with no dependency on the agent or
FastAPI, so each is unit-tested in isolation:

```powershell
.venv\Scripts\pytest tests\test_sql_executor.py -v      # safety checks + real sqlite execution
.venv\Scripts\pytest tests\test_analytics.py -v           # percentage/average/trend math
.venv\Scripts\pytest tests\test_verification.py -v          # pass/fail rules for every tool shape
.venv\Scripts\pytest tests\test_sentiment.py -v               # JSON parsing, mocked Mistral call
.venv\Scripts\pytest tests\test_query_database.py -v            # generator+executor compose/retry, mocked
```

Or all of them (this whole suite makes zero network calls - safe to run
anytime, no API key needed):
```powershell
.venv\Scripts\pytest
```

Notable design choices:
- **`query_database`** is the single agent-facing tool that composes
  `sql_generator.generate_sql()` (Mistral writes SQL, grounded in the real
  schema + few-shot examples) and `sql_executor.run_sql()` (sqlite3,
  read-only connection, SELECT-only allow-list, forbidden-keyword
  block-list, hard row cap). If execution fails, `query_database` feeds the
  error back into `generate_sql()` once for a self-corrected retry before
  giving up - this is the "retry/correct failed SQL" requirement.
- **`run_analytics`** exists *alongside* `query_database`, not instead of
  it, specifically for percentage/average/trend questions - those are
  computed deterministically in pandas rather than trusted to freeform SQL
  every time, which is both more reliable and cheaper.
- **`verification.py`** is deliberately rule-based, not another LLM call:
  it needs to be fast, free, and 100% consistent, and it must run on every
  tool result unconditionally (see `app/agent.py::_run_tool`), not only
  when the model decides to check.

## Step 4 - The Mistral agent (`app/agent.py`)

**What**: the tool-calling loop. Mistral picks a tool + arguments -> Python
executes it -> Python verifies it -> the raw result *and* the verification
outcome go back to Mistral -> repeat (up to `AGENT_MAX_TOOL_ITERATIONS`,
default 5) until Mistral has enough verified data to answer in plain
language, or the loop gives up and says so honestly.

**Run it** (interactive, prints the full tool trace with `--trace`):
```powershell
.venv\Scripts\python scripts\ask.py --trace "How many survey responses are there?"
.venv\Scripts\python scripts\ask.py --trace "What percentage of responses are Very Dissatisfied?"
.venv\Scripts\python scripts\ask.py --trace "What do employees say about compensation?"
.venv\Scripts\python scripts\ask.py                       # interactive loop, no single question
```

Read the `TOOL TRACE` output to see exactly which tool(s) ran, with what
arguments, what came back, and whether verification passed - this is the
main way to debug/extend the agent.

## Step 5 - FastAPI (`app/main.py`)

**What**: `POST /chat {"message": "..."}` -> `{"answer", "tool_trace",
"iterations_used", "gave_up"}`. `GET /health` reports whether the DB/index
exist and which models are configured. `POST /admin/rebuild` re-runs
ingestion/indexing without restarting the process.

**Run it**:
```powershell
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```
Then, in another terminal:
```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"How many survey responses are there?\"}"
```
Or open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

## Troubleshooting: "429 Rate limit exceeded" on every chat call

If `scripts/ask.py`, `/chat`, or any tool that calls Mistral chat completion
fails with a `429` **even for a single bare request with nothing else
running**, check the response headers before assuming it's transient
traffic:

```powershell
.venv\Scripts\python -c "
from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError
from app import config
c = Mistral(api_key=config.MISTRAL_API_KEY)
try:
    c.chat.complete(model='mistral-small-latest', messages=[{'role':'user','content':'hi'}])
    print('SUCCESS')
except SDKError as e:
    print(e.status_code, dict(e.headers))
"
```

If the response includes `x-ratelimit-limit-req-minute: 0`, that is **not**
a transient burst limit that backoff/retry can wait out - it means chat
completions are allowed **zero requests per minute** for this API
key/workspace, account-wide, regardless of which model you pick. This
happened on the key used while building this POC: `mistral-embed`
(embeddings) worked perfectly the whole time (see Step 2 - the full
5000-row index built successfully), but every chat-completion call to
`mistral-large-latest` (403 `tier_not_allowed`) and `mistral-small-latest`
(429, limit-req-minute=0) failed, because this workspace's chat completions
access hadn't been enabled/billed, even though `models.list()` still listed
those models as available.

**Fix**: log into https://console.mistral.ai/, check Billing/Usage and the
workspace's rate limits, and enable chat completions (a payment method is
typically required even for light usage - the embeddings-only free
allowance is separate from the chat-completions allowance). Once
`x-ratelimit-limit-req-minute` is non-zero for your key, everything in
`app/agent.py`, `app/tools/sql_generator.py`, and `app/tools/sentiment.py`
will work as-is with no code changes - the tool-calling message/schema
shapes were verified directly against the installed `mistralai==2.9.4` SDK,
and all 43 tests in `tests/` (which mock the chat-completion call) already
pass. Re-run `scripts/ask.py` to confirm live.

## Known limitations (intentional, for a POC)

- Single in-process SQLite file, no concurrent-write handling - fine for a
  read-mostly analytics chatbot, not for a multi-writer production system.
- `/admin/rebuild` has no auth - would need it before this went anywhere
  shared.
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
