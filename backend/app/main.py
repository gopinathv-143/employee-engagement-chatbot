"""
Step 5: FastAPI layer exposing the chatbot as an HTTP API.

Endpoints:
  GET  /health              - liveness/readiness check (also confirms DB + index exist)
  POST /chat                - ask the agent a question, block until answered
  POST /chat/jobs           - ask the agent a question, return immediately with a job_id
  GET  /chat/jobs/{job_id}  - poll a job started via POST /chat/jobs
  POST /admin/rebuild       - (re)build the SQLite DB and/or the vector index

This file wires HTTP <-> app.agent.chat(). It contains no business logic of
its own on purpose - everything the agent needs already exists as plain
importable Python functions, so it stays testable without spinning up a
server (see app.agent, and the manual test script scripts/ask.py).

Why both a blocking /chat and a job-based /chat/jobs: a multi-tool question
(e.g. retrieval + sentiment) makes several sequential Groq calls, each
wrapped in its own rate-limit retry/backoff (see retry_utils.py) - under
sustained rate limiting that can legitimately take minutes even though it
eventually succeeds. A single HTTP request with a client-side timeout has no
way to distinguish "still working" from "hung", so any fixed timeout short
enough to fail fast on a real hang will also cut off slow-but-succeeding
turns. The job endpoints split "start the work" from "check on it": each
poll is a cheap, near-instant request, so the client can wait as long as it
wants without needing a long-lived connection or a compromise timeout value.
The frontend (frontend/app.py) uses the job endpoints; /chat is kept for
direct/simple callers (scripts, curl) that just want one blocking call.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import agent, config, data_processing, indexing


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Optional prior turns, oldest first, as {role, content}.",
    )


class ChatResponse(BaseModel):
    answer: str
    tool_trace: list[dict]
    iterations_used: int
    gave_up: bool


class ChatJobCreated(BaseModel):
    job_id: str
    status: str


class ChatJobStatus(BaseModel):
    status: str  # "running" | "done" | "error"
    result: ChatResponse | None = None
    error: str | None = None


# In-memory job store for the async /chat/jobs flow. A plain dict guarded by
# a lock is enough for a single-process POC server - each job's actual work
# runs in its own background thread (agent.chat() manages its own asyncio
# event loop internally via asyncio.run(), so it must NOT be awaited on the
# server's own event loop; a separate thread sidesteps that entirely).
_chat_jobs: dict[str, dict] = {}
_chat_jobs_lock = threading.Lock()
_CHAT_JOB_TTL_SECONDS = 30 * 60


def _prune_stale_chat_jobs() -> None:
    cutoff = time.time() - _CHAT_JOB_TTL_SECONDS
    stale_ids = [jid for jid, job in _chat_jobs.items() if job["created_at"] < cutoff]
    for jid in stale_ids:
        del _chat_jobs[jid]


class RebuildRequest(BaseModel):
    rebuild_db: bool = True
    rebuild_index: bool = False
    index_limit: int | None = Field(
        default=None,
        description="Optional cap on rows to index, for a fast smoke test.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the SQLite DB exists at startup so the very first chat request
    # doesn't pay the one-time ingestion cost. The vector index is loaded
    # lazily on first retrieval call (it may not exist yet in a fresh
    # checkout until scripts/build_index.py has been run once).
    try:
        data_processing.build_dataset(force_rebuild=False)
    except FileNotFoundError as exc:
        print(f"[startup warning] {exc}")
    yield


app = FastAPI(
    title="Employee Engagement Chatbot (POC)",
    description=(
        "Groq-orchestrated agent over an employee engagement survey: "
        "SQL analytics + LlamaIndex retrieval + sentiment, all verified "
        "before answering."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    db_exists = config.SQLITE_DB_PATH.exists()
    response_count = None
    if db_exists:
        # Cheap enough to read on every health check (single COUNT(*) on an
        # indexed table) - lets the frontend show a live "N responses"
        # figure without a dedicated endpoint.
        try:
            conn = sqlite3.connect(config.SQLITE_DB_PATH)
            try:
                response_count = conn.execute(
                    f"SELECT COUNT(*) FROM {config.SQL_TABLE_NAME}"
                ).fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error:
            response_count = None

    return {
        "status": "ok",
        "db_exists": db_exists,
        "index_exists": config.INDEX_STORAGE_DIR.exists()
        and any(config.INDEX_STORAGE_DIR.iterdir()),
        "chat_model": config.GROQ_CHAT_MODEL,
        "embed_model": config.EMBED_MODEL_NAME,
        "response_count": response_count,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    history = [m.model_dump() for m in request.history]
    result = agent.chat(user_message=request.message, history=history)
    return ChatResponse(
        answer=result.answer,
        tool_trace=result.tool_trace,
        iterations_used=result.iterations_used,
        gave_up=result.gave_up,
    )


@app.post("/chat/jobs", response_model=ChatJobCreated)
def create_chat_job(request: ChatRequest) -> ChatJobCreated:
    """Start one agent turn in a background thread and return immediately.
    Poll GET /chat/jobs/{job_id} for the result - see the module docstring
    for why this exists alongside the blocking /chat."""
    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    history = [m.model_dump() for m in request.history]
    job_id = uuid.uuid4().hex

    with _chat_jobs_lock:
        _prune_stale_chat_jobs()
        _chat_jobs[job_id] = {
            "status": "running", "result": None, "error": None,
            "created_at": time.time(),
        }

    def _run_job() -> None:
        try:
            result = agent.chat(user_message=request.message, history=history)
            response = ChatResponse(
                answer=result.answer,
                tool_trace=result.tool_trace,
                iterations_used=result.iterations_used,
                gave_up=result.gave_up,
            )
            with _chat_jobs_lock:
                _chat_jobs[job_id]["status"] = "done"
                _chat_jobs[job_id]["result"] = response
        except Exception as exc:  # noqa: BLE001 - surface to the poller, not a crashed thread
            with _chat_jobs_lock:
                _chat_jobs[job_id]["status"] = "error"
                _chat_jobs[job_id]["error"] = str(exc)

    threading.Thread(target=_run_job, daemon=True).start()
    return ChatJobCreated(job_id=job_id, status="running")


@app.get("/chat/jobs/{job_id}", response_model=ChatJobStatus)
def get_chat_job(job_id: str) -> ChatJobStatus:
    with _chat_jobs_lock:
        job = _chat_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id.")
    return ChatJobStatus(status=job["status"], result=job["result"], error=job["error"])


@app.post("/admin/rebuild")
def rebuild(request: RebuildRequest) -> dict:
    """Manual trigger to rebuild the DB and/or index without restarting the
    process. Not authenticated - fine for a local POC, would need auth
    before this ever went near a shared environment."""
    report = {}
    if request.rebuild_db:
        df, cleaning_report = data_processing.build_dataset(force_rebuild=True)
        report["db_rows"] = len(df)
        report["cleaning_report"] = (
            cleaning_report.as_dict() if cleaning_report else "unchanged (was cached)"
        )
    if request.rebuild_index:
        index = indexing.build_or_load_index(force_rebuild=True, limit=request.index_limit)
        report["index_documents"] = len(index.docstore.docs)
    return report
