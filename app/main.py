"""
Step 5: FastAPI layer exposing the chatbot as an HTTP API.

Endpoints:
  GET  /health          - liveness/readiness check (also confirms DB + index exist)
  POST /chat            - ask the agent a question
  POST /admin/rebuild   - (re)build the SQLite DB and/or the vector index

This file wires HTTP <-> app.agent.chat(). It contains no business logic of
its own on purpose - everything the agent needs already exists as plain
importable Python functions, so it stays testable without spinning up a
server (see app.agent, and the manual test script scripts/ask.py).
"""

from __future__ import annotations

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
    return {
        "status": "ok",
        "db_exists": config.SQLITE_DB_PATH.exists(),
        "index_exists": config.INDEX_STORAGE_DIR.exists()
        and any(config.INDEX_STORAGE_DIR.iterdir()),
        "chat_model": config.GROQ_CHAT_MODEL,
        "embed_model": config.MISTRAL_EMBED_MODEL,
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
