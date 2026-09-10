"""
Step 2: LlamaIndex retrieval over employee free-text comments.

Builds one Document per survey row that has a non-empty Comment, embeds it
with Mistral's embedding model, and persists a VectorStoreIndex to disk so
it only needs to be built once. Metadata (department, role, theme,
question, rating, date, etc.) is preserved on every node so retrieved
results can be filtered, cited, and cross-referenced with the SQL/analytics
tools.

This module is intentionally retrieval-only: it returns raw matched nodes
with their metadata, NOT an LLM-synthesized answer. Keeping retrieval and
answer-generation separate is what lets the Verification tool check "did we
actually find relevant comments" before the Groq agent is allowed to
write a final answer from them.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import pandas as pd
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.mistralai import MistralAIEmbedding

from app import config
from app.data_processing import build_dataset
from app.retry_utils import rate_limit_retry

METADATA_KEYS = [
    "Response_ID", "Company", "Respondent_Type", "Department", "Role",
    "Theme", "Question", "Rating", "Employee_Feedback", "Response_Date",
]

# Metadata that helps semantic matching but shouldn't dominate the vector
# (ids/dates/company are noise for similarity search, though still useful
# for the LLM to see and for filtering).
_EXCLUDED_FROM_EMBEDDING = ["Response_ID", "Company", "Response_Date"]

# This project's Mistral tier has a fairly low embeddings rate limit, so the
# index is built in small, paced, retryable batches rather than one bulk
# from_documents() call (which has no backoff and dies on the first 429).
_INDEX_INSERT_BATCH_SIZE = 20
_INDEX_BATCH_PAUSE_SECONDS = 1.5

_index_cache: VectorStoreIndex | None = None


@rate_limit_retry
def _insert_batch(index: VectorStoreIndex, nodes_batch: list[Document]) -> None:
    index.insert_nodes(nodes_batch)


def _configure_settings() -> None:
    # No Settings.llm here on purpose: this module only ever retrieves
    # (index.as_retriever().retrieve(...)), which needs embed_model only.
    # Answer generation from retrieved comments is Groq's job, in app.agent.
    Settings.embed_model = MistralAIEmbedding(
        model_name=config.MISTRAL_EMBED_MODEL, api_key=config.MISTRAL_API_KEY
    )


def _rows_to_documents(df: pd.DataFrame) -> list[Document]:
    documents = []
    for _, row in df.iterrows():
        comment = str(row["Comment"]).strip()
        if not comment:
            continue
        metadata = {key: row[key] for key in METADATA_KEYS}
        documents.append(
            Document(
                text=f"[{row['Theme']}] {comment}",
                metadata=metadata,
                excluded_embed_metadata_keys=_EXCLUDED_FROM_EMBEDDING,
                excluded_llm_metadata_keys=[],
                id_=str(row["Response_ID"]),
            )
        )
    return documents


def build_or_load_index(
    force_rebuild: bool = False, limit: int | None = None
) -> VectorStoreIndex:
    """Load the persisted index from disk, or build + persist it if it
    doesn't exist yet (or force_rebuild=True).

    `limit`: only index the first N rows with a comment - useful for a fast
    smoke test of the whole pipeline before paying for/waiting on embedding
    all 5000 rows. Never used when loading an already-persisted index.
    """
    global _index_cache
    if _index_cache is not None and not force_rebuild:
        return _index_cache

    _configure_settings()
    storage_dir = config.INDEX_STORAGE_DIR

    has_persisted_index = storage_dir.exists() and any(storage_dir.iterdir())
    if has_persisted_index and not force_rebuild:
        storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
        _index_cache = load_index_from_storage(storage_context)
        return _index_cache

    df, _ = build_dataset(force_rebuild=False)
    if limit is not None:
        df = df.head(limit)
    documents = _rows_to_documents(df)
    if not documents:
        raise RuntimeError("No documents with non-empty Comment text were found to index.")

    # llama_index.core.Document IS-A TextNode, so it can be embedded/inserted
    # directly via insert_nodes() without a separate node-parsing pass -
    # these comments are short enough that no chunking is needed anyway.
    index = VectorStoreIndex(nodes=[])
    total = len(documents)
    for i in range(0, total, _INDEX_INSERT_BATCH_SIZE):
        batch = documents[i : i + _INDEX_INSERT_BATCH_SIZE]
        _insert_batch(index, batch)
        done = min(i + _INDEX_INSERT_BATCH_SIZE, total)
        print(f"  indexed {done}/{total} documents", file=sys.stderr)
        if done < total:
            time.sleep(_INDEX_BATCH_PAUSE_SECONDS)

    storage_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_dir))

    _index_cache = index
    return index


@dataclass
class RetrievedComment:
    response_id: str
    text: str
    score: float | None
    department: str
    role: str
    theme: str
    question: str
    rating: int
    employee_feedback: str
    response_date: str

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class RetrievalResult:
    ok: bool
    query: str
    results: list[dict] = field(default_factory=list)
    count: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "query": self.query,
            "results": self.results, "count": self.count, "error": self.error,
        }


def _matches_filters(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        node_value = metadata.get(key)
        if node_value is None or str(node_value).lower() != str(value).lower():
            return False
    return True


def search_employee_comments(
    query: str, top_k: int = 8, filters: dict | None = None
) -> RetrievalResult:
    """Retrieve the most semantically relevant employee comments for
    `query`, optionally restricted to exact-match metadata filters such as
    {"Department": "Finance"} or {"Theme": "Manager Support"}.
    """
    try:
        index = build_or_load_index()
    except Exception as exc:  # noqa: BLE001 - surface as a tool-level error
        return RetrievalResult(ok=False, query=query, error=str(exc))

    try:
        # Over-fetch, then apply metadata filters in Python, then trim -
        # keeps filtering logic simple and version-independent.
        retriever = index.as_retriever(similarity_top_k=max(top_k * 4, 20))
        nodes = retriever.retrieve(query)
    except Exception as exc:  # noqa: BLE001
        return RetrievalResult(ok=False, query=query, error=f"Retrieval failed: {exc}")

    matched = [n for n in nodes if _matches_filters(n.metadata, filters)][:top_k]

    results = []
    for n in matched:
        md = n.metadata
        results.append(
            RetrievedComment(
                response_id=str(md.get("Response_ID")),
                text=_strip_theme_prefix(n.get_content()),
                score=float(n.score) if n.score is not None else None,
                department=md.get("Department", ""),
                role=md.get("Role", ""),
                theme=md.get("Theme", ""),
                question=md.get("Question", ""),
                rating=int(md.get("Rating", 0)),
                employee_feedback=md.get("Employee_Feedback", ""),
                response_date=str(md.get("Response_Date", "")),
            ).as_dict()
        )

    return RetrievalResult(ok=True, query=query, results=results, count=len(results))


def _strip_theme_prefix(text: str) -> str:
    """Documents are stored as '[Theme] comment text' to help embedding
    relevance; strip that prefix back off before showing the raw comment
    to the LLM/user."""
    if text.startswith("[") and "]" in text:
        return text.split("]", 1)[1].strip()
    return text
