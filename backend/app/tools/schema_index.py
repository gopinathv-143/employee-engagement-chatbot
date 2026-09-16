"""
Semantic schema resolver.

Free-text questions rarely quote the database's stored category/question
values verbatim - "pay fairness" vs the real Question text "How fair do you
consider your compensation compared with your responsibilities?", or
"the accounts team" vs the real Department value "Finance". Both the SQL
generator and the analytics filter matcher previously had to guess those
exact strings blind (via exact/case-insensitive equality or trial-and-error
LIKE queries), burning agent retries - or failing outright - whenever the
guess didn't match.

Every "closed vocabulary" column in this dataset (Question, Theme,
Department, Role, Employee_Feedback, Respondent_Type, Company) has a small,
fixed set of distinct values (see data_processing.py). This module embeds
each column's distinct values once, with the same local embedding model
already used for comment retrieval, and exposes resolve() to rank the real
stored values by similarity to a piece of free text. Callers use the top
match(es) as grounding/candidates instead of inventing text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app import config
from app.data_processing import build_dataset

SEMANTIC_COLUMNS = [
    "Question", "Theme", "Department", "Role",
    "Employee_Feedback", "Respondent_Type", "Company",
]

# Shared confidence floor for auto-substituting a filter value with its
# closest semantic match, used by every caller that does exact-match-then-
# semantic-fallback filtering (analytics.py, indexing.py). Calibrated
# against real vs. bogus paraphrases of these columns' values: genuine
# paraphrases ("pay and benefits" -> 'Compensation & Benefits', "the
# finance folks" -> 'Finance') scored 0.72-0.86 cosine similarity; an
# ambiguous/wrong guess ("accounts team" -> 'Sales Support & Operations')
# scored 0.66. Below this floor, callers should report the mismatch
# instead of guessing.
DEFAULT_MATCH_THRESHOLD = 0.72

_embed_model_cache: HuggingFaceEmbedding | None = None
_column_index_cache: dict[str, tuple[list[str], np.ndarray]] = {}


def _get_embed_model() -> HuggingFaceEmbedding:
    global _embed_model_cache
    if _embed_model_cache is None:
        _embed_model_cache = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)
    return _embed_model_cache


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _build_column_index(column: str) -> tuple[list[str], np.ndarray]:
    df, _ = build_dataset(force_rebuild=False)
    values = sorted(v for v in df[column].astype(str).unique() if v.strip())
    if not values:
        return values, np.zeros((0, 0))
    vectors = np.array(_get_embed_model().get_text_embedding_batch(values))
    return values, _normalize(vectors)


def reset_cache() -> None:
    """Used by tests / after re-ingesting data mid-process."""
    global _column_index_cache
    _column_index_cache = {}


def distinct_values(column: str) -> list[str]:
    """The column's real distinct values (sorted), for exact-match checks
    that shouldn't pay for a query embedding."""
    if column not in SEMANTIC_COLUMNS:
        raise ValueError(f"'{column}' is not a semantically-indexed column.")
    if column not in _column_index_cache:
        _column_index_cache[column] = _build_column_index(column)
    return _column_index_cache[column][0]


@dataclass
class Match:
    value: str
    score: float

    def as_dict(self) -> dict:
        return {"value": self.value, "score": round(self.score, 4)}


def resolve(column: str, text: str, top_k: int = 5) -> list[Match]:
    """Rank `column`'s real stored distinct values by cosine similarity to
    `text`. Returns up to top_k matches, highest similarity first."""
    if column not in SEMANTIC_COLUMNS:
        raise ValueError(f"'{column}' is not a semantically-indexed column.")
    if not text or not text.strip():
        return []

    if column not in _column_index_cache:
        _column_index_cache[column] = _build_column_index(column)
    values, vectors = _column_index_cache[column]
    if not values:
        return []

    query_vec = np.array(_get_embed_model().get_query_embedding(text))
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    query_vec = query_vec / query_norm

    scores = vectors @ query_vec
    top_idx = np.argsort(-scores)[:top_k]
    return [Match(value=values[i], score=float(scores[i])) for i in top_idx]


def best_match(column: str, text: str, min_score: float) -> Match | None:
    """Convenience helper for exact-match fallbacks: the single best match
    for `column`, or None if nothing clears `min_score`."""
    matches = resolve(column, text, top_k=1)
    if matches and matches[0].score >= min_score:
        return matches[0]
    return None
