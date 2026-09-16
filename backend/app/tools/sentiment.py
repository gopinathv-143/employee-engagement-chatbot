"""
Sentiment tool.

Classifies employee comments as Positive / Neutral / Negative using Groq,
batched into a single JSON-in / JSON-out call per chunk (fast + cheap
compared to one API call per comment, which matters once this is wired to
search_employee_comments results in the agent loop).

Independent of app.indexing - it takes plain {response_id, text} items, so
it can be tested against hand-written text without touching the vector
index or the database (see tests/test_sentiment.py, which mocks the Groq
client).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app import config
from app.groq_client import chat_complete

_ALLOWED = {"Positive", "Neutral", "Negative"}
_CHUNK_SIZE = 20

_SYSTEM_PROMPT = """You are an HR sentiment classifier. For each numbered employee \
comment, classify its sentiment as exactly one of: Positive, Neutral, Negative.

Respond with ONLY a JSON object of this exact shape, nothing else:
{"results": [{"response_id": "<id>", "sentiment": "Positive|Neutral|Negative", \
"rationale": "<one short sentence>"}, ...]}

Include exactly one entry per comment given, using the same response_id."""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class SentimentResult:
    ok: bool
    results: list[dict] = field(default_factory=list)
    count: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return {"ok": self.ok, "results": self.results, "count": self.count, "error": self.error}


def _build_user_prompt(chunk: list[dict]) -> str:
    lines = [f'{i + 1}. [response_id: {item["response_id"]}] {item["text"]}'
              for i, item in enumerate(chunk)]
    return "Comments:\n" + "\n".join(lines)


def _classify_chunk(chunk: list[dict]) -> list[dict] | None:
    try:
        response = chat_complete(
            model=config.GROQ_CHAT_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(chunk)},
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception:  # noqa: BLE001 - API/network failure, let caller fall back
        return None

    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    items = parsed.get("results")
    if not isinstance(items, list):
        return None

    by_id = {}
    for item in items:
        rid = str(item.get("response_id", ""))
        sentiment = item.get("sentiment", "Neutral")
        if sentiment not in _ALLOWED:
            sentiment = "Neutral"
        by_id[rid] = {
            "response_id": rid,
            "sentiment": sentiment,
            "rationale": item.get("rationale", ""),
        }

    # Guarantee full coverage: any input item the model silently skipped
    # gets an explicit fallback entry rather than disappearing.
    results = []
    for item in chunk:
        rid = str(item["response_id"])
        if rid in by_id:
            results.append(by_id[rid])
        else:
            results.append({
                "response_id": rid, "sentiment": "Neutral",
                "rationale": "Fallback: model did not return a classification for this item.",
            })
    return results


def analyze_sentiment(items: list[dict]) -> SentimentResult:
    """items: list of {"response_id": str, "text": str}."""
    if not items:
        return SentimentResult(ok=False, error="No items provided to classify.")

    all_results: list[dict] = []
    for i in range(0, len(items), _CHUNK_SIZE):
        chunk = items[i : i + _CHUNK_SIZE]
        classified = _classify_chunk(chunk)
        if classified is None:
            return SentimentResult(
                ok=False,
                error="Sentiment classification failed (could not parse a valid "
                      "response from Groq).",
            )
        all_results.extend(classified)

    return SentimentResult(ok=True, results=all_results, count=len(all_results))
