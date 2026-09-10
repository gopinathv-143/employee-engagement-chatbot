"""
Unit tests for app.tools.sentiment. The Mistral client is mocked - these
tests check the JSON parsing / fallback-coverage logic, not the live API
(that's covered by the manual smoke test in scripts/ask.py once you have a
working key).

Run with:
    .venv\\Scripts\\pytest tests\\test_sentiment.py -v
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from app.tools.sentiment import analyze_sentiment


def _fake_response(response_text: str):
    message = SimpleNamespace(content=response_text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_analyze_sentiment_happy_path():
    payload = json.dumps({
        "results": [
            {"response_id": "R1", "sentiment": "Positive", "rationale": "Praises manager."},
            {"response_id": "R2", "sentiment": "Negative", "rationale": "Complains about pay."},
        ]
    })
    with patch("app.tools.sentiment.chat_complete", return_value=_fake_response(payload)):
        result = analyze_sentiment([
            {"response_id": "R1", "text": "My manager is great."},
            {"response_id": "R2", "text": "Pay is too low."},
        ])
    assert result.ok is True
    by_id = {r["response_id"]: r["sentiment"] for r in result.results}
    assert by_id == {"R1": "Positive", "R2": "Negative"}


def test_analyze_sentiment_normalizes_invalid_label():
    payload = json.dumps({"results": [{"response_id": "R1", "sentiment": "Furious", "rationale": ""}]})
    with patch("app.tools.sentiment.chat_complete", return_value=_fake_response(payload)):
        result = analyze_sentiment([{"response_id": "R1", "text": "..."}])
    assert result.results[0]["sentiment"] == "Neutral"


def test_analyze_sentiment_fills_in_missing_ids():
    # Model only returned a classification for R1, not R2.
    payload = json.dumps({"results": [{"response_id": "R1", "sentiment": "Positive", "rationale": ""}]})
    with patch("app.tools.sentiment.chat_complete", return_value=_fake_response(payload)):
        result = analyze_sentiment([
            {"response_id": "R1", "text": "Great."},
            {"response_id": "R2", "text": "Also great."},
        ])
    assert result.ok is True
    assert result.count == 2
    r2 = next(r for r in result.results if r["response_id"] == "R2")
    assert r2["sentiment"] == "Neutral"
    assert "Fallback" in r2["rationale"]


def test_analyze_sentiment_handles_unparseable_response():
    with patch("app.tools.sentiment.chat_complete", return_value=_fake_response("not json at all")):
        result = analyze_sentiment([{"response_id": "R1", "text": "Great."}])
    assert result.ok is False
    assert result.error


def test_analyze_sentiment_rejects_empty_input():
    result = analyze_sentiment([])
    assert result.ok is False
