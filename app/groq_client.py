"""
Single shared Groq SDK client instance, plus a retry-wrapped
chat-completion helper.

Kept in one place so every module (sql_generator, sentiment, agent) talks to
Groq the same way, with the same key/config and the same rate-limit
resilience, instead of each constructing its own client and re-solving the
same 429 problem separately.

Groq's chat.completions API is OpenAI-compatible, so response shapes
(response.choices[0].message, .tool_calls, .function.arguments as a JSON
string) match what app/agent.py already expects - no call-site changes were
needed beyond swapping the import and the model name.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from groq import Groq

from app import config
from app.retry_utils import rate_limit_retry


@lru_cache(maxsize=1)
def get_client() -> Groq:
    return Groq(api_key=config.require_groq_key())


@rate_limit_retry
def chat_complete(**kwargs: Any):
    """client.chat.completions.create(...) with automatic retry/backoff on 429s.

    Every call site (sql_generator, sentiment, agent) should use this
    instead of calling client.chat.completions.create directly.
    """
    return get_client().chat.completions.create(**kwargs)
