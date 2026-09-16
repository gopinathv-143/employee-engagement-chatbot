"""
Shared retry/backoff policy for Groq API calls.

Groq (chat completions, the only external API this project calls) rate-
limits under real usage. Every direct call into the SDK goes through this
same retry policy so a transient rate limit never surfaces as a hard
failure to the user or to a tool's verification step.
"""

from __future__ import annotations

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limited" in text


rate_limit_retry = retry(
    retry=retry_if_exception(is_rate_limit_error),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
