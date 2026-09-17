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


def is_daily_quota_error(exc: BaseException) -> bool:
    """A 'tokens/requests per day' limit is a different failure class from
    a short per-minute burst limit: Groq's own error tells the caller to
    wait minutes, not seconds, so retrying it on this decorator's backoff
    schedule (maxing out at 60s a step) can never succeed within the
    retry budget - it just spends ~4 minutes finding that out. Per-minute/
    per-second limits, by contrast, genuinely do clear inside that window,
    so only THOSE should go through the retry loop below."""
    text = str(exc).lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text


rate_limit_retry = retry(
    retry=retry_if_exception(
        lambda exc: is_rate_limit_error(exc) and not is_daily_quota_error(exc)
    ),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
