"""
Composes the SQL Generator and SQL Executor into the single tool the agent
actually calls: "answer this natural-language question using SQL".

Why a composed tool instead of exposing generate_sql / run_sql separately
to the LLM as two tool-calls: SQL generation isn't really a *tool call* in
the function-calling sense (there's no external system to invoke - it's the
model writing text), it's a specialised reasoning step. So this file plays
the role of "the SQL Generator step calls the SQL Executor step", while
sql_generator.py and sql_executor.py remain independent, separately
testable Python modules exactly as required. The agent only ever sees one
tool, `query_database`, which keeps the tool-calling loop simple.

Includes two automatic self-correction retries, neither of which costs an
extra agent iteration or an extra Groq call for the second one:
  1. If the first generated SQL fails to EXECUTE, the error is fed back
     into the generator once (one extra Groq call).
  2. If the SQL executes but a Question-column filter matched nothing (a
     full-sentence exact match, or a LIKE guess that missed), retry once
     with the real Question text closest to `question` by embedding
     similarity (see schema_index) - pure local embedding + SQLite, no LLM
     call at all. Whether the LLM chooses to keep searching for a better
     match instead of using a good one it already found is inherently
     nondeterministic (a one-word difference in phrasing can change its
     exploration path); this makes the "use the closest real match" step
     itself deterministic instead of hoping the model follows the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.tools import schema_index, sql_executor, sql_generator

_QUESTION_CONDITION_RE = re.compile(
    r"Question\s*(?:=|LIKE)\s*'[^']*'(?:\s+AND\s+Question\s*(?:=|LIKE)\s*'[^']*')*",
    re.IGNORECASE,
)

# Below schema_index.DEFAULT_MATCH_THRESHOLD (0.72) a candidate isn't
# confident enough to silently swap in - real paraphrases of the SAME
# question scored 0.79-0.86 in calibration, while a genuinely different
# (if related) topic like "HR team" or "manager support for growth" (vs.
# the real "manager support for work challenges") scored 0.69-0.71. Below
# this lower floor a candidate is unrelated noise (an off-topic nonsense
# string scored ~0.5) and not worth surfacing at all.
_SUGGEST_THRESHOLD = 0.55


@dataclass
class QueryDatabaseResult:
    ok: bool
    question: str
    sql: str
    attempts: int
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool
    error: str | None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__


# Same aggregate-name convention verification.py uses: AVG/SUM/MIN/MAX/MEAN
# return NULL (not 0) when nothing matched, unlike COUNT. A row can
# legitimately mix the two - e.g. {"avg_rating": None, "response_count": 0}
# - so "no matching data" must check the aggregate columns specifically,
# not require every column in the row to be None.
_AGGREGATE_NAME_HINTS = ("avg", "sum", "min", "max", "mean")


def _no_matching_data(result: sql_executor.SqlResult) -> bool:
    """True if the query executed but found nothing: zero rows, or a single
    row where an AVG/SUM/MIN/MAX/MEAN-named column is NULL - SQL's normal
    behaviour for an aggregate over a WHERE clause that matched no rows."""
    if result.row_count == 0:
        return True
    if result.row_count == 1:
        row = result.rows[0]
        if not row:
            return False
        has_null_aggregate = any(
            value is None and any(hint in key.lower() for hint in _AGGREGATE_NAME_HINTS)
            for key, value in row.items()
        )
        return has_null_aggregate or all(v is None for v in row.values())
    return False


def _substitute_question_filter(sql: str, value: str) -> str | None:
    """Rewrite the SQL's Question filter(s) to a single exact match on
    `value`. Returns None if the SQL doesn't contain a recognizable
    Question filter to rewrite - fail safe, skip the fallback rather than
    risk mangling a query shape this pattern doesn't understand."""
    escaped = value.replace("'", "''")
    new_sql, n = _QUESTION_CONDITION_RE.subn(f"Question = '{escaped}'", sql, count=1)
    return new_sql if n else None


def query_database(question: str) -> QueryDatabaseResult:
    sql = sql_generator.generate_sql(question)
    result = sql_executor.run_sql(sql)
    attempts = 1

    if not result.ok:
        retry_sql = sql_generator.generate_sql(question, error_feedback=result.error)
        retry_result = sql_executor.run_sql(retry_sql)
        attempts = 2
        sql, result = retry_sql, retry_result

    notes: list[str] = []
    if result.ok and _no_matching_data(result) and _QUESTION_CONDITION_RE.search(sql):
        # Only worth reasoning about Question candidates when the SQL is
        # actually filtering on Question - otherwise (e.g. a Department
        # filter matched nothing) a "closest Question" suggestion would be
        # irrelevant noise.
        candidates = schema_index.resolve("Question", question, top_k=1)
        top = candidates[0] if candidates else None

        if top is not None and top.score >= schema_index.DEFAULT_MATCH_THRESHOLD:
            # Confident enough to be the same question, just worded
            # differently - swap it in automatically.
            fallback_sql = _substitute_question_filter(sql, top.value)
            if fallback_sql is not None:
                fallback_result = sql_executor.run_sql(fallback_sql)
                attempts += 1
                if fallback_result.ok and not _no_matching_data(fallback_result):
                    notes.append(
                        f"The generated SQL's Question filter matched nothing; "
                        f"automatically retried using the closest real survey "
                        f"question '{top.value}' (similarity {top.score:.2f}) "
                        f"instead."
                    )
                    sql, result = fallback_sql, fallback_result
        elif top is not None and top.score >= _SUGGEST_THRESHOLD:
            # Not confident enough to silently substitute - a lower score
            # here usually means a genuinely different (if related) topic,
            # e.g. "manager support for growth" vs. the real question "manager
            # support for work challenges". Surfacing it as a named
            # suggestion is safer than picking it silently, but still far
            # more useful than the caller getting nothing to work with.
            notes.append(
                f"No survey question is a close enough match to answer this "
                f"directly (closest real question: '{top.value}', similarity "
                f"{top.score:.2f} - likely a related but different topic). "
                f"Consider asking about that specific question if it's "
                f"relevant, rather than treating this as a verified answer to "
                f"the original wording."
            )

    return QueryDatabaseResult(
        ok=result.ok,
        question=question,
        sql=sql,
        attempts=attempts,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        truncated=result.truncated,
        error=result.error,
        notes=notes,
    )
