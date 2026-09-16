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

Includes exactly one automatic self-correction retry: if the first
generated SQL fails to execute, the error is fed back into the generator
once before giving up.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.tools import sql_executor, sql_generator


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

    def as_dict(self) -> dict:
        return self.__dict__


def query_database(question: str) -> QueryDatabaseResult:
    sql = sql_generator.generate_sql(question)
    result = sql_executor.run_sql(sql)
    attempts = 1

    if not result.ok:
        retry_sql = sql_generator.generate_sql(question, error_feedback=result.error)
        retry_result = sql_executor.run_sql(retry_sql)
        attempts = 2
        if retry_result.ok:
            sql, result = retry_sql, retry_result
        else:
            # Keep the second attempt's error - it's the most recent context.
            sql, result = retry_sql, retry_result

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
    )
