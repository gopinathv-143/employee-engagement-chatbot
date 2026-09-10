"""
SQL Executor tool.

Takes a SQL string (produced by app.tools.sql_generator, or typed by hand
during testing) and runs it SAFELY against the read-only SQLite database.

This module has NO dependency on Mistral or LlamaIndex - it is pure
sqlite3 + string safety checks, so it can be fully unit-tested without an
API key (see tests/test_sql_executor.py).

Safety model (defense in depth, appropriate for a POC - not a hardened
multi-tenant system):
  1. Only a single SELECT (or WITH ... SELECT) statement is allowed.
  2. A block-list of DDL/DML keywords is rejected even if disguised inside
     a WITH clause or subquery.
  3. The connection itself is opened read-only via a SQLite URI, so even a
     bug in the checks above cannot mutate the database.
  4. Row count is hard-capped (config.SQL_MAX_ROWS) to protect the LLM
     context window.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from app import config

_FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "truncate",
    "grant", "revoke",
]

_ALLOWED_START = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)


@dataclass
class SqlResult:
    ok: bool
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "error": self.error,
        }


def validate_sql(sql: str) -> str | None:
    """Return an error message if the SQL is not allowed, else None."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return "Empty SQL."

    if ";" in stripped:
        return "Multiple statements are not allowed (found ';' inside the query)."

    if not _ALLOWED_START.match(stripped):
        return "Only SELECT (or WITH ... SELECT) statements are allowed."

    lowered = stripped.lower()
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            return f"Forbidden keyword detected: '{kw}'."

    if config.SQL_TABLE_NAME.lower() not in lowered:
        return (
            f"Query does not reference the '{config.SQL_TABLE_NAME}' table. "
            f"Only that table exists."
        )

    return None


def run_sql(sql: str, db_path: Path | str | None = None) -> SqlResult:
    """Safely execute a validated SELECT statement and return the results."""
    db_path = Path(db_path) if db_path is not None else config.SQLITE_DB_PATH

    error = validate_sql(sql)
    if error:
        return SqlResult(ok=False, sql=sql, error=error)

    if not db_path.exists():
        return SqlResult(
            ok=False, sql=sql,
            error=f"Database not found at {db_path}. Run scripts/build_db.py first.",
        )

    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            fetched = cursor.fetchmany(config.SQL_MAX_ROWS + 1)
            columns = [d[0] for d in cursor.description] if cursor.description else []
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return SqlResult(ok=False, sql=sql, error=f"SQLite error: {exc}")

    truncated = len(fetched) > config.SQL_MAX_ROWS
    rows = [dict(r) for r in fetched[: config.SQL_MAX_ROWS]]

    return SqlResult(
        ok=True,
        sql=sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
    )
