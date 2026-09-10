"""
Unit tests for app.tools.sql_executor. No API key or network access
required - builds a tiny throwaway SQLite file per test.

Run with:
    .venv\\Scripts\\pytest tests\\test_sql_executor.py -v
"""

import sqlite3

import pytest

from app.tools.sql_executor import run_sql, validate_sql


@pytest.fixture()
def small_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE engagement (Response_ID TEXT, Department TEXT, Rating INTEGER)"
    )
    conn.executemany(
        "INSERT INTO engagement VALUES (?, ?, ?)",
        [("R1", "Finance", 4), ("R2", "Finance", 2), ("R3", "Ops", 5)],
    )
    conn.commit()
    conn.close()
    return db_path


def test_validate_sql_allows_select():
    assert validate_sql("SELECT * FROM engagement;") is None


def test_validate_sql_blocks_insert():
    assert validate_sql("INSERT INTO engagement VALUES ('x','y',1);") is not None


def test_validate_sql_blocks_multiple_statements():
    assert validate_sql("SELECT * FROM engagement; DROP TABLE engagement;") is not None


def test_validate_sql_blocks_missing_table_reference():
    assert validate_sql("SELECT 1;") is not None


def test_validate_sql_allows_with_clause():
    sql = "WITH t AS (SELECT * FROM engagement) SELECT COUNT(*) FROM t;"
    assert validate_sql(sql) is None


def test_run_sql_executes_and_returns_rows(small_db):
    result = run_sql("SELECT COUNT(*) AS c FROM engagement;", db_path=small_db)
    assert result.ok is True
    assert result.rows == [{"c": 3}]


def test_run_sql_rejects_unsafe_query(small_db):
    result = run_sql("DELETE FROM engagement;", db_path=small_db)
    assert result.ok is False
    assert result.error  # rejected for *some* clear reason (not a silent DB mutation)


def test_run_sql_rejects_forbidden_keyword_disguised_in_with_clause(small_db):
    sql = "WITH x AS (DELETE FROM engagement RETURNING *) SELECT * FROM x;"
    result = run_sql(sql, db_path=small_db)
    assert result.ok is False
    assert "Forbidden keyword" in result.error


def test_run_sql_reports_sqlite_errors(small_db):
    result = run_sql("SELECT nonexistent_column FROM engagement;", db_path=small_db)
    assert result.ok is False
    assert "SQLite error" in result.error


def test_run_sql_missing_db_reports_clear_error(tmp_path):
    result = run_sql("SELECT * FROM engagement;", db_path=tmp_path / "missing.db")
    assert result.ok is False
    assert "not found" in result.error
