"""
Unit tests for app.tools.query_database - specifically the compose +
one-retry-on-failure behaviour. Both sql_generator.generate_sql and
sql_executor.run_sql are mocked, so this needs no API key and no real
database file.

Run with:
    .venv\\Scripts\\pytest tests\\test_query_database.py -v
"""

from unittest.mock import patch

from app.tools.query_database import query_database
from app.tools.sql_executor import SqlResult


def test_query_database_succeeds_on_first_try():
    with patch("app.tools.query_database.sql_generator.generate_sql", return_value="SELECT 1;") as gen, \
         patch("app.tools.query_database.sql_executor.run_sql",
               return_value=SqlResult(ok=True, sql="SELECT 1;", columns=["c"],
                                       rows=[{"c": 1}], row_count=1)) as run:
        result = query_database("how many rows?")

    assert result.ok is True
    assert result.attempts == 1
    assert gen.call_count == 1
    assert run.call_count == 1


def test_query_database_retries_once_on_failure_then_succeeds():
    bad = SqlResult(ok=False, sql="SELECT bogus;", error="SQLite error: no such column: bogus")
    good = SqlResult(ok=True, sql="SELECT 1;", columns=["c"], rows=[{"c": 1}], row_count=1)

    with patch("app.tools.query_database.sql_generator.generate_sql",
               side_effect=["SELECT bogus;", "SELECT 1;"]) as gen, \
         patch("app.tools.query_database.sql_executor.run_sql", side_effect=[bad, good]) as run:
        result = query_database("how many rows?")

    assert result.ok is True
    assert result.attempts == 2
    assert gen.call_count == 2
    # The retry must have been told *why* the first attempt failed.
    _, kwargs = gen.call_args_list[1]
    assert "error_feedback" in kwargs or gen.call_args_list[1].args
    assert run.call_count == 2


def test_query_database_reports_failure_after_retry_also_fails():
    bad1 = SqlResult(ok=False, sql="SELECT bogus;", error="syntax error")
    bad2 = SqlResult(ok=False, sql="SELECT still_bogus;", error="still failing")

    with patch("app.tools.query_database.sql_generator.generate_sql",
               side_effect=["SELECT bogus;", "SELECT still_bogus;"]), \
         patch("app.tools.query_database.sql_executor.run_sql", side_effect=[bad1, bad2]):
        result = query_database("how many rows?")

    assert result.ok is False
    assert result.attempts == 2
    assert "still failing" in result.error
