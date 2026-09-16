"""
Unit tests for app.tools.verification. Pure logic, no dependencies at all.

Run with:
    .venv\\Scripts\\pytest tests\\test_verification.py -v
"""

from app.tools.verification import (
    verify_analytics,
    verify_query_database,
    verify_retrieval,
    verify_sentiment,
    verify_tool_result,
)


def test_verify_query_database_flags_failed_sql():
    result = verify_query_database({"ok": False, "error": "syntax error", "rows": [], "columns": []})
    assert result["valid"] is False
    assert "syntax error" in result["issues"][0]


def test_verify_query_database_passes_good_result():
    result = verify_query_database(
        {"ok": True, "sql": "SELECT 1;", "rows": [{"c": 1}], "columns": ["c"], "row_count": 1}
    )
    assert result["valid"] is True
    assert result["issues"] == []


def test_verify_query_database_flags_empty_result_softly():
    result = verify_query_database(
        {"ok": True, "sql": "SELECT 1;", "rows": [], "columns": ["c"], "row_count": 0}
    )
    assert result["valid"] is True
    assert any("zero rows" in i for i in result["issues"])


def test_verify_analytics_flags_bad_percentage():
    result = verify_analytics({
        "ok": True, "operation": "percentage",
        "data": [{"matching_count": 10, "total_count": 5, "percentage": 200.0}],
    })
    assert result["valid"] is False


def test_verify_analytics_flags_out_of_range_average():
    result = verify_analytics({
        "ok": True, "operation": "average_by_group",
        "data": [{"Department": "Finance", "avg_rating": 9.0}],
    })
    assert result["valid"] is False


def test_verify_analytics_passes_good_result():
    result = verify_analytics({
        "ok": True, "operation": "average_by_group",
        "data": [{"Department": "Finance", "avg_rating": 3.5}],
    })
    assert result["valid"] is True


def test_verify_retrieval_flags_empty_results():
    result = verify_retrieval({"results": []}, query="management support")
    assert result["valid"] is False


def test_verify_retrieval_passes_nonempty_results():
    result = verify_retrieval(
        {"results": [{"response_id": "R1", "text": "Great manager.", "score": 0.8}]},
        query="management support",
    )
    assert result["valid"] is True


def test_verify_sentiment_flags_invalid_label():
    result = verify_sentiment({"results": [{"response_id": "R1", "sentiment": "Angry"}]})
    assert result["valid"] is False


def test_verify_sentiment_passes_valid_labels():
    result = verify_sentiment({"results": [{"response_id": "R1", "sentiment": "Positive"}]})
    assert result["valid"] is True


def test_verify_tool_result_dispatches_by_name():
    result = verify_tool_result("run_analytics", {
        "ok": True, "operation": "average_by_group",
        "data": [{"Department": "Finance", "avg_rating": 3.5}],
    })
    assert result["valid"] is True


def test_verify_tool_result_unknown_tool_passes_through():
    result = verify_tool_result("some_future_tool", {"anything": True})
    assert result["valid"] is True
