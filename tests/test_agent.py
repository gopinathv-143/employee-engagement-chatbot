from types import SimpleNamespace
from unittest.mock import patch

from app.agent import _call_query_database, _is_specific_question_average


def test_specific_question_average_is_detected():
    assert _is_specific_question_average(
        "What is the average rating for "
        "'How useful is the guidance you receive from your manager?'"
    )
    assert not _is_specific_question_average("What is the average rating by department?")


def test_query_database_result_includes_rating_scale():
    fake_result = SimpleNamespace(as_dict=lambda: {"ok": True, "rows": []})
    with patch("app.agent.query_database.query_database", return_value=fake_result):
        result = _call_query_database({"question": "average rating for a question"})

    assert result["rating_scale"]["database"] == (
        "Rating is INTEGER on a 1-5 scale (1 worst, 5 best)"
    )
    assert "multiply a verified average by 2" in result["rating_scale"]["zero_to_ten_equivalent"]