from types import SimpleNamespace
from unittest.mock import patch

from app.agent import _call_query_database, _default_give_up_message, _is_specific_question_average


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


def test_give_up_message_flags_no_matching_data():
    trace = [{
        "tool": "search_employee_comments",
        "verification": {"valid": False, "issues": [
            "No employee comments were retrieved for query: 'quantum sandwiches'."
        ]},
    }]
    message = _default_give_up_message(trace)
    assert "couldn't find anything in the survey data" in message


def test_give_up_message_falls_back_to_generic_for_other_failures():
    trace = [{
        "tool": "run_analytics",
        "verification": {"valid": False, "issues": ["Implausible percentage value: 200.0"]},
    }]
    message = _default_give_up_message(trace)
    assert "couldn't find anything in the survey data" not in message
    assert "rephrase" in message


def test_give_up_message_handles_empty_trace():
    assert "rephrase" in _default_give_up_message([])