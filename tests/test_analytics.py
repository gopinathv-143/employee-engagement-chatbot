"""
Unit tests for app.tools.analytics. No API key or network access required -
injects a small synthetic dataframe directly into the module cache instead
of reading the real SQLite database.

Run with:
    .venv\\Scripts\\pytest tests\\test_analytics.py -v
"""

import pandas as pd
import pytest

import app.tools.analytics as analytics_mod
from app.tools.analytics import run_analytics


@pytest.fixture(autouse=True)
def sample_df():
    df = pd.DataFrame([
        {"Department": "Finance", "Role": "Analyst", "Theme": "Job Satisfaction",
         "Respondent_Type": "Employee", "Company": "Acme", "Rating": 4,
         "Employee_Feedback": "Satisfied", "Response_Month": "2025-01"},
        {"Department": "Finance", "Role": "Manager", "Theme": "Job Satisfaction",
         "Respondent_Type": "Employee", "Company": "Acme", "Rating": 2,
         "Employee_Feedback": "Dissatisfied", "Response_Month": "2025-02"},
        {"Department": "Ops", "Role": "Analyst", "Theme": "Job Satisfaction",
         "Respondent_Type": "Worker", "Company": "Acme", "Rating": 5,
         "Employee_Feedback": "Very Satisfied", "Response_Month": "2025-02"},
        {"Department": "Ops", "Role": "Analyst", "Theme": "Job Satisfaction",
         "Respondent_Type": "Worker", "Company": "Acme", "Rating": 1,
         "Employee_Feedback": "Very Dissatisfied", "Response_Month": "2025-02"},
    ])
    analytics_mod._df_cache = df
    yield df
    analytics_mod.reset_cache()


def test_percentage_dissatisfied():
    result = run_analytics("percentage", {"column": "Employee_Feedback", "value": "Dissatisfied"})
    assert result.ok is True
    assert result.data[0]["matching_count"] == 1
    assert result.data[0]["total_count"] == 4
    assert result.data[0]["percentage"] == 25.0


def test_percentage_with_filter():
    result = run_analytics(
        "percentage",
        {"column": "Rating", "value": 5, "filters": {"Department": "Ops"}},
    )
    assert result.data[0]["total_count"] == 2
    assert result.data[0]["matching_count"] == 1
    assert result.data[0]["percentage"] == 50.0


def test_average_by_group():
    result = run_analytics("average_by_group", {"group_by": "Department"})
    assert result.ok is True
    by_dept = {row["Department"]: row["avg_rating"] for row in result.data}
    assert by_dept["Finance"] == 3.0
    assert by_dept["Ops"] == 3.0


def test_count_by_group():
    result = run_analytics("count_by_group", {"group_by": "Department"})
    by_dept = {row["Department"]: row["count"] for row in result.data}
    assert by_dept["Finance"] == 2
    assert by_dept["Ops"] == 2


def test_rating_distribution_sums_to_100():
    result = run_analytics("rating_distribution", {})
    total_pct = sum(r["percentage"] for r in result.data)
    assert 99.0 <= total_pct <= 101.0


def test_trend_avg_rating_by_month():
    result = run_analytics("trend", {"metric": "avg_rating"})
    by_month = {row["month"]: row["avg_rating"] for row in result.data}
    assert by_month["2025-01"] == 4.0
    assert by_month["2025-02"] == pytest.approx((2 + 5 + 1) / 3, abs=0.01)


def test_unknown_operation_fails_cleanly():
    result = run_analytics("not_a_real_op", {})
    assert result.ok is False
    assert "Unknown operation" in result.error


def test_unsupported_group_by_column_fails_cleanly():
    result = run_analytics("average_by_group", {"group_by": "Response_ID"})
    assert result.ok is False
