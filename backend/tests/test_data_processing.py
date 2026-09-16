"""
Unit tests for app.data_processing. No API key or network access required -
this only exercises pandas logic against a small synthetic dataframe.

Run with:
    .venv\\Scripts\\pytest tests\\test_data_processing.py -v
"""

import pandas as pd

from app.data_processing import REQUIRED_COLUMNS, clean


def _make_raw_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "Response_ID": None, "Company": "Acme", "Respondent_Type": "Employee",
        "Department": "Finance", "Role": "Analyst", "Tenure_Years": 2.5,
        "Theme": "Job Satisfaction", "Question": "How satisfied are you?",
        "Rating": 4, "Comment": "Good place to work.",
        "Employee_Feedback": "Satisfied", "Response_Date": "2025-01-15",
    }
    full_rows = []
    for i, overrides in enumerate(rows):
        row = {**base, "Response_ID": f"T-{i:04d}"}
        row.update(overrides)
        full_rows.append(row)
    return pd.DataFrame(full_rows)[REQUIRED_COLUMNS]


def test_clean_keeps_valid_rows():
    df = _make_raw_df([{}, {"Rating": 2}, {"Rating": 5}])
    cleaned, report = clean(df)
    assert report.rows_in == 3
    assert report.rows_out == 3
    assert list(cleaned["Rating"]) == [4, 2, 5]
    assert "Response_Month" in cleaned.columns
    assert cleaned.loc[0, "Response_Month"] == "2025-01"


def test_clean_drops_invalid_rating():
    df = _make_raw_df([{"Rating": 4}, {"Rating": 7}, {"Rating": 0}, {"Rating": "n/a"}])
    cleaned, report = clean(df)
    assert report.dropped_invalid_rating >= 2
    assert set(cleaned["Rating"]) <= {1, 2, 3, 4, 5}


def test_clean_drops_missing_required_fields():
    df = _make_raw_df([{}, {"Department": None}, {"Theme": ""}])
    # empty string Theme should count as missing after strip
    df.loc[2, "Theme"] = None
    cleaned, report = clean(df)
    assert report.dropped_missing_required == 2
    assert len(cleaned) == 1


def test_clean_deduplicates_by_response_id():
    df = _make_raw_df([{}, {}])
    df.loc[1, "Response_ID"] = df.loc[0, "Response_ID"]  # force a duplicate id
    cleaned, report = clean(df)
    assert report.dropped_duplicates == 1
    assert len(cleaned) == 1


def test_clean_allows_empty_comment():
    df = _make_raw_df([{"Comment": None}])
    cleaned, _ = clean(df)
    assert cleaned.loc[0, "Comment"] == ""
