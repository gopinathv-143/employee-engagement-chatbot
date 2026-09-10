"""
Step 1: Data ingestion.

Responsibilities of this module ONLY:
  - Read the raw survey file (xlsx or csv) with pandas.
  - Clean / validate it into a predictable shape.
  - Persist the cleaned data into SQLite (the source of truth for every
    structured-data tool: SQL generator/executor, analytics).

This module does NOT know about LlamaIndex, Mistral, or FastAPI. Keeping it
self-contained means it can be unit-tested with plain pandas/sqlite, with no
API key required (see tests/test_data_processing.py).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app import config

# The exact columns we expect in the source file. Extra columns are dropped;
# missing required columns raise an error early (fail fast, not deep inside
# a SQL query at chat time).
REQUIRED_COLUMNS = [
    "Response_ID",
    "Company",
    "Respondent_Type",
    "Department",
    "Role",
    "Tenure_Years",
    "Theme",
    "Question",
    "Rating",
    "Comment",
    "Employee_Feedback",
    "Response_Date",
]


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    dropped_missing_required: int
    dropped_invalid_rating: int
    dropped_duplicates: int

    def as_dict(self) -> dict:
        return self.__dict__


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw survey file. Supports .xlsx and .csv transparently."""
    path = Path(path) if path is not None else config.DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. Set DATA_FILE in .env or place "
            f"the survey export there."
        )

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data file extension: {path.suffix}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Source file is missing required columns: {missing}")

    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and normalize the raw dataframe.

    Rules (kept deliberately simple for a POC):
      - Trim whitespace on all text columns.
      - Parse Response_Date to a real datetime; derive Response_Month
        (YYYY-MM) for trend queries.
      - Rating must be an integer 1-5; rows outside that range are dropped
        (a survey rating outside 1-5 indicates corrupted data, not a
        legitimate response).
      - Rows missing Response_ID, Department, Theme, or Rating are dropped
        (cannot be reliably analyzed or retrieved).
      - Comment / Employee_Feedback may be empty strings - that's a valid
        "no free-text comment left" case, not an error.
      - Exact duplicate rows (by Response_ID) are dropped, keeping the
        first occurrence.
    """
    rows_in = len(df)
    df = df.copy()

    df = df[REQUIRED_COLUMNS]

    text_cols = ["Company", "Respondent_Type", "Department", "Role", "Theme",
                 "Question", "Comment", "Employee_Feedback"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Tenure_Years"] = pd.to_numeric(df["Tenure_Years"], errors="coerce")
    df["Response_Date"] = pd.to_datetime(df["Response_Date"], errors="coerce")

    required_nonnull = ["Response_ID", "Department", "Theme", "Rating", "Response_Date"]
    missing_mask = df[required_nonnull].isna().any(axis=1)
    dropped_missing_required = int(missing_mask.sum())
    df = df[~missing_mask]

    invalid_rating_mask = ~df["Rating"].isin([1, 2, 3, 4, 5])
    dropped_invalid_rating = int(invalid_rating_mask.sum())
    df = df[~invalid_rating_mask]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Response_ID"], keep="first")
    dropped_duplicates = before_dedup - len(df)

    df["Rating"] = df["Rating"].astype(int)
    df["Response_Month"] = df["Response_Date"].dt.strftime("%Y-%m")
    df["Response_Date"] = df["Response_Date"].dt.strftime("%Y-%m-%d")
    df["Comment"] = df["Comment"].fillna("")
    df["Employee_Feedback"] = df["Employee_Feedback"].fillna("")

    df = df.reset_index(drop=True)

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(df),
        dropped_missing_required=dropped_missing_required,
        dropped_invalid_rating=dropped_invalid_rating,
        dropped_duplicates=dropped_duplicates,
    )
    return df, report


def write_sqlite(df: pd.DataFrame, db_path: Path | str | None = None) -> None:
    """Persist the cleaned dataframe into SQLite as the single source table."""
    db_path = Path(db_path) if db_path is not None else config.SQLITE_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(config.SQL_TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_department '
            f'ON {config.SQL_TABLE_NAME}(Department)'
        )
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_theme '
            f'ON {config.SQL_TABLE_NAME}(Theme)'
        )
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_month '
            f'ON {config.SQL_TABLE_NAME}(Response_Month)'
        )
        conn.commit()
    finally:
        conn.close()


def build_dataset(force_rebuild: bool = False) -> tuple[pd.DataFrame, CleaningReport | None]:
    """Ensure the SQLite database exists and return the cleaned dataframe.

    If the DB already exists and force_rebuild is False, we still re-read it
    from SQLite (not the source file) so that every consumer (SQL tools,
    analytics tool, indexing step) works off exactly the same cleaned rows.
    """
    if config.SQLITE_DB_PATH.exists() and not force_rebuild:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        try:
            df = pd.read_sql(f"SELECT * FROM {config.SQL_TABLE_NAME}", conn)
        finally:
            conn.close()
        return df, None

    raw = load_raw()
    df, report = clean(raw)
    write_sqlite(df)
    return df, report


def get_schema_description() -> str:
    """Human-readable schema description used to ground the SQL generator."""
    return f"""Table: {config.SQL_TABLE_NAME}
Columns:
  Response_ID TEXT       - unique survey response identifier, e.g. 'ABG-00001'
  Company TEXT            - company name (mostly a single constant value in this dataset)
  Respondent_Type TEXT     - one of 'Employee', 'Worker', 'HR'
  Department TEXT           - department name, e.g. 'Finance', 'Operations'
  Role TEXT                  - job role/title, e.g. 'Analyst', 'Manager'
  Tenure_Years REAL            - years of tenure, decimal
  Theme TEXT                    - survey theme/category, e.g. 'Job Satisfaction',
                                   'Compensation & Benefits', 'Leadership & Trust',
                                   'Manager Support', 'Performance Management',
                                   'Work Environment', 'Team Collaboration',
                                   'Learning & Development', 'Employee Wellbeing',
                                   'Workplace Safety', 'Work-Life Balance',
                                   'Communication', 'Career Growth & Mobility',
                                   'Rewards & Recognition', 'Technology & Tools'
  Question TEXT                  - the exact survey question text
  Rating INTEGER                  - 1 (worst) to 5 (best)
  Comment TEXT                     - free-text employee comment (may be empty)
  Employee_Feedback TEXT            - short label derived from rating, one of
                                       'Very Dissatisfied','Dissatisfied','Neutral',
                                       'Satisfied','Very Satisfied'
  Response_Date TEXT                 - ISO date 'YYYY-MM-DD'
  Response_Month TEXT                 - 'YYYY-MM', derived, for trend grouping

Notes for SQL generation:
  - This is SQLite. Use SQLite functions only (e.g. strftime, not DATE_TRUNC).
  - Only SELECT statements are allowed.
  - Always alias aggregate columns with a readable name (e.g. AVG(Rating) AS avg_rating).
"""
