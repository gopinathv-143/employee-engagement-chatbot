"""
Step 1 test/CLI: read the Excel file, clean it, write it to SQLite.

Run from the project root (with the venv active):
    python scripts/build_db.py

This is intentionally the FIRST thing you should run and verify before
touching LlamaIndex, tools, or the agent - everything downstream depends on
this SQLite database being correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, data_processing


def main() -> None:
    print(f"Reading source file: {config.DATA_FILE}")
    df, report = data_processing.build_dataset(force_rebuild=True)

    print("\nCleaning report:")
    for k, v in report.as_dict().items():
        print(f"  {k}: {v}")

    print(f"\nWrote {len(df)} rows to {config.SQLITE_DB_PATH} "
          f"(table '{config.SQL_TABLE_NAME}')")

    print("\nSample rows:")
    print(df.head(3).to_string())

    print("\nQuick sanity aggregates:")
    print("  Rating value counts:")
    print(df["Rating"].value_counts().sort_index().to_string())
    print("\n  Departments:", df["Department"].nunique())
    print("  Themes:", df["Theme"].nunique())
    print("  Date range:", df["Response_Date"].min(), "to", df["Response_Date"].max())


if __name__ == "__main__":
    main()
