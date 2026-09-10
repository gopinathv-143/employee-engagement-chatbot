"""
Analytics tool.

A small set of deterministic pandas computations (percentages, group
averages, counts, simple monthly trends) for the numeric/categorical
questions that are error-prone to hand off to freeform SQL generation every
single time (e.g. "percentage dissatisfied"). This tool complements
query_database rather than replacing it - the agent picks whichever is the
better fit for a given question.

No Mistral/LlamaIndex dependency here either - pure pandas, so it is fully
unit-testable (see tests/test_analytics.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.data_processing import build_dataset

_ALLOWED_GROUP_COLUMNS = {
    "Department", "Role", "Theme", "Respondent_Type", "Company",
    "Employee_Feedback", "Response_Month",
}
_ALLOWED_FILTER_COLUMNS = _ALLOWED_GROUP_COLUMNS | {"Rating"}

_df_cache: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache, _ = build_dataset(force_rebuild=False)
    return _df_cache


def reset_cache() -> None:
    """Used by tests / after re-ingesting data mid-process."""
    global _df_cache
    _df_cache = None


@dataclass
class AnalyticsResult:
    ok: bool
    operation: str
    params: dict
    data: list[dict] = field(default_factory=list)
    total_matched: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "operation": self.operation, "params": self.params,
            "data": self.data, "total_matched": self.total_matched, "error": self.error,
        }


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> tuple[pd.DataFrame, str | None]:
    if not filters:
        return df, None
    for col, val in filters.items():
        if col not in _ALLOWED_FILTER_COLUMNS:
            return df, f"Cannot filter on column '{col}'."
        if col == "Rating":
            df = df[df["Rating"] == int(val)]
        else:
            df = df[df[col].astype(str).str.lower() == str(val).lower()]
    return df, None


def _percentage(df: pd.DataFrame, params: dict) -> AnalyticsResult:
    column = params.get("column")
    value = params.get("value")
    if column not in _ALLOWED_FILTER_COLUMNS:
        return AnalyticsResult(False, "percentage", params, error=f"Unsupported column '{column}'.")

    filtered, err = _apply_filters(df, params.get("filters"))
    if err:
        return AnalyticsResult(False, "percentage", params, error=err)

    total = len(filtered)
    if total == 0:
        return AnalyticsResult(True, "percentage", params, data=[], total_matched=0)

    if column == "Rating":
        matches = int((filtered["Rating"] == int(value)).sum())
    else:
        matches = int((filtered[column].astype(str).str.lower() == str(value).lower()).sum())

    pct = round(100.0 * matches / total, 2)
    return AnalyticsResult(
        True, "percentage", params,
        data=[{"matching_count": matches, "total_count": total, "percentage": pct}],
        total_matched=total,
    )


def _average_by_group(df: pd.DataFrame, params: dict) -> AnalyticsResult:
    group_by = params.get("group_by")
    if group_by not in _ALLOWED_GROUP_COLUMNS:
        return AnalyticsResult(False, "average_by_group", params,
                                error=f"Unsupported group_by column '{group_by}'.")

    filtered, err = _apply_filters(df, params.get("filters"))
    if err:
        return AnalyticsResult(False, "average_by_group", params, error=err)

    grouped = (
        filtered.groupby(group_by)["Rating"]
        .agg(avg_rating="mean", count="size")
        .reset_index()
        .sort_values("avg_rating", ascending=False)
    )
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    return AnalyticsResult(
        True, "average_by_group", params,
        data=grouped.to_dict(orient="records"),
        total_matched=len(filtered),
    )


def _count_by_group(df: pd.DataFrame, params: dict) -> AnalyticsResult:
    group_by = params.get("group_by")
    if group_by not in _ALLOWED_GROUP_COLUMNS:
        return AnalyticsResult(False, "count_by_group", params,
                                error=f"Unsupported group_by column '{group_by}'.")

    filtered, err = _apply_filters(df, params.get("filters"))
    if err:
        return AnalyticsResult(False, "count_by_group", params, error=err)

    counts = (
        filtered.groupby(group_by).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return AnalyticsResult(
        True, "count_by_group", params,
        data=counts.to_dict(orient="records"),
        total_matched=len(filtered),
    )


def _rating_distribution(df: pd.DataFrame, params: dict) -> AnalyticsResult:
    filtered, err = _apply_filters(df, params.get("filters"))
    if err:
        return AnalyticsResult(False, "rating_distribution", params, error=err)

    total = len(filtered)
    dist = filtered["Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    data = [
        {"rating": int(r), "count": int(c),
         "percentage": round(100.0 * c / total, 2) if total else 0.0}
        for r, c in dist.items()
    ]
    return AnalyticsResult(True, "rating_distribution", params, data=data, total_matched=total)


def _trend(df: pd.DataFrame, params: dict) -> AnalyticsResult:
    metric = params.get("metric", "avg_rating")
    if metric not in ("avg_rating", "count"):
        return AnalyticsResult(False, "trend", params, error="metric must be 'avg_rating' or 'count'.")

    filtered, err = _apply_filters(df, params.get("filters"))
    if err:
        return AnalyticsResult(False, "trend", params, error=err)

    if metric == "avg_rating":
        series = filtered.groupby("Response_Month")["Rating"].mean().round(2)
        data = [{"month": m, "avg_rating": v} for m, v in series.sort_index().items()]
    else:
        series = filtered.groupby("Response_Month").size()
        data = [{"month": m, "count": int(v)} for m, v in series.sort_index().items()]

    return AnalyticsResult(True, "trend", params, data=data, total_matched=len(filtered))


_OPERATIONS = {
    "percentage": _percentage,
    "average_by_group": _average_by_group,
    "count_by_group": _count_by_group,
    "rating_distribution": _rating_distribution,
    "trend": _trend,
}


def run_analytics(operation: str, params: dict | None = None) -> AnalyticsResult:
    params = params or {}
    handler = _OPERATIONS.get(operation)
    if handler is None:
        return AnalyticsResult(
            False, operation, params,
            error=f"Unknown operation '{operation}'. Valid: {sorted(_OPERATIONS)}",
        )
    df = _get_df()
    return handler(df, params)
