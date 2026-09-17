"""
Verification tool.

Runs deterministic, rule-based sanity checks on the output of every other
tool BEFORE it is allowed to reach the final answer. This is what
guarantees the agent never presents an invented or silently-broken number.

Design choice: verification is plain Python (no LLM call) on purpose - it
needs to be fast, free, and 100% consistent, and it needs to run on EVERY
tool result, every time, not only when the model "decides" to check. The
agent (app/agent.py) calls the matching verify_* function automatically
right after every tool execution; it is not something the LLM can skip.

Each verify_* function returns:
    {"valid": bool, "issues": [str, ...]}
`valid=False` with a non-empty issues list tells the agent the result must
not be used to answer the user, and should trigger a retry or a graceful
"I don't have a reliable answer" response instead.
"""

from __future__ import annotations

_ALLOWED_SENTIMENTS = {"Positive", "Neutral", "Negative"}

# Column-name substrings that mark a value as an aggregate (AVG/SUM/MIN/MAX -
# unlike COUNT, these return NULL, not 0, when the WHERE clause matched zero
# underlying rows). A NULL here almost always means the generated SQL's
# filter value (e.g. an exact Question= match) doesn't exist verbatim in the
# data - most often because the model guessed at free text instead of using
# a real stored value - not a legitimate "the average of nothing" answer.
_AGGREGATE_NAME_HINTS = ("avg", "sum", "min", "max", "mean")


def _result(valid: bool, issues: list[str]) -> dict:
    return {"valid": valid, "issues": issues}


def verify_query_database(result: dict) -> dict:
    issues: list[str] = []
    if not result.get("ok"):
        issues.append(f"SQL execution failed: {result.get('error')}")
        return _result(False, issues)

    if not result.get("sql", "").strip():
        issues.append("No SQL was generated.")

    rows = result.get("rows", [])
    columns = result.get("columns", [])
    if columns and rows:
        row_keys = set(rows[0].keys())
        if not row_keys.issubset(set(columns)) and not set(columns).issubset(row_keys):
            issues.append("Row keys do not match reported columns.")

    null_aggregate_issues = [
        f"Aggregate column '{key}' is NULL - the query's filter almost certainly "
        f"matched zero underlying rows (e.g. an exact-match value that isn't a real "
        f"stored value). Do not report this as a real answer - find the correct "
        f"filter value (list distinct values if unsure) and retry."
        for row in rows
        for key, value in row.items()
        if value is None and any(hint in key.lower() for hint in _AGGREGATE_NAME_HINTS)
    ]
    if null_aggregate_issues:
        return _result(False, null_aggregate_issues)

    notes = list(result.get("notes") or [])

    if result.get("row_count", 0) == 0:
        # Not necessarily wrong (a legitimately empty answer is possible),
        # but the agent should say so explicitly rather than guess - flag it
        # as a soft issue the caller can choose to surface, not a hard fail.
        issues.append("Query returned zero rows - confirm this is expected before answering.")
        return _result(True, issues + notes)

    return _result(True, issues + notes)


def verify_analytics(result: dict) -> dict:
    issues: list[str] = []
    if not result.get("ok"):
        issues.append(f"Analytics computation failed: {result.get('error')}")
        return _result(False, issues)

    op = result.get("operation")
    data = result.get("data", [])

    # "Hard" issues mean the numbers are internally inconsistent / impossible
    # and must block the answer. "Soft" issues (e.g. legitimately empty
    # results) are surfaced but don't block - a true zero/empty answer is
    # still a valid answer.
    hard_issues: list[str] = []

    if op == "percentage":
        for row in data:
            pct = row.get("percentage")
            if pct is None or not (0.0 <= pct <= 100.0):
                hard_issues.append(f"Implausible percentage value: {pct}")
            if row.get("matching_count", 0) > row.get("total_count", 0):
                hard_issues.append("matching_count exceeds total_count.")

    if op == "rating_distribution":
        total_pct = sum(row.get("percentage", 0) for row in data)
        if data and not (95.0 <= total_pct <= 105.0):
            hard_issues.append(f"Rating distribution percentages sum to {total_pct}, expected ~100.")

    if op in ("average_by_group", "trend"):
        for row in data:
            avg = row.get("avg_rating")
            if avg is not None and not (1.0 <= avg <= 5.0):
                hard_issues.append(f"Implausible avg_rating value: {avg} (expected 1-5).")

    if not data:
        issues.append("Analytics returned no data - confirm this is expected before answering.")

    # Surface any filter-resolution notes from _apply_filters (e.g. a
    # semantic fallback substitution, or a filter value that matched
    # nothing even after semantic lookup) so the agent sees exactly what
    # happened rather than just a raw number.
    issues = hard_issues + issues + list(result.get("notes") or [])
    return _result(len(hard_issues) == 0, issues)


def verify_retrieval(result: dict, query: str) -> dict:
    issues: list[str] = []
    notes = list(result.get("notes") or [])
    results = result.get("results", [])
    if not results:
        issues.append(f"No employee comments were retrieved for query: '{query}'.")
        return _result(False, issues + notes)

    for r in results:
        if not r.get("text", "").strip():
            issues.append(f"Retrieved node {r.get('response_id')} has empty text.")
        if r.get("score") is not None and r["score"] < 0:
            issues.append(f"Suspicious negative similarity score on {r.get('response_id')}.")

    return _result(True, issues + notes)


def verify_sentiment(result: dict) -> dict:
    issues: list[str] = []
    items = result.get("results", [])
    if not items:
        issues.append("No sentiment results were produced.")
        return _result(False, issues)

    for item in items:
        label = item.get("sentiment")
        if label not in _ALLOWED_SENTIMENTS:
            issues.append(f"Invalid sentiment label '{label}' for response {item.get('response_id')}.")

    valid = all(item.get("sentiment") in _ALLOWED_SENTIMENTS for item in items)
    return _result(valid, issues)


_VERIFIERS = {
    "query_database": verify_query_database,
    "run_analytics": verify_analytics,
    "search_employee_comments": verify_retrieval,
    "analyze_sentiment": verify_sentiment,
}


def verify_tool_result(tool_name: str, tool_output: dict, **extra) -> dict:
    """Dispatch to the right verifier by tool name. Unknown tools pass
    through as valid (nothing to check), rather than blocking the agent."""
    verifier = _VERIFIERS.get(tool_name)
    if verifier is None:
        return _result(True, [])
    if tool_name == "search_employee_comments":
        return verifier(tool_output, extra.get("query", ""))
    return verifier(tool_output)
