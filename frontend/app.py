"""
Streamlit frontend for the Employee Engagement Chatbot.

Talks only to the FastAPI backend's /health and /chat endpoints - no
business logic lives here. The one thing this file is responsible for is
presentation: turning the raw tool_trace (SQL, analytics data, retrieved
comments, sentiment labels, and the deterministic verification result for
each) into something a non-technical reader can scan in a few seconds,
instead of a JSON blob.
"""

import time

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

# A multi-tool question (retrieval + sentiment, say) makes several sequential
# Groq calls, each with its own rate-limit retry/backoff - under sustained
# rate limiting that can legitimately take minutes even though it eventually
# succeeds. Rather than one blocking POST /chat with a single fixed timeout
# (which either cuts off slow-but-succeeding turns or waits too long on a
# truly hung one), the backend runs the turn as a job (POST /chat/jobs) and
# this polls its status (GET /chat/jobs/{id}) with short, cheap requests -
# so it can wait much longer overall without needing a long-lived connection.
CHAT_JOB_POLL_INTERVAL_SECONDS = 1.5
CHAT_JOB_MAX_WAIT_SECONDS = 600

st.set_page_config(
    page_title="Employee Engagement Chatbot",
    page_icon="💬",
    layout="wide",
)

# -----------------------------
# Styling
#
# Scoped to elements this file adds (header banner, badges, tool cards) -
# deliberately does not touch Streamlit's own chrome (.stApp background,
# sidebar, chat bubbles), so it looks right in both light and dark theme.
# -----------------------------

st.markdown(
    """
    <style>
    .app-header {
        padding: 1.35rem 1.75rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        margin-bottom: 1.1rem;
    }
    .app-header h1 { margin: 0; font-size: 1.55rem; }
    .app-header p { margin: 0.35rem 0 0; opacity: 0.92; font-size: 0.92rem; }

    .badge {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .badge-ok   { background: #dcfce7; color: #166534; }
    .badge-warn { background: #fef9c3; color: #854d0e; }
    .badge-fail { background: #fee2e2; color: #991b1b; }

    .tool-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .tool-card-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .comment-card {
        border-left: 3px solid #7c3aed;
        padding: 0.4rem 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
        background: rgba(124, 58, 237, 0.06);
    }
    .comment-meta {
        font-size: 0.78rem;
        opacity: 0.75;
        margin-bottom: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# -----------------------------
# Header
# -----------------------------

st.markdown(
    """
    <div class="app-header">
        <h1>💬 Employee Engagement Chatbot</h1>
        <p>Ask about ratings, trends, and what employees are actually saying -
        every number is pulled live from the survey data and independently
        verified before you see it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------

EXAMPLE_QUESTIONS = [
    "What is the average employee rating?",
    "What percentage of employees are dissatisfied with compensation?",
    "How has average rating trended by month?",
    "What are employees saying about workload?",
    "What is the sentiment regarding leadership?",
]

with st.sidebar:
    st.subheader("System status")

    try:
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        health = health_response.json() if health_response.status_code == 200 else None
    except requests.exceptions.RequestException:
        health = None

    if health is None:
        st.error("Backend offline")
    else:
        st.success("Backend online")
        cols = st.columns(2)
        cols[0].metric("Survey responses", health.get("response_count") or "—")
        cols[1].metric(
            "Search index",
            "Ready" if health.get("index_exists") else "Missing",
        )
        st.caption(f"Chat model: `{health.get('chat_model', 'unknown')}`")
        st.caption(f"Embedding model: `{health.get('embed_model', 'unknown')}`")

    st.divider()

    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try asking")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example::{q}", width="stretch"):
            st.session_state.pending_question = q

# -----------------------------
# Tool-trace rendering
#
# One block per tool the agent called, showing exactly what it did and
# whether the deterministic verification step approved the result - not a
# JSON dump of the raw API response.
# -----------------------------

TOOL_ICONS = {
    "query_database": "🗄️",
    "run_analytics": "📊",
    "search_employee_comments": "🔎",
    "analyze_sentiment": "🙂",
}
TOOL_LABELS = {
    "query_database": "Database query",
    "run_analytics": "Analytics",
    "search_employee_comments": "Comment search",
    "analyze_sentiment": "Sentiment analysis",
}
SENTIMENT_DOT = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}


def verification_badge(verification: dict) -> str:
    valid = verification.get("valid", True)
    has_issues = bool(verification.get("issues"))
    if valid and not has_issues:
        return '<span class="badge badge-ok">✅ Verified</span>'
    if valid and has_issues:
        return '<span class="badge badge-warn">🟡 Verified · note below</span>'
    return '<span class="badge badge-fail">🔴 Failed verification</span>'


def render_query_database(entry: dict) -> None:
    result = entry["result"]
    sql = result.get("sql")
    if sql:
        st.code(sql, language="sql")
    rows = result.get("rows") or []
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    elif result.get("ok"):
        st.caption("Query ran successfully but returned no rows.")
    if result.get("error"):
        st.error(result["error"])
    for note in result.get("notes") or []:
        st.info(note, icon="🧭")


def render_run_analytics(entry: dict) -> None:
    result = entry["result"]
    op = result.get("operation", "")
    data = result.get("data") or []
    st.caption(f"Operation: `{op}`")

    if not data:
        st.caption("No matching data for this filter.")
    else:
        df = pd.DataFrame(data)
        st.dataframe(df, width="stretch", hide_index=True)
        try:
            if op == "percentage" and "percentage" in df.columns:
                st.metric("Percentage", f"{data[0]['percentage']}%")
            elif op == "average_by_group" and "avg_rating" in df.columns:
                label_col = next(c for c in df.columns if c not in ("avg_rating", "count"))
                st.bar_chart(df.set_index(label_col)["avg_rating"])
            elif op == "count_by_group" and "count" in df.columns:
                label_col = next(c for c in df.columns if c != "count")
                st.bar_chart(df.set_index(label_col)["count"])
            elif op == "rating_distribution" and "rating" in df.columns:
                st.bar_chart(df.set_index("rating")["count"])
            elif op == "trend" and "month" in df.columns:
                y_col = "avg_rating" if "avg_rating" in df.columns else "count"
                st.line_chart(df.set_index("month")[y_col])
        except (StopIteration, KeyError):
            pass  # chart is a nice-to-have; the table above already has the numbers

    for note in result.get("notes") or []:
        st.info(note, icon="🧭")


def render_search_employee_comments(entry: dict) -> None:
    result = entry["result"]
    for note in result.get("notes") or []:
        st.info(note, icon="🧭")

    results = result.get("results") or []
    if not results:
        st.caption("No matching comments found.")
        return

    for r in results:
        score = r.get("score")
        score_txt = f" · match {score:.2f}" if isinstance(score, (int, float)) else ""
        st.markdown(
            f"""
            <div class="comment-card">
                <div class="comment-meta">
                    {r.get('department', '')} · {r.get('role', '')} ·
                    rating {r.get('rating', '?')}/5 ·
                    {r.get('employee_feedback', '')}{score_txt}
                </div>
                <div>“{r.get('text', '')}”</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analyze_sentiment(entry: dict) -> None:
    result = entry["result"]
    items = result.get("results") or []
    if not items:
        st.caption("No sentiment results.")
        return

    counts = pd.Series([i["sentiment"] for i in items]).value_counts()
    st.bar_chart(counts)

    shown = items[:6]
    for i in shown:
        dot = SENTIMENT_DOT.get(i.get("sentiment"), "⚪")
        st.markdown(f"{dot} **{i.get('sentiment')}** — {i.get('rationale', '')}")
    if len(items) > len(shown):
        st.caption(f"… and {len(items) - len(shown)} more comments classified.")


RENDERERS = {
    "query_database": render_query_database,
    "run_analytics": render_run_analytics,
    "search_employee_comments": render_search_employee_comments,
    "analyze_sentiment": render_analyze_sentiment,
}


def render_tool_trace(tool_trace: list) -> None:
    for i, entry in enumerate(tool_trace, start=1):
        tool = entry.get("tool", "")
        icon = TOOL_ICONS.get(tool, "🔧")
        label = TOOL_LABELS.get(tool, tool)
        badge = verification_badge(entry.get("verification", {}))

        st.markdown(
            f'<div class="tool-card-title">{i}. {icon} {label} &nbsp; {badge}</div>',
            unsafe_allow_html=True,
        )

        renderer = RENDERERS.get(tool)
        if renderer:
            renderer(entry)

        for issue in entry.get("verification", {}).get("issues") or []:
            st.caption(f"⚠️ {issue}")

        if i < len(tool_trace):
            st.divider()


def _is_plain_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def render_headline(tool_trace: list, gave_up: bool = False) -> None:
    """Surface the single most decision-relevant number/chart from this
    turn immediately below the answer, with no click required - the
    verification badges and full SQL/tables stay in the "How this answer
    was put together" expander for anyone who wants to audit them, but an
    HR reader scanning many answers needs the number-in-context up front.

    Two correctness rules, not just a nice-to-have:
    - If the agent gave up, show nothing here. tool_trace can still contain
      an earlier tool call that technically succeeded before a later step
      failed - headlining that number would flatly contradict the "I
      wasn't able to produce a verified answer" text right above it.
    - Walk the trace newest-first and skip anything verification didn't
      pass. A multi-iteration turn can contain a superseded early attempt
      with different (wrong) numbers before the agent corrected itself;
      the headline must match what the final answer is actually based on,
      not the first thing that happens to look numeric."""
    if gave_up:
        return

    for entry in reversed(tool_trace):
        tool = entry.get("tool")
        result = entry.get("result") or {}
        if not entry.get("verification", {}).get("valid", True):
            continue

        if tool == "run_analytics" and result.get("ok") and result.get("data"):
            op = result["operation"]
            data = result["data"]
            df = pd.DataFrame(data)

            with st.container(border=True):
                if op == "percentage" and "percentage" in df.columns:
                    row = data[0]
                    c1, c2 = st.columns([1, 2])
                    c1.metric("Result", f"{row['percentage']}%")
                    c1.caption(f"{row.get('matching_count')} of {row.get('total_count')} responses")
                    c2.progress(min(max(row["percentage"] / 100, 0.0), 1.0))
                    return
                if op == "average_by_group" and "avg_rating" in df.columns:
                    label_col = next(
                        (c for c in df.columns if c not in ("avg_rating", "count")), None
                    )
                    if label_col:
                        st.caption(f"Average rating by {label_col} (1-5 scale)")
                        st.bar_chart(df.set_index(label_col)["avg_rating"])
                        return
                if op == "count_by_group" and "count" in df.columns:
                    label_col = next((c for c in df.columns if c != "count"), None)
                    if label_col:
                        st.caption(f"Response count by {label_col}")
                        st.bar_chart(df.set_index(label_col)["count"])
                        return
                if op == "rating_distribution" and "rating" in df.columns:
                    st.caption("Rating distribution (1 = worst, 5 = best)")
                    st.bar_chart(df.set_index("rating")["count"])
                    return
                if op == "trend" and "month" in df.columns:
                    y_col = "avg_rating" if "avg_rating" in df.columns else "count"
                    st.caption("Trend over time")
                    st.line_chart(df.set_index("month")[y_col])
                    return

        if tool == "query_database" and result.get("ok") and result.get("rows"):
            rows = result["rows"]
            if len(rows) == 1:
                numeric_items = {k: v for k, v in rows[0].items() if _is_plain_number(v)}
                # A single aggregate row (AVG/COUNT) - e.g. {"avg_rating": 3.7,
                # "response_count": 145}. Multi-column detail rows (a raw
                # SELECT * lookup) fall through to the table in the expander
                # instead, since there's no single number to headline.
                if 1 <= len(numeric_items) <= 2:
                    with st.container(border=True):
                        cols = st.columns(len(numeric_items))
                        for col, (key, value) in zip(cols, numeric_items.items()):
                            label = key.replace("_", " ").title()
                            shown = round(value, 2) if isinstance(value, float) else value
                            col.metric(label, shown)
                        rating_key = next(
                            (k for k in numeric_items if "rating" in k.lower()), None
                        )
                        if rating_key:
                            st.progress(min(max(numeric_items[rating_key] / 5, 0.0), 1.0))
                            st.caption("Scale: 1 (worst) – 5 (best)")
                    return


# -----------------------------
# Conversation history
# -----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("gave_up"):
            st.warning(
                "This response is incomplete - see the message above for why, "
                "and what to do next.",
                icon="⚠️",
            )
        if message.get("tool_trace"):
            render_headline(message["tool_trace"], gave_up=message.get("gave_up", False))
            with st.expander("🔍 How this answer was put together"):
                render_tool_trace(message["tool_trace"])


# -----------------------------
# Ask the backend and render the exchange
# -----------------------------


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("_Starting..._")

        try:
            create_response = requests.post(
                f"{BACKEND_URL}/chat/jobs",
                json={"message": question, "history": history},
                timeout=15,
            )
        except requests.exceptions.ConnectionError:
            status.empty()
            st.error(
                "Could not connect to the backend. Make sure the FastAPI "
                "server is running on port 8000."
            )
            return
        except Exception as exc:  # noqa: BLE001
            status.empty()
            st.error(f"Unexpected error starting the request: {exc}")
            return

        if create_response.status_code != 200:
            status.empty()
            st.error(f"Backend error: {create_response.status_code}")
            st.code(create_response.text)
            return

        job_id = create_response.json()["job_id"]

        # Poll with short, cheap requests instead of one long blocking call -
        # a slow-but-succeeding turn (heavy rate-limit backoff on the
        # backend) is indistinguishable from a hung one to a single fixed
        # timeout, so it either cuts off good answers or waits forever on
        # bad ones. Polling lets this wait much longer in total while each
        # individual request stays fast, and shows real elapsed time instead
        # of a static spinner.
        started_at = time.time()
        job_result = None
        job_error = None
        while True:
            elapsed = time.time() - started_at
            if elapsed > CHAT_JOB_MAX_WAIT_SECONDS:
                job_error = (
                    f"Still no answer after {int(CHAT_JOB_MAX_WAIT_SECONDS)}s - the "
                    "backend may be stuck or heavily rate-limited. It may still "
                    "finish in the background; try asking again in a bit."
                )
                break

            status.markdown(f"⏳ Analyzing employee engagement data... ({int(elapsed)}s)")

            try:
                poll_response = requests.get(f"{BACKEND_URL}/chat/jobs/{job_id}", timeout=10)
            except requests.exceptions.RequestException:
                time.sleep(CHAT_JOB_POLL_INTERVAL_SECONDS)
                continue

            if poll_response.status_code != 200:
                time.sleep(CHAT_JOB_POLL_INTERVAL_SECONDS)
                continue

            payload = poll_response.json()
            if payload["status"] == "done":
                job_result = payload["result"]
                break
            if payload["status"] == "error":
                job_error = payload.get("error") or "The backend failed to produce an answer."
                break

            time.sleep(CHAT_JOB_POLL_INTERVAL_SECONDS)

        status.empty()

        if job_error is not None:
            st.error(job_error)
            return

        answer = job_result.get("answer", "No answer was returned.")
        tool_trace = job_result.get("tool_trace", [])
        gave_up = job_result.get("gave_up", False)

        st.markdown(answer)
        if gave_up:
            st.warning(
                "This response is incomplete - see the message above for why, "
                "and what to do next.",
                icon="⚠️",
            )
        if tool_trace:
            render_headline(tool_trace, gave_up=gave_up)
            with st.expander("🔍 How this answer was put together"):
                render_tool_trace(tool_trace)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "tool_trace": tool_trace,
                "gave_up": gave_up,
            }
        )


user_question = st.chat_input("Ask about employee engagement...")
if user_question:
    ask(user_question)
elif st.session_state.pending_question:
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    ask(pending)
