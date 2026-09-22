"""
Streamlit frontend for the Employee Engagement Chatbot.

Talks only to the FastAPI backend's /health and /chat endpoints - no
business logic lives here. The one thing this file is responsible for is
presentation: turning the raw tool_trace (SQL, analytics data, retrieved
comments, sentiment labels, and the deterministic verification result for
each) into something a non-technical reader can scan in a few seconds,
instead of a JSON blob.
"""

import random
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

# Rotated underneath the elapsed-time counter while a job is running, so a
# slow (but healthy) answer reads as active progress rather than a stuck
# spinner - purely cosmetic, cycled by elapsed time, not tied to real stages.
THINKING_PHRASES = [
    "Reading through survey responses",
    "Crunching the numbers",
    "Cross-checking the data",
    "Verifying the result",
]

USER_AVATAR = "🧑‍💼"
ASSISTANT_AVATAR = "✨"

st.set_page_config(
    page_title="Employee Engagement Chatbot",
    page_icon="💬",
    layout="wide",
)

# -----------------------------
# Styling
#
# Scoped to elements this file adds (header banner, chat bubbles, sidebar,
# metric cards) plus a light re-skin of Streamlit's own chrome via its
# stable data-testid hooks - kept to colors/spacing/typography so it holds
# up in both light and dark theme.
# -----------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class^="st-"], [class*=" st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    /* Streamlit's own icon glyphs (collapse arrow, expander chevron, etc.)
       are ligature text in an icon font - the rule above was overriding
       that font and leaving the literal ligature name ("keyboard_double_
       arrow_left") visible instead of the icon. */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    :root {
        --brand-1: #4f46e5;
        --brand-2: #7c3aed;
        --brand-soft: rgba(124, 58, 237, 0.08);
        --brand-border: rgba(124, 58, 237, 0.18);
    }

    /* App background - plain white */
    .stApp, [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }
    [data-testid="stBottom"] > div { background: #ffffff; }
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: #ffffff !important;
    }

    /* Clean up Streamlit's own chrome for a demo-ready look */
    [data-testid="stAppDeployButton"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    /* Streamlit adds a hover "link to heading" icon to every h1-h6 (incl.
       raw HTML ones in st.markdown) - not useful outside a docs page. */
    [data-testid="stHeaderActionElements"] { display: none; }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.35); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.55); }

    /* ---------- Header banner ---------- */
    .app-header {
        position: relative;
        overflow: hidden;
        padding: 1.6rem 1.9rem;
        border-radius: 18px;
        background: linear-gradient(135deg, var(--brand-1) 0%, var(--brand-2) 100%);
        color: #ffffff;
        margin-bottom: 1.3rem;
        box-shadow: 0 10px 30px -12px rgba(79, 70, 229, 0.55);
    }
    .app-header::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .app-header p {
        margin: 0.4rem 0 0;
        opacity: 0.94;
        font-size: 0.95rem;
        max-width: 640px;
        line-height: 1.45;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #f5f4fb;
        border-right: 1px solid var(--brand-border);
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--brand-2);
        margin-top: 0.2rem;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        border-radius: 10px;
        border: 1px solid var(--brand-border);
        background: #ffffff;
        text-align: left;
        justify-content: flex-start;
        font-size: 0.86rem;
        padding: 0.5rem 0.8rem;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        border-color: var(--brand-2);
        background: var(--brand-soft);
        color: var(--brand-2);
        transform: translateX(2px);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.4rem;
    }
    .sidebar-brand .icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
    }
    .sidebar-brand .title {
        font-weight: 700;
        font-size: 1.0rem;
        color: #1f2937;
        line-height: 1.1;
    }
    .sidebar-brand .subtitle {
        font-size: 0.72rem;
        color: #6b7280;
    }
    .history-item {
        display: flex;
        gap: 0.55rem;
        padding: 0.4rem 0.1rem;
        font-size: 0.82rem;
        color: #3f3f46;
        line-height: 1.35;
    }
    .history-item .bullet {
        flex-shrink: 0;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: var(--brand-soft);
        color: var(--brand-2);
        font-size: 0.68rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0.1rem;
    }
    .history-empty {
        font-size: 0.82rem;
        color: #9ca3af;
        font-style: italic;
    }

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.85rem 1.05rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(0,0,0,0.05);
        animation: fadeIn 0.25s ease;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--brand-soft);
        border-color: var(--brand-border);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #ffffff;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, var(--brand-1), var(--brand-2)) !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #f59e0b, #ea580c) !important;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
    }

    /* ---------- Headline metric cards ---------- */
    [data-testid="stMetric"] {
        background: var(--brand-soft);
        border-left: 4px solid var(--brand-2);
        border-radius: 10px;
        padding: 0.7rem 0.9rem 0.5rem;
    }
    [data-testid="stMetricValue"] { color: var(--brand-2); font-weight: 700; }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] { border-radius: 12px; }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align: center;
        padding: 2.6rem 1.5rem 1.8rem;
        color: #4b5563;
    }
    .empty-state .icon {
        font-size: 2.4rem;
        margin-bottom: 0.6rem;
    }
    .empty-state-title {
        margin: 0 0 0.35rem;
        color: #1f2937;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .empty-state p {
        margin: 0 auto 1.1rem;
        max-width: 420px;
        font-size: 0.9rem;
        color: #6b7280;
    }

    /* ---------- Thinking indicator ---------- */
    .thinking {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #4b5563;
        font-size: 0.92rem;
    }
    .thinking .dots span {
        display: inline-block;
        width: 6px;
        height: 6px;
        margin-right: 3px;
        border-radius: 50%;
        background: var(--brand-2);
        animation: bounce 1.1s infinite ease-in-out both;
    }
    .thinking .dots span:nth-child(1) { animation-delay: -0.24s; }
    .thinking .dots span:nth-child(2) { animation-delay: -0.12s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
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

# A larger pool than we display at once, spanning the same tool categories
# as before (a simple average, a percentage, a trend, a comment search, a
# sentiment read) - `get_example_questions` below draws a random 5 from this
# each time the app process starts, so the sidebar doesn't show the exact
# same five prompts on every run without risking a question the backend
# can't actually answer (these all map to real Theme/Rating columns).
EXAMPLE_QUESTION_POOL = [
    "What is the average employee rating?",
    "What percentage of employees are dissatisfied with compensation?",
    "How has average rating trended by month?",
    "What are employees saying about workload?",
    "What is the sentiment regarding leadership?",
    "What is the average rating for work-life balance?",
    "Which theme has the lowest average rating?",
    "What percentage of employees rated workplace safety below 3?",
    "What are employees saying about career growth opportunities?",
    "What is the sentiment regarding manager support?",
    "How has the average rating for employee wellbeing trended by month?",
    "What percentage of employees are satisfied with recognition and rewards?",
    "What are employees saying about communication from leadership?",
    "What is the average rating by department?",
    "What is the sentiment regarding compensation and benefits?",
]


@st.cache_resource
def get_example_questions() -> list[str]:
    """Picked once per running app process (cache_resource persists for the
    server's lifetime across sessions/reruns, and resets on restart) so the
    prompts refresh each time the project is (re)started, without shuffling
    under a user's feet on every click within one run."""
    return random.sample(EXAMPLE_QUESTION_POOL, k=5)


EXAMPLE_QUESTIONS = get_example_questions()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="icon">💬</div>
            <div>
                <div class="title">Engagement Assistant</div>
                <div class="subtitle">Survey insights, on demand</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Question history")

    asked_questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if asked_questions:
        history_html = "".join(
            f'<div class="history-item"><div class="bullet">{i}</div><div>{q}</div></div>'
            for i, q in enumerate(asked_questions, start=1)
        )
        st.markdown(history_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="history-empty">No questions asked yet.</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try asking")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example::{q}", width="stretch"):
            st.session_state.pending_question = q


def _is_plain_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def render_headline(tool_trace: list, gave_up: bool = False) -> None:
    """Surface the single most decision-relevant number/chart from this
    turn immediately below the answer, with no click required - an HR
    reader scanning many answers needs the number-in-context up front.

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
# Conversation history / empty state
# -----------------------------

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">💬</div>
            <div class="empty-state-title">What would you like to know?</div>
            <p>Ask a question below, or pick one from the sidebar, to get a
            verified answer straight from the survey data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("gave_up"):
            st.warning(
                "This response is incomplete - see the message above for why, "
                "and what to do next.",
                icon="⚠️",
            )
        if message.get("tool_trace"):
            render_headline(message["tool_trace"], gave_up=message.get("gave_up", False))


# -----------------------------
# Ask the backend and render the exchange
# -----------------------------


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        status = st.empty()
        status.markdown(
            '<div class="thinking">Starting'
            '<span class="dots"><span></span><span></span><span></span></span></div>',
            unsafe_allow_html=True,
        )

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

            phrase = THINKING_PHRASES[int(elapsed // 4) % len(THINKING_PHRASES)]
            status.markdown(
                f'<div class="thinking">{phrase} · {int(elapsed)}s'
                '<span class="dots"><span></span><span></span><span></span></span></div>',
                unsafe_allow_html=True,
            )

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
