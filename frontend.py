import streamlit as st
import requests

# -----------------------------
# Configuration
# -----------------------------

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Employee Engagement Chatbot",
    page_icon="💬",
    layout="centered"
)

# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Header
# -----------------------------

st.title("💬 Employee Engagement Chatbot")

st.caption(
    "AI-powered employee engagement analysis using "
    "LlamaIndex, sentiment analysis and an agentic backend."
)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("System Status")

    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5
        )

        if response.status_code == 200:
            health = response.json()

            st.success("Backend Online")

            st.write(
                f"**Database:** "
                f"{'Available' if health.get('db_exists') else 'Unavailable'}"
            )

            st.write(
                f"**LlamaIndex:** "
                f"{'Available' if health.get('index_exists') else 'Unavailable'}"
            )

            st.write(
                f"**Chat Model:** "
                f"{health.get('chat_model', 'Unknown')}"
            )

            st.write(
                f"**Embedding Model:** "
                f"{health.get('embed_model', 'Unknown')}"
            )

        else:
            st.error("Backend returned an error")

    except requests.exceptions.RequestException:
        st.error("Backend Offline")

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Example Questions")

    st.write("• How many employees are there?")
    st.write("• What is the average employee rating?")
    st.write("• What are employees saying about workload?")
    st.write("• What is the sentiment regarding leadership?")
    st.write("• What are the main employee concerns?")


# -----------------------------
# Display Previous Messages
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

user_question = st.chat_input(
    "Ask about employee engagement..."
)


if user_question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Prepare conversation history
    history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages[:-1]
    ]

    # Call backend
    with st.chat_message("assistant"):

        with st.spinner("Analyzing employee engagement data..."):

            try:

                payload = {
                    "message": user_question,
                    "history": history
                }

                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:

                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer was returned."
                    )

                    st.markdown(answer)

                    # Save assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    # Optional tool trace
                    tool_trace = result.get("tool_trace", [])

                    if tool_trace:

                        with st.expander("🔍 View Analysis Details"):

                            st.json(tool_trace)

                else:

                    st.error(
                        f"Backend error: "
                        f"{response.status_code}"
                    )

                    st.code(response.text)

            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the backend. "
                    "Make sure FastAPI is running on port 8000."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )