"""
Central configuration for the Employee Engagement Chatbot POC.

Everything that varies between environments (API keys, file paths, model
names) is read from environment variables (loaded from a local .env file).
Nothing here should ever be hard-coded with a real secret.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder this file's parent (app/) lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root regardless of the current working directory.
load_dotenv(PROJECT_ROOT / ".env")


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# --- Groq (chat completions: the tool-calling agent, SQL generation,
# sentiment classification) - the only API key this project needs. ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")

# --- Embeddings (comment retrieval only). Runs locally via HuggingFace/
# sentence-transformers - no API key, no external service. ---
EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# --- Data ---
DATA_FILE: Path = _resolve(os.getenv("DATA_FILE", "data/employee_engagement_5000.xlsx"))
SQLITE_DB_PATH: Path = _resolve(os.getenv("SQLITE_DB_PATH", "db/engagement.db"))
INDEX_STORAGE_DIR: Path = _resolve(os.getenv("INDEX_STORAGE_DIR", "storage"))

# --- Agent behaviour ---
AGENT_MAX_TOOL_ITERATIONS: int = int(os.getenv("AGENT_MAX_TOOL_ITERATIONS", "5"))

# The single table our structured data lives in.
SQL_TABLE_NAME = "engagement"

# Hard cap on rows returned by any SQL query, regardless of what the
# generated SQL asks for. Protects the LLM context window and the API
# response from being flooded by a badly-scoped query.
SQL_MAX_ROWS = 200


def require_groq_key() -> str:
    """Raise a clear, early error if the Groq API key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill "
            "in a real key from https://console.groq.com/keys."
        )
    return GROQ_API_KEY
