"""
LlamaIndex Retrieval tool.

Thin wrapper around app.indexing so the agent's tool registry has one
obvious place per tool, matching sql_generator / sql_executor / analytics /
sentiment / verification. All the actual index build/load/query logic
lives in app.indexing (it's substantial enough, and reused at startup, to
deserve its own module) - this file is the stable "tool" entry point.
"""

from __future__ import annotations

from app.indexing import RetrievalResult, search_employee_comments

__all__ = ["search_employee_comments", "RetrievalResult"]
