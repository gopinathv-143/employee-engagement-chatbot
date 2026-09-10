"""
SQL Generator tool.

Turns a natural-language question into a single SQLite SELECT statement,
using Groq with the table schema and a few worked examples as grounding.

Deliberately separate from sql_executor.py: this module only ever *produces
text*. It never touches the database. That separation is what lets the
agent (or a human) swap in a different SQL-writing strategy later without
touching execution/safety code at all.
"""

from __future__ import annotations

import re

from app import config
from app.data_processing import get_schema_description
from app.groq_client import chat_complete

_FEW_SHOT = """Examples:

Q: How many employees are there in total?
SQL: SELECT COUNT(DISTINCT Response_ID) AS employee_count FROM engagement;

Q: What is the average rating for the Compensation & Benefits theme?
SQL: SELECT AVG(Rating) AS avg_rating FROM engagement WHERE Theme = 'Compensation & Benefits';

Q: How many responses are Very Dissatisfied in the Finance department?
SQL: SELECT COUNT(*) AS count FROM engagement WHERE Department = 'Finance' AND Employee_Feedback = 'Very Dissatisfied';

Q: Show the average rating per department, highest first.
SQL: SELECT Department, AVG(Rating) AS avg_rating FROM engagement GROUP BY Department ORDER BY avg_rating DESC;

Q: How has average rating trended by month for Work-Life Balance?
SQL: SELECT Response_Month, AVG(Rating) AS avg_rating FROM engagement WHERE Theme = 'Work-Life Balance' GROUP BY Response_Month ORDER BY Response_Month;
"""

_SYSTEM_PROMPT_TEMPLATE = """You are a SQL generator for a SQLite database of employee \
engagement survey responses.

{schema}

{few_shot}

Rules:
- Output ONLY the SQL statement. No markdown code fences, no explanation, no comments.
- Exactly one SELECT (or WITH ... SELECT) statement, ending with a semicolon.
- Never use INSERT/UPDATE/DELETE/DROP/ALTER or any other write statement.
- Use single quotes for string literals and match the exact category values \
given in the schema notes when possible.
"""

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_sql(text: str) -> str:
    """Strip markdown fences / chatty wrapping the LLM might add anyway."""
    text = text.strip()
    fence_match = _SQL_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # If the model added a leading "SQL:" label, drop it.
    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def generate_sql(question: str, error_feedback: str | None = None) -> str:
    """Generate a single SQLite SELECT statement for `question`.

    If `error_feedback` is given (the error message from a previously failed
    execution attempt), it is included so the model can self-correct - this
    is what the agent uses for its one automatic retry.
    """
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        schema=get_schema_description(), few_shot=_FEW_SHOT
    )

    user_content = f"Question: {question}"
    if error_feedback:
        user_content += (
            f"\n\nYour previous SQL failed with this error:\n{error_feedback}\n"
            f"Write a corrected SQL statement that fixes this problem."
        )

    response = chat_complete(
        model=config.GROQ_CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _extract_sql(raw)
