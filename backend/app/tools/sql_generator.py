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
from app.tools import schema_index

_FEW_SHOT = """Examples:

Q: How many employees are there in total?
SQL: SELECT COUNT(DISTINCT Response_ID) AS employee_count FROM engagement;

Q: What is the average rating for the Compensation & Benefits theme?
SQL: SELECT AVG(Rating) AS avg_rating, COUNT(*) AS response_count FROM engagement WHERE Theme = 'Compensation & Benefits';

Q: How many responses are Very Dissatisfied in the Finance department?
SQL: SELECT COUNT(*) AS count FROM engagement WHERE Department = 'Finance' AND Employee_Feedback = 'Very Dissatisfied';

Q: Show the average rating per department, highest first.
SQL: SELECT Department, AVG(Rating) AS avg_rating, COUNT(*) AS response_count FROM engagement GROUP BY Department ORDER BY avg_rating DESC;

Q: How has average rating trended by month for Work-Life Balance?
SQL: SELECT Response_Month, AVG(Rating) AS avg_rating, COUNT(*) AS response_count FROM engagement WHERE Theme = 'Work-Life Balance' GROUP BY Response_Month ORDER BY Response_Month;
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
- The user's question may be prefixed with a "Semantic grounding" block listing \
the real stored Question/Theme/Department/Role values closest to the question. \
When present, use the exact string from that block for any filter it applies to \
- do not write your own paraphrase of it.
- Whenever you compute AVG(...), also SELECT a COUNT(*) (or COUNT(DISTINCT \
Response_ID) if the query already groups rows) alongside it, aliased something \
like response_count - a bare average with no sample size isn't useful to an HR \
reader.
"""

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

# Columns worth grounding with semantic candidates: Question is free text
# with no shared vocabulary to lean on; Theme/Department/Role are
# high-cardinality-enough (15/36/18 values) that a paraphrase ("pay
# fairness", "the accounts team") won't share tokens with the real value.
_GROUNDING_COLUMNS = [
    ("Question", "Closest matching survey Question values", 5),
    ("Theme", "Closest matching Theme values", 3),
    ("Department", "Closest matching Department values", 3),
    ("Role", "Closest matching Role values", 3),
]


def _semantic_grounding_block(question: str) -> str:
    """Rank each grounded column's real stored values by similarity to the
    user's question, so the model can copy an exact value instead of
    guessing/paraphrasing it into a WHERE clause that will match nothing.
    Best-effort: if the embedding model isn't available for some reason,
    generation still proceeds without grounding rather than failing."""
    lines = []
    question_top_score = None
    for column, label, top_k in _GROUNDING_COLUMNS:
        try:
            matches = schema_index.resolve(column, question, top_k=top_k)
        except Exception:
            continue
        if not matches:
            continue
        if column == "Question":
            question_top_score = matches[0].score
        rendered = "; ".join(f"'{m.value}' (similarity {m.score:.2f})" for m in matches)
        lines.append(f"  {label}: {rendered}")

    if not lines:
        return ""

    block = (
        "\n\nSemantic grounding - these are the REAL stored values ranked by "
        "similarity to the question above. If the question is asking about one of "
        "these, you MUST copy the exact string shown (spelling and casing) into your "
        "WHERE clause - never paraphrase, shorten, or invent a Question/Theme/"
        "Department/Role value:\n" + "\n".join(lines)
    )

    # A full sentence can never equal a stored Question verbatim, so when
    # even the BEST candidate is a weak match, writing
    # `WHERE Question = '<the user's own wording>'` is guaranteed to return
    # nothing - it's not a real attempt, just a wasted round trip (the topic
    # likely isn't in the data at all, e.g. a team/department that isn't a
    # real Theme/Department). Tell the model explicitly rather than leaving
    # it to infer this from scores alone.
    if question_top_score is not None and question_top_score < schema_index.DEFAULT_MATCH_THRESHOLD:
        block += (
            f"\n\nNone of the Question candidates above are a confident match "
            f"(best similarity {question_top_score:.2f}, below "
            f"{schema_index.DEFAULT_MATCH_THRESHOLD:.2f}). Do NOT write "
            f"WHERE Question = '<the user's own wording>' - that is guaranteed to "
            f"match nothing. Either use Question LIKE '%keyword%' with one short, "
            f"distinctive keyword from the topic, or, if the topic genuinely has no "
            f"matching survey data, write a query that makes that clear (e.g. "
            f"SELECT DISTINCT Question FROM engagement WHERE Question LIKE "
            f"'%keyword%') instead of guessing at an exact-match filter."
        )

    return block


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

    user_content = f"Question: {question}" + _semantic_grounding_block(question)
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
