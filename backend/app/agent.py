"""
Step 4: The Groq agent - orchestrates tool selection, execution,
verification, and retry, then writes the final answer.

Design:
  - Groq does the reasoning (which tool(s) to call, with what arguments,
    and how to phrase the final answer). It NEVER computes a number or
    quotes a comment itself - it is only allowed to report what a tool
    returned.
  - Every tool call is followed, in Python (not by asking the LLM), by the
    matching verification function from app.tools.verification. The
    verification outcome is appended alongside the raw tool result before
    being sent back to Groq, so the model always sees "here is the data,
    and here is whether it passed sanity checks" together.
  - If a tool result fails verification, the model is told explicitly and
    is expected to either retry (e.g. a different query/operation) or,
    after AGENT_MAX_TOOL_ITERATIONS, the loop gives up and returns an
    honest "couldn't get a reliable answer" message rather than letting the
    model paper over the failure.

Orchestration is implemented as a LlamaIndex event-driven Workflow (see
`AgentWorkflow` below), which replaces the previous plain `for` loop with
explicit Event classes and `@step` methods:

    StartEvent (user_message, history)
        |
        v
    RouteEvent  <---------------------------------------------+
        |  (LLM picks a tool per TOOL_SCHEMAS, or answers)     |
        +--> FinalAnswerEvent / GiveUpEvent                    |
        |                                                      |
        +--> ToolCallEvent (one per requested tool call)       |
                   |                                           |
                   v                                           |
             ToolResultEvent (tool execution + the existing    |
                               verify_* check - this IS the    |
                               VerificationEvent step)          |
                   |                                           |
                   v                                           |
             aggregate_results (collects the whole batch,      |
                                 re-routes for another turn) ---+
        |
        v
    finalize -> StopEvent -> ChatResult (answer, tool_trace,
                                          iterations_used, gave_up)

Everything downstream of "which tool(s) got called" (tool implementations,
verification rules, retry policy, response shape) is unchanged - only the
sequencing is now expressed as LlamaIndex events/steps instead of a manual
loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import groq
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from app import config
from app.groq_client import chat_complete
from app.tools import analytics, query_database, retrieval, sentiment, verification

_SERVICE_UNAVAILABLE_MESSAGE = (
    "The AI service is temporarily at capacity and couldn't process this "
    "request. This isn't about your question - please try again in a few "
    "minutes."
)

SYSTEM_PROMPT = """You are an HR analytics assistant for an employee engagement \
survey dataset. You answer questions ONLY using the tools available to you - you \
must never invent numbers, percentages, counts, trends, or employee quotes.

Scope check - do this BEFORE choosing a tool: this dataset only covers survey \
responses (Department, Role, Theme, Question, Rating, Employee_Feedback, free-text \
Comment, Respondent_Type, Company, Response_Date). If the question is not about this \
survey data at all - general knowledge, small talk, requests unrelated to workplace/ \
HR topics, or a question addressed to you personally as if you were an employee \
("your experience", "your opinion") with no reasonable reading as "what do the \
survey/comments say" - do NOT call any tool. Answer directly, in one or two \
sentences: say plainly that you can only answer questions about this employee \
engagement survey (ratings, departments, themes, comments, sentiment), and invite \
the user to ask something in that scope. Do not spend tool calls guessing at an \
interpretation that forces an unrelated question into a tool it doesn't fit.

Tool selection guide:
- Counts, totals, simple lookups, group-bys, filters -> query_database
- Percentages, averages, rating distributions, trends over time -> run_analytics \
  (prefer this over query_database for "percentage X" and "average rating by Y" \
  questions - it is computed deterministically, not written as freeform SQL)
- A simple average rating for one specific survey question (especially when the \
    question text is quoted) -> query_database. Do not use run_analytics with \
    group_by="Question"; Question is not an analytics group-by.
- "What do employees say about X", concerns, opinions, complaints, themes -> \
  search_employee_comments, and then call analyze_sentiment on the retrieved \
  comments' texts before describing the overall sentiment/tone
- "Why", "reasons", "explain" as a FOLLOW-UP about a rating/percentage/trend \
  you already reported earlier in this conversation -> identify the specific \
  theme/department/filter that number was about (from your own prior answer, \
  or the filters you used in the tool call that produced it) and call \
  search_employee_comments with that same filter, then analyze_sentiment on \
  the results. Do not just restate the earlier number - a "why" question \
  needs real comment text as evidence, not a repeated statistic. If nothing \
  in the conversation so far establishes what "the rating" refers to, ask \
  the user to clarify rather than guessing at a filter.
- If a question needs more than one of these (e.g. "what's the sentiment on \
  compensation, and what percentage rated it below 3"), call multiple tools \
  and combine their verified results.

After every tool call you will receive the raw result AND a "verification" \
block computed independently of you. If verification says a result is not \
valid, do NOT use it to answer - either adjust your tool call (different \
operation/filters/wording) and try again, or, if you are out of reasonable \
options, tell the user plainly that you could not find a reliable answer \
instead of guessing.

Once a tool call has PASSED verification against a Question/Theme that is \
genuinely related to what the user asked - even if it isn't an exact wording \
match - stop searching and use it. Compute the actual number from it and \
answer, stating plainly that it's the closest matching survey item rather \
than a literal match (e.g. "The survey doesn't ask that exact question, but \
the closest related one is 'X', averaging Y"). Continuing to search for a \
more perfectly-worded match instead of using a good verified result you \
already have risks running out of attempts and giving up with nothing to \
show, which is worse than an honest, clearly-caveated partial answer.

When you do have verified results, write an answer an HR reader can act on \
without opening the underlying data - not just a bare number. Do not show raw \
JSON to the user. Rating is an INTEGER on a 1-5 scale (1 worst, 5 best), so \
report database averages on that scale. Only provide a 0-10 equivalent when \
the user explicitly requests it, calculated as the verified 1-5 average \
multiplied by 2, and label it as a 0-10 equivalent.

Whenever a tool result includes it, state the sample size behind a number \
(e.g. "3.7 out of 5, based on 145 responses") - a number with no denominator \
isn't decision-grade for HR. Briefly characterize where it falls (e.g. "on \
the higher end", "roughly average", "a clear concern area") rather than \
leaving the reader to judge a bare figure in isolation. If the result breaks \
down by a group (department, theme, month, etc.), call out the highest and \
lowest instead of just repeating the average across the board, and mention \
that the full breakdown is visible below. Stay to 3-5 sentences - add \
substance (n, context, the standout group), not filler.
"""

_SPECIFIC_QUESTION_AVERAGE_RE = re.compile(
    r"\b(?:average|mean)\s+rating\b.*\bfor\b.*(?:['\"].+['\"]|\bquestion\b)",
    re.IGNORECASE | re.DOTALL,
)


def _is_specific_question_average(user_message: str) -> bool:
    """Identify averages for one survey question before LLM tool selection."""
    return bool(_SPECIFIC_QUESTION_AVERAGE_RE.search(user_message))


_NO_DATA_ISSUE_MARKERS = (
    "no employee comments were retrieved",
    "query returned zero rows",
    "analytics returned no data",
)


def _default_give_up_message(trace: list[dict]) -> str:
    """Pick a give-up message that matches what actually happened, instead of
    one generic sentence for every failure. When every attempt in the trace
    came back with a "found nothing" issue (empty retrieval/rows/analytics -
    as opposed to a malformed query, an implausible number, or some other
    real error), the honest and more actionable message is "this doesn't
    look covered by the data", not "please rephrase" - rephrasing a question
    about a topic that genuinely isn't in the dataset won't help."""
    all_issues = [
        issue
        for entry in trace
        for issue in (entry.get("verification", {}).get("issues") or [])
    ]
    every_entry_has_issues = trace and all(
        entry.get("verification", {}).get("issues") for entry in trace
    )
    if every_entry_has_issues and all_issues and all(
        any(marker in issue.lower() for marker in _NO_DATA_ISSUE_MARKERS)
        for issue in all_issues
    ):
        return (
            "I couldn't find anything in the survey data covering that - no matching "
            "comments, rows, or metrics. This assistant only answers from what's "
            "actually in the survey (departments, roles, themes like compensation or "
            "management, ratings, and free-text comments), so this topic may simply "
            "not be represented in it. Try a different topic, or a specific "
            "department/theme/time period."
        )
    return (
        "I wasn't able to produce a fully verified answer to that question "
        "after several attempts. Could you rephrase it, or narrow it down "
        "(e.g. a specific department, theme, or time period)?"
    )

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Answer a natural-language question by generating and safely "
                "executing SQL against the employee engagement SQLite database. "
                "Best for counts, lookups, filters, and simple group-bys."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The natural-language question to answer with SQL.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analytics",
            "description": (
                "Compute a deterministic metric: percentages, average rating by "
                "group, counts by group, rating distribution, or a monthly trend."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "percentage", "average_by_group", "count_by_group",
                            "rating_distribution", "trend",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "For 'percentage': {column, value, filters?}. "
                            "For 'average_by_group'/'count_by_group': {group_by, filters?}. "
                            "For 'rating_distribution': {filters?}. "
                            "For 'trend': {metric: 'avg_rating'|'count', filters?}. "
                            "'column'/'group_by' must be one of Department, Role, Theme, "
                            "Respondent_Type, Company, Employee_Feedback, Response_Month, "
                            "or Rating (for 'column' only). 'filters' is a dict of "
                            "column: exact_value, e.g. {\"Department\": \"Finance\"}."
                        ),
                    },
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_employee_comments",
            "description": (
                "Semantically search employee free-text comments (e.g. about "
                "management, career growth, compensation). Returns the most "
                "relevant real comments with metadata, not a generated summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "description": "Default 8, max ~20."},
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional exact-match metadata filters, e.g. "
                            "{\"Department\": \"Finance\"} or {\"Theme\": \"Manager Support\"}."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": (
                "Classify a list of employee comments as Positive/Neutral/Negative. "
                "Call this on the comments returned by search_employee_comments "
                "before describing overall sentiment/tone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "response_id": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["response_id", "text"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
]


def _call_query_database(args: dict, user_message: str = "") -> dict:
    result = query_database.query_database(
        question=args["question"], match_question=user_message or None
    ).as_dict()
    result["rating_scale"] = {
        "database": "Rating is INTEGER on a 1-5 scale (1 worst, 5 best)",
        "zero_to_ten_equivalent": "multiply a verified average by 2 only when explicitly requested",
    }
    return result


def _call_run_analytics(args: dict, user_message: str = "") -> dict:
    return analytics.run_analytics(
        operation=args["operation"], params=args.get("params") or {}
    ).as_dict()


def _call_search_employee_comments(args: dict, user_message: str = "") -> dict:
    return retrieval.search_employee_comments(
        query=args["query"],
        top_k=int(args.get("top_k") or 8),
        filters=args.get("filters"),
    ).as_dict()


def _call_analyze_sentiment(args: dict, user_message: str = "") -> dict:
    return sentiment.analyze_sentiment(items=args.get("items") or []).as_dict()


_TOOL_IMPLS: dict[str, Callable[[dict, str], dict]] = {
    "query_database": _call_query_database,
    "run_analytics": _call_run_analytics,
    "search_employee_comments": _call_search_employee_comments,
    "analyze_sentiment": _call_analyze_sentiment,
}


@dataclass
class ToolTraceEntry:
    tool: str
    arguments: dict
    result: dict
    verification: dict

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class ChatResult:
    answer: str
    tool_trace: list[dict] = field(default_factory=list)
    iterations_used: int = 0
    gave_up: bool = False


def _parse_tool_arguments(raw_arguments: Any) -> dict:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            return json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _assistant_message_to_dict(message: Any) -> dict:
    """Convert the SDK's AssistantMessage object into a plain dict we can
    safely re-send as-is in the next request (avoids depending on the
    exact pydantic serialization the SDK expects)."""
    out: dict = {"role": "assistant", "content": message.content or ""}
    if getattr(message, "tool_calls", None):
        out["tool_calls"] = []
        for tc in message.tool_calls:
            args = tc.function.arguments
            if not isinstance(args, str):
                args = json.dumps(args)
            out["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": args},
            })
    return out


def _run_tool(name: str, args: dict, user_message: str = "") -> tuple[dict, dict]:
    """Execute one tool call and its mandatory verification step.
    Returns (result_dict, verification_dict). Never raises - any exception
    from the underlying tool is turned into a failed result so the agent
    loop can react to it instead of crashing the whole request.

    `user_message` is the turn's original, unmodified user message (as
    opposed to `args`, which for query_database may be an LLM-rephrased
    sub-question) - passed through so query_database's closest-real-Question
    fallback matches against what the user actually asked. See
    query_database.query_database's docstring for why that distinction
    matters.

    Some tools (query_database's SQL generation, analyze_sentiment) make
    their own Groq call, separate from the routing call in `route()`. A
    Groq outage there is marked with `service_unavailable: True` instead of
    a generic error, so `aggregate_results` can fail the whole turn fast
    instead of re-routing into another doomed attempt at the same call."""
    impl = _TOOL_IMPLS.get(name)
    if impl is None:
        result = {"ok": False, "error": f"Unknown tool '{name}'."}
        return result, {"valid": False, "issues": [result["error"]]}

    try:
        result = impl(args, user_message)
    except groq.APIError as exc:
        result = {"ok": False, "error": str(exc), "service_unavailable": True}
        return result, {"valid": False, "issues": [_SERVICE_UNAVAILABLE_MESSAGE]}
    except Exception as exc:  # noqa: BLE001 - a broken tool must not crash the agent
        result = {"ok": False, "error": f"Tool '{name}' raised an exception: {exc}"}

    extra = {}
    if name == "search_employee_comments":
        extra["query"] = args.get("query", "")
    verification_outcome = verification.verify_tool_result(name, result, **extra)
    return result, verification_outcome


# ---------------------------------------------------------------------------
# LlamaIndex Event classes
#
# Each Event is a typed, pydantic-validated message passed between @step
# methods below. This is what makes the routing "real" LlamaIndex Workflow
# orchestration rather than a description of one: the workflow engine
# dispatches every one of these to whichever @step declares it as a
# parameter type, and `Context.send_event` / `Context.collect_events` do the
# actual fan-out (one event per requested tool call) and fan-in (wait for
# every tool in a multi-tool question before continuing) shown in the
# module docstring's diagram.
# ---------------------------------------------------------------------------


class RouteEvent(Event):
    """StartEvent/aggregate_results -> RouteEvent: ask the LLM what to do
    next - answer directly, or call one or more of query_database /
    run_analytics / search_employee_comments / analyze_sentiment."""

    messages: list[dict]
    trace: list[dict]
    iteration: int
    user_message: str


class ToolCallEvent(Event):
    """RouteEvent -> ToolCallEvent: one tool the LLM asked to run. A single
    LLM turn can emit several of these (multi-step questions), each becoming
    its own event dispatched via `Context.send_event`."""

    tool_call_id: str
    tool_name: str
    arguments: dict
    messages: list[dict]
    trace: list[dict]
    iteration: int
    batch_size: int
    user_message: str


class ToolResultEvent(Event):
    """ToolCallEvent -> ToolResultEvent: the tool has run AND been checked by
    the existing `verification.verify_*` function - this is the
    "VerificationEvent" from the required flow. `valid=False` here is what
    drives the retry/re-route decision in `aggregate_results`."""

    tool_call_id: str
    tool_name: str
    arguments: dict
    result: dict
    verification: dict
    messages: list[dict]
    trace: list[dict]
    iteration: int
    batch_size: int
    user_message: str


class FinalAnswerEvent(Event):
    """RouteEvent -> FinalAnswerEvent: the LLM produced an answer without
    requesting another tool call."""

    answer: str
    trace: list[dict]
    iteration: int


class GiveUpEvent(Event):
    """RouteEvent -> GiveUpEvent: AGENT_MAX_TOOL_ITERATIONS reached without a
    verified answer - same honest-failure behaviour as before, just reached
    via an event instead of a `for` loop running out.

    `message`, when set, overrides the default "couldn't verify" text - used
    for a distinct failure class (the Groq API itself being unavailable/
    rate-limited) where "try rephrasing your question" would be actively
    misleading, since the question was never the problem."""

    trace: list[dict]
    iteration: int
    message: str | None = None


class AgentWorkflow(Workflow):
    """LlamaIndex event-driven orchestrator for one chat turn.

    This is the chatbot's brain: it owns *sequencing* (route -> tool(s) ->
    verify -> re-route or answer). All actual decisions (which tool, what
    arguments, how to phrase the answer) still come from the LLM via
    `chat_complete`/TOOL_SCHEMAS, and all correctness checks still come from
    the deterministic `verification` module - neither of those changed.
    """

    @step
    async def route(
        self, ctx: Context, ev: StartEvent | RouteEvent
    ) -> ToolCallEvent | FinalAnswerEvent | GiveUpEvent:
        # StartEvent ---> RouteEvent: intent routing. Specific-question
        # averages take the deterministic query_database fast path; all other
        # questions use the same LLM tool menu as before.
        if isinstance(ev, StartEvent):
            user_message = ev.get("user_message")
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(ev.get("history") or [])
            messages.append({"role": "user", "content": user_message})
            trace: list[dict] = []
            iteration = 1
        else:
            messages = ev.messages
            trace = ev.trace
            iteration = ev.iteration
            user_message = ev.user_message

        if iteration > config.AGENT_MAX_TOOL_ITERATIONS:
            return GiveUpEvent(trace=trace, iteration=config.AGENT_MAX_TOOL_ITERATIONS)

        if isinstance(ev, StartEvent) and _is_specific_question_average(user_message):
            tool_call_id = "direct-question-average"
            arguments = {"question": user_message}
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "query_database",
                        "arguments": json.dumps(arguments),
                    },
                }],
            })
            return ToolCallEvent(
                tool_call_id=tool_call_id,
                tool_name="query_database",
                arguments=arguments,
                messages=messages,
                trace=trace,
                iteration=iteration,
                batch_size=1,
                user_message=user_message,
            )

        try:
            response = await asyncio.to_thread(
                chat_complete,
                model=config.GROQ_CHAT_MODEL,
                temperature=0.2,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except groq.APIError:
            # groq_client.chat_complete already retried this internally
            # (see retry_utils.rate_limit_retry - up to 8 attempts with
            # growing backoff) before giving up, so a further per-iteration
            # retry here would just repeat that same multi-minute wait for
            # an outage that isn't going to clear in the next few seconds.
            # Fail fast with an honest, distinct message instead of letting
            # this propagate as an unhandled exception (-> a raw 500) or
            # silently eating the rest of the iteration budget.
            return GiveUpEvent(
                trace=trace, iteration=iteration, message=_SERVICE_UNAVAILABLE_MESSAGE
            )
        message = response.choices[0].message

        if not getattr(message, "tool_calls", None):
            # RouteEvent -> FinalAnswerEvent: no tool needed, done.
            return FinalAnswerEvent(answer=message.content or "", trace=trace, iteration=iteration)

        messages = messages + [_assistant_message_to_dict(message)]
        tool_calls = message.tool_calls

        # RouteEvent -> ToolCallEvent (fan-out). Multi-step questions (e.g.
        # "sentiment on compensation AND percentage rated below 3") arrive
        # here as multiple tool_calls in one LLM turn; each becomes its own
        # ToolCallEvent so `execute_tool` runs once per tool and
        # `aggregate_results` combines all of their verified results.
        for tool_call in tool_calls:
            ctx.send_event(ToolCallEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                arguments=_parse_tool_arguments(tool_call.function.arguments),
                messages=messages,
                trace=trace,
                iteration=iteration,
                batch_size=len(tool_calls),
                user_message=user_message,
            ))
        return None

    @step
    async def execute_tool(self, ctx: Context, ev: ToolCallEvent) -> ToolResultEvent:
        # ToolCallEvent -> tool execution -> ToolResultEvent. `_run_tool` is
        # exactly the function used by the old manual loop: it calls the
        # real tool implementation (query_database / run_analytics /
        # search_employee_comments / analyze_sentiment) and then the
        # matching `verification.verify_*` check. Nothing about tool
        # behaviour or verification rules changed - only that it now runs as
        # a workflow step, off the event loop via `asyncio.to_thread`.
        result, verification_outcome = await asyncio.to_thread(
            _run_tool, ev.tool_name, ev.arguments, ev.user_message
        )
        return ToolResultEvent(
            tool_call_id=ev.tool_call_id,
            tool_name=ev.tool_name,
            arguments=ev.arguments,
            result=result,
            verification=verification_outcome,
            messages=ev.messages,
            trace=ev.trace,
            iteration=ev.iteration,
            batch_size=ev.batch_size,
            user_message=ev.user_message,
        )

    @step
    async def aggregate_results(
        self, ctx: Context, ev: ToolResultEvent
    ) -> RouteEvent | GiveUpEvent | None:
        # ToolResultEvent -> VerificationEvent (already applied in
        # execute_tool) -> fan-in: wait for every tool call from this round,
        # fold their (verified-or-not) results back into the conversation,
        # and re-route. This is the retry mechanism: a failed verification
        # is visible to the LLM on the next `route` turn, and SYSTEM_PROMPT
        # instructs it to adjust its tool call, try again, or admit it
        # cannot answer - exactly the same retry contract as the old loop,
        # just re-entering `route` as an event instead of looping in Python.
        batch = ctx.collect_events(ev, [ToolResultEvent] * ev.batch_size)
        if batch is None:
            return None

        messages = list(ev.messages)
        trace = list(ev.trace)
        for r in batch:
            trace.append(ToolTraceEntry(
                tool=r.tool_name, arguments=r.arguments,
                result=r.result, verification=r.verification,
            ).as_dict())
            messages.append({
                "role": "tool",
                "tool_call_id": r.tool_call_id,
                "name": r.tool_name,
                "content": json.dumps(
                    {"result": r.result, "verification": r.verification}, default=str
                ),
            })

        # If any tool in this round hit a Groq outage (SQL generation,
        # sentiment classification - see _run_tool), re-routing would just
        # ask the LLM to try again and very likely hit the same wall, one
        # iteration at a time until AGENT_MAX_TOOL_ITERATIONS. Fail fast
        # with the same honest message `route()` uses for a routing-call
        # outage, instead of burning the rest of the budget on retries that
        # can't succeed.
        if any(r.result.get("service_unavailable") for r in batch):
            return GiveUpEvent(trace=trace, iteration=ev.iteration, message=_SERVICE_UNAVAILABLE_MESSAGE)

        return RouteEvent(
            messages=messages, trace=trace, iteration=ev.iteration + 1, user_message=ev.user_message
        )

    @step
    async def finalize(
        self, ctx: Context, ev: FinalAnswerEvent | GiveUpEvent
    ) -> StopEvent:
        # FinalAnswerEvent / GiveUpEvent -> StopEvent: build the same
        # ChatResult shape the FastAPI /chat endpoint has always expected.
        if isinstance(ev, GiveUpEvent):
            result = ChatResult(
                answer=ev.message or _default_give_up_message(ev.trace),
                tool_trace=ev.trace,
                iterations_used=ev.iteration,
                gave_up=True,
            )
        else:
            result = ChatResult(
                answer=ev.answer,
                tool_trace=ev.trace,
                iterations_used=ev.iteration,
                gave_up=False,
            )
        return StopEvent(result=result)


# Workflow config (steps/events) is static, so one instance is reused across
# requests - each `.run(...)` call gets its own fresh Context, so concurrent
# /chat requests do not share routing/trace state. `timeout=None` matches
# the old loop's behaviour of having no wall-clock cutoff of its own.
_workflow = AgentWorkflow(timeout=None)


async def _run_workflow(user_message: str, history: list[dict]) -> ChatResult:
    handler = _workflow.run(user_message=user_message, history=history)
    return await handler


def chat(user_message: str, history: list[dict] | None = None) -> ChatResult:
    """Run one user turn through the LlamaIndex Workflow (StartEvent ->
    RouteEvent -> tool event(s) -> verification -> retry/re-route ->
    FinalAnswerEvent/GiveUpEvent -> StopEvent). Stays a plain synchronous
    function so existing callers (the FastAPI /chat endpoint, scripts/ask.py)
    need no changes."""
    return asyncio.run(_run_workflow(user_message, history or []))
