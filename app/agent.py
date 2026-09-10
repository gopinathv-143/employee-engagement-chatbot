"""
Step 4: The Mistral agent - orchestrates tool selection, execution,
verification, and retry, then writes the final answer.

Design:
  - Mistral does the reasoning (which tool(s) to call, with what arguments,
    and how to phrase the final answer). It NEVER computes a number or
    quotes a comment itself - it is only allowed to report what a tool
    returned.
  - Every tool call is followed, in Python (not by asking the LLM), by the
    matching verification function from app.tools.verification. The
    verification outcome is appended alongside the raw tool result before
    being sent back to Mistral, so the model always sees "here is the data,
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
from dataclasses import dataclass, field
from typing import Any, Callable

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from app import config
from app.groq_client import chat_complete
from app.tools import analytics, query_database, retrieval, sentiment, verification

SYSTEM_PROMPT = """You are an HR analytics assistant for an employee engagement \
survey dataset. You answer questions ONLY using the tools available to you - you \
must never invent numbers, percentages, counts, trends, or employee quotes.

Tool selection guide:
- Counts, totals, simple lookups, group-bys, filters -> query_database
- Percentages, averages, rating distributions, trends over time -> run_analytics \
  (prefer this over query_database for "percentage X" and "average rating by Y" \
  questions - it is computed deterministically, not written as freeform SQL)
- "What do employees say about X", concerns, opinions, complaints, themes -> \
  search_employee_comments, and then call analyze_sentiment on the retrieved \
  comments' texts before describing the overall sentiment/tone
- If a question needs more than one of these (e.g. "what's the sentiment on \
  compensation, and what percentage rated it below 3"), call multiple tools \
  and combine their verified results.

After every tool call you will receive the raw result AND a "verification" \
block computed independently of you. If verification says a result is not \
valid, do NOT use it to answer - either adjust your tool call (different \
operation/filters/wording) and try again, or, if you are out of reasonable \
options, tell the user plainly that you could not find a reliable answer \
instead of guessing.

When you do have verified results, write a clear, concise natural-language \
answer that cites the actual numbers/quotes returned by the tools. Do not \
show raw JSON to the user. Keep answers focused and readable.
"""

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


def _call_query_database(args: dict) -> dict:
    return query_database.query_database(question=args["question"]).as_dict()


def _call_run_analytics(args: dict) -> dict:
    return analytics.run_analytics(
        operation=args["operation"], params=args.get("params") or {}
    ).as_dict()


def _call_search_employee_comments(args: dict) -> dict:
    return retrieval.search_employee_comments(
        query=args["query"],
        top_k=int(args.get("top_k") or 8),
        filters=args.get("filters"),
    ).as_dict()


def _call_analyze_sentiment(args: dict) -> dict:
    return sentiment.analyze_sentiment(items=args.get("items") or []).as_dict()


_TOOL_IMPLS: dict[str, Callable[[dict], dict]] = {
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


def _run_tool(name: str, args: dict) -> tuple[dict, dict]:
    """Execute one tool call and its mandatory verification step.
    Returns (result_dict, verification_dict). Never raises - any exception
    from the underlying tool is turned into a failed result so the agent
    loop can react to it instead of crashing the whole request."""
    impl = _TOOL_IMPLS.get(name)
    if impl is None:
        result = {"ok": False, "error": f"Unknown tool '{name}'."}
        return result, {"valid": False, "issues": [result["error"]]}

    try:
        result = impl(args)
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


class FinalAnswerEvent(Event):
    """RouteEvent -> FinalAnswerEvent: the LLM produced an answer without
    requesting another tool call."""

    answer: str
    trace: list[dict]
    iteration: int


class GiveUpEvent(Event):
    """RouteEvent -> GiveUpEvent: AGENT_MAX_TOOL_ITERATIONS reached without a
    verified answer - same honest-failure behaviour as before, just reached
    via an event instead of a `for` loop running out."""

    trace: list[dict]
    iteration: int


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
        # StartEvent ---> RouteEvent: intent routing. TOOL_SCHEMAS is the
        # same tool menu as before (query_database / run_analytics /
        # search_employee_comments / analyze_sentiment); the LLM decides
        # which one(s) apply to the question, unchanged from the old loop.
        if isinstance(ev, StartEvent):
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(ev.get("history") or [])
            messages.append({"role": "user", "content": ev.get("user_message")})
            trace: list[dict] = []
            iteration = 1
        else:
            messages = ev.messages
            trace = ev.trace
            iteration = ev.iteration

        if iteration > config.AGENT_MAX_TOOL_ITERATIONS:
            return GiveUpEvent(trace=trace, iteration=config.AGENT_MAX_TOOL_ITERATIONS)

        response = await asyncio.to_thread(
            chat_complete,
            model=config.GROQ_CHAT_MODEL,
            temperature=0.2,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
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
            _run_tool, ev.tool_name, ev.arguments
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
        )

    @step
    async def aggregate_results(
        self, ctx: Context, ev: ToolResultEvent
    ) -> RouteEvent | None:
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

        return RouteEvent(messages=messages, trace=trace, iteration=ev.iteration + 1)

    @step
    async def finalize(
        self, ctx: Context, ev: FinalAnswerEvent | GiveUpEvent
    ) -> StopEvent:
        # FinalAnswerEvent / GiveUpEvent -> StopEvent: build the same
        # ChatResult shape the FastAPI /chat endpoint has always expected.
        if isinstance(ev, GiveUpEvent):
            result = ChatResult(
                answer=(
                    "I wasn't able to produce a fully verified answer to that question "
                    "after several attempts. Could you rephrase it, or narrow it down "
                    "(e.g. a specific department, theme, or time period)?"
                ),
                tool_trace=ev.trace,
                iterations_used=config.AGENT_MAX_TOOL_ITERATIONS,
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
