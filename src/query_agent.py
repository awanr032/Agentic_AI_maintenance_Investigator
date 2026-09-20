"""Free-form question answering over the validated corpus.

Phase 2 in CLAUDE.md's original scope, now built as an extension of the
same grounding discipline used everywhere else in this codebase: the model
never states a count, asset type, or example text it didn't get from an
actual tool call. Architecturally this is pattern_agent.py's multi-turn
tool-calling loop, generalized from "investigate the top asset types and
report findings" to "answer whatever the caller asked, using the same two
tools." Both tools are the exact same deterministic functions in
src/tools.py — no new counting/lookup logic, only a second entry point
onto data that already has a trusted computation path.

Unlike PatternFinding (a fixed structured shape), an answer here is free
text, so there's no PatternFinding-style field-by-field verification
possible. Grounding is checked instead by requiring every number the
answer states to appear in some tool result actually observed during the
run (`_verify_grounded()`) — same "don't trust the model's arithmetic"
principle as drafting_agent.py's hard numeric check, adapted to text
instead of a fixed template.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv

from src import tools
from src.tools import get_failure_history, list_common_asset_types

load_dotenv()

PROVIDER = os.environ.get("QUERY_PROVIDER", "deepseek")  # "deepseek" | "claude"
MAX_TOOL_TURNS = 8


@dataclass
class QueryResult:
    question: str
    answer: str
    supporting_facts: list[dict[str, Any]]
    grounded: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TOOL_SCHEMAS = [
    {
        "name": "list_common_asset_types",
        "description": (
            "List the most frequent PhysicalObject taxonomy types in the validated corpus, "
            "each with its exact dotted-path type string and how many records mention it. "
            "Use this first to find the exact taxonomy string for whatever asset the "
            "question is about (e.g. to find how 'pumps' are actually named in this data) "
            "before calling get_failure_history — never guess a taxonomy path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "Max asset types to return, default 20."},
            },
            "required": [],
        },
    },
    {
        "name": "get_failure_history",
        "description": (
            "Deterministically look up how often a given asset type co-occurs with a "
            "State/Process entity (a failure or maintenance-process signature) across the "
            "validated corpus. Returns up to top_n records sorted by occurrence_count, most "
            "frequent first, each with example source texts. This is the ONLY source of "
            "truth for counts and example texts — never state a count or cite a text you "
            "did not get from a call to this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "description": (
                        "An exact PhysicalObject taxonomy type string as returned by "
                        "list_common_asset_types, e.g. 'PhysicalObject/DrivingObject/"
                        "CombustionEngine'. Also matches taxonomy descendants of this type."
                    ),
                },
                "top_n": {"type": "integer", "description": "Max records to return, default 20."},
            },
            "required": ["asset_type"],
        },
    },
]

SYSTEM_PROMPT = """\
You are a maintenance data query assistant. You answer questions about recurring failure \
and maintenance patterns in a validated corpus of maintenance work order (MWO) records, \
using exactly two tools: list_common_asset_types and get_failure_history. These tools are \
the ONLY source of truth for any count, asset type name, or example text in your answer.

Typical approach: call list_common_asset_types first to see what asset types exist and \
how they're actually named in this taxonomy (a plain-English term like "pump" may map to \
a specific taxonomy path you need to find), then call get_failure_history for the ones \
relevant to the question. Call it more than once if the question spans multiple asset \
types. If nothing in the tool results is relevant to the question, say so plainly instead \
of guessing.

When you have enough information, respond with a final JSON object only (no more tool \
calls, no prose, no markdown fences) with this exact shape:
{"answer": "<a short, plain-English answer, citing specific counts and examples you actually observed from a tool call>"}

Every number and example text in your answer must be copied exactly from a tool result. \
If you cannot answer from the available tools, say that plainly in "answer" rather than \
inventing a number.
"""


def _execute_tool_call(name: str, arguments_json: str, split: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Same validate-before-calling discipline as pattern_agent.py's
    _execute_tool_call — never trust a provider's schema enforcement blindly."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return {"error": "arguments were not valid JSON"}

    if name == "list_common_asset_types":
        top_n = args.get("top_n", 20)
        if not isinstance(top_n, int) or top_n <= 0:
            top_n = 20
        return list_common_asset_types(split=split, top_n=top_n)

    if name == "get_failure_history":
        asset_type = args.get("asset_type")
        if not isinstance(asset_type, str) or not asset_type:
            return {"error": "asset_type must be a non-empty string"}
        top_n = args.get("top_n", 20)
        if not isinstance(top_n, int) or top_n <= 0:
            top_n = 20
        return get_failure_history(asset_type=asset_type, top_n=top_n, split=split)

    return {"error": f"unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Provider-specific tool-calling loops (structurally identical to
# pattern_agent.py's, generalized to a two-tool schema list)
# ---------------------------------------------------------------------------

def _run_deepseek(system_prompt: str, user_prompt: str, split: str) -> tuple[str, list[dict[str, Any]]]:
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    tools = [{"type": "function", "function": schema} for schema in TOOL_SCHEMAS]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    observed: list[dict[str, Any]] = []

    for _ in range(MAX_TOOL_TURNS):
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tools,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        if not message.tool_calls:
            content = message.content or ""
            if _is_valid_final_answer(content):
                return content, observed
            # Malformed JSON on what looked like the final turn: correct it
            # WITHOUT discarding the conversation so far. The old behavior
            # (a top-level retry starting a brand-new messages=[system,
            # user] list) threw away every tool result already gathered —
            # caught via a real case where the model had 54 observed
            # records but the discard-and-restart retry path told the
            # caller "no tool results are available," a confusing, overly
            # conservative non-answer despite having real data on hand.
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That wasn't valid JSON in the required shape. Respond with ONLY "
                        '{"answer": "..."}, using the tool results already gathered above — '
                        "no more tool calls needed."
                    ),
                }
            )
            continue
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            }
        )
        for tc in message.tool_calls:
            result = _execute_tool_call(tc.function.name, tc.function.arguments, split)
            if isinstance(result, list):
                observed.extend(result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

    messages.append({"role": "user", "content": "Please respond now with your final JSON answer, no more tool calls."})
    response = client.chat.completions.create(
        model="deepseek-flash", messages=messages, temperature=0, extra_body={"thinking": {"type": "disabled"}}
    )
    return response.choices[0].message.content or "", observed


def _run_claude(system_prompt: str, user_prompt: str, split: str) -> tuple[str, list[dict[str, Any]]]:
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = Anthropic(api_key=api_key)

    tools = [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in TOOL_SCHEMAS]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    observed: list[dict[str, Any]] = []

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2048,
            temperature=0,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            content = "".join(b.text for b in response.content if b.type == "text")
            if _is_valid_final_answer(content):
                return content, observed
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That wasn't valid JSON in the required shape. Respond with ONLY "
                        '{"answer": "..."}, using the tool results already gathered above — '
                        "no more tool calls needed."
                    ),
                }
            )
            continue
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tu in tool_uses:
            result = _execute_tool_call(tu.name, json.dumps(tu.input), split)
            if isinstance(result, list):
                observed.extend(result)
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "user", "content": tool_results})

    messages.append({"role": "user", "content": "Please respond now with your final JSON answer, no more tool calls."})
    response = client.messages.create(model="claude-sonnet-5", max_tokens=2048, temperature=0, system=system_prompt, messages=messages)
    return "".join(b.text for b in response.content if b.type == "text"), observed


def _call_model(system_prompt: str, user_prompt: str, split: str) -> tuple[str, list[dict[str, Any]]]:
    if PROVIDER == "claude":
        return _run_claude(system_prompt, user_prompt, split)
    return _run_deepseek(system_prompt, user_prompt, split)


def _parse_model_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _is_valid_final_answer(raw: str) -> bool:
    """Used inside the tool-calling loops to decide whether a turn with no
    tool calls is genuinely done, or needs an in-conversation correction —
    see the loops' comments for why this replaced a discard-and-restart retry."""
    try:
        parsed = _parse_model_response(raw)
        return isinstance(parsed.get("answer"), str)
    except (json.JSONDecodeError, AttributeError):
        return False


def _verify_grounded(answer: str, question: str, observed_records: list[dict[str, Any]]) -> bool:
    """Every integer literal in the answer must either come from the
    question itself (e.g. "top 3" echoing the caller's own phrasing — not a
    claim about the data) or appear as some observed record's
    occurrence_count/count — the same "don't trust the model's own
    arithmetic" principle as pattern_agent.py's _verify_against_observed,
    adapted from a fixed structured field to free text. Excluding the
    question's own numbers was added after a real false positive: "What are
    the top 3 ... " got its "3" flagged as an unverified claim even though
    every actual count in the answer was correct — a number the caller
    supplied isn't something the model could have fabricated. This is a
    necessary-but-not-sufficient sanity check, same advisory spirit as
    drafting_agent.py's soft groundedness score, not a proof of correctness.
    """
    numbers_in_question = {int(n) for n in re.findall(r"\b\d+\b", question)}
    numbers_in_answer = {int(n) for n in re.findall(r"\b\d+\b", answer)} - numbers_in_question
    if not numbers_in_answer:
        return True  # nothing numeric claimed beyond the question itself
    observed_numbers = {
        record[key]
        for record in observed_records
        for key in ("occurrence_count", "count")
        if key in record
    }
    return numbers_in_answer <= observed_numbers


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(question: str, split: str = "silver") -> QueryResult:
    """Answer a free-form question about `split`'s validated corpus
    (design-spec.md's originally-deferred Query Agent).

    Not persisted via store.py — unlike find_patterns()'s one saved
    document, a query's answer is only useful to the caller that asked it,
    same non-persistence rationale as run_drafting.py's drafted text.
    """
    tools.clear_cache()  # a warm Lambda container must never reuse a previous invocation's data
    raw, observed = _call_model(SYSTEM_PROMPT, question, split)
    try:
        parsed = _parse_model_response(raw)
        answer = str(parsed["answer"])
    except (json.JSONDecodeError, KeyError, TypeError):
        raw_retry, observed_retry = _call_model(
            SYSTEM_PROMPT,
            "Your previous response was not valid JSON. Respond with ONLY the final JSON "
            "object, no more tool calls, nothing else.\n\n" + question,
            split,
        )
        observed = observed + observed_retry
        parsed = _parse_model_response(raw_retry)  # let this raise if it fails a second time
        answer = str(parsed["answer"])

    return QueryResult(
        question=question,
        answer=answer,
        supporting_facts=observed,
        grounded=_verify_grounded(answer, question, observed),
    )
