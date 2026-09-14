"""Cross-text pattern-finding agent (design-spec.md §5.3).

Batch, not per-text: runs once over the whole validated `split` and returns
a handful of PatternFinding results — not one call per input text, unlike
extraction_agent.py/validation_agent.py. The LLM's only job is choosing
which asset types are worth investigating and narrating a human-readable
description for each pattern it finds compelling; it never computes a count
or invents a source text itself — src/tools.py's get_failure_history() does
that deterministically, and every finding is cross-checked in
`_verify_against_observed()` below against what a tool call actually
returned before being accepted, not just trusted because the model said so.
Same evidence-trail principle as extraction_agent.py's span resolution and
validation_agent.py's structural checks: verify what the model claims
against what's actually known, don't just take its word for it.

Implemented as a real multi-turn tool-calling loop against each provider's
raw API — not LangGraph/CrewAI (CLAUDE.md's "plain sequential Python" for
orchestration is about not needing a framework for the four *stages*; an
agentic loop with one tool inside a single stage isn't the kind of
branching/looping complexity that non-goal was written about). DeepSeek's
tool-calling STRICT mode is deliberately avoided (CLAUDE.md flags it as
beta with its own base URL); plain function-calling is used instead, with
every tool call's arguments validated in `_execute_tool_call()` before
get_failure_history() is ever invoked with them.

Provider isolated behind `_call_model()`. Default DeepSeek (`deepseek-flash`);
set PATTERN_PROVIDER=claude for the comparison pass — Sonnet 5, per
CLAUDE.md's model choices (the only agent in this pipeline NOT using Haiku
4.5 on the Claude side, since cross-text pattern investigation is judged to
need more capability than per-text extraction/validation/drafting).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv

from src import store
from src.tools import get_failure_history, list_common_asset_types

load_dotenv()

PROVIDER = os.environ.get("PATTERN_PROVIDER", "deepseek")  # "deepseek" | "claude"
MAX_TOOL_TURNS = 8  # safety bound on the agentic loop, not a production retry system


@dataclass
class PatternFinding:
    asset_type: str
    pattern: str
    occurrence_count: int
    example_source_texts: list[str]
    supporting_entity_types: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TOOL_SCHEMA = {
    "name": "get_failure_history",
    "description": (
        "Deterministically look up how often a given asset type co-occurs with a "
        "State/Process entity (a failure or maintenance-process signature) across the "
        "validated corpus, via hasParticipant/hasProperty relations. Returns up to top_n "
        "records sorted by occurrence_count, most frequent first, each with example source "
        "texts. This is the ONLY source of truth for counts and example texts in this task "
        "— never state a count or cite a text you did not get from a call to this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "asset_type": {
                "type": "string",
                "description": (
                    "A PhysicalObject taxonomy type, e.g. 'PhysicalObject/DrivingObject/"
                    "CombustionEngine'. Also matches taxonomy descendants of this type."
                ),
            },
            "top_n": {"type": "integer", "description": "Max records to return, default 20."},
        },
        "required": ["asset_type"],
    },
}

SYSTEM_PROMPT = """\
You are a maintenance-pattern investigator. You look for recurring failure/maintenance \
signatures across a large corpus of validated maintenance work order (MWO) extractions, \
using a single tool, get_failure_history, which is the ONLY source of truth for counts \
and example texts.

You will be given a shortlist of the most common asset (PhysicalObject) types in the \
corpus. Call get_failure_history for the ones you think are worth investigating — you \
don't have to call it for every one, and you may call it more than once for different \
asset types. Stop calling it once you have enough to report a handful of genuinely \
recurring patterns (occurrence_count noticeably above 1 — a single occurrence is not a \
pattern).

When you're done investigating, respond with a final JSON object only (no more tool \
calls, no prose, no markdown fences) with this exact shape:
{"findings": [{"asset_type": "...", "pattern": "<a short, human-readable description of the recurring issue>", "occurrence_count": <int, copied exactly from a get_failure_history result>, "example_source_texts": [<copied exactly from that result>], "supporting_entity_types": ["<the asset_type>", "<the failure_type from that result>"]}]}

Only report findings backed by an actual get_failure_history call you made — every \
occurrence_count and example_source_texts value must be copied exactly from a tool \
result, never invented, estimated, or rounded.
"""


def _execute_tool_call(name: str, arguments_json: str, split: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse and validate a tool call's arguments before ever touching
    get_failure_history() — per CLAUDE.md's explicit warning not to trust a
    provider's tool-calling schema enforcement blindly.

    `split` is threaded through from find_patterns() explicitly, not left
    to get_failure_history()'s own default — otherwise every tool call would
    silently query "silver" regardless of what split find_patterns() was
    actually asked to investigate (caught by testing this against a fixture
    split before this fix).
    """
    if name != "get_failure_history":
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError:
        return {"error": "arguments were not valid JSON"}
    asset_type = args.get("asset_type")
    if not isinstance(asset_type, str) or not asset_type:
        return {"error": "asset_type must be a non-empty string"}
    top_n = args.get("top_n", 20)
    if not isinstance(top_n, int) or top_n <= 0:
        top_n = 20
    return get_failure_history(asset_type=asset_type, top_n=top_n, split=split)


# ---------------------------------------------------------------------------
# Provider-specific tool-calling loops
# ---------------------------------------------------------------------------

def _run_deepseek(system_prompt: str, user_prompt: str, split: str) -> tuple[str, list[dict[str, Any]]]:
    from openai import OpenAI  # imported lazily: don't require this SDK when PROVIDER == "claude"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    tools = [{"type": "function", "function": TOOL_SCHEMA}]
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
            # see extraction_agent.py's note: deepseek-flash's default reasoning pass
            # is unneeded overhead here (this loop already worked without it going
            # wrong, but was paying the reasoning-token cost on every turn).
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or "", observed
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

    # Ran out of turns without a final answer -- ask directly, no further tool calls implied.
    messages.append({"role": "user", "content": "Please respond now with your final JSON findings, no more tool calls."})
    response = client.chat.completions.create(
        model="deepseek-flash", messages=messages, temperature=0, extra_body={"thinking": {"type": "disabled"}}
    )
    return response.choices[0].message.content or "", observed


def _run_claude(system_prompt: str, user_prompt: str, split: str) -> tuple[str, list[dict[str, Any]]]:
    from anthropic import Anthropic  # imported lazily: don't require this SDK when PROVIDER == "deepseek"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = Anthropic(api_key=api_key)

    tools = [{"name": TOOL_SCHEMA["name"], "description": TOOL_SCHEMA["description"], "input_schema": TOOL_SCHEMA["parameters"]}]
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
            return "".join(b.text for b in response.content if b.type == "text"), observed
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tu in tool_uses:
            result = _execute_tool_call(tu.name, json.dumps(tu.input), split)
            if isinstance(result, list):
                observed.extend(result)
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "user", "content": tool_results})

    messages.append({"role": "user", "content": "Please respond now with your final JSON findings, no more tool calls."})
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


def _verify_against_observed(finding: PatternFinding, observed_records: list[dict[str, Any]]) -> bool:
    """Does some record actually returned by a get_failure_history call
    during this run support this finding's count and example texts? A
    finding that doesn't match any observed record is fabricated (a wrong
    count, an invented example) and is dropped, not trusted on the model's
    say-so — same principle as the structural checks elsewhere in this
    codebase."""
    return any(
        record["occurrence_count"] == finding.occurrence_count
        and set(finding.example_source_texts) <= set(record["example_source_texts"])
        for record in observed_records
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def find_patterns(split: str = "silver", num_candidates: int = 20) -> list[PatternFinding]:
    """Run the Pattern Agent once over `split`'s validated data (design-spec.md §5.3).

    Persists to store/patterns/findings.json via src/store.py, matching
    scoring.py's evaluate_gold_split() precedent for this codebase's
    batch-level modules (unlike extraction_agent.py/validation_agent.py,
    which are per-text and leave persistence to their batch scripts).
    """
    candidates = list_common_asset_types(split=split, top_n=num_candidates)
    user_prompt = (
        f"Most common asset types in the validated {split} corpus (asset_type, count):\n"
        + json.dumps(candidates, ensure_ascii=False)
    )

    raw, observed = _call_model(SYSTEM_PROMPT, user_prompt, split)
    try:
        parsed = _parse_model_response(raw)
    except json.JSONDecodeError:
        # One retry, same single-pass policy as extraction_agent.py/validation_agent.py.
        # Note this re-runs the WHOLE tool-calling loop from scratch (a fresh call to
        # _call_model), not just a reformatting nudge on the same conversation — an
        # accepted inefficiency for v1 rather than building conversation-resuming retry.
        raw, observed_retry = _call_model(
            SYSTEM_PROMPT,
            "Your previous response was not valid JSON. Respond with ONLY the final JSON "
            "object, no more tool calls, nothing else.\n\n" + user_prompt,
            split,
        )
        observed = observed + observed_retry
        parsed = _parse_model_response(raw)  # let this raise if it fails a second time

    findings: list[PatternFinding] = []
    for f in parsed.get("findings", []):
        try:
            finding = PatternFinding(
                asset_type=str(f["asset_type"]),
                pattern=str(f["pattern"]),
                occurrence_count=int(f["occurrence_count"]),
                example_source_texts=[str(t) for t in f.get("example_source_texts", [])],
                supporting_entity_types=[str(t) for t in f.get("supporting_entity_types", [])],
            )
        except (KeyError, TypeError, ValueError):
            continue  # malformed finding — structural sanity only, same principle as extraction_agent.py
        if _verify_against_observed(finding, observed):
            findings.append(finding)

    result = [f.to_dict() for f in findings]
    store.write_doc("patterns/findings", result)
    return findings
