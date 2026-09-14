"""Bounded prose drafting agent (design-spec.md §5.4).

`draft(finding) -> str` matches the literal contract from §5.4: a plain
string, not a richer object — honored exactly, even though this module also
builds an advisory faithfulness checker (see below), which is exposed as a
SEPARATE function rather than smuggled into draft()'s return type.

**The hard constraint** (design-spec.md calls this "the most important line
in this spec"): never state a fact, cause, severity judgment, or
recommendation that isn't in the structured input. Enforced two ways here:

1. Prompt-level (the only enforcement §5.4 actually requires for v1): the
   system prompt states the rule up front, plus one paired good/bad example
   per input kind, showing exactly the failure mode to avoid (inventing a
   cause or a recommendation) rather than just asserting the rule abstractly.
2. `check_faithfulness()` — an ADVISORY, NOT REQUIRED-for-v1 automated check
   (design-spec.md §9 open question 5's stretch goal, built now since it's
   the same "verify what the model claims, don't just trust it" principle
   used everywhere else in this codebase — extraction_agent.py's span
   resolution, validation_agent.py's structural checks, pattern_agent.py's
   _verify_against_observed). Two signals, deliberately not one, because
   they have very different reliability:
   - `numeric_issues`: any number in the drafted text that doesn't match a
     real number from the input (occurrence_count, example count). Numbers
     have no paraphrase ambiguity, so this is a genuinely reliable
     fabrication signal.
   - `groundedness_score`: fraction of the draft's content words traceable
     to the input's own vocabulary (span text, type names split on "/" and
     camelCase, confidence levels, source texts). This is inherently fuzzy —
     ordinary faithful paraphrasing uses connector words ("resolved",
     "showing", "on") that legitimately won't appear verbatim in the input —
     so a low score is a prompt to go look, not proof of a problem. This is
     explicitly NOT a substitute for §5.4's actual acceptance criteria
     (manually spot-check drafts against their input); it's a triage aid.

Pattern-finding drafts get a fixed disclaimer sentence appended
deterministically in Python, not left to the model to remember — one less
thing that can silently go missing (same reasoning as computing
start_token/end_token deterministically in extraction_agent.py rather than
trusting the model to count).
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("DRAFTING_PROVIDER", "deepseek")  # "deepseek" | "claude"

PATTERN_DISCLAIMER = "This pattern was not independently verified beyond the source texts shown."

EXTRACTION_SYSTEM_PROMPT = """\
You are a report-writing agent. You turn one structured Extraction Agent result for a \
single maintenance work order into a short, readable paragraph.

THE MOST IMPORTANT RULE: only restate facts explicitly present in the structured finding \
given to you. Never infer a cause, never judge severity, never suggest a next step or \
recommendation, never add any detail not in the given fields — even if it seems obvious \
or highly likely. If something isn't given, leave it out rather than guessing.

Example of a CORRECT, bounded paragraph for a finding with entities "bearing wear" \
(State: DegradationState/Wear) and "pump" (PhysicalObject), connected by a \
hasParticipant/hasPatient relation, plus an Activity "change out" \
(MaintenanceActivity/Replace):
"This work order recorded a bearing wear issue (State: DegradationState/Wear) on a pump, \
resolved by replacement (Activity: MaintenanceActivity/Replace). Confidence: high."

Example of an INCORRECT paragraph for the SAME finding — wrong because it invents a \
cause ("inadequate lubrication") and a judgment ("the correct fix") not present in the \
input, and a future-prevention claim nobody stated:
"This work order recorded a bearing wear issue on a pump, likely caused by inadequate \
lubrication, and replacement was the correct fix to prevent future failures."

Respond with the paragraph text only — no JSON, no markdown, no preamble like "Here is \
the paragraph:".
"""

PATTERN_SYSTEM_PROMPT = """\
You are a report-writing agent. You turn one structured Pattern Agent finding (a \
recurring signature across many work orders) into a short, readable paragraph.

THE MOST IMPORTANT RULE: only restate facts explicitly present in the structured finding \
given to you. Never infer a cause, never judge severity, never suggest a next step or \
recommendation (e.g. "inspection intervals should be shortened"), never add any detail \
not in the given fields — even if it seems obvious or highly likely.

Example of a CORRECT, bounded paragraph for a finding with pattern "recurring bearing \
wear failures" on combustion engines, occurrence_count 12, and two example texts:
"Bearing wear on combustion engines recurred 12 times in the reviewed records, most \
recently in: [the two example texts]."

Example of an INCORRECT paragraph for the SAME finding — wrong because it invents a cause \
("poor maintenance scheduling") and a recommendation ("inspection intervals should be \
shortened") not present in the input:
"Bearing wear on combustion engines recurred 12 times, likely due to poor maintenance \
scheduling, and inspection intervals should be shortened to address this recurring issue."

Write ONLY the sentence(s) restating the pattern, the exact occurrence_count, and the \
example texts — do NOT add any disclaimer/caveat sentence yourself, one is appended \
automatically afterward. Respond with that text only — no JSON, no markdown, no preamble.
"""


def _is_pattern_finding(finding: dict[str, Any]) -> bool:
    return "asset_type" in finding and "pattern" in finding


def _build_extraction_user_prompt(finding: dict[str, Any]) -> str:
    import json

    payload = {
        "text": finding["text"],
        "entities": [
            {"span_text": e["span_text"], "type": e["type"], "confidence": e["confidence"]}
            for e in finding.get("entities", [])
        ],
        "relations": [
            {"head_span": r["head_span"], "tail_span": r["tail_span"], "type": r["type"]}
            for r in finding.get("relations", [])
        ],
    }
    return (
        "Structured finding (an Extraction Agent result for one work order):\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nWrite the bounded paragraph now, restating only what's given above."
    )


def _build_pattern_user_prompt(finding: dict[str, Any]) -> str:
    import json

    payload = {
        "asset_type": finding["asset_type"],
        "pattern": finding["pattern"],
        "occurrence_count": finding["occurrence_count"],
        "example_source_texts": finding["example_source_texts"],
        "supporting_entity_types": finding.get("supporting_entity_types", []),
    }
    return (
        "Structured finding (a Pattern Agent result across many work orders):\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nWrite ONLY the sentence(s) restating the pattern, count, and example texts given above."
    )


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def _call_deepseek(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI  # imported lazily: don't require this SDK when PROVIDER == "claude"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # No response_format="json_object" here -- the output is prose, not JSON,
        # unlike extraction_agent.py/validation_agent.py.
        temperature=0,
        # Real bug found and fixed here: with max_tokens=300 and thinking left enabled
        # (deepseek-flash's default), this call was silently returning an EMPTY string
        # for 8 of 15 real pattern-finding drafts in one run — finish_reason="length"
        # because the model's internal reasoning pass (reasoning_content, billed as
        # output tokens) consumed the entire 300-token budget before writing any actual
        # answer. draft() had no check for this, so an empty response silently became
        # "This pattern was not independently verified..." with the actual pattern
        # content missing entirely — caught only because check_faithfulness() flagged
        # groundedness=0.00 on the result. Fixed two ways: disabling thinking (see
        # extraction_agent.py's note — also a real, separate cost reduction, unrelated
        # to this bug) removes the reasoning-token consumption entirely; max_tokens is
        # also raised as defense-in-depth in case a future model reintroduces reasoning
        # by default.
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=500,
    )
    return response.choices[0].message.content or ""


def _call_claude(system_prompt: str, user_prompt: str) -> str:
    from anthropic import Anthropic  # imported lazily: don't require this SDK when PROVIDER == "deepseek"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_model(system_prompt: str, user_prompt: str) -> str:
    if PROVIDER == "claude":
        return _call_claude(system_prompt, user_prompt)
    return _call_deepseek(system_prompt, user_prompt)


def _clean(raw: str) -> str:
    """Strip whitespace and a stray ```-fence or wrapping quotes, despite
    the prompt asking for neither — same defensive-but-minimal pattern as
    the other agents' _parse_model_response()."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def draft(finding: dict[str, Any]) -> str:
    """Draft bounded prose from one structured finding (design-spec.md §5.4).

    `finding` is either an ExtractionResult-shaped dict (has "text",
    "entities", "relations") or a PatternFinding-shaped dict (has
    "asset_type", "pattern", "occurrence_count", "example_source_texts",
    "supporting_entity_types") — distinguished by which keys are present.
    """
    if _is_pattern_finding(finding):
        raw = _call_model(PATTERN_SYSTEM_PROMPT, _build_pattern_user_prompt(finding))
        text = _clean(raw)
        if text and not text.endswith((".", "!", "?")):
            text += "."
        return (text + " " + PATTERN_DISCLAIMER).strip()
    raw = _call_model(EXTRACTION_SYSTEM_PROMPT, _build_extraction_user_prompt(finding))
    return _clean(raw)


# ---------------------------------------------------------------------------
# Advisory faithfulness check (design-spec.md §9 open question 5) — NOT part
# of draft()'s contract, called separately by whoever wants the signal.
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"[A-Z][a-z0-9]*|[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "was", "were", "be", "been",
    "being", "and", "or", "but", "not", "no", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "it", "its", "work", "order", "recorded", "reviewed", "records",
    "most", "recently", "shown", "given", "above", "has", "have", "had", "issue",
    "resolved", "found", "detected", "showing", "signs", "again", "confidence",
}


def _split_camel_case(s: str) -> list[str]:
    return [w.lower() for w in _CAMEL_RE.findall(s)]


def _words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)}


def _type_words(type_str: str) -> set[str]:
    words: set[str] = set()
    for segment in type_str.split("/"):
        words |= set(_split_camel_case(segment))
    return words


def _build_vocabulary(finding: dict[str, Any], is_pattern: bool) -> set[str]:
    vocab: set[str] = set()
    if is_pattern:
        vocab |= _type_words(finding["asset_type"])
        vocab |= _words(finding["pattern"])
        vocab |= {str(finding["occurrence_count"])}
        for t in finding.get("example_source_texts", []):
            vocab |= _words(t)
        for t in finding.get("supporting_entity_types", []):
            vocab |= _type_words(t)
    else:
        vocab |= _words(finding["text"])
        for e in finding.get("entities", []):
            vocab |= _words(e["span_text"])
            vocab |= _type_words(e["type"])
            vocab.add(str(e["confidence"]).lower())
        for r in finding.get("relations", []):
            vocab |= _words(r["head_span"])
            vocab |= _words(r["tail_span"])
            vocab |= _type_words(r["type"])
    return vocab


def _groundedness_score(text: str, vocabulary: set[str]) -> tuple[float, list[str]]:
    # Match on the ORIGINAL-case tokens too, so a drafted word that cites a
    # type name verbatim in camelCase (e.g. "MaintenanceActivity", written
    # with no internal space) can still match a vocabulary that was built by
    # camelCase-splitting that same type name into "maintenance"+"activity" --
    # without this, faithfully citing a type name looks like a mismatch.
    raw_words = re.findall(r"[A-Za-z0-9']+", text)
    content_pairs = [(w.lower(), w) for w in raw_words if w.lower() not in _STOPWORDS and not w.lower().isdigit()]
    if not content_pairs:
        return 1.0, []
    unmatched = [
        lower for lower, raw in content_pairs
        if lower not in vocabulary and not set(_split_camel_case(raw)) <= vocabulary
    ]
    score = 1.0 - (len(unmatched) / len(content_pairs))
    return score, sorted(set(unmatched))


def _check_numbers(text: str, finding: dict[str, Any], is_pattern: bool) -> list[str]:
    """Numbers have no paraphrase ambiguity the way words do, so this is a
    much more reliable fabrication signal than the word-level score."""
    drafted_numbers = {int(n) for n in re.findall(r"\b\d+\b", text)}
    allowed_numbers: set[int] = set()
    if is_pattern:
        allowed_numbers.add(int(finding["occurrence_count"]))
        allowed_numbers.add(len(finding.get("example_source_texts", [])))
    bad = sorted(drafted_numbers - allowed_numbers)
    return [
        f"drafted text mentions the number {n}, which does not match occurrence_count "
        f"or example count in the input"
        for n in bad
    ]


def check_faithfulness(drafted_text: str, finding: dict[str, Any]) -> dict[str, Any]:
    """Advisory-only check, NOT required for v1 (design-spec.md §9 open
    question 5) and NOT a substitute for §5.4's real acceptance criteria
    (manually spot-check drafts against their source input). See module
    docstring for why `groundedness_score` is inherently fuzzy while
    `numeric_issues` is a reliable signal.
    """
    is_pattern = _is_pattern_finding(finding)
    vocabulary = _build_vocabulary(finding, is_pattern)
    score, unmatched_words = _groundedness_score(drafted_text, vocabulary)
    return {
        "groundedness_score": score,
        "unmatched_content_words": unmatched_words,
        "numeric_issues": _check_numbers(drafted_text, finding, is_pattern),
    }
