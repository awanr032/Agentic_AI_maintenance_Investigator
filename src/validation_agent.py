"""Validation/QA agent — reviews one Extraction Agent result (design-spec.md §5.2).

This agent does NOT re-extract from scratch. It checks a single
ExtractionResult two ways: deterministic schema/structure checks (cheap,
unambiguous, done in plain Python before any LLM call), and the LLM's own
independent read of the source text (catching things no rule table can,
like a technically-valid type that's still wrong for this text, a missed
entity, or a miscalibrated confidence). `status` is fully determined by
whether the union of both kinds of issues is empty — the model doesn't
report its own pass/flagged verdict, only issues; this keeps `status` from
ever disagreeing with its own `issues` list.

Design decision, resolved not guessed at silently: §5.2 says to check
"does every relation connect entity types that relation is defined for in
SCHEME.md" — but data/scheme.json has no domain/range metadata whatsoever
(every relation node's `description` and `example_terms` are empty strings/
lists, confirmed by inspecting the raw file directly). SCHEME.md doesn't
define this. So `_check_unusual_type_pairs()` below substitutes an
empirical proxy built from gold_release.json instead: across all 1,076 gold
texts, relations connect only 18 distinct (head_top_class, relation_type,
tail_top_class) triples, out of 5x5x7=175 theoretically possible — a
combination never observed there is a genuinely useful "worth a second
look" signal, even though it's data-derived rather than schema-declared.
It's deliberately phrased as a soft finding, not an error: a handful of the
18 observed triples occur only once in gold, so "unobserved" in 1,076
examples doesn't mean "impossible."

Kept cheaper than extraction_agent.py by design (§5.2: "smaller prompt,
narrower job") — no full 224-type list injected, no few-shot examples. The
model only needs this one extraction's own types and the source text to
form a judgment; re-deriving taxonomy membership is already handled
deterministically above, not asked of the model again.

Provider isolated behind `_call_model()`, same one-function-to-swap pattern
as extraction_agent.py. Default DeepSeek (`deepseek-flash`); set
VALIDATION_PROVIDER=claude for the Claude comparison pass (Haiku 4.5, per
CLAUDE.md's model choices).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from src.extraction_agent import ExtractionResult

load_dotenv()

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_release.json"

Confidence = Literal["high", "medium", "low"]
Status = Literal["pass", "flagged"]

PROVIDER = os.environ.get("VALIDATION_PROVIDER", "deepseek")  # "deepseek" | "claude"


@dataclass
class RevisedConfidence:
    entity_index: int
    confidence: Confidence


@dataclass
class ValidationResult:
    text: str
    status: Status
    issues: list[str]
    revised_confidence: RevisedConfidence | None = None

    def to_dict(self) -> dict[str, Any]:
        """For persisting via src/store.py's put_record()."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Deterministic checks — no LLM call, unambiguous, run before the model sees anything
# ---------------------------------------------------------------------------

def _check_types_in_taxonomy(
    extraction: ExtractionResult, allowed_entity_types: list[str], allowed_relation_types: list[str]
) -> list[str]:
    """Every entity/relation type must actually exist in data/scheme.json."""
    allowed_e = set(allowed_entity_types)
    allowed_r = set(allowed_relation_types)
    issues: list[str] = []
    for i, e in enumerate(extraction.entities):
        if e.type not in allowed_e:
            issues.append(f'entity {i} ("{e.span_text}") has type "{e.type}", which is not in the allowed entity taxonomy')
    for i, r in enumerate(extraction.relations):
        if r.type not in allowed_r:
            issues.append(f'relation {i} ("{r.head_span}"->"{r.tail_span}") has type "{r.type}", which is not in the allowed relation taxonomy')
    return issues


def _check_duplicate_or_overlapping_spans(extraction: ExtractionResult) -> list[str]:
    """Flag entities whose token ranges overlap — v1 doesn't model nested/
    overlapping entities in general (see design-spec.md §9 open question 2),
    so any overlap here is treated as suspicious rather than intentional,
    with one deliberate exception: extraction_agent.py's prompt now teaches
    a compound-noun decomposition pattern (e.g. "bend pulley" isA "pulley")
    where one span is a token-subsequence of the other BY DESIGN. An overlap
    between exactly the two spans an "isA" relation connects is that pattern
    working as intended, not an error, so it's excluded here rather than
    flagged every single time the Extraction Agent does the right thing."""
    issues: list[str] = []
    seen: list[tuple[int, int, int]] = []
    isa_pairs = {frozenset((r.head_span, r.tail_span)) for r in extraction.relations if r.type == "isA"}
    for i, e in enumerate(extraction.entities):
        for start, end, j in seen:
            if e.start_token < end and start < e.end_token:
                other = extraction.entities[j]
                if frozenset((e.span_text, other.span_text)) in isa_pairs:
                    continue  # the deliberate decomposition pattern, not an error
                issues.append(
                    f'entity {i} ("{e.span_text}", tokens {e.start_token}-{e.end_token}) overlaps '
                    f'entity {j} ("{other.span_text}", tokens {start}-{end})'
                )
        seen.append((e.start_token, e.end_token, i))
    return issues


def _check_dangling_relations(extraction: ExtractionResult) -> list[str]:
    """Every relation's head_span/tail_span must match an entity actually
    present in this same extraction."""
    known_spans = {e.span_text for e in extraction.entities}
    issues: list[str] = []
    for i, r in enumerate(extraction.relations):
        if r.head_span not in known_spans:
            issues.append(f'relation {i} references head_span "{r.head_span}", which is not among this extraction\'s entities')
        if r.tail_span not in known_spans:
            issues.append(f'relation {i} references tail_span "{r.tail_span}", which is not among this extraction\'s entities')
    return issues


@lru_cache(maxsize=1)
def _observed_type_pair_triples() -> frozenset[tuple[str, str, str]]:
    """(head_top_class, relation_type, tail_top_class) triples actually
    observed anywhere in gold_release.json — the empirical proxy described
    in the module docstring, computed once and cached."""
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        gold = json.load(f)
    triples: set[tuple[str, str, str]] = set()
    for item in gold:
        entities = item["entities"]
        for r in item.get("relations", []):
            head_top = entities[r["head"]]["type"].split("/")[0]
            tail_top = entities[r["tail"]]["type"].split("/")[0]
            triples.add((head_top, r["type"], tail_top))
    return frozenset(triples)


def _check_unusual_type_pairs(extraction: ExtractionResult) -> list[str]:
    """Flag relations whose (head top-class, relation type, tail top-class)
    was never observed in the 1,076-text gold corpus. Soft signal, not a
    hard rule — see module docstring for why scheme.json can't give us a
    hard one."""
    observed = _observed_type_pair_triples()
    span_to_type = {e.span_text: e.type for e in extraction.entities}
    issues: list[str] = []
    for i, r in enumerate(extraction.relations):
        head_type = span_to_type.get(r.head_span)
        tail_type = span_to_type.get(r.tail_span)
        if head_type is None or tail_type is None:
            continue  # already reported by _check_dangling_relations
        triple = (head_type.split("/")[0], r.type, tail_type.split("/")[0])
        if triple not in observed:
            issues.append(
                f'relation {i} ("{r.head_span}" --{r.type}--> "{r.tail_span}") connects top-level classes '
                f'{triple[0]} and {triple[2]}, a combination never observed in the gold corpus — worth a '
                f"second look, not necessarily wrong"
            )
    return issues


def _run_deterministic_checks(
    extraction: ExtractionResult, allowed_entity_types: list[str], allowed_relation_types: list[str]
) -> list[str]:
    return (
        _check_types_in_taxonomy(extraction, allowed_entity_types, allowed_relation_types)
        + _check_duplicate_or_overlapping_spans(extraction)
        + _check_dangling_relations(extraction)
        + _check_unusual_type_pairs(extraction)
    )


# ---------------------------------------------------------------------------
# LLM judgment — only for what a rule table can't catch
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a QA reviewer for an entity/relation extraction pipeline over short maintenance \
work order (MWO) texts, following the MaintIE annotation scheme. You are reviewing ONE \
extraction result someone else already produced — you are NOT re-extracting from scratch.

One MaintIE convention you should NOT flag as an error: when a multi-word noun phrase \
names one object at increasing specificity (a modifier plus a head noun, e.g. "bend \
pulley", where "pulley" is the general head noun and "bend" narrows it), it's correct — \
not a fabricated entity — for BOTH the full phrase AND the bare head noun to appear as \
separate entities, connected by an "isA" relation (full phrase isA head noun). The head \
noun being a token-subsequence of the full phrase is expected there, not a sign the \
shorter span is fake or "not a separate mention". The same applies to "hasPart" between \
a named larger unit and a named sub-part in the same phrase (e.g. "PTO" hasPart "shaft").

You will be given the source text, the entities/relations already extracted (with their \
types and confidence levels), and a list of issues an automated check already found.

Your job:
- Read the source text and judge whether the extraction looks right: does any assigned \
type look wrong given what the text actually says, is there an obvious entity the \
extraction missed, does any confidence level look miscalibrated (too high for something \
ambiguous, too low for something obvious)?
- Only report NEW issues beyond the automated ones already listed — do not restate them.
- If you believe exactly one entity's confidence should be revised, say which one (its \
0-indexed position in the entities list) and to what level. Suggest at most one revision, \
or none if the existing levels look right.
- If you find nothing new, return an empty issues list — do not invent a problem just to \
have something to say.

Respond with a single JSON object only — no prose, no markdown code fences — exactly this shape:
{"additional_issues": ["..."], "revised_confidence": {"entity_index": 0, "confidence": "high|medium|low"} or null}
"""


def _build_user_prompt(extraction: ExtractionResult, deterministic_issues: list[str]) -> str:
    payload = {
        "text": extraction.text,
        "entities": [
            {"index": i, "span_text": e.span_text, "type": e.type, "confidence": e.confidence}
            for i, e in enumerate(extraction.entities)
        ],
        "relations": [
            {"head_span": r.head_span, "tail_span": r.tail_span, "type": r.type} for r in extraction.relations
        ],
        "issues_already_found": deterministic_issues,
    }
    return json.dumps(payload, ensure_ascii=False)


def _call_deepseek(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI  # imported lazily: don't require this SDK when PROVIDER == "claude"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-flash",  # see extraction_agent.py's note: deepseek-chat was retired 2026-07-24
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},  # requires the word "json" in the prompt — see SYSTEM_PROMPT
        temperature=0,
        # see extraction_agent.py's note: deepseek-flash's default reasoning pass is
        # unneeded overhead here and disabling it cuts real cost substantially.
        extra_body={"thinking": {"type": "disabled"}},
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
        max_tokens=512,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_model(system_prompt: str, user_prompt: str) -> str:
    if PROVIDER == "claude":
        return _call_claude(system_prompt, user_prompt)
    return _call_deepseek(system_prompt, user_prompt)


def _parse_model_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate(
    extraction: ExtractionResult, allowed_entity_types: list[str], allowed_relation_types: list[str]
) -> ValidationResult:
    """Review one Extraction Agent result (design-spec.md §5.2)."""
    deterministic_issues = _run_deterministic_checks(extraction, allowed_entity_types, allowed_relation_types)

    system_prompt = SYSTEM_PROMPT
    user_prompt = _build_user_prompt(extraction, deterministic_issues)

    raw = _call_model(system_prompt, user_prompt)
    try:
        parsed = _parse_model_response(raw)
    except json.JSONDecodeError:
        # One retry with a stricter follow-up — same single-pass policy as extraction_agent.py.
        raw = _call_model(
            system_prompt,
            user_prompt + "\n\nYour previous response was not valid JSON. "
            "Respond with ONLY a single valid JSON object and nothing else.",
        )
        parsed = _parse_model_response(raw)  # let this raise if it fails a second time

    additional_issues = [str(i) for i in parsed.get("additional_issues", []) if str(i).strip()]

    revised_confidence: RevisedConfidence | None = None
    rc = parsed.get("revised_confidence")
    if isinstance(rc, dict):
        idx = rc.get("entity_index")
        conf = rc.get("confidence")
        if isinstance(idx, int) and 0 <= idx < len(extraction.entities) and conf in ("high", "medium", "low"):
            revised_confidence = RevisedConfidence(entity_index=idx, confidence=conf)
        # else: malformed suggestion — dropped, same structural-sanity-only principle as extraction_agent.py

    all_issues = deterministic_issues + additional_issues
    status: Status = "flagged" if all_issues else "pass"

    return ValidationResult(text=extraction.text, status=status, issues=all_issues, revised_confidence=revised_confidence)
