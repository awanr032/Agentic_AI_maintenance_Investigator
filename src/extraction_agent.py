"""Per-text entity/relation extraction agent (design-spec.md §5.1).

One LLM call per MWO text. Batching many independent calls across a whole
split is the caller's job (scripts/run_extraction.py, not built yet), not
this module's — MWOs are short and independent so that parallelizes
trivially at the call site (see CLAUDE.md).

Design decisions made here, resolved deliberately rather than guessed at:

- **Tokenization.** design-spec.md §3.1 flags that gold `start`/`end` are
  token indices, not character offsets, and says we must "keep the same
  tokenization the agent used, or re-tokenize consistently." Verified
  directly against gold_release.json: plain `text.split()` reproduces
  MaintIE's own `tokens` array exactly for 1,071/1,076 records (99.5%); the
  5 mismatches are MaintIE's own spelling normalization in `tokens`
  ("eng"->"engine", "cab"->"cabin") plus one genuine data anomaly (a "TBD"
  text paired with unrelated tokens) — not tokenizer disagreements. Token
  *boundaries* survive even the normalization cases. `_tokenize()` below is
  `text.split()` on that basis.

- **The model never reports token indices itself.** The output schema
  (§5.1) wants `start_token`/`end_token` per entity, but LLMs are unreliable
  at counting token positions. So the model is only asked for `span_text`
  (an exact, contiguous run of tokens — unambiguous string matching), and
  `_find_span()` resolves `start_token`/`end_token` deterministically by
  matching that string against our own `_tokenize()` output. Same principle
  the Pattern Agent's tool uses elsewhere in this design: the LLM chooses
  and narrates, code computes.

- **`<id>` placeholder tokens.** 687/1,076 gold texts (64%) start with the
  literal token `<id>` (a sanitization placeholder — see §3.1). The system
  prompt tells the model explicitly to ignore it, never extract an entity
  from it, while keeping it in the token list so indices stay aligned with
  gold's.

- **Type-taxonomy checking is deliberately NOT done here.** This function
  does not verify that a returned `type` string actually exists in
  data/scheme.json, or that a relation connects entity types the schema
  allows for it. That is the Validation/QA Agent's job (§5.2) — checking it
  here too would mean invalid extractions get silently dropped before the
  Validation Agent (and its honest "12% flagged" reporting) ever sees them.
  What *is* checked here is structural, not schema-legality: can the
  model's span_text be located in the text at all, and does every relation
  reference an entity that's actually present in this same result.

- **Provider.** Isolated behind `_call_model()`, the one function that
  talks to an LLM here (per CLAUDE.md: "keep each agent's API call isolated
  behind a small function so swapping providers is a one-line change").
  Default is DeepSeek (`deepseek-chat`, primary build target); set
  EXTRACTION_PROVIDER=claude (or change the default below) to run the
  Claude comparison pass (Haiku 4.5, per CLAUDE.md's model choices) instead
  — nothing else in this file changes either way.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_release.json"

# Reserved for few-shot prompting (pulled directly from gold_release.json,
# not hand-written, per §5.1). These specific indices MUST be excluded from
# any held-out evaluation sample scoring.py draws later — otherwise the
# model would be scored on texts it was shown the answer to in its own
# prompt. Exported so that future sampling code can exclude them by import
# rather than by re-stating this constant.
FEW_SHOT_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4)

Confidence = Literal["high", "medium", "low"]

PROVIDER = os.environ.get("EXTRACTION_PROVIDER", "deepseek")  # "deepseek" | "claude"


@dataclass
class ExtractedEntity:
    span_text: str
    start_token: int
    end_token: int
    type: str
    confidence: Confidence


@dataclass
class ExtractedRelation:
    head_span: str
    tail_span: str
    type: str


@dataclass
class ExtractionResult:
    text: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """For persisting via src/store.py's put_record(), which takes a
        plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Tokenization / span resolution
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization — see module docstring for the empirical
    validation against MaintIE's own `tokens` arrays."""
    return text.split()


def _find_span(tokens: list[str], span_text: str) -> tuple[int, int] | None:
    """Locate `span_text` as a contiguous run within `tokens`.

    Returns (start_token, end_token) using the same half-open convention as
    gold data, or None if it can't be found — treated by extract() as a
    malformed entity to drop, not an error to raise on (one bad entity
    shouldn't sink an otherwise-good extraction).
    """
    span_tokens = span_text.split()
    if not span_tokens:
        return None
    n = len(span_tokens)
    for start in range(len(tokens) - n + 1):
        if tokens[start:start + n] == span_tokens:
            return start, start + n
    return None


# ---------------------------------------------------------------------------
# Few-shot examples (from gold data, reshaped to this agent's output schema)
# ---------------------------------------------------------------------------

def _entity_span_text(tokens: list[str], entity: dict[str, Any]) -> str:
    return " ".join(tokens[entity["start"]:entity["end"]])


def _gold_item_to_example(item: dict[str, Any]) -> dict[str, Any]:
    """Reshape one gold_release.json record into exactly the subset of
    fields the model is asked to produce (span_text/type/confidence for
    entities, head_span/tail_span/type for relations) — NOT the full
    ExtractionResult schema, since we don't ask the model for
    start_token/end_token either (see module docstring)."""
    tokens = item["tokens"]
    entities = [
        {
            "span_text": _entity_span_text(tokens, e),
            "type": e["type"],
            "confidence": "high",  # gold labels are correct by definition
        }
        for e in item["entities"]
    ]
    relations = [
        {
            "head_span": _entity_span_text(tokens, item["entities"][r["head"]]),
            "tail_span": _entity_span_text(tokens, item["entities"][r["tail"]]),
            "type": r["type"],
        }
        for r in item["relations"]
    ]
    return {"text": item["text"], "entities": entities, "relations": relations}


@lru_cache(maxsize=1)
def _few_shot_examples() -> list[dict[str, Any]]:
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        gold = json.load(f)
    return [_gold_item_to_example(gold[i]) for i in FEW_SHOT_INDICES]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an information extraction agent for short maintenance work order (MWO) texts, \
following the MaintIE annotation scheme. Respond with a single JSON object only — no \
prose, no markdown code fences.

Allowed entity types (every entity's "type" must be EXACTLY one of these strings):
{entity_types}

Allowed relation types (every relation's "type" must be EXACTLY one of these strings):
{relation_types}

Rules:
- You are given a whitespace-tokenized list of tokens for one MWO text. Extract only from those tokens.
- If the first token is the literal placeholder "<id>", ignore it — never extract an entity from it.
- Every entity's "span_text" must be an exact, contiguous run of the given tokens (one or more consecutive tokens joined by a single space) — not a paraphrase or a partial word.
- Every entity needs a "confidence" of "high", "medium", or "low", reflecting how sure you are that both the span and the type are correct.
- Every relation connects two entities you extracted, referenced by their exact "span_text" values, as "head_span" and "tail_span".
- Only use types from the allowed lists above. If nothing in the text fits an allowed type, do not force a label — omit that entity or relation. Before answering, check that each type string you used appears EXACTLY, character-for-character, in the allowed list — do not assemble a plausible-looking path yourself (e.g. combining a real leaf name with the wrong parent category). If you are not fully sure a type string is really in the list, pick a shorter, more general prefix of it that you ARE sure is in the list (a real ancestor category is always safer than a specific-looking invented one).
- Prefer the most specific type available. Only use a bare top-level category (just "Activity", "PhysicalObject", "Process", "Property", or "State", with no more specific path) if you genuinely cannot narrow it down further — this should be rare. If you're unsure between a few specific subtypes, commit to your best specific guess and mark it "low" confidence rather than defaulting to the bare category.
- A word describing something ONGOING or HAPPENING (e.g. "leaking", "weeping", "overheating", "dripping") is almost always a Process (e.g. Process/UndesirableProcess), not a State. A State describes a static condition something IS IN (e.g. "broken", "worn", "cracked", "blown") — it doesn't happen over time, it just is. Do not label an ongoing-event word like "leaking" as a State/DegradedState just because it sounds like a bad condition.
- A pump, compressor, blower, or fan that MOVES OR PRODUCES a flow of liquid or gas is a GeneratingObject (e.g. PhysicalObject/GeneratingObject/LiquidFlowGeneratingObject for something moving liquid, PhysicalObject/GeneratingObject/GaseousFlowGeneratingObject for something moving gas) — NOT a DrivingObject, even though "fluid-powered" sounds like a plausible match. DrivingObject is for something that USES power to actuate mechanical motion (a hydraulic cylinder, an engine, a motor) — the difference is "moves the fluid/gas itself" (GeneratingObject) vs. "is driven by fluid/gas power to move something else" (DrivingObject).
- A tyre/tire or wheel is a PhysicalObject/GuidingObject/MechanicalEnergyGuidingObject (it guides mechanical motion) — do not fall back to a bare PhysicalObject for it.
- Before extracting a multi-word noun phrase as ONE entity, check whether it names one object at increasing specificity, or two DIFFERENT named components joined together — these need different treatment:
  * ONE object, increasing specificity: a modifier narrows what it is, but the last word (the head noun) alone is still fundamentally the exact same KIND of thing as the whole phrase — just less specific (e.g. "bend pulley": a bend pulley is still fundamentally a pulley). Extract BOTH the full phrase AND the bare head noun as separate entities, connected by an "isA" relation (full phrase isA head noun). Example: for "bend pulley", extract {{"span_text": "bend pulley", "type": "PhysicalObject/GuidingObject/MechanicalEnergyGuidingObject", "confidence": "high"}} AND {{"span_text": "pulley", "type": "PhysicalObject/GuidingObject/MechanicalEnergyGuidingObject", "confidence": "high"}}, plus {{"head_span": "bend pulley", "tail_span": "pulley", "type": "isA"}}.
  * TWO different named components joined together: the first word is not a mere descriptive modifier but names its own separate, different kind of object, and the phrase describes one component being part of / attached to the other (e.g. "brake hose": a hose that is part of a brake system — a hose is not a kind of brake, and a brake is not a kind of hose, they're two different real components). In this case do NOT extract the combined phrase as one entity at all — extract EACH word as its own separate entity directly, and connect them with "hasPart" (the containing/larger component hasPart the sub-part). Example: for "brake hose", extract {{"span_text": "brake", "type": "...", "confidence": "high"}} and {{"span_text": "hose", "type": "...", "confidence": "high"}} as two separate entities (never "brake hose" as a single combined span), plus {{"head_span": "brake", "tail_span": "hose", "type": "hasPart"}}.
  * If you're not confident which of these two cases applies, prefer leaving the phrase as a single, whole entity rather than guessing — an unnecessary split is also a mistake, not just a missing one.
- Output valid JSON with exactly this shape:
{{"text": "<the input text>", "entities": [{{"span_text": "...", "type": "...", "confidence": "high|medium|low"}}], "relations": [{{"head_span": "...", "tail_span": "...", "type": "..."}}]}}

Worked examples from the MaintIE gold corpus:

{examples}
"""


def _format_example(example: dict[str, Any]) -> str:
    tokens = _tokenize(example["text"])
    return f'tokens: {json.dumps(tokens)}\noutput: {json.dumps(example, ensure_ascii=False)}'


def _build_system_prompt(allowed_entity_types: list[str], allowed_relation_types: list[str]) -> str:
    examples = "\n\n".join(_format_example(ex) for ex in _few_shot_examples())
    return SYSTEM_PROMPT_TEMPLATE.format(
        entity_types=", ".join(allowed_entity_types),
        relation_types=", ".join(allowed_relation_types),
        examples=examples,
    )


def _build_user_prompt(tokens: list[str]) -> str:
    return f"tokens: {json.dumps(tokens)}"


# ---------------------------------------------------------------------------
# Provider calls — the only functions in this file that talk to an LLM
# ---------------------------------------------------------------------------

def _call_deepseek(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI  # imported lazily: don't require this SDK when PROVIDER == "claude"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set — copy .env.example to .env and fill it in.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        # "deepseek-chat" (CLAUDE.md's original model choice) was retired
        # 2026-07-24 — "deepseek-flash" is the current cheap/general model
        # (confirmed against api-docs.deepseek.com; verify again if this
        # starts erroring, DeepSeek's naming has already changed once).
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},  # requires the word "json" in the prompt — see SYSTEM_PROMPT_TEMPLATE
        temperature=0,
        # deepseek-flash defaults to an internal "thinking"/reasoning pass (observed:
        # 2,347 reasoning tokens for a 5-word input) that's billed as output tokens and
        # isn't needed for this bounded extraction task — disabling it cuts real cost
        # substantially (157 total tokens vs. thousands, observed on the same input) with
        # no correctness loss seen in spot checks. Discovered while debugging drafting_agent.py
        # silently returning empty output (finish_reason="length" — reasoning consumed the
        # entire max_tokens budget before any content was written); this agent has no
        # explicit max_tokens cap so it likely wasn't hitting that failure mode itself, but
        # was still paying the reasoning-token cost on every single call this whole project.
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
        max_tokens=1024,
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
    """Parse the model's JSON response, stripping a stray ```-fence if one
    slipped through despite the system prompt forbidding it — one
    defensive strip, not a general markdown parser."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract(text: str, allowed_entity_types: list[str], allowed_relation_types: list[str]) -> ExtractionResult:
    """Extract entities and relations from one MWO text (design-spec.md §5.1)."""
    tokens = _tokenize(text)
    system_prompt = _build_system_prompt(allowed_entity_types, allowed_relation_types)
    user_prompt = _build_user_prompt(tokens)

    raw = _call_model(system_prompt, user_prompt)
    try:
        parsed = _parse_model_response(raw)
    except json.JSONDecodeError:
        # One retry with a stricter follow-up — per design-spec.md §9 open
        # question 1's explicit recommendation, not a general retry system
        # (see the non-goals in §2: no retry/backoff sophistication for v1).
        raw = _call_model(
            system_prompt,
            user_prompt + "\n\nYour previous response was not valid JSON. "
            "Respond with ONLY a single valid JSON object and nothing else.",
        )
        parsed = _parse_model_response(raw)  # let this raise if it fails a second time

    entities: list[ExtractedEntity] = []
    for raw_entity in parsed.get("entities", []):
        span_text = raw_entity.get("span_text", "")
        span = _find_span(tokens, span_text)
        if span is None:
            continue  # unresolvable span: structurally malformed, drop (see module docstring)
        start_token, end_token = span
        confidence = raw_entity.get("confidence")
        if confidence not in ("high", "medium", "low"):
            confidence = "low"  # defensive default — never silently upgrade a missing value to "high"
        entities.append(
            ExtractedEntity(
                span_text=span_text,
                start_token=start_token,
                end_token=end_token,
                type=raw_entity.get("type", ""),
                confidence=confidence,
            )
        )

    known_spans = {e.span_text for e in entities}
    relations: list[ExtractedRelation] = []
    for raw_relation in parsed.get("relations", []):
        head_span = raw_relation.get("head_span", "")
        tail_span = raw_relation.get("tail_span", "")
        if head_span not in known_spans or tail_span not in known_spans:
            continue  # references an entity that was dropped above or never existed
        relations.append(ExtractedRelation(head_span=head_span, tail_span=tail_span, type=raw_relation.get("type", "")))

    return ExtractionResult(text=text, entities=entities, relations=relations)
