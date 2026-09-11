"""Deterministic tools the Pattern Agent calls.

Per design-spec.md §5.3: "The LLM's job is choosing which asset types to
investigate and narrating the pattern, not computing the counts." Every
function here is plain Python — no LLM calls, no randomness, same input
always gives the same output.

Reads validated results exclusively through src/store.py (never open()/
json.load() directly on pipeline data, per CLAUDE.md).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src import store

_FAILURE_RELATION_TYPES = {"hasParticipant/hasAgent", "hasParticipant/hasPatient", "hasProperty"}
_FAILURE_ENTITY_CLASSES = {"State", "Process"}


def _is_or_under(entity_type: str, asset_type: str) -> bool:
    """True if entity_type == asset_type or is a taxonomy descendant of it.

    Matched on "/"-delimited path segments, not raw substring, so
    "PhysicalObject/Driving" never matches "PhysicalObject/DrivingXYZ".
    """
    et = entity_type.split("/")
    at = asset_type.split("/")
    return len(et) >= len(at) and et[: len(at)] == at


def _validated_pass_records(split: str) -> dict[int, dict[str, Any]]:
    """Extracted records for `split`, restricted to indices the Validation
    Agent marked "pass" — design-spec.md §5.3: patterns are only mined from
    trusted data."""
    validated = dict(store.iter_records("validated", split))
    extracted = dict(store.iter_records("extracted", split))
    return {idx: extracted[idx] for idx, v in validated.items() if v.get("status") == "pass" and idx in extracted}


def list_common_asset_types(split: str = "silver", top_n: int = 20) -> list[dict[str, Any]]:
    """The most frequent PhysicalObject types across validated `split`
    extractions — candidates for the Pattern Agent to consider.

    Not itself an LLM-callable tool (there's no parameter worth an agent
    choosing — it's a fixed, one-shot lookup): computed once up front and
    handed to the model as context in the initial prompt.
    """
    counts: Counter[str] = Counter()
    for record in _validated_pass_records(split).values():
        for e in record.get("entities", []):
            if e["type"].split("/")[0] == "PhysicalObject":
                counts[e["type"]] += 1
    return [{"asset_type": t, "count": c} for t, c in counts.most_common(top_n)]


def get_failure_history(asset_type: str, top_n: int = 20, split: str = "silver") -> list[dict[str, Any]]:
    """For every validated (status=pass) text in `split` where an entity is
    `asset_type` or a taxonomy descendant of it, find State/Process entities
    connected to it via hasParticipant/hasProperty relations, and group by
    that entity's type — i.e. by failure/process signature.

    `occurrence_count` counts distinct source TEXTS showing the pairing, not
    raw relation instances — a text with two relations linking the same
    asset-failure pair (or the same pair in both connection directions)
    still counts once, matching what "recurred N times in the reviewed
    records" (design-spec.md §5.3's own example) should mean.

    Returns up to `top_n` records, most frequent first:
        {"asset_type": ..., "failure_type": ..., "occurrence_count": int,
         "example_source_texts": [...]}  # capped at 3 examples per record
    """
    records = _validated_pass_records(split)
    counts: Counter[str] = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)

    for record in records.values():
        entities = record.get("entities", [])
        asset_spans = {e["span_text"] for e in entities if _is_or_under(e["type"], asset_type)}
        if not asset_spans:
            continue
        failure_spans = {e["span_text"]: e["type"] for e in entities if e["type"].split("/")[0] in _FAILURE_ENTITY_CLASSES}
        if not failure_spans:
            continue

        seen_this_text: set[str] = set()
        for r in record.get("relations", []):
            if r["type"] not in _FAILURE_RELATION_TYPES:
                continue
            # Either side of the relation could be the asset or the failure
            # entity -- check both orientations rather than assuming a fixed
            # head/tail direction (see the design-spec.md correction: don't
            # assume a relation's direction without checking real data).
            for asset_span, other_span in ((r["head_span"], r["tail_span"]), (r["tail_span"], r["head_span"])):
                if asset_span in asset_spans and other_span in failure_spans:
                    failure_type = failure_spans[other_span]
                    if failure_type not in seen_this_text:
                        counts[failure_type] += 1
                        seen_this_text.add(failure_type)
                        if len(examples[failure_type]) < 3:
                            examples[failure_type].append(record["text"])

    return [
        {
            "asset_type": asset_type,
            "failure_type": failure_type,
            "occurrence_count": count,
            "example_source_texts": examples[failure_type],
        }
        for failure_type, count in counts.most_common(top_n)
    ]
