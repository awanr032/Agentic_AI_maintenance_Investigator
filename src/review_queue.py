"""Human-in-the-loop review layer (added after v1's original six-module build
order — see docs/design-spec.md's Human Review section for the rationale).

This module owns three things, all pure/testable logic with no LLM calls:

1. **Building a review queue** — not "review everything," which doesn't
   scale, but a targeted subset: every item the Validation Agent flagged
   (the known-questionable ones), plus a small random sample of items it
   passed (to catch the blind spot a flag-only process can never see: a
   case where Extraction and Validation both agreed and were both wrong —
   found concretely in this project's own investigation sessions).
2. **Recording a reviewer's decision** — accept / reject / fix — against
   one record, persisted through src/store.py like everything else.
3. **Merging corrections into the trusted record set** — `effective_records()`
   is what downstream consumers (src/tools.py's Pattern Agent input, and
   potentially a future re-scoring pass) should read instead of raw
   validated-pass records, so a correction actually takes effect rather
   than sitting in a file nobody reads.

Storage: corrections/{split}.jsonl, same append-only keyed-by-index shape
as extracted/validated (via store.py's generic collection support — no
change to store.py was needed). One correction record looks like:
    {"decision": "accept" | "reject" | "fix",
     "reviewer": str | None,
     "reviewed_at": "<ISO 8601 timestamp>",
     "original_status": "pass" | "flagged",
     "original_issues": [...],  # snapshot at review time, for an audit trail
     "corrected_record": {...} | None}  # required (and only meaningful) for "fix"
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from src import store

Decision = Literal["accept", "reject", "fix"]


@dataclass
class QueueItem:
    """One record awaiting human review, with enough context to judge it
    without opening extracted/validated files by hand."""

    index: int
    text: str
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    status: str  # "pass" | "flagged" — the Validation Agent's original verdict
    issues: list[str]
    reason_in_queue: Literal["flagged", "spot_check"]


def build_review_queue(split: str, spot_check_fraction: float = 0.03, seed: int = 42) -> list[QueueItem]:
    """Every flagged item, plus a small deterministic random sample of
    passed items (spot_check_fraction, default 3%) — see module docstring
    for why passed items are sampled too, not just flagged ones. Excludes
    indices already reviewed (a decision already recorded), so re-running
    this after some reviews only shows what's left.
    """
    extracted = dict(store.iter_records("extracted", split))
    validated = dict(store.iter_records("validated", split))
    already_reviewed = store.completed_indices("corrections", split)

    flagged = [idx for idx, v in validated.items() if v.get("status") == "flagged" and idx not in already_reviewed]

    passed = [idx for idx, v in validated.items() if v.get("status") == "pass" and idx not in already_reviewed]
    rng = random.Random(seed)
    spot_check = rng.sample(passed, k=round(len(passed) * spot_check_fraction)) if passed else []

    items: list[QueueItem] = []
    for idx in sorted(flagged):
        v, e = validated[idx], extracted.get(idx)
        if e is None:
            continue
        items.append(QueueItem(idx, e["text"], e["entities"], e["relations"], v["status"], v["issues"], "flagged"))
    for idx in sorted(spot_check):
        v, e = validated[idx], extracted.get(idx)
        if e is None:
            continue
        items.append(QueueItem(idx, e["text"], e["entities"], e["relations"], v["status"], v["issues"], "spot_check"))
    return items


def record_decision(
    split: str,
    index: int,
    decision: Decision,
    corrected_record: dict[str, Any] | None = None,
    reviewer: str | None = None,
) -> None:
    """Persist one reviewer decision. `corrected_record` is required for
    "fix" (an ExtractionResult-shaped dict — text/entities/relations) and
    ignored otherwise. `original_status`/`original_issues` are snapshotted
    from validated/{split}.jsonl at call time, not re-derived later, so the
    audit trail reflects what the reviewer actually saw.
    """
    if decision == "fix" and not corrected_record:
        raise ValueError('decision "fix" requires a corrected_record')
    validated = dict(store.iter_records("validated", split))
    v = validated.get(index, {})
    record = {
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "original_status": v.get("status"),
        "original_issues": v.get("issues", []),
        "corrected_record": corrected_record if decision == "fix" else None,
    }
    store.put_record("corrections", split, index, record)


def effective_records(split: str) -> dict[int, dict[str, Any]]:
    """The trusted record set for `split`, after applying human review —
    what src/tools.py and any other pattern-mining/reporting consumer
    should read, instead of duplicating this merge logic themselves.

    Base: validated (status == "pass") extractions — the same rule that
    applied before human review existed. Overlaid by any recorded
    correction:
      - "accept": include the ORIGINAL extraction even though it was
        flagged — the reviewer looked and judged the flag a false alarm
        (a real, observed case in this project: the overlap check firing
        on a deliberate isA-decomposition pair before that was fixed).
      - "fix": include the reviewer's corrected_record in place of the
        original.
      - "reject": exclude entirely, even if validation had marked it
        "pass" — this is what catches the blind spot a flag-only process
        can't: Extraction and Validation both agreeing and both being
        wrong. Without this override, a passed-but-actually-wrong record
        would silently stay trusted forever.
    """
    extracted = dict(store.iter_records("extracted", split))
    validated = dict(store.iter_records("validated", split))
    corrections = dict(store.iter_records("corrections", split))

    records: dict[int, dict[str, Any]] = {
        idx: extracted[idx] for idx, v in validated.items() if v.get("status") == "pass" and idx in extracted
    }
    for idx, c in corrections.items():
        decision = c.get("decision")
        if decision == "reject":
            records.pop(idx, None)
        elif decision == "accept" and idx in extracted:
            records[idx] = extracted[idx]
        elif decision == "fix" and c.get("corrected_record"):
            records[idx] = c["corrected_record"]
    return records


def review_stats(split: str) -> dict[str, Any]:
    """Summary counts — how much of the queue has been worked through and
    what reviewers decided. Useful for a status line in scripts/review.py
    and for the "shrinking loop" framing discussed for this feature: the
    review queue should get smaller over time as corrections improve the
    extraction prompt, not stay a fixed ongoing cost.
    """
    validated = dict(store.iter_records("validated", split))
    corrections = dict(store.iter_records("corrections", split))
    flagged_total = sum(1 for v in validated.values() if v.get("status") == "flagged")
    by_decision: dict[str, int] = {}
    for c in corrections.values():
        by_decision[c["decision"]] = by_decision.get(c["decision"], 0) + 1
    return {
        "flagged_total": flagged_total,
        "flagged_reviewed": sum(
            1 for idx, c in corrections.items() if validated.get(idx, {}).get("status") == "flagged"
        ),
        "reviewed_total": len(corrections),
        "by_decision": by_decision,
    }
