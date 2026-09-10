"""Deterministic precision/recall/F1 scoring of extractions vs. gold labels.

NOT an LLM call — pure Python only, per CLAUDE.md. Reads gold_release.json
directly (a source input, same exemption schema.py/extraction_agent.py use
for scheme.json — see store.py's module docstring); reads pipeline-generated
extracted/validated results exclusively through src/store.py, never via
open()/json.load() directly, per the same rule.

Design decision, resolved not guessed at silently (design-spec.md §9 open
question 2, "need exact-match vs partial-match F1 defined before writing
scoring.py"):

- **Exact match is the primary, headline metric** — an entity counts as
  correct only if its (start_token, end_token, type) all match a gold
  entity exactly; a relation counts as correct only if both endpoint
  entities AND the relation type match exactly. This is the strict,
  standard convention (matches how MaintIE's own SpERT baseline is scored,
  making the two numbers comparable per §5.1's acceptance criteria).
- **Two well-defined, unambiguous "looser" diagnostics are reported
  alongside it**, not one fuzzy "partial match": `entity_boundary_only`
  (span must match exactly, type ignored — separates "found the right
  words" from "labeled them right") and `relation_pair_only` (the same
  idea for relations — right two entities connected, relation type
  ignored). Deliberately NOT building an overlap-threshold partial-match
  metric (e.g. "50%+ token overlap counts") — that needs its own arbitrary
  threshold nobody has specified, which would just be a second undocumented
  guess layered on top of the first. Boundary/pair-only matching needs no
  such threshold: it's exact match on a subset of the fields.
- Confidence never factors into scoring — it's a separate "how sure is the
  model" signal (§5.1), orthogonal to whether a label is actually correct.

Both a `score_item()` restricted to one text (pure, no I/O, independently
testable with a fixture per CLAUDE.md's code conventions) and an
`evaluate_gold_split()` orchestrator (does the I/O, aggregates, persists
the report) are provided — matching the same split-testable-core /
impure-orchestrator shape as the rest of this codebase.

`evaluate_gold_split()` reports two aggregates side by side: `all_extracted`
(every sample item that made it through Extraction, regardless of
Validation Agent status) and `trusted_only` (validation status == "pass"
only). Comparing the two is a coarse, free signal toward §5.2's own
acceptance criterion — "check whether flagged items correlate with actual
extraction errors" — a materially higher F1 in `trusted_only` suggests the
Validation Agent is adding real signal, not noise.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import store
from src.extraction_agent import FEW_SHOT_INDICES

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_release.json"

EntityTuple = tuple[int, int, str]  # (start_token, end_token, type)
RelationTuple = tuple[EntityTuple, EntityTuple, str]  # (head, tail, relation_type)

_UNRESOLVED: EntityTuple = (-1, -1, "<UNRESOLVED>")  # never matches a real gold tuple


@lru_cache(maxsize=1)
def load_gold() -> list[dict[str, Any]]:
    """Read data/gold_release.json directly — a source input, not pipeline
    data (see module docstring)."""
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def held_out_gold_indices(sample_size: int | None = 100, seed: int = 42) -> list[int]:
    """The gold indices scoring should be run against.

    Always excludes FEW_SHOT_INDICES (imported from extraction_agent, not
    restated) — those texts were shown to the model as worked examples, so
    scoring on them would inflate F1 dishonestly. `sample_size=None` returns
    every eligible gold index instead of a sample (e.g. for a final,
    fuller-than-~100-texts case-study run once a smaller test looks good).
    """
    eligible = [i for i in range(len(load_gold())) if i not in FEW_SHOT_INDICES]
    if sample_size is None:
        return eligible
    rng = random.Random(seed)
    return sorted(rng.sample(eligible, min(sample_size, len(eligible))))


# ---------------------------------------------------------------------------
# Tuple extraction — predicted (our dict shape) and gold (MaintIE's shape)
# ---------------------------------------------------------------------------

def _predicted_entity_tuples(predicted: dict[str, Any]) -> list[EntityTuple]:
    return [(e["start_token"], e["end_token"], e["type"]) for e in predicted.get("entities", [])]


def _entity_lookup(predicted: dict[str, Any]) -> dict[str, EntityTuple]:
    """span_text -> (start_token, end_token, type), for resolving relations'
    head_span/tail_span back to the entity they refer to. If two entities
    share identical span_text (should already be caught upstream by
    validation_agent.py's overlap check), the last one wins — a known,
    accepted imprecision, not a case worth building bipartite resolution
    for."""
    return {e["span_text"]: (e["start_token"], e["end_token"], e["type"]) for e in predicted.get("entities", [])}


def _predicted_relation_tuples(predicted: dict[str, Any]) -> list[RelationTuple]:
    lookup = _entity_lookup(predicted)
    tuples: list[RelationTuple] = []
    for r in predicted.get("relations", []):
        head = lookup.get(r["head_span"], _UNRESOLVED)
        tail = lookup.get(r["tail_span"], _UNRESOLVED)
        # Unresolvable references (shouldn't happen for our own
        # extraction_agent.py output, but scoring.py may be fed fixtures or
        # externally-built data) become _UNRESOLVED, which can never match a
        # real gold tuple — they always land as an FP, rather than being
        # silently dropped and understating how wrong the prediction was.
        tuples.append((head, tail, r["type"]))
    return tuples


def _gold_entity_tuples(gold_item: dict[str, Any]) -> list[EntityTuple]:
    return [(e["start"], e["end"], e["type"]) for e in gold_item["entities"]]


def _gold_relation_tuples(gold_item: dict[str, Any]) -> list[RelationTuple]:
    entities = gold_item["entities"]
    tuples: list[RelationTuple] = []
    for r in gold_item.get("relations", []):
        head_e = entities[r["head"]]
        tail_e = entities[r["tail"]]
        tuples.append(((head_e["start"], head_e["end"], head_e["type"]), (tail_e["start"], tail_e["end"], tail_e["type"]), r["type"]))
    return tuples


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _match_multiset(predicted: list[Any], gold: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    """Generic exact multiset matcher — works for entity tuples, relation
    tuples, or any hashable key. Returns (tp, fp, fn) as lists of the
    matched/unmatched items themselves, so callers can inspect or bucket
    them (e.g. by type) afterwards."""
    pred_counter = Counter(predicted)
    gold_counter = Counter(gold)
    tp = list((pred_counter & gold_counter).elements())
    fp = list((pred_counter - gold_counter).elements())
    fn = list((gold_counter - pred_counter).elements())
    return tp, fp, fn


@dataclass
class ItemScore:
    """Per-text match results — pure data, no I/O. entity_exact/relation_exact
    tuples carry a type; entity_boundary/relation_pair tuples deliberately
    don't (see module docstring)."""

    entity_exact: tuple[list[EntityTuple], list[EntityTuple], list[EntityTuple]]
    entity_boundary: tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]
    relation_exact: tuple[list[RelationTuple], list[RelationTuple], list[RelationTuple]]
    relation_pair: tuple[list[tuple[EntityTuple, EntityTuple]], list[tuple[EntityTuple, EntityTuple]], list[tuple[EntityTuple, EntityTuple]]]


def score_item(predicted: dict[str, Any], gold_item: dict[str, Any]) -> ItemScore:
    """Score one ExtractionResult-shaped dict against its matching gold_release.json item.

    Pure function, no I/O — independently testable with a fixture pair, per
    CLAUDE.md's code conventions.
    """
    pred_entities = _predicted_entity_tuples(predicted)
    gold_entities = _gold_entity_tuples(gold_item)
    entity_exact = _match_multiset(pred_entities, gold_entities)
    entity_boundary = _match_multiset(
        [(s, e) for s, e, _ in pred_entities], [(s, e) for s, e, _ in gold_entities]
    )

    pred_relations = _predicted_relation_tuples(predicted)
    gold_relations = _gold_relation_tuples(gold_item)
    relation_exact = _match_multiset(pred_relations, gold_relations)
    relation_pair = _match_multiset(
        [(h, t) for h, t, _ in pred_relations], [(h, t) for h, t, _ in gold_relations]
    )

    return ItemScore(entity_exact, entity_boundary, relation_exact, relation_pair)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _counts_dict(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _aggregate_typed(tp: list[Any], fp: list[Any], fn: list[Any], type_index: int) -> dict[str, Any]:
    """Overall counts plus a per-type breakdown, bucketing each tuple by its
    element at `type_index` (2 for both entity tuples (start,end,type) and
    relation tuples (head,tail,type) — a happy coincidence of the schemas,
    not a coincidence I depend on elsewhere)."""
    result = _counts_dict(len(tp), len(fp), len(fn))
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for t in tp:
        by_type[t[type_index]]["tp"] += 1
    for t in fp:
        by_type[t[type_index]]["fp"] += 1
    for t in fn:
        by_type[t[type_index]]["fn"] += 1
    result["by_type"] = {name: _counts_dict(c["tp"], c["fp"], c["fn"]) for name, c in sorted(by_type.items())}
    return result


def aggregate_scores(item_scores: list[ItemScore]) -> dict[str, Any]:
    """Sum per-item match results into overall + per-type precision/recall/F1."""
    e_exact_tp: list[EntityTuple] = []
    e_exact_fp: list[EntityTuple] = []
    e_exact_fn: list[EntityTuple] = []
    e_bound_tp: list[Any] = []
    e_bound_fp: list[Any] = []
    e_bound_fn: list[Any] = []
    r_exact_tp: list[RelationTuple] = []
    r_exact_fp: list[RelationTuple] = []
    r_exact_fn: list[RelationTuple] = []
    r_pair_tp: list[Any] = []
    r_pair_fp: list[Any] = []
    r_pair_fn: list[Any] = []

    for s in item_scores:
        e_exact_tp += s.entity_exact[0]; e_exact_fp += s.entity_exact[1]; e_exact_fn += s.entity_exact[2]
        e_bound_tp += s.entity_boundary[0]; e_bound_fp += s.entity_boundary[1]; e_bound_fn += s.entity_boundary[2]
        r_exact_tp += s.relation_exact[0]; r_exact_fp += s.relation_exact[1]; r_exact_fn += s.relation_exact[2]
        r_pair_tp += s.relation_pair[0]; r_pair_fp += s.relation_pair[1]; r_pair_fn += s.relation_pair[2]

    return {
        "num_items": len(item_scores),
        "entity_exact_match": _aggregate_typed(e_exact_tp, e_exact_fp, e_exact_fn, type_index=2),
        "entity_boundary_only": _counts_dict(len(e_bound_tp), len(e_bound_fp), len(e_bound_fn)),
        "relation_exact_match": _aggregate_typed(r_exact_tp, r_exact_fp, r_exact_fn, type_index=2),
        "relation_pair_only": _counts_dict(len(r_pair_tp), len(r_pair_fp), len(r_pair_fn)),
    }


# ---------------------------------------------------------------------------
# Orchestration — the one impure function in this module
# ---------------------------------------------------------------------------

def evaluate_gold_split(sample_size: int | None = 100, seed: int = 42) -> dict[str, Any]:
    """Score whatever's already in the store for the gold split's held-out
    sample, write scores/eval_report.json via store.py, and return it.

    Only scores indices that are actually present in extracted/gold.jsonl —
    silently skipping the rest would misrepresent progress, so
    `items_missing_from_store` reports how many of the requested sample
    haven't been run through Extraction yet, rather than pretending the
    sample was smaller than requested.
    """
    gold = load_gold()
    indices = held_out_gold_indices(sample_size, seed)
    extracted = dict(store.iter_records("extracted", "gold"))
    validated = dict(store.iter_records("validated", "gold"))

    all_scores: list[ItemScore] = []
    trusted_scores: list[ItemScore] = []
    missing = 0
    for idx in indices:
        pred = extracted.get(idx)
        if pred is None:
            missing += 1
            continue
        item_score = score_item(pred, gold[idx])
        all_scores.append(item_score)
        v = validated.get(idx)
        if v is not None and v.get("status") == "pass":
            trusted_scores.append(item_score)

    report = {
        "sample_size_requested": sample_size,
        "sample_indices_count": len(indices),
        "items_missing_from_store": missing,
        "all_extracted": aggregate_scores(all_scores),
        "trusted_only": aggregate_scores(trusted_scores),
    }
    store.write_doc("scores/eval_report", report)
    return report
