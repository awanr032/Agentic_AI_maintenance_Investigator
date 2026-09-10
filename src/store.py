"""Flat-file local storage layer for the maintenance-investigator pipeline.

This is the ONLY module allowed to touch the filesystem for pipeline data
(data/scheme.json, gold_release.json, silver_release.json are source inputs
and are exempt — schema.py reads those directly). Every agent, script, and
the scoring module must read/write pipeline results through the functions
below, never via open()/json.load()/pathlib directly. Reason: on AWS, local
disk is ephemeral (Lambda/Fargate wipe it between runs), so this file is the
single change point when that migration happens — see design-spec.md §6.1.

Disk layout, relative to STORE_ROOT (the repo root):
    extracted/{split}.jsonl   one ExtractionResult per line, keyed by index
    validated/{split}.jsonl   one ValidationResult per line, keyed by index
    scores/eval_report.json   single JSON document (scoring module output)
    patterns/findings.json    single JSON document (Pattern Agent output)

"split" is a caller-chosen label such as "gold" or "silver". "index" is the
item's position in that source dataset — the key every downstream stage
joins on.

Keyed collections ("extracted", "validated") are append-only JSONL logs, not
read-modify-write files. This keeps put_record() a single cheap append with
no read-before-write race, which matters because §6.2 requires every batch
script to be resumable: check completed_indices() once at the start of a
run, skip anything already in it, and only call put_record() for the rest.
If a record for the same index is appended twice (e.g. a script re-run
without checking first), the later line wins — iter_records()/get_record()
scan the whole log and keep the last occurrence per index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

STORE_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Keyed, append-only collections: extracted/{split}.jsonl, validated/{split}.jsonl
# ---------------------------------------------------------------------------

def _jsonl_path(collection: str, split: str) -> Path:
    return STORE_ROOT / collection / f"{split}.jsonl"


def completed_indices(collection: str, split: str) -> set[int]:
    """Indices already recorded in collection/split.

    Call this once at the start of a batch run (not per item — it scans the
    whole log) and skip any index already in the returned set. This is the
    resumability check required by design-spec.md §6.2: without it, a crash
    or rate-limit death partway through ~8,076 LLM calls means re-running,
    and re-paying for, everything already done.
    """
    path = _jsonl_path(collection, split)
    if not path.exists():
        return set()
    indices: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(json.loads(line)["index"])
    return indices


def put_record(collection: str, split: str, index: int, record: dict[str, Any]) -> None:
    """Append `record` for `index` to collection/split.

    Callers must check completed_indices() first — this does not
    deduplicate on write (append-only, no read-before-write). Writing the
    same index twice is harmless (readers keep the last occurrence) but
    wasteful, and defeats the point of checking first.
    """
    path = _jsonl_path(collection, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"index": index, "record": record}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def iter_records(collection: str, split: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (index, record) for every index in collection/split.

    If an index appears more than once in the log, only its last occurrence
    is yielded, in index order (not file order) — safe to call after
    resumed/re-run batches without seeing stale duplicates.
    """
    path = _jsonl_path(collection, split)
    if not path.exists():
        return
    latest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                latest[row["index"]] = row["record"]
    for index in sorted(latest):
        yield index, latest[index]


def get_record(collection: str, split: str, index: int) -> dict[str, Any] | None:
    """Fetch a single record by index, or None if it isn't present.

    Convenience wrapper over iter_records() for one-off lookups (e.g. the
    Drafting Agent pulling one Extraction Agent result for a per-order
    report). Batch scoring/pattern-mining over a whole split should use
    iter_records() instead — it reads the file once.
    """
    for i, record in iter_records(collection, split):
        if i == index:
            return record
    return None


# ---------------------------------------------------------------------------
# Single-document outputs: scores/eval_report.json, patterns/findings.json
# ---------------------------------------------------------------------------

def _doc_path(name: str) -> Path:
    """`name` is a collection/document pair, e.g. "scores/eval_report"."""
    return STORE_ROOT / f"{name}.json"


def write_doc(name: str, data: Any) -> None:
    """Overwrite a single JSON document, e.g. write_doc("scores/eval_report", {...}).

    Unlike the keyed collections, these are whole-document outputs (the
    scoring module's report, the Pattern Agent's findings) produced once per
    run, not accumulated per-item — so overwrite-on-write is correct here,
    not an append log.
    """
    path = _doc_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_doc(name: str) -> Any | None:
    """Read a single JSON document written by write_doc(), or None if absent."""
    path = _doc_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
