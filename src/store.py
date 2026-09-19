"""Storage layer for the maintenance-investigator pipeline — local flat
files by default, S3 when STORE_S3_BUCKET is set.

This is the ONLY module allowed to touch the filesystem/S3 for pipeline
data (data/scheme.json, gold_release.json, silver_release.json are source
inputs and are exempt — schema.py reads those directly). Every agent,
script, and the scoring module must read/write pipeline results through
the functions below, never via open()/json.load()/pathlib/boto3 directly.

Backend selection (design-spec.md §6.1's "single change point"): if the
STORE_S3_BUCKET environment variable is set, every function below reads
and writes S3 objects in that bucket instead of local files. Local dev
work is unaffected — nothing sets that variable unless you're running in
(or targeting) Lambda, where local disk is ephemeral. This is why the
migration didn't need to touch extraction_agent.py, tools.py, or any
batch script: they only ever called store.py's public functions, exactly
as designed.

Local disk layout, relative to STORE_ROOT (the repo root):
    extracted/{split}.jsonl   one ExtractionResult per line, keyed by index
    validated/{split}.jsonl   one ValidationResult per line, keyed by index
    scores/eval_report.json   single JSON document (scoring module output)
    patterns/findings.json    single JSON document (Pattern Agent output)

S3 layout (same collection/split/name vocabulary, different physical
shape — one object per record instead of one growing JSONL file):
    s3://{bucket}/extracted/{split}/{index}.json
    s3://{bucket}/validated/{split}/{index}.json
    s3://{bucket}/scores/eval_report.json
    s3://{bucket}/patterns/findings.json

One-object-per-record (rather than porting the JSONL-append shape as-is)
was a deliberate choice, not an arbitrary translation: S3 has no efficient
in-place append, so a single growing JSONL object would mean a full
read-modify-write on every put_record() call — exactly the kind of
race/contention Lambda's concurrent, stateless invocations are prone to.
One object per record makes completed_indices() a plain prefix listing,
put_record() a single independent PUT with no read-before-write, and
resumability (design-spec.md §6.2) fall out for free — the same property
the local JSONL log's "last occurrence wins" logic had to work harder for.

"split" is a caller-chosen label such as "gold" or "silver". "index" is the
item's position in that source dataset — the key every downstream stage
joins on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

STORE_ROOT = Path(__file__).resolve().parent.parent
S3_BUCKET = os.environ.get("STORE_S3_BUCKET")

_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3  # imported lazily: local dev never needs this installed

        _s3_client = boto3.client("s3")
    return _s3_client


# ---------------------------------------------------------------------------
# Keyed collections: extracted/{split}.jsonl or validated/{split}/{index}.json
# ---------------------------------------------------------------------------

def _jsonl_path(collection: str, split: str) -> Path:
    return STORE_ROOT / collection / f"{split}.jsonl"


def _s3_prefix(collection: str, split: str) -> str:
    return f"{collection}/{split}/"


def _s3_key(collection: str, split: str, index: int) -> str:
    return f"{collection}/{split}/{index}.json"


def completed_indices(collection: str, split: str) -> set[int]:
    """Indices already recorded in collection/split.

    Call this once at the start of a batch run (not per item) and skip any
    index already in the returned set — the resumability check required by
    design-spec.md §6.2: without it, a crash or rate-limit death partway
    through ~8,076 LLM calls means re-running, and re-paying for, everything
    already done.
    """
    if S3_BUCKET:
        indices: set[int] = set()
        paginator = _s3().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=_s3_prefix(collection, split)):
            for obj in page.get("Contents", []):
                filename = obj["Key"].rsplit("/", 1)[-1]
                indices.add(int(filename[: -len(".json")]))
        return indices

    path = _jsonl_path(collection, split)
    if not path.exists():
        return set()
    indices = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(json.loads(line)["index"])
    return indices


def put_record(collection: str, split: str, index: int, record: dict[str, Any]) -> None:
    """Store `record` for `index` in collection/split.

    Local backend: append to the JSONL log (callers must check
    completed_indices() first — this does not deduplicate on write).
    S3 backend: an independent PUT of one object — no read-before-write,
    so double-writing the same index is harmless AND cheap, not just
    harmless, unlike the local log's "scan and dedupe on read" approach.
    """
    if S3_BUCKET:
        _s3().put_object(
            Bucket=S3_BUCKET,
            Key=_s3_key(collection, split, index),
            Body=json.dumps(record, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        return

    path = _jsonl_path(collection, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"index": index, "record": record}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def iter_records(collection: str, split: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (index, record) for every index in collection/split, in index order.

    Local backend: if an index appears more than once in the log, only its
    last occurrence is yielded — safe after a resumed/re-run batch. S3
    backend: no duplicates are possible by construction (one object per
    index, later PUTs simply overwrite).
    """
    if S3_BUCKET:
        # Fetched in parallel, not one GET per record in a loop: a full-split
        # scan (what tools.py's _validated_pass_records() does on EVERY tool
        # call — the read pattern Pattern/Query Agent actually use, not an
        # edge case) means ~500 individual network round-trips for the
        # silver split. Sequentially that's slow enough to blow past even a
        # generous Lambda timeout -- caught for real: the deployed Query
        # Agent timed out at 30s on its first live question. A thread pool
        # is safe here because these are independent, read-only GETs with no
        # ordering dependency between them (the final sort below restores
        # index order regardless of completion order).
        import concurrent.futures

        keys: list[tuple[int, str]] = []
        paginator = _s3().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=_s3_prefix(collection, split)):
            for obj in page.get("Contents", []):
                filename = obj["Key"].rsplit("/", 1)[-1]
                keys.append((int(filename[: -len(".json")]), obj["Key"]))

        def _fetch(item: tuple[int, str]) -> tuple[int, dict[str, Any]]:
            index, key = item
            body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            return index, json.loads(body)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            results = list(pool.map(_fetch, keys))
        for index, record in sorted(results):
            yield index, record
        return

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

    S3 backend: one direct GET (cheap, O(1)), unlike the local backend's
    iter_records()-based scan — a nice side effect of one-object-per-record,
    not the reason it was chosen (that was resumability/contention, above).
    """
    if S3_BUCKET:
        try:
            body = _s3().get_object(Bucket=S3_BUCKET, Key=_s3_key(collection, split, index))["Body"].read()
            return json.loads(body)
        except _s3().exceptions.NoSuchKey:
            return None

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

    Unlike the keyed collections, these are whole-document outputs produced
    once per run, not accumulated per-item — overwrite-on-write is correct
    here on both backends, not an append log.
    """
    if S3_BUCKET:
        _s3().put_object(
            Bucket=S3_BUCKET,
            Key=f"{name}.json",
            Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return

    path = _doc_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_doc(name: str) -> Any | None:
    """Read a single JSON document written by write_doc(), or None if absent."""
    if S3_BUCKET:
        try:
            body = _s3().get_object(Bucket=S3_BUCKET, Key=f"{name}.json")["Body"].read()
            return json.loads(body)
        except _s3().exceptions.NoSuchKey:
            return None

    path = _doc_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
