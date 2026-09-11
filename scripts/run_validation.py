"""Batch-runs the Validation Agent over one split's extracted results
(design-spec.md §5.2, repo layout §8). Run from the repo root as:

    .venv\\Scripts\\python.exe -m scripts.run_validation --split gold

Processes whatever's already in extracted/{split}.jsonl (i.e. whatever
scripts/run_extraction.py has produced) — not an independent sample
selection of its own. Resumable (§6.2): skips indices already present in
validated/{split}.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import schema, store  # noqa: E402
from src.extraction_agent import ExtractedEntity, ExtractedRelation, ExtractionResult  # noqa: E402
from src.validation_agent import validate  # noqa: E402


def _to_extraction_result(record: dict) -> ExtractionResult:
    """store.py hands back plain dicts (see its module docstring) —
    reconstruct the dataclass validate() actually expects."""
    return ExtractionResult(
        text=record["text"],
        entities=[ExtractedEntity(**e) for e in record["entities"]],
        relations=[ExtractedRelation(**r) for r in record["relations"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run the Validation Agent over one split's extracted results.")
    parser.add_argument("--split", choices=["gold", "silver"], required=True)
    args = parser.parse_args()

    extracted = dict(store.iter_records("extracted", args.split))
    already_done = store.completed_indices("validated", args.split)
    todo = sorted(i for i in extracted if i not in already_done)
    entity_types = schema.entity_types()
    relation_types = schema.relation_types()

    print(f"split={args.split} extracted_available={len(extracted)} already_validated={len(extracted) - len(todo)} to_process={len(todo)}")

    succeeded = failed = flagged = 0
    for n, idx in enumerate(todo, 1):
        extraction = _to_extraction_result(extracted[idx])
        try:
            result = validate(extraction, entity_types, relation_types)
            store.put_record("validated", args.split, idx, result.to_dict())
        except Exception as e:  # one bad item shouldn't kill the whole batch
            print(f"[{n}/{len(todo)}] FAILED index={idx}: {e}")
            failed += 1
            continue
        succeeded += 1
        if result.status == "flagged":
            flagged += 1
        print(f"[{n}/{len(todo)}] OK index={idx}: status={result.status} issues={len(result.issues)}")

    print(f"done: succeeded={succeeded} failed={failed} flagged={flagged} ({flagged}/{succeeded} of processed items)")


if __name__ == "__main__":
    main()
