"""Batch-runs the Extraction Agent over one split (design-spec.md §5.1, repo
layout §8). Run from the repo root as:

    .venv\\Scripts\\python.exe -m scripts.run_extraction --split gold

Per CLAUDE.md's guardrail ("test on a 10-50 text sample first"), defaults to
a small sample rather than the full ~1,076/~7,000-text split — pass --all to
override, after checking docs/design-spec.md's cost section.

Resumable (design-spec.md §6.2): skips any index already present in
extracted/{split}.jsonl, so a crash or interruption partway through doesn't
mean re-running (and re-paying for) everything already done.

Reads data/gold_release.json / data/silver_release.json directly — source
inputs, same exemption as schema.py/extraction_agent.py/scoring.py use for
their own source-data reads (see store.py's module docstring). Only the
Extraction Agent's own output goes through store.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import schema, scoring, store  # noqa: E402
from src.extraction_agent import extract  # noqa: E402

SILVER_PATH = Path(__file__).resolve().parent.parent / "data" / "silver_release.json"


@lru_cache(maxsize=1)
def _load_silver() -> list[dict]:
    with SILVER_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_texts(split: str) -> list[dict]:
    return scoring.load_gold() if split == "gold" else _load_silver()


def _select_indices(split: str, sample_size: int | None, seed: int) -> list[int]:
    """gold reuses scoring.held_out_gold_indices() (excludes few-shot
    indices, deterministic per seed) so a later scoring run is scoped
    consistently. silver has no held-out/leakage concept — it's just a
    plain sample for pattern-mining volume."""
    if split == "gold":
        return scoring.held_out_gold_indices(sample_size=sample_size, seed=seed)
    total = len(_load_silver())
    eligible = list(range(total))
    if sample_size is None:
        return eligible
    rng = random.Random(seed)
    return sorted(rng.sample(eligible, min(sample_size, total)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run the Extraction Agent over one split.")
    parser.add_argument("--split", choices=["gold", "silver"], required=True)
    parser.add_argument("--sample-size", type=int, default=20, help="Texts to process (default 20). Ignored if --all is given.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed, for reproducibility.")
    parser.add_argument(
        "--all", action="store_true",
        help="Process every eligible text instead of a sample. Expensive on silver "
        "(~7,000 texts) — check docs/design-spec.md's cost section first.",
    )
    args = parser.parse_args()

    sample_size = None if args.all else args.sample_size
    indices = _select_indices(args.split, sample_size, args.seed)
    texts = _load_texts(args.split)
    already_done = store.completed_indices("extracted", args.split)
    entity_types = schema.entity_types()
    relation_types = schema.relation_types()

    todo = [i for i in indices if i not in already_done]
    print(f"split={args.split} requested={len(indices)} already_done={len(indices) - len(todo)} to_process={len(todo)}")

    succeeded = failed = 0
    for n, idx in enumerate(todo, 1):
        text = texts[idx]["text"]
        try:
            result = extract(text, entity_types, relation_types)
            store.put_record("extracted", args.split, idx, result.to_dict())
        except Exception as e:  # one bad item shouldn't kill the whole batch — no retry/backoff built for v1, just a skip
            print(f"[{n}/{len(todo)}] FAILED index={idx}: {e}")
            failed += 1
            continue
        succeeded += 1
        print(f"[{n}/{len(todo)}] OK index={idx}: {text!r} -> {len(result.entities)} entities, {len(result.relations)} relations")

    print(f"done: succeeded={succeeded} failed={failed} skipped_already_done={len(indices) - len(todo)}")


if __name__ == "__main__":
    main()
