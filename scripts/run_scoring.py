"""Thin CLI wrapper around src/scoring.py's evaluate_gold_split() (repo
layout §8 lists this as a separate script from src/scoring.py itself — the
module holds the scoring logic, this just runs it and prints the result).
Run from the repo root as:

    .venv\\Scripts\\python.exe -m scripts.run_scoring --sample-size 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import scoring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score extracted/validated gold-split results against gold labels.")
    parser.add_argument("--sample-size", type=int, default=100, help="Held-out gold sample size (default 100, per §5.1's acceptance criteria).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Score every eligible gold text instead of a sample.")
    args = parser.parse_args()

    sample_size = None if args.all else args.sample_size
    report = scoring.evaluate_gold_split(sample_size=sample_size, seed=args.seed)
    print(json.dumps(report, indent=2))
    print()
    print("written to scores/eval_report.json")
    if report["items_missing_from_store"]:
        print(
            f"note: {report['items_missing_from_store']} of the {report['sample_indices_count']} sample "
            "items haven't been run through scripts/run_extraction.py yet (run it first for the same "
            "--split gold with a matching or larger --sample-size)."
        )


if __name__ == "__main__":
    main()
