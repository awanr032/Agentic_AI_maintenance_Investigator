"""Runs the Pattern Agent over one split's validated data (design-spec.md
§5.3, repo layout §8). Run from the repo root as:

    .venv\\Scripts\\python.exe -m scripts.run_patterns --split silver

Not really "resumable" the way extraction/validation are — this is a single
batch investigation over the whole validated split each time, not a
per-text loop, so re-running just re-investigates from scratch (and
overwrites patterns/findings.json). See src/pattern_agent.py's own
find_patterns() for what happens inside one run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pattern_agent import find_patterns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pattern Agent over one split's validated data.")
    parser.add_argument("--split", choices=["gold", "silver"], default="silver")
    parser.add_argument("--num-candidates", type=int, default=20, help="Asset-type shortlist size shown to the model.")
    args = parser.parse_args()

    findings = find_patterns(split=args.split, num_candidates=args.num_candidates)
    print(f"{len(findings)} pattern(s) survived verification (see src/pattern_agent.py's _verify_against_observed):")
    for f in findings:
        print(f" - {f.asset_type}: {f.pattern} (occurrence_count={f.occurrence_count})")
        for example in f.example_source_texts:
            print(f"     e.g. {example!r}")
    print()
    print("written to patterns/findings.json")


if __name__ == "__main__":
    main()
