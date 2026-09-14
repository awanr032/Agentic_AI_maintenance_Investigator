"""Interactive human review CLI for src/review_queue.py. Run from the repo
root as:

    .venv\\Scripts\\python.exe -m scripts.review --split silver
    .venv\\Scripts\\python.exe -m scripts.review --split silver --stats

For each item in the queue (every flagged record, plus a small random
sample of passed ones — see review_queue.build_review_queue()'s docstring
for why), shows the source text, extraction, and the Validation Agent's
issues (if any), then asks the reviewer to decide:

    a  accept as-is (include in downstream pattern-mining, even if flagged)
    r  reject (exclude from downstream pattern-mining, even if it passed)
    f  fix (paste the path to a corrected-record JSON file)
    s  skip (leave unreviewed, ask again next run)
    q  quit (stop the session; unreviewed items stay in the queue)

A "fix" JSON file must be one ExtractionResult-shaped object: the same
{"text", "entities", "relations"} dict extraction_agent.py itself produces
— that's a deliberate constraint, not a limitation: it means a reviewer's
correction slots into effective_records() exactly like a normal extraction
result, no special-casing needed downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import review_queue  # noqa: E402


def _print_item(item: review_queue.QueueItem) -> None:
    print(f"\n{'=' * 70}")
    print(f"index={item.index}  status={item.status}  queued_because={item.reason_in_queue}")
    print(f"text: {item.text!r}")
    print(f"entities: {[(e['span_text'], e['type'], e['confidence']) for e in item.entities]}")
    print(f"relations: {[(r['head_span'], r['type'], r['tail_span']) for r in item.relations]}")
    if item.issues:
        print("issues:")
        for i in item.issues:
            print(f"  - {i}")
    else:
        print("issues: (none — this is a random spot-check of a passed item)")


def _load_corrected_record(path_str: str) -> dict | None:
    path = Path(path_str.strip())
    if not path.is_file():
        print(f"  no such file: {path}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  not valid JSON: {e}")
        return None
    if not all(k in data for k in ("text", "entities", "relations")):
        print('  file must be a JSON object with "text", "entities", "relations" keys')
        return None
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively review flagged (and spot-checked) extractions.")
    parser.add_argument("--split", choices=["gold", "silver"], required=True)
    parser.add_argument("--spot-check-fraction", type=float, default=0.03)
    parser.add_argument("--reviewer", default=None, help="Optional name/id recorded with each decision.")
    parser.add_argument("--stats", action="store_true", help="Print review progress and exit, no interactive session.")
    args = parser.parse_args()

    if args.stats:
        print(json.dumps(review_queue.review_stats(args.split), indent=2))
        return

    queue = review_queue.build_review_queue(args.split, spot_check_fraction=args.spot_check_fraction)
    if not queue:
        print("Nothing to review — queue is empty (everything already reviewed, or nothing flagged).")
        return

    print(f"{len(queue)} item(s) to review for split={args.split}.")
    for n, item in enumerate(queue, 1):
        _print_item(item)
        while True:
            choice = input(f"[{n}/{len(queue)}] (a)ccept / (r)eject / (f)ix / (s)kip / (q)uit: ").strip().lower()
            if choice == "q":
                print("Stopping — remaining items stay in the queue for next time.")
                return
            if choice == "s":
                break
            if choice == "a":
                review_queue.record_decision(args.split, item.index, "accept", reviewer=args.reviewer)
                break
            if choice == "r":
                review_queue.record_decision(args.split, item.index, "reject", reviewer=args.reviewer)
                break
            if choice == "f":
                path_str = input("  path to corrected-record JSON file: ")
                corrected = _load_corrected_record(path_str)
                if corrected is None:
                    continue  # re-prompt the a/r/f/s/q choice
                review_queue.record_decision(args.split, item.index, "fix", corrected_record=corrected, reviewer=args.reviewer)
                break
            print("  please enter a, r, f, s, or q")

    print("\nDone with this session.")
    print(json.dumps(review_queue.review_stats(args.split), indent=2))


if __name__ == "__main__":
    main()
