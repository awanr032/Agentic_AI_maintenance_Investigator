"""Ask the Query Agent a free-form question about one split's validated
data. Run from the repo root as:

    .venv\\Scripts\\python.exe -m scripts.run_query "What are the most common pump failures?" --split silver

Not a batch/resumable script like run_extraction.py — one question in,
one answer out, same one-shot spirit as run_patterns.py's single
investigation pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query_agent import answer_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Query Agent a question about one split's validated data.")
    parser.add_argument("question")
    parser.add_argument("--split", choices=["gold", "silver"], default="silver")
    args = parser.parse_args()

    result = answer_question(args.question, split=args.split)
    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
    print()
    print(f"grounded: {result.grounded}" + ("" if result.grounded else "  <-- WARNING: unverified number(s) in answer"))
    print(f"supporting facts observed: {len(result.supporting_facts)}")


if __name__ == "__main__":
    main()
