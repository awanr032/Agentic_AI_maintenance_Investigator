"""Ad-hoc stress test for the Query Agent: 30 real questions spanning
in-scope lookups, ambiguous natural-language-to-taxonomy mapping, asset
types that don't exist in this data, questions demanding precision the
data can't support, and fully out-of-scope questions. Not a formal
accuracy benchmark (there's no fixed ground-truth answer key for free-text
Q&A the way scoring.py has for extraction) — this is a manual-inspection
harness, same spirit as the early hand-investigation passes done for the
other four agents before trusting them.

Run from repo root: .venv\\Scripts\\python.exe -m scripts.eval_query_agent
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query_agent import answer_question  # noqa: E402

QUESTIONS = [
    # --- in-scope, real domain terms (ambiguous -> taxonomy mapping needed) ---
    "What are the most common failures for pumps?",
    "What issues are typically reported for hydraulic hoses?",
    "What are the most common problems with bearings?",
    "What failures are associated with conveyor belts?",
    "What is the most common issue with tyres?",
    "What kind of failures occur most often on engines?",
    "What problems are reported for filters?",
    "What issues happen most with valves?",
    "What are common seal failures?",
    "What is the most frequent failure mode for motors?",
    "What issues are seen with sensors?",
    "What are common problems with cables?",
    "What failures occur on radiators?",
    "What issues are reported for switches?",
    "What problems occur with fans?",
    # --- aggregate / ranking ---
    "What are the top 5 most common asset types in the data?",
    "How many distinct asset types appear in the top 20 most common list?",
    "What is the single most frequently mentioned asset type?",
    "List the three least common asset types among the top 20.",
    "Compare the failure counts between pumps and engines - which has more recorded issues?",
    # --- edge case: asset types that almost certainly don't exist in this data ---
    "What are the most common failures for spaceship thrusters?",
    "What issues occur with quantum flux capacitors?",
    "What is the failure rate of the warp core?",
    # --- adversarial: demanding precision the data can't support ---
    "Exactly how many hours of downtime have pumps caused this year?",
    "What is the average cost of repairing a leaking pump?",
    "How many times will the conveyor belt fail next month?",
    "What is the mean time between failures for bearings, in days?",
    # --- fully out of scope ---
    "What is the weather like today?",
    "Who is the current president of the United States?",
    "Can you write me a poem about maintenance?",
]


def main() -> None:
    total_turns_proxy = 0
    grounded_count = 0
    results = []
    start = time.time()

    for i, q in enumerate(QUESTIONS, 1):
        try:
            result = answer_question(q, split="silver")
            status = "OK"
            answer = result.answer
            grounded = result.grounded
            facts = len(result.supporting_facts)
        except Exception as e:  # noqa: BLE001 -- this is a stress test, we want to see every failure
            status = "ERROR"
            answer = f"{type(e).__name__}: {e}"
            grounded = None
            facts = 0

        if grounded:
            grounded_count += 1
        total_turns_proxy += facts
        results.append((i, q, status, answer, grounded, facts))
        print(f"[{i}/{len(QUESTIONS)}] {status} grounded={grounded} facts={facts}")
        print(f"    Q: {q}")
        print(f"    A: {answer}")
        print()

    elapsed = time.time() - start
    print("=" * 70)
    print(f"{len(QUESTIONS)} questions run in {elapsed:.0f}s")
    print(f"grounded: {grounded_count}/{len(QUESTIONS)}")
    print(f"errors: {sum(1 for r in results if r[2] == 'ERROR')}")
    print(f"total observed tool-result records across all questions: {total_turns_proxy}")


if __name__ == "__main__":
    main()
