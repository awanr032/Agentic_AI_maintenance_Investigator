"""Runs the Drafting Agent over Pattern Agent findings and/or individual
extraction results (design-spec.md §5.4, repo layout §8). Run from the repo
root as:

    .venv\\Scripts\\python.exe -m scripts.run_drafting --patterns
    .venv\\Scripts\\python.exe -m scripts.run_drafting --split gold --sample-size 5

Deliberately doesn't persist drafted text anywhere — design-spec.md §6's
store layout lists only extracted/, validated/, scores/, patterns/, no
"drafts" collection. This script exists to support §5.4's actual acceptance
criteria directly: "manually spot-check a sample of drafted paragraphs
against their source structured input" — it prints each draft next to its
check_faithfulness() advisory flags so that spot-check is easy to do, not
to produce a new persisted artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store  # noqa: E402
from src.drafting_agent import check_faithfulness, draft  # noqa: E402


def _report_one(label: str, finding: dict) -> None:
    text = draft(finding)
    check = check_faithfulness(text, finding)
    print(f"- [{label}] {text}")
    if check["numeric_issues"] or check["groundedness_score"] < 0.5:
        print(f"    [faithfulness flag] groundedness={check['groundedness_score']:.2f}")
        if check["numeric_issues"]:
            print(f"    numeric_issues: {check['numeric_issues']}")
        if check["unmatched_content_words"]:
            print(f"    unmatched_content_words: {check['unmatched_content_words']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft prose from Pattern Agent findings and/or individual extraction results.")
    parser.add_argument("--patterns", action="store_true", help="Draft every finding in patterns/findings.json.")
    parser.add_argument("--split", choices=["gold", "silver"], help="Also draft a sample of individual extraction results from this split.")
    parser.add_argument("--sample-size", type=int, default=5, help="How many per-order extraction results to draft, if --split is given.")
    args = parser.parse_args()

    if not args.patterns and not args.split:
        parser.error("pass --patterns and/or --split")

    if args.patterns:
        findings = store.read_doc("patterns/findings") or []
        print(f"drafting {len(findings)} pattern finding(s) from patterns/findings.json...")
        for f in findings:
            _report_one(f["asset_type"], f)
        print()

    if args.split:
        extracted = dict(store.iter_records("extracted", args.split))
        sample_indices = sorted(extracted)[: args.sample_size]
        print(f"drafting {len(sample_indices)} per-order result(s) from extracted/{args.split}.jsonl...")
        for idx in sample_indices:
            _report_one(f"{args.split}#{idx}", extracted[idx])


if __name__ == "__main__":
    main()
