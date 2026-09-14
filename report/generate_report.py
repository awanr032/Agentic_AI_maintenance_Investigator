"""Builds report.html from drafted Pattern Agent findings (design-spec.md §8,
the final deliverable named in the repo layout). Run from the repo root as:

    .venv\\Scripts\\python.exe -m report.generate_report --split silver

Drafted prose is cached per finding via a new report_drafts/{split}.jsonl
collection (uses store.py's existing generic collection support — no
changes needed there), keyed by the finding's index in patterns/findings.json.
Regenerating the report re-reads the cache instead of re-calling the
Drafting Agent every time — deliberate, for two reasons found firsthand in
this project: (1) it avoids repeat API cost on every report rebuild, and
(2) a live drafting call can hit a transient provider-side timeout (observed
directly in this project's own session), which a report-generation step
has no business being fragile to. Delete a specific report_drafts/{split}.jsonl
line (or the whole file) to force a fresh draft for that finding.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import review_queue, store  # noqa: E402
from src.drafting_agent import check_faithfulness, draft  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "report.html"


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _get_or_draft(split: str, index: int, finding: dict[str, Any]) -> str:
    """Cached drafted prose for one finding — see module docstring for why
    this is cached rather than always calling draft() live."""
    cached = store.get_record("report_drafts", split, index)
    if cached is not None:
        return cached["text"]
    text = draft(finding)
    store.put_record("report_drafts", split, index, {"text": text})
    return text


CSS = """
:root {
  --ink: #1f2937; --muted: #6b7280; --bg: #f7f5f0; --card: #ffffff;
  --border: #e5e0d5; --accent: #9a5b28; --accent-bg: #f4e9dd; --warn: #b45309;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
}
.container { max-width: 780px; margin: 0 auto; padding: 40px 24px 80px; }
header h1 { font-size: 1.6rem; margin: 0 0 6px; }
header .subtitle { color: var(--muted); margin: 0 0 32px; font-size: 0.92rem; }
.stats { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 40px; }
.stat {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 20px; min-width: 130px; flex: 1;
}
.stat-value { display: block; font-size: 1.5rem; font-weight: 600; color: var(--accent); }
.stat-label { display: block; font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
h2 { font-size: 1.15rem; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 40px 0 20px; }
.finding {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 22px; margin-bottom: 14px;
}
.finding-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.count {
  background: var(--accent-bg); color: var(--accent); font-weight: 700; font-size: 0.85rem;
  padding: 3px 10px; border-radius: 999px;
}
.asset-type code { font-size: 0.78rem; color: var(--muted); }
.drafted { margin: 0; }
.drafted .flag { color: var(--warn); font-weight: 700; margin-left: 4px; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--accent); font-size: 0.85rem; }
.examples { margin: 8px 0 6px; padding-left: 20px; font-size: 0.88rem; color: var(--muted); }
.types { font-size: 0.82rem; color: var(--muted); margin: 4px 0 0; }
.types code { background: var(--accent-bg); padding: 1px 6px; border-radius: 4px; }
footer { margin-top: 48px; font-size: 0.82rem; color: var(--muted); border-top: 1px solid var(--border); padding-top: 20px; }
code { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }
"""


def build_report(split: str = "silver") -> str:
    findings: list[dict[str, Any]] = store.read_doc("patterns/findings") or []
    findings_sorted = sorted(enumerate(findings), key=lambda pair: -pair[1]["occurrence_count"])

    finding_cards: list[str] = []
    for idx, finding in findings_sorted:
        text = _get_or_draft(split, idx, finding)
        check = check_faithfulness(text, finding)
        flag = "" if check["groundedness_score"] >= 0.5 and not check["numeric_issues"] else ' <span class="flag">⚠ spot-check</span>'
        examples_html = "".join(f"<li>{_html_escape(t)}</li>" for t in finding["example_source_texts"])
        types_html = ", ".join(f"<code>{_html_escape(t)}</code>" for t in finding["supporting_entity_types"])
        finding_cards.append(
            f'<article class="finding">'
            f'<div class="finding-header"><span class="count">{finding["occurrence_count"]}×</span>'
            f'<span class="asset-type"><code>{_html_escape(finding["asset_type"])}</code></span></div>'
            f'<p class="drafted">{_html_escape(text)}{flag}</p>'
            f"<details><summary>Evidence</summary>"
            f'<ul class="examples">{examples_html}</ul>'
            f'<p class="types">Supporting entity types: {types_html}</p>'
            f"</details></article>"
        )

    extracted_count = len(dict(store.iter_records("extracted", split)))
    validated = dict(store.iter_records("validated", split))
    flagged_count = sum(1 for v in validated.values() if v.get("status") == "flagged")
    flagged_pct = (flagged_count / len(validated) * 100) if validated else 0.0

    accuracy_stats = ""
    eval_report = store.read_doc("scores/eval_report")
    if eval_report:
        e = eval_report["all_extracted"]["entity_exact_match"]
        r = eval_report["all_extracted"]["relation_exact_match"]
        n = eval_report["sample_indices_count"]
        accuracy_stats = (
            f'<div class="stat"><span class="stat-value">{e["f1"] * 100:.0f}%</span>'
            f'<span class="stat-label">Entity F1 vs. gold (n={n})</span></div>'
            f'<div class="stat"><span class="stat-value">{r["f1"] * 100:.0f}%</span>'
            f'<span class="stat-label">Relation F1 vs. gold</span></div>'
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maintenance Pattern Findings</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Agentic AI Maintenance Investigator</h1>
    <p class="subtitle">Recurring failure patterns mined from {extracted_count} validated maintenance work orders
      ({_html_escape(split)} corpus) · generated {generated_at}</p>
  </header>

  <section class="stats">
    <div class="stat"><span class="stat-value">{extracted_count}</span><span class="stat-label">Texts extracted</span></div>
    <div class="stat"><span class="stat-value">{flagged_pct:.0f}%</span><span class="stat-label">Flagged for review</span></div>
    <div class="stat"><span class="stat-value">{len(findings)}</span><span class="stat-label">Recurring patterns found</span></div>
    {accuracy_stats}
  </section>

  <section class="findings">
    <h2>Recurring Failure Patterns</h2>
    {"".join(finding_cards) if finding_cards else "<p>No patterns found — run scripts/run_patterns.py first.</p>"}
  </section>

  <footer>
    <p>Every finding above is cross-checked against a real tool-call result before being included here —
    occurrence counts and example texts are never invented. A ⚠ marks a drafted paragraph that failed an
    automated faithfulness check (see src/drafting_agent.py's check_faithfulness()) and should be
    spot-checked by hand against its source data before being relied on.</p>
    <p>This report reflects a portfolio/practice project built on the public, MIT-licensed MaintIE dataset
    (github.com/nlp-tlp/maintie) — it is not a production accuracy claim. See docs/design-spec.md for
    methodology, known limitations, and the human-in-the-loop review process (src/review_queue.py) this
    data has (or has not yet) been through.</p>
  </footer>
</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report.html from drafted Pattern Agent findings.")
    parser.add_argument("--split", choices=["gold", "silver"], default="silver")
    args = parser.parse_args()

    html = build_report(args.split)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
