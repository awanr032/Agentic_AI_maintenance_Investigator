# Agentic AI Maintenance Investigator

Portfolio project: a five-agent pipeline that turns short, messy maintenance work order
(MWO) text into structured, evidence-backed data — and lets you ask it questions about
the failure patterns hiding in that data — benchmarked against the public MaintIE dataset.

**The five agents:**

| Agent | What it does | Evidence discipline |
|---|---|---|
| **Extraction** | Per-text: raw MWO text → structured entities/relations | Every span must appear verbatim in the source text |
| **Validation** | Deterministic structural checks on each extraction (taxonomy membership, dangling relations, overlaps) | No LLM judgment call trusted without a rule-based check behind it |
| **Pattern** | Cross-text: finds recurring failure signatures using a real multi-turn tool-calling loop | Every finding verified against actual tool output before being reported — a fabricated count or example is dropped, not trusted on the model's word |
| **Drafting** | Turns one structured finding into a short, bounded prose paragraph | Hard constraint: never introduces a fact not present in its input |
| **Query** | Free-form question answering over the same trusted data Pattern Agent reads | Every number in an answer must trace back to an actual observed tool result |

None of these agents talk to each other directly — they coordinate through a shared,
auditable data store (`src/store.py`), each stage reading what the previous one wrote.
See `docs/design-spec.md` §4 for the full architecture, and §5 for each agent's exact
input/output contract.

## Why this exists

Real maintenance systems (CMMS software — Maximo, SAP PM, Fiix, UpKeep) accumulate years
of free-text work order descriptions that are never systematically analyzed. This project
demonstrates turning that unstructured text into structured, queryable data — without
requiring technicians to change how they write work orders — plus an audit trail (every
claim traceable to real evidence) for anyone who needs to trust the output.

## Results

Extraction accuracy was benchmarked against MaintIE's own published academic baseline
(SpERT, LREC-COLING 2024) and iterated through 9+ documented find-fix-verify cycles (see
git history) — taxonomy hallucinations, systematic type confusions, and a major hidden
reasoning-token cost bug were all found by hand-investigating real mismatches, fixed with
targeted evidenced prompt rules, and verified by re-running. Full numbers in
`scores/eval_report.json` after running the pipeline (see Setup below).

A human-in-the-loop review layer (`src/review_queue.py`, design-spec.md §8.1) catches the
one class of error automated checks structurally can't: cases where Extraction and
Validation both agreed and were both wrong.

## Deployed

Three of the five agents run live on AWS Lambda, provisioned via Terraform
(`infra/terraform/`), each with its own separately-scoped IAM role (least privilege: no
role has a permission its agent doesn't specifically need):

- **Extraction Agent** — IAM role scoped to exactly two permissions, the DeepSeek API key
  in SSM Parameter Store (never a plaintext env var), and guardrails (input length cap,
  generic error responses) appropriate for a metered API behind a public endpoint.
- **Query Agent** and **Pattern Agent** — read/write the validated corpus via S3
  (`src/store.py`'s S3 backend, populated via `scripts/migrate_store_to_s3.py`) instead of
  local disk. Both surfaced real bugs on their first live deployment, each found and fixed
  the same way as every bug in this project — evidence first, root cause, targeted fix,
  re-verified:
  - **Sequential S3 refetch timeout** (Query Agent): `tools.py` scans the whole split on
    every tool call; the one-object-per-record S3 layout turned that into ~500 sequential
    network round-trips, timing out the Lambda at 30s. Fixed by parallelizing the reads
    (verified: 500 records in ~4s) and adding a per-invocation cache so multiple tool calls
    in one run share a single fetch — applied proactively to Pattern Agent too, since its
    whole job is investigating multiple asset types per run.
  - **Verification loophole** (Pattern Agent): a live run returned a finding with
    `occurrence_count: 1` and **no example texts at all** — checked against the real data
    directly and confirmed fabricated. Root cause: the verification check's subset test
    (`set(finding.examples) <= set(record.examples)`) is vacuously true when the
    finding's examples are empty. Fixed by requiring non-empty examples, justified
    structurally (a real record can never have zero examples).
  - **Discard-context retry** (both agents): when the model's JSON response had a prose
    preamble the parser didn't expect, the retry logic threw away the entire conversation —
    including every tool result already gathered — and asked again with "no more tool
    calls," producing an empty result despite having real data moments earlier. Fixed by
    validating and correcting within the same conversation instead of restarting it.

All three verified end-to-end via direct invocation, matching local output exactly. See
`docs/design-spec.md` §7 for the full deployment writeup, including a known open issue
shared by all three (each Function URL is currently blocked by what looks like a
new-AWS-account anti-abuse restriction — worked around by calling them directly via
CLI/console instead).

Validation and Drafting Agents are built and tested locally but not yet deployed — both
would reuse the same S3-backed `store.py` pattern the other three already prove out.

## Setup

```bash
cp .env.example .env   # then fill in your API key(s)
pip install -r requirements.txt
python scripts/fetch_data.py   # clones MaintIE's gold/silver data into data/
```

## Running the pipeline

Each stage is a separate batch script, run in order (later stages depend on earlier ones'
output — see the Architecture section above):

```bash
python -m scripts.run_extraction --split silver
python -m scripts.run_validation --split silver
python -m scripts.run_scoring          # deterministic, no LLM calls
python -m scripts.run_patterns --split silver
python -m scripts.run_drafting --split silver

# Ask the Query Agent a one-off question, anytime after extraction+validation have run:
python -m scripts.run_query "What are the most common failures for pumps?" --split silver
```

`scripts/review.py` is the interactive human review CLI (design-spec.md §8.1).
`report/generate_report.py` builds a static HTML report from drafted findings.

## Data attribution

This project fetches data from [MaintIE](https://github.com/nlp-tlp/maintie) (MIT licensed).
The dataset itself is not vendored into this repo — `scripts/fetch_data.py` pulls it at setup
time. Credit to the MaintIE authors for the gold/silver corpora and taxonomy this project is
built on top of and benchmarked against.
