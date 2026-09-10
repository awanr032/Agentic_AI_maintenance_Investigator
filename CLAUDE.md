# Agentic AI Maintenance Investigator

## What this project is

A four-agent pipeline that extracts structured entities/relations from short maintenance
work order (MWO) texts, validates the extraction, finds recurring failure patterns across
many work orders, and drafts readable findings — all with an evidence trail back to source
text. Built as a portfolio/practice project using the public, MIT-licensed MaintIE dataset
(github.com/nlp-tlp/maintie) as both training/eval data and the gold-standard benchmark.

**Full design spec**: see `docs/design-spec.md` — read this before writing any agent code.
It defines the exact input/output contract for every agent, the repo structure, and five
open questions that should be resolved deliberately, not guessed at silently.

## Current scope (v1) — build in this order

1. `src/schema.py` — load MaintIE's entity/relation taxonomy from `data/scheme.json` (a
   nested tree with `entity` and `relation` top-level keys; each node has a `fullname`
   field giving the exact dotted-path string used in the data's `"type"` field, e.g.
   `PhysicalObject/EmittingObject/ElectricCoolingObject`). Walk it recursively and flatten
   into usable label lists — don't hardcode the 224 leaf types.
2. `src/extraction_agent.py` — per-text extraction agent (§5.1 of the spec).
3. `src/validation_agent.py` — reviews extraction output (§5.2).
4. `src/scoring.py` — deterministic precision/recall vs. gold labels. NOT an LLM call.
5. `src/pattern_agent.py` — cross-text pattern finding (§5.3), uses `src/tools.py`.
6. `src/drafting_agent.py` — bounded prose from structured findings (§5.4). Hard constraint:
   must never introduce a fact not present in its input. This is the most important
   correctness rule in the whole project — treat violations as bugs, not style issues.

Query Agent and Recommendation Agent are explicitly phase 2 — do not build them yet.

## Model / provider choices

- Primary build target: **DeepSeek** (deepseek-chat / deepseek-reasoner), OpenAI-compatible
  API surface. Cheap enough to iterate freely — see cost estimates in the design spec.
- Comparison run: same pipeline against **Claude** (Haiku 4.5 for Extraction/Validation/
  Drafting, Sonnet 5 for Pattern Agent) once the DeepSeek version works, for the case study.
- Keep each agent's API call isolated behind a small function so swapping providers is a
  one-line change, not a rewrite. Don't hardcode a provider inside agent logic.
- DeepSeek JSON mode requires the word "json" somewhere in the prompt or the API rejects
  the request — handle this in every agent that uses `response_format: json_object`.
- DeepSeek tool-calling strict mode is beta and needs a separate beta base URL — validate
  the Pattern Agent's tool call output carefully, don't assume schema enforcement is solid.

## Orchestration

Plain sequential Python. No LangGraph/CrewAI/framework for v1 — four sequential batch
stages don't need one. Each stage runs as a batch over the whole dataset before the next
starts (extract everything → validate everything → score → find patterns → draft).

## Data

- `data/gold_release.json` — 1,076 double-annotated texts. Ground truth for scoring ONLY.
  Never train/mine patterns on this — it's the answer key.
- `data/silver_release.json` — 7,000 weakly-labelled texts. Use for pattern-mining volume,
  after running our own Extraction + Validation agents on it (don't reuse MaintIE's own
  silver labels — see open question 4 in the design spec for why).
- Both files need to be fetched from the MaintIE repo — not included in this scaffold.

## Cost / safety guardrails (non-negotiable)

- API keys live in `.env`, which is gitignored. Never hardcode a key, never commit one.
- Before running a full batch over all ~8,076 texts, test on a 10-50 text sample first.
- Check account balance/spend limits before large batch runs — see `docs/design-spec.md`
  cost section for expected order of magnitude (a few dollars on DeepSeek, ~$15-20 on
  Claude, per full pass).
- Use the Batch/async API where the provider supports it for the Extraction and Validation
  stages specifically — that's the high-volume part of the pipeline.

## Code conventions

- Python, type hints on every agent function signature (see §5 of the spec for exact
  signatures expected).
- **All data reading and writing goes through `src/store.py`. No exceptions.** No agent,
  script, or scoring module may call `open()`, `json.load()`, `pathlib`, or otherwise touch
  the filesystem for pipeline data directly. Reason: on AWS, local filesystems are ephemeral
  (Lambda/Fargate wipe them between runs), so this all moves to S3 later. If every read/write
  is behind `store.py`, that's a one-file change. If file handling is scattered across agent
  files, it's a painful retrofit. This rule is about future portability, not style — treat
  a direct file access in an agent file as a bug.
- **Every run must be resumable.** Key every stored result by its source text ID (index in
  the source dataset). Before processing an item, `run_extraction.py` and `run_validation.py`
  must check the store and skip anything already completed. Reason: a batch of ~8,076 LLM
  calls will eventually die partway through (crash, timeout, rate limit, spot interruption
  on AWS). Without this, a failure at item 6,000 means re-running and re-paying for all
  6,000. This is ~10 lines now and a genuine headache to retrofit later.
- Local flat-file storage for v1 (`src/store.py`) — no database, no AWS. Keep the storage
  interface simple enough to swap later without touching agent code.
- Every agent function should be independently testable with a fixture input/output pair,
  since the whole point of this design is that each stage is separately checkable.
