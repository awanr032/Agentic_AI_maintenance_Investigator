# Agentic AI Maintenance Investigator — System Design Spec

Status: draft v1
Purpose: design-first spec to hand to Claude Code for implementation. No code yet — this defines contracts, responsibilities, and acceptance criteria.

---

## 1. Goal

Build a four-agent pipeline (revised from the original two-agent version — see §4 for rationale):
1. **Extraction Agent** — extracts structured entities/relations from short maintenance work order (MWO) texts, evaluated against a published gold-standard benchmark (MaintIE).
2. **Validation/QA Agent** — reviews extraction output for schema and logical errors before it's trusted downstream. Mirrors MaintIE's own double-annotation methodology (the gold corpus itself was double-checked by two human experts).
3. **Pattern Agent** — aggregates validated extractions across many work orders to surface recurring failure patterns per asset — the cross-document "investigation" layer the raw dataset doesn't provide.
4. **Drafting Agent** — turns structured findings (from either the Extraction Agent for a single work order, or the Pattern Agent for a recurring signature) into readable prose. Strictly bounded: it may only restate facts present in the structured input it's given — no inferred causes, no recommendations, no elaboration beyond the data.

Success is measured three ways: extraction quality (precision/recall vs. gold labels), usefulness of surfaced patterns (qualitative — do the patterns look like real findings a maintenance engineer would care about), and drafted-report faithfulness (qualitative — does every claim in the drafted text trace back to a specific entity/relation/pattern it was given, with no invented facts).

Two further agents are deliberately **out of scope for v1** and listed as phase 2 in §10: a Query Agent (ad-hoc questions against extracted data) and a Recommendation Agent (prescriptive suggestions from patterns). Both are lower priority than getting the four core agents right, and the Recommendation Agent in particular carries more trust/liability weight (see tender-project evidence-trail discussion) than is worth taking on before the extraction and pattern layers are proven.

## 2. Non-goals (v1)

- No AWS, no S3, no MCP server, no multi-tenant deployment. All local.
- No fine-tuning. Prompt-based extraction only for v1; fine-tuning is a stated stretch goal, not required.
- No UI framework decisions yet beyond "simple enough to demo" — a CLI + a single HTML report page is enough.
- No production error handling / retries / rate-limit backoff sophistication — note where it would go, don't build it yet.

## 3. Data source

Repo: `github.com/nlp-tlp/maintie` (MIT licensed).

Relevant files after cloning:
- `data/gold_release.json` — 1,076 expert double-annotated texts (ground truth for evaluation) — verified
- `data/silver_release.json` — 7,000 weakly-labelled texts (larger volume, use for the pattern-mining agent, not for scoring accuracy) — verified
- `data/scheme.json` — the full 224-leaf entity/relation taxonomy as a nested tree (`entity`
  and `relation` top-level keys, each node has a `fullname` field giving the exact dotted
  path used in `"type"` fields below) — verified; load this programmatically, not `SCHEME.md`

### 3.1 Source data format (verified against the actual repo — corrected from an earlier draft assumption)

```json
{
  "text": "<id> air conditioner thermostat not working",
  "tokens": ["<id>", "air", "conditioner", "thermostat", "not", "working"],
  "entities": [
    {"start": 1, "end": 3, "type": "PhysicalObject/EmittingObject/ElectricCoolingObject"},
    {"start": 3, "end": 4, "type": "PhysicalObject/SensingObject/TemperatureSensingObject"},
    {"start": 4, "end": 6, "type": "State/UndesirableState/FailedState"}
  ],
  "relations": [
    {"head": 0, "tail": 1, "type": "hasPart"},
    {"head": 2, "tail": 1, "type": "hasParticipant/hasPatient"}
  ]
}
```

Note the real example also shows the `<id>` sanitization placeholder mentioned in §3 discussion — worth handling explicitly in the Extraction Agent prompt (skip/ignore placeholder tokens rather than trying to extract entities from them).

Top-level entity classes (5): `Activity`, `PhysicalObject`, `Process`, `Property`, `State`
Relation types (6): `contains`, `hasPart`, `hasParticipant/hasAgent`, `hasParticipant/hasPatient`, `hasProperty`, `isA`

Leaf entity types are hierarchical strings (e.g. `PhysicalObject/DrivingObject/CombustionEngine`) — 224 total. **Design decision: load the full leaf taxonomy from `SCHEME.md` (or a derived JSON, if we pre-parse it once) and inject the relevant subset into the extraction prompt as the allowed label set, rather than hand-authoring the taxonomy.** This keeps the system correct if the schema is ever updated and avoids transcription errors.

Note: `start`/`end` in the source are **token indices**, not character offsets. Our citation trail needs to map back to token spans → we keep the same tokenization the agent used, or re-tokenize consistently, so extracted spans are comparable to gold spans for scoring.

## 4. System architecture

```
gold_release.json / silver_release.json
            │
            ▼
   ┌─────────────────────┐
   │ Extraction Agent     │   input: 1 MWO text + allowed label set
   │ (per-text, LLM call) │   output: entities[], relations[] (see §5.1)
   └──────────┬───────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Validation/QA Agent  │   input: one Extraction Agent output
   │ (per-text, LLM call) │   output: pass/flagged + corrected confidence (see §5.2)
   └──────────┬───────────┘
              │  (validated results persisted to local store, §6)
              ▼
   ┌─────────────────────┐
   │ Scoring module       │   input: validated extractions vs gold (gold subset only)
   │ (deterministic code, │   output: precision/recall/F1 per entity & relation type
   │  not an LLM call)    │
   └──────────┬───────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Pattern Agent        │   input: validated entities/relations across N texts
   │ (batch, LLM + tool)  │   output: recurring failure signatures (see §5.3)
   └──────────┬───────────┘
              │  calls
              ▼
   ┌─────────────────────┐
   │ get_failure_history() │  deterministic function/tool, queries local store
   └─────────────────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Drafting Agent        │   input: a Pattern Agent finding (or a single
   │ (per-finding, LLM call)│   Extraction Agent result for a per-order report)
   └──────────┬─────────────┘   output: bounded prose, no invented facts (see §5.4)
              │
              ▼
      report.html / CLI output
```

**Why four agents, not two.** The original v1 draft had just Extraction → Pattern. Two additions came out of working through the design in conversation: (1) extraction errors compound into pattern-mining errors, so a Validation/QA Agent that mirrors MaintIE's own human double-annotation process is worth adding before trusting anything downstream; (2) raw structured JSON isn't itself the deliverable a maintenance engineer reads — a Drafting Agent that's *strictly bounded to restating given facts* turns findings into prose without reopening the hallucination risk the structured/citation approach was built to avoid. Each agent has one narrow, separately-checkable job — this is deliberate: the alternative (one agent that extracts, judges, aggregates, and writes) is harder to debug and harder to score.

Orchestration for v1: **plain sequential Python, no framework.** Each stage runs as a batch step over the full dataset before the next stage starts (extract everything, then validate everything, then score, then find patterns, then draft). No graph, no state machine, no framework needed at this scale — see conversation rationale: framework overhead isn't justified until there's real branching/looping/long-running state, which four sequential batch steps don't require.

## 5. Agent contracts

### 5.1 Extraction Agent

**Input:**
```python
def extract(text: str, allowed_entity_types: list[str], allowed_relation_types: list[str]) -> ExtractionResult
```

**Output schema:**
```json
{
  "text": "change out engine",
  "entities": [
    {"span_text": "change out", "start_token": 0, "end_token": 2, "type": "Activity/MaintenanceActivity/Replace", "confidence": "high|medium|low"},
    {"span_text": "engine", "start_token": 2, "end_token": 3, "type": "PhysicalObject/DrivingObject/CombustionEngine", "confidence": "high"}
  ],
  "relations": [
    {"head_span": "change out", "tail_span": "engine", "type": "hasParticipant/hasPatient"}
  ]
}
```

*(Corrected 2026-09-11: this example originally had head_span/tail_span reversed —
`hasParticipant/hasPatient` runs event→participant, e.g. "change out" (the activity)
→ "engine" (what it acts on), never the other way. This was inconsistent with both the
§3.1 example above and the actual gold corpus, where that direction holds across all
1,076 texts with zero exceptions.)*

Design decisions:
- **`confidence` field is required output**, even though MaintIE's gold data doesn't have one — this is what makes the extraction defensible/citable later (matches the evidence-trail principle from the tender project). Low-confidence extractions should be flagged for review in the report, not silently included in pattern-mining.
- Prompt strategy: few-shot, using 3-5 examples pulled directly from `gold_release.json` (not hand-written), plus the allowed label list for that call. Given texts average 5.4 tokens, keep the prompt tight — don't over-engineer with retrieval for this step (see §7, RAG is out of scope for extraction).
- Batch these calls; MWOs are short and independent, so this parallelizes trivially.

**Acceptance criteria:** on a held-out sample of ~100 gold texts, entity-type F1 and relation-type F1 are computed and reported. No hard target pre-set — the number itself, reported honestly, is the deliverable. (Reference point: MaintIE's own SpERT baseline scores are in `RESULTS.md` in the repo — worth pulling those numbers into the case study for comparison, not as a target to beat.)

### 5.2 Validation/QA Agent

**Input:**
```python
def validate(extraction: ExtractionResult, allowed_entity_types: list[str], allowed_relation_types: list[str]) -> ValidationResult
```

**Output schema:**
```json
{
  "text": "change out engine",
  "status": "pass | flagged",
  "issues": ["relation connects two entities where neither type supports this relation per SCHEME.md"],
  "revised_confidence": {"entity_index": 1, "confidence": "medium"}
}
```

Design decisions:
- This agent does **not** re-extract from scratch — it reviews the Extraction Agent's specific output against the schema (does every type actually exist in the taxonomy? does every relation connect entity types that relation is defined for in `SCHEME.md`? are there duplicate/overlapping spans?) and against its own independent judgment of the source text.
- Distinct from a second full extraction pass: cheaper (smaller prompt, narrower job), and produces a clear pass/flagged signal rather than a second, possibly-conflicting set of extractions to reconcile.
- **Flagged results are excluded from Pattern Agent input and from the accuracy scoring's "trusted" subset**, but are kept in the store and reported separately (e.g. "12% of extractions flagged for review") — this number is itself a useful, honest metric for the case study.

**Acceptance criteria:** on the same held-out gold sample used in §5.1, check whether flagged items correlate with actual extraction errors (compare against gold labels) — this tells you whether the Validation Agent is adding real signal or just noise.

### 5.3 Pattern Agent

**Input:** the full set of **validated** (status: pass) entities/relations across many texts (start with the silver corpus subset — larger volume, more realistic for finding recurring patterns — clearly labelled as weakly-labelled-then-validated data in any output, since silver labels are themselves machine-generated even before our own extraction/validation runs on them — see open question 4).

**Tool it calls:**
```python
def get_failure_history(asset_type: str, top_n: int = 20) -> list[FailureRecord]
```
Deterministic — queries the local extracted-data store (grouping by `PhysicalObject` leaf type co-occurring with `State`/`Process` entities via `hasParticipant`/`hasProperty` relations), returns structured records. The LLM's job is choosing which asset types to investigate and narrating the pattern, not computing the counts.

**Output schema:**
```json
{
  "asset_type": "PhysicalObject/DrivingObject/CombustionEngine",
  "pattern": "recurring bearing wear failures",
  "occurrence_count": 12,
  "example_source_texts": ["...", "...", "..."],
  "supporting_entity_types": ["State/DegradationState/Wear", "PhysicalObject/...Bearing"]
}
```

Every pattern finding must cite back to `example_source_texts` — same evidence-trail principle as the Extraction Agent's confidence field.

**Acceptance criteria:** qualitative — run against the silver corpus, manually review whether the top 5-10 surfaced patterns look like genuine, sensible recurring issues (not a numeric target).

### 5.4 Drafting Agent

**Input:**
```python
def draft(finding: PatternFinding | ExtractionResult) -> str
```

**Output:** a short prose paragraph, e.g.:

> "This work order recorded a bearing wear issue (State: DegradationState/Wear) on a pump, resolved by replacement (Activity: MaintenanceActivity/Replace). Confidence: high."

or for a pattern finding:

> "Bearing wear on combustion engines recurred 12 times in the reviewed records, most recently in: [source text excerpts]. This pattern was not independently verified beyond the source texts shown."

**Hard constraint (the most important line in this spec):** the Drafting Agent's system prompt must explicitly forbid introducing any fact, cause, severity judgment, or recommendation not present in the structured input it receives. This is a prompt-level constraint for v1 — there's no automated check enforcing it yet (see open question 5). Practically: the prompt should include the structured finding, the instruction "only restate what is given below; do not infer causes, severity, or next steps," and 1-2 few-shot examples showing correctly-bounded vs. over-reaching output.

**Acceptance criteria:** qualitative — manually spot-check a sample of drafted paragraphs against their source structured input; every claim in the prose should be traceable to a specific field in the input. Any invented detail is a failure of this agent, full stop — this is not a "mostly right" acceptance bar.

### 5.5 Query Agent (moved up from §10 Phase 2, built post-v1)

**Input:**
```python
def answer_question(question: str, split: str = "silver") -> QueryResult
```

**Output:**
```python
@dataclass
class QueryResult:
    question: str
    answer: str                        # free-text, not a fixed template
    supporting_facts: list[dict]        # every tool result observed while answering
    grounded: bool                      # advisory only, see below
```

Free-form natural-language Q&A over the same trusted data Pattern Agent reads (validated-pass records via `review_queue.effective_records()`), reusing Pattern Agent's exact architecture: a multi-turn tool-calling loop against two deterministic tools (`list_common_asset_types`, `get_failure_history` — both already existed in `src/tools.py` for Pattern Agent; Query Agent added no new counting logic, only a second entry point onto it). Unlike Pattern Agent's fixed structured `PatternFinding` output, an answer here is free text, so there's no field-by-field verification possible — grounding is checked instead by requiring every number the answer states to trace back to an actual observed tool result (excluding numbers echoed from the question itself, e.g. "top 3").

**Why this was safe to move up from Phase 2 ahead of the Recommendation Agent:** it only *describes* data (same as Pattern/Drafting), it never suggests an action — the liability/evidence-trail concern that keeps the Recommendation Agent deliberately last (§10) doesn't apply here.

**Verified via a 30-question stress test** (`scripts/eval_query_agent.py`) spanning: real domain questions requiring ambiguous natural-language-to-taxonomy mapping ("pumps" → `LiquidFlowGeneratingObject`), asset types absent from the data (correctly declined, not fabricated), questions demanding precision the data can't support — cost, downtime hours, forecasts, MTBF (all correctly declined with an explanation of exactly what the tools don't contain), and fully out-of-scope questions (correctly declined). Run three times across ~90 total question-runs: zero fabricated facts found. One real bug was found and fixed this way — see git history ("Fix Query Agent retry discarding gathered tool context") — where a JSON-formatting retry discarded all tool results already gathered instead of preserving them, occasionally producing a misleadingly unhelpful non-answer despite having real data on hand.

**Known limitation, not fixed:** the grounding check occasionally flags a benign number as "unverified" — e.g. the model summarizing several exact counts as an approximate range ("1-3 records each"), or citing a tool call's own parameter (`top_n=50`) rather than a data claim. This is a safe direction for a false positive (over-cautious, not under-cautious) and was left as-is rather than over-tuned.

## 6. Local data store (v1)

No database needed yet. Flat files are sufficient at this scale (~8,000 texts max):
- `extracted/{split}.jsonl` — one extraction result per line, **keyed by source text index**
- `validated/{split}.jsonl` — one validation result per line, keyed by the same index
- `scores/eval_report.json` — output of the scoring module
- `patterns/findings.json` — output of the Pattern Agent
- `corrections/{split}.jsonl` — human review decisions, keyed by source text index (§8.1)
- `report_drafts/{split}.jsonl` — cached Drafting Agent output per pattern finding, keyed by
  the finding's index in `patterns/findings.json`; `report/generate_report.py` reads this
  cache instead of re-calling the Drafting Agent on every report rebuild

Two hard requirements on this layer, both cheap now and painful to retrofit:

**6.1 Everything goes through `store.py`.** No agent, script, or scoring module touches the
filesystem directly for pipeline data. Reason: this is the single change point when moving
to S3/DynamoDB on AWS, where local filesystems are ephemeral. The storage interface is the
stable contract; flat files are an implementation detail behind it.

**6.2 Runs must be resumable.** Every stored result is keyed by source text index, and each
batch script checks the store and skips already-completed items before processing. Reason:
a run of ~8,076 LLM calls will die partway through eventually — crash, timeout, rate limit,
or (on AWS) a spot interruption. Without resumability that means re-running and re-paying
for everything already done.

### 6.3 Known scaling limit (documented, not fixed in v1)

`get_failure_history()` scanning JSONL is fine at ~8,000 records. At a real client's volume
(100k+ work orders) it needs an indexed store — DynamoDB, Athena over S3, or Postgres. This
is a bounded swap, not a redesign, *provided* §6.1 holds. Worth naming in the case study as
a known limit rather than leaving it implicit.

## 7. Where RAG, MCP, and AWS genuinely fit (deferred, not absent)

- **RAG**: not needed for per-text extraction (texts are too short to need retrieval). Would become relevant if the Pattern Agent needs semantic matching across worded-differently failures at larger scale than keyword/structured grouping handles — defer until §5.2's structured grouping proves insufficient.
- **MCP**: wrap `get_failure_history()` as an MCP tool once the local version works — this is a mechanical wrapping step, not a design change, so it's appropriately last.
- **AWS — actually done (Extraction, Query, and Pattern Agents):** `infra/terraform/` provisions three real Lambda deployments, each with its own separately-scoped IAM role (least privilege: no role has a permission its agent doesn't specifically need).
  - **Extraction Agent**: role has exactly two permissions (CloudWatch logs, read one SSM parameter). `infra/lambda_build/lambda_handler.py` wraps `extraction_agent.py`'s `extract()` unchanged — AWS decides *where* the code runs, nothing about extraction itself changed. Guardrails: input length cap, generic error responses so internals aren't leaked to an anonymous caller.
  - **Query Agent** and **Pattern Agent**: role additionally has S3 read/write on a dedicated store bucket (`infra/terraform/store.tf`), populated via `scripts/migrate_store_to_s3.py`. This is what made `store.py`'s S3 backend (§6, below) real rather than theoretical — the whole point of building it. Deploying these two surfaced **three real bugs**, each found and fixed with the same evidence-first discipline as every fix earlier in this document:
    1. **Sequential S3 refetch timeout.** `tools.py`'s `_validated_pass_records()` scans the *entire* split on every tool call (the actual read pattern these agents use, not a rare case) — locally that's one fast file read, but the S3 backend's one-object-per-record layout turned it into ~500 sequential network round-trips, timing out the Query Agent Lambda at its original 30s limit. Fixed by parallelizing the reads in `iter_records()`'s S3 branch (verified: 500 records in ~4s from outside AWS) plus a per-process cache in `tools.py` so multiple tool calls within one run share a single fetch — applied to Pattern Agent proactively before its first deployment, since investigating several asset types per run is its normal behavior, not an edge case, making the redundant-refetch cost guaranteed rather than merely possible.
    2. **Verification loophole.** Pattern Agent's first live run returned a finding with `occurrence_count: 1` and **zero example texts** — checked directly against `tools.get_failure_history()` for that exact asset type and confirmed no real record matched at all. Root cause: `_verify_against_observed()`'s subset check (`set(finding.example_source_texts) <= set(record["example_source_texts"])`) is vacuously true when the finding's examples are empty — a fabricated finding with no cited evidence could pass the one check this agent exists to have. Fixed by requiring non-empty examples, justified structurally: `get_failure_history()` guarantees every real record with `occurrence_count >= 1` has at least one example, so a genuine match can never have zero. Latent since Pattern Agent was first built; never triggered before because every prior real run happened to also copy real examples alongside its counts.
    3. **Discard-context retry.** Immediately after fixing #2, a live run returned `[]` — zero findings — despite working moments earlier with no code changes in between. Reproduced locally 3/3 times and traced turn-by-turn: the model sometimes prefaces its final JSON with a plain sentence (e.g. "I have enough to report the genuinely recurring patterns.\n\n{...}"), which `_parse_model_response()`'s bare `json.loads()` rejected outright. That triggered the top-level retry, which discarded the entire conversation — including 25+ already-investigated asset types — and asked again with "no more tool calls," so the model correctly-but-uselessly reported nothing. This was the exact same class of bug already found and fixed in Query Agent (a discard-and-restart retry instead of correcting within the same conversation) — it had just never been ported back to Pattern Agent, the agent it was originally adapted from. Fixed in both agents: `_parse_model_response()` now extracts JSON starting from the first `{` instead of requiring the whole string to be pure JSON, and both provider loops validate before returning, correcting within the same conversation on failure rather than restarting it.
  - All three agents verified end-to-end via direct `aws lambda invoke`, matching local output exactly.
  - **Known gap, partially resolved**: each Lambda's own Function URL still returns 403 despite a provably correct IAM resource policy — most likely a new-AWS-account anti-abuse restriction on anonymous Function URL access (the account also showed a below-default concurrency limit of 10), and retested unchanged after several more days, ruling out a transient account-age issue. Rather than wait on AWS, `infra/terraform/query_agent_api.tf` puts **API Gateway** (HTTP API) in front of Query Agent instead — a different invocation path entirely (`lambda:InvokeFunction` via API Gateway's own resource policy, not `lambda:InvokeFunctionUrl`), confirmed to sidestep the restriction completely. This is also the standard, production-grade way to expose a Lambda publicly. Query Agent is now genuinely publicly callable with CORS enabled, backing a real demo page anyone can open and use with no terminal or AWS credentials.
    - **A fourth deployment bug surfaced getting there**: the first live API Gateway request failed with `AttributeError: module 'src.tools' has no attribute 'clear_cache'`. Root cause: `tools.py`'s caching fix (added for the WO-12-class bug) had been copied into Pattern Agent's Lambda package (built fresh afterward) but never re-copied into Query Agent's already-existing package — a packaging oversight, not a design flaw. Confirmed by diffing every shared source file against both Lambda packages; only that one file was stale. Fixed by re-copying and rebuilding.
    - Pattern and Extraction Agents' Function URLs remain unresolved; the same API Gateway pattern would apply to either if a public endpoint for them is ever wanted.
  - **A fifth deployment bug, and a real end-user demo**: `infra/web/index.html` + `infra/terraform/web.tf` put an actual webpage — "Ask The Investigator" — on top of the API Gateway endpoint, so a real, non-technical person can ask a question with no terminal or AWS credentials at all. First attempt hosted this page as a Claude artifact instead of a real website, and it failed for every real user with `TypeError: Failed to fetch`, despite the API's CORS configuration being correct. Root cause: published Claude artifacts run in a sandbox that blocks `fetch()` to arbitrary external hosts — a client-side platform restriction no server-side CORS setting can work around. Fixed by hosting the identical page as a genuine S3 static website instead, keeping the whole demo (frontend and backend) under this project's own AWS account. Verified both by direct HTTP testing (page load, CORS preflight headers) and by a real user asking a real question in a real browser.
  - **Not yet done**: Validation and Drafting Agents are still local-only. Both would reuse the exact S3-backed `store.py` and IAM pattern the other three already prove out.

## 8. Suggested repo structure for Claude Code

```
agentic-ai-maintenance-investigator/
  data/                    # cloned MaintIE data (gitignored, fetched by a setup script)
  src/
    schema.py               # loads/parses SCHEME.md into usable label lists
    extraction_agent.py      # §5.1
    validation_agent.py       # §5.2
    scoring.py                 # deterministic precision/recall vs gold (validated subset)
    pattern_agent.py            # §5.3
    drafting_agent.py            # §5.4
    query_agent.py                # §5.5, built post-v1
    tools.py                      # get_failure_history() and any other deterministic tools
    store.py                       # flat-file read/write per §6
    review_queue.py                 # human-in-the-loop review layer, §8.1
  scripts/
    run_extraction.py          # batch-runs extraction over a split
    run_validation.py            # batch-runs validation over extraction output
    run_scoring.py                 # scores validated extractions vs gold
    run_patterns.py                  # runs Pattern Agent over validated data
    run_drafting.py                    # runs Drafting Agent over patterns and/or per-order findings
    run_query.py                         # asks the Query Agent a single question, §5.5
    eval_query_agent.py                    # 30-question stress test, §5.5
    review.py                            # interactive human review CLI, §8.1
  report/
    generate_report.py                  # builds report.html from drafted output
  infra/
    terraform/                          # IAM role, SSM parameter, Lambda function + URL, §7
    lambda_build/lambda_handler.py       # thin Lambda adapter around extraction_agent.py's extract()
  README.md                        # case study writeup (final deliverable)
```

### 8.1 Human-in-the-loop review (added post-v1)

Not part of the original six-module build order — added once real batch runs on gold and
silver both showed a consistent ~20% flagged rate, and manual investigation of flagged
items repeatedly found genuine, fixable extraction problems (the isA/hasPart decomposition
issue documented in git history is the clearest example: found by hand, fixed by prompt
edit, verified by re-running). That manual find→fix→verify cycle is the pattern this layer
is built to support on an ongoing basis, not a one-time investigation.

Design decisions, resolved deliberately:

- **Review targets flagged items, plus a small (default 3%) random sample of PASSED
  items — not every record.** Reviewing everything doesn't scale with corpus size,
  defeating the point of automating extraction at all. But a flag-only review process has
  a real blind spot it can never see on its own: a case where the Extraction Agent and
  Validation Agent both agreed and were both wrong. The random spot-check on passed items
  is the only way to catch that class of error.
- **Three decisions, not a binary approve/reject**: `accept` (the flag was a false alarm —
  keep the original), `reject` (exclude even a passed record — catches the blind spot
  above), `fix` (substitute the reviewer's corrected extraction). A fix's corrected record
  must be the same `{"text", "entities", "relations"}` shape `extract()` itself produces —
  so it slots into downstream consumers with no special-casing.
- **Corrections are additive, not destructive**: `src/review_queue.py`'s
  `effective_records()` merges corrections on top of the original validated-pass set at
  read time; the original `extracted/{split}.jsonl` and `validated/{split}.jsonl` files are
  never rewritten. `src/tools.py` (and therefore the Pattern Agent) reads exclusively
  through `effective_records()`, so a correction takes effect without touching pipeline
  contracts elsewhere.
- **The review queue is meant to shrink over time, not stay a fixed ongoing cost.** The
  intended longer-term loop (not yet built — see below) is: recurring corrections get fed
  back into the Extraction Agent's few-shot examples or prompt rules, the same way this
  project's own manual investigations already did by hand three times. If the flagged rate
  doesn't trend down as corrections accumulate, that is itself a signal the feedback loop
  isn't working — not that more reviewers are needed.
- **Not built yet, deliberately deferred**: automatic feedback of "fix" corrections into
  `extraction_agent.py`'s few-shot examples (would need a policy for how many/which
  corrected examples to include without bloating the prompt); routing based on
  cross-provider disagreement (DeepSeek vs. Claude both extracting the same text, only
  human-reviewing where they disagree) rather than a flat flagged/spot-check queue — a
  smarter, likely smaller queue, but needs both providers actually run against the same
  data first, which hasn't happened yet in this project.

## 9. Open questions to resolve before/during implementation

1. Which LLM calls the Extraction Agent — plain Claude API with structured output, or something requiring stricter JSON schema enforcement? (Recommend: Claude API with a strict system prompt + output validation/retry on malformed JSON, no extra framework.)
2. How to handle multi-token entity spans and nested/overlapping entities in scoring — need exact-match vs. partial-match F1 defined before writing `scoring.py`, since this materially changes the reported number.
3. Confidence calibration: is "high/medium/low" from the model itself trustworthy, or should low-confidence be inferred separately (e.g. self-consistency across repeated calls)? Fine to start naive and note as a limitation.
4. Silver corpus is weakly labelled — for the Pattern Agent, decide whether to extract independently with our own Extraction Agent (consistent methodology, more LLM calls) or reuse MaintIE's provided silver labels (faster, but inconsistent with our own extraction's error profile). Recommend: use our own agent's extractions for consistency, clearly documented as a choice in the case study.
5. Drafting Agent faithfulness is currently enforced only via prompt instruction (§5.4), which is not a hard guarantee. A stretch improvement: a lightweight automated check that every noun phrase/fact in the drafted text can be matched back to a field in the structured input, flagging drafts that introduce unmatched content for human review. Not required for v1, but worth deciding whether to build before presenting drafted output as trustworthy in the case study.

## 10. Phase 2 (explicitly out of scope for this build)

- **Query Agent**: ~~ad-hoc natural-language questions against already-extracted/validated data~~ — **built post-v1, see §5.5.** It turned out cheap to add exactly as predicted here (reused Pattern Agent's tools and architecture with no new extraction logic), and moved up ahead of the Recommendation Agent below because it only describes data, never suggests an action.
- **Recommendation Agent**: prescriptive suggestions (e.g. "consider shortening inspection interval") derived from Pattern Agent findings. Still deliberately last — this is the point in the pipeline where output starts resembling advice rather than reporting, which raises the same evidence-trail and liability considerations discussed for the tender-investigator project's Bid/No-Bid Agent. Should not be built until the Drafting Agent's faithfulness (open question 5) is well understood. A concrete constraint identified in discussion: it would have to recommend an *action type* ("schedule an inspection") grounded in real occurrence data, but must not invent specifics the data can't support (e.g. a numeric inspection interval) — the same "hedge rather than fabricate" discipline already forced into the Extraction Agent's prompt.
