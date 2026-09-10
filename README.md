# Agentic AI Maintenance Investigator

Practice/portfolio project: a four-agent pipeline (Extraction → Validation → Pattern →
Drafting) that turns short maintenance work order text into structured, evidence-backed
findings, benchmarked against the public MaintIE dataset.

See `docs/design-spec.md` for the full design and `CLAUDE.md` for build conventions.

## Setup

```bash
cp .env.example .env   # then fill in your API key(s)
pip install -r requirements.txt
python scripts/fetch_data.py   # clones MaintIE's gold/silver data into data/
```

## Data attribution

This project fetches data from [MaintIE](https://github.com/nlp-tlp/maintie) (MIT licensed).
The dataset itself is not vendored into this repo — `scripts/fetch_data.py` pulls it at setup
time. Credit to the MaintIE authors for the gold/silver corpora and taxonomy this project is
built on top of and benchmarked against.
