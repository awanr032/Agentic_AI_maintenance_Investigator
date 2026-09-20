"""AWS Lambda entry point wrapping src/pattern_agent.py's find_patterns().

Same thin-adapter pattern as the Extraction and Query Agent handlers, and
the same S3-backed store.py dependency as the Query Agent's (see that
handler's docstring for why STORE_S3_BUCKET is what makes this work at
all against Lambda's ephemeral filesystem).

Unlike the other two agents, this one is a batch investigation over a
whole split rather than a single per-request answer -- num_candidates is
capped (see MAX_NUM_CANDIDATES) both as a cost guardrail on the public
endpoint and because src/tools.py's clear_cache()-based fix means every
asset type investigated in one run shares a single underlying data fetch,
but each candidate can still trigger its own DeepSeek tool-call turn.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

_deepseek_key_cache: str | None = None

MAX_NUM_CANDIDATES = 50


def _load_deepseek_key() -> str:
    global _deepseek_key_cache
    if _deepseek_key_cache is not None:
        return _deepseek_key_cache
    import boto3  # available in the Lambda runtime by default, not bundled

    param_name = os.environ["DEEPSEEK_API_KEY_PARAM"]
    ssm = boto3.client("ssm")
    response = ssm.get_parameter(Name=param_name, WithDecryption=True)
    _deepseek_key_cache = response["Parameter"]["Value"]
    return _deepseek_key_cache


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """event body: {"split": "silver", "num_candidates": 20} (both optional).
    Returns an API-Gateway/Function-URL-shaped response (statusCode + JSON
    body). Findings are also persisted to S3 (patterns/findings.json) by
    find_patterns() itself, same as the local batch script's behavior."""
    os.environ["DEEPSEEK_API_KEY"] = _load_deepseek_key()

    from src.pattern_agent import find_patterns

    body = event.get("body")
    payload = json.loads(body) if isinstance(body, str) else (body or event or {})
    split = payload.get("split", "silver")
    num_candidates = payload.get("num_candidates", 20)

    if split not in ("gold", "silver"):
        return {"statusCode": 400, "body": json.dumps({"error": '"split" must be "gold" or "silver"'})}
    if not isinstance(num_candidates, int) or not (1 <= num_candidates <= MAX_NUM_CANDIDATES):
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f'"num_candidates" must be an integer from 1 to {MAX_NUM_CANDIDATES}'}),
        }

    try:
        findings = find_patterns(split=split, num_candidates=num_candidates)
    except Exception as e:  # noqa: BLE001
        print(f"pattern finding failed: {e!r}")
        return {"statusCode": 502, "body": json.dumps({"error": "pattern finding failed"})}

    return {"statusCode": 200, "body": json.dumps([f.to_dict() for f in findings])}
