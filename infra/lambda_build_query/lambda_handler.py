"""AWS Lambda entry point wrapping src/query_agent.py's answer_question().

Same thin-adapter pattern as infra/lambda_build/lambda_handler.py (the
Extraction Agent's handler): translate Lambda's event/context shape into a
normal function call, nothing about query_agent.py itself changes.

The one thing genuinely new here versus the Extraction handler: this
agent's tools (src/tools.py, via src/review_queue.py, via src/store.py)
read the validated corpus, and Lambda's filesystem is ephemeral — so this
only works because store.py switches to its S3 backend when
STORE_S3_BUCKET is set (see src/store.py's module docstring). Setting that
environment variable is what makes "the same code that works locally
against JSONL files" also work here against S3, with zero changes to
query_agent.py/tools.py/review_queue.py themselves.

Expects the DeepSeek API key in SSM Parameter Store (see
DEEPSEEK_API_KEY_PARAM), fetched once per cold start and cached for the
execution environment's lifetime, and the validated corpus in S3 (see
STORE_S3_BUCKET) -- populated once via scripts/migrate_store_to_s3.py.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

_deepseek_key_cache: str | None = None

# Same guardrail rationale as the Extraction handler: this is a public,
# unauthenticated endpoint backed by a metered API.
MAX_QUESTION_LENGTH = 500


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
    """event body: {"question": "<a question>", "split": "silver"}. Returns
    an API-Gateway/Function-URL-shaped response (statusCode + JSON body)."""
    os.environ["DEEPSEEK_API_KEY"] = _load_deepseek_key()

    from src.query_agent import answer_question

    body = event.get("body")
    payload = json.loads(body) if isinstance(body, str) else (body or event)
    question = payload.get("question")
    split = payload.get("split", "silver")

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": '"question" is required'})}
    if len(question) > MAX_QUESTION_LENGTH:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"question exceeds {MAX_QUESTION_LENGTH} character limit"}),
        }
    if split not in ("gold", "silver"):
        return {"statusCode": 400, "body": json.dumps({"error": '"split" must be "gold" or "silver"'})}

    try:
        result = answer_question(question, split=split)
    except Exception as e:  # noqa: BLE001
        print(f"query failed: {e!r}")
        return {"statusCode": 502, "body": json.dumps({"error": "query failed"})}

    return {"statusCode": 200, "body": json.dumps(result.to_dict())}
