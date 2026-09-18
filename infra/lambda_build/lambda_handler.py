"""AWS Lambda entry point wrapping src/extraction_agent.py's extract().

This file, plus a copy of src/extraction_agent.py, src/schema.py, and
data/{gold_release.json,scheme.json}, form the whole Lambda deployment
package — extraction_agent.py itself is unchanged, this is purely a thin
adapter translating Lambda's event/context shape into a normal function
call, per the plan discussed: AWS decides *where the code runs*, nothing
about how extraction itself works needs to change.

Expects the DeepSeek API key in SSM Parameter Store (not a Lambda
environment variable in plaintext) at the path given by the
DEEPSEEK_API_KEY_PARAM environment variable, fetched once per cold start
and cached for the lifetime of this execution environment.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

_deepseek_key_cache: str | None = None

# Guardrail: this is a public, unauthenticated endpoint backed by a metered
# API. Capping input length bounds the worst-case cost of a single abusive
# call (a very long text costs more tokens per request than a normal MWO).
MAX_TEXT_LENGTH = 2000


def _load_deepseek_key() -> str:
    """Fetch the API key from SSM Parameter Store once per cold start,
    rather than baking it into a plaintext Lambda environment variable."""
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
    """event body: {"text": "<one MWO text>"}. Returns an API-Gateway/
    Function-URL-shaped response (statusCode + JSON body) so this same
    handler works whether it's invoked directly or through an HTTP front
    door later."""
    os.environ["DEEPSEEK_API_KEY"] = _load_deepseek_key()

    from src.extraction_agent import extract
    from src.schema import entity_types, relation_types

    body = event.get("body")
    payload = json.loads(body) if isinstance(body, str) else (body or event)
    text = payload.get("text")
    if not text:
        return {"statusCode": 400, "body": json.dumps({"error": '"text" is required'})}
    if len(text) > MAX_TEXT_LENGTH:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"text exceeds {MAX_TEXT_LENGTH} character limit"}),
        }

    try:
        result = extract(text, entity_types(), relation_types())
    except Exception as e:  # noqa: BLE001
        # Log the real exception to CloudWatch (this print lands in the Lambda's
        # log group) but don't echo internal details back to an anonymous caller.
        print(f"extraction failed: {e!r}")
        return {"statusCode": 502, "body": json.dumps({"error": "extraction failed"})}

    return {"statusCode": 200, "body": json.dumps(result.to_dict())}
