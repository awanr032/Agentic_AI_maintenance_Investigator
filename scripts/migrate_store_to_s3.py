"""One-time copy of existing local store data into an S3 bucket, in the
key format src/store.py's S3 backend expects (one object per record).

This does NOT set STORE_S3_BUCKET for itself -- it deliberately reads via
store.py's LOCAL backend (the source of truth on this machine) and writes
directly to S3 via boto3, so one process can copy in one direction without
needing to toggle store.py's backend mid-run.

Run from repo root, after the S3 bucket exists (see infra/terraform):
    .venv\\Scripts\\python.exe -m scripts.migrate_store_to_s3 --bucket YOUR_BUCKET_NAME
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boto3  # noqa: E402

from src import store  # noqa: E402  -- used in its default LOCAL-backend mode

KEYED_COLLECTIONS = ["extracted", "validated"]
SPLITS = ["gold", "silver"]
SINGLE_DOCS = ["scores/eval_report", "patterns/findings"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy local store data into an S3 bucket.")
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()

    s3 = boto3.client("s3")
    total_records = 0

    for collection in KEYED_COLLECTIONS:
        for split in SPLITS:
            n = 0
            for index, record in store.iter_records(collection, split):
                s3.put_object(
                    Bucket=args.bucket,
                    Key=f"{collection}/{split}/{index}.json",
                    Body=json.dumps(record, ensure_ascii=False).encode("utf-8"),
                    ContentType="application/json",
                )
                n += 1
            if n:
                print(f"{collection}/{split}: uploaded {n} records")
            total_records += n

    for name in SINGLE_DOCS:
        doc = store.read_doc(name)
        if doc is None:
            continue
        s3.put_object(
            Bucket=args.bucket,
            Key=f"{name}.json",
            Body=json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"{name}.json: uploaded (single document)")

    print(f"\ndone: {total_records} keyed records + single documents uploaded to s3://{args.bucket}/")


if __name__ == "__main__":
    main()
