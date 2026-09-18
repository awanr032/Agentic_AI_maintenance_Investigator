# S3 bucket backing src/store.py's S3 mode (STORE_S3_BUCKET) — shared by
# any agent Lambda that needs the validated corpus (currently: Query Agent).
# Bucket names are globally unique across ALL of AWS, not just this
# account, so a random suffix avoids a name collision with someone else's
# bucket rather than requiring you to hand-pick a unique name.

resource "random_id" "store_bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "store" {
  bucket = "agentic-ai-maintenance-investigator-store-${random_id.store_bucket_suffix.hex}"
}

# No public access whatsoever — every reader/writer is a Lambda function
# authenticated via its own IAM role, never an anonymous HTTP request the
# way the Function URLs are. This is a private data bucket, not a website.
resource "aws_s3_bucket_public_access_block" "store" {
  bucket                  = aws_s3_bucket.store.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "store_bucket_name" {
  value       = aws_s3_bucket.store.bucket
  description = "Run scripts/migrate_store_to_s3.py --bucket <this> to upload your existing local data."
}
