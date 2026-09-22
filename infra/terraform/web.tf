# A real, independently-hosted static website for "Ask The Investigator"
# -- NOT a Claude artifact. Found the hard way: published artifacts run in
# a sandboxed environment that blocks fetch() to arbitrary external hosts
# (no CORS setting on the AWS side can work around that; it's a platform
# restriction on the artifact, not a server-side problem). Hosting the
# same page as a genuine S3 static website sidesteps this entirely, and
# keeps the whole demo -- frontend and backend -- under this project's own
# AWS account rather than split across two different platforms.

resource "aws_s3_bucket" "web" {
  bucket = "agentic-ai-maintenance-investigator-web-${random_id.store_bucket_suffix.hex}"
}

resource "aws_s3_bucket_website_configuration" "web" {
  bucket = aws_s3_bucket.web.id
  index_document {
    suffix = "index.html"
  }
}

# Unlike store.tf's bucket (private, Lambda-only access via IAM), this one
# must be genuinely public -- it's a website, not a data store. The same
# guardrails already on the API Gateway endpoint itself (question length
# cap, generic errors) are what actually protect against abuse; this
# bucket only ever serves the static HTML/CSS/JS, never any of the
# maintenance data.
resource "aws_s3_bucket_public_access_block" "web" {
  bucket                  = aws_s3_bucket.web.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "web" {
  bucket = aws_s3_bucket.web.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "PublicReadGetObject"
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.web.arn}/*"
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.web]
}

resource "aws_s3_object" "index" {
  bucket       = aws_s3_bucket.web.id
  key          = "index.html"
  source       = "${path.module}/../web/index.html"
  content_type = "text/html"
  etag         = filemd5("${path.module}/../web/index.html")
}

output "web_url" {
  value       = "http://${aws_s3_bucket.web.bucket}.s3-website-${var.aws_region}.amazonaws.com"
  description = "The real, public, working 'Ask The Investigator' page"
}
