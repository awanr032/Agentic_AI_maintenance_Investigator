terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.aws_region
  # No access keys here — the AWS provider uses the same credential chain as the
  # aws CLI (the ~/.aws/credentials file written by `aws configure`), so whatever
  # already authenticates `aws sts get-caller-identity` authenticates Terraform too.
}

# ---------------------------------------------------------------------------
# Secret: the DeepSeek API key, stored in SSM Parameter Store as a SecureString
# (encrypted at rest with the AWS-managed KMS key), not as a plaintext Lambda
# environment variable. lambda_handler.py reads this by name at cold start.
# ---------------------------------------------------------------------------
resource "aws_ssm_parameter" "deepseek_api_key" {
  name        = "/agentic-ai-maintenance-investigator/deepseek-api-key"
  description = "DeepSeek API key used by the Extraction Agent Lambda"
  type        = "SecureString"
  value       = var.deepseek_api_key
}

# ---------------------------------------------------------------------------
# IAM: the role Lambda assumes when it runs, plus exactly two permissions —
# write CloudWatch logs, and read this one SSM parameter. Least privilege:
# this role can't touch any other AWS resource in the account.
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_exec" {
  name = "agentic-ai-extraction-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "basic_execution" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "read_deepseek_param" {
  name = "read-deepseek-api-key-param"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = aws_ssm_parameter.deepseek_api_key.arn
    }]
  })
}

# ---------------------------------------------------------------------------
# The Lambda function itself. filename points at the zip we built by hand
# (infra/lambda_build/build.zip); source_code_hash tells Terraform to redeploy
# whenever that zip's contents change.
# ---------------------------------------------------------------------------
resource "aws_lambda_function" "extraction_agent" {
  function_name = "agentic-ai-extraction-agent"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "lambda_handler.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 512

  # No reserved_concurrent_executions: this account's total concurrency
  # ceiling is only 10 right now (a brand-new-account default AWS raises
  # over time), which already bounds this function tighter than any
  # per-function reservation we could set without violating AWS's rule
  # that unreserved capacity can't drop below 10.

  filename         = "${path.module}/../lambda_build/build.zip"
  source_code_hash = filebase64sha256("${path.module}/../lambda_build/build.zip")

  environment {
    variables = {
      DEEPSEEK_API_KEY_PARAM = aws_ssm_parameter.deepseek_api_key.name
    }
  }
}

# ---------------------------------------------------------------------------
# Function URL: the simplest way to get a real callable HTTPS endpoint without
# standing up API Gateway yet. authorization_type = NONE means anyone with the
# URL can call it — acceptable for a demo endpoint hitting a metered API, but
# worth knowing: there is no auth in front of this by default.
# ---------------------------------------------------------------------------
resource "aws_lambda_function_url" "extraction_url" {
  function_name      = aws_lambda_function.extraction_agent.function_name
  authorization_type = "NONE"
}

# authorization_type = NONE only sets the URL's own auth mode — AWS still
# requires this separate resource-based permission statement granting the
# public ("*") principal the right to actually invoke it. Without this,
# every call gets a 403 Forbidden regardless of the URL's auth setting.
resource "aws_lambda_permission" "public_url_invoke" {
  statement_id           = "AllowPublicFunctionUrlInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.extraction_agent.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

output "function_url" {
  value       = aws_lambda_function_url.extraction_url.function_url
  description = "POST {\"text\": \"...\"} here to run the Extraction Agent"
}

output "lambda_function_name" {
  value = aws_lambda_function.extraction_agent.function_name
}
