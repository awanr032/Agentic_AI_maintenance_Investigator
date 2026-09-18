# Query Agent Lambda: its own IAM role, separate from the Extraction
# Agent's (agentic-ai-extraction-lambda-role in main.tf) — least privilege
# means each function gets exactly the permissions IT needs, not the union
# of everything every Lambda in this project needs. Extraction never
# touches S3; Query Agent needs the DeepSeek key (like Extraction) PLUS
# read/write on the store bucket (unlike Extraction).

resource "aws_iam_role" "query_agent_lambda_exec" {
  name = "agentic-ai-query-agent-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "query_agent_basic_execution" {
  role       = aws_iam_role.query_agent_lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "query_agent_read_deepseek_param" {
  name = "read-deepseek-api-key-param"
  role = aws_iam_role.query_agent_lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = aws_ssm_parameter.deepseek_api_key.arn
    }]
  })
}

resource "aws_iam_role_policy" "query_agent_store_access" {
  name = "read-write-store-bucket"
  role = aws_iam_role.query_agent_lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.store.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.store.arn
      }
    ]
  })
}

resource "aws_lambda_function" "query_agent" {
  function_name = "agentic-ai-query-agent"
  role          = aws_iam_role.query_agent_lambda_exec.arn
  handler       = "lambda_handler.handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 512

  filename         = "${path.module}/../lambda_build_query/build.zip"
  source_code_hash = filebase64sha256("${path.module}/../lambda_build_query/build.zip")

  environment {
    variables = {
      DEEPSEEK_API_KEY_PARAM = aws_ssm_parameter.deepseek_api_key.name
      STORE_S3_BUCKET        = aws_s3_bucket.store.bucket
    }
  }
}

# Same pattern (and same known caveat) as the Extraction Agent's Function
# URL in main.tf: authorization_type = NONE plus this explicit permission
# statement are both required for anonymous invoke to actually work, and
# on a brand-new AWS account, anonymous invoke may still be blocked by an
# account-level anti-abuse restriction regardless — direct invocation
# (aws lambda invoke / the console Test tab) is the reliable path either way.
resource "aws_lambda_function_url" "query_agent_url" {
  function_name      = aws_lambda_function.query_agent.function_name
  authorization_type = "NONE"
}

resource "aws_lambda_permission" "query_agent_public_url_invoke" {
  statement_id           = "AllowPublicFunctionUrlInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.query_agent.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

output "query_agent_function_url" {
  value       = aws_lambda_function_url.query_agent_url.function_url
  description = "POST {\"question\": \"...\", \"split\": \"silver\"} here"
}

output "query_agent_function_name" {
  value = aws_lambda_function.query_agent.function_name
}
