# Pattern Agent Lambda: its own IAM role, same least-privilege reasoning as
# query_agent.tf -- needs the DeepSeek key (like every agent) plus
# read/write on the same store bucket Query Agent uses (it reads the
# validated corpus and writes patterns/findings.json).

resource "aws_iam_role" "pattern_agent_lambda_exec" {
  name = "agentic-ai-pattern-agent-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "pattern_agent_basic_execution" {
  role       = aws_iam_role.pattern_agent_lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "pattern_agent_read_deepseek_param" {
  name = "read-deepseek-api-key-param"
  role = aws_iam_role.pattern_agent_lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = aws_ssm_parameter.deepseek_api_key.arn
    }]
  })
}

resource "aws_iam_role_policy" "pattern_agent_store_access" {
  name = "read-write-store-bucket"
  role = aws_iam_role.pattern_agent_lambda_exec.id

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

resource "aws_lambda_function" "pattern_agent" {
  function_name = "agentic-ai-pattern-agent"
  role          = aws_iam_role.pattern_agent_lambda_exec.arn
  handler       = "lambda_handler.handler"
  runtime       = "python3.12"
  # Higher than Query Agent's 60s: up to MAX_TOOL_TURNS=8 turns each
  # potentially calling the LLM, versus Query Agent's typical 1-3 -- set
  # generously up front based on the lesson already learned from Query
  # Agent's real timeout, not discovered the same way twice. The
  # clear_cache() fix in tools.py means every asset type investigated in
  # one run shares a single S3 fetch, so this is bounded by LLM call
  # latency across turns, not multiplied S3 I/O.
  timeout       = 90
  memory_size   = 512

  filename         = "${path.module}/../lambda_build_pattern/build.zip"
  source_code_hash = filebase64sha256("${path.module}/../lambda_build_pattern/build.zip")

  environment {
    variables = {
      DEEPSEEK_API_KEY_PARAM = aws_ssm_parameter.deepseek_api_key.name
      STORE_S3_BUCKET        = aws_s3_bucket.store.bucket
    }
  }
}

# Same authorization_type=NONE + explicit permission pattern, and the same
# known caveat (a new-AWS-account restriction may still block anonymous
# invoke regardless) as the other two agents' Function URLs.
resource "aws_lambda_function_url" "pattern_agent_url" {
  function_name      = aws_lambda_function.pattern_agent.function_name
  authorization_type = "NONE"
}

resource "aws_lambda_permission" "pattern_agent_public_url_invoke" {
  statement_id           = "AllowPublicFunctionUrlInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.pattern_agent.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

output "pattern_agent_function_url" {
  value       = aws_lambda_function_url.pattern_agent_url.function_url
  description = "POST {\"split\": \"silver\", \"num_candidates\": 20} here"
}

output "pattern_agent_function_name" {
  value = aws_lambda_function.pattern_agent.function_name
}
