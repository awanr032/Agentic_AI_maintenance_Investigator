# A real, public, browser-callable HTTP endpoint in front of the Query
# Agent Lambda -- separate from its Function URL (main.tf/query_agent.tf),
# which remains blocked by what looks like a new-AWS-account restriction
# on anonymous *Function URL* access specifically. API Gateway invokes
# Lambda through a different mechanism entirely (lambda:InvokeFunction via
# its own resource policy, granted below, rather than
# lambda:InvokeFunctionUrl) -- a different AWS feature, not just a
# different door into the same one, so it isn't expected to hit the same
# restriction. This is also the standard, production-grade way to expose
# a Lambda publicly, rather than relying on a bare Function URL long-term.
#
# HTTP API (API Gateway v2), not the older REST API type: simpler,
# cheaper, and its CORS support is one config block instead of manually
# wiring up an OPTIONS method -- needed here since a browser page will
# call this directly with fetch().

resource "aws_apigatewayv2_api" "query_agent" {
  name          = "agentic-ai-query-agent-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["content-type"]
  }
}

resource "aws_apigatewayv2_integration" "query_agent" {
  api_id                 = aws_apigatewayv2_api.query_agent.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.query_agent.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "query_agent" {
  api_id    = aws_apigatewayv2_api.query_agent.id
  route_key = "POST /ask"
  target    = "integrations/${aws_apigatewayv2_integration.query_agent.id}"
}

resource "aws_apigatewayv2_stage" "query_agent" {
  api_id      = aws_apigatewayv2_api.query_agent.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "query_agent_apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.query_agent.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.query_agent.execution_arn}/*/*"
}

output "query_agent_api_url" {
  value       = "${aws_apigatewayv2_api.query_agent.api_endpoint}/ask"
  description = "POST {\"question\": \"...\", \"split\": \"silver\"} here -- the real, public, browser-callable endpoint"
}
