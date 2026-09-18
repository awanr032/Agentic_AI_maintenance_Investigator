variable "aws_region" {
  description = "AWS region to deploy into. us-east-1 has the broadest free-tier service availability."
  type        = string
  default     = "us-east-1"
}

variable "deepseek_api_key" {
  description = <<-EOT
    The real DeepSeek API key. Passed in at apply-time (TF_VAR_deepseek_api_key env var
    or a gitignored terraform.tfvars) so it never appears in a .tf file that could be
    committed. Terraform still writes it into terraform.tfstate in plaintext (this is
    normal for any provider-managed secret resource) — that's why terraform.tfstate* is
    gitignored below, same reasoning as .env for the local app.
  EOT
  type        = string
  sensitive   = true
}
