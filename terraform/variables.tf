variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "my_ip_cidr" {
  description = "CIDR duoc phep SSH vao Bastion Host. Mac dinh la IP public cua nguoi lam lab (least-privilege thay vi mo 0.0.0.0/0)"
  type        = string
  default     = "203.171.27.42/32"
}

variable "hf_token" {
  description = "Hugging Face Token for gated models (like Gemma)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "model_id" {
  description = "Hugging Face Model ID to serve"
  type        = string
  default     = "google/gemma-4-E2B-it"
}

variable "enable_gpu" {
  description = "Set to true to deploy the optional GPU + vLLM LLM inference node instead of the default CPU + LightGBM node"
  type        = bool
  default     = false
}

variable "cpu_instance_type" {
  description = "Instance type for the default CPU (LightGBM) compute node. Lab chi dinh t3.medium (2 vCPU / 4 GB), nhung tai khoan AWS Free Plan chi cho phep launch cac type free-tier-eligible. c7i-flex.large co cung 2 vCPU / 4 GB va nam trong danh sach do, dong thoi la x86_64 nen dung chung Ubuntu AMI amd64"
  type        = string
  default     = "c7i-flex.large"
}

variable "gpu_instance_type" {
  description = "Instance type for the optional GPU (vLLM) compute node"
  type        = string
  default     = "g4dn.xlarge"
}