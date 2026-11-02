variable "project_name" {
  description = "Prefix used for all resource names (lowercase letters/digits; ACR names must be globally unique)"
  type        = string
  default     = "projectapex"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "centralindia"
}

variable "image_tag" {
  description = "Tag of the project-apex image pushed to ACR"
  type        = string
  default     = "latest"
}

variable "cpu" {
  description = "vCPU per container replica"
  type        = number
  default     = 1.0
}

variable "memory" {
  description = "Memory per container replica (must pair with cpu, e.g. 1.0 => 2Gi)"
  type        = string
  default     = "2Gi"
}

variable "min_replicas" {
  type    = number
  default = 1
}

variable "max_replicas" {
  type    = number
  default = 3
}

variable "mongo_uri" {
  description = "MongoDB Atlas connection string (pass via TF_VAR_mongo_uri, never commit)"
  type        = string
  sensitive   = true
}

variable "gemini_api_key" {
  description = "Google Gemini API key (optional)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "tags" {
  type = map(string)
  default = {
    project     = "project-apex"
    environment = "dev"
    managed_by  = "terraform"
  }
}
