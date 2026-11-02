terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
  }

  # Optional remote state – uncomment and configure for team use
  # backend "azurerm" {
  #   resource_group_name  = "tfstate-rg"
  #   storage_account_name = "tfstateapex"
  #   container_name       = "tfstate"
  #   key                  = "project-apex.tfstate"
  # }
}

provider "azurerm" {
  features {}
}
