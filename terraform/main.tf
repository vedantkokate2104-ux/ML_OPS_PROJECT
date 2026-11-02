# Azure Container Apps hosting for the EventOracle Streamlit dashboard.
#
#   Resource Group ─┬─ Container Registry (ACR)  <- docker push project-apex image
#                   ├─ Log Analytics workspace
#                   └─ Container Apps Environment ── Container App (port 8501, HTTPS ingress)
#
# To extend to AKS: replace the Container Apps resources with
# azurerm_kubernetes_cluster + an AcrPull azurerm_role_assignment, then apply ../k8s/.

resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-rg"
  location = var.location
  tags     = var.tags
}

resource "azurerm_container_registry" "acr" {
  name                = "${var.project_name}acr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = var.tags
}

resource "azurerm_log_analytics_workspace" "logs" {
  name                = "${var.project_name}-logs"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

resource "azurerm_container_app_environment" "env" {
  name                       = "${var.project_name}-env"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
  tags                       = var.tags
}

resource "azurerm_container_app" "app" {
  name                         = "${var.project_name}-app"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.env.id
  revision_mode                = "Single"
  tags                         = var.tags

  registry {
    server               = azurerm_container_registry.acr.login_server
    username             = azurerm_container_registry.acr.admin_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.acr.admin_password
  }

  secret {
    name  = "mongo-uri"
    value = var.mongo_uri
  }

  secret {
    name  = "gemini-api-key"
    value = var.gemini_api_key == "" ? "unset" : var.gemini_api_key
  }

  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    container {
      name   = "streamlit"
      image  = "${azurerm_container_registry.acr.login_server}/project-apex:${var.image_tag}"
      cpu    = var.cpu
      memory = var.memory

      env {
        name        = "MONGO_URI"
        secret_name = "mongo-uri"
      }

      env {
        name        = "GEMINI_API_KEY"
        secret_name = "gemini-api-key"
      }

      env {
        name  = "ENABLE_METRICS"
        value = "true"
      }

      liveness_probe {
        transport = "HTTP"
        port      = 8501
        path      = "/_stcore/health"
      }

      readiness_probe {
        transport = "HTTP"
        port      = 8501
        path      = "/_stcore/health"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8501
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
}
