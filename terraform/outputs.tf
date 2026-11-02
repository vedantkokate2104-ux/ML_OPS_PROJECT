output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "acr_login_server" {
  description = "Push images here: docker push <login_server>/project-apex:<tag>"
  value       = azurerm_container_registry.acr.login_server
}

output "app_url" {
  description = "Public HTTPS URL of the Streamlit dashboard"
  value       = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}
