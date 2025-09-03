output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.mlops.name
}

output "resource_group_location" {
  description = "Location of the resource group"
  value       = azurerm_resource_group.mlops.location
}

output "vm_name" {
  description = "Name of the virtual machine"
  value       = azurerm_linux_virtual_machine.mlops.name
}

output "public_ip_address" {
  description = "Public IP address of the virtual machine"
  value       = data.azurerm_public_ip.mlops.ip_address
}

output "ssh_connection_command" {
  description = "Command to SSH into the virtual machine"
  value       = "ssh -i ~/.ssh/ehb_azure ${var.admin_username}@${data.azurerm_public_ip.mlops.ip_address}"
}

output "api_url" {
  description = "URL to access the API service"
  value       = "http://${data.azurerm_public_ip.mlops.ip_address}:8000"
}

output "api_docs_url" {
  description = "URL to access the API documentation"
  value       = "http://${data.azurerm_public_ip.mlops.ip_address}:8000/docs"
}

output "streamlit_url" {
  description = "URL to access the Streamlit web interface"
  value       = "http://${data.azurerm_public_ip.mlops.ip_address}:8501"
}

output "vm_size" {
  description = "Size of the deployed virtual machine"
  value       = azurerm_linux_virtual_machine.mlops.size
}

output "os_disk_size" {
  description = "Size of the OS disk in GB"
  value       = "${azurerm_linux_virtual_machine.mlops.os_disk[0].disk_size_gb}GB"
}

output "network_security_group_name" {
  description = "Name of the network security group"
  value       = azurerm_network_security_group.mlops.name
}

output "virtual_network_name" {
  description = "Name of the virtual network"
  value       = azurerm_virtual_network.mlops.name
}

output "deployment_summary" {
  description = "Summary of the deployed infrastructure"
  value = {
    resource_group = azurerm_resource_group.mlops.name
    location       = azurerm_resource_group.mlops.location
    vm_name        = azurerm_linux_virtual_machine.mlops.name
    vm_size        = azurerm_linux_virtual_machine.mlops.size
    public_ip      = data.azurerm_public_ip.mlops.ip_address
    api_url        = "http://${data.azurerm_public_ip.mlops.ip_address}:8000"
    web_url        = "http://${data.azurerm_public_ip.mlops.ip_address}:8501"
  }
}
