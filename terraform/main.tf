terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.117"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# Resource Group
resource "azurerm_resource_group" "mlops" {
  name     = "mlops-simple-rg"
  location = var.azure_location
}

# Public IP - Student optimized
resource "azurerm_public_ip" "mlops" {
  name                = "mlops-public-ip"
  location            = azurerm_resource_group.mlops.location
  resource_group_name = azurerm_resource_group.mlops.name
  allocation_method   = "Static"
  sku                 = "Standard"
  
  depends_on = [azurerm_resource_group.mlops]
}

# Network Security Group
resource "azurerm_network_security_group" "mlops" {
  name                = "mlops-nsg"
  location            = azurerm_resource_group.mlops.location
  resource_group_name = azurerm_resource_group.mlops.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "HTTP-8000"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "HTTP-8501"
    priority                   = 1003
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8501"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  depends_on = [azurerm_resource_group.mlops]
}

# Virtual Network
resource "azurerm_virtual_network" "mlops" {
  name                = "mlops-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.mlops.location
  resource_group_name = azurerm_resource_group.mlops.name
  
  depends_on = [azurerm_resource_group.mlops]
}

# Subnet
resource "azurerm_subnet" "mlops" {
  name                 = "mlops-subnet"
  resource_group_name  = azurerm_resource_group.mlops.name
  virtual_network_name = azurerm_virtual_network.mlops.name
  address_prefixes     = ["10.0.1.0/24"]
  
  depends_on = [azurerm_virtual_network.mlops]
}

# Network Interface
resource "azurerm_network_interface" "mlops" {
  name                = "mlops-nic"
  location            = azurerm_resource_group.mlops.location
  resource_group_name = azurerm_resource_group.mlops.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.mlops.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.mlops.id
  }
  
  depends_on = [azurerm_subnet.mlops, azurerm_public_ip.mlops]
}

# Associate NSG to NIC
resource "azurerm_network_interface_security_group_association" "mlops" {
  network_interface_id      = azurerm_network_interface.mlops.id
  network_security_group_id = azurerm_network_security_group.mlops.id
  
  depends_on = [azurerm_network_interface.mlops, azurerm_network_security_group.mlops]
}

# Virtual Machine - Student optimized
resource "azurerm_linux_virtual_machine" "mlops" {
  name                = "mlops-simple-vm"
  location            = azurerm_resource_group.mlops.location
  resource_group_name = azurerm_resource_group.mlops.name
  size                = "Standard_B1s"
  admin_username      = "azureuser"

  disable_password_authentication = true

  network_interface_ids = [
    azurerm_network_interface.mlops.id,
  ]

  admin_ssh_key {
    username   = "azureuser"
    public_key = file("~/.ssh/ehb_azure.pub")
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    disk_size_gb         = 30
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  custom_data = base64encode(<<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io docker-compose git
    systemctl start docker
    systemctl enable docker
    usermod -aG docker azureuser
    
    cd /home/azureuser
    git clone https://github.com/codesapienbe/mlpops.git
    cd mlpops
    chown -R azureuser:azureuser /home/azureuser/mlpops
    
    sudo -u azureuser docker-compose up -d
  EOF
  )

  depends_on = [azurerm_network_interface_security_group_association.mlops]
}

# Data source to get public IP
data "azurerm_public_ip" "mlops" {
  name                = azurerm_public_ip.mlops.name
  resource_group_name = azurerm_resource_group.mlops.name
  depends_on          = [azurerm_linux_virtual_machine.mlops]
}
