variable "subscription_id" {
  description = "Azure subscription ID for student account"
  type        = string
  default     = "c0b1d897-0053-4d32-a69a-7009b075dc67"
}

variable "azure_location" {
  description = "Azure region - must be from your allowed student regions"
  type        = string
  default     = "francecentral"
  
  validation {
    condition = contains([
      "norwayeast",
      "polandcentral", 
      "spaincentral",
      "francecentral",
      "switzerlandnorth"
    ], var.azure_location)
    error_message = "The azure_location must be one of your allowed student subscription regions."
  }
}

variable "admin_username" {
  description = "Admin username for the VM"
  type        = string
  default     = "azureuser"
}

variable "vm_size" {
  description = "Size of the virtual machine"
  type        = string
  default     = "Standard_B1s"
  
  validation {
    condition = contains([
      "Standard_B1s",
      "Standard_B1ms",
      "Standard_B2s"
    ], var.vm_size)
    error_message = "VM size must be a valid free tier option."
  }
}

variable "disk_size_gb" {
  description = "Size of OS disk in GB"
  type        = number
  default     = 30
  
  validation {
    condition     = var.disk_size_gb >= 30 && var.disk_size_gb <= 64
    error_message = "Disk size must be between 30 and 64 GB for free tier."
  }
}
