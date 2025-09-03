# MLOps Project - Complete Automation

This project provides a complete MLOps pipeline with automated infrastructure deployment and application management using Terraform and Docker Compose.

## 🚀 Quick Start

### Prerequisites
- Terraform installed
- Docker and Docker Compose installed
- SSH key pair for Azure VM access (`~/.ssh/ehb_azure`)
- Azure CLI configured with proper permissions
- Azure subscription with sufficient permissions

### One-Command Deployment
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Deploy everything
./scripts/run.sh deploy

# Check status
./scripts/run.sh status

# View logs
./scripts/run.sh logs

# Cleanup when done
./scripts/run.sh cleanup
```

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `./scripts/run.sh deploy` | Deploy infrastructure and application |
| `./scripts/run.sh status` | Check deployment status and health |
| `./scripts/run.sh cleanup` | Destroy infrastructure and cleanup |
| `./scripts/run.sh logs` | Show recent deployment logs |
| `./scripts/run.sh help` | Show help message |

## 🏗️ Architecture

### Infrastructure (Terraform + Azure)
- **Resource Group**: `mlops-simple-rg`
- **Virtual Machine**: Ubuntu 22.04 LTS (Standard_B1s)
- **Network**: Custom VNet with NSG rules for ports 22, 8000, 8501
- **Public IP**: Static IP for external access
- **Location**: Configurable via `terraform/variables.tf`

### Application (Docker Compose)
- **MLOps API**: FastAPI service on port 8000
- **MLOps Web**: Streamlit UI on port 8501
- **MLOps Training**: Pipeline execution service

## 🔧 Automation Features

### Complete Deployment Pipeline
1. **Infrastructure Provisioning**: Terraform creates Azure resources
2. **VM Setup**: Automatic Docker installation and configuration
3. **Application Deployment**: Docker Compose builds and starts services
4. **Health Monitoring**: Continuous health checks and status monitoring
5. **Logging**: Comprehensive logging for troubleshooting
6. **Azure Integration**: Native Azure services and networking

### Health Checks
- API endpoint health verification
- Web UI accessibility checks
- Docker container status monitoring
- VM connectivity validation

### Error Handling
- Graceful failure handling
- Detailed error logging
- Automatic retry mechanisms
- Cleanup on failure

## 📊 Monitoring & Logs

### Log Files
- `deployment.log`: Complete deployment process logs
- `status.log`: Status check logs
- `health.log`: VM-side health monitoring logs

### Health Endpoints
- API Health: `http://<instance-ip>:8000/health`
- Web UI: `http://<instance-ip>:8501`

## 🛠️ Manual Operations

### Individual Scripts
```bash
# Deploy infrastructure and application
./scripts/deploy.sh

# Check deployment status
./scripts/status.sh

# Cleanup resources
./scripts/cleanup.sh
```

### Terraform Operations (Azure)
```bash
cd terraform

# Initialize
terraform init

# Plan changes
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy

# Show outputs
terraform output

# Configure Azure variables
# Edit terraform/variables.tf for location, subscription, etc.
```

### Docker Compose Operations
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs

# Check status
docker-compose ps
```

## 🔒 Security Features

- SSH key-based authentication
- Network Security Groups with minimal port exposure
- Non-root user execution
- Secure Docker configuration

## 📈 Scaling & Maintenance

### Adding New Services
1. Update `docker-compose.yml`
2. Add health check endpoints
3. Update monitoring scripts
4. Redeploy with `./scripts/run.sh deploy`

### Updating Application
1. Push changes to Git repository
2. Run `./scripts/run.sh deploy` to redeploy
3. Scripts automatically pull latest changes

### Monitoring & Alerts
- Built-in health monitoring on VM
- Structured logging for external monitoring systems
- Health check endpoints for load balancer integration
- Azure Monitor integration ready
- Azure Log Analytics compatible

## 🚨 Troubleshooting

### Common Issues

#### VM Not Accessible
```bash
# Check VM status
./scripts/run.sh status

# Verify SSH key configuration
ls -la ~/.ssh/ehb_azure*

# Check Azure portal for VM status
# Azure CLI: az vm show --name mlops-simple-vm --resource-group mlops-simple-rg
```

#### Services Not Starting
```bash
# Check container logs
./scripts/run.sh status

# SSH into VM and check manually
ssh azureuser@<instance-ip>
cd mlpops
docker-compose logs
```

#### Health Check Failures
```bash
# View detailed logs
./scripts/run.sh logs

# Check service endpoints
curl http://<instance-ip>:8000/health
curl http://<instance-ip>:8501
```

### Debug Mode
```bash
# Enable verbose logging
export TF_LOG=DEBUG
export TF_LOG_PATH=terraform.log

# Run deployment with detailed output
./scripts/run.sh deploy

# Azure CLI debugging
az account show
az vm list --resource-group mlops-simple-rg
```

## 📚 Additional Resources

- [Terraform Azure Provider Documentation](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Azure VM Documentation](https://docs.microsoft.com/en-us/azure/virtual-machines/)
- [Azure CLI Documentation](https://docs.microsoft.com/en-us/cli/azure/)
- [Azure Resource Manager](https://docs.microsoft.com/en-us/azure/azure-resource-manager/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `./scripts/run.sh deploy`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
