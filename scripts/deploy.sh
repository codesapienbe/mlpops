#!/bin/bash
set -e

# Configuration
LOG_FILE="deployment.log"
TERRAFORM_DIR="terraform"
DOCKER_COMPOSE_FILE="docker-compose.yml"
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
HEALTH_CHECK_INTERVAL=10  # 10 seconds

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "❌ ERROR: $1"
    exit 1
}

# Health check function
check_service_health() {
    local service_url=$1
    local service_name=$2
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / HEALTH_CHECK_INTERVAL))
    local attempt=1
    
    log "🏥 Checking $service_name health at $service_url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$service_url" > /dev/null 2>&1; then
            log "✅ $service_name is healthy!"
            return 0
        fi
        
        log "⏳ Attempt $attempt/$max_attempts: $service_name not ready yet, waiting ${HEALTH_CHECK_INTERVAL}s..."
        sleep $HEALTH_CHECK_INTERVAL
        ((attempt++))
    done
    
    error_exit "$service_name failed health check after $HEALTH_CHECK_TIMEOUT seconds"
}

# Wait for VM to be ready
wait_for_vm_ready() {
    local instance_ip=$1
    local max_attempts=60
    local attempt=1
    
    log "🖥️ Waiting for VM to be ready at $instance_ip..."
    
    while [ $attempt -le $max_attempts ]; do
        if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null azureuser@"$instance_ip" "echo 'VM is ready'" 2>/dev/null; then
            log "✅ VM is ready and accessible!"
            return 0
        fi
        
        log "⏳ Attempt $attempt/$max_attempts: VM not ready yet, waiting 10s..."
        sleep 10
        ((attempt++))
    done
    
    error_exit "VM failed to become ready after $((max_attempts * 10)) seconds"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "🏗️ Deploying infrastructure with Terraform..."
    
    cd "$TERRAFORM_DIR" || error_exit "Failed to change to Terraform directory"
    
    # Initialize Terraform
    log "🔧 Initializing Terraform..."
    terraform init || error_exit "Terraform init failed"
    
    # Apply Terraform configuration
    log "🚀 Applying Terraform configuration..."
    terraform apply -auto-approve || error_exit "Terraform apply failed"
    
    # Get instance IP
    log "📡 Getting instance IP..."
    INSTANCE_IP=$(terraform output -raw instance_ip 2>/dev/null) || error_exit "Failed to get instance IP"
    log "📍 Instance IP: $INSTANCE_IP"
    
    cd - > /dev/null || error_exit "Failed to return to root directory"
    
    echo "$INSTANCE_IP"
}

# Deploy application with Docker Compose
deploy_application() {
    local instance_ip=$1
    
    log "🐳 Deploying application with Docker Compose on $instance_ip..."
    
    # Wait for VM to be ready
    wait_for_vm_ready "$instance_ip"
    
    # Ensure Docker Compose is running
    log "🔧 Ensuring Docker Compose is running..."
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null azureuser@"$instance_ip" << 'EOF'
        cd /home/azureuser/mlpops
        
        # Stop any existing containers
        docker-compose down || true
        
        # Pull latest changes
        git pull origin main || true
        
        # Build and start containers
        docker-compose build --no-cache
        docker-compose up -d
        
        # Wait for containers to start
        sleep 30
        
        # Check container status
        docker-compose ps
EOF
    
    if [ $? -ne 0 ]; then
        error_exit "Failed to deploy application on VM"
    fi
    
    log "✅ Application deployment completed!"
}

# Verify deployment
verify_deployment() {
    local instance_ip=$1
    
    log "🔍 Verifying deployment..."
    
    # Check API health
    check_service_health "http://$instance_ip:8000/health" "API"
    
    # Check Web UI health
    check_service_health "http://$instance_ip:8501" "Web UI"
    
    log "🎉 All services are healthy and running!"
}

# Main deployment function
main() {
    log "🚀 Starting complete MLOps deployment..."
    
    # Check prerequisites
    command -v terraform >/dev/null 2>&1 || error_exit "Terraform is not installed"
    command -v docker >/dev/null 2>&1 || error_exit "Docker is not installed"
    command -v docker-compose >/dev/null 2>&1 || error_exit "Docker Compose is not installed"
    
    # Deploy infrastructure
    INSTANCE_IP=$(deploy_infrastructure)
    
    # Deploy application
    deploy_application "$INSTANCE_IP"
    
    # Verify deployment
    verify_deployment "$INSTANCE_IP"
    
    # Final status
    log "🎯 Deployment Summary:"
    log "   📍 Instance IP: $INSTANCE_IP"
    log "   📊 API: http://$INSTANCE_IP:8000"
    log "   🌐 Web UI: http://$INSTANCE_IP:8501"
    log "   📋 Logs: $LOG_FILE"
    
    log "✅ Complete deployment successful!"
}

# Run main function
main "$@"
