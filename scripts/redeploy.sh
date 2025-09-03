#!/bin/bash
set -e

# Configuration
LOG_FILE="redeploy.log"
TERRAFORM_DIR="terraform"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Get instance IP from Terraform
get_instance_ip() {
    if [ -d "$TERRAFORM_DIR" ]; then
        cd "$TERRAFORM_DIR" 2>/dev/null || return 1
        INSTANCE_IP=$(terraform output -raw instance_ip 2>/dev/null) || return 1
        cd - > /dev/null 2>/dev/null || true
        echo "$INSTANCE_IP"
    else
        return 1
    fi
}

# Redeploy application
redeploy_application() {
    local instance_ip=$1
    
    log "🔄 Redeploying application on $instance_ip..."
    
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null azureuser@"$instance_ip" << 'EOF'
        cd /home/azureuser/mlpops
        
        # Pull latest changes
        git pull origin main || echo "Git pull failed, continuing with existing code"
        
        # Stop containers
        docker-compose down
        
        # Rebuild and start containers
        docker-compose build --no-cache
        docker-compose up -d
        
        # Wait for services to start
        sleep 30
        
        # Check container status
        docker-compose ps
        
        # Show recent logs
        echo "=== API Logs ==="
        docker-compose logs --tail=10 mlops-api
        echo "=== Web Logs ==="
        docker-compose logs --tail=10 mlops-web
EOF
    
    if [ $? -ne 0 ]; then
        log "❌ Failed to redeploy application"
        return 1
    fi
    
    log "✅ Application redeployment completed!"
}

# Verify services
verify_services() {
    local instance_ip=$1
    
    log "🔍 Verifying services after redeployment..."
    
    # Wait a bit for services to fully start
    sleep 10
    
    # Check API health
    if curl -f -s "http://$instance_ip:8000/health" > /dev/null 2>&1; then
        log "✅ API is healthy"
    else
        log "❌ API health check failed"
        return 1
    fi
    
    # Check Web UI
    if curl -f -s "http://$instance_ip:8501" > /dev/null 2>&1; then
        log "✅ Web UI is accessible"
    else
        log "❌ Web UI check failed"
        return 1
    fi
    
    log "🎉 All services are healthy!"
}

# Main function
main() {
    log "🚀 Starting application redeployment..."
    
    # Get instance IP
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        log "❌ Could not retrieve instance IP from Terraform"
        log "💡 Run 'terraform apply' first or check terraform directory"
        exit 1
    fi
    
    log "📍 Instance IP: $INSTANCE_IP"
    
    # Redeploy application
    redeploy_application "$INSTANCE_IP"
    
    # Verify services
    verify_services "$INSTANCE_IP"
    
    # Final status
    log "🎯 Redeployment Summary:"
    log "   📍 Instance IP: $INSTANCE_IP"
    log "   📊 API: http://$INSTANCE_IP:8000"
    log "   🌐 Web UI: http://$INSTANCE_IP:8501"
    log "   📋 Logs: $LOG_FILE"
    
    log "✅ Redeployment completed successfully!"
}

# Run main function
main "$@" 