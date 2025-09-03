#!/bin/bash
set -e

# Configuration
LOG_FILE="cleanup.log"
TERRAFORM_DIR="terraform"
STATUS_LOG="status.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Confirmation prompt
confirm_cleanup() {
    echo -e "${RED}⚠️ WARNING: This will destroy all infrastructure and data!${NC}"
    echo -e "${YELLOW}This action cannot be undone.${NC}"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        echo -e "${GREEN}Cleanup cancelled.${NC}"
        exit 0
    fi
}

# Get instance IP before cleanup
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

# Stop Docker containers gracefully
stop_containers() {
    local instance_ip=$1
    
    if [ -n "$instance_ip" ]; then
        log "🐳 Stopping Docker containers on $instance_ip..."
        if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null azureuser@"$instance_ip" "cd /home/azureuser/mlpops && docker-compose down" 2>/dev/null; then
            log "✅ Containers stopped successfully"
        else
            log "⚠️ Could not stop containers (VM may already be down)"
        fi
    fi
}

# Clean up local files
cleanup_local() {
    log "🧹 Cleaning up local files..."
    
    # Remove log files
    [ -f "$LOG_FILE" ] && rm -f "$LOG_FILE" && log "✅ Removed $LOG_FILE"
    [ -f "$STATUS_LOG" ] && rm -f "$STATUS_LOG" && log "✅ Removed $STATUS_LOG"
    
    # Remove Terraform state files (optional)
    if [ -d "$TERRAFORM_DIR" ]; then
        cd "$TERRAFORM_DIR"
        if [ -f ".terraform.lock.hcl" ]; then
            rm -f .terraform.lock.hcl && log "✅ Removed Terraform lock file"
        fi
        cd - > /dev/null
    fi
    
    log "✅ Local cleanup completed"
}

# Destroy infrastructure
destroy_infrastructure() {
    log "🏗️ Destroying infrastructure with Terraform..."
    
    if [ ! -d "$TERRAFORM_DIR" ]; then
        log "⚠️ Terraform directory not found, skipping infrastructure cleanup"
        return 0
    fi
    
    cd "$TERRAFORM_DIR" || error_exit "Failed to change to Terraform directory"
    
    # Check if Terraform is initialized
    if [ ! -d ".terraform" ]; then
        log "🔧 Initializing Terraform..."
        terraform init || log "⚠️ Terraform init failed, continuing with cleanup"
    fi
    
    # Destroy infrastructure
    log "💥 Destroying all resources..."
    terraform destroy -auto-approve || log "⚠️ Terraform destroy failed, some resources may remain"
    
    cd - > /dev/null || true
    
    log "✅ Infrastructure destruction completed"
}

# Main cleanup function
main() {
    echo -e "${BLUE}🧹 MLOps Infrastructure Cleanup${NC}"
    echo "================================="
    
    # Confirm cleanup
    confirm_cleanup
    
    # Get instance IP before cleanup
    INSTANCE_IP=$(get_instance_ip)
    
    # Stop containers gracefully
    if [ -n "$INSTANCE_IP" ]; then
        stop_containers "$INSTANCE_IP"
    fi
    
    # Destroy infrastructure
    destroy_infrastructure
    
    # Clean up local files
    cleanup_local
    
    echo ""
    echo -e "${GREEN}🎉 Cleanup completed successfully!${NC}"
    echo -e "${BLUE}📋 Cleanup logs saved to: $LOG_FILE${NC}"
}

# Run main function
main "$@"
