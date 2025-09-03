#!/bin/bash
set -e

# Configuration
LOG_FILE="deployment.log"
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
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$STATUS_LOG"
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

# Check service health
check_service() {
    local url=$1
    local service_name=$2
    local timeout=10
    
    if curl -f -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $service_name: Healthy"
        return 0
    else
        echo -e "${RED}❌${NC} $service_name: Unhealthy"
        return 1
    fi
}

# Check VM connectivity
check_vm_connectivity() {
    local instance_ip=$1
    
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null azureuser@"$instance_ip" "echo 'VM accessible'" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} VM Connectivity: Accessible"
        return 0
    else
        echo -e "${RED}❌${NC} VM Connectivity: Not accessible"
        return 1
    fi
}

# Check Docker containers status
check_docker_containers() {
    local instance_ip=$1
    
    echo -e "${BLUE}🐳 Checking Docker containers on $instance_ip...${NC}"
    
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null azureuser@"$instance_ip" << 'EOF' 2>/dev/null || return 1
        cd /home/azureuser/mlpops
        docker-compose ps
        echo "--- Container logs summary ---"
        docker-compose logs --tail=5 mlops-api
        echo "--- End logs ---"
EOF
}

# Show deployment logs
show_deployment_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}📋 Recent deployment logs:${NC}"
        tail -20 "$LOG_FILE" | while IFS= read -r line; do
            echo "   $line"
        done
    else
        echo -e "${YELLOW}⚠️ No deployment logs found${NC}"
    fi
}

# Main status check
main() {
    echo -e "${BLUE}🔍 MLOps Deployment Status Check${NC}"
    echo "=================================="
    
    # Get instance IP
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        echo -e "${RED}❌ Could not retrieve instance IP from Terraform${NC}"
        echo -e "${YELLOW}💡 Run 'terraform apply' first or check terraform directory${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}📍 Instance IP: $INSTANCE_IP${NC}"
    echo ""
    
    # Check VM connectivity
    if check_vm_connectivity "$INSTANCE_IP"; then
        echo ""
        
        # Check Docker containers
        check_docker_containers "$INSTANCE_IP"
        echo ""
        
        # Check service health
        echo -e "${BLUE}🏥 Service Health Checks:${NC}"
        check_service "http://$INSTANCE_IP:8000/health" "API (Port 8000)"
        check_service "http://$INSTANCE_IP:8501" "Web UI (Port 8501)"
        
        echo ""
        echo -e "${BLUE}🌐 Service URLs:${NC}"
        echo -e "   📊 API: ${GREEN}http://$INSTANCE_IP:8000${NC}"
        echo -e "   🌐 Web UI: ${GREEN}http://$INSTANCE_IP:8501${NC}"
        
    else
        echo -e "${RED}❌ Cannot access VM. Services may not be running.${NC}"
        echo -e "${YELLOW}💡 Check if VM is running and SSH key is properly configured${NC}"
    fi
    
    echo ""
    show_deployment_logs
    
    echo ""
    echo -e "${BLUE}📊 Status check completed at $(date)${NC}"
}

# Run main function
main "$@" 