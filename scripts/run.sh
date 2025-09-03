#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Show usage
show_usage() {
    echo -e "${BLUE}🚀 MLOps Deployment Manager${NC}"
    echo "================================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo -e "  ${GREEN}deploy${NC}     - Deploy infrastructure and application"
    echo -e "  ${GREEN}redeploy${NC}   - Redeploy application only (keep infrastructure)"
    echo -e "  ${GREEN}status${NC}     - Check deployment status and health"
    echo -e "  ${GREEN}cleanup${NC}    - Destroy infrastructure and cleanup"
    echo -e "  ${GREEN}logs${NC}       - Show recent deployment logs"
    echo -e "  ${GREEN}help${NC}       - Show this help message"
    echo ""
    echo "Examples:"
echo "  $0 deploy    # Full deployment"
echo "  $0 redeploy  # Redeploy application only"
echo "  $0 status    # Check status"
echo "  $0 cleanup   # Cleanup everything"
    echo ""
}

# Show logs
show_logs() {
    if [ -f "deployment.log" ]; then
        echo -e "${BLUE}📋 Recent deployment logs:${NC}"
        tail -30 deployment.log
    else
        echo -e "${YELLOW}⚠️ No deployment logs found${NC}"
        echo "Run 'deploy' first to generate logs"
    fi
}

# Main function
main() {
    case "${1:-help}" in
        deploy)
            echo -e "${GREEN}🚀 Starting deployment...${NC}"
            ./scripts/deploy.sh
            ;;
        redeploy)
            echo -e "${YELLOW}🔄 Starting redeployment...${NC}"
            ./scripts/redeploy.sh
            ;;
        status)
            echo -e "${BLUE}🔍 Checking status...${NC}"
            ./scripts/status.sh
            ;;
        cleanup)
            echo -e "${RED}🧹 Starting cleanup...${NC}"
            ./scripts/cleanup.sh
            ;;
        logs)
            show_logs
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 