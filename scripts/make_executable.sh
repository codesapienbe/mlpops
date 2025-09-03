#!/bin/bash
# Make all deployment scripts executable
chmod +x scripts/*.sh
echo "✅ All scripts are now executable!"
echo ""
echo "Available commands:"
echo "  ./scripts/run.sh deploy     - Deploy everything"
echo "  ./scripts/run.sh redeploy   - Redeploy application only"
echo "  ./scripts/run.sh status     - Check status"
echo "  ./scripts/run.sh cleanup    - Cleanup resources"
echo "  ./scripts/run.sh logs       - View logs"
echo "  ./scripts/run.sh help       - Show help" 