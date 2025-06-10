#!/bin/bash
set -e

echo "🧹 Cleaning up..."

# Stop local services
docker-compose down
docker system prune -f

# Cleanup AWS (if deployed)
if [ -d "terraform" ] && [ -f "terraform/terraform.tfstate" ]; then
    cd terraform
    terraform destroy -auto-approve
fi

echo "✅ Cleanup completed!"
