#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# Local deployment
echo "📦 Starting local services..."
docker-compose down
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Test local deployment
echo "🧪 Testing local deployment..."
curl -f http://localhost:8000/health || echo "❌ API health check failed"

echo "✅ Local deployment completed!"
echo "📊 API: http://localhost:8000"
echo "🌐 Web: http://localhost:8501"

# AWS deployment (optional)
read -p "Deploy to AWS? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "☁️ Deploying to AWS..."
    cd terraform
    terraform init
    terraform apply -auto-approve
    
    INSTANCE_IP=$(terraform output -raw instance_ip)
    echo "🎉 AWS deployment completed!"
    echo "📊 API: http://$INSTANCE_IP:8000"
    echo "🌐 Web: http://$INSTANCE_IP:8501"
fi
