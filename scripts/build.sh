#!/bin/bash
set -e

echo "🐳 Building Docker images..."

# Build images
docker-compose build

echo "✅ Build completed!"
