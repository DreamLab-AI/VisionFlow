#!/bin/bash

# Production Deployment Script for VisionFlow Multi-Agent System

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== VisionFlow Production Deployment ==="
echo ""

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then 
   echo -e "${YELLOW}Warning: Running as root is not recommended${NC}"
fi

# Check for required files
echo "1. Checking required files..."
REQUIRED_FILES=(
    "docker-compose.prod.yml"
    ".env.production"
    "Dockerfile"
    "client/Dockerfile"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗${NC} Missing required file: $file"
        if [ "$file" == ".env.production" ]; then
            echo "  Please copy .env.production.template to .env.production and configure it"
        fi
        exit 1
    else
        echo -e "${GREEN}✓${NC} Found $file"
    fi
done

# Validate environment variables
echo ""
echo "2. Validating environment configuration..."
source .env.production

# Check critical environment variables
CRITICAL_VARS=(
    "GITHUB_TOKEN"
    "ANTHROPIC_API_KEY"
    "MCP_RELAY_FALLBACK_TO_MOCK"
)

for var in "${CRITICAL_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}✗${NC} Missing critical environment variable: $var"
        exit 1
    else
        echo -e "${GREEN}✓${NC} $var is set"
    fi
done

# Verify MCP_RELAY_FALLBACK_TO_MOCK is false
if [ "$MCP_RELAY_FALLBACK_TO_MOCK" != "false" ]; then
    echo -e "${RED}✗${NC} MCP_RELAY_FALLBACK_TO_MOCK must be 'false' in production"
    exit 1
fi

# Build images
echo ""
echo "3. Building Docker images..."
echo "   This may take several minutes..."

# Build backend
echo "   Building backend image..."
docker build -t ghcr.io/your-org/multi-agent-backend:latest . || {
    echo -e "${RED}✗${NC} Backend build failed"
    exit 1
}
echo -e "${GREEN}✓${NC} Backend image built"

# Build frontend
echo "   Building frontend image..."
docker build -t ghcr.io/your-org/multi-agent-client:latest ./client || {
    echo -e "${RED}✗${NC} Frontend build failed"
    exit 1
}
echo -e "${GREEN}✓${NC} Frontend image built"

# Pull or build claude-flow image
echo "   Preparing claude-flow image..."
# TODO: Build or pull claude-flow image
echo -e "${YELLOW}!${NC} Claude-flow image needs to be built separately"

# Stop existing services
echo ""
echo "4. Stopping existing services..."
docker-compose -f docker-compose.prod.yml down || true

# Start services
echo ""
echo "5. Starting production services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo ""
echo "6. Waiting for services to be healthy..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    BACKEND_HEALTH=$(docker inspect multi-agent-backend-prod --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    CLIENT_HEALTH=$(docker inspect multi-agent-client-prod --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    
    if [ "$BACKEND_HEALTH" == "healthy" ] && [ "$CLIENT_HEALTH" == "healthy" ]; then
        echo -e "${GREEN}✓${NC} All services are healthy"
        break
    fi
    
    echo "   Backend: $BACKEND_HEALTH, Client: $CLIENT_HEALTH (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)"
    sleep 5
    ATTEMPT=$((ATTEMPT+1))
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}✗${NC} Services failed to become healthy"
    echo "   Check logs with: docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi

# Run post-deployment checks
echo ""
echo "7. Running post-deployment checks..."

# Check API health
echo -n "   Checking API health endpoint... "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/health || echo "000")
if [ "$HTTP_CODE" == "200" ]; then
    echo -e "${GREEN}✓${NC} API is responding"
else
    echo -e "${RED}✗${NC} API health check failed (HTTP $HTTP_CODE)"
fi

# Check WebSocket
echo -n "   Checking WebSocket endpoint... "
# Simple WebSocket test would go here
echo -e "${YELLOW}!${NC} Manual verification required"

# Check MCP connection
echo -n "   Checking MCP connection... "
MCP_STATUS=$(curl -s http://localhost:8080/api/bots | jq -r '.mcpConnected' 2>/dev/null || echo "error")
if [ "$MCP_STATUS" == "true" ]; then
    echo -e "${GREEN}✓${NC} MCP is connected"
elif [ "$MCP_STATUS" == "false" ]; then
    echo -e "${YELLOW}!${NC} MCP is not connected (may need initialization)"
else
    echo -e "${RED}✗${NC} Failed to check MCP status"
fi

# Summary
echo ""
echo "=== Deployment Summary ==="
echo ""
echo "Services deployed:"
echo "  - Backend:  http://localhost:8080"
echo "  - Frontend: http://localhost:3000"
echo "  - WebSocket: ws://localhost:8080/ws"
echo ""
echo "Next steps:"
echo "  1. Access the application at http://localhost:3000"
echo "  2. Initialize a swarm to test MCP integration"
echo "  3. Monitor logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "  4. Set up monitoring and alerting"
echo ""
echo -e "${GREEN}Deployment complete!${NC}"