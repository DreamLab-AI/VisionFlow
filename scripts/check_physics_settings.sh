#!/bin/bash

# VisionFlow Physics Settings Verification Script
# Quick check of current physics parameters

CONTAINER_IP="172.18.0.10"
PORT="4000"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Current VisionFlow Physics Settings${NC}"
echo "==================================="

curl -s "http://${CONTAINER_IP}:${PORT}/api/settings" | \
jq '.visualisation.graphs.visionflow.physics | {springK, repelK, maxVelocity, maxForce, damping, temperature}' 2>/dev/null || \
echo "Failed to retrieve settings"

echo -e "\n${GREEN}Physics settings check completed${NC}"