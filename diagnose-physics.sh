#!/bin/bash
# Diagnostic script for physics simulation issues

echo "========================================="
echo "Physics Simulation Diagnostic Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Checking recent logs for GPU and physics activity..."
echo ""

# Check GPU initialization
echo -e "${YELLOW}1. GPU Initialization:${NC}"
grep -E "GPU: InitializeGPU|GPU initialization|GPU Device detected" logs/rust.log | tail -5
echo ""

# Check graph data
echo -e "${YELLOW}2. Graph Data:${NC}"
grep -E "GraphServiceActor: Sending.*nodes|Built initial graph|metadata entries" logs/rust.log | tail -5
echo ""

# Check physics settings
echo -e "${YELLOW}3. Physics Settings Updates:${NC}"
grep -E "Physics settings extraction|Sending physics to GPU|GPU: Received physics update" logs/rust.log | tail -5
echo ""

# Check simulation activity
echo -e "${YELLOW}4. Simulation Activity:${NC}"
grep -E "GPU: ComputeForces called|GPU kernel params|Starting physics simulation loop" logs/rust.log | tail -5
echo ""

# Check for errors
echo -e "${YELLOW}5. Recent Errors:${NC}"
grep -E "GPU NOT INITIALIZED|No nodes to compute|Failed to" logs/rust.log | tail -10
echo ""

# Check if simulation is sending updates
echo -e "${YELLOW}6. Position Updates:${NC}"
grep -E "BroadcastNodePositions|UpdateNodePositions" logs/rust.log | tail -5
echo ""

echo "========================================="
echo "Real-time monitoring (last 20 lines):"
echo "========================================="
tail -f logs/rust.log | grep -E "GPU:|GraphServiceActor:|Physics|simulation"