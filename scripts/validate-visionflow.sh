#!/bin/bash

# VisionFlow End-to-End Validation Script

echo "=== VisionFlow End-to-End Validation ==="
echo "This script helps validate the complete integration of the VisionFlow upgrade"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if services are running
check_service() {
    local service=$1
    local port=$2
    
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $service is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $service is not running on port $port"
        return 1
    fi
}

echo "1. Checking services..."
check_service "Backend (Actix-web)" 8080
check_service "Frontend (React)" 3000

echo ""
echo "2. Testing API endpoints..."

# Test health check
echo -n "   Testing /api/health... "
if curl -s http://localhost:8080/api/health > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Test bots endpoint
echo -n "   Testing /api/bots... "
response=$(curl -s http://localhost:8080/api/bots)
if [ ! -z "$response" ]; then
    echo -e "${GREEN}✓${NC}"
    agent_count=$(echo "$response" | jq '.agents | length' 2>/dev/null || echo "0")
    echo "      Found $agent_count agents"
else
    echo -e "${RED}✗${NC}"
fi

echo ""
echo "3. WebSocket connection test..."
echo "   To test WebSocket manually:"
echo "   - Open browser developer console"
echo "   - Check Network tab for WebSocket connection to ws://localhost:8080/ws"
echo "   - Verify 'bots-full-update' messages are received"

echo ""
echo "4. Manual UI validation checklist:"
echo ""
echo "   [ ] Open http://localhost:3000 in browser"
echo "   [ ] Navigate to the 3D visualization view"
echo "   [ ] Check that the right panel shows new UI components:"
echo "       [ ] System Health panel"
echo "       [ ] Activity Log panel (collapsed by default)"
echo "       [ ] Agent Details panel (collapsed by default)"
echo ""
echo "   [ ] Trigger swarm initialization:"
echo "       [ ] Click 'Initialize Swarm' button (if visible)"
echo "       [ ] Or use keyboard shortcut if implemented"
echo "       [ ] SwarmInitializationPrompt should appear with:"
echo "           [ ] Topology selection (mesh, hierarchical, ring, star)"
echo "           [ ] Maximum agents slider (3-20)"
echo "           [ ] Agent type checkboxes (12 types including queen)"
echo "           [ ] Neural enhancements toggle"
echo "           [ ] Task description textarea"
echo ""
echo "   [ ] Fill in the initialization form:"
echo "       [ ] Select 'hierarchical' topology"
echo "       [ ] Enable 'queen' agent type"
echo "       [ ] Add task: 'Build a REST API with user authentication'"
echo "       [ ] Click 'Spawn Hive Mind'"
echo ""
echo "   [ ] Verify backend processing:"
echo "       [ ] Check backend logs for 'initialize-swarm' API call"
echo "       [ ] Verify MCP tool request is sent"
echo "       [ ] Check for polling mechanism activation"
echo ""
echo "   [ ] Verify frontend updates:"
echo "       [ ] System Health panel updates with swarm metrics"
echo "       [ ] Activity Log shows agent spawn messages"
echo "       [ ] 3D visualization shows new agent nodes"
echo "       [ ] Agent nodes have enhanced visual features:"
echo "           [ ] Performance rings"
echo "           [ ] State indicators"
echo "           [ ] Capability badges"
echo "           [ ] Message flow lines"
echo ""
echo "   [ ] Test agent interactions:"
echo "       [ ] Click on an agent node in 3D view"
echo "       [ ] Verify Agent Details panel shows selected agent"
echo "       [ ] Check that agent metrics update in real-time"
echo ""
echo "5. Performance checks:"
echo "   - Monitor browser console for errors"
echo "   - Check network tab for excessive requests"
echo "   - Verify smooth 3D rendering without lag"
echo ""
echo "=== End of validation checklist ==="