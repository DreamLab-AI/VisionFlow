#!/bin/bash

# Test script to verify physics settings updates are working
# Run this after rebuilding the server with the fixes

echo "=== Physics Settings Update Test ==="
echo "This script will test if physics settings updates are propagating correctly"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check logs for propagation message
check_propagation() {
    echo -e "${YELLOW}Checking for physics propagation in logs...${NC}"
    if tail -100 /workspace/ext/logs/rust-error.log | grep -q "Physics setting changed, propagating to GPU actors"; then
        echo -e "${GREEN}✓ Physics propagation message found${NC}"
        return 0
    else
        echo -e "${RED}✗ No physics propagation message found${NC}"
        return 1
    fi
}

# Function to check current physics values in logs
check_physics_values() {
    echo -e "${YELLOW}Current physics values in server:${NC}"
    tail -20 /workspace/ext/logs/rust-error.log | grep -E "repel_k|spring_k|max_velocity" | tail -3
}

# Function to update a physics setting via API
update_physics_setting() {
    local path=$1
    local value=$2
    echo -e "${YELLOW}Updating $path to $value via API...${NC}"
    
    curl -X POST http://localhost:4000/api/settings/batch \
        -H "Content-Type: application/json" \
        -d "{\"updates\":[{\"path\":\"$path\",\"value\":$value}]}" \
        -s -o /dev/null -w "%{http_code}"
}

# Test 1: Check initial state
echo "=== Test 1: Initial State ==="
check_physics_values
echo ""

# Test 2: Update repelK
echo "=== Test 2: Update repelK ==="
http_code=$(update_physics_setting "visualisation.graphs.logseq.physics.repelK" "100.0")
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ API request successful (HTTP $http_code)${NC}"
else
    echo -e "${RED}✗ API request failed (HTTP $http_code)${NC}"
fi
sleep 2
check_propagation
check_physics_values
echo ""

# Test 3: Update springK
echo "=== Test 3: Update springK ==="
http_code=$(update_physics_setting "visualisation.graphs.logseq.physics.springK" "5.0")
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ API request successful (HTTP $http_code)${NC}"
else
    echo -e "${RED}✗ API request failed (HTTP $http_code)${NC}"
fi
sleep 2
check_propagation
check_physics_values
echo ""

# Test 4: Check settings.yaml was updated
echo "=== Test 4: Verify settings.yaml ==="
echo -e "${YELLOW}Current values in settings.yaml:${NC}"
grep -E "repelK|springK" /workspace/ext/data/settings.yaml | grep -A1 logseq | head -2
echo ""

# Summary
echo "=== Test Summary ==="
echo "If all tests passed, you should see:"
echo "1. ✓ API requests returning HTTP 200"
echo "2. ✓ Physics propagation messages in logs"
echo "3. ✓ Updated physics values being used by the server"
echo "4. ✓ settings.yaml containing the new values"
echo ""
echo "If tests failed, the server may need to be rebuilt with the fixes."