#!/bin/bash

# Comprehensive MCP Agent System Test - For VisionFlow Container
# Run this INSIDE the visionflow_container to test full agent lifecycle
#
# Usage: bash test_from_visionflow.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration - MUST use multi-agent-container hostname
MCP_HOST="multi-agent-container"
MCP_PORT="9500"
TIMEOUT=3

echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Comprehensive MCP Agent Test from VisionFlow         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${CYAN}[INFO]${NC} Testing MCP server at $MCP_HOST:$MCP_PORT"
echo -e "${CYAN}[INFO]${NC} Running from: $(hostname) ($(hostname -i))"
echo -e "${CYAN}[INFO]${NC} Test includes: swarm init, agent spawn, list, status, destroy"
echo

# Check dependencies
if ! command -v nc >/dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} netcat (nc) is required but not installed"
    echo "Install with: apt-get update && apt-get install -y netcat-openbsd"
    exit 1
fi

# Test counter
PASSED=0
FAILED=0

# Helper function to run a test
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo -n -e "${BLUE}[TEST]${NC} $test_name... "
    
    if eval "$test_cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Debug function for verbose output
debug_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo -e "${BLUE}[DEBUG TEST]${NC} $test_name"
    echo -e "${CYAN}Command:${NC} $test_cmd"
    echo -e "${CYAN}Output:${NC}"
    eval "$test_cmd" 2>&1 || true
    echo
}

# Test 1: DNS Resolution
test_dns() {
    getent hosts "$MCP_HOST" >/dev/null 2>&1 || \
    ping -c 1 "$MCP_HOST" >/dev/null 2>&1 || \
    host "$MCP_HOST" >/dev/null 2>&1
}

# Test 2: TCP Connectivity
test_tcp() {
    timeout 2 bash -c "echo > /dev/tcp/$MCP_HOST/$MCP_PORT" 2>/dev/null
}

# Test 3: MCP Initialize
test_initialize() {
    local response=$(
        (printf '{"jsonrpc":"2.0","id":"init-test","method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"VisionFlow-Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}\n'; sleep 0.2) | \
        timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null
    )
    echo "$response" | grep -q '"id":"init-test".*"result"'
}

# Test 4: Tools List
test_tools_list() {
    local response=$(
        (printf '{"jsonrpc":"2.0","id":"tools-test","method":"tools/list","params":{}}\n'; sleep 0.2) | \
        timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null
    )
    echo "$response" | grep -q '"id":"tools-test".*"result"'
}

# Test 5: Agent List (VisionFlow's main use case)
test_agent_list() {
    local response=$(
        (printf '{"jsonrpc":"2.0","id":"agent-test","method":"tools/call","params":{"name":"agent_list","arguments":{"filter":"all"}}}\n'; sleep 0.2) | \
        timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null
    )
    echo "$response" | grep -q '"id":"agent-test"'
}

# Function to send MCP request and show response
send_mcp_request() {
    local method="$1"
    local params="$2"
    local test_name="$3"
    local request_id="test-$(date +%s)-$$-$(uuidgen | cut -c1-8)"
    
    echo -e "${BLUE}[TEST]${NC} $test_name"
    
    local request="{\"jsonrpc\":\"2.0\",\"id\":\"$request_id\",\"method\":\"$method\",\"params\":$params}"
    echo -e "  ${YELLOW}→ Request:${NC} $method"
    echo -e "  ${CYAN}ID:${NC} $request_id"
    
    # Send request and keep connection open briefly for response
    local response=$(echo "$request" | nc -w 2 "$MCP_HOST" "$MCP_PORT" 2>&1 || true)
    
    # Parse JSON response (skip non-JSON lines)
    local json_found=false
    local line_count=0
    while IFS= read -r line; do
        line_count=$((line_count + 1))
        # Debug: Show first few lines if needed
        if [ $line_count -le 2 ] && [ "$LOG_LEVEL" = "debug" ]; then
            echo -e "  ${CYAN}[Line $line_count]:${NC} ${line:0:80}..."
        fi
        
        if [[ $line == '{"jsonrpc":'* ]]; then
            # Check if this response matches our request ID
            if [[ $line == *"\"id\":\"$request_id\""* ]]; then
                echo -e "  ${GREEN}← Response:${NC}"
                # Try to pretty print, fall back to raw
                if command -v jq >/dev/null 2>&1; then
                    echo "$line" | jq -C '.' 2>/dev/null || echo "$line"
                elif command -v python3 >/dev/null 2>&1; then
                    echo "$line" | python3 -m json.tool 2>/dev/null || echo "$line"
                else
                    echo "$line"
                fi
                json_found=true
                break
            elif [[ $line == *'"error":'* ]] && [[ $line == *"\"id\":\"$request_id\""* ]]; then
                echo -e "  ${RED}← Error:${NC}"
                if command -v jq >/dev/null 2>&1; then
                    echo "$line" | jq -C '.' 2>/dev/null || echo "$line"
                else
                    echo "$line"
                fi
                json_found=true
                break
            fi
        fi
    done <<< "$response"
    
    if [ "$json_found" = false ]; then
        echo -e "  ${RED}← No valid response${NC}"
        if [ -n "$response" ]; then
            echo -e "  ${YELLOW}First line:${NC} $(echo "$response" | head -1 | cut -c1-80)"
        else
            echo -e "  ${YELLOW}No output received${NC}"
        fi
    fi
    echo
}

# Run all tests
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Phase 1: Basic Connectivity${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo

run_test "DNS Resolution" test_dns
run_test "TCP Connectivity" test_tcp

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Basic connectivity failed!${NC}"
    echo "Troubleshooting:"
    echo "  - Check if multi-agent-container is running"
    echo "  - Verify network configuration"
    exit 1
fi

echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Phase 2: MCP Protocol Initialization${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo

send_mcp_request "initialize" '{
    "protocolVersion": "2024-11-05",
    "capabilities": {
        "tools": {"listChanged": true},
        "resources": {"subscribe": true, "listChanged": true}
    },
    "clientInfo": {
        "name": "visionflow-test",
        "version": "1.0.0"
    }
}' "Initialize MCP Connection"

echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Phase 3: Swarm Operations${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo

# Test swarm_init
send_mcp_request "tools/call" '{
    "name": "swarm_init",
    "arguments": {
        "topology": "mesh",
        "maxAgents": 5,
        "strategy": "balanced"
    }
}' "Initialize Swarm (mesh topology)"

sleep 1

# Test agent_spawn multiple times
send_mcp_request "tools/call" '{
    "name": "agent_spawn",
    "arguments": {
        "type": "coordinator",
        "name": "test-coordinator",
        "capabilities": ["orchestration", "monitoring"]
    }
}' "Spawn Coordinator Agent"

send_mcp_request "tools/call" '{
    "name": "agent_spawn",
    "arguments": {
        "type": "researcher",
        "name": "test-researcher",
        "capabilities": ["analysis", "search"]
    }
}' "Spawn Researcher Agent"

send_mcp_request "tools/call" '{
    "name": "agent_spawn",
    "arguments": {
        "type": "coder",
        "name": "test-coder",
        "capabilities": ["implementation", "testing"]
    }
}' "Spawn Coder Agent"

sleep 1

echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Phase 4: Agent Queries${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo

# Test agent_list
send_mcp_request "tools/call" '{
    "name": "agent_list",
    "arguments": {
        "filter": "all"
    }
}' "List All Agents"

# Test swarm_status
send_mcp_request "tools/call" '{
    "name": "swarm_status",
    "arguments": {
        "verbose": true
    }
}' "Get Swarm Status"

# Test agent_metrics
send_mcp_request "tools/call" '{
    "name": "agent_metrics",
    "arguments": {
        "metric": "all"
    }
}' "Get Agent Metrics"

echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Phase 5: Task Execution${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo

# Test task orchestration
send_mcp_request "tools/call" '{
    "name": "task_orchestrate",
    "arguments": {
        "task": "Test task: Analyze system performance and generate report",
        "strategy": "adaptive",
        "priority": "medium",
        "maxAgents": 3
    }
}' "Orchestrate Test Task"

sleep 2

echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Phase 6: Alternative Method Formats${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo

# Try different method name formats
echo -e "${BLUE}Testing direct method names (without tools/call wrapper):${NC}"
echo

send_mcp_request "agent_list" '{
    "filter": "all"
}' "Direct agent_list call"

send_mcp_request "mcp__claude-flow__agent_list" '{
    "filter": "all"
}' "Prefixed agent_list call"

echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Phase 7: Cleanup${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
echo

# Test swarm destruction
send_mcp_request "tools/call" '{
    "name": "swarm_destroy",
    "arguments": {}
}' "Destroy Swarm"

# Summary
echo
echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                     TEST COMPLETE                      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${YELLOW}Test Summary:${NC}"
echo "  - Basic connectivity test passed"
echo "  - MCP protocol initialization shown"
echo "  - Swarm lifecycle tested (init → spawn → list → destroy)"
echo "  - Multiple method formats tested"
echo
echo -e "${GREEN}Integration Notes for VisionFlow:${NC}"
echo "  1. Host: multi-agent-container (NOT localhost)"
echo "  2. Port: 9500"
echo "  3. Use 'tools/call' wrapper for agent methods"
echo "  4. Parse responses by matching request ID"
echo "  5. Skip non-JSON lines in responses"
echo
echo -e "${CYAN}Environment Variables for Rust:${NC}"
echo "  export MCP_HOST=multi-agent-container"
echo "  export MCP_TCP_PORT=9500"
echo
echo -e "${CYAN}Quick Test Commands:${NC}"
echo "  # List tools available:"
echo '  echo '"'"'{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}'"'"' | nc multi-agent-container 9500'
echo
echo "  # Initialize swarm:"
echo '  echo '"'"'{"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"swarm_init","arguments":{"topology":"mesh"}}}'"'"' | nc multi-agent-container 9500'
echo
echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"