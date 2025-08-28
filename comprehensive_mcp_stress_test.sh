#!/bin/bash

# =============================================================================
# Comprehensive MCP Connection Stress Test Script
# =============================================================================
# This script performs extensive testing of the MCP connection between
# visionflow_container and multi-agent-container with graceful failure handling
# 
# Usage: ./comprehensive_mcp_stress_test.sh [options]
#   -h, --host HOST       MCP host (default: multi-agent-container)
#   -p, --port PORT       MCP port (default: 9500)
#   -v, --verbose         Enable verbose output
#   -q, --quick           Run quick tests only
#   -s, --stress          Run stress tests (high load)
# =============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
MCP_HOST="${MCP_HOST:-multi-agent-container}"
MCP_PORT="${MCP_PORT:-9500}"
VERBOSE=false
QUICK_MODE=false
STRESS_MODE=false
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            MCP_HOST="$2"
            shift 2
            ;;
        -p|--port)
            MCP_PORT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -s|--stress)
            STRESS_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  -h, --host HOST    MCP host (default: multi-agent-container)"
            echo "  -p, --port PORT    MCP port (default: 9500)"
            echo "  -v, --verbose      Enable verbose output"
            echo "  -q, --quick        Run quick tests only"
            echo "  -s, --stress       Run stress tests (high load)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Utility functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
    ((WARNINGS++))
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
    ((TOTAL_TESTS++))
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# Test result tracking
track_result() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    if [ "$result" = "pass" ]; then
        log_success "$test_name: PASSED"
        [ -n "$details" ] && log_verbose "  Details: $details"
    elif [ "$result" = "fail" ]; then
        log_error "$test_name: FAILED"
        [ -n "$details" ] && echo "  Reason: $details"
    else
        log_warning "$test_name: WARNING"
        [ -n "$details" ] && echo "  Note: $details"
    fi
}

# =============================================================================
# CONNECTIVITY TESTS
# =============================================================================

test_dns_resolution() {
    log_test "Testing DNS resolution of $MCP_HOST"
    
    if getent hosts "$MCP_HOST" > /dev/null 2>&1; then
        local ip=$(getent hosts "$MCP_HOST" | awk '{ print $1 }')
        track_result "DNS Resolution" "pass" "Resolved to $ip"
    elif host "$MCP_HOST" > /dev/null 2>&1; then
        local ip=$(host "$MCP_HOST" | grep "has address" | head -1 | awk '{ print $4 }')
        track_result "DNS Resolution" "pass" "Resolved to $ip (via host command)"
    else
        track_result "DNS Resolution" "fail" "Cannot resolve $MCP_HOST"
        return 1
    fi
}

test_tcp_connectivity() {
    log_test "Testing TCP connectivity to $MCP_HOST:$MCP_PORT"
    
    if timeout 2 bash -c "echo > /dev/tcp/$MCP_HOST/$MCP_PORT" 2>/dev/null; then
        track_result "TCP Connectivity" "pass" "Port $MCP_PORT is open"
    elif nc -zv -w2 "$MCP_HOST" "$MCP_PORT" 2>/dev/null; then
        track_result "TCP Connectivity" "pass" "Port $MCP_PORT is open (via netcat)"
    else
        track_result "TCP Connectivity" "fail" "Cannot connect to $MCP_HOST:$MCP_PORT"
        return 1
    fi
}

test_ping_latency() {
    log_test "Testing network latency to $MCP_HOST"
    
    if ping -c 3 -W 2 "$MCP_HOST" > /tmp/ping_result 2>&1; then
        local avg_time=$(grep "avg" /tmp/ping_result | cut -d'/' -f5)
        if [ -n "$avg_time" ]; then
            track_result "Network Latency" "pass" "Average RTT: ${avg_time}ms"
        else
            track_result "Network Latency" "warning" "Ping succeeded but couldn't parse timing"
        fi
    else
        track_result "Network Latency" "warning" "ICMP blocked or host unreachable"
    fi
    rm -f /tmp/ping_result
}

# =============================================================================
# MCP PROTOCOL TESTS
# =============================================================================

test_mcp_initialize() {
    log_test "Testing MCP initialization handshake"
    
    # MCP server sends startup message first, then notifications, then response
    # We need to filter for the actual response with our ID
    local response=$(echo '{"jsonrpc":"2.0","id":"init-test","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"StressTest","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}' | \
        timeout 3 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"init-test"')
    
    if echo "$response" | grep -q '"result"'; then
        local version=$(echo "$response" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
        track_result "MCP Initialize" "pass" "Server version: $version"
        log_verbose "Full response: $response"
    else
        track_result "MCP Initialize" "fail" "Invalid or no response"
        log_verbose "Response: $response"
        return 1
    fi
}

test_mcp_tools_list() {
    log_test "Testing MCP tools listing"
    
    local response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"tools-test","method":"tools/list","params":{}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"tools-test"')
    
    if echo "$response" | grep -q '"tools":\['; then
        local tool_count=$(echo "$response" | grep -o '"name":"[^"]*"' | wc -l)
        track_result "Tools List" "pass" "Found $tool_count tools"
        
        # List some key tools if verbose
        if [ "$VERBOSE" = true ]; then
            echo "$response" | grep -o '"name":"[^"]*"' | head -10 | while read tool; do
                log_verbose "  - $(echo $tool | cut -d'"' -f4)"
            done
        fi
    else
        track_result "Tools List" "fail" "Could not retrieve tools list"
        return 1
    fi
}

test_swarm_init() {
    log_test "Testing swarm initialization (direct method)"
    
    local swarm_id="test-swarm-$$-$(date +%s)"
    local response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"'$swarm_id'","method":"swarm_init","params":{"topology":"mesh","maxAgents":4,"strategy":"balanced"}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep "\"id\":\"$swarm_id\"")
    
    # The response has nested JSON in content[0].text, check for result field first
    if echo "$response" | grep -q '"result".*"content"'; then
        # Extract the text content which contains the actual swarm data
        if echo "$response" | grep -q '"success":.*true.*"swarmId"'; then
            # Parse swarmId from the nested JSON - it's in the text field
            local extracted_id=$(echo "$response" | sed -n 's/.*"swarmId"[[:space:]]*:[[:space:]]*"\\*\([^"\\]*\).*/\1/p' | head -1)
            if [ -n "$extracted_id" ]; then
                track_result "Swarm Init (Direct)" "pass" "Created swarm: $extracted_id"
                echo "$extracted_id" > /tmp/last_swarm_id
            else
                track_result "Swarm Init (Direct)" "pass" "Swarm created (ID parsing failed)"
                log_verbose "Response: $response"
            fi
        else
            track_result "Swarm Init (Direct)" "fail" "Swarm creation failed"
            log_verbose "Response: $response"
        fi
    else
        track_result "Swarm Init (Direct)" "fail" "Could not initialize swarm"
        log_verbose "Response: $response"
    fi
}

test_swarm_init_tools_call() {
    log_test "Testing swarm initialization (tools/call method)"
    
    local response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"swarm-tools-test","method":"tools/call","params":{"name":"swarm_init","arguments":{"topology":"hierarchical","maxAgents":6,"strategy":"adaptive"}}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"swarm-tools-test"')
    
    if echo "$response" | grep -q '"swarmId"'; then
        local swarm_id=$(echo "$response" | grep -o '"swarmId":"[^"]*"' | cut -d'"' -f4)
        track_result "Swarm Init (tools/call)" "pass" "Created swarm: $swarm_id"
        echo "$swarm_id" > /tmp/last_swarm_id_tools
    else
        track_result "Swarm Init (tools/call)" "warning" "Method may not be supported"
    fi
}

test_agent_spawn() {
    log_test "Testing agent spawning"
    
    local swarm_id=""
    if [ -f /tmp/last_swarm_id ]; then
        swarm_id=$(cat /tmp/last_swarm_id)
    fi
    
    if [ -z "$swarm_id" ]; then
        track_result "Agent Spawn" "warning" "No swarm ID available, skipping"
        return
    fi
    
    local response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"agent-test","method":"agent_spawn","params":{"type":"researcher","swarmId":"'$swarm_id'","capabilities":["analysis","search"]}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"agent-test"')
    
    if echo "$response" | grep -q '"success":true.*"agentId"'; then
        local agent_id=$(echo "$response" | grep -o '"agentId":"[^"]*"' | cut -d'"' -f4)
        track_result "Agent Spawn" "pass" "Created agent: $agent_id"
    else
        track_result "Agent Spawn" "fail" "Could not spawn agent"
    fi
}

test_swarm_status() {
    log_test "Testing swarm status query"
    
    local swarm_id=""
    if [ -f /tmp/last_swarm_id ]; then
        swarm_id=$(cat /tmp/last_swarm_id)
    fi
    
    if [ -z "$swarm_id" ]; then
        track_result "Swarm Status" "warning" "No swarm ID available, skipping"
        return
    fi
    
    local response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"status-test","method":"swarm_status","params":{"swarmId":"'$swarm_id'"}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"status-test"')
    
    if echo "$response" | grep -q '"success":true'; then
        track_result "Swarm Status" "pass" "Successfully queried swarm status"
        if [ "$VERBOSE" = true ]; then
            local agent_count=$(echo "$response" | grep -o '"agentCount":[0-9]*' | cut -d':' -f2)
            local task_count=$(echo "$response" | grep -o '"taskCount":[0-9]*' | cut -d':' -f2)
            log_verbose "  Agents: $agent_count, Tasks: $task_count"
        fi
    else
        track_result "Swarm Status" "fail" "Could not query swarm status"
    fi
}

test_memory_operations() {
    log_test "Testing memory store operations"
    
    local test_key="test-key-$$"
    local test_value="test-value-$(date +%s)"
    
    # Test store
    local store_response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"mem-store","method":"memory_usage","params":{"action":"store","key":"'$test_key'","value":"'$test_value'","namespace":"test"}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"mem-store"')
    
    if echo "$store_response" | grep -q '"success":true'; then
        log_verbose "Memory store succeeded"
        
        # Test retrieve
        local retrieve_response=$((
            echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
            sleep 0.5
            echo '{"jsonrpc":"2.0","id":"mem-retrieve","method":"memory_usage","params":{"action":"retrieve","key":"'$test_key'","namespace":"test"}}'
        ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"mem-retrieve"')
        
        if echo "$retrieve_response" | grep -q "$test_value"; then
            track_result "Memory Operations" "pass" "Store and retrieve working"
        else
            track_result "Memory Operations" "warning" "Store succeeded but retrieve failed"
        fi
    else
        track_result "Memory Operations" "fail" "Could not store to memory"
    fi
}

test_task_orchestration() {
    log_test "Testing task orchestration"
    
    local response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"task-test","method":"task_orchestrate","params":{"task":"Analyze system performance metrics","strategy":"parallel","priority":"medium","maxAgents":3}}'
    ) | timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"task-test"')
    
    if echo "$response" | grep -q '"success":true.*"taskId"'; then
        local task_id=$(echo "$response" | grep -o '"taskId":"[^"]*"' | cut -d'"' -f4)
        track_result "Task Orchestration" "pass" "Created task: $task_id"
    else
        track_result "Task Orchestration" "fail" "Could not orchestrate task"
    fi
}

# =============================================================================
# STRESS TESTS
# =============================================================================

test_concurrent_connections() {
    if [ "$STRESS_MODE" != true ]; then
        return
    fi
    
    log_test "Testing concurrent connections (10 simultaneous)"
    
    local success_count=0
    local pids=""
    
    for i in {1..10}; do
        (
            echo '{"jsonrpc":"2.0","id":"concurrent-'$i'","method":"tools/list","params":{}}' | \
            timeout 5 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | \
            grep -q '"tools":\[' && echo "SUCCESS"
        ) > /tmp/concurrent_$i.txt &
        pids="$pids $!"
    done
    
    # Wait for all background jobs
    for pid in $pids; do
        wait $pid
    done
    
    # Count successes
    for i in {1..10}; do
        if [ -f /tmp/concurrent_$i.txt ] && grep -q "SUCCESS" /tmp/concurrent_$i.txt; then
            ((success_count++))
        fi
        rm -f /tmp/concurrent_$i.txt
    done
    
    if [ $success_count -eq 10 ]; then
        track_result "Concurrent Connections" "pass" "All 10 connections succeeded"
    elif [ $success_count -ge 7 ]; then
        track_result "Concurrent Connections" "warning" "$success_count/10 connections succeeded"
    else
        track_result "Concurrent Connections" "fail" "Only $success_count/10 connections succeeded"
    fi
}

test_rapid_requests() {
    if [ "$STRESS_MODE" != true ]; then
        return
    fi
    
    log_test "Testing rapid sequential requests (50 requests)"
    
    local success_count=0
    local start_time=$(date +%s)
    
    for i in {1..50}; do
        local response=$(echo '{"jsonrpc":"2.0","id":"rapid-'$i'","method":"tools/list","params":{}}' | \
            timeout 2 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | head -1)
        
        if echo "$response" | grep -q '"tools":\['; then
            ((success_count++))
        fi
        
        # Small delay to prevent overwhelming
        sleep 0.05
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $success_count -eq 50 ]; then
        track_result "Rapid Requests" "pass" "All 50 requests succeeded in ${duration}s"
    elif [ $success_count -ge 45 ]; then
        track_result "Rapid Requests" "warning" "$success_count/50 succeeded in ${duration}s"
    else
        track_result "Rapid Requests" "fail" "Only $success_count/50 succeeded"
    fi
}

test_large_payload() {
    if [ "$STRESS_MODE" != true ]; then
        return
    fi
    
    log_test "Testing large payload handling"
    
    # Create a large task description (about 10KB)
    local large_text=$(head -c 10000 < /dev/zero | tr '\0' 'A')
    
    local response=$((
        echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":{"major":2024,"minor":11,"patch":5},"clientInfo":{"name":"Test","version":"1.0.0"},"capabilities":{"tools":{"listChanged":true}}}}'
        sleep 0.5
        echo '{"jsonrpc":"2.0","id":"large-test","method":"task_orchestrate","params":{"task":"'$large_text'","strategy":"parallel"}}'
    ) | timeout 10 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | grep '"id":"large-test"')
    
    if echo "$response" | grep -q '"id":"large-test"'; then
        if echo "$response" | grep -q '"success":true'; then
            track_result "Large Payload" "pass" "10KB payload handled successfully"
        else
            track_result "Large Payload" "warning" "Server responded but rejected large payload"
        fi
    else
        track_result "Large Payload" "fail" "No response for large payload"
    fi
}

test_connection_recovery() {
    log_test "Testing connection recovery after disconnect"
    
    # First connection
    local response1=$(echo '{"jsonrpc":"2.0","id":"recovery-1","method":"tools/list","params":{}}' | \
        timeout 3 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | head -1)
    
    if echo "$response1" | grep -q '"tools":\['; then
        log_verbose "First connection successful"
        
        # Wait a moment
        sleep 1
        
        # Second connection (simulating reconnect)
        local response2=$(echo '{"jsonrpc":"2.0","id":"recovery-2","method":"tools/list","params":{}}' | \
            timeout 3 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | head -1)
        
        if echo "$response2" | grep -q '"tools":\['; then
            track_result "Connection Recovery" "pass" "Reconnection successful"
        else
            track_result "Connection Recovery" "fail" "Could not reconnect"
        fi
    else
        track_result "Connection Recovery" "fail" "Initial connection failed"
    fi
}

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

test_invalid_json() {
    log_test "Testing invalid JSON handling"
    
    local response=$(echo 'this is not json' | timeout 2 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null)
    
    if echo "$response" | grep -q '"error"'; then
        track_result "Invalid JSON" "pass" "Server properly rejected invalid JSON"
    else
        track_result "Invalid JSON" "warning" "Server response unclear for invalid JSON"
    fi
}

test_unknown_method() {
    log_test "Testing unknown method handling"
    
    local response=$(echo '{"jsonrpc":"2.0","id":"unknown-test","method":"this_method_does_not_exist","params":{}}' | \
        timeout 3 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null)
    
    if echo "$response" | grep -q '"error".*"Method not found"'; then
        track_result "Unknown Method" "pass" "Server properly rejected unknown method"
    else
        track_result "Unknown Method" "warning" "Server response unclear for unknown method"
    fi
}

test_timeout_handling() {
    log_test "Testing timeout handling"
    
    # Send partial request and wait
    (
        echo -n '{"jsonrpc":"2.0","id":"timeout-test","method":"tools/list"'
        sleep 6
    ) | timeout 8 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null > /tmp/timeout_response
    
    if [ -s /tmp/timeout_response ]; then
        track_result "Timeout Handling" "warning" "Server responded despite incomplete request"
    else
        track_result "Timeout Handling" "pass" "Connection properly timed out"
    fi
    rm -f /tmp/timeout_response
}

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

test_response_time() {
    log_test "Testing average response time"
    
    local total_time=0
    local count=0
    
    for i in {1..5}; do
        local start=$(date +%s%N)
        local response=$(echo '{"jsonrpc":"2.0","id":"perf-'$i'","method":"tools/list","params":{}}' | \
            timeout 3 nc "$MCP_HOST" "$MCP_PORT" 2>/dev/null | head -1)
        local end=$(date +%s%N)
        
        if echo "$response" | grep -q '"tools":\['; then
            local duration=$((($end - $start) / 1000000)) # Convert to milliseconds
            total_time=$((total_time + duration))
            ((count++))
            log_verbose "Request $i took ${duration}ms"
        fi
    done
    
    if [ $count -gt 0 ]; then
        local avg_time=$((total_time / count))
        if [ $avg_time -lt 100 ]; then
            track_result "Response Time" "pass" "Average response time: ${avg_time}ms (excellent)"
        elif [ $avg_time -lt 500 ]; then
            track_result "Response Time" "pass" "Average response time: ${avg_time}ms (good)"
        elif [ $avg_time -lt 1000 ]; then
            track_result "Response Time" "warning" "Average response time: ${avg_time}ms (slow)"
        else
            track_result "Response Time" "fail" "Average response time: ${avg_time}ms (too slow)"
        fi
    else
        track_result "Response Time" "fail" "Could not measure response time"
    fi
}

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

main() {
    echo "============================================================================="
    echo "MCP Connection Comprehensive Stress Test"
    echo "============================================================================="
    echo "Target: $MCP_HOST:$MCP_PORT"
    echo "Mode: $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full") $([ "$STRESS_MODE" = true ] && echo "+ Stress" || echo "")"
    echo "Started: $(date)"
    echo "============================================================================="
    echo ""
    
    # Basic connectivity tests
    echo -e "${CYAN}[PHASE 1]${NC} Testing Basic Connectivity"
    echo "-----------------------------------------------------------------------------"
    test_dns_resolution
    test_tcp_connectivity
    test_ping_latency
    echo ""
    
    # MCP protocol tests
    echo -e "${CYAN}[PHASE 2]${NC} Testing MCP Protocol"
    echo "-----------------------------------------------------------------------------"
    test_mcp_initialize
    test_mcp_tools_list
    echo ""
    
    if [ "$QUICK_MODE" != true ]; then
        # Core functionality tests
        echo -e "${CYAN}[PHASE 3]${NC} Testing Core Functionality"
        echo "-----------------------------------------------------------------------------"
        test_swarm_init
        test_swarm_init_tools_call
        test_agent_spawn
        test_swarm_status
        test_memory_operations
        test_task_orchestration
        echo ""
        
        # Edge cases and error handling
        echo -e "${CYAN}[PHASE 4]${NC} Testing Edge Cases"
        echo "-----------------------------------------------------------------------------"
        test_invalid_json
        test_unknown_method
        test_timeout_handling
        test_connection_recovery
        echo ""
        
        # Performance tests
        echo -e "${CYAN}[PHASE 5]${NC} Testing Performance"
        echo "-----------------------------------------------------------------------------"
        test_response_time
        echo ""
    fi
    
    # Stress tests (if enabled)
    if [ "$STRESS_MODE" = true ]; then
        echo -e "${CYAN}[PHASE 6]${NC} Running Stress Tests"
        echo "-----------------------------------------------------------------------------"
        test_concurrent_connections
        test_rapid_requests
        test_large_payload
        echo ""
    fi
    
    # Cleanup
    rm -f /tmp/last_swarm_id /tmp/last_swarm_id_tools
    
    # Final report
    echo "============================================================================="
    echo "TEST RESULTS SUMMARY"
    echo "============================================================================="
    echo "Total Tests:    $TOTAL_TESTS"
    echo -e "Passed:         ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed:         ${RED}$FAILED_TESTS${NC}"
    echo -e "Warnings:       ${YELLOW}$WARNINGS${NC}"
    
    local success_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    fi
    
    echo "Success Rate:   $success_rate%"
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}✅ ALL TESTS PASSED - MCP connection is fully operational!${NC}"
        exit 0
    elif [ $success_rate -ge 80 ]; then
        echo -e "${YELLOW}⚠️  MOSTLY PASSING - MCP connection is operational with some issues${NC}"
        exit 1
    elif [ $success_rate -ge 50 ]; then
        echo -e "${YELLOW}⚠️  PARTIAL SUCCESS - MCP connection has significant issues${NC}"
        exit 2
    else
        echo -e "${RED}❌ CRITICAL FAILURE - MCP connection is not working properly${NC}"
        exit 3
    fi
}

# Run main function
main