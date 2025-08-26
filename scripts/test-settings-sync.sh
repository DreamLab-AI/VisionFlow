#!/bin/bash
# Integration test script for settings sync functionality
# Tests the complete settings sync flow with bloom field validation

set -e

# Configuration
SERVER_URL="http://localhost:8080"
TEST_TIMEOUT=30
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m' # No Color

# Test counters
TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

# Helper functions
log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_NC} $1"
}

log_success() {
    echo -e "${COLOR_GREEN}[PASS]${COLOR_NC} $1"
    ((PASS_COUNT++))
}

log_error() {
    echo -e "${COLOR_RED}[FAIL]${COLOR_NC} $1"
    ((FAIL_COUNT++))
}

log_warning() {
    echo -e "${COLOR_YELLOW}[WARN]${COLOR_NC} $1"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((TEST_COUNT++))
    log_info "Running test: $test_name"
    
    if eval "$test_command"; then
        log_success "$test_name"
    else
        log_error "$test_name"
    fi
    echo
}

# Test helper functions
make_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local expected_status="$4"
    
    local url="${SERVER_URL}${endpoint}"
    local response
    local status_code
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$url" || echo "000")
    elif [ "$method" = "POST" ]; then
        if [ -n "$data" ]; then
            response=$(curl -s -w "\n%{http_code}" -X POST \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$url" || echo "000")
        else
            response=$(curl -s -w "\n%{http_code}" -X POST "$url" || echo "000")
        fi
    fi
    
    status_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "$expected_status" ]; then
        return 0
    else
        echo "Expected status $expected_status, got $status_code"
        echo "Response: $response_body"
        return 1
    fi
}

validate_json() {
    local json_string="$1"
    echo "$json_string" | jq . > /dev/null 2>&1
}

validate_settings_structure() {
    local response="$1"
    
    # Check for required top-level keys
    echo "$response" | jq -e '.visualisation' > /dev/null || return 1
    echo "$response" | jq -e '.system' > /dev/null || return 1
    echo "$response" | jq -e '.xr' > /dev/null || return 1
    
    # Check for bloom/glow settings
    echo "$response" | jq -e '.visualisation.glow' > /dev/null || return 1
    echo "$response" | jq -e '.visualisation.glow.enabled' > /dev/null || return 1
    echo "$response" | jq -e '.visualisation.glow.intensity' > /dev/null || return 1
    echo "$response" | jq -e '.visualisation.glow.nodeGlowStrength' > /dev/null || return 1
    
    # Check for physics settings in both graphs
    echo "$response" | jq -e '.visualisation.graphs.logseq.physics' > /dev/null || return 1
    echo "$response" | jq -e '.visualisation.graphs.visionflow.physics' > /dev/null || return 1
    
    return 0
}

# Test data
VALID_BLOOM_UPDATE='{
  "visualisation": {
    "glow": {
      "enabled": true,
      "intensity": 2.5,
      "radius": 0.9,
      "threshold": 0.2,
      "baseColor": "#ff6b6b",
      "nodeGlowStrength": 3.5,
      "edgeGlowStrength": 4.0
    },
    "graphs": {
      "logseq": {
        "physics": {
          "springK": 0.15,
          "repelK": 2.5,
          "damping": 0.9,
          "iterations": 75
        }
      }
    }
  }
}'

INVALID_BLOOM_UPDATE='{
  "visualisation": {
    "glow": {
      "intensity": -1.0
    }
  }
}'

INVALID_PHYSICS_UPDATE='{
  "damping": 2.0,
  "iterations": -10
}'

VALID_PHYSICS_UPDATE='{
  "springK": 0.2,
  "repelK": 3.0,
  "damping": 0.95,
  "maxVelocity": 10.0,
  "iterations": 100
}'

# Check if server is running
check_server() {
    log_info "Checking if server is running at $SERVER_URL"
    if curl -s -f "$SERVER_URL/health" > /dev/null 2>&1; then
        log_success "Server is responding"
        return 0
    else
        log_error "Server is not responding at $SERVER_URL"
        log_warning "Please ensure the server is running with: cargo run --bin server"
        return 1
    fi
}

# Core API tests
test_get_settings() {
    local response
    response=$(curl -s "$SERVER_URL/api/settings")
    local status=$?
    
    if [ $status -eq 0 ] && validate_json "$response" && validate_settings_structure "$response"; then
        return 0
    else
        echo "Failed to get valid settings response"
        return 1
    fi
}

test_update_bloom_settings_valid() {
    make_request "POST" "/api/settings" "$VALID_BLOOM_UPDATE" "200"
}

test_update_bloom_settings_invalid() {
    make_request "POST" "/api/settings" "$INVALID_BLOOM_UPDATE" "400"
}

test_physics_endpoint() {
    make_request "POST" "/api/physics/update" "$VALID_PHYSICS_UPDATE" "200"
}

test_physics_validation() {
    make_request "POST" "/api/physics/update" "$INVALID_PHYSICS_UPDATE" "400"
}

test_settings_reset() {
    make_request "POST" "/api/settings/reset" "" "200"
}

test_validation_stats() {
    make_request "GET" "/api/settings/validation/stats" "" "200"
}

# Rate limiting tests
test_rate_limiting() {
    log_info "Testing rate limiting with rapid requests"
    local rate_limited=false
    
    for i in {1..15}; do
        local response
        response=$(curl -s -w "%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d '{"test": '$i'}' \
            "$SERVER_URL/api/settings")
        
        local status_code
        status_code=$(echo "$response" | tail -c 4)
        
        if [ "$status_code" = "429" ]; then
            rate_limited=true
            break
        fi
        
        sleep 0.1
    done
    
    if [ "$rate_limited" = true ]; then
        return 0
    else
        echo "Rate limiting not triggered after 15 requests"
        return 1
    fi
}

# Bidirectional sync test
test_bidirectional_sync() {
    log_info "Testing bidirectional sync"
    
    # Step 1: Update settings
    local update_response
    update_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$VALID_BLOOM_UPDATE" \
        "$SERVER_URL/api/settings")
    
    if ! validate_json "$update_response"; then
        echo "Invalid JSON response from update"
        return 1
    fi
    
    # Step 2: Fetch settings and verify changes
    local fetch_response
    fetch_response=$(curl -s "$SERVER_URL/api/settings")
    
    if ! validate_json "$fetch_response"; then
        echo "Invalid JSON response from fetch"
        return 1
    fi
    
    # Verify the update was applied
    local intensity
    intensity=$(echo "$fetch_response" | jq -r '.visualisation.glow.intensity')
    
    if [ "$intensity" = "2.5" ]; then
        return 0
    else
        echo "Expected intensity 2.5, got $intensity"
        return 1
    fi
}

# Nostr authentication tests (basic structure)
test_nostr_auth_endpoint() {
    local nostr_event='{
        "id": "test_event_id_12345",
        "pubkey": "test_pubkey_abcdef1234567890",
        "content": "Authenticate to LogseqSpringThing",
        "sig": "test_signature_fedcba0987654321",
        "created_at": 1640995200,
        "kind": 22242,
        "tags": [["relay", "wss://relay.damus.io"], ["challenge", "test_challenge_uuid"]]
    }'
    
    # This will likely return 401 with test data, but we test the endpoint exists
    local response
    response=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$nostr_event" \
        "$SERVER_URL/auth/nostr")
    
    local status_code
    status_code=$(echo "$response" | tail -c 4)
    
    # Accept both 200 (valid signature) or 401 (invalid signature)
    if [ "$status_code" = "200" ] || [ "$status_code" = "401" ]; then
        return 0
    else
        echo "Expected 200 or 401, got $status_code"
        return 1
    fi
}

test_nostr_verify_endpoint() {
    local verify_payload='{
        "pubkey": "test_pubkey_abcdef1234567890",
        "token": "test_token_12345"
    }'
    
    make_request "POST" "/auth/nostr/verify" "$verify_payload" "200"
}

# Error handling tests
test_malformed_json() {
    local response
    response=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "{ invalid json }" \
        "$SERVER_URL/api/settings")
    
    local status_code
    status_code=$(echo "$response" | tail -c 4)
    
    if [ "$status_code" = "400" ]; then
        return 0
    else
        echo "Expected 400 for malformed JSON, got $status_code"
        return 1
    fi
}

test_large_payload() {
    log_info "Testing large payload handling"
    
    # Create a large JSON payload
    local large_data='{
        "visualisation": {
            "glow": {
                "massiveData": "'
    
    # Add 100KB of data
    large_data+=$(python3 -c "print('x' * 100000)")
    large_data+='"}
        }
    }'
    
    local response
    response=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$large_data" \
        "$SERVER_URL/api/settings")
    
    local status_code
    status_code=$(echo "$response" | tail -c 4)
    
    # Should reject large payloads
    if [ "$status_code" = "413" ] || [ "$status_code" = "400" ]; then
        return 0
    else
        echo "Expected 413 or 400 for large payload, got $status_code"
        return 1
    fi
}

# Performance tests
test_response_time() {
    log_info "Testing response time"
    
    local start_time
    local end_time
    local duration
    
    start_time=$(date +%s%3N)
    curl -s "$SERVER_URL/api/settings" > /dev/null
    end_time=$(date +%s%3N)
    
    duration=$((end_time - start_time))
    
    if [ $duration -lt 1000 ]; then  # Less than 1 second
        log_success "Response time: ${duration}ms (< 1000ms)"
        return 0
    else
        echo "Response time too slow: ${duration}ms"
        return 1
    fi
}

test_concurrent_requests() {
    log_info "Testing concurrent request handling"
    
    local pids=()
    local results=()
    
    # Launch 5 concurrent requests
    for i in {1..5}; do
        {
            curl -s -w "%{http_code}" -X POST \
                -H "Content-Type: application/json" \
                -d '{"test": '$i'}' \
                "$SERVER_URL/api/settings" > "/tmp/concurrent_test_$i.result"
        } &
        pids+=($!)
    done
    
    # Wait for all requests to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    # Check results
    local all_success=true
    for i in {1..5}; do
        local result
        result=$(cat "/tmp/concurrent_test_$i.result" 2>/dev/null || echo "000")
        local status_code
        status_code=$(echo "$result" | tail -c 4)
        
        if [ "$status_code" != "200" ] && [ "$status_code" != "429" ]; then
            all_success=false
            echo "Request $i failed with status $status_code"
        fi
        
        rm -f "/tmp/concurrent_test_$i.result"
    done
    
    if [ "$all_success" = true ]; then
        return 0
    else
        echo "Some concurrent requests failed"
        return 1
    fi
}

# Main test execution
main() {
    log_info "Starting comprehensive settings sync integration tests"
    log_info "Server URL: $SERVER_URL"
    echo
    
    # Check prerequisites
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed"
        exit 1
    fi
    
    # Check server availability
    if ! check_server; then
        exit 1
    fi
    
    echo
    log_info "Running core API tests..."
    echo
    
    # Core API functionality
    run_test "GET /api/settings" "test_get_settings"
    run_test "POST /api/settings (valid bloom)" "test_update_bloom_settings_valid"
    run_test "POST /api/settings (invalid bloom)" "test_update_bloom_settings_invalid"
    run_test "POST /api/physics/update (valid)" "test_physics_endpoint"
    run_test "POST /api/physics/update (invalid)" "test_physics_validation"
    run_test "POST /api/settings/reset" "test_settings_reset"
    run_test "GET /api/settings/validation/stats" "test_validation_stats"
    
    echo
    log_info "Running advanced tests..."
    echo
    
    # Advanced functionality
    run_test "Bidirectional sync" "test_bidirectional_sync"
    run_test "Rate limiting" "test_rate_limiting"
    
    echo
    log_info "Running authentication tests..."
    echo
    
    # Authentication tests
    run_test "POST /auth/nostr" "test_nostr_auth_endpoint"
    run_test "POST /auth/nostr/verify" "test_nostr_verify_endpoint"
    
    echo
    log_info "Running error handling tests..."
    echo
    
    # Error handling
    run_test "Malformed JSON handling" "test_malformed_json"
    run_test "Large payload handling" "test_large_payload"
    
    echo
    log_info "Running performance tests..."
    echo
    
    # Performance tests
    run_test "Response time" "test_response_time"
    run_test "Concurrent requests" "test_concurrent_requests"
    
    echo
    log_info "Test Summary:"
    log_info "Total tests: $TEST_COUNT"
    log_success "Passed: $PASS_COUNT"
    
    if [ $FAIL_COUNT -gt 0 ]; then
        log_error "Failed: $FAIL_COUNT"
        echo
        log_error "Some tests failed. Please review the output above."
        exit 1
    else
        echo
        log_success "All tests passed! Settings sync is working correctly."
        exit 0
    fi
}

# Handle script arguments
case "${1:-}" in
    --server-url)
        if [ -n "${2:-}" ]; then
            SERVER_URL="$2"
            shift 2
        else
            log_error "--server-url requires a URL argument"
            exit 1
        fi
        ;;
    --help|-h)
        echo "Usage: $0 [--server-url URL]"
        echo
        echo "Options:"
        echo "  --server-url URL    Server URL (default: http://localhost:8080)"
        echo "  --help, -h          Show this help message"
        echo
        echo "This script tests the complete settings sync functionality including:"
        echo "  - REST API endpoints with bloom field validation"
        echo "  - Server acceptance and processing of bloom settings"
        echo "  - Bidirectional sync between client and server"
        echo "  - Nostr authentication integration"
        echo "  - Rate limiting and security measures"
        echo "  - Error handling and recovery scenarios"
        exit 0
        ;;
esac

# Run the tests
main
