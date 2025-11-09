#!/bin/bash
# P1-1 Stress Majorization Configuration Endpoint Test Script

set -e

BASE_URL="${BASE_URL:-http://localhost:8080}"
API_PATH="/api/analytics/stress-majorization"

echo "================================"
echo "P1-1 Configuration Endpoint Tests"
echo "================================"
echo ""
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Get current configuration
echo "Test 1: GET current configuration"
echo "-----------------------------------"
curl -s -X GET "$BASE_URL$API_PATH/config" | jq '.'
echo ""

# Test 2: Configure with valid parameters
echo "Test 2: POST valid configuration"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "learning_rate": 0.15,
    "momentum": 0.6,
    "max_iterations": 150,
    "auto_run_interval": 450
  }' | jq '.'
echo ""

# Test 3: Verify configuration was applied
echo "Test 3: Verify configuration update"
echo "-----------------------------------"
curl -s -X GET "$BASE_URL$API_PATH/config" | jq '.'
echo ""

# Test 4: Partial update (only learning_rate)
echo "Test 4: Partial update (learning_rate only)"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/configure" \
  -H "Content-Type: application/json" \
  -d '{"learning_rate": 0.2}' | jq '.'
echo ""

# Test 5: Validation - learning_rate too high
echo "Test 5: Validation error - learning_rate too high"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/configure" \
  -H "Content-Type: application/json" \
  -d '{"learning_rate": 0.6}' | jq '.'
echo ""

# Test 6: Validation - momentum too high
echo "Test 6: Validation error - momentum too high"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/configure" \
  -H "Content-Type: application/json" \
  -d '{"momentum": 1.5}' | jq '.'
echo ""

# Test 7: Validation - max_iterations too low
echo "Test 7: Validation error - max_iterations too low"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/configure" \
  -H "Content-Type: application/json" \
  -d '{"max_iterations": 5}' | jq '.'
echo ""

# Test 8: Validation - auto_run_interval out of range
echo "Test 8: Validation error - auto_run_interval too low"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/configure" \
  -H "Content-Type: application/json" \
  -d '{"auto_run_interval": 20}' | jq '.'
echo ""

# Test 9: Multiple parameter update
echo "Test 9: Update multiple parameters"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "learning_rate": 0.12,
    "momentum": 0.55,
    "max_iterations": 120
  }' | jq '.'
echo ""

# Test 10: Get final configuration state
echo "Test 10: GET final configuration"
echo "-----------------------------------"
curl -s -X GET "$BASE_URL$API_PATH/config" | jq '.'
echo ""

# Test 11: Trigger stress majorization with new config
echo "Test 11: Trigger stress majorization"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL$API_PATH/trigger" | jq '.'
echo ""

# Test 12: Check stats after trigger
echo "Test 12: GET stats after trigger"
echo "-----------------------------------"
curl -s -X GET "$BASE_URL$API_PATH/stats" | jq '.'
echo ""

echo "================================"
echo "All tests completed!"
echo "================================"
