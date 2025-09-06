#!/bin/bash

# Test script for settings batch endpoints
API_URL="http://localhost:3001/api/settings"

echo "=== Testing Settings Batch Endpoints ==="
echo ""

# Test 1: Batch GET
echo "1. Testing Batch GET endpoint..."
curl -X POST "${API_URL}/batch" \
  -H "Content-Type: application/json" \
  -d '{"paths": ["physics.springK", "physics.repelK", "visualisation.bloom.intensity"]}' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null | jq '.' || echo "Failed"

echo ""
echo "2. Testing Batch UPDATE endpoint..."
# Test 2: Batch UPDATE
curl -X PUT "${API_URL}/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {"path": "physics.springK", "value": 0.005},
      {"path": "physics.repelK", "value": 50.0},
      {"path": "visualisation.bloom.intensity", "value": 1.5}
    ]
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null | jq '.' || echo "Failed"

echo ""
echo "3. Testing individual path GET..."
curl -X GET "${API_URL}/path?path=physics.springK" \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null | jq '.' || echo "Failed"

echo ""
echo "4. Testing individual path UPDATE..."
curl -X PUT "${API_URL}/path" \
  -H "Content-Type: application/json" \
  -d '{"path": "physics.damping", "value": 0.9}' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null | jq '.' || echo "Failed"

echo ""
echo "=== Tests Complete ==="