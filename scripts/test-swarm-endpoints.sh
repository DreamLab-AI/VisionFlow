#!/bin/bash

# Test script to check swarm endpoints connectivity

echo "🔍 Testing Swarm Endpoints..."
echo "============================="

# Base URL - adjust based on your environment
BASE_URL="http://localhost:3001"

# Test the swarm data endpoint
echo -e "\n📊 Testing GET /api/swarm/data..."
curl -X GET "$BASE_URL/api/swarm/data" -H "Content-Type: application/json" -v 2>&1 | grep -E "(< HTTP|< Location|{|])"

# Test swarm data with test flag
echo -e "\n\n🧪 Testing GET /api/swarm/data?test=true..."
curl -X GET "$BASE_URL/api/swarm/data?test=true" -H "Content-Type: application/json" -s | jq '.'

# Test the MCP WebSocket endpoint
echo -e "\n\n🔌 Testing WebSocket /ws/mcp..."
echo "WebSocket endpoint available at: ws://localhost:3001/ws/mcp"

# Check if the backend is running
echo -e "\n\n🏥 Checking backend health..."
curl -X GET "$BASE_URL/api/health" -s | jq '.'

# Show the live powerdev endpoint info
echo -e "\n\n🌐 Live PowerDev Endpoint Info:"
echo "The VisionFlow visualization should connect to:"
echo "- Primary: ws://localhost:3001/ws/mcp (MCP Relay)"
echo "- Fallback: http://localhost:3001/api/swarm/data (REST API)"
echo ""
echo "If no agents are showing, make sure:"
echo "1. The orchestrator service is running"
echo "2. Claude Flow swarm is initialized"
echo "3. The MCP relay can connect to orchestrator at ws://orchestrator:8080/ws"