#!/bin/bash

echo "🚀 Starting MCP Bot Observability Server..."
echo ""
echo "Available tools: 47 observability tools"
echo "Categories: agent.*, swarm.*, message.*, performance.*, visualization.*, neural.*, memory.*"
echo ""
echo "Physics Engine: 60 FPS spring-directed graph"
echo "Agent Capacity: 1000+ concurrent agents"
echo ""
echo "Starting server on stdio..."
echo "=================================="
echo ""

cd "$(dirname "$0")"
exec node src/index.js