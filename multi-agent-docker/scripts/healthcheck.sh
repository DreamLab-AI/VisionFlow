#!/bin/bash
# Health check script for Docker container monitoring

# Try to check API endpoint if available
if curl -f http://localhost:8080/health 2>/dev/null; then
    exit 0
fi

# Fallback: Check if agentic-flow is accessible
if command -v agentic-flow > /dev/null 2>&1; then
    exit 0
fi

# If both fail, container is unhealthy
exit 1
