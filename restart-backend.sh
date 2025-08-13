#!/bin/bash
# Restart backend to apply new endpoint

echo "🔄 Restarting backend to add multi-agent endpoint..."
echo ""

# Restart the Rust backend inside the container
docker exec visionflow_container supervisorctl restart rust-backend 2>/dev/null || {
    echo "Note: Could not restart via supervisorctl, trying direct restart..."
    docker exec visionflow_container pkill -f webxr
    sleep 2
    docker exec -d visionflow_container /app/webxr
}

echo ""
echo "✅ Backend restart initiated!"
echo ""
echo "The new endpoint /api/bots/initialize-multi-agent is now available."
echo "It will return a mock response for now, allowing the UI to work."
echo ""
echo "To connect to the real multi-agent-container on TCP port 9500,"
echo "the TCP client code needs to be implemented in the handler."