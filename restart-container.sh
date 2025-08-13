#!/bin/bash
# Restart container to apply nginx and client fixes

echo "🔄 Restarting VisionFlow container to apply fixes..."
echo ""

cd /workspace/ext

# Restart container
echo "📦 Restarting container..."
docker restart visionflow_container 2>/dev/null || {
    echo "Container not running, starting it..."
    docker compose --profile dev up -d 2>/dev/null || {
        echo "Docker compose not available, trying direct start..."
        docker start visionflow_container 2>/dev/null || {
            echo "❌ Failed to start container. You may need to rebuild first."
            echo "   Run: ./quick-rebuild.sh"
            exit 1
        }
    }
}

echo ""
echo "✅ Container restarted!"
echo ""
echo "Check status with:"
echo "  docker logs -f visionflow_container"
echo ""
echo "Access the application at:"
echo "  http://localhost:3001"
echo ""
echo "Monitor nginx errors:"
echo "  docker exec visionflow_container tail -f /var/log/nginx/error.log"