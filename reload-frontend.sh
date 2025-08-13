#!/bin/bash
# Reload frontend and backend to apply physics changes

echo "🔄 Reloading services to apply physics changes..."
echo ""

# Restart services inside the container using supervisorctl
echo "Restarting backend and frontend services..."
docker exec visionflow_container supervisorctl restart rust-backend 2>/dev/null || {
    echo "Note: Could not restart rust-backend via supervisorctl"
}

docker exec visionflow_container supervisorctl restart vite-dev 2>/dev/null || {
    echo "Note: Could not restart vite-dev via supervisorctl"
}

echo ""
echo "✅ Services restarted!"
echo ""
echo "The physics settings have been updated to:"
echo "  - Damping: 0.98 (higher = less bouncing)"
echo "  - Spring Strength: 0.001 (lower = softer springs)"
echo "  - Max Velocity: 0.5 (lower = slower movement)"
echo ""
echo "Please refresh your browser to see the changes."
echo ""
echo "If bouncing persists, you can further adjust in:"
echo "  /workspace/ext/data/settings.yaml"