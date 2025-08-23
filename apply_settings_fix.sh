#!/bin/bash
# Script to apply the fixed settings.yaml to the running Docker container

echo "Applying fixed settings.yaml to Docker container..."
echo "================================================"
echo ""

# Check if the container is running
if docker ps | grep -q visionflow_container; then
    echo "✅ Container 'visionflow_container' is running"
    echo ""
    
    # Copy the fixed settings.yaml to the container
    echo "Copying fixed settings.yaml to container..."
    docker cp data/settings.yaml visionflow_container:/app/settings.yaml
    
    if [ $? -eq 0 ]; then
        echo "✅ Settings file copied successfully"
        echo ""
        
        # Restart the rust-backend service in the container
        echo "Restarting rust-backend service..."
        docker exec visionflow_container supervisorctl restart rust-backend
        
        if [ $? -eq 0 ]; then
            echo "✅ Rust backend restarted"
            echo ""
            
            # Wait a moment and check the status
            sleep 2
            echo "Checking service status..."
            docker exec visionflow_container supervisorctl status rust-backend
            
            echo ""
            echo "✅ Fix applied successfully!"
            echo ""
            echo "The rust-backend should now be running without YAML errors."
        else
            echo "⚠️  Could not restart rust-backend via supervisorctl"
            echo "You may need to restart the entire container with:"
            echo "  ./scripts/launch.sh restart"
        fi
    else
        echo "❌ Failed to copy settings file to container"
        exit 1
    fi
else
    echo "❌ Container 'visionflow_container' is not running"
    echo ""
    echo "Please start the container first with:"
    echo "  ./scripts/launch.sh"
    exit 1
fi

echo ""
echo "To verify the fix worked, check the logs with:"
echo "  tail -f logs/rust-error.log"