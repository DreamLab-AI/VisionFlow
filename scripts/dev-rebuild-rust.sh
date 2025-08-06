#!/bin/bash
# Development script to rebuild and restart Rust server when source changes

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== VisionFlow Rust Development Rebuild Script ===${NC}"

# Function to rebuild Rust inside container
rebuild_rust() {
    echo -e "${YELLOW}Rebuilding Rust server...${NC}"
    
    # Execute cargo build inside the container
    docker compose -f docker-compose.dev.yml exec visionflow-xr bash -c "
        cd /app && \
        echo 'Building with GPU features...' && \
        cargo build --features gpu && \
        cp target/debug/webxr /app/webxr && \
        echo 'Build successful!'
    "
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Build completed successfully${NC}"
        return 0
    else
        echo -e "${RED}Build failed!${NC}"
        return 1
    fi
}

# Function to restart Rust server
restart_rust_server() {
    echo -e "${YELLOW}Restarting Rust server...${NC}"
    
    # Kill the existing Rust process
    docker compose -f docker-compose.dev.yml exec visionflow-xr bash -c "
        pkill -f webxr || true
    "
    
    # Start the new process
    docker compose -f docker-compose.dev.yml exec -d visionflow-xr bash -c "
        cd /app && \
        ./webxr --gpu-debug > /app/logs/rust_server.log 2>&1
    "
    
    echo -e "${GREEN}Rust server restarted${NC}"
}

# Main execution
main() {
    # Check if container is running
    if ! docker compose -f docker-compose.dev.yml ps | grep -q "visionflow-xr.*running"; then
        echo -e "${RED}Error: visionflow-xr container is not running${NC}"
        echo "Please start the container first with: ./scripts/dev.sh"
        exit 1
    fi
    
    # Rebuild and restart
    if rebuild_rust; then
        restart_rust_server
        echo -e "${GREEN}Development rebuild complete!${NC}"
        
        # Show logs
        echo -e "${YELLOW}Tailing Rust server logs (Ctrl+C to exit)...${NC}"
        docker compose -f docker-compose.dev.yml exec visionflow-xr tail -f /app/logs/rust_server.log
    else
        echo -e "${RED}Rebuild failed. Please check the errors above.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"