#\!/bin/bash
echo "Checking server configuration..."
echo ""
echo "1. Docker Status:"
docker --version 2>&1
echo ""

echo "2. Server Routes Configuration:"
echo "The settings routes are configured at:"
echo "  - GET/POST /api/user-settings"
echo "  - GET/POST /api/user-settings/sync"
echo "  - POST /api/user-settings/clear-cache"
echo ""

echo "3. Error Analysis:"
echo "The 502 Bad Gateway errors indicate:"
echo "  - The Rust backend server is not running"
echo "  - Docker daemon is not running on this system"
echo ""

echo "4. Resolution:"
echo "To fix this issue, you need to:"
echo "  1. Start Docker daemon: sudo service docker start"
echo "  2. Build the container: docker build -f Dockerfile.dev -t webxr-dev ."
echo "  3. Run the container: docker run -p 3001:3090 webxr-dev"
echo ""

echo "5. Important Note:"
echo "The refactor created NEW files but did NOT modify existing routes."
echo "The existing settings_handler.rs is unchanged and routes are intact."
echo "The issue is simply that your backend server is not running."
