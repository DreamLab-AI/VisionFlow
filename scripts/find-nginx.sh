#!/bin/bash
# Helper script to find nginx in the container

echo "Finding nginx binary location..."
docker exec visionflow_container which nginx 2>/dev/null || echo "nginx not found in PATH"

echo ""
echo "Searching filesystem for nginx..."
docker exec visionflow_container find / -name nginx -type f 2>/dev/null | head -10

echo ""
echo "Checking common locations..."
for loc in /usr/bin/nginx /usr/sbin/nginx /usr/local/bin/nginx /usr/local/nginx/sbin/nginx; do
    docker exec visionflow_container ls -la "$loc" 2>/dev/null && echo "Found: $loc"
done

echo ""
echo "Package info..."
docker exec visionflow_container dpkg -l | grep nginx 2>/dev/null || echo "nginx package not installed"