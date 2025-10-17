#!/bin/bash

echo "=== Fixing Kokoro Network Configuration ==="

# Add Kokoro container to the ragflow network
echo "Adding Kokoro container to docker_ragflow network..."
docker network connect docker_ragflow friendly_dewdney

# Verify the connection
echo -e "\nVerifying network connection..."
docker inspect friendly_dewdney | grep -A 10 "Networks" | grep -E "(docker_ragflow|IPAddress)"

# Get the new IP address
KOKORO_IP=$(docker inspect friendly_dewdney -f '{{range .NetworkSettings.Networks}}{{if eq .NetworkID "b0c38a1301451c0329969ef53fdedde5221b1b05b063ad94d66017a45d3ddaa3"}}{{.IPAddress}}{{end}}{{end}}')

if [ -n "$KOKORO_IP" ]; then
    echo -e "\n✓ Kokoro is now accessible at: $KOKORO_IP:8880"
    echo "  Internal hostname: friendly_dewdney"

    # Update settings via API
    echo -e "\nUpdating Kokoro URL via Settings API..."
    curl -X POST http://localhost:4000/api/settings/batch \
        -H "Content-Type: application/json" \
        -d "{\"updates\":[{\"path\":\"voice.tts.kokoro.apiUrl\",\"value\":\"http://$KOKORO_IP:8880\"}]}" \
        -s -o /dev/null -w "API Response: %{http_code}\n"

    echo "✓ Settings updated!"

    # Test the connection
    echo -e "\nTesting Kokoro connection..."
    curl -s "http://$KOKORO_IP:8880/health" | head -5
else
    echo "✗ Failed to add Kokoro to network"
fi