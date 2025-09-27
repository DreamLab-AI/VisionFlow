#!/bin/bash

# VisionFlow Physics Settings Direct Update Script
# Updates physics parameters using the correct nested structure

set -e  # Exit on any error

# Configuration
CONTAINER_IP="172.18.0.10"
PORT="4000"
TIMEOUT=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}VisionFlow Physics Settings Direct Update Script${NC}"
echo "=============================================="

BASE_URL="http://${CONTAINER_IP}:${PORT}"

# Create the correct nested JSON structure
JSON_PAYLOAD='{
    "visualisation": {
        "graphs": {
            "visionflow": {
                "physics": {
                    "springK": 5.0,
                    "repelK": 50.0,
                    "maxVelocity": 20.0,
                    "maxForce": 200.0,
                    "damping": 0.2,
                    "temperature": 0.1
                }
            }
        }
    }
}'

echo -e "${BLUE}JSON Payload:${NC}"
echo "$JSON_PAYLOAD" | jq '.' 2>/dev/null || echo "$JSON_PAYLOAD"

# Function to update physics settings
update_physics() {
    echo -e "\n${YELLOW}Updating physics settings...${NC}"

    # Try different API endpoints with the nested structure
    local endpoints=(
        "/api/settings"
        "/api/config"
        "/api/update"
    )

    for endpoint in "${endpoints[@]}"; do
        for method in POST PUT PATCH; do
            echo -e "${YELLOW}  Trying ${method} ${endpoint}...${NC}"

            local response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
                --connect-timeout $TIMEOUT \
                -X $method \
                -H "Content-Type: application/json" \
                -d "$JSON_PAYLOAD" \
                "${BASE_URL}${endpoint}" 2>/dev/null || echo "000")

            if [[ "$response" =~ ^[23] ]]; then
                echo -e "${GREEN}  ✓ Update successful via ${method} ${endpoint} (HTTP ${response})${NC}"
                if [ -s /tmp/curl_response.txt ]; then
                    echo -e "${BLUE}  Response size: $(wc -c < /tmp/curl_response.txt) bytes${NC}"
                fi
                return 0
            else
                echo -e "${RED}  ✗ Failed with HTTP ${response}${NC}"
            fi
        done
    done

    return 1
}

# Function to verify the update
verify_update() {
    echo -e "\n${YELLOW}Verifying physics settings update...${NC}"

    local current_settings=$(curl -s "${BASE_URL}/api/settings" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully retrieved current settings${NC}"

        # Extract visionflow physics settings
        local visionflow_physics=$(echo "$current_settings" | jq '.visualisation.graphs.visionflow.physics' 2>/dev/null)

        if [ "$visionflow_physics" != "null" ] && [ -n "$visionflow_physics" ]; then
            echo -e "${BLUE}Current VisionFlow Physics Settings:${NC}"
            echo "$visionflow_physics" | jq '{springK, repelK, maxVelocity, maxForce, damping, temperature}'

            # Check if our values were applied
            local springK=$(echo "$visionflow_physics" | jq -r '.springK')
            local repelK=$(echo "$visionflow_physics" | jq -r '.repelK')
            local maxVelocity=$(echo "$visionflow_physics" | jq -r '.maxVelocity')
            local maxForce=$(echo "$visionflow_physics" | jq -r '.maxForce')
            local damping=$(echo "$visionflow_physics" | jq -r '.damping')
            local temperature=$(echo "$visionflow_physics" | jq -r '.temperature')

            echo -e "\n${BLUE}Verification Results:${NC}"
            if [ "$springK" = "5" ] || [ "$springK" = "5.0" ]; then
                echo -e "${GREEN}✓ springK: $springK (target: 5.0)${NC}"
            else
                echo -e "${RED}✗ springK: $springK (target: 5.0)${NC}"
            fi

            if [ "$repelK" = "50" ] || [ "$repelK" = "50.0" ]; then
                echo -e "${GREEN}✓ repelK: $repelK (target: 50.0)${NC}"
            else
                echo -e "${RED}✗ repelK: $repelK (target: 50.0)${NC}"
            fi

            if [ "$maxVelocity" = "20" ] || [ "$maxVelocity" = "20.0" ]; then
                echo -e "${GREEN}✓ maxVelocity: $maxVelocity (target: 20.0)${NC}"
            else
                echo -e "${RED}✗ maxVelocity: $maxVelocity (target: 20.0)${NC}"
            fi

            if [ "$maxForce" = "200" ] || [ "$maxForce" = "200.0" ]; then
                echo -e "${GREEN}✓ maxForce: $maxForce (target: 200.0)${NC}"
            else
                echo -e "${RED}✗ maxForce: $maxForce (target: 200.0)${NC}"
            fi

            if [ "$damping" = "0.2" ]; then
                echo -e "${GREEN}✓ damping: $damping (target: 0.2)${NC}"
            else
                echo -e "${RED}✗ damping: $damping (target: 0.2)${NC}"
            fi

            if [ "$temperature" = "0.1" ]; then
                echo -e "${GREEN}✓ temperature: $temperature (target: 0.1)${NC}"
            else
                echo -e "${RED}✗ temperature: $temperature (target: 0.1)${NC}"
            fi
        else
            echo -e "${RED}✗ Could not find visionflow physics settings${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to retrieve current settings${NC}"
        return 1
    fi
}

# Function to try individual parameter updates if batch fails
try_individual_updates() {
    echo -e "\n${YELLOW}Trying individual parameter updates...${NC}"

    declare -A PARAMS=(
        ["springK"]="5.0"
        ["repelK"]="50.0"
        ["maxVelocity"]="20.0"
        ["maxForce"]="200.0"
        ["damping"]="0.2"
        ["temperature"]="0.1"
    )

    for param in "${!PARAMS[@]}"; do
        local value="${PARAMS[$param]}"
        echo -e "\n${YELLOW}Updating ${param} = ${value}${NC}"

        # Create JSON for individual parameter update
        local individual_json="{
            \"visualisation\": {
                \"graphs\": {
                    \"visionflow\": {
                        \"physics\": {
                            \"${param}\": ${value}
                        }
                    }
                }
            }
        }"

        local response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
            --connect-timeout $TIMEOUT \
            -X POST \
            -H "Content-Type: application/json" \
            -d "$individual_json" \
            "${BASE_URL}/api/settings" 2>/dev/null || echo "000")

        if [[ "$response" =~ ^[23] ]]; then
            echo -e "${GREEN}  ✓ ${param} updated successfully${NC}"
        else
            echo -e "${RED}  ✗ Failed to update ${param}${NC}"
        fi
    done
}

# Function to trigger reload or restart
trigger_reload() {
    echo -e "\n${YELLOW}Attempting to trigger settings reload...${NC}"

    local reload_endpoints=(
        "/api/reload"
        "/api/refresh"
        "/api/restart"
        "/api/physics/reload"
        "/api/config/reload"
    )

    for endpoint in "${reload_endpoints[@]}"; do
        local response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
            --connect-timeout $TIMEOUT \
            -X POST \
            "${BASE_URL}${endpoint}" 2>/dev/null || echo "000")

        if [[ "$response" =~ ^[23] ]]; then
            echo -e "${GREEN}  ✓ Reload triggered via ${endpoint} (HTTP ${response})${NC}"
            return 0
        fi
    done

    echo -e "${YELLOW}  No reload endpoint responded (this may be normal)${NC}"
    return 1
}

# Main execution
main() {
    # Test connectivity
    echo -e "${YELLOW}Testing connectivity to ${BASE_URL}...${NC}"
    if ! curl -s --connect-timeout $TIMEOUT "${BASE_URL}/api/settings" > /dev/null; then
        echo -e "${RED}✗ Cannot connect to VisionFlow API${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Connected to VisionFlow API${NC}"

    # Try batch update first
    if update_physics; then
        echo -e "${GREEN}✓ Batch update completed${NC}"
    else
        echo -e "${YELLOW}Batch update failed, trying individual parameter updates...${NC}"
        try_individual_updates
    fi

    # Trigger reload
    trigger_reload

    # Verify the update
    verify_update

    echo -e "\n${GREEN}Physics settings update process completed${NC}"
}

# Cleanup function
cleanup() {
    rm -f /tmp/curl_response.txt
}

trap cleanup EXIT

# Run main function
main