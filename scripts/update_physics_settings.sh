#!/bin/bash

# VisionFlow Physics Settings Update Script
# Updates physics parameters via API endpoints

set -e  # Exit on any error

# Configuration
CONTAINER_IP="172.18.0.10"
PORTS=(3000 4000)
TIMEOUT=10

# Physics parameters to update
declare -A PHYSICS_PARAMS=(
    ["visualisation.graphs.visionflow.physics.springK"]="5.0"
    ["visualisation.graphs.visionflow.physics.repelK"]="50.0"
    ["visualisation.graphs.visionflow.physics.maxVelocity"]="20.0"
    ["visualisation.graphs.visionflow.physics.maxForce"]="200.0"
    ["visualisation.graphs.visionflow.physics.damping"]="0.2"
    ["visualisation.graphs.visionflow.physics.temperature"]="0.1"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}VisionFlow Physics Settings Update Script${NC}"
echo "========================================"

# Function to test connectivity to a port
test_port() {
    local port=$1
    echo -e "${YELLOW}Testing connectivity to ${CONTAINER_IP}:${port}...${NC}"

    if curl -s --connect-timeout $TIMEOUT "http://${CONTAINER_IP}:${port}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Port ${port} is accessible${NC}"
        return 0
    elif curl -s --connect-timeout $TIMEOUT "http://${CONTAINER_IP}:${port}/" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Port ${port} is accessible (no health endpoint)${NC}"
        return 0
    else
        echo -e "${RED}✗ Port ${port} is not accessible${NC}"
        return 1
    fi
}

# Function to update a single parameter
update_parameter() {
    local base_url=$1
    local param_key=$2
    local param_value=$3

    echo -e "${YELLOW}Updating ${param_key} = ${param_value}${NC}"

    # Try different API endpoints
    local endpoints=(
        "/api/config"
        "/api/settings"
        "/api/physics"
        "/api/update"
        "/config"
        "/settings"
        "/physics"
    )

    for endpoint in "${endpoints[@]}"; do
        # Try POST with JSON body
        local response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
            --connect-timeout $TIMEOUT \
            -X POST \
            -H "Content-Type: application/json" \
            -d "{\"${param_key}\": ${param_value}}" \
            "${base_url}${endpoint}" 2>/dev/null || echo "000")

        if [[ "$response" =~ ^[23] ]]; then
            echo -e "${GREEN}  ✓ Successfully updated via ${endpoint} (HTTP ${response})${NC}"
            if [ -s /tmp/curl_response.txt ]; then
                echo -e "${BLUE}  Response: $(cat /tmp/curl_response.txt)${NC}"
            fi
            return 0
        fi

        # Try PUT method
        response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
            --connect-timeout $TIMEOUT \
            -X PUT \
            -H "Content-Type: application/json" \
            -d "{\"${param_key}\": ${param_value}}" \
            "${base_url}${endpoint}" 2>/dev/null || echo "000")

        if [[ "$response" =~ ^[23] ]]; then
            echo -e "${GREEN}  ✓ Successfully updated via ${endpoint} (HTTP ${response})${NC}"
            if [ -s /tmp/curl_response.txt ]; then
                echo -e "${BLUE}  Response: $(cat /tmp/curl_response.txt)${NC}"
            fi
            return 0
        fi

        # Try PATCH method
        response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
            --connect-timeout $TIMEOUT \
            -X PATCH \
            -H "Content-Type: application/json" \
            -d "{\"${param_key}\": ${param_value}}" \
            "${base_url}${endpoint}" 2>/dev/null || echo "000")

        if [[ "$response" =~ ^[23] ]]; then
            echo -e "${GREEN}  ✓ Successfully updated via ${endpoint} (HTTP ${response})${NC}"
            if [ -s /tmp/curl_response.txt ]; then
                echo -e "${BLUE}  Response: $(cat /tmp/curl_response.txt)${NC}"
            fi
            return 0
        fi
    done

    echo -e "${RED}  ✗ Failed to update ${param_key}${NC}"
    return 1
}

# Function to send batch update
send_batch_update() {
    local base_url=$1

    echo -e "${YELLOW}Attempting batch update...${NC}"

    # Create JSON payload with all parameters
    local json_payload="{"
    local first=true
    for key in "${!PHYSICS_PARAMS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            json_payload+=","
        fi
        json_payload+="\"${key}\": ${PHYSICS_PARAMS[$key]}"
    done
    json_payload+="}"

    echo -e "${BLUE}JSON Payload: ${json_payload}${NC}"

    local endpoints=(
        "/api/config/batch"
        "/api/settings/batch"
        "/api/physics/batch"
        "/api/batch"
        "/api/config"
        "/api/settings"
        "/api/physics"
        "/batch"
        "/config"
        "/settings"
    )

    for endpoint in "${endpoints[@]}"; do
        for method in POST PUT PATCH; do
            echo -e "${YELLOW}  Trying ${method} ${endpoint}...${NC}"

            local response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
                --connect-timeout $TIMEOUT \
                -X $method \
                -H "Content-Type: application/json" \
                -d "$json_payload" \
                "${base_url}${endpoint}" 2>/dev/null || echo "000")

            if [[ "$response" =~ ^[23] ]]; then
                echo -e "${GREEN}  ✓ Batch update successful via ${method} ${endpoint} (HTTP ${response})${NC}"
                if [ -s /tmp/curl_response.txt ]; then
                    echo -e "${BLUE}  Response: $(cat /tmp/curl_response.txt)${NC}"
                fi
                return 0
            fi
        done
    done

    echo -e "${RED}  ✗ Batch update failed${NC}"
    return 1
}

# Function to trigger reload
trigger_reload() {
    local base_url=$1

    echo -e "${YELLOW}Triggering settings reload...${NC}"

    local reload_endpoints=(
        "/api/reload"
        "/api/refresh"
        "/api/config/reload"
        "/api/settings/reload"
        "/api/physics/reload"
        "/reload"
        "/refresh"
    )

    for endpoint in "${reload_endpoints[@]}"; do
        for method in POST PUT GET; do
            local response=$(curl -s -w "%{http_code}" -o /tmp/curl_response.txt \
                --connect-timeout $TIMEOUT \
                -X $method \
                "${base_url}${endpoint}" 2>/dev/null || echo "000")

            if [[ "$response" =~ ^[23] ]]; then
                echo -e "${GREEN}  ✓ Reload triggered via ${method} ${endpoint} (HTTP ${response})${NC}"
                if [ -s /tmp/curl_response.txt ]; then
                    echo -e "${BLUE}  Response: $(cat /tmp/curl_response.txt)${NC}"
                fi
                return 0
            fi
        done
    done

    echo -e "${YELLOW}  No reload endpoint found (this may be normal)${NC}"
    return 1
}

# Main execution
main() {
    local success=false

    for port in "${PORTS[@]}"; do
        echo -e "\n${BLUE}=== Testing Port ${port} ===${NC}"

        if test_port $port; then
            local base_url="http://${CONTAINER_IP}:${port}"

            # Try batch update first
            if send_batch_update "$base_url"; then
                success=true
                trigger_reload "$base_url"
                break
            else
                # Fall back to individual parameter updates
                echo -e "${YELLOW}Batch update failed, trying individual updates...${NC}"
                local individual_success=true

                for key in "${!PHYSICS_PARAMS[@]}"; do
                    if ! update_parameter "$base_url" "$key" "${PHYSICS_PARAMS[$key]}"; then
                        individual_success=false
                    fi
                done

                if [ "$individual_success" = true ]; then
                    success=true
                    trigger_reload "$base_url"
                    break
                fi
            fi
        fi
    done

    echo -e "\n${BLUE}=== Summary ===${NC}"
    if [ "$success" = true ]; then
        echo -e "${GREEN}✓ Physics settings update completed successfully${NC}"
        echo -e "${BLUE}Updated parameters:${NC}"
        for key in "${!PHYSICS_PARAMS[@]}"; do
            echo -e "  ${key} = ${PHYSICS_PARAMS[$key]}"
        done
    else
        echo -e "${RED}✗ Failed to update physics settings${NC}"
        echo -e "${YELLOW}This could mean:${NC}"
        echo -e "  1. VisionFlow container is not running"
        echo -e "  2. API endpoints have different paths"
        echo -e "  3. Authentication is required"
        echo -e "  4. Different HTTP methods are needed"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    rm -f /tmp/curl_response.txt
}

trap cleanup EXIT

# Run main function
main

echo -e "\n${GREEN}Script execution completed${NC}"