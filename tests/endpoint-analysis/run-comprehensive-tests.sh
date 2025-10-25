#!/bin/bash

# Comprehensive Endpoint Testing Script
# Testing all VisionFlow backend endpoints systematically

OUTPUT_DIR="/home/devuser/workspace/project/tests/endpoint-analysis"
RESULTS_FILE="$OUTPUT_DIR/endpoint-test-results.json"
LOG_FILE="$OUTPUT_DIR/test-execution.log"

# Initialize results
echo "{" > $RESULTS_FILE
echo "  \"test_timestamp\": \"$(date -Iseconds)\"," >> $RESULTS_FILE
echo "  \"endpoints\": {" >> $RESULTS_FILE

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    
    echo "===== Testing: $name =====" | tee -a $LOG_FILE
    echo "URL: $url" | tee -a $LOG_FILE
    echo "Method: $method" | tee -a $LOG_FILE
    echo "Time: $(date -Iseconds)" | tee -a $LOG_FILE
    
    # Capture start time
    start_time=$(date +%s.%N)
    
    # Execute curl with comprehensive metrics
    response=$(docker exec visionflow_container curl -s -w "\n__METRICS__\nhttp_code:%{http_code}\ntime_total:%{time_total}\ntime_connect:%{time_connect}\ntime_starttransfer:%{time_starttransfer}\nsize_download:%{size_download}\n" \
        --max-time 10 \
        -X $method \
        "$url" 2>&1)
    
    exit_code=$?
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Parse metrics
    http_code=$(echo "$response" | grep "http_code:" | cut -d: -f2)
    time_total=$(echo "$response" | grep "time_total:" | cut -d: -f2)
    time_connect=$(echo "$response" | grep "time_connect:" | cut -d: -f2)
    size_download=$(echo "$response" | grep "size_download:" | cut -d: -f2)
    
    # Get response body (everything before __METRICS__)
    body=$(echo "$response" | sed '/__METRICS__/,$d')
    
    # Check if backend is still running
    backend_running=$(docker exec visionflow_container pgrep -f "node.*server.js" | wc -l)
    
    # Store results
    echo "    \"$name\": {" >> $RESULTS_FILE
    echo "      \"url\": \"$url\"," >> $RESULTS_FILE
    echo "      \"method\": \"$method\"," >> $RESULTS_FILE
    echo "      \"http_code\": \"$http_code\"," >> $RESULTS_FILE
    echo "      \"curl_exit_code\": $exit_code," >> $RESULTS_FILE
    echo "      \"time_total\": \"$time_total\"," >> $RESULTS_FILE
    echo "      \"time_connect\": \"$time_connect\"," >> $RESULTS_FILE
    echo "      \"actual_duration\": \"$duration\"," >> $RESULTS_FILE
    echo "      \"size_download\": \"$size_download\"," >> $RESULTS_FILE
    echo "      \"backend_running_after\": $backend_running," >> $RESULTS_FILE
    echo "      \"response_preview\": \"$(echo "$body" | head -c 200 | tr '\n' ' ')\"," >> $RESULTS_FILE
    echo "      \"status\": \"$([ "$http_code" = "200" ] && echo "SUCCESS" || echo "FAILED")\"" >> $RESULTS_FILE
    echo "    }," >> $RESULTS_FILE
    
    # Log summary
    echo "Result: HTTP $http_code, Exit $exit_code, Duration ${duration}s, Backend Running: $backend_running" | tee -a $LOG_FILE
    echo "Response preview: $(echo "$body" | head -c 100)" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    
    # Small delay between tests
    sleep 2
}

# Test all endpoints systematically
echo "Starting comprehensive endpoint testing..." | tee $LOG_FILE

# Health endpoints (known working)
test_endpoint "health" "http://localhost:4000/api/health"

# Config endpoint (known problematic)
test_endpoint "config" "http://localhost:4000/api/config"

# Settings endpoints
test_endpoint "settings_root" "http://localhost:4000/api/settings"
test_endpoint "settings_system" "http://localhost:4000/api/settings/system"
test_endpoint "settings_visualisation" "http://localhost:4000/api/settings/visualisation"
test_endpoint "settings_database" "http://localhost:4000/api/settings/database"
test_endpoint "settings_api" "http://localhost:4000/api/settings/api"

# Graph endpoints
test_endpoint "graph_data" "http://localhost:4000/api/graph/data"
test_endpoint "graph_nodes" "http://localhost:4000/api/graph/nodes"
test_endpoint "graph_edges" "http://localhost:4000/api/graph/edges"
test_endpoint "graph_stats" "http://localhost:4000/api/graph/stats"

# Ontology endpoints
test_endpoint "ontology_classes" "http://localhost:4000/api/ontology/classes"
test_endpoint "ontology_properties" "http://localhost:4000/api/ontology/properties"
test_endpoint "ontology_individuals" "http://localhost:4000/api/ontology/individuals"

# Search endpoints
test_endpoint "search_nodes" "http://localhost:4000/api/search/nodes"
test_endpoint "search_semantic" "http://localhost:4000/api/search/semantic"

# Layout endpoints
test_endpoint "layout_force" "http://localhost:4000/api/layout/force"
test_endpoint "layout_hierarchical" "http://localhost:4000/api/layout/hierarchical"

# Close JSON
echo "    \"test_complete\": true" >> $RESULTS_FILE
echo "  }" >> $RESULTS_FILE
echo "}" >> $RESULTS_FILE

echo "Testing complete. Results saved to $RESULTS_FILE"
