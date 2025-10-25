#!/bin/bash

echo "===== CRASH TIMING ANALYSIS ====="
echo ""

test_crash_timing() {
    local endpoint=$1
    local name=$2
    
    echo "Testing: $name ($endpoint)"
    echo "Starting at: $(date +%H:%M:%S.%N)"
    
    # Use timeout to track exact timing
    timeout --preserve-status 15s time -p docker exec visionflow_container curl -v --max-time 10 "$endpoint" 2>&1 | {
        while IFS= read -r line; do
            echo "$(date +%H:%M:%S.%N) | $line"
        done
    }
    
    echo "Ended at: $(date +%H:%M:%S.%N)"
    echo ""
}

# Test various endpoints with precise timing
test_crash_timing "http://localhost:4000/api/health" "Health (working baseline)"
sleep 3

test_crash_timing "http://localhost:4000/api/config" "Config (known crash)"
sleep 3

test_crash_timing "http://localhost:4000/api/settings/system" "Settings System"
sleep 3

test_crash_timing "http://localhost:4000/api/graph/data" "Graph Data"
sleep 3

echo "===== TCP CONNECTION DETAILS ====="
docker exec visionflow_container curl -v --trace-time --max-time 5 http://localhost:4000/api/config 2>&1 | grep -E "(Connected|Closing|timeout|Expire)"

