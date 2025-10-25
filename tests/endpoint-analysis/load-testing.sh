#!/bin/bash

echo "===== LOAD TESTING: SEQUENTIAL REQUEST PATTERNS ====="
echo ""

# Test 1: Multiple requests to working endpoint
echo "Test 1: Five sequential requests to /api/health (known working)"
for i in {1..5}; do
    echo "Request $i:"
    docker exec visionflow_container curl -s -w "HTTP: %{http_code}, Time: %{time_total}s\n" http://localhost:4000/api/health | head -n 1
    sleep 1
done

echo ""
echo "Test 2: Multiple requests to /api/config (known crashing)"
for i in {1..3}; do
    echo "Request $i:"
    start=$(date +%s)
    docker exec visionflow_container curl -s -w "HTTP: %{http_code}, Time: %{time_total}s\n" --max-time 5 http://localhost:4000/api/config 2>&1 | head -n 2
    end=$(date +%s)
    echo "Actual duration: $((end - start))s"
    
    # Check if backend still running
    backend_count=$(docker exec visionflow_container pgrep -f "node.*server.js" | wc -l)
    echo "Backend processes running: $backend_count"
    
    sleep 3
done

echo ""
echo "Test 3: Alternating working/crashing endpoints"
for i in {1..3}; do
    echo "Cycle $i - Health:"
    docker exec visionflow_container curl -s -w "HTTP: %{http_code}\n" http://localhost:4000/api/health | tail -n 1
    
    sleep 1
    
    echo "Cycle $i - Config:"
    docker exec visionflow_container curl -s -w "HTTP: %{http_code}\n" --max-time 5 http://localhost:4000/api/config 2>&1 | tail -n 1
    
    sleep 2
done

echo ""
echo "Test 4: Recovery test - wait 30s after crash, then retry"
echo "Triggering crash with /api/config..."
docker exec visionflow_container curl -s --max-time 5 http://localhost:4000/api/config > /dev/null 2>&1

echo "Waiting 30 seconds for potential recovery..."
sleep 30

echo "Testing if endpoint recovered:"
docker exec visionflow_container curl -s -w "HTTP: %{http_code}, Time: %{time_total}s\n" http://localhost:4000/api/config

echo ""
echo "Backend status after recovery test:"
docker exec visionflow_container pgrep -f "node.*server.js" && echo "Backend running" || echo "Backend NOT running"

