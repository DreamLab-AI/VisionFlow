#!/usr/bin/env bash

# VisionFlow Settings System - Load Testing Script
#
# Comprehensive load testing for settings infrastructure including:
# - Database read/write performance
# - Hot-reload system
# - WebSocket broadcast
# - Settings search performance
# - Concurrent client simulation
# - Memory and CPU profiling
#
# Requirements:
# - ab (Apache Bench): sudo apt-get install apache2-utils
# - jq: sudo apt-get install jq
# - wrk: sudo apt-get install wrk
# - parallel: sudo apt-get install parallel
#
# Usage:
#   ./load_test_settings.sh [test_suite]
#
# Test Suites:
#   quick  - Basic smoke tests (1 min)
#   medium - Standard load tests (5 min)
#   full   - Comprehensive tests (15 min)

set -euo pipefail

# Configuration
API_BASE="${API_BASE:-http://localhost:8080/api}"
DB_PATH="${DB_PATH:-data/settings.db}"
WS_URL="${WS_URL:-ws://localhost:8080/api/settings/ws}"
TEST_SUITE="${1:-medium}"
OUTPUT_DIR="${OUTPUT_DIR:-load_test_results}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test configuration based on suite
case "$TEST_SUITE" in
    quick)
        DURATION=60
        CONCURRENT_USERS=10
        REQUESTS_PER_SECOND=100
        ;;
    medium)
        DURATION=300
        CONCURRENT_USERS=50
        REQUESTS_PER_SECOND=500
        ;;
    full)
        DURATION=900
        CONCURRENT_USERS=100
        REQUESTS_PER_SECOND=1000
        ;;
    *)
        log_error "Unknown test suite: $TEST_SUITE"
        exit 1
        ;;
esac

log_info "Running $TEST_SUITE test suite"
log_info "Duration: ${DURATION}s, Concurrent Users: $CONCURRENT_USERS, RPS Target: $REQUESTS_PER_SECOND"

# 1. Database Read Performance Test
test_db_reads() {
    log_info "Testing database read performance..."

    local output="$OUTPUT_DIR/db_reads.txt"

    ab -n 10000 -c 50 -g "$OUTPUT_DIR/db_reads_gnuplot.tsv" \
        "$API_BASE/settings/" > "$output" 2>&1

    local rps=$(grep "Requests per second" "$output" | awk '{print $4}')
    local latency=$(grep "Time per request.*mean" "$output" | head -1 | awk '{print $4}')

    log_success "Database reads: $rps req/s, ${latency}ms avg latency"

    # Validation
    if (( $(echo "$latency > 10" | bc -l) )); then
        log_warning "Database read latency exceeds 10ms target"
    fi
}

# 2. Database Write Performance Test
test_db_writes() {
    log_info "Testing database write performance..."

    local output="$OUTPUT_DIR/db_writes.txt"

    # Generate test data
    local test_data='{"key":"test.performance","value":"true"}'

    ab -n 1000 -c 10 \
        -p <(echo "$test_data") \
        -T "application/json" \
        "$API_BASE/settings/test.performance" > "$output" 2>&1

    local rps=$(grep "Requests per second" "$output" | awk '{print $4}')
    local latency=$(grep "Time per request.*mean" "$output" | head -1 | awk '{print $4}')

    log_success "Database writes: $rps req/s, ${latency}ms avg latency"

    # Validation
    if (( $(echo "$latency > 50" | bc -l) )); then
        log_warning "Database write latency exceeds 50ms target"
    fi
}

# 3. Settings Search Performance Test
test_search_performance() {
    log_info "Testing settings search performance..."

    local output="$OUTPUT_DIR/search.txt"

    # Test various search queries
    local queries=("physics" "visualization" "agent" "performance" "gpu" "xr")

    for query in "${queries[@]}"; do
        local start=$(date +%s%N)

        curl -s "$API_BASE/settings/search?q=$query" > /dev/null

        local end=$(date +%s%N)
        local duration=$(( (end - start) / 1000000 ))

        echo "Search '$query': ${duration}ms" >> "$output"

        if (( duration > 100 )); then
            log_warning "Search for '$query' exceeded 100ms: ${duration}ms"
        fi
    done

    log_success "Search performance test complete"
}

# 4. Hot-Reload Performance Test
test_hot_reload() {
    log_info "Testing hot-reload system..."

    # Backup database
    cp "$DB_PATH" "$DB_PATH.backup"

    local start=$(date +%s%N)

    # Modify database
    sqlite3 "$DB_PATH" "UPDATE settings SET value = 'true' WHERE key = 'test.hotreload'"

    # Wait for hot-reload (500ms debounce + processing)
    sleep 1

    local end=$(date +%s%N)
    local duration=$(( (end - start) / 1000000 ))

    log_success "Hot-reload latency: ${duration}ms"

    # Restore database
    mv "$DB_PATH.backup" "$DB_PATH"

    if (( duration > 1000 )); then
        log_warning "Hot-reload exceeded 1000ms target"
    fi
}

# 5. WebSocket Broadcast Load Test
test_websocket_broadcast() {
    log_info "Testing WebSocket broadcast..."

    local output="$OUTPUT_DIR/websocket.txt"

    # Use wscat or custom script
    if command -v wscat &> /dev/null; then
        for i in $(seq 1 $CONCURRENT_USERS); do
            (
                wscat -c "$WS_URL" &
                local pid=$!
                sleep $DURATION
                kill $pid 2>/dev/null || true
            ) &
        done

        wait

        log_success "WebSocket broadcast test complete ($CONCURRENT_USERS concurrent connections)"
    else
        log_warning "wscat not installed, skipping WebSocket test"
    fi
}

# 6. Concurrent Settings Updates
test_concurrent_updates() {
    log_info "Testing concurrent settings updates..."

    local output="$OUTPUT_DIR/concurrent_updates.txt"

    # Parallel updates
    seq 1 $CONCURRENT_USERS | parallel -j $CONCURRENT_USERS \
        "curl -X PUT -H 'Content-Type: application/json' \
        -d '{\"value\": \"test_{}\"}' \
        $API_BASE/settings/test.concurrent{} -s -w 'Time: %{time_total}s\n' >> $output"

    local avg_time=$(awk '{sum+=$2; count++} END {print sum/count}' "$output")

    log_success "Concurrent updates: ${avg_time}s average"

    if (( $(echo "$avg_time > 0.1" | bc -l) )); then
        log_warning "Concurrent update latency exceeds 100ms target"
    fi
}

# 7. Memory Usage Monitoring
test_memory_usage() {
    log_info "Monitoring memory usage..."

    local output="$OUTPUT_DIR/memory.txt"

    # Start memory monitoring
    (
        while true; do
            local mem=$(ps aux | grep "visionflow" | grep -v grep | awk '{print $6}')
            echo "$(date +%s),$mem" >> "$output"
            sleep 5
        done
    ) &
    local monitor_pid=$!

    # Run load for duration
    sleep $DURATION

    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true

    # Analyze memory
    local max_mem=$(sort -t',' -k2 -n "$output" | tail -1 | cut -d',' -f2)
    local avg_mem=$(awk -F',' '{sum+=$2; count++} END {print sum/count}' "$output")

    log_success "Memory: ${avg_mem}KB average, ${max_mem}KB peak"

    # Validation (assuming < 500MB target)
    if (( max_mem > 512000 )); then
        log_warning "Memory usage exceeded 500MB target"
    fi
}

# 8. CPU Usage Monitoring
test_cpu_usage() {
    log_info "Monitoring CPU usage..."

    local output="$OUTPUT_DIR/cpu.txt"

    # Start CPU monitoring
    (
        while true; do
            local cpu=$(ps aux | grep "visionflow" | grep -v grep | awk '{print $3}')
            echo "$(date +%s),$cpu" >> "$output"
            sleep 5
        done
    ) &
    local monitor_pid=$!

    # Run load for duration
    sleep $DURATION

    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true

    # Analyze CPU
    local max_cpu=$(sort -t',' -k2 -n "$output" | tail -1 | cut -d',' -f2)
    local avg_cpu=$(awk -F',' '{sum+=$2; count++} END {print sum/count}' "$output")

    log_success "CPU: ${avg_cpu}% average, ${max_cpu}% peak"
}

# 9. Preset Application Load Test
test_preset_application() {
    log_info "Testing preset application..."

    local output="$OUTPUT_DIR/presets.txt"

    local presets=("low" "medium" "high" "ultra")

    for preset in "${presets[@]}"; do
        local start=$(date +%s%N)

        curl -X POST -H "Content-Type: application/json" \
            -d "{\"preset\":\"$preset\"}" \
            "$API_BASE/settings/preset" -s > /dev/null

        local end=$(date +%s%N)
        local duration=$(( (end - start) / 1000000 ))

        echo "Preset $preset: ${duration}ms" >> "$output"

        if (( duration > 500 )); then
            log_warning "Preset '$preset' application exceeded 500ms: ${duration}ms"
        fi
    done

    log_success "Preset application test complete"
}

# 10. Sustained Load Test
test_sustained_load() {
    log_info "Running sustained load test ($DURATION seconds)..."

    local output="$OUTPUT_DIR/sustained_load.txt"

    wrk -t$CONCURRENT_USERS -c$CONCURRENT_USERS -d${DURATION}s \
        --latency "$API_BASE/settings/" > "$output" 2>&1

    local rps=$(grep "Requests/sec" "$output" | awk '{print $2}')
    local latency_avg=$(grep "Latency" "$output" | awk '{print $2}')
    local latency_99=$(grep "99%" "$output" | awk '{print $2}')

    log_success "Sustained load: $rps req/s, ${latency_avg} avg, ${latency_99} p99"
}

# Generate Summary Report
generate_report() {
    log_info "Generating summary report..."

    local report="$OUTPUT_DIR/summary.md"

    cat > "$report" <<EOF
# VisionFlow Settings System - Load Test Report

**Test Suite**: $TEST_SUITE
**Date**: $(date)
**Duration**: ${DURATION}s
**Concurrent Users**: $CONCURRENT_USERS

## Results Summary

### Database Performance
$(cat "$OUTPUT_DIR/db_reads.txt" | grep "Requests per second" || echo "N/A")
$(cat "$OUTPUT_DIR/db_writes.txt" | grep "Requests per second" || echo "N/A")

### Search Performance
\`\`\`
$(cat "$OUTPUT_DIR/search.txt" 2>/dev/null || echo "N/A")
\`\`\`

### Hot-Reload
$(grep "Hot-reload" "$OUTPUT_DIR"/*.txt 2>/dev/null || echo "N/A")

### Resource Usage
- Memory: $(tail -1 "$OUTPUT_DIR/memory.txt" 2>/dev/null || echo "N/A")
- CPU: $(tail -1 "$OUTPUT_DIR/cpu.txt" 2>/dev/null || echo "N/A")

### Sustained Load
$(cat "$OUTPUT_DIR/sustained_load.txt" 2>/dev/null | grep -A 3 "Latency" || echo "N/A")

## Validation

EOF

    # Add validation checks
    local warnings=$(grep -r "\[WARNING\]" "$OUTPUT_DIR" | wc -l)
    local errors=$(grep -r "\[ERROR\]" "$OUTPUT_DIR" | wc -l)

    echo "- Warnings: $warnings" >> "$report"
    echo "- Errors: $errors" >> "$report"

    if [[ $errors -eq 0 && $warnings -eq 0 ]]; then
        echo -e "\n✅ **All tests passed!**" >> "$report"
    else
        echo -e "\n⚠️ **Some tests had warnings or errors**" >> "$report"
    fi

    log_success "Report generated: $report"
}

# Main execution
main() {
    log_info "Starting VisionFlow Settings Load Tests"
    log_info "Test suite: $TEST_SUITE"

    # Run tests
    test_db_reads
    test_db_writes
    test_search_performance
    test_hot_reload
    test_websocket_broadcast
    test_concurrent_updates
    test_memory_usage &
    test_cpu_usage &
    test_preset_application
    test_sustained_load

    # Wait for monitoring to complete
    wait

    # Generate report
    generate_report

    log_success "Load testing complete! Results in: $OUTPUT_DIR"
}

# Execute
main "$@"
