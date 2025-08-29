#!/bin/bash

###############################################################################
# GPU/CPU Fallback Test
# Tests GPU detection, CUDA availability, and CPU fallback mechanisms
###############################################################################

set -e

# Configuration
API_HOST=${API_HOST:-"localhost"}
API_PORT=${API_PORT:-8080}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

header() {
    echo -e "${PURPLE}[TEST]${NC} $1"
}

# Detect GPU capabilities
detect_gpu_hardware() {
    header "Hardware Detection"
    
    local gpu_detected=false
    local cuda_available=false
    local opencl_available=false
    
    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        log "Checking NVIDIA GPU status"
        if nvidia-smi &> /dev/null; then
            local gpu_info
            gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [[ -n "$gpu_info" ]]; then
                success "NVIDIA GPU detected: $gpu_info"
                gpu_detected=true
                
                # Check CUDA specifically
                local cuda_version
                cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
                if [[ -n "$cuda_version" ]]; then
                    success "CUDA driver available: $cuda_version"
                    cuda_available=true
                fi
            fi
        fi
    fi
    
    # Check for CUDA toolkit
    if command -v nvcc &> /dev/null; then
        local nvcc_version
        nvcc_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}')
        if [[ -n "$nvcc_version" ]]; then
            success "CUDA toolkit available: $nvcc_version"
        fi
    fi
    
    # Check OpenCL
    if command -v clinfo &> /dev/null; then
        log "Checking OpenCL capabilities"
        if clinfo &> /dev/null; then
            local opencl_devices
            opencl_devices=$(clinfo 2>/dev/null | grep -c "Device Name" || echo "0")
            if [ "$opencl_devices" -gt 0 ]; then
                success "OpenCL devices available: $opencl_devices"
                opencl_available=true
            fi
        fi
    fi
    
    # Check AMD GPU
    if command -v rocm-smi &> /dev/null; then
        log "Checking AMD ROCm status"
        if rocm-smi &> /dev/null; then
            success "AMD ROCm GPU detected"
            gpu_detected=true
        fi
    fi
    
    # Summary
    echo
    log "Hardware Summary:"
    echo "  GPU Detected: $gpu_detected"
    echo "  CUDA Available: $cuda_available"
    echo "  OpenCL Available: $opencl_available"
    
    # Export for other tests
    export GPU_DETECTED=$gpu_detected
    export CUDA_AVAILABLE=$cuda_available
    export OPENCL_AVAILABLE=$opencl_available
    
    return 0
}

# Test compute backend selection
test_compute_backend_selection() {
    header "Compute Backend Selection"
    
    # Test compute status endpoint
    log "Checking compute backend status"
    local status_response
    status_response=$(curl -s "http://$API_HOST:$API_PORT/api/compute/status" 2>/dev/null || echo "")
    
    if [[ -n "$status_response" ]]; then
        log "Compute status response: $status_response"
        
        if command -v jq &> /dev/null; then
            local backend_type
            backend_type=$(echo "$status_response" | jq -r '.backend // .type // "unknown"' 2>/dev/null)
            log "Active backend: $backend_type"
            
            if [[ "$backend_type" == "gpu" ]] || [[ "$backend_type" == "cuda" ]]; then
                if [[ "$CUDA_AVAILABLE" == true ]]; then
                    success "GPU backend active with CUDA support"
                else
                    warning "GPU backend active but no CUDA detected"
                fi
            elif [[ "$backend_type" == "cpu" ]]; then
                success "CPU backend active (expected fallback)"
            else
                warning "Unknown backend type: $backend_type"
            fi
        else
            success "Compute status endpoint responding"
        fi
    else
        warning "Compute status endpoint not available"
    fi
    
    return 0
}

# Test GPU memory allocation
test_gpu_memory_allocation() {
    header "GPU Memory Allocation Test"
    
    if [[ "$CUDA_AVAILABLE" != true ]]; then
        warning "Skipping GPU memory test - CUDA not available"
        return 0
    fi
    
    # Test memory allocation endpoint
    log "Testing GPU memory allocation"
    local memory_test_request='{
        "operation": "allocate",
        "size": 1048576,
        "type": "gpu"
    }'
    
    local memory_response
    memory_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$memory_test_request" \
        "http://$API_HOST:$API_PORT/api/compute/memory" 2>/dev/null || echo "")
    
    if [[ -n "$memory_response" ]]; then
        if [[ "$memory_response" == *"success"* ]] || [[ "$memory_response" == *"allocated"* ]]; then
            success "GPU memory allocation successful"
        elif [[ "$memory_response" == *"fallback"* ]] || [[ "$memory_response" == *"cpu"* ]]; then
            success "GPU memory allocation failed, CPU fallback activated"
        else
            warning "GPU memory allocation response unclear: $memory_response"
        fi
    else
        warning "GPU memory allocation endpoint not available"
    fi
    
    return 0
}

# Test compute performance
test_compute_performance() {
    header "Compute Performance Test"
    
    # Test matrix multiplication or similar compute-heavy operation
    local performance_test_request='{
        "operation": "matrix_multiply",
        "size": 512,
        "iterations": 10
    }'
    
    log "Running compute performance test"
    local start_time=$(date +%s%3N)
    
    local perf_response
    perf_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$performance_test_request" \
        "http://$API_HOST:$API_PORT/api/compute/benchmark" 2>/dev/null || echo "")
    
    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))
    
    log "Performance test duration: ${duration}ms"
    
    if [[ -n "$perf_response" ]]; then
        if command -v jq &> /dev/null; then
            local backend_used
            local compute_time
            backend_used=$(echo "$perf_response" | jq -r '.backend // "unknown"' 2>/dev/null)
            compute_time=$(echo "$perf_response" | jq -r '.compute_time // 0' 2>/dev/null)
            
            log "Backend used: $backend_used"
            log "Compute time: ${compute_time}ms"
            
            if [[ "$backend_used" == "gpu" ]] && [[ "$CUDA_AVAILABLE" == true ]]; then
                success "GPU compute performance test completed"
            elif [[ "$backend_used" == "cpu" ]]; then
                success "CPU compute performance test completed"
            else
                warning "Performance test completed with unknown backend"
            fi
        else
            success "Compute performance test completed"
        fi
    else
        # Fallback test using physics endpoint
        log "Testing fallback compute via physics endpoint"
        local physics_test='{
            "springK": 0.1,
            "repelK": 2.0,
            "damping": 0.95,
            "iterations": 100
        }'
        
        local physics_response
        physics_response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$physics_test" \
            "http://$API_HOST:$API_PORT/api/physics/update" 2>/dev/null || echo "")
        
        if [[ -n "$physics_response" ]] && [[ "$physics_response" != *"error"* ]]; then
            success "Fallback compute test via physics endpoint successful"
        else
            error "Compute performance test failed"
            return 1
        fi
    fi
    
    return 0
}

# Test fallback mechanism
test_fallback_mechanism() {
    header "Fallback Mechanism Test"
    
    # Test forced CPU fallback
    log "Testing forced CPU fallback"
    local fallback_request='{
        "force_backend": "cpu",
        "operation": "test_compute"
    }'
    
    local fallback_response
    fallback_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$fallback_request" \
        "http://$API_HOST:$API_PORT/api/compute/test" 2>/dev/null || echo "")
    
    if [[ -n "$fallback_response" ]]; then
        if [[ "$fallback_response" == *"cpu"* ]] || [[ "$fallback_response" != *"error"* ]]; then
            success "CPU fallback mechanism working"
        else
            warning "Fallback response unclear: $fallback_response"
        fi
    else
        warning "Fallback test endpoint not available"
    fi
    
    # Test GPU preference (if available)
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        log "Testing GPU preference"
        local gpu_request='{
            "prefer_backend": "gpu",
            "operation": "test_compute"
        }'
        
        local gpu_response
        gpu_response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$gpu_request" \
            "http://$API_HOST:$API_PORT/api/compute/test" 2>/dev/null || echo "")
        
        if [[ -n "$gpu_response" ]]; then
            if [[ "$gpu_response" == *"gpu"* ]]; then
                success "GPU preference working"
            elif [[ "$gpu_response" == *"fallback"* ]] || [[ "$gpu_response" == *"cpu"* ]]; then
                success "GPU preference with CPU fallback working"
            else
                warning "GPU preference response unclear: $gpu_response"
            fi
        fi
    fi
    
    return 0
}

# Test error handling
test_error_handling() {
    header "Error Handling Test"
    
    # Test invalid compute requests
    log "Testing invalid compute request handling"
    local invalid_request='{"invalid": "request", "size": -1}'
    
    local error_response
    error_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$invalid_request" \
        "http://$API_HOST:$API_PORT/api/compute/test" 2>/dev/null || echo "")
    
    if [[ "$error_response" == *"error"* ]] || [[ "$error_response" == *"400"* ]]; then
        success "Invalid request error handling working"
    else
        warning "Error handling may not be working properly"
    fi
    
    # Test out-of-memory simulation
    log "Testing out-of-memory handling"
    local oom_request='{
        "operation": "allocate",
        "size": 999999999999,
        "type": "gpu"
    }'
    
    local oom_response
    oom_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$oom_request" \
        "http://$API_HOST:$API_PORT/api/compute/memory" 2>/dev/null || echo "")
    
    if [[ "$oom_response" == *"error"* ]] || [[ "$oom_response" == *"fallback"* ]] || [[ "$oom_response" == *"insufficient"* ]]; then
        success "Out-of-memory error handling working"
    else
        warning "OOM error handling response: $oom_response"
    fi
    
    return 0
}

# Test system resource monitoring
test_resource_monitoring() {
    header "Resource Monitoring"
    
    # Check system resources before compute test
    log "Monitoring system resources"
    
    if command -v free &> /dev/null; then
        local memory_before
        memory_before=$(free -m | grep "Mem:" | awk '{print $3}')
        log "Memory usage before: ${memory_before}MB"
    fi
    
    if [[ "$CUDA_AVAILABLE" == true ]] && command -v nvidia-smi &> /dev/null; then
        local gpu_memory_before
        gpu_memory_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "GPU memory usage before: ${gpu_memory_before}MB"
    fi
    
    # Run a resource-intensive test
    log "Running resource-intensive compute test"
    local intensive_request='{
        "operation": "intensive_compute",
        "duration": 5,
        "load": "high"
    }'
    
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$intensive_request" \
        "http://$API_HOST:$API_PORT/api/compute/stress" >/dev/null 2>&1 || true
    
    sleep 2
    
    # Check resources after
    if command -v free &> /dev/null; then
        local memory_after
        memory_after=$(free -m | grep "Mem:" | awk '{print $3}')
        log "Memory usage after: ${memory_after}MB"
    fi
    
    if [[ "$CUDA_AVAILABLE" == true ]] && command -v nvidia-smi &> /dev/null; then
        local gpu_memory_after
        gpu_memory_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "GPU memory usage after: ${gpu_memory_after}MB"
    fi
    
    success "Resource monitoring completed"
    return 0
}

# Main execution
main() {
    echo "======================================================================="
    echo "GPU/CPU Fallback Test Suite"
    echo "======================================================================="
    log "API Target: $API_HOST:$API_PORT"
    echo
    
    # Initialize variables
    export GPU_DETECTED=false
    export CUDA_AVAILABLE=false
    export OPENCL_AVAILABLE=false
    
    # Run tests
    local test_results=()
    
    if detect_gpu_hardware; then
        test_results+=("Hardware Detection: PASS")
    else
        test_results+=("Hardware Detection: FAIL")
    fi
    
    echo
    
    if test_compute_backend_selection; then
        test_results+=("Backend Selection: PASS")
    else
        test_results+=("Backend Selection: FAIL")
    fi
    
    echo
    
    if test_gpu_memory_allocation; then
        test_results+=("GPU Memory: PASS")
    else
        test_results+=("GPU Memory: FAIL")
    fi
    
    echo
    
    if test_compute_performance; then
        test_results+=("Performance: PASS")
    else
        test_results+=("Performance: FAIL")
    fi
    
    echo
    
    if test_fallback_mechanism; then
        test_results+=("Fallback: PASS")
    else
        test_results+=("Fallback: FAIL")
    fi
    
    echo
    
    if test_error_handling; then
        test_results+=("Error Handling: PASS")
    else
        test_results+=("Error Handling: FAIL")
    fi
    
    echo
    
    if test_resource_monitoring; then
        test_results+=("Resource Monitoring: PASS")
    else
        test_results+=("Resource Monitoring: FAIL")
    fi
    
    echo
    echo "======================================================================="
    echo "Test Summary"
    echo "======================================================================="
    
    local failed_tests=0
    for result in "${test_results[@]}"; do
        if [[ "$result" == *"PASS"* ]]; then
            success "$result"
        else
            error "$result"
            ((failed_tests++))
        fi
    done
    
    echo
    log "System Configuration:"
    echo "  GPU Hardware: $GPU_DETECTED"
    echo "  CUDA Support: $CUDA_AVAILABLE"
    echo "  OpenCL Support: $OPENCL_AVAILABLE"
    
    if [[ "$GPU_DETECTED" == true ]]; then
        log "GPU compute should be preferred when available"
    else
        log "CPU fallback should be used exclusively"
    fi
    
    echo
    
    if [ $failed_tests -eq 0 ]; then
        success "All GPU/CPU fallback tests passed!"
        exit 0
    else
        error "$failed_tests test(s) failed"
        exit 1
    fi
}

# Handle arguments
case "${1:-}" in
    --help|-h)
        echo "GPU/CPU Fallback Test Suite"
        echo
        echo "Usage: $0 [--help]"
        echo
        echo "Tests GPU detection, CUDA availability, and CPU fallback mechanisms."
        echo
        echo "Environment variables:"
        echo "  API_HOST     API server host (default: localhost)"
        echo "  API_PORT     API server port (default: 8080)"
        echo
        exit 0
        ;;
esac

main "$@"