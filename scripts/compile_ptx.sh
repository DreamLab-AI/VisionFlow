#!/bin/bash
# Comprehensive GPU Kernel Compilation Script
# Optimized for NVIDIA A6000 (SM_86) and other compute capabilities

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
CUDA_ARCH=${CUDA_ARCH:-86}  # Default to A6000 (SM_86)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UTILS_DIR="$PROJECT_ROOT/src/utils"
LOG_FILE="$PROJECT_ROOT/logs/ptx_compilation.log"

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Logging functions
log_info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

log_warn() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] $*" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v nvcc &> /dev/null; then
        log_error "nvcc not found. Please install CUDA Toolkit."
        exit 1
    fi
    
    local nvcc_version
    nvcc_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    log_info "Found NVCC version: $nvcc_version"
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found. Cannot verify GPU compatibility."
    else
        log_info "Available GPUs:"
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tee -a "$LOG_FILE"
    fi
}

# Validate compute capability
validate_compute_capability() {
    case $CUDA_ARCH in
        75) log_info "Target: RTX 20 series / Tesla T4 (SM_75)" ;;
        86) log_info "Target: RTX 30 series / A6000 / A100 (SM_86)" ;;
        89) log_info "Target: RTX 40 series / H100 (SM_89)" ;;
        90) log_info "Target: H100 / H200 (SM_90)" ;;
        *) log_warn "Unknown compute capability: SM_$CUDA_ARCH. Proceeding anyway..." ;;
    esac
}

# Compile a single kernel
compile_kernel() {
    local kernel_name="$1"
    local source_file="$UTILS_DIR/${kernel_name}.cu"
    local output_file="$UTILS_DIR/${kernel_name}.ptx"
    
    if [ ! -f "$source_file" ]; then
        log_error "Kernel source not found: $source_file"
        return 1
    fi
    
    log_info "Compiling kernel: $kernel_name"
    
    # Common NVCC flags optimized for research-grade performance
    local nvcc_flags=(
        -arch=sm_${CUDA_ARCH}
        -O3
        --use_fast_math
        -ptx
        -rdc=true
        --compiler-options -fPIC
        --compiler-options -Wall
        --compiler-options -Wextra
        -lineinfo
        --ptxas-options=-v
        -DCUDA_ARCH=${CUDA_ARCH}
    )
    
    # Add debug info in debug builds
    if [[ "${DEBUG:-0}" == "1" ]]; then
        nvcc_flags+=(
            -g
            -G
            --device-debug
        )
    fi
    
    # Compile with detailed output
    local compile_output
    if compile_output=$(nvcc "${nvcc_flags[@]}" "$source_file" -o "$output_file" 2>&1); then
        log_info "Successfully compiled $kernel_name"
        echo "$compile_output" >> "$LOG_FILE"
        
        # Set appropriate permissions
        chmod 644 "$output_file"
        
        # Log PTX file size
        local ptx_size
        ptx_size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file")
        log_info "Generated PTX size: $ptx_size bytes"
        
        return 0
    else
        log_error "Failed to compile $kernel_name"
        log_error "$compile_output"
        return 1
    fi
}

# Main compilation function
main() {
    log_info "Starting GPU kernel compilation (SM_$CUDA_ARCH)"
    log_info "Project root: $PROJECT_ROOT"
    
    check_prerequisites
    validate_compute_capability
    
    # Discover kernels: use CLI args that exist, otherwise auto-detect *.cu files
    local kernels=()
    if [ "$#" -gt 0 ]; then
        for name in "$@"; do
            if [ -f "$UTILS_DIR/${name}.cu" ]; then
                kernels+=("$name")
            else
                log_warn "Requested kernel '$name' not found at $UTILS_DIR/${name}.cu; skipping."
            fi
        done
    fi
    
    if [ "${#kernels[@]}" -eq 0 ]; then
        while IFS= read -r cu; do
            kernels+=("$(basename "$cu" .cu)")
        done < <(find "$UTILS_DIR" -maxdepth 1 -type f -name "*.cu" | sort)
    fi
    
    local success_count=0
    local total_kernels=${#kernels[@]}
    
    if [ "$total_kernels" -eq 0 ]; then
        log_warn "No CUDA .cu files found in $UTILS_DIR. Nothing to compile."
        exit 0
    fi
    
    log_info "Found $total_kernels kernels to compile"
    
    # Compile each kernel
    for kernel in "${kernels[@]}"; do
        if compile_kernel "$kernel"; then
            ((success_count++))
        else
            log_error "Compilation failed for $kernel"
            if [[ "${FAIL_FAST:-1}" == "1" ]]; then
                log_error "Stopping compilation due to failure (FAIL_FAST=1)"
                exit 1
            fi
        fi
        echo  # Add spacing between kernels
    done
    
    # Summary
    log_info "Compilation Summary:"
    log_info "  Successful: $success_count/$total_kernels"
    log_info "  Failed: $((total_kernels - success_count))/$total_kernels"
    
    if [ $success_count -eq $total_kernels ]; then
        log_info "All kernels compiled successfully!"
        
        # List generated PTX files
        log_info "Generated PTX files:"
        ls -la "$UTILS_DIR"/*.ptx | tee -a "$LOG_FILE"
        
        exit 0
    else
        log_error "Some kernels failed to compile. Check the log for details."
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --clean        Remove existing PTX files before compilation"
        echo "  --debug        Enable debug compilation flags"
        echo ""
        echo "Environment Variables:"
        echo "  CUDA_ARCH      Target compute capability (default: 86 for A6000)"
        echo "  FAIL_FAST      Stop on first failure (default: 1)"
        echo "  DEBUG          Enable debug flags (default: 0)"
        exit 0
        ;;
    --clean)
        log_info "Cleaning existing PTX files..."
        rm -f "$UTILS_DIR"/*.ptx
        shift
        ;;
    --debug)
        export DEBUG=1
        log_info "Debug mode enabled"
        shift
        ;;
esac

# Run main function
main "$@"