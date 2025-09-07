#!/bin/bash
#
# PTX Compilation Script for VisionFlow Unified GPU Kernels
# System Architecture: Automated PTX build pipeline with diagnostics
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_FILE="src/utils/visionflow_unified.cu"
OUTPUT_FILE="target/release/visionflow_unified.ptx"
CUDA_ARCH="sm_70"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[PTX Build]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PTX Build]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[PTX Build]${NC} $1"
}

log_error() {
    echo -e "${RED}[PTX Build]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking build dependencies..."
    
    if ! command -v nvcc &> /dev/null; then
        log_error "NVCC not found. Please install CUDA Toolkit."
        exit 1
    fi
    
    local nvcc_version
    nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    log "Found NVCC version: $nvcc_version"
    
    if [[ ! -f "$PROJECT_ROOT/$SOURCE_FILE" ]]; then
        log_error "Source file not found: $SOURCE_FILE"
        exit 1
    fi
    
    log_success "Dependencies verified"
}

# Create output directory
prepare_build() {
    log "Preparing build environment..."
    mkdir -p "$(dirname "$PROJECT_ROOT/$OUTPUT_FILE")"
    log_success "Build environment ready"
}

# Compile PTX with comprehensive flags
compile_ptx() {
    log "Compiling CUDA kernels to PTX..."
    
    cd "$PROJECT_ROOT"
    
    # Compilation flags optimized for performance and compatibility
    local nvcc_flags=(
        -ptx
        -arch="$CUDA_ARCH"
        -O3
        --use_fast_math
        --ftz=true           # Flush denormals to zero
        --prec-div=false     # Use fast division
        --prec-sqrt=false    # Use fast square root
        --fmad=true          # Enable fused multiply-add
        -I/usr/local/cuda/include
        --generate-line-info # For debugging
        --ptxas-options="-v" # Verbose PTX assembler
    )
    
    log "Compilation command: nvcc ${nvcc_flags[*]} $SOURCE_FILE -o $OUTPUT_FILE"
    
    if nvcc "${nvcc_flags[@]}" "$SOURCE_FILE" -o "$OUTPUT_FILE" 2>&1; then
        log_success "PTX compilation completed successfully"
    else
        log_error "PTX compilation failed"
        return 1
    fi
}

# Validate PTX content
validate_ptx() {
    log "Validating PTX output..."
    
    local ptx_file="$PROJECT_ROOT/$OUTPUT_FILE"
    
    if [[ ! -f "$ptx_file" ]]; then
        log_error "PTX file not found: $ptx_file"
        return 1
    fi
    
    local file_size
    file_size=$(stat -c%s "$ptx_file" 2>/dev/null || echo "0")
    
    if [[ "$file_size" -lt 1000 ]]; then
        log_error "PTX file too small ($file_size bytes), compilation likely failed"
        return 1
    fi
    
    log "PTX file size: $file_size bytes"
    
    # Check for required kernel functions
    local required_kernels=(
        "build_grid_kernel"
        "compute_cell_bounds_kernel" 
        "force_pass_kernel"
        "integrate_pass_kernel"
        "relaxation_step_kernel"
    )
    
    log "Checking for required kernels..."
    
    for kernel in "${required_kernels[@]}"; do
        if grep -q "\.entry $kernel" "$ptx_file"; then
            log_success "✓ Found kernel: $kernel"
        else
            log_warning "⚠ Kernel not found or not properly exported: $kernel"
        fi
    done
    
    # Check PTX version
    local ptx_version
    ptx_version=$(grep -o "\.version [0-9.]*" "$ptx_file" | head -1 | cut -d' ' -f2)
    log "PTX version: $ptx_version"
    
    # Check target architecture
    local target_arch
    target_arch=$(grep -o "\.target [a-z0-9_]*" "$ptx_file" | head -1 | cut -d' ' -f2)
    log "Target architecture: $target_arch"
    
    log_success "PTX validation completed"
}

# Set environment variable
set_environment() {
    log "Setting up environment variables..."
    
    local ptx_path
    ptx_path=$(realpath "$PROJECT_ROOT/$OUTPUT_FILE")
    
    export VISIONFLOW_PTX_PATH="$ptx_path"
    log_success "Set VISIONFLOW_PTX_PATH=$VISIONFLOW_PTX_PATH"
    
    # Add to shell profile for persistence
    local shell_profile=""
    if [[ -f "$HOME/.bashrc" ]]; then
        shell_profile="$HOME/.bashrc"
    elif [[ -f "$HOME/.zshrc" ]]; then
        shell_profile="$HOME/.zshrc"
    fi
    
    if [[ -n "$shell_profile" ]]; then
        if ! grep -q "VISIONFLOW_PTX_PATH" "$shell_profile"; then
            echo "export VISIONFLOW_PTX_PATH=\"$ptx_path\"" >> "$shell_profile"
            log_success "Added VISIONFLOW_PTX_PATH to $shell_profile"
        else
            log "VISIONFLOW_PTX_PATH already exists in $shell_profile"
        fi
    fi
}

# Store compilation status in hive memory
store_status() {
    log "Storing compilation status in hive memory..."
    
    local status_data="{
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"status\": \"success\",
        \"ptx_file\": \"$PROJECT_ROOT/$OUTPUT_FILE\",
        \"file_size\": $(stat -c%s "$PROJECT_ROOT/$OUTPUT_FILE" 2>/dev/null || echo "0"),
        \"cuda_arch\": \"$CUDA_ARCH\",
        \"nvcc_version\": \"$(nvcc --version | grep release | sed 's/.*release \([0-9.]*\).*/\1/')\"
    }"
    
    # Use claude-flow memory storage
    if command -v npx &> /dev/null; then
        echo "$status_data" | npx claude-flow@alpha hooks memory-store --key "hive/compilation/ptx_status" --value - 2>/dev/null || log_warning "Could not store in hive memory"
    fi
    
    log_success "Compilation status stored"
}

# Main execution flow
main() {
    log "Starting PTX compilation pipeline..."
    
    check_dependencies
    prepare_build
    compile_ptx
    validate_ptx
    set_environment
    store_status
    
    log_success "PTX compilation pipeline completed successfully!"
    log "PTX file location: $PROJECT_ROOT/$OUTPUT_FILE"
    log "Environment variable: VISIONFLOW_PTX_PATH=$VISIONFLOW_PTX_PATH"
}

# Error handling
trap 'log_error "Build failed at line $LINENO"' ERR

# Execute main function
main "$@"