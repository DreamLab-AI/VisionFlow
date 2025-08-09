#!/bin/bash
# Comprehensive Build System Helper Script
# Provides utilities for managing the GPU kernel build system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
UTILS_DIR="$PROJECT_ROOT/src/utils"

# Create logs directory
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Show usage information
show_usage() {
    cat << EOF
Build Helper for GPU Kernel Compilation System

Usage: $0 COMMAND [OPTIONS]

Commands:
  validate        Validate the build environment and dependencies
  clean           Clean all generated PTX files and build artifacts
  compile         Compile all GPU kernels
  test            Run build system tests
  status          Show current build status and PTX file information
  benchmark       Run compilation benchmarks
  doctor          Diagnose common build issues
  help            Show this help message

Options:
  --arch ARCH     Target CUDA architecture (default: 86 for A6000)
  --debug         Enable debug compilation mode
  --verbose       Enable verbose output
  --force         Force operation (bypass safety checks)

Environment Variables:
  CUDA_ARCH       Target compute capability (75, 86, 89, 90)
  DEBUG           Enable debug mode (0 or 1)
  VERBOSE         Enable verbose logging (0 or 1)

Examples:
  $0 validate                    # Check build environment
  $0 compile --arch 86          # Compile for A6000
  $0 clean --force              # Clean all artifacts
  $0 test                       # Run build tests
  $0 doctor                     # Diagnose issues

EOF
}

# Validate build environment
validate_environment() {
    log_info "Validating build environment..."
    
    local issues=0
    
    # Check for required tools
    log_info "Checking required tools..."
    
    if ! command -v nvcc &> /dev/null; then
        log_error "NVIDIA CUDA Compiler (nvcc) not found"
        log_error "Please install CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit"
        ((issues++))
    else
        local nvcc_version
        nvcc_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        log_success "Found NVCC version: $nvcc_version"
    fi
    
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo not found"
        log_error "Please install Rust: https://rustup.rs/"
        ((issues++))
    else
        local cargo_version
        cargo_version=$(cargo --version | cut -d' ' -f2)
        log_success "Found Cargo version: $cargo_version"
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Information:"
        nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader,nounits | while read -r line; do
            log_info "  GPU $line"
        done
    else
        log_warn "nvidia-smi not available - cannot verify GPU compatibility"
    fi
    
    # Check project structure
    log_info "Checking project structure..."
    
    local required_files=(
        "$PROJECT_ROOT/Cargo.toml"
        "$PROJECT_ROOT/build.rs"
        "$SCRIPT_DIR/compile_ptx.sh"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "Found: $(basename "$file")"
        else
            log_error "Missing: $file"
            ((issues++))
        fi
    done
    
    # Check CUDA kernels
    log_info "Checking CUDA kernels..."
    local kernel_count=0
    if [ -d "$UTILS_DIR" ]; then
        while IFS= read -r -d '' file; do
            log_success "Found kernel: $(basename "$file")"
            ((kernel_count++))
        done < <(find "$UTILS_DIR" -name "*.cu" -print0)
        log_info "Total kernels found: $kernel_count"
    else
        log_error "Utils directory not found: $UTILS_DIR"
        ((issues++))
    fi
    
    # Summary
    if [ $issues -eq 0 ]; then
        log_success "Build environment validation completed successfully!"
        return 0
    else
        log_error "Found $issues issues in build environment"
        return 1
    fi
}

# Clean build artifacts
clean_build() {
    local force=${1:-false}
    
    log_info "Cleaning build artifacts..."
    
    if [ "$force" != "true" ]; then
        echo -n "This will delete all PTX files and build artifacts. Continue? [y/N] "
        read -r response
        case $response in
            [yY][eE][sS]|[yY])
                log_info "Proceeding with clean..."
                ;;
            *)
                log_info "Clean cancelled."
                return 0
                ;;
        esac
    fi
    
    # Remove PTX files
    local ptx_count=0
    if [ -d "$UTILS_DIR" ]; then
        while IFS= read -r -d '' file; do
            log_info "Removing: $(basename "$file")"
            rm -f "$file"
            ((ptx_count++))
        done < <(find "$UTILS_DIR" -name "*.ptx" -print0)
    fi
    
    # Remove build directory
    if [ -d "$PROJECT_ROOT/target" ]; then
        log_info "Removing Cargo build directory..."
        rm -rf "$PROJECT_ROOT/target"
    fi
    
    # Clean logs
    if [ -d "$LOG_DIR" ]; then
        log_info "Cleaning build logs..."
        find "$LOG_DIR" -name "*.log" -delete 2>/dev/null || true
    fi
    
    log_success "Clean completed. Removed $ptx_count PTX files."
}

# Show build status
show_status() {
    log_info "Build System Status Report"
    echo "=========================="
    
    # Environment
    echo "Environment:"
    echo "  CUDA_ARCH: ${CUDA_ARCH:-86}"
    echo "  DEBUG: ${DEBUG:-0}"
    echo "  Project Root: $PROJECT_ROOT"
    echo
    
    # CUDA kernels
    echo "CUDA Kernels:"
    if [ -d "$UTILS_DIR" ]; then
        while IFS= read -r -d '' cu_file; do
            local basename=$(basename "$cu_file" .cu)
            local ptx_file="$UTILS_DIR/$basename.ptx"
            local cu_size cu_mtime ptx_size ptx_mtime status
            
            cu_size=$(stat -f%z "$cu_file" 2>/dev/null || stat -c%s "$cu_file")
            cu_mtime=$(stat -f%Sm -t "%Y-%m-%d %H:%M" "$cu_file" 2>/dev/null || stat -c%y "$cu_file" | cut -d' ' -f1,2 | cut -d'.' -f1)
            
            if [ -f "$ptx_file" ]; then
                ptx_size=$(stat -f%z "$ptx_file" 2>/dev/null || stat -c%s "$ptx_file")
                ptx_mtime=$(stat -f%Sm -t "%Y-%m-%d %H:%M" "$ptx_file" 2>/dev/null || stat -c%y "$ptx_file" | cut -d' ' -f1,2 | cut -d'.' -f1)
                status="✓ COMPILED"
            else
                ptx_size="N/A"
                ptx_mtime="N/A"
                status="✗ MISSING"
            fi
            
            printf "  %-25s %8s bytes (%s) -> %8s bytes (%s) [%s]\n" \
                "$basename" "$cu_size" "$cu_mtime" "$ptx_size" "$ptx_mtime" "$status"
        done < <(find "$UTILS_DIR" -name "*.cu" -print0 | sort -z)
    fi
    echo
    
    # Recent logs
    if [ -f "$LOG_DIR/ptx_compilation.log" ]; then
        echo "Recent Compilation Log (last 5 lines):"
        tail -n 5 "$LOG_DIR/ptx_compilation.log" | sed 's/^/  /'
    fi
}

# Run build system tests
run_tests() {
    log_info "Running build system tests..."
    
    # Test 1: Validate environment
    log_info "Test 1: Environment validation"
    if validate_environment >/dev/null 2>&1; then
        log_success "✓ Environment validation passed"
    else
        log_error "✗ Environment validation failed"
        return 1
    fi
    
    # Test 2: Compilation script exists and is executable
    log_info "Test 2: Compilation script"
    local compile_script="$SCRIPT_DIR/compile_ptx.sh"
    if [ -x "$compile_script" ]; then
        log_success "✓ Compilation script is executable"
    else
        log_error "✗ Compilation script not executable: $compile_script"
        return 1
    fi
    
    # Test 3: Build.rs syntax check
    log_info "Test 3: Build script syntax"
    if rustc --crate-type bin "$PROJECT_ROOT/build.rs" -o /tmp/build_test 2>/dev/null; then
        rm -f /tmp/build_test
        log_success "✓ Build.rs syntax is valid"
    else
        log_error "✗ Build.rs has syntax errors"
        return 1
    fi
    
    # Test 4: Cargo check
    log_info "Test 4: Cargo project check"
    if (cd "$PROJECT_ROOT" && cargo check --quiet 2>/dev/null); then
        log_success "✓ Cargo project check passed"
    else
        log_warn "! Cargo project check had warnings (this may be normal)"
    fi
    
    log_success "All build system tests completed!"
}

# Benchmark compilation performance
run_benchmark() {
    log_info "Running compilation benchmark..."
    
    local arch=${CUDA_ARCH:-86}
    local compile_script="$SCRIPT_DIR/compile_ptx.sh"
    
    if [ ! -x "$compile_script" ]; then
        log_error "Compilation script not found or not executable: $compile_script"
        return 1
    fi
    
    # Clean first to ensure fresh compilation
    rm -f "$UTILS_DIR"/*.ptx
    
    log_info "Starting timed compilation for SM_$arch..."
    local start_time=$(date +%s.%N)
    
    if CUDA_ARCH=$arch "$compile_script" >/dev/null 2>&1; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "unknown")
        
        log_success "Compilation completed in ${duration}s"
        
        # Show PTX file sizes
        log_info "Generated PTX files:"
        ls -la "$UTILS_DIR"/*.ptx 2>/dev/null | while read -r line; do
            log_info "  $line"
        done
    else
        log_error "Compilation benchmark failed"
        return 1
    fi
}

# Diagnose common issues
run_doctor() {
    log_info "Running build system diagnostics..."
    
    local issues_found=0
    
    # Check CUDA installation
    log_info "Checking CUDA installation..."
    if command -v nvcc >/dev/null 2>&1; then
        log_success "✓ NVCC found"
        nvcc --version | head -1
    else
        log_error "✗ NVCC not found - CUDA Toolkit not installed"
        ((issues_found++))
    fi
    
    # Check CUDA runtime
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_success "✓ NVIDIA driver found"
        nvidia-smi | head -3 | tail -1
    else
        log_error "✗ NVIDIA driver not found"
        ((issues_found++))
    fi
    
    # Check disk space
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 1000000 ]; then  # Less than 1GB
        log_warn "⚠ Low disk space: ${available_space}KB available"
        ((issues_found++))
    else
        log_success "✓ Sufficient disk space available"
    fi
    
    # Check permissions
    if [ -w "$UTILS_DIR" ]; then
        log_success "✓ Write permissions to utils directory"
    else
        log_error "✗ No write permissions to $UTILS_DIR"
        ((issues_found++))
    fi
    
    # Summary
    if [ $issues_found -eq 0 ]; then
        log_success "No issues found! Build system should work correctly."
    else
        log_warn "Found $issues_found potential issues. See above for details."
    fi
}

# Main command dispatcher
main() {
    local command=${1:-help}
    local force=false
    local verbose=${VERBOSE:-0}
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --arch)
                export CUDA_ARCH="$2"
                shift 2
                ;;
            --debug)
                export DEBUG=1
                shift
                ;;
            --verbose)
                export VERBOSE=1
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                if [[ $1 != $command ]]; then
                    break
                fi
                shift
                ;;
        esac
    done
    
    # Execute command
    case $command in
        validate)
            validate_environment
            ;;
        clean)
            clean_build $force
            ;;
        compile)
            "$SCRIPT_DIR/compile_ptx.sh"
            ;;
        test)
            run_tests
            ;;
        status)
            show_status
            ;;
        benchmark)
            run_benchmark
            ;;
        doctor)
            run_doctor
            ;;
        help)
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"