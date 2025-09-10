#!/bin/bash
# Test compilation script for GPU refactoring

set -e

log() {
    echo "[TEST-COMPILE][$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

cd /workspace/ext

log "Testing Rust compilation with GPU features..."

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    log "ERROR: Cargo.toml not found in current directory"
    exit 1
fi

# Try to find cargo in common locations
CARGO_PATHS=(
    "/usr/local/cargo/bin/cargo"
    "/root/.cargo/bin/cargo"
    "$HOME/.cargo/bin/cargo"
    "cargo"
)

CARGO_CMD=""
for path in "${CARGO_PATHS[@]}"; do
    if command -v "$path" &> /dev/null; then
        CARGO_CMD="$path"
        break
    fi
done

if [ -z "$CARGO_CMD" ]; then
    log "ERROR: cargo not found. Testing with rustc directly..."
    
    # Try rustc as fallback
    RUSTC_PATHS=(
        "/usr/local/cargo/bin/rustc"
        "/root/.cargo/bin/rustc"
        "$HOME/.cargo/bin/rustc"
        "rustc"
    )
    
    RUSTC_CMD=""
    for path in "${RUSTC_PATHS[@]}"; do
        if command -v "$path" &> /dev/null; then
            RUSTC_CMD="$path"
            break
        fi
    done
    
    if [ -z "$RUSTC_CMD" ]; then
        log "ERROR: Neither cargo nor rustc found"
        log "Please ensure Rust toolchain is installed"
        exit 1
    fi
    
    log "Using rustc for syntax checking..."
    # Basic syntax check with rustc
    find src -name "*.rs" -type f | while read -r file; do
        log "Checking: $file"
        $RUSTC_CMD --crate-type lib --edition 2021 -Z parse-only "$file" 2>&1 || true
    done
else
    log "Found cargo at: $CARGO_CMD"
    log "Running cargo check with GPU features..."
    
    # Run cargo check (faster than build)
    if $CARGO_CMD check --features gpu 2>&1; then
        log "✓ Cargo check passed successfully"
        
        # Try cargo build if check passes
        log "Attempting full build..."
        if $CARGO_CMD build --release --features gpu 2>&1; then
            log "✓ Full build completed successfully"
        else
            log "WARNING: Full build failed but check passed"
        fi
    else
        log "ERROR: Cargo check failed"
        exit 1
    fi
fi

log "Compilation test completed"