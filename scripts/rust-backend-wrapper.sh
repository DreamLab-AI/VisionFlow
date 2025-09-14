#!/bin/bash
# Wrapper script for rust-backend that ensures rebuild on startup
# This is used by supervisord in development mode

set -e

# Set Docker environment variable to ensure PTX compilation at runtime
export DOCKER_ENV=1

log() {
    echo "[RUST-WRAPPER][$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Always rebuild in dev mode unless explicitly skipped
if [ "${SKIP_RUST_REBUILD:-false}" != "true" ]; then
    log "Rebuilding Rust backend with GPU support to apply code changes..."
    cd /app
    
    # Build with GPU features (debug build for development)
    if cargo build --features gpu; then
        log "✓ Rust backend rebuilt successfully (debug build)"
    else
        log "ERROR: Failed to rebuild Rust backend"
        exit 1
    fi
    
    RUST_BINARY="/app/target/debug/webxr"
else
    log "Skipping Rust rebuild (SKIP_RUST_REBUILD=true)"
    RUST_BINARY="/app/webxr"
fi

# Verify binary exists
if [ ! -f "${RUST_BINARY}" ]; then
    log "ERROR: Rust binary not found at ${RUST_BINARY}"
    exit 1
fi

log "Starting Rust backend from ${RUST_BINARY}..."
exec ${RUST_BINARY}