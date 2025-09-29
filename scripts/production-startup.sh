#!/bin/bash
set -e

echo "[STARTUP] Starting production environment..."

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check GPU availability
log "Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
    if [ -n "$GPU_INFO" ]; then
        log "GPU detected: $GPU_INFO"
    fi
fi

# Create necessary directories
mkdir -p /app/logs /var/log/nginx /var/run/nginx

# Build the production binary if not exists
if [ ! -f /app/target/release/webxr ]; then
    log "Building production binary..."
    cd /app
    cargo build --release --features gpu
else
    log "Using existing production binary"
fi

# Copy PTX files from build output to expected locations
log "Ensuring PTX files are in place..."
mkdir -p /app/src/utils/ptx
find /app/target/release/build -name 'visionflow_unified.ptx' -exec cp {} /app/src/utils/ptx/ \; 2>/dev/null || true
if [ -f /app/src/utils/ptx/visionflow_unified.ptx ]; then
    log "PTX file copied successfully"
    ls -la /app/src/utils/ptx/
else
    log "WARNING: PTX file not found in build output"
fi

# Use supervisord for production
if [ -f /app/supervisord.production.conf ]; then
    log "Starting production services with supervisord..."
    exec supervisord -c /app/supervisord.production.conf
else
    # Fallback to direct execution
    log "Starting services directly (no supervisord config found)..."

    # Start Rust backend on port 4001
    log "Starting Rust backend on port 4001..."
    RUST_LOG=warn /app/target/release/webxr --port 4001 --gpu-debug &
    BACKEND_PID=$!

    # Wait for backend to be ready
    log "Waiting for backend to start..."
    for i in {1..30}; do
        if nc -z localhost 4001; then
            log "Backend is ready on port 4001"
            break
        fi
        sleep 1
    done

    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        log "ERROR: Backend crashed during startup"
        exit 1
    fi

    # Start nginx on port 4000 to serve frontend and proxy API
    log "Starting nginx on port 4000..."
    nginx -g "daemon off;"
fi