#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[ENTRYPOINT][$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check for GPU availability
log "Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
    if [ -n "$GPU_INFO" ]; then
        log "GPU detected: $GPU_INFO"
        nvidia-smi
    else
        log "WARNING: No GPU detected, running in CPU mode"
    fi
else
    log "WARNING: nvidia-smi not available"
fi

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Ensure settings file exists
if [ ! -f /app/settings.yaml ]; then
    log "ERROR: settings.yaml not found at /app/settings.yaml"
    exit 1
fi

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Start services with supervisord
log "Starting production environment services..."
exec supervisord -c /app/supervisord.production.conf