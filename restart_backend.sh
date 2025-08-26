#!/bin/bash
# Script to restart the Rust backend after fixing the bloom/glow field issue

echo "=== Restarting Rust Backend ==="
echo "Stopping any existing processes..."

# Kill existing rust backend processes
pkill -f webxr || true
pkill -f rust-backend || true

echo "Starting backend..."
cd /workspace/ext

# Try to start the backend directly
if [ -f "./target/release/webxr" ]; then
    echo "Starting from release build..."
    RUST_LOG=error ./target/release/webxr &
elif [ -f "./target/debug/webxr" ]; then
    echo "Starting from debug build..."
    RUST_LOG=error ./target/debug/webxr &
else
    echo "No built binary found. The code changes need to be compiled."
    echo "Since cargo is not available in this environment, the changes will take effect"
    echo "when the system is rebuilt in the production environment."
fi

# Check if it started
sleep 2
if pgrep -f webxr > /dev/null; then
    echo "✅ Backend started successfully!"
    ps aux | grep webxr | grep -v grep
else
    echo "❌ Backend failed to start. Check logs at /workspace/ext/logs/rust-error.log"
    tail -20 /workspace/ext/logs/rust-error.log
fi