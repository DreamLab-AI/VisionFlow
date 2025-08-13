#!/bin/bash
# Start Rust backend with fallback compilation

cd /app

# Check if pre-built binary exists and is executable
if [ -x "/app/webxr" ]; then
    echo "[Rust Backend] Starting pre-built binary..."
    exec /app/webxr
else
    echo "[Rust Backend] Binary not found, compiling..."
    
    # Try to build the binary
    if cargo build --features gpu; then
        cp target/debug/webxr /app/webxr
        chmod +x /app/webxr
        echo "[Rust Backend] Build successful, starting..."
        exec /app/webxr
    else
        echo "[Rust Backend] Build failed, trying without GPU features..."
        if cargo build; then
            cp target/debug/webxr /app/webxr
            chmod +x /app/webxr
            echo "[Rust Backend] Build successful (no GPU), starting..."
            exec /app/webxr
        else
            echo "[Rust Backend] FATAL: Cannot build or run backend"
            exit 1
        fi
    fi
fi