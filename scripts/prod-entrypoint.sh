#!/bin/bash
set -e

echo "[PROD-ENTRYPOINT] Starting production environment..."

# Build the rust backend fresh for this container's architecture
echo "[PROD-ENTRYPOINT] Building Rust backend for production..."
cd /app
cargo build --release --features gpu

# Copy PTX files to expected location
echo "[PROD-ENTRYPOINT] Copying PTX files..."
mkdir -p /app/src/utils/ptx
find /app/target/release/build -name 'visionflow_unified.ptx' -exec cp {} /app/src/utils/ptx/ \; 2>/dev/null || true

# Build client for production
echo "[PROD-ENTRYPOINT] Building client for production..."
cd /app/client
npm run build

# Start services with supervisord
echo "[PROD-ENTRYPOINT] Starting services with supervisord..."
exec supervisord -c /app/supervisord.production.conf