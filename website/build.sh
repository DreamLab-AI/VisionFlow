#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Building wasm-mesh-hero..."
wasm-pack build wasm-mesh-hero --target web --out-dir ../dist/wasm/mesh-hero --no-typescript --release

echo "==> Building wasm-particle-field..."
wasm-pack build wasm-particle-field --target web --out-dir ../dist/wasm/particle-field --no-typescript --release

echo "==> Copying static assets..."
cp -r static/* dist/

echo "==> Writing CNAME..."
echo "www.visionflow.info" > dist/CNAME

echo "==> Copying repo images..."
mkdir -p dist/img
cp -r ../assets/diagrams/* dist/img/ 2>/dev/null || true
cp -r ../assets/generated/* dist/img/ 2>/dev/null || true
cp -r ../assets/heroes/* dist/img/ 2>/dev/null || true
cp -r ../assets/screenshots/* dist/img/ 2>/dev/null || true

echo "==> Build complete. Output in dist/"
ls -la dist/
