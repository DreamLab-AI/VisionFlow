#!/bin/bash
# Build test script for VisionFlow with updated dependencies

set -e

echo "=== VisionFlow Build Test ==="
echo "Testing Rust compilation with updated dependencies..."
echo ""

# Check if we're in the correct directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found. Please run from project root."
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf target/
rm -f Cargo.lock

# Check Rust version
echo "Rust version:"
rustc --version || echo "Rust not installed"
cargo --version || echo "Cargo not installed"
echo ""

# Attempt build with minimal features (no GPU to test in container)
echo "Building with CPU-only features..."
if command -v cargo >/dev/null 2>&1; then
    RUST_BACKTRACE=1 cargo build --features cpu --no-default-features 2>&1 | head -100
else
    echo "Cargo not available in this environment"
fi

echo ""
echo "Build test complete!"