#!/bin/bash
# Generate Cargo.lock file for fresh builds

set -e

echo "=== Generating Cargo.lock ==="

# Check if we're in the correct directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found. Please run from project root."
    exit 1
fi

# Check if Cargo.lock already exists
if [ -f "Cargo.lock" ]; then
    echo "Cargo.lock already exists."
    read -p "Do you want to regenerate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing Cargo.lock"
        exit 0
    fi
    echo "Removing existing Cargo.lock..."
    rm -f Cargo.lock
fi

# Generate Cargo.lock
echo "Generating new Cargo.lock file..."
if command -v cargo >/dev/null 2>&1; then
    # Try to generate lock file without building
    cargo generate-lockfile 2>/dev/null || cargo metadata --format-version 1 > /dev/null 2>&1 || true
    
    if [ -f "Cargo.lock" ]; then
        echo "✓ Cargo.lock generated successfully"
        echo "File size: $(wc -c < Cargo.lock) bytes"
    else
        echo "⚠ Warning: Cargo.lock generation may have failed"
        echo "The lock file will be generated during the Docker build"
    fi
else
    echo "⚠ Cargo not available in this environment"
    echo "Cargo.lock will be generated during Docker build"
fi

echo ""
echo "Note: The Docker build will generate Cargo.lock automatically if it's missing."
echo "This is normal for fresh checkouts or after dependency updates."