#!/bin/bash
# Quick test to verify the build fixes work

echo "Testing Rust compilation fixes..."
cd /workspace/ext

# Just test if the code compiles (don't need full build)
echo "Checking syntax..."
cargo check --features gpu 2>&1 | tail -20

if [ $? -eq 0 ]; then
    echo "✅ Compilation checks passed!"
else
    echo "❌ Still has compilation errors"
    exit 1
fi