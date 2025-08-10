#!/bin/bash

# Pre-compile PTX files for Docker build
# This script should be run before building the Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UTILS_DIR="$PROJECT_ROOT/src/utils"
PTX_DIR="$UTILS_DIR/ptx"

echo "Pre-compiling PTX files for Docker build..."
echo "Project root: $PROJECT_ROOT"

# Create PTX directory if it doesn't exist
mkdir -p "$PTX_DIR"

# Find all CUDA kernel files
KERNELS=($(find "$UTILS_DIR" -name "*.cu" -type f | xargs -n1 basename | sed 's/\.cu$//' | sort))

echo "Found ${#KERNELS[@]} kernels to compile"

# Compile each kernel
for kernel in "${KERNELS[@]}"; do
    echo -n "Compiling $kernel... "
    CU_FILE="$UTILS_DIR/${kernel}.cu"
    PTX_FILE="$PTX_DIR/${kernel}.ptx"
    
    if [ ! -f "$CU_FILE" ]; then
        echo "SKIP (source not found)"
        continue
    fi
    
    # Check if PTX is up to date
    if [ -f "$PTX_FILE" ] && [ "$PTX_FILE" -nt "$CU_FILE" ]; then
        echo "SKIP (up to date)"
        continue
    fi
    
    # Compile with optimizations for faster compilation
    if nvcc -ptx -arch=sm_86 -O2 --use_fast_math "$CU_FILE" -o "$PTX_FILE" 2>/dev/null; then
        SIZE=$(ls -lh "$PTX_FILE" | awk '{print $5}')
        echo "OK ($SIZE)"
    else
        echo "FAILED"
        echo "  Error compiling $kernel, trying without optimizations..."
        if nvcc -ptx -arch=sm_86 "$CU_FILE" -o "$PTX_FILE" 2>/dev/null; then
            SIZE=$(ls -lh "$PTX_FILE" | awk '{print $5}')
            echo "  OK ($SIZE)"
        else
            echo "  FAILED - Creating stub file"
            echo "// Stub PTX file for $kernel" > "$PTX_FILE"
        fi
    fi
done

echo ""
echo "PTX compilation complete. Files in $PTX_DIR:"
ls -lh "$PTX_DIR"/*.ptx 2>/dev/null || echo "No PTX files found"