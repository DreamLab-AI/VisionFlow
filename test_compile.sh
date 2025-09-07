#!/bin/bash
echo "Testing CUDA compilation..."
nvcc -ptx -arch sm_75 -o /tmp/test.ptx src/utils/visionflow_unified.cu --use_fast_math -O3 2>&1
if [ $? -eq 0 ]; then
    echo "✅ CUDA compilation successful"
else
    echo "❌ CUDA compilation failed"
    exit 1
fi

echo "Checking Rust syntax..."
/home/ubuntu/.cargo/bin/cargo check --message-format short 2>&1 | grep -E "error\[E" | head -5
if [ $? -eq 0 ]; then
    echo "⚠️  Rust errors found (showing first 5)"
else
    echo "✅ No Rust compilation errors detected"
fi