# Hybrid CPU-WASM/GPU SSSP Build Instructions

## Prerequisites

### System Requirements
- **Rust**: 1.75.0 or later
- **CUDA Toolkit**: 11.8 or later (for GPU kernels)
- **wasm-pack**: For building WASM modules
- **Node.js**: 16+ (for WASM runtime testing)

### Installation

#### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### 2. Add WASM Target
```bash
rustup target add wasm32-unknown-unknown
```

#### 3. Install wasm-pack
```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### 4. Install CUDA (if not present)
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

## Build Instructions

### 1. Standard Rust Build (CPU + GPU)
```bash
cd /workspace/ext
cargo build --release --features gpu
```

### 2. WASM Module Build
```bash
# Build WASM module for hybrid SSSP
wasm-pack build --target web --out-dir pkg/wasm \
  --features wasm \
  -- --no-default-features

# For Node.js environment
wasm-pack build --target nodejs --out-dir pkg/node \
  --features wasm \
  -- --no-default-features
```

### 3. CUDA Kernels Compilation
```bash
# Compile CUDA kernels to PTX
nvcc -ptx -arch=sm_70 \
  src/utils/visionflow_unified.cu \
  -o target/visionflow_unified.ptx

nvcc -ptx -arch=sm_70 \
  src/utils/sssp_compact.cu \
  -o target/sssp_compact.ptx

# For hybrid SSSP kernels (when implemented in CUDA)
# nvcc -ptx -arch=sm_70 \
#   src/gpu/hybrid_sssp/kernels.cu \
#   -o target/hybrid_sssp.ptx
```

### 4. Development Build with Checks
```bash
# Run cargo check for quick compilation verification
cargo check --all-features

# Build with all warnings
RUSTFLAGS="-W warnings" cargo build

# Build with specific GPU architecture
CUDA_ARCH=sm_80 cargo build --release --features gpu
```

## Configuration

### Cargo.toml Features
```toml
[features]
default = ["gpu"]
gpu = ["cudarc/driver"]     # GPU support with CUDA
wasm = []                    # WASM compilation
hybrid-sssp = ["gpu", "wasm"] # Hybrid SSSP feature
```

### Environment Variables
```bash
# CUDA configuration
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# WASM optimization
export WASM_OPT_FLAGS="-O3 --enable-simd"

# Hybrid SSSP configuration
export HYBRID_SSSP_ENABLE=true
export HYBRID_SSSP_USE_PINNED_MEMORY=true
```

## Testing

### Unit Tests
```bash
# Run all tests
cargo test --all-features

# Run hybrid SSSP tests specifically
cargo test --test hybrid_sssp_tests

# Run with logging
RUST_LOG=debug cargo test -- --nocapture
```

### WASM Tests
```bash
# Test WASM module in browser
cd pkg/wasm
python3 -m http.server 8000
# Open http://localhost:8000/test.html

# Test in Node.js
cd pkg/node
node test.js
```

### GPU Kernel Verification
```bash
# Verify PTX compilation
cuobjdump -ptx target/*.ptx

# Check kernel signatures
nvdisasm target/hybrid_sssp.ptx | grep ".entry"

# Profile kernel execution
nvprof cargo run --release --features gpu
```

## Benchmarking

### Performance Benchmarks
```bash
# Run criterion benchmarks
cargo bench --features gpu

# Specific hybrid SSSP benchmarks
cargo bench --bench hybrid_sssp_bench

# Generate flamegraph
cargo flamegraph --features gpu --bin webxr
```

### Complexity Verification
```bash
# Run complexity analysis tool
cargo run --bin verify_complexity -- \
  --algorithm hybrid-sssp \
  --graph-sizes "1000,10000,100000,1000000" \
  --expected "O(m*log^(2/3)*n)"
```

## Deployment

### Production Build
```bash
# Optimized release build
cargo build --release \
  --features "gpu hybrid-sssp" \
  -Z build-std=std,panic_abort \
  -Z build-std-features=panic_immediate_abort

# Strip debug symbols
strip target/release/webxr
```

### WASM Deployment
```bash
# Optimize WASM size
wasm-opt -O3 -o optimized.wasm pkg/wasm/webxr_bg.wasm

# Generate TypeScript bindings
wasm-bindgen --typescript --out-dir pkg/ts \
  target/wasm32-unknown-unknown/release/webxr.wasm
```

### Docker Build
```dockerfile
FROM rust:1.75 as builder

# Install CUDA
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Install wasm-pack
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

WORKDIR /app
COPY . .

# Build both native and WASM
RUN cargo build --release --features gpu
RUN wasm-pack build --target web --out-dir pkg/wasm

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
COPY --from=builder /app/target/release/webxr /usr/local/bin/
COPY --from=builder /app/pkg/wasm /var/www/wasm/

CMD ["webxr"]
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```bash
# Verify CUDA installation
nvcc --version

# Set CUDA_PATH explicitly
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
```

#### 2. WASM Build Fails
```bash
# Clear cache and rebuild
rm -rf target/wasm32-unknown-unknown
cargo clean
wasm-pack build --dev
```

#### 3. Compilation Errors
```bash
# Check with verbose output
RUST_BACKTRACE=1 cargo build -vv

# Fix common async recursion issue
# Convert recursive async functions to iterative
```

#### 4. PTX Compilation Issues
```bash
# Check GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv

# Use appropriate architecture
nvcc -ptx -arch=sm_XX ... # Replace XX with your GPU's compute capability
```

## Integration with Existing System

### Using Hybrid SSSP in Code
```rust
use webxr::gpu::hybrid_sssp::{HybridSSPExecutor, HybridSSPConfig};

// Configure hybrid SSSP
let config = HybridSSPConfig {
    enable_hybrid: true,
    use_pinned_memory: true,
    enable_profiling: cfg!(debug_assertions),
    ..Default::default()
};

// Initialize executor
let mut executor = HybridSSPExecutor::new(config);
executor.initialize().await?;

// Execute SSSP
let result = executor.execute(
    num_nodes,
    num_edges,
    &sources,
    &csr_row_offsets,
    &csr_col_indices,
    &csr_weights,
).await?;

// Use results
println!("Complexity factor: {:.2}x theoretical",
         result.metrics.complexity_factor);
```

### JavaScript Integration
```javascript
// Load WASM module
import init, { WASMSSPSolver } from './pkg/wasm/webxr.js';

await init();

// Create solver
const solver = new WASMSSPSolver();

// Solve SSSP
const result = await solver.solve_sssp(
    sources,
    numNodes,
    rowOffsets,
    colIndices,
    weights
);
```

## Performance Monitoring

### Enable Profiling
```rust
// In Cargo.toml
[profile.release]
debug = true  # Enable debug symbols in release

// Run with profiling
HYBRID_SSSP_PROFILE=true cargo run --release
```

### Metrics Output
```
=== Hybrid SSSP Performance Metrics ===
Total time: 234.56 ms
  CPU orchestration: 45.23 ms (19.3%)
  GPU computation: 156.78 ms (66.9%)
  CPU-GPU transfer: 32.55 ms (13.8%)
Recursion levels: 3
Pivots selected: 127
Total relaxations: 3234567
Complexity factor: 1.12x theoretical
=====================================
```

## Further Resources

- [Breaking the Sorting Barrier Paper](https://arxiv.org/abs/XXX)
- [WASM Documentation](https://rustwasm.github.io/docs/book/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Project Architecture](./hybrid_cpu_wasm_gpu_architecture.md)
- [Performance Analysis](./hybrid_sssp_performance_analysis.md)

---

*Last updated: 2025-09-15*