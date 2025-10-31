# VisionFlow - Build System & Dependency Analysis

**Analysis Date:** 2025-10-23
**Project:** VisionFlow (WebXR Knowledge Graph Platform)
**Repository:** `/home/devuser/workspace/project`

---

## Table of Contents

1. [Build System Overview](#1-build-system-overview)
2. [Rust Backend Build Process](#2-rust-backend-build-process)
3. [Frontend Build Process](#3-frontend-build-process)
4. [CUDA/GPU Compilation](#4-cudagpu-compilation)
5. [Docker Build Strategy](#5-docker-build-strategy)
6. [Dependency Analysis](#6-dependency-analysis)
7. [Build Scripts Reference](#7-build-scripts-reference)
8. [Configuration Files](#8-configuration-files)
9. [Known Build Issues](#9-known-build-issues)
10. [Build Performance Optimization](#10-build-performance-optimization)

---

## 1. Build System Overview

### Multi-Stage Build Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Build Stage                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 1. Base Image: nvidia/cuda:12.4.1-devel-ubuntu   │   │
│  │ 2. System Dependencies (apt-get)                 │   │
│  │ 3. Rust Toolchain (rustup)                       │   │
│  │ 4. Node.js 20.x (nodesource)                     │   │
│  │ 5. Source Code Copy                              │   │
│  │ 6. Cargo Fetch (Rust deps)                       │   │
│  │ 7. NPM Install (Frontend deps)                   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                Runtime Build Stage                      │
│  (Executed on every container start)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 1. Check GPU availability                        │   │
│  │ 2. Compile CUDA kernels to PTX (build.rs)       │   │
│  │ 3. Build Rust backend (cargo build --release)    │   │
│  │ 4. Generate TypeScript types (specta)           │   │
│  │ 5. Start Rust server (port 4000)                │   │
│  │ 6. Start Vite dev server (port 5173)            │   │
│  │ 7. Start Nginx reverse proxy (port 3001)        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Build Stages Summary

| Stage | When | Duration | Cached |
|-------|------|----------|--------|
| **Docker Build** | `docker-compose build` | 10-15 min | ✅ Yes (layers) |
| **Rust Compilation** | Every container start | 3-5 min | ✅ Yes (target/) |
| **CUDA PTX Generation** | With Rust build | 1-2 min | ✅ Yes (OUT_DIR) |
| **TypeScript Build** | With Vite | 10-30 sec | ✅ Yes (node_modules) |
| **Service Startup** | After builds | 10-15 sec | N/A |

---

## 2. Rust Backend Build Process

### Cargo Configuration

**File:** `Cargo.toml`

```toml
[package]
name = "webxr"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]  # Dual library types

[features]
default = ["gpu", "ontology"]
gpu = ["cudarc", "cust", "cust_core"]
ontology = ["horned-owl", "horned-functional", "whelk", "walkdir", "clap"]
redis = ["dep:redis"]

[profile.release]
opt-level = 3        # Maximum optimization
lto = true           # Link-time optimization
codegen-units = 1    # Single code generation unit
panic = "unwind"     # Unwind on panic
strip = true         # Strip symbols
```

### Build Script (build.rs)

**Purpose:** Compile 8 CUDA kernels to PTX format

**Process:**

1. **Check GPU Feature:** Skip if GPU feature disabled
2. **Locate CUDA Toolkit:** `/usr/local/cuda` or `$CUDA_PATH`
3. **Determine Architecture:** `$CUDA_ARCH` (default: 75)
4. **Compile Each Kernel:**
   ```bash
   nvcc -ptx -arch sm_86 -o output.ptx source.cu --use_fast_math -O3
   ```
5. **Verify PTX Files:** Check size > 0 bytes
6. **Export Environment Variables:** `{KERNEL}_PTX_PATH`
7. **Compile Thrust Wrapper:** For legacy compatibility
8. **Device Linking:** Required for Thrust library
9. **Create Static Library:** `libthrust_wrapper.a`
10. **Link CUDA Libraries:** `cudart`, `cuda`, `cudadevrt`, `stdc++`

### CUDA Kernel Compilation

**Kernels Compiled:**

| Kernel File | Output PTX | Purpose |
|-------------|------------|---------|
| `visionflow_unified.cu` | `visionflow_unified.ptx` | Unified physics engine |
| `gpu_clustering_kernels.cu` | `gpu_clustering_kernels.ptx` | Leiden clustering |
| `dynamic_grid.cu` | `dynamic_grid.ptx` | Spatial hashing grid |
| `gpu_aabb_reduction.cu` | `gpu_aabb_reduction.ptx` | AABB reduction |
| `gpu_landmark_apsp.cu` | `gpu_landmark_apsp.ptx` | All-pairs shortest path |
| `sssp_compact.cu` | `sssp_compact.ptx` | Single-source shortest path |
| `visionflow_unified_stability.cu` | `visionflow_unified_stability.ptx` | Stability optimization |
| `ontology_constraints.cu` | `ontology_constraints.ptx` | Semantic validation |

**NVCC Flags:**
- `-ptx` - Generate PTX intermediate representation
- `-arch sm_86` - Target Ampere architecture (configurable)
- `-o output.ptx` - Output file
- `--use_fast_math` - Fast math operations
- `-O3` - Maximum optimization
- `-Xcompiler -fPIC` - Position-independent code
- `-dc` - Device code linking (for Thrust)

### Dependency Compilation

**Major Dependencies with Build Steps:**

| Dependency | Build Type | Notes |
|------------|------------|-------|
| `cudarc` | Native | CUDA driver bindings |
| `rusqlite` | Native (bundled) | Includes SQLite C library |
| `horned-owl` | Pure Rust | OWL ontology parser |
| `whelk-rs` | Local path | Reasoning engine (local dependency) |
| `actix-web` | Pure Rust | Web framework |

### Build Command Examples

**Development Build:**
```bash
cargo build --features gpu,ontology
```

**Release Build:**
```bash
cargo build --release --features gpu
```

**CPU-Only Build:**
```bash
cargo build --release --no-default-features --features ontology
```

**Type Generation:**
```bash
cargo run --bin generate_types
```

---

## 3. Frontend Build Process

### Package Configuration

**File:** `client/package.json`

```json
{
  "name": "visionflow-client",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "npm run types:generate && vite build",
    "types:generate": "cd .. && cargo run --bin generate_types",
    "preinstall": "node scripts/block-test-packages.cjs",
    "security:check": "node scripts/block-test-packages.cjs && npm audit"
  }
}
```

### Vite Configuration

**File:** `client/vite.config.ts`

**Key Settings:**

```typescript
{
  server: {
    host: '0.0.0.0',
    port: 5173,
    hmr: {
      clientPort: 3001,        // Proxy through Nginx
      path: '/vite-hmr'        // WebSocket path
    },
    watch: {
      usePolling: true,        // For Docker
      interval: 1000
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
}
```

### Build Process Flow

1. **Security Check:** Block vulnerable test packages
2. **Type Generation:** Run Rust type generator (specta)
3. **TypeScript Compilation:** TSC type checking
4. **Vite Build:** Bundle React application
5. **Asset Optimization:** Minify JS/CSS, optimize images
6. **Output:** `client/dist/` directory

### Frontend Dependencies

**Core Framework:**
- React 18.2.0
- TypeScript 5.8.3
- Vite 6.2.6

**3D Rendering:**
- Three.js 0.175.0
- @react-three/fiber 8.15.0
- @react-three/drei 9.80.0
- @babylonjs/core 8.28.0

**UI Components:**
- @radix-ui/* (multiple packages)
- Tailwind CSS 4.1.3
- Framer Motion 12.6.5

**Build Tools:**
- @vitejs/plugin-react 4.3.4
- PostCSS 8.5.3
- Autoprefixer 10.4.21

---

## 4. CUDA/GPU Compilation

### CUDA Toolkit Requirements

**Version:** 12.4.1
**Components Required:**
- nvcc (CUDA compiler)
- CUDA driver API
- CUDA runtime API
- Device runtime (cudadevrt) for Thrust

### Architecture Targets

**Supported Compute Capabilities:**

| Architecture | Compute Capability | GPU Examples |
|--------------|-------------------|--------------|
| Volta | sm_70 | Tesla V100 |
| Turing | sm_75 | RTX 2080, GTX 1660 |
| Ampere | **sm_86** | RTX 3090, A6000 (default) |
| Ampere | sm_80 | A100 |
| Ada | sm_89 | RTX 4090 |

**Configuration:**
```bash
export CUDA_ARCH=86  # Set target architecture
```

### PTX Compilation Verification

**Script:** `scripts/check_ptx_compilation.sh`

```bash
#!/bin/bash
# Verify all PTX files were generated correctly
OUT_DIR=$(cargo metadata --format-version 1 | jq -r '.target_directory')
PTX_DIR="${OUT_DIR}/release/build/webxr-*/out"

for kernel in visionflow_unified gpu_clustering_kernels dynamic_grid \
              gpu_aabb_reduction gpu_landmark_apsp sssp_compact \
              visionflow_unified_stability ontology_constraints; do
    PTX_FILE=$(find ${PTX_DIR} -name "${kernel}.ptx" 2>/dev/null)
    if [ -f "$PTX_FILE" ]; then
        SIZE=$(stat -c%s "$PTX_FILE")
        echo "✓ ${kernel}.ptx: ${SIZE} bytes"
    else
        echo "✗ ${kernel}.ptx: MISSING"
    fi
done
```

### GPU Detection at Runtime

**Environment Variables:**
- `NVIDIA_VISIBLE_DEVICES=0` - GPU to use
- `CUDA_DEVICE=0` - CUDA device index
- `NO_GPU_COMPUTE=true` - Force CPU fallback

---

## 5. Docker Build Strategy

### Development Dockerfile

**File:** `Dockerfile.dev`

**Base Image:** `nvidia/cuda:12.4.1-devel-ubuntu22.04`

**Layer Optimization:**

```dockerfile
# Layer 1: System packages (cached)
RUN apt-get update && apt-get install -y \
    curl git gcc-11 g++-11 build-essential pkg-config \
    libssl-dev netcat-openbsd lsof gzip expect \
    docker.io supervisor nginx

# Layer 2: Rust toolchain (cached)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable

# Layer 3: Node.js (cached)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Layer 4: Source code (invalidated on changes)
COPY Cargo.toml build.rs ./
COPY src ./src
COPY whelk-rs ./whelk-rs
COPY client ./client

# Layer 5: Dependencies (cached if Cargo.toml unchanged)
RUN cargo fetch

# Layer 6: NPM dependencies (cached if package.json unchanged)
WORKDIR /app/client
RUN npm install
```

### Volume Mounts (Development)

```yaml
volumes:
  - ./client:/app/client                    # Live client code
  - ./src:/app/src                          # Live backend code
  - ./Cargo.toml:/app/Cargo.toml           # Cargo config
  - npm-cache:/root/.npm                   # NPM cache
  - cargo-cache:/root/.cargo/registry      # Cargo registry cache
  - cargo-git-cache:/root/.cargo/git       # Cargo git cache
  - cargo-target-cache:/app/target         # Build artifacts
```

**Benefits:**
- ✅ Live code reloading
- ✅ Persistent dependency caches
- ✅ Faster rebuild times

### Production Dockerfile

**File:** `Dockerfile.production`

**Differences from Dev:**
- Multi-stage build
- No dev dependencies
- Stripped binaries
- Optimized layers

---

## 6. Dependency Analysis

### Rust Dependencies (Cargo.toml)

#### Web Framework
```toml
actix-web = "4.11.0"
actix-cors = "0.7.1"
actix-files = "0.6"
actix-web-actors = "4.3"
tokio = { version = "1.47.1", features = ["full"] }
```

#### Database
```toml
rusqlite = { version = "0.37", features = ["bundled"] }
r2d2 = "0.8"
r2d2_sqlite = "0.31"
redis = { version = "0.27", optional = true }
```

#### GPU Computing
```toml
cudarc = { version = "0.12.1", optional = true }
cust = { version = "0.3.2", optional = true }
cust_core = { version = "0.1.1", optional = true }
bytemuck = { version = "1.21", features = ["derive"] }
```

#### Serialization
```toml
serde = { version = "1.0.219", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
specta = { version = "2.0.0-rc.22", features = ["derive", "export"] }
```

#### Ontology
```toml
horned-owl = { version = "1.2.0", features = ["remote"], optional = true }
whelk = { path = "./whelk-rs", optional = true }
```

#### Total Dependency Count
- **Direct dependencies:** ~80
- **Transitive dependencies:** ~200+

### Frontend Dependencies (package.json)

#### Core Framework
```json
{
  "react": "18.2.0",
  "react-dom": "18.2.0",
  "typescript": "5.8.3",
  "vite": "6.2.6"
}
```

#### 3D Graphics
```json
{
  "@babylonjs/core": "8.28.0",
  "three": "0.175.0",
  "@react-three/fiber": "8.15.0",
  "@react-three/drei": "9.80.0",
  "@mediapipe/tasks-vision": "0.10.21"
}
```

#### UI Components
```json
{
  "@radix-ui/react-dialog": "1.1.7",
  "@radix-ui/react-dropdown-menu": "2.1.7",
  "@radix-ui/themes": "3.2.1",
  "tailwindcss": "4.1.3",
  "framer-motion": "12.6.5"
}
```

#### Utilities
```json
{
  "lodash": "4.17.21",
  "uuid": "11.1.0",
  "immer": "10.1.1",
  "nostr-tools": "2.12.0"
}
```

#### Total Dependency Count
- **Direct dependencies:** ~60
- **Transitive dependencies:** ~500+

### Security Overrides

**Package.json overrides:**
```json
{
  "ansi-regex": "6.1.0",
  "esbuild": "0.25.9",
  "prismjs": "1.30.0"
}
```

**Reason:** Security vulnerability fixes

---

## 7. Build Scripts Reference

### Key Build Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `dev-entrypoint.sh` | `/scripts/` | Container startup orchestration |
| `build_ptx.sh` | `/scripts/` | Manual PTX compilation |
| `check_ptx_compilation.sh` | `/scripts/` | Verify PTX build |
| `verify_ptx.sh` | `/scripts/` | Validate PTX files |
| `launch.sh` | `/scripts/` | Production startup |
| `rust-backend-wrapper.sh` | `/scripts/` | Rust service wrapper |

### dev-entrypoint.sh Flow

```bash
#!/bin/bash
set -e

# 1. Setup Docker group
# 2. Create log directories
# 3. Set up cleanup trap
# 4. Free port 4000 if in use (lsof/fuser)
# 5. Build Rust backend (cargo build --release --features gpu)
# 6. Verify binary exists
# 7. Start Rust server (background)
# 8. Start Vite dev server (background)
# 9. Start Nginx (foreground)
# 10. Wait for Nginx or sleep infinity
```

**Environment Variables Used:**
- `SKIP_RUST_REBUILD` - Skip rebuild (default: false)
- `SYSTEM_NETWORK_PORT` - API port (default: 4000)
- `RUST_LOG` - Log level
- `DOCKER_ENV` - Flag for Docker build

---

## 8. Configuration Files

### Build Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `Cargo.toml` | Rust package manifest | TOML |
| `build.rs` | Rust build script | Rust |
| `package.json` | NPM package manifest | JSON |
| `vite.config.ts` | Vite bundler config | TypeScript |
| `tsconfig.json` | TypeScript compiler | JSON |
| `tailwind.config.js` | Tailwind CSS config | JavaScript |
| `.eslintrc.json` | ESLint config | JSON |
| `.prettierrc` | Prettier config | JSON |

### Runtime Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `.env` | Environment variables | ENV |
| `settings.yaml` | Application settings | YAML |
| `nginx.dev.conf` | Nginx reverse proxy | Nginx |
| `supervisord.dev.conf` | Process supervisor | INI |
| `docker-compose.yml` | Container orchestration | YAML |

### Configuration Hierarchy

```
Environment Variables (.env)
    ↓
Application Config (settings.yaml)
    ↓
Database State (settings.db)
    ↓
Runtime Overrides (API calls)
```

---

## 9. Known Build Issues

### Issue 1: PTX Compilation Failures

**Symptom:** Build fails with "CUDA PTX compilation failed"

**Causes:**
- CUDA toolkit not installed
- Wrong CUDA_ARCH for GPU
- nvcc not in PATH

**Solutions:**
```bash
# Check CUDA installation
which nvcc
nvcc --version

# Set correct architecture
export CUDA_ARCH=86  # For RTX 30-series

# Verify GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Issue 2: Rust Rebuild Every Time

**Symptom:** Cargo rebuilds entire project on every start

**Cause:** Intentional for development (ensures code changes applied)

**Workaround:**
```bash
export SKIP_RUST_REBUILD=true
```

**Note:** Only use for testing, not for development

### Issue 3: Port 4000 Already in Use

**Symptom:** "Address already in use" error

**Cause:** Previous Rust server not properly terminated

**Solution:**
```bash
# Kill process on port 4000
lsof -ti:4000 | xargs kill -9

# Or restart container
docker-compose restart
```

### Issue 4: Vite HMR Not Working

**Symptom:** Changes not reflected in browser

**Causes:**
- File watching disabled in Docker
- Nginx proxy misconfigured

**Solutions:**
```typescript
// vite.config.ts
server: {
  watch: {
    usePolling: true,  // Required for Docker
    interval: 1000
  }
}
```

### Issue 5: TypeScript Type Errors

**Symptom:** "Cannot find module" or type errors

**Cause:** Types not generated from Rust backend

**Solution:**
```bash
cd client
npm run types:generate
```

---

## 10. Build Performance Optimization

### Caching Strategy

#### Docker Layer Caching

**Optimal Layer Order:**
1. System packages (rarely changes)
2. Language runtimes (rarely changes)
3. Dependency manifests (Cargo.toml, package.json)
4. Dependencies (cargo fetch, npm install)
5. Source code (changes frequently)

#### Cargo Build Caching

**Persistent Volumes:**
```yaml
volumes:
  - cargo-cache:/root/.cargo/registry
  - cargo-git-cache:/root/.cargo/git
  - cargo-target-cache:/app/target
```

**Cache Locations:**
- `/root/.cargo/registry` - Crate registry
- `/root/.cargo/git` - Git dependencies
- `/app/target` - Build artifacts

**Cache Hit Rate:** ~90% for unchanged dependencies

#### NPM Build Caching

**Persistent Volumes:**
```yaml
volumes:
  - npm-cache:/root/.npm
```

**Cache Locations:**
- `/root/.npm` - Package cache
- `client/node_modules` - Installed packages

### Build Time Benchmarks

| Operation | Cold Build | Warm Build | Cache Hit |
|-----------|------------|------------|-----------|
| Docker build | 10-15 min | 2-3 min | 80% |
| Rust compilation | 5-7 min | 30-60 sec | 90% |
| CUDA PTX generation | 1-2 min | 10-20 sec | 95% |
| NPM install | 2-3 min | 10-15 sec | 95% |
| Vite build | 1-2 min | 10-20 sec | 85% |
| **Total (dev start)** | **15-20 min** | **3-5 min** | - |

### Optimization Tips

1. **Use BuildKit:**
   ```bash
   export DOCKER_BUILDKIT=1
   docker-compose build
   ```

2. **Parallel Cargo Builds:**
   ```bash
   cargo build -j$(nproc)
   ```

3. **Incremental Compilation:**
   ```toml
   [profile.dev]
   incremental = true
   ```

4. **Skip Unused Features:**
   ```bash
   cargo build --no-default-features --features ontology
   ```

5. **Use sccache (Rust):**
   ```bash
   cargo install sccache
   export RUSTC_WRAPPER=sccache
   ```

---

## Summary

### Build System Strengths

✅ **Multi-stage optimization** - Docker layer caching
✅ **Persistent caching** - Cargo, NPM, CUDA artifacts
✅ **Feature flags** - GPU, ontology, redis
✅ **Type safety** - Rust → TypeScript code generation
✅ **GPU acceleration** - 8 CUDA kernels, 40+ functions

### Build System Challenges

⚠️ **Rebuild on every start** - Intentional but slow
⚠️ **CUDA complexity** - Requires proper toolkit setup
⚠️ **Large dependency tree** - 200+ Rust deps, 500+ NPM deps
⚠️ **No CI/CD** - Manual build process only

### Recommended Improvements

1. Add GitHub Actions CI pipeline
2. Implement staged Dockerfile (separate build stage)
3. Add build caching to CI
4. Create prebuilt Docker images for common configs
5. Document GPU architecture detection
6. Add build time monitoring

---

**Analysis Complete**
**Document Version:** 1.0
**Last Updated:** 2025-10-23
