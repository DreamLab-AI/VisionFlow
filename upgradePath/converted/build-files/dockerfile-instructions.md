# Build Configuration Instructions - Dockerfile Updates

## **Task 3.1: Refactor Dockerfile.dev to Multi-Stage Build**
*   **Goal:** Implement multi-stage Docker build for better optimization and security
*   **Actions:**
    1. In `Dockerfile.dev`: Change base image and add multi-stage build:
       - Change from `FROM nvidia/cuda:12.8.1-devel-ubuntu22.04` to multi-stage approach
       - **Stage 1 Builder**: Use `FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder`
       - **Stage 2 Final**: Use `FROM nvidia/cuda:12.4.1-devel-ubuntu22.04`
    
    2. Configure Stage 1 (Builder):
       - Install Rust toolchain:
         ```bash
         RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update -o Acquire::Retries=3 && apt-get install -y curl && \
             curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
         ENV PATH="/root/.cargo/bin:${PATH}"
         ```
       - Install build dependencies:
         ```bash
         RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update -o Acquire::Retries=3 && apt-get install -y --no-install-recommends \
             build-essential \
             pkg-config \
             libssl-dev
         ```
       - Copy only necessary files for dependency caching:
         ```bash
         COPY Cargo.toml ./
         RUN cargo fetch
         ```
       - Copy source and build:
         ```bash
         COPY src ./src
         RUN cargo build --release
         ```
    
    3. Configure Stage 2 (Final Image):
       - Remove build-time dependencies:
         - Remove `build-essential` from runtime packages
         - Remove `pkg-config` from runtime packages  
         - Remove `libssl-dev` from runtime packages
       - Keep only runtime dependencies:
         - `curl`, `git`, `gcc-11`, `g++-11`
         - `netcat-openbsd`, `lsof`, `gzip`
         - Node.js and Nginx
    
    4. Remove inline Rust compilation:
       - Remove cargo build section from final stage
       - Remove `COPY Cargo.toml ./`, `COPY build.rs ./`, `COPY src ./src`
       - Replace with: `COPY --from=builder /app/target/release/webxr /app/webxr`
    
    5. Simplify build process:
       - Remove `cargo fetch` from final stage
       - Remove complex GPU build flags and environment variables
       - Remove PTX compilation comments and logic

## **Task 3.2: Remove build.rs Build Script**
*   **Goal:** Eliminate complex CUDA build script that was causing issues
*   **Actions:**
    1. Delete `build.rs` file entirely:
       - Remove 235 lines of CUDA compilation logic
       - Remove PTX file generation complexity
       - Remove environment variable dependencies
    
    2. Simplify build process:
       - Rely on cudarc crate for CUDA integration
       - Remove custom CUDA kernel compilation
       - Use standard Rust build process

## **Implementation Notes:**
- Multi-stage build reduces final image size by excluding build tools
- CUDA downgrade from 12.8.1 to 12.4.1 for better compatibility with cudarc 0.12.1
- Removes complex build script that was causing compilation issues
- Maintains GPU support through cudarc crate rather than custom kernels
- Improves build caching by separating dependency fetch and source compilation

## **Build Process Changes:**
1. **Before**: Single-stage build with inline compilation
2. **After**: Multi-stage build with separate builder and runtime stages
3. **Result**: Smaller images, better caching, more reliable builds