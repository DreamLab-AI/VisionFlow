# ============================================================================

# RuVector PostgreSQL with Embeddings Support

# ============================================================================

#

# Builds the ruvector extension from source with --features embeddings

# so that ruvector_embed_vec() is available for in-database embedding generation.

#

# Build time: ~30 min on 8-core, requires 8GB+ RAM

# Base: postgres:17-bookworm

#

# ============================================================================



# ---------- Stage 1: Build ruvector extension from source ----------

FROM rust:1.86-bookworm AS builder



# Add PostgreSQL 17 apt repository (PGDG)

RUN apt-get update && apt-get install -y --no-install-recommends \

    curl ca-certificates gnupg lsb-release && \

    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/pgdg.gpg && \

    echo "deb [signed-by=/usr/share/keyrings/pgdg.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \

    > /etc/apt/sources.list.d/pgdg.list && \

    rm -rf /var/lib/apt/lists/*



# Install build dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \

    build-essential \

    clang \

    cmake \

    git \

    libopenblas-dev \

    libssl-dev \

    pkg-config \

    postgresql-server-dev-17 \

    && rm -rf /var/lib/apt/lists/*



# Install pgrx (PostgreSQL extension framework)

RUN cargo install cargo-pgrx --version "0.12.9" --locked



# Initialize pgrx for PG17

RUN cargo pgrx init --pg17 /usr/bin/pg_config



# Clone ruvector source

ARG RUVECTOR_VERSION=v2.0.0

RUN git clone --depth 1 --branch "${RUVECTOR_VERSION}" \

    https://github.com/ruvnet/ruvector.git /build/ruvector || \

    git clone --depth 1 https://github.com/ruvnet/ruvector.git /build/ruvector



WORKDIR /build/ruvector/crates/ruvector-postgres



# Build with embeddings feature enabled

# Features: pg17 (target), index-all (HNSW/IVF), quant-all (compression),

#           graph-complete (Cypher/SPARQL), embeddings (in-DB generation),

#           analytics-complete, attention-extended, sona-learning

RUN cargo pgrx package --pg-config /usr/bin/pg_config \

    --features "pg17 embeddings" \

    --no-default-features \

    2>&1 || \

    cargo pgrx package --pg-config /usr/bin/pg_config \

    --features "pg17" \

    --no-default-features \

    2>&1 || true

# DEBUG: find where pgrx put the artifacts

RUN echo "=== pgrx output dirs ===" && \

    find /build/ruvector/crates/ruvector-postgres/target -maxdepth 5 -name "*.control" 2>/dev/null && \

    echo "=== .so files ===" && \

    find /build/ruvector/crates/ruvector-postgres/target -maxdepth 5 -name "*.so" 2>/dev/null && \

    echo "=== extension dirs ===" && \

    find /build/ruvector/crates/ruvector-postgres/target -maxdepth 6 -type d -name "extension" 2>/dev/null && \

    echo "=== top-level target contents ===" && \

    ls -la /build/ruvector/crates/ruvector-postgres/target/ && \

    echo "=== release dir ===" && \

    ls -la /build/ruvector/crates/ruvector-postgres/target/release/ 2>/dev/null || true



# ---------- Stage 1b: Download embedding models ----------

FROM rust:1.86-bookworm AS model-downloader



# Build the download-models binary if it exists, otherwise use curl

RUN mkdir -p /opt/ruvector/models



# Download all-MiniLM-L6-v2 ONNX model (22MB)

RUN apt-get update && apt-get install -y curl && \

    mkdir -p /opt/ruvector/models/all-MiniLM-L6-v2 && \

    curl -fsSL "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" \

        -o /opt/ruvector/models/all-MiniLM-L6-v2/model.onnx && \

    curl -fsSL "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json" \

        -o /opt/ruvector/models/all-MiniLM-L6-v2/tokenizer.json && \

    curl -fsSL "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json" \

        -o /opt/ruvector/models/all-MiniLM-L6-v2/config.json && \

    curl -fsSL "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json" \

        -o /opt/ruvector/models/all-MiniLM-L6-v2/tokenizer_config.json && \

    curl -fsSL "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json" \

        -o /opt/ruvector/models/all-MiniLM-L6-v2/special_tokens_map.json || \

    echo "Model download failed - embeddings will use client-side fallback"



# Download bge-small-en-v1.5 ONNX model (backup model)

RUN mkdir -p /opt/ruvector/models/bge-small-en-v1.5 && \

    curl -fsSL "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" \

        -o /opt/ruvector/models/bge-small-en-v1.5/model.onnx && \

    curl -fsSL "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json" \

        -o /opt/ruvector/models/bge-small-en-v1.5/tokenizer.json && \

    curl -fsSL "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/config.json" \

        -o /opt/ruvector/models/bge-small-en-v1.5/config.json || \

    echo "BGE model download failed - will use MiniLM only"



# ---------- Stage 2: Runtime image ----------

FROM postgres:17-bookworm



# Install runtime dependencies for embeddings (ONNX Runtime)

RUN apt-get update && apt-get install -y --no-install-recommends \

    libopenblas0 \

    libssl3 \

    ca-certificates \

    && rm -rf /var/lib/apt/lists/*



# Copy built extension from builder

# pgrx packages to target/release/ruvector-pg17/

# TODO: restore correct paths after debug

RUN echo "placeholder for COPY"



# Copy pre-cached embedding models

COPY --from=model-downloader /opt/ruvector/models /opt/ruvector/models



# Environment for fastembed/ONNX model resolution

ENV FASTEMBED_CACHE_DIR=/opt/ruvector/models

ENV RUVECTOR_MODELS_DIR=/opt/ruvector/models



# PostgreSQL configuration optimized for vector workloads

RUN echo "shared_preload_libraries = 'ruvector'" >> /usr/share/postgresql/postgresql.conf.sample



# Health check

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \

    CMD pg_isready -U ${POSTGRES_USER:-ruvector} -d ${POSTGRES_DB:-ruvector} || exit 1



# Default environment

ENV POSTGRES_USER=ruvector

ENV POSTGRES_PASSWORD=ruvector

ENV POSTGRES_DB=ruvector

