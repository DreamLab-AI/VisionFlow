# Configuration Guide

This guide covers all configuration options available in AutoSchema KG Rust, including environment variables, configuration files, and runtime settings.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Configuration Files](#configuration-files)
- [Environment Variables](#environment-variables)
- [Module-Specific Configuration](#module-specific-configuration)
- [Runtime Configuration](#runtime-configuration)
- [Production Configuration](#production-configuration)
- [Configuration Validation](#configuration-validation)

## Configuration Overview

AutoSchema KG Rust uses a hierarchical configuration system that supports:

- **TOML configuration files** for structured settings
- **Environment variables** for deployment flexibility
- **Command-line arguments** for runtime overrides
- **Default values** for quick setup

Configuration is loaded in this priority order:
1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration files
4. Default values (lowest priority)

## Configuration Files

### Main Configuration File

The primary configuration file is `config.toml` in the project root:

```toml
# config.toml - Main configuration file

[general]
# Application settings
app_name = "autoschema_kg_rust"
version = "0.1.0"
log_level = "info"
max_concurrent_tasks = 100

[knowledge_graph]
# Neo4j database configuration
neo4j_uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"
database = "neo4j"

# Connection pool settings
max_connections = 10
min_connections = 2
connection_timeout = 30
acquire_timeout = 10
idle_timeout = 600

# Transaction settings
transaction_timeout = 60
retry_attempts = 3
retry_delay = 1000

[retriever]
# Multi-hop retrieval settings
max_hops = 3
top_k = 10
similarity_threshold = 0.7
enable_reranking = true

# Cache configuration
cache_size = 1000
cache_ttl = 3600
enable_cache = true

# Query processing
enable_query_expansion = true
expansion_count = 3
enable_query_rewriting = true

[llm]
# Default LLM provider
provider = "openai"
model = "gpt-4"
api_key = "${OPENAI_API_KEY}"

# Generation settings
temperature = 0.1
max_tokens = 2048
top_p = 0.9
frequency_penalty = 0.0
presence_penalty = 0.0

# Rate limiting
requests_per_minute = 60
tokens_per_minute = 50000
concurrent_requests = 5

# Retry configuration
max_retries = 3
retry_delay = 1000
backoff_multiplier = 2.0

[vector_store]
# Embedding model configuration
model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
device = "cpu"
batch_size = 32

# Index configuration
index_type = "hnsw"
m = 16
ef_construction = 200
ef_search = 100

# Storage settings
persist_index = true
index_path = "data/vector_index"
compression = true

[processing]
# Batch processing settings
batch_size = 1000
parallel_workers = 4
chunk_size = 10000

# Validation settings
enable_validation = true
strict_validation = false
skip_invalid_records = true

# Memory management
max_memory_usage = "2GB"
enable_streaming = true
stream_chunk_size = 1000

[monitoring]
# Metrics collection
enable_metrics = true
metrics_port = 9090
metrics_path = "/metrics"

# Health checks
health_check_port = 8080
health_check_path = "/health"

# Logging
log_format = "json"
log_file = "logs/autoschema.log"
log_rotation = "daily"
log_retention = 30

[security]
# API security
enable_auth = false
api_key_header = "X-API-Key"
rate_limit_per_ip = 1000

# TLS configuration
enable_tls = false
cert_file = "certs/server.crt"
key_file = "certs/server.key"
```

### Environment-Specific Configuration

Create separate configuration files for different environments:

#### Development Configuration (`config/development.toml`)
```toml
[general]
log_level = "debug"
max_concurrent_tasks = 10

[knowledge_graph]
neo4j_uri = "bolt://localhost:7687"
max_connections = 5

[llm]
provider = "local"
model = "llama2"
api_endpoint = "http://localhost:11434"

[monitoring]
enable_metrics = false
```

#### Testing Configuration (`config/testing.toml`)
```toml
[general]
log_level = "warn"

[knowledge_graph]
neo4j_uri = "bolt://localhost:7688"  # Different port for tests
database = "test"

[vector_store]
index_path = "test_data/vector_index"
persist_index = false

[processing]
batch_size = 100
parallel_workers = 1
```

#### Production Configuration (`config/production.toml`)
```toml
[general]
log_level = "info"
max_concurrent_tasks = 500

[knowledge_graph]
neo4j_uri = "${NEO4J_URI}"
username = "${NEO4J_USERNAME}"
password = "${NEO4J_PASSWORD}"
max_connections = 50

[llm]
api_key = "${OPENAI_API_KEY}"
requests_per_minute = 300
concurrent_requests = 20

[monitoring]
enable_metrics = true
log_format = "json"

[security]
enable_auth = true
enable_tls = true
```

## Environment Variables

### Core Environment Variables

```bash
# Application Configuration
export APP_ENV="production"                    # Environment: development, testing, production
export APP_LOG_LEVEL="info"                   # Log level: trace, debug, info, warn, error
export APP_CONFIG_FILE="config/production.toml"  # Configuration file path

# Database Configuration
export NEO4J_URI="bolt://neo4j-cluster:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-secure-password"
export NEO4J_DATABASE="knowledge_graph"
export NEO4J_MAX_CONNECTIONS="20"

# LLM API Keys
export OPENAI_API_KEY="sk-your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export HUGGINGFACE_API_KEY="your-huggingface-token"

# Vector Store Configuration
export VECTOR_STORE_PATH="/data/vectors"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export VECTOR_DEVICE="cuda"                   # cpu, cuda, mps

# Processing Configuration
export PROCESSING_BATCH_SIZE="5000"
export PROCESSING_WORKERS="8"
export MAX_MEMORY_USAGE="8GB"

# Monitoring and Logging
export METRICS_ENABLED="true"
export METRICS_PORT="9090"
export HEALTH_CHECK_PORT="8080"
export LOG_FORMAT="json"
export LOG_FILE="/var/log/autoschema/app.log"

# Security
export API_KEY="your-api-key"
export TLS_CERT_FILE="/etc/ssl/certs/server.crt"
export TLS_KEY_FILE="/etc/ssl/private/server.key"

# Performance Tuning
export RUST_LOG="autoschema_kg_rust=info,neo4rs=warn"
export RUST_BACKTRACE="1"
export MALLOC_CONF="background_thread:true,metadata_thp:auto"
```

### Docker Environment Variables

For Docker deployment, use an environment file:

```bash
# .env.production
APP_ENV=production
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=production-password
OPENAI_API_KEY=sk-production-key
VECTOR_DEVICE=cpu
PROCESSING_WORKERS=4
METRICS_ENABLED=true
LOG_LEVEL=info
```

## Module-Specific Configuration

### Knowledge Graph Construction Configuration

```rust
use kg_construction::{GraphConfig, ProcessingConfig, ValidationConfig};

// Graph connection configuration
let graph_config = GraphConfig::builder()
    .neo4j_uri("bolt://localhost:7687")
    .username("neo4j")
    .password("password")
    .database("knowledge_graph")
    .max_connections(10)
    .connection_timeout(Duration::from_secs(30))
    .transaction_timeout(Duration::from_secs(60))
    .retry_attempts(3)
    .build();

// Processing configuration
let processing_config = ProcessingConfig::builder()
    .batch_size(1000)
    .parallel_workers(4)
    .chunk_size(10000)
    .enable_validation(true)
    .skip_invalid_records(true)
    .max_memory_usage(2_000_000_000) // 2GB
    .build();

// Validation configuration
let validation_config = ValidationConfig::builder()
    .strict_mode(false)
    .allow_unknown_types(true)
    .max_property_length(1000)
    .required_fields(vec!["id".to_string(), "type".to_string()])
    .build();
```

### LLM Generator Configuration

```rust
use llm_generator::{GenerationConfig, RateConfig, RetryConfig, OpenAIProvider};

// Generation configuration
let generation_config = GenerationConfig::builder()
    .model("gpt-4")
    .temperature(0.1)
    .max_tokens(2048)
    .top_p(0.9)
    .frequency_penalty(0.0)
    .presence_penalty(0.0)
    .stop_sequences(vec!["\\n\\n".to_string()])
    .build();

// Rate limiting configuration
let rate_config = RateConfig::builder()
    .requests_per_minute(60)
    .tokens_per_minute(50000)
    .concurrent_requests(5)
    .burst_allowance(10)
    .build();

// Retry configuration
let retry_config = RetryConfig::builder()
    .max_retries(3)
    .base_delay(Duration::from_millis(1000))
    .max_delay(Duration::from_secs(60))
    .backoff_multiplier(2.0)
    .jitter(true)
    .build();

// Provider configuration
let provider = OpenAIProvider::builder()
    .api_key(&std::env::var("OPENAI_API_KEY")?)
    .base_url("https://api.openai.com/v1")
    .organization("your-org-id")
    .timeout(Duration::from_secs(60))
    .max_retries(3)
    .build();
```

### Retriever Configuration

```rust
use retriever::{RetrieverConfig, CacheConfig, QueryConfig, RankingConfig};

// Main retriever configuration
let retriever_config = RetrieverConfig::builder()
    .max_hops(3)
    .top_k(10)
    .similarity_threshold(0.7)
    .enable_reranking(true)
    .traversal_strategy(TraversalStrategy::Adaptive)
    .build();

// Cache configuration
let cache_config = CacheConfig::builder()
    .size(1000)
    .ttl(Duration::from_secs(3600))
    .enable_cache(true)
    .cache_type(CacheType::Memory)
    .eviction_policy(EvictionPolicy::LRU)
    .build();

// Query processing configuration
let query_config = QueryConfig::builder()
    .enable_expansion(true)
    .expansion_count(3)
    .enable_rewriting(true)
    .min_query_length(3)
    .max_query_length(500)
    .build();

// Ranking configuration
let ranking_config = RankingConfig::builder()
    .strategy(RankingStrategy::Fusion)
    .weights(vec![0.4, 0.3, 0.3]) // Semantic, lexical, graph
    .normalize_scores(true)
    .min_score(0.1)
    .build();
```

### Vector Store Configuration

```rust
use vectorstore::{EmbeddingConfig, IndexConfig, StorageConfig};

// Embedding model configuration
let embedding_config = EmbeddingConfig::builder()
    .model("sentence-transformers/all-MiniLM-L6-v2")
    .dimension(384)
    .device("cuda")
    .batch_size(32)
    .max_length(512)
    .normalize_embeddings(true)
    .build();

// Vector index configuration
let index_config = IndexConfig::builder()
    .index_type(IndexType::HNSW)
    .m(16)
    .ef_construction(200)
    .ef_search(100)
    .max_m(16)
    .ml(1.0 / (2.0_f32).ln())
    .build();

// Storage configuration
let storage_config = StorageConfig::builder()
    .persist_index(true)
    .index_path("data/vector_index")
    .compression(true)
    .backup_enabled(true)
    .backup_interval(Duration::from_secs(3600))
    .build();
```

## Runtime Configuration

### Tokio Runtime Configuration

```rust
use tokio::runtime::{Builder, Runtime};

// Custom runtime configuration
fn create_optimized_runtime() -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .max_blocking_threads(512)
        .thread_stack_size(2 * 1024 * 1024) // 2MB stack
        .thread_name("autoschema-worker")
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime")
}

// Usage
#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Application code
    Ok(())
}
```

### Memory Allocator Configuration

```rust
// Use jemalloc for better memory management
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// Configure jemalloc at startup
fn configure_memory() {
    use std::env;

    // Enable background threads for better performance
    env::set_var("MALLOC_CONF",
        "background_thread:true,metadata_thp:auto,dirty_decay_ms:1000,muzzy_decay_ms:1000");
}
```

### Logging Configuration

```rust
use tracing::{Level, subscriber::set_global_default};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn init_logging() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("autoschema_kg_rust=info"));

    let file_appender = tracing_appender::rolling::daily("logs", "autoschema.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_thread_names(true)
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .json()
        )
        .init();

    Ok(())
}
```

## Production Configuration

### Load Balancer Configuration

#### Nginx Configuration (`nginx.conf`)
```nginx
upstream autoschema_backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8082 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.autoschema.com;

    # Request size limits
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types application/json text/plain application/xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://autoschema_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Keep alive
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    location /metrics {
        proxy_pass http://autoschema_backend;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }
}
```

### Systemd Service Configuration

#### Service file (`/etc/systemd/system/autoschema-kg-rust.service`)
```ini
[Unit]
Description=AutoSchema KG Rust Service
After=network.target neo4j.service
Wants=neo4j.service

[Service]
Type=simple
User=autoschema
Group=autoschema
WorkingDirectory=/opt/autoschema-kg-rust
ExecStart=/opt/autoschema-kg-rust/bin/autoschema_kg_rust
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
KillSignal=SIGTERM
Restart=always
RestartSec=5

# Environment
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1
EnvironmentFile=/etc/autoschema/environment

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/autoschema-kg-rust/data /var/log/autoschema

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

### Docker Compose Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  autoschema-kg-rust:
    image: autoschema-kg-rust:latest
    restart: unless-stopped

    environment:
      - APP_ENV=production
      - RUST_LOG=info
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}

    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config:ro

    ports:
      - "8080:8080"
      - "9090:9090"

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

    depends_on:
      neo4j:
        condition: service_healthy

    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 1G
          cpus: '0.5'

  neo4j:
    image: neo4j:5.11
    restart: unless-stopped

    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_heap_initial__size: 1G
      NEO4J_dbms_memory_heap_max__size: 2G
      NEO4J_dbms_memory_pagecache_size: 1G

    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

    ports:
      - "7474:7474"
      - "7687:7687"

    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped

    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus

    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

    ports:
      - "9000:9090"

volumes:
  neo4j_data:
  neo4j_logs:
  prometheus_data:
```

## Configuration Validation

### Configuration Schema Validation

```rust
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

#[derive(Debug, Deserialize, Serialize, Validate)]
pub struct AppConfig {
    #[validate(nested)]
    pub general: GeneralConfig,

    #[validate(nested)]
    pub knowledge_graph: KnowledgeGraphConfig,

    #[validate(nested)]
    pub llm: LLMConfig,

    #[validate(nested)]
    pub vector_store: VectorStoreConfig,
}

#[derive(Debug, Deserialize, Serialize, Validate)]
pub struct GeneralConfig {
    #[validate(length(min = 1, max = 100))]
    pub app_name: String,

    #[validate(range(min = 1, max = 10000))]
    pub max_concurrent_tasks: usize,

    #[validate(custom = "validate_log_level")]
    pub log_level: String,
}

#[derive(Debug, Deserialize, Serialize, Validate)]
pub struct KnowledgeGraphConfig {
    #[validate(url)]
    pub neo4j_uri: String,

    #[validate(length(min = 1))]
    pub username: String,

    #[validate(length(min = 1))]
    pub password: String,

    #[validate(range(min = 1, max = 100))]
    pub max_connections: usize,

    #[validate(range(min = 1, max = 300))]
    pub connection_timeout: u64,
}

fn validate_log_level(level: &str) -> Result<(), ValidationError> {
    match level {
        "trace" | "debug" | "info" | "warn" | "error" => Ok(()),
        _ => Err(ValidationError::new("invalid_log_level")),
    }
}

impl AppConfig {
    pub fn load_and_validate() -> Result<Self, ConfigError> {
        let config = Self::load()?;
        config.validate()
            .map_err(|e| ConfigError::ValidationFailed(format!("{:?}", e)))?;
        Ok(config)
    }

    pub fn validate_runtime(&self) -> Result<(), ConfigError> {
        // Additional runtime validations
        if self.knowledge_graph.max_connections > 50 {
            log::warn!("High connection count may impact performance");
        }

        if self.vector_store.dimension % 8 != 0 {
            log::warn!("Vector dimension not aligned for SIMD optimization");
        }

        Ok(())
    }
}
```

### Configuration Testing

```rust
#[cfg(test)]
mod config_tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_valid_config() {
        let config_content = r#"
            [general]
            app_name = "test_app"
            max_concurrent_tasks = 100
            log_level = "info"

            [knowledge_graph]
            neo4j_uri = "bolt://localhost:7687"
            username = "neo4j"
            password = "password"
            max_connections = 10
            connection_timeout = 30
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let config = AppConfig::from_file(temp_file.path()).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config_content = r#"
            [general]
            app_name = ""  # Invalid: empty string
            max_concurrent_tasks = 0  # Invalid: below minimum
            log_level = "invalid"  # Invalid: not a valid log level
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let result = AppConfig::from_file(temp_file.path())
            .and_then(|config| config.validate().map_err(|e| e.into()));

        assert!(result.is_err());
    }
}
```

### Configuration Hot Reloading

```rust
use notify::{Watcher, RecursiveMode, watcher};
use std::sync::{Arc, RwLock};
use std::sync::mpsc::channel;
use std::time::Duration;

pub struct ConfigManager {
    config: Arc<RwLock<AppConfig>>,
    config_path: PathBuf,
}

impl ConfigManager {
    pub fn new(config_path: PathBuf) -> Result<Self, ConfigError> {
        let config = AppConfig::load_from_file(&config_path)?;

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            config_path,
        })
    }

    pub fn start_watching(&self) -> Result<(), ConfigError> {
        let (tx, rx) = channel();
        let mut watcher = watcher(tx, Duration::from_secs(1))?;

        watcher.watch(&self.config_path, RecursiveMode::NonRecursive)?;

        let config = Arc::clone(&self.config);
        let config_path = self.config_path.clone();

        tokio::spawn(async move {
            while let Ok(event) = rx.recv() {
                match event {
                    notify::DebouncedEvent::Write(_) => {
                        match AppConfig::load_from_file(&config_path) {
                            Ok(new_config) => {
                                if new_config.validate().is_ok() {
                                    *config.write().unwrap() = new_config;
                                    log::info!("Configuration reloaded successfully");
                                } else {
                                    log::error!("Invalid configuration, keeping current config");
                                }
                            },
                            Err(e) => {
                                log::error!("Failed to reload configuration: {}", e);
                            }
                        }
                    },
                    _ => {}
                }
            }
        });

        Ok(())
    }

    pub fn get_config(&self) -> Arc<RwLock<AppConfig>> {
        Arc::clone(&self.config)
    }
}
```

This comprehensive configuration guide covers all aspects of configuring AutoSchema KG Rust for different environments and use cases. The hierarchical configuration system provides flexibility while maintaining type safety and validation.