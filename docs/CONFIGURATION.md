# Configuration Guide

## Overview

VisionFlow's MCP TCP client is configured through environment variables and builder patterns. This guide covers all configuration options and best practices.

## Environment Variables

### Required Variables

```bash
# MCP Server Location
CLAUDE_FLOW_HOST=multi-agent-container
MCP_TCP_PORT=9500

# Transport Selection
MCP_TRANSPORT=tcp
```

### Optional Performance Tuning

```bash
# Connection Retry Configuration
MCP_RECONNECT_ATTEMPTS=3        # Number of retry attempts (default: 3)
MCP_RECONNECT_DELAY=1000        # Delay between retries in ms (default: 1000)
MCP_CONNECTION_TIMEOUT=30000    # Connection timeout in ms (default: 30000)

# Logging
MCP_LOG_LEVEL=info              # Log level: trace, debug, info, warn, error
RUST_LOG=info,claude_flow=debug # Rust logging configuration
```

### Other Services

```bash
# RAGFlow Integration
RAGFLOW_API_BASE_URL=http://ragflow-server:9380
RAGFLOW_AGENT_ID=aa2e328812ef11f083dc0a0d6226f61b

# External APIs
PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}
GITHUB_TOKEN=${GITHUB_TOKEN}
OPENAI_API_KEY=${OPENAI_API_KEY}
```

## Docker Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  visionflow:
    container_name: logseq_spring_thing_webxr
    build:
      context: .
      dockerfile: Dockerfile.production
    
    networks:
      - docker_ragflow  # Must be same network as multi-agent-container
    
    depends_on:
      multi-agent:
        condition: service_healthy
    
    environment:
      # MCP TCP Configuration
      - CLAUDE_FLOW_HOST=multi-agent-container
      - MCP_TCP_PORT=9500
      - MCP_TRANSPORT=tcp
      - MCP_RECONNECT_ATTEMPTS=3
      - MCP_RECONNECT_DELAY=1000
      - MCP_CONNECTION_TIMEOUT=30000
      
      # Logging
      - RUST_LOG=info,claude_flow=debug,tcp=debug
      
      # Other services
      - RAGFLOW_API_BASE_URL=http://ragflow-server:9380
      - RAGFLOW_AGENT_ID=${RAGFLOW_AGENT_ID}
    
    ports:
      - "3001:3001"  # VisionFlow UI
      - "4000:4000"  # Internal API
    
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

networks:
  docker_ragflow:
    external: true
```

### Dockerfile.production

```dockerfile
FROM rust:1.75 as builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/visionflow /usr/local/bin/

# Create app user
RUN useradd -m -u 1000 app
USER app

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 3001 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4000/api/health || exit 1

# Run the application
CMD ["visionflow"]
```

## Code Configuration

### Client Builder Configuration

```rust
use std::time::Duration;
use crate::services::claude_flow::client_builder::ClaudeFlowClientBuilder;

// Default configuration (reads from environment)
let client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .build()
    .await?;

// Custom configuration
let client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .host("custom-mcp-host")
    .port(9600)
    .with_retry(5, Duration::from_secs(2))
    .with_timeout(Duration::from_secs(60))
    .build()
    .await?;

// Lazy initialisation (connect manually)
let mut client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .build_lazy()
    .await;

client.connect().await?;
client.initialise().await?;
```

### Actor Configuration

```rust
use crate::actors::claude_flow_actor_tcp::ClaudeFlowActorTcp;
use actix::prelude::*;

// Basic actor
let actor = ClaudeFlowActorTcp::new().start();

// With custom configuration
std::env::set_var("MCP_TCP_PORT", "9600");
std::env::set_var("MCP_RECONNECT_ATTEMPTS", "5");
let actor = ClaudeFlowActorTcp::new().start();
```

### Transport Configuration

```rust
use crate::services::claude_flow::transport::tcp::TcpTransport;

// Environment-based configuration
let transport = TcpTransport::new_with_defaults();

// Explicit configuration
let transport = TcpTransport::new("multi-agent-container", 9500);
```

## Network Configuration

### Docker Network Setup

```bash
# Create shared network
docker network create docker_ragflow

# Verify network
docker network inspect docker_ragflow

# List connected containers
docker network inspect docker_ragflow | jq '.Containers'
```

### Port Forwarding

For development access from host:

```bash
# Forward TCP port
ssh -L 9500:localhost:9500 user@docker-host

# Or use Docker port mapping
docker run -p 9500:9500 ...
```

### Firewall Rules

If using firewall, allow TCP port 9500:

```bash
# UFW
sudo ufw allow 9500/tcp

# iptables
sudo iptables -A INPUT -p tcp --dport 9500 -j ACCEPT
```

## Logging Configuration

### Log Levels

Configure via `RUST_LOG` environment variable:

```bash
# Global level
export RUST_LOG=info

# Module-specific levels
export RUST_LOG=warn,visionflow=info,claude_flow=debug,tcp=trace

# Filter by module
export RUST_LOG=visionflow::services::claude_flow=debug
```

### Log Output Format

```rust
// In main.rs
use env_logger;

fn init_logging() {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("info")
    )
    .format_timestamp_millis()
    .format_module_path(true)
    .init();
}
```

### Structured Logging

```rust
use tracing;
use tracing_subscriber;

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter("visionflow=debug,claude_flow=debug")
        .with_target(true)
        .with_thread_ids(true)
        .json()
        .init();
}
```

## Performance Configuration

### Connection Pooling

```rust
// config.rs
pub struct MCPConfig {
    pub pool_size: usize,
    pub pool_timeout: Duration,
    pub idle_timeout: Duration,
}

impl Default for MCPConfig {
    fn default() -> Self {
        Self {
            pool_size: std::env::var("MCP_POOL_SIZE")
                .unwrap_or("4".to_string())
                .parse()
                .unwrap_or(4),
            pool_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
        }
    }
}
```

### Buffer Sizes

```rust
// Optimize TCP buffer sizes
use tokio::net::TcpStream;

async fn configure_socket(stream: &TcpStream) -> Result<()> {
    use socket2::{Socket, Domain, Type};
    
    let socket = Socket::from(stream.as_raw_fd());
    socket.set_send_buffer_size(65536)?;  // 64KB
    socket.set_recv_buffer_size(65536)?;  // 64KB
    socket.set_nodelay(true)?;            // Disable Nagle's algorithm
    
    Ok(())
}
```

### Timeout Configuration

```rust
pub struct TimeoutConfig {
    pub connect: Duration,
    pub request: Duration,
    pub idle: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(30),
            request: Duration::from_secs(60),
            idle: Duration::from_secs(300),
        }
    }
}
```

## Security Configuration

### TLS Support (Future)

```bash
# TLS configuration (when implemented)
MCP_USE_TLS=true
MCP_TLS_PORT=9543
MCP_TLS_CERT=/path/to/cert.pem
MCP_TLS_KEY=/path/to/key.pem
MCP_TLS_CA=/path/to/ca.pem
```

### Authentication (Future)

```bash
# Token-based authentication (when implemented)
MCP_AUTH_ENABLED=true
MCP_AUTH_TOKEN=your-secret-token
MCP_AUTH_METHOD=bearer
```

### Rate Limiting

```rust
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;

fn create_rate_limiter() -> RateLimiter {
    let quota = Quota::per_second(
        NonZeroU32::new(
            std::env::var("MCP_RATE_LIMIT")
                .unwrap_or("100".to_string())
                .parse()
                .unwrap_or(100)
        ).unwrap()
    );
    
    RateLimiter::direct(quota)
}
```

## Development Configuration

### Local Development

`.env.development`:
```bash
# Local development
CLAUDE_FLOW_HOST=localhost
MCP_TCP_PORT=9500
MCP_TRANSPORT=tcp
MCP_LOG_LEVEL=debug
RUST_LOG=debug
```

### Testing

`.env.test`:
```bash
# Test configuration
CLAUDE_FLOW_HOST=test-mcp-server
MCP_TCP_PORT=9501
MCP_TRANSPORT=tcp
MCP_CONNECTION_TIMEOUT=5000
MCP_RECONNECT_ATTEMPTS=1
```

### CI/CD

`.env.ci`:
```bash
# CI configuration
CLAUDE_FLOW_HOST=ci-mcp-server
MCP_TCP_PORT=9500
MCP_TRANSPORT=tcp
MCP_LOG_LEVEL=warn
```

## Configuration Validation

```rust
use serde::Deserialize;
use config::{Config, ConfigError, Environment};

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub claude_flow_host: String,
    pub mcp_tcp_port: u16,
    pub mcp_transport: String,
    pub mcp_reconnect_attempts: u32,
    pub mcp_reconnect_delay: u64,
    pub mcp_connection_timeout: u64,
}

impl AppConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        Config::builder()
            .add_source(Environment::with_prefix(""))
            .build()?
            .try_deserialize()
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.mcp_tcp_port == 0 {
            return Err("Invalid port number".to_string());
        }
        
        if self.mcp_transport != "tcp" {
            return Err("Only TCP transport is supported".to_string());
        }
        
        Ok(())
    }
}
```

## Monitoring Configuration

### Metrics Collection

```bash
# Prometheus metrics
METRICS_ENABLED=true
METRICS_PORT=9090
METRICS_PATH=/metrics
```

### Health Check

```rust
// Health check endpoint configuration
pub struct HealthConfig {
    pub enabled: bool,
    pub port: u16,
    pub path: String,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 4000,
            path: "/api/health".to_string(),
        }
    }
}
```

---

*Configuration Guide Version: 1.0*
*Last Updated: 2025-08-12*