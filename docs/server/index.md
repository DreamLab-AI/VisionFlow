# Server Documentation

The VisionFlow backend is built with Rust using the Actix web framework and an actor-based architecture for high-performance, concurrent operations.

## Architecture Overview

### Core Architecture
- **[System Architecture](architecture.md)** - Actix-based web server design
- **[Actor System](actors.md)** - Actor model for concurrent state management
- **[GPU Compute](gpu-compute.md)** - GPU-accelerated physics simulation

### Key Features
- **Binary Protocol**: Efficient 28-byte position streaming format
- **Actor-Based Concurrency**: Safe state management with message passing
- **GPU Physics**: Optional GPU acceleration for large-scale simulations
- **MCP Integration**: Full Claude Flow agent control via TCP bridge

## Components

### Request Handling
- **[HTTP Handlers](handlers.md)** - REST API request handlers
- **[WebSocket Handlers](handlers.md#websocket-handlers)** - Real-time communication
- **[Binary Protocol Handler](handlers.md#binary-protocol)** - High-performance position updates

### Services & Integration
- **[Core Services](services.md)** - Business logic and external integrations
- **[AI Services](ai-services.md)** - RAGFlow, Perplexity, and Speech services
- **[Claude Flow Integration](features/claude-flow-mcp-integration.md)** - MCP agent orchestration

### Data & Configuration
- **[Data Models](models.md)** - Rust structs and database schemas
- **[Configuration System](config.md)** - Settings and environment management
- **[Feature Access Control](feature-access.md)** - User permissions and features

### Utilities
- **[Helper Functions](utils.md)** - Common utilities and helpers
- **[Type Definitions](types.md)** - Shared types and enums

## Quick Start

### Running the Server
```bash
# Development mode
cargo run --features gpu

# Production build
cargo build --release --features gpu
./target/release/agent-server

# With Docker
docker-compose up webxr
```

### Environment Variables
```bash
# Core settings
RUST_LOG=info
BIND_ADDRESS=0.0.0.0:3001

# Agent control
AGENT_CONTROL_URL=multi-agent-container:9500

# GPU physics
ENABLE_GPU_PHYSICS=true
PHYSICS_UPDATE_RATE=60

# WebSocket
WEBSOCKET_PORT=8080
WEBSOCKET_COMPRESSION=true
```

## API Endpoints

### Core APIs
- `GET /api/health` - Health check
- `GET /api/graph/data` - Graph data
- `POST /api/graph/update` - Update graph

### Agent APIs
- `GET /api/bots/status` - Agent status
- `GET /api/bots/data` - Full agent graph
- `POST /api/bots/initialize-multi-agent` - Spawn Multi Agent

### WebSocket
- `/ws` - Main WebSocket endpoint for real-time updates

## Performance Optimization

### Concurrency Model
- **Actors**: Isolated state with message passing
- **Async Runtime**: Tokio for non-blocking I/O
- **Thread Pool**: Configurable worker threads

### Memory Management
- **Arc/Mutex**: Shared state for read-heavy workloads
- **Actor Addresses**: Lightweight references
- **Binary Protocol**: Minimal allocation overhead

### Monitoring
- **Metrics**: Prometheus-compatible metrics
- **Logging**: Structured logging with `tracing`
- **Health Checks**: Liveness and readiness probes

## Development Guide

### Project Structure
```
src/
├── actors/          # Actor implementations
├── handlers/        # HTTP/WebSocket handlers
├── services/        # Business logic
├── models/          # Data structures
├── config/          # Configuration
├── utils/           # Utilities
└── main.rs          # Entry point
```

### Adding New Features
1. Define data models in `models/`
2. Create service in `services/`
3. Add handler in `handlers/`
4. Register routes in `main.rs`
5. Update actor if needed

### Testing
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with GPU tests
cargo test --features gpu
```

## Troubleshooting

### Common Issues

1. **Actor Mailbox Full**
   - Increase mailbox capacity
   - Add backpressure handling
   - Check for message loops

2. **WebSocket Disconnections**
   - Check compression settings
   - Monitor message size
   - Verify heartbeat interval

3. **GPU Initialization Failed**
   - Verify GPU drivers
   - Check CUDA/Vulkan support
   - Fall back to CPU mode

### Debug Commands
```bash
# Enable debug logging
RUST_LOG=debug cargo run

# Profile performance
cargo flamegraph

# Check for memory leaks
valgrind ./target/debug/agent-server
```