# Backend Services

Rust services and actors powering the VisionFlow backend infrastructure.

## Overview

The VisionFlow backend is built with Rust and the Actix framework, providing:
- **Actor-based Architecture** for fault tolerance
- **High-performance APIs** with Actix-Web
- **GPU Acceleration** with CUDA integration
- **Real-time Communication** via WebSockets
- **Microservice Architecture** for scalability

## Core Services

### Graph Service
Central graph data management and processing.

#### Features
- Knowledge graph parsing and storage
- Agent graph real-time updates
- Semantic relationship analysis
- Graph topology management

#### API Endpoints
- `GET /api/graph/data` - Retrieve graph structure
- `POST /api/graph/update` - Update graph data
- `POST /api/graph/refresh` - Refresh from sources

### GPU Compute Service
High-performance physics and analytics computation.

#### Capabilities
- Force-directed layout algorithms
- Parallel graph analysis
- Real-time clustering
- Constraint solving

#### Performance
- **60 FPS** physics simulation
- **100,000+** node capacity
- **Sub-millisecond** computation
- **Multi-GPU** support

### WebSocket Service
Real-time communication infrastructure.

#### Protocol Support
- Binary position updates
- JSON control messages
- MCP protocol relay
- Compressed data streams

#### Connection Management
- Auto-reconnection handling
- Message queuing
- Priority-based routing
- Rate limiting

### Authentication Service
Security and access control.

#### Features
- Nostr-based authentication
- Session management
- Role-based access control
- API key management

## Actor System

### Core Actors

#### GraphServiceActor
Manages graph data and operations.
- Thread-safe data structures
- Concurrent read/write handling
- Change notification system
- Data persistence

#### GPUComputeActor
Coordinates GPU computation.
- CUDA kernel management
- Memory pool allocation
- Computation scheduling
- Error handling and recovery

#### ClientManagerActor
Manages WebSocket connections.
- Connection lifecycle management
- Message broadcasting
- Client state tracking
- Load balancing

#### SettingsActor
Configuration management.
- Dynamic settings updates
- Validation and constraints
- Change propagation
- Persistence

#### EnhancedClaudeFlowActor
AI agent coordination.
- MCP protocol handling
- Agent lifecycle management
- Task orchestration
- Performance monitoring

### Actor Communication

```rust
// Message passing example
#[derive(Message)]
#[rtype(result = "Result<GraphData, GraphError>")]
pub struct GetGraphData {
    pub filter: Option<GraphFilter>,
}

// Actor implementation
impl Handler<GetGraphData> for GraphServiceActor {
    type Result = Result<GraphData, GraphError>;

    fn handle(&mut self, msg: GetGraphData, _: &mut Context<Self>) -> Self::Result {
        self.get_filtered_graph(msg.filter)
    }
}
```

## API Architecture

### REST Endpoints
Synchronous request/response patterns.
- Graph data operations
- Configuration management
- System health checks
- Analytics queries

### WebSocket Streams
Asynchronous real-time updates.
- Position streaming
- Agent visualization
- Settings synchronization
- Event notifications

### Binary Protocol
Optimized data serialization.
- 34-byte node updates
- Zero-copy optimization
- Cross-platform compatibility
- Version negotiation

## Data Flow

```
External Sources → Graph Service → GPU Compute → WebSocket → Clients
     ↓                ↓              ↓            ↓
   GitHub           Database      CUDA Kernel   Binary Protocol
   API/MCP         Persistence    Physics       Real-time Updates
```

## Performance Features

### High-Performance Computing
- **CUDA Integration** for parallel processing
- **Memory Pool Management** for allocation efficiency
- **Zero-Copy Networking** for minimal overhead
- **Async I/O** with Tokio runtime

### Scalability
- **Horizontal Scaling** with stateless design
- **Load Balancing** across multiple instances
- **Connection Pooling** for external services
- **Caching Strategies** for frequently accessed data

### Reliability
- **Fault Tolerance** with actor supervision
- **Graceful Degradation** when services fail
- **Health Monitoring** with metrics collection
- **Automatic Recovery** from transient failures

## Service Integration

### External Services

#### Claude Flow MCP
AI agent orchestration platform.
- Direct TCP connection on port 9500
- JSON-RPC 2.0 protocol
- Multi-swarm management
- Real-time telemetry

#### GitHub API
Source code and repository integration.
- Repository analysis
- Issue tracking
- Pull request management
- Webhook handling

#### RAGFlow
Knowledge processing service.
- Document analysis
- Semantic extraction
- Graph generation
- Content enrichment

### Internal Services

#### Database Layer
Data persistence and caching.
- SQLite for local storage
- Redis for caching
- Connection pooling
- Migration management

#### Monitoring Stack
Observability and metrics.
- Prometheus metrics
- Structured logging
- Distributed tracing
- Health checks

## Configuration Management

### Service Configuration
```toml
[services]
graph_service = { enabled = true, workers = 4 }
gpu_compute = { enabled = true, device_id = 0 }
websocket = { port = 3001, max_connections = 1000 }
```

### Environment Variables
- `VISIONFLOW_LOG_LEVEL` - Logging verbosity
- `VISIONFLOW_GPU_DEVICE` - GPU device selection
- `VISIONFLOW_WS_PORT` - WebSocket port
- `VISIONFLOW_DB_URL` - Database connection

## Development Guidelines

### Code Standards
- Rust idioms and best practices
- Error handling with Result types
- Async/await for I/O operations
- Comprehensive testing

### Testing Strategy
- Unit tests for individual components
- Integration tests for service interaction
- Performance benchmarks
- Load testing

### Documentation Standards
- Rust doc comments for all public APIs
- Architecture decision records
- API documentation
- Deployment guides

## Related Documentation

- [Server Architecture](../server/architecture.md)
- [Actor System Details](../architecture/actor-model.md)
- [GPU Compute Integration](../architecture/gpu-compute.md)
- [API Documentation](../api/README.md)

---

[← Back to Documentation](../README.md)