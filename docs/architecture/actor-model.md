# Actor Model Architecture

*[Architecture](../index.md)*

## Overview

VisionFlow uses the Actor Model for concurrent, fault-tolerant system design built on Actix actors.

## Actor System Design

For detailed information about our actor-based architecture, see:

- [System Overview](./system-overview.md) - Complete system architecture
- [Server Architecture](../server/architecture.md) - Server-side actor implementation

## Key Actors

Our main actors include:
- **ClaudeFlowActor**: MCP integration and agent orchestration (refactored for resilience)
- **GraphActor**: Graph data management with position update filtering
- **GPUActor**: GPU compute coordination with stability monitoring
- **ClientManager**: WebSocket client management with connection pooling

## Actor System Improvements (Based on Archive Analysis)

### ClaudeFlowActor Refactoring

The ClaudeFlowActor has been refactored into three focused components for improved reliability:

#### 1. Connection Management Actor
- **Fresh TCP Connections**: Uses connection pooling instead of persistent connections for MCP compatibility
- **Circuit Breaker Pattern**: Automatic connection health monitoring with recovery
- **Exponential Backoff**: 1s to 30s reconnection delays to prevent connection storms
- **Heartbeat Monitoring**: 30-second ping/pong cycles with timeout detection

#### 2. Message Processing Actor
- **Message Queuing**: Persistent queues with retry logic during disconnections
- **Priority Processing**: Agent nodes receive preferential treatment in message queues
- **Batch Operations**: Bulk update processing with deduplication
- **Error Recovery**: Structured error handling with fallback mechanisms

#### 3. Agent Lifecycle Actor
- **Agent Spawning**: Dynamic agent creation with type selection
- **Status Monitoring**: Real-time agent telemetry and health checking
- **Swarm Management**: Initialize, scale, and destroy agent swarms
- **Graceful Shutdown**: Proper cleanup and resource deallocation

### Performance Improvements

#### Binary Protocol Optimization
- **34-byte Wire Format**: Optimized from previous 28-byte format with SSSP fields
- **84.8% Bandwidth Reduction**: Through selective compression and delta updates
- **Node Type Flags**: Bit-level encoding for agent/knowledge discrimination
- **SSSP Integration**: Shortest-path data included in binary stream

#### Position Update Filtering
- **Micro-movement Detection**: Filters position changes smaller than threshold
- **KE-based Processing**: Reduces unnecessary updates when system is stable
- **Priority Queuing**: Agent nodes processed with higher priority
- **Batch Updates**: Groups multiple position changes for efficient transmission

## Known Issues and Current Status

### Actor System Health
- ✅ **Connection Resilience**: Circuit breakers and exponential backoff implemented
- ✅ **Message Reliability**: Queuing and retry mechanisms operational
- ✅ **Performance Optimization**: Binary protocol achieving 84.8% bandwidth reduction
- ⚠️ **GPU Integration**: Position updates continue when KE=0 (requires stability gates)
- ✅ **Agent Management**: Full lifecycle management implemented

### Monitoring and Debugging

#### Actor Health Checks
```bash
# Monitor actor system health
docker logs visionflow-backend | grep -E "(Actor|Connection|Circuit)"

# Check MCP connection status
curl -X POST http://localhost:3000/api/bots/check-mcp-connection

# Monitor WebSocket connections
docker logs visionflow-backend | grep -E "(WebSocket|Binary.*bytes)"
```

#### Performance Metrics
```bash
# Check binary protocol efficiency
docker logs visionflow-backend | grep "bandwidth reduction"

# Monitor position update filtering
docker logs visionflow-backend | grep "position.*filtered"
```

## Related Documentation

- [Architecture Overview](./index.md)
- [System Overview](./system-overview.md)
- [Server Architecture](../server/architecture.md)
- [GPU Compute Improvements](gpu-compute-improvements.md)
- [ClaudeFlow Actor Details](claude-flow-actor.md)