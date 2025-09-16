# Real-time Communication Architecture

*This file covers WebSocket protocols and binary communication formats.*

## Overview

VisionFlow implements high-performance real-time communication through:
- Binary WebSocket protocols
- Optimized data serialization
- Efficient message routing
- Connection management

## Binary Protocol

### Protocol Specification
See [Binary Protocol](../api/binary-protocol.md) for complete specification.

**34-byte node update format:**
- Node ID (2 bytes)
- Position XYZ (12 bytes)
- Velocity XYZ (12 bytes)
- SSSP distance (8 bytes)

### Performance Benefits
- **84.8% bandwidth reduction** vs JSON
- **Sub-millisecond** serialization
- **60 FPS** update capability
- **Zero-copy** optimizations

## WebSocket Streams

### Stream Types
1. **Position Updates** - Real-time node positions
2. **Agent Visualization** - AI agent telemetry
3. **MCP Relay** - AI agent communication
4. **Settings Sync** - Configuration updates

### Connection Management
- Automatic reconnection with exponential backoff
- Heartbeat monitoring
- Message queuing during disconnections
- Priority-based message handling

## Message Routing

### Client-Server Communication
```
Client → WebSocket Handler → Actor System → Services
```

### Message Types
- **Binary Protocol** - Position updates
- **JSON Messages** - Configuration and control
- **MCP Protocol** - AI agent communication
- **Control Messages** - System commands

## Performance Characteristics

| Metric | Performance |
|--------|-------------|
| Latency | <10ms |
| Throughput | 1000+ msg/sec |
| Concurrent Connections | 1000+ |
| Update Rate | 5-60 Hz |

## Related Documentation

- [Binary Protocol Specification](../api/binary-protocol.md)
- [WebSocket API Reference](../api/websocket.md)
- [WebSocket Protocols Overview](../api/websocket-protocols.md)
- [Client WebSocket Integration](../client/websocket.md)

---

[← Back to Architecture Documentation](README.md)