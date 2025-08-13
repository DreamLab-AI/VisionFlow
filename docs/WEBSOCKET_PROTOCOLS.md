# WebSocket Protocols and Message Schemas

## Overview

This document details the WebSocket protocols used for real-time communication between the agent visualisation frontend and the backend orchestration system. We support both binary and JSON protocols for different use cases.

## Connection Establishment

### WebSocket Endpoints

1. **Primary Agent Data Stream**: `wss://[host]/api/visualisation/agents/ws`
2. **Graph Data Stream**: `wss://[host]/api/graph/ws`
3. **Control Channel**: `wss://[host]/api/control/ws`

### Authentication

```javascript
// Connection with JWT authentication
const ws = new WebSocket('wss://api.example.com/agents/ws', {
  headers: {
    'Authorization': `Bearer ${authToken}`,
    'X-Client-Version': '1.0.0',
    'X-Requested-Protocols': 'binary,json'
  }
});
```

### Handshake Protocol

```json
// Client -> Server: Initial handshake
{
  "type": "handshake",
  "version": "1.0.0",
  "capabilities": ["binary", "compression", "multiplexing"],
  "clientId": "client-uuid",
  "timestamp": "2024-01-10T12:00:00Z"
}

// Server -> Client: Handshake response
{
  "type": "handshake_ack",
  "sessionId": "session-uuid",
  "protocols": ["binary", "json"],
  "heartbeatInterval": 30000,
  "compression": "zlib",
  "serverTime": "2024-01-10T12:00:00Z"
}
```

## Binary Protocol Specification

### Message Structure

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Magic (4)  │  Type (1)   │  Flags (1)  │  Length (2) │
├─────────────┴─────────────┴─────────────┴─────────────┤
│                    Payload (variable)                   │
└─────────────────────────────────────────────────────────┘
```

### Header Fields

| Field | Size | Description |
|-------|------|-------------|
| Magic | 4 bytes | 0x41474E54 ("AGNT") |
| Type | 1 byte | Message type identifier |
| Flags | 1 byte | Compression, priority, etc. |
| Length | 2 bytes | Payload length (max 65535) |

### Message Types

```typescript
enum BinaryMessageType {
  POSITION_UPDATE = 0x01,
  STATE_BATCH = 0x02,
  PERFORMANCE_METRICS = 0x03,
  COORDINATION_EVENT = 0x04,
  COMPRESSED_BATCH = 0x05,
  HEARTBEAT = 0x06,
  ERROR = 0xFF
}
```

### Position Update Format

```
// Type: 0x01 (POSITION_UPDATE)
// Payload structure for N agents:
┌────────────┬──────────────────────────────────┐
│ Count (4)  │ Agent Data Array (28 * N bytes)  │
└────────────┴──────────────────────────────────┘

// Each agent data (28 bytes):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ ID (4)  │ PosX(4) │ PosY(4) │ PosZ(4) │ VelX(4) │ VelY(4) │ VelZ(4) │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

### State Batch Format

```
// Type: 0x02 (STATE_BATCH)
// Variable length encoded states
┌────────────┬─────────────────────────┐
│ Count (4)  │ State Records (variable)│
└────────────┴─────────────────────────┘

// Each state record:
┌────────┬────────┬────────┬───────────────┐
│ ID (4) │State(1)│Flags(1)│ Metadata (var)│
└────────┴────────┴────────┴───────────────┘
```

### Performance Metrics Format

```
// Type: 0x03 (PERFORMANCE_METRICS)
┌────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ ID (4) │Tasks (4) │Success(4)│AvgMs(4) │ CPU% (4) │ Mem (8)  │
└────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## JSON Protocol Specification

### Message Envelope

```typescript
interface WSMessage<T = any> {
  id: string;           // Message ID for request/response matching
  type: string;         // Message type identifier
  timestamp: string;    // ISO 8601 timestamp
  payload: T;           // Type-specific payload
  metadata?: {
    correlationId?: string;
    priority?: 'low' | 'normal' | 'high' | 'critical';
    ttl?: number;       // Time to live in ms
  };
}
```

### Agent Event Messages

#### Agent State Change

```json
{
  "id": "msg-123",
  "type": "agent.state.changed",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "agentId": "agent-001",
    "previousState": "IDLE",
    "newState": "EXECUTING",
    "reason": "Task assigned",
    "metadata": {
      "taskId": "task-456",
      "estimatedDuration": 5000
    }
  }
}
```

#### Agent Performance Update

```json
{
  "id": "msg-124",
  "type": "agent.performance.update",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "agentId": "agent-001",
    "metrics": {
      "tasksCompleted": 42,
      "successRate": 0.95,
      "averageResponseTime": 234,
      "resourceUtilization": 0.67,
      "communicationEfficiency": 0.89,
      "uptime": 3600000,
      "messageQueueSize": 3,
      "connectionCount": 7,
      "memoryUsage": 104857600
    },
    "trends": {
      "successRateTrend": "improving",
      "responseTrend": "stable",
      "loadTrend": "increasing"
    }
  }
}
```

#### Agent Capability Update

```json
{
  "id": "msg-125",
  "type": "agent.capability.update",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "agentId": "agent-001",
    "action": "add",
    "capability": {
      "name": "advanced_analysis",
      "category": "analysis",
      "level": 4,
      "description": "Advanced data analysis with ML models"
    }
  }
}
```

### Message Flow Events

#### Message Sent

```json
{
  "id": "msg-126",
  "type": "message.sent",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "messageId": "msg-789",
    "from": { "id": "agent-001", "namespace": "research" },
    "to": [{ "id": "agent-002", "namespace": "analysis" }],
    "messageType": "REQUEST",
    "priority": "HIGH",
    "content": {
      "action": "analyze_data",
      "data": { "source": "dataset-123" }
    },
    "expiry": "2024-01-10T12:05:00Z"
  }
}
```

#### Message Flow Statistics

```json
{
  "id": "msg-127",
  "type": "message.flow.stats",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "flowId": "agent-001-agent-002",
    "statistics": {
      "messageCount": 156,
      "averageLatency": 45.2,
      "maxLatency": 234,
      "minLatency": 12,
      "successRate": 0.98,
      "bandwidth": 0.75,
      "lastActivity": "2024-01-10T11:59:45Z"
    }
  }
}
```

### Coordination Events

#### Coordination Pattern Initiated

```json
{
  "id": "msg-128",
  "type": "coordination.initiated",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "coordinationId": "coord-123",
    "pattern": "CONSENSUS",
    "initiator": { "id": "agent-001" },
    "participants": [
      { "id": "agent-002", "role": "voter" },
      { "id": "agent-003", "role": "voter" },
      { "id": "agent-004", "role": "observer" }
    ],
    "parameters": {
      "consensusThreshold": 0.66,
      "timeout": 30000,
      "proposal": {
        "type": "resource_allocation",
        "details": { "cpu": 4, "memory": "8GB" }
      }
    }
  }
}
```

#### Coordination Progress

```json
{
  "id": "msg-129",
  "type": "coordination.progress",
  "timestamp": "2024-01-10T12:00:15Z",
  "payload": {
    "coordinationId": "coord-123",
    "pattern": "CONSENSUS",
    "progress": 0.66,
    "status": "VOTING",
    "votes": {
      "agent-001": { "vote": "approve", "timestamp": "2024-01-10T12:00:10Z" },
      "agent-002": { "vote": "approve", "timestamp": "2024-01-10T12:00:12Z" },
      "agent-003": { "vote": "pending" }
    },
    "estimatedCompletion": "2024-01-10T12:00:30Z"
  }
}
```

### System Events

#### System Metrics Update

```json
{
  "id": "msg-130",
  "type": "system.metrics",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "activeAgents": 42,
    "totalAgents": 50,
    "systemLoad": {
      "cpu": 0.67,
      "memory": 0.45,
      "gpu": 0.23
    },
    "messageMetrics": {
      "rate": 1523.4,
      "queueDepth": 234,
      "averageLatency": 23.5,
      "errorRate": 0.002
    },
    "networkHealth": 0.98,
    "coordinationEfficiency": 0.87,
    "alerts": [
      {
        "level": "warning",
        "message": "High message queue depth detected",
        "component": "message-bus",
        "timestamp": "2024-01-10T11:59:50Z"
      }
    ]
  }
}
```

### Control Messages

#### Execute Command

```json
{
  "id": "msg-131",
  "type": "control.execute",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "targetAgent": "agent-001",
    "command": "execute_tool",
    "parameters": {
      "tool": "web_search",
      "query": "latest AI research papers",
      "maxResults": 10
    },
    "timeout": 30000,
    "callback": {
      "type": "webhook",
      "url": "https://api.example.com/callbacks/cmd-131"
    }
  }
}
```

#### multi-agent Configuration Update

```json
{
  "id": "msg-132",
  "type": "control.multi-agent.configure",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "multi-agentId": "multi-agent-001",
    "configuration": {
      "topology": "HIERARCHICAL",
      "coordinationStrategy": "consensus",
      "faultTolerance": {
        "enabled": true,
        "redundancy": 2,
        "heartbeatInterval": 5000
      },
      "performanceTargets": {
        "maxLatency": 100,
        "minThroughput": 1000,
        "targetSuccessRate": 0.99
      }
    }
  }
}
```

## Error Handling

### Error Message Format

```json
{
  "id": "msg-133",
  "type": "error",
  "timestamp": "2024-01-10T12:00:00Z",
  "payload": {
    "code": "AGENT_NOT_FOUND",
    "message": "Agent with ID 'agent-999' not found",
    "details": {
      "requestId": "msg-131",
      "component": "agent-manager",
      "stackTrace": "..."
    },
    "recoverable": true,
    "suggestedAction": "verify_agent_id"
  }
}
```

### Error Codes

| Code | Description | Recoverable |
|------|-------------|-------------|
| CONNECTION_FAILED | WebSocket connection failed | Yes |
| AUTH_FAILED | Authentication failed | Yes |
| INVALID_MESSAGE | Message validation failed | Yes |
| AGENT_NOT_FOUND | Referenced agent doesn't exist | Yes |
| COMMAND_FAILED | Command execution failed | Maybe |
| RATE_LIMIT_EXCEEDED | Too many requests | Yes |
| INTERNAL_ERROR | Server error | No |

## Compression

### Compression Negotiation

```json
{
  "type": "compression.negotiate",
  "algorithms": ["gzip", "deflate", "brotli"],
  "preferredLevel": 6
}
```

### Compressed Message Format

```
┌──────────┬──────────┬──────────────┬───────────────┐
│Magic (4) │Type (1)  │Original (4)  │Compressed(...)|
└──────────┴──────────┴──────────────┴───────────────┘
```

## Multiplexing

### Channel Management

```json
{
  "type": "channel.create",
  "channelId": "perf-metrics",
  "priority": "high",
  "dedicated": true,
  "bufferSize": 1000
}
```

### Channel Message

```json
{
  "channel": "perf-metrics",
  "sequence": 12345,
  "message": { /* actual message */ }
}
```

## Best Practices

### Message Design

1. **Keep payloads small**: Use binary protocol for high-frequency updates
2. **Batch when possible**: Group multiple updates in single message
3. **Use appropriate priorities**: Critical messages should bypass queues
4. **Include correlation IDs**: For request/response tracking
5. **Set TTL for time-sensitive messages**: Prevent stale message processing

### Connection Management

1. **Implement exponential backoff**: For reconnection attempts
2. **Use heartbeats**: Detect connection issues early
3. **Handle partial messages**: Network issues may fragment messages
4. **Implement message acknowledgment**: For critical operations
5. **Monitor connection metrics**: Track latency and throughput

### Performance Optimization

1. **Use binary protocol for positions**: 85% bandwidth reduction
2. **Enable compression for large payloads**: 30-50% additional savings
3. **Implement message deduplication**: Prevent duplicate processing
4. **Use connection pooling**: For high-throughput scenarios
5. **Consider WebRTC for P2P**: Agent-to-agent direct communication

### Security

1. **Always use WSS (TLS)**: Encrypt all communications
2. **Implement rate limiting**: Prevent DoS attacks
3. **Validate all messages**: Use schemas for validation
4. **Sanitize user inputs**: Prevent injection attacks
5. **Rotate authentication tokens**: Limit exposure window

## Protocol Versioning

### Version Negotiation

```json
{
  "type": "version.negotiate",
  "supported": ["1.0", "1.1", "2.0"],
  "preferred": "2.0",
  "features": {
    "1.0": ["basic"],
    "1.1": ["compression", "binary"],
    "2.0": ["multiplexing", "streaming"]
  }
}
```

### Backward Compatibility

- Version 2.x clients can connect to 1.x servers
- Binary protocol available from version 1.1
- Multiplexing requires version 2.0 or higher
- Feature detection via capabilities negotiation