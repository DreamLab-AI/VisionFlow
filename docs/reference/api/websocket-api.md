# WebSocket API Reference

**Version:** 3.1.0
**Last Updated:** 2025-10-25

---

## WebSocket Endpoints

### Available Streams

- **Knowledge Graph:** `ws://localhost:3030/api/graph/stream`
- **Ontology Graph:** `ws://localhost:3030/api/ontology/graph/stream`
- **Agent Visualization:** `ws://localhost:3030/api/agents/stream`

---

## Connection & Authentication

### Establishing Connection

```javascript
const ws = new WebSocket('ws://localhost:3030/api/graph/stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: '<jwt_token>'
  }));
};
```

### Message Handling

```javascript
ws.onmessage = (event) => {
  // Handle incoming messages (binary or JSON)
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

---

## Server → Client Messages

### Node Update (Binary Protocol V2)

The most frequent message type - 36-byte binary node updates:

```
Format: 36 bytes total
[msg_type: u8] [node_id: u32] [pos_x: f32] [pos_y: f32] [pos_z: f32]
[vel_x: f32] [vel_y: f32] [vel_z: f32] [color_rgba: u32] [flags: u8][u8][u8]
```

### Edge Update

```json
{
  "type": "edge_update",
  "edge": {
    "id": "edge_1",
    "source": 1,
    "target": 2,
    "weight": 1.0
  }
}
```

### Simulation State

```json
{
  "type": "simulation_state",
  "running": true,
  "fps": 59.8,
  "iteration": 1523,
  "converged": false
}
```

### Error Message

```json
{
  "type": "error",
  "code": "INVALID_NODE",
  "message": "Node ID does not exist"
}
```

---

## Client → Server Messages

### Request Full Sync

```json
{
  "type": "request_sync"
}
```

### Update Node Position

```json
{
  "type": "update_position",
  "node_id": 1,
  "position": { "x": 15.5, "y": 25.3, "z": 10.1 }
}
```

### Subscribe to Graph Mode

```json
{
  "type": "subscribe",
  "graph_mode": "knowledge"
}
```

Available modes: `knowledge`, `ontology`, `agent`

### Apply Constraints

```json
{
  "type": "constraints",
  "constraints": [
    {
      "type": "pin",
      "node_id": 1,
      "position": { "x": 0, "y": 0, "z": 0 }
    }
  ]
}
```

### Heartbeat (Keep-Alive)

```json
{
  "type": "ping"
}
```

Server responds with:

```json
{
  "type": "pong"
}
```

---

## Adaptive Broadcasting

The server adapts message frequency based on simulation state:

### Active State (Physics Running)
- **Frequency:** 60 FPS (16.6ms interval)
- **Message Type:** Full node position updates (binary)

### Settled State (Physics Converged)
- **Frequency:** 5 Hz (200ms interval)
- **Message Type:** Delta updates only

### On-Demand
- **Trigger:** Client sends `request_sync`
- **Message Type:** Complete graph dump (JSON)

---

## Performance Characteristics

### Bandwidth Usage

```
Binary Protocol V2: 36 bytes/node × 100k nodes × 60 FPS = 3.6 MB/s
JSON Equivalent:    ~200 bytes/node × 100k nodes × 60 FPS = 20 MB/s
Reduction: 82% bandwidth savings
```

### Latency

- **Binary update:** <10ms end-to-end
- **Server processing:** <2ms
- **Client parsing:** <3ms

---

## Related Documentation

- **[Binary Protocol Specification](./binary-protocol.md)** - Wire format details
- **[REST API](./rest-api.md)** - HTTP endpoints
- **[API Overview](./README.md)** - API introduction

---

**Document Maintained By:** VisionFlow API Team
**Last Review:** 2025-10-25
**Next Review:** 2025-11-25
