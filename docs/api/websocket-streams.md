# WebSocket Streams

*[API Documentation](README.md) > WebSocket Streams*

## Overview

VisionFlow's WebSocket infrastructure provides real-time bidirectional communication using an optimized binary protocol alongside JSON control messages. The system achieves **84.8% bandwidth reduction** through intelligent compression and binary encoding.

## Connection Architecture

```mermaid
graph TB
    subgraph "Client Applications"
        Browser[Web Browser]
        XR[XR/VR Devices]
        Mobile[Mobile Apps]
        CLI[CLI Tools]
    end

    subgraph "WebSocket Endpoints"
        Primary[/wss<br/>Binary Position Stream]
        Speech[/ws/speech<br/>Voice Interaction]
        MCP[/ws/mcp-relay<br/>JSON-RPC 2.0]
        Agents[/ws/bots_visualization<br/>Agent Monitoring]
    end

    subgraph "Backend Services"
        GPU[GPU Compute Engine]
        Actors[Actor System]
        McpServer[MCP Server<br/>Port 9500]
    end

    Browser --> Primary
    Browser --> Speech
    Browser --> MCP
    Browser --> Agents
    XR --> Primary
    CLI --> MCP

    Primary --> GPU
    Speech --> Actors
    MCP --> McpServer
    Agents --> Actors
```

## WebSocket Endpoints

### 1. Primary Graph Stream
- **Endpoint**: `/wss`
- **Purpose**: Real-time position updates for graph visualization
- **Protocol**: Binary (34-byte format) + JSON control messages
- **Performance**: 5-60 Hz updates, 94% bandwidth reduction vs JSON

### 2. Speech Interaction
- **Endpoint**: `/ws/speech`
- **Purpose**: Voice commands and audio streaming
- **Protocol**: JSON control + binary audio data
- **Features**: Real-time transcription, TTS, voice commands

### 3. MCP Protocol Relay
- **Endpoint**: `/ws/mcp-relay`
- **Purpose**: Agent orchestration via MCP protocol
- **Protocol**: JSON-RPC 2.0 relay to TCP port 9500
- **Features**: Multi-swarm management, tool invocation

### 4. Agent Visualization
- **Endpoint**: `/ws/bots_visualization`
- **Purpose**: Real-time agent state monitoring
- **Protocol**: JSON with 60 FPS position updates
- **Features**: Performance metrics, swarm coordination

## Binary Protocol Specification

### Position Update Format (34 bytes)

```
Offset  Size  Type    Description
0       2     u16     Node ID (with type flags in high bits)
2       4     f32     Position X (IEEE 754 little-endian)
6       4     f32     Position Y (IEEE 754 little-endian)
10      4     f32     Position Z (IEEE 754 little-endian)
14      4     f32     Velocity X
18      4     f32     Velocity Y
22      4     f32     Velocity Z
26      4     f32     SSSP Distance (shortest path)
30      4     i32     SSSP Parent (for path reconstruction)
```

### Node Type Flags

Type information is encoded in the high bits of the Node ID:

| Bit Mask | Type | Description |
|----------|------|-------------|
| `0x8000` | Agent | AI agent node |
| `0x4000` | Knowledge | Knowledge graph node |
| `0x0000` | Standard | Default node type |

**Node ID extraction**: `actualId = nodeId & 0x3FFF`

### Binary Message Processing

#### Encoding Example (Rust)
```rust
pub struct WireNodeDataItem {
    pub id: u16,              // 2 bytes with type flags
    pub position: Vec3Data,   // 12 bytes
    pub velocity: Vec3Data,   // 12 bytes
    pub sssp_distance: f32,   // 4 bytes
    pub sssp_parent: i32,     // 4 bytes
}

// Example: Agent node ID=1, pos=(10,20,30), vel=(0.1,0.2,0.3)
let wire_data = WireNodeDataItem {
    id: 0x8001,              // Agent flag + ID 1
    position: Vec3Data { x: 10.0, y: 20.0, z: 30.0 },
    velocity: Vec3Data { x: 0.1, y: 0.2, z: 0.3 },
    sssp_distance: 5.5,
    sssp_parent: 42,
};
```

#### Decoding Example (JavaScript)
```javascript
function decodeBinaryPositions(buffer) {
  const BYTES_PER_NODE = 34;
  const view = new DataView(buffer);
  const positions = new Map();

  for (let i = 0; i < buffer.byteLength; i += BYTES_PER_NODE) {
    const flaggedId = view.getUint16(i, true);

    // Extract type flags
    const isAgent = (flaggedId & 0x8000) !== 0;
    const isKnowledge = (flaggedId & 0x4000) !== 0;
    const actualId = flaggedId & 0x3FFF;

    const position = {
      x: view.getFloat32(i + 2, true),
      y: view.getFloat32(i + 6, true),
      z: view.getFloat32(i + 10, true)
    };

    const velocity = {
      x: view.getFloat32(i + 14, true),
      y: view.getFloat32(i + 18, true),
      z: view.getFloat32(i + 22, true)
    };

    const ssspDistance = view.getFloat32(i + 26, true);
    const ssspParent = view.getInt32(i + 30, true);

    positions.set(actualId, {
      position,
      velocity,
      ssspDistance,
      ssspParent,
      type: isAgent ? 'agent' : isKnowledge ? 'knowledge' : 'standard'
    });
  }

  return positions;
}
```

## JSON Control Messages

### Base Message Format
```typescript
interface WebSocketMessage<T = any> {
  type: string;          // Message type identifier
  payload?: T;           // Type-specific payload
  timestamp?: string;    // ISO 8601 timestamp
  id?: string;          // Message ID for correlation
}
```

### Primary Stream (/wss)

#### Connection Flow
```json
// 1. Server sends on connection
{"type": "connection_established", "timestamp": 1694851200000}

// 2. Client requests initial data
{"type": "requestInitialData"}

// 3. Server confirms updates started
{"type": "updatesStarted", "timestamp": 1694851205000}

// 4. Binary position updates begin automatically
// (34-byte binary messages at 5-60 Hz)
```

#### Control Messages
```json
// Request full graph snapshot
{"type": "request_full_snapshot", "graphs": ["knowledge", "agent"]}

// Request agent positions specifically
{"type": "requestBotsPositions"}

// Heartbeat (every 30 seconds)
{"type": "ping", "timestamp": 1694851300000}
{"type": "pong", "timestamp": 1694851300000}

// Loading state during computation
{"type": "loading", "message": "Calculating initial layout..."}
```

### Agent Visualization (/ws/bots_visualization)

#### Initialization
```json
{
  "type": "multi-agent-init",
  "payload": {
    "swarmId": "swarm_1757880683494_yl81sece5",
    "topology": "hierarchical",
    "maxAgents": 10,
    "coordinator": "agent_1757967065850_coord"
  }
}
```

#### Agent State Updates
```json
{
  "type": "agent-state-update",
  "payload": {
    "agentId": "agent-001",
    "status": "executing",         // "idle", "busy", "error", "offline"
    "currentTask": "Analyzing research papers",
    "progress": 0.65,
    "health": 0.92,
    "workload": 0.78,
    "metrics": {
      "tokensUsed": 1523,
      "tasksCompleted": 42,
      "successRate": 0.95,
      "avgResponseTime": 250
    },
    "position": {
      "x": 150.5,
      "y": 200.3,
      "z": 50.0
    }
  }
}
```

#### Full Agent Update (Bulk)
```json
{
  "type": "bots-full-update",
  "payload": {
    "agents": [
      {
        "id": "agent-001",
        "name": "Research Agent Alpha",
        "type": "researcher",
        "status": "active",
        "capabilities": ["web_search", "document_analysis"],
        "metrics": {
          "tasksCompleted": 42,
          "successRate": 0.95,
          "tokenRate": 1523.4
        }
      }
    ],
    "connections": [
      {
        "from": "agent-001",
        "to": "agent-002",
        "strength": 0.8,
        "messageRate": 12.5,
        "bandwidth": 1024
      }
    ],
    "swarmMetrics": {
      "throughput": 15.3,
      "efficiency": 0.87,
      "coordination": 0.93
    }
  }
}
```

### Speech Interaction (/ws/speech)

#### Text-to-Speech
```json
{
  "type": "textToSpeech",
  "payload": {
    "text": "Graph analysis complete. Found 3 clusters with high modularity.",
    "voice": "neural",     // "standard", "neural", "premium"
    "speed": 1.0,         // 0.5 to 2.0
    "pitch": 1.0,         // 0.5 to 2.0
    "stream": true        // Stream audio chunks
  }
}
```

#### Speech-to-Text Control
```json
{
  "type": "sttAction",
  "payload": {
    "action": "start",    // "start", "stop", "pause", "resume"
    "language": "en-US",
    "model": "whisper",   // "whisper", "google", "azure"
    "continuousMode": true,
    "confidenceThreshold": 0.7
  }
}
```

#### Audio Data (Binary)
- Binary WebSocket frames containing PCM audio data
- Sample rate: 16kHz, 16-bit, mono
- Chunk size: 1024 samples (64ms at 16kHz)
- Format: Little-endian signed integers

### MCP Protocol Relay (/ws/mcp-relay)

#### Tool Invocation
```json
{
  "jsonrpc": "2.0",
  "id": "call-123",
  "method": "tools/call",
  "params": {
    "name": "swarm_init",
    "arguments": {
      "topology": "hierarchical",
      "maxAgents": 10,
      "swarmId": "swarm_1757880683494_yl81sece5"
    }
  }
}
```

#### Tool Response
```json
{
  "jsonrpc": "2.0",
  "id": "call-123",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Swarm initialized successfully with 10 agents in hierarchical topology"
      }
    ]
  }
}
```

#### Agent Command Execution
```json
{
  "type": "execute-command",
  "payload": {
    "swarmId": "swarm_1757880683494_yl81sece5",
    "agentId": "agent-001",
    "command": "analyze_document",
    "parameters": {
      "documentUrl": "https://example.com/research.pdf",
      "extractInsights": true,
      "outputFormat": "structured"
    },
    "priority": "high",
    "timeout": 30000
  }
}
```

## Performance Characteristics

### Bandwidth Optimization

| Scenario | JSON Protocol | Binary Protocol | Reduction |
|----------|---------------|-----------------|----------|
| 100 nodes @ 60fps | 3.0 MB/s | 200 KB/s | **93%** |
| 500 nodes @ 60fps | 15 MB/s | 1.0 MB/s | **93%** |
| 1000 nodes @ 60fps | 30 MB/s | 2.0 MB/s | **93%** |
| Mixed (agents + knowledge) | 25 MB/s | 1.2 MB/s | **95%** |

### Compression Strategy
- **Selective Compression**: Messages >256 bytes use GZIP
- **Binary Optimization**: 34-byte fixed format vs 150+ byte JSON
- **Delta Encoding**: Only changed values for settings
- **Priority Queuing**: Agent nodes processed first

### Update Rates
- **Primary Stream**: 5 Hz minimum, up to 60 Hz for active sessions
- **Agent Visualization**: 60 FPS for smooth monitoring
- **Speech**: Real-time streaming with <200ms latency
- **MCP Relay**: Request/response based, <500ms typical

## Connection Management

### Reconnection Strategy
```typescript
class WebSocketManager {
  private reconnectAttempts = 0;
  private maxAttempts = 5;
  private baseDelay = 1000; // 1 second

  async reconnect() {
    if (this.reconnectAttempts >= this.maxAttempts) {
      throw new Error('Max reconnection attempts reached');
    }

    // Exponential backoff with jitter
    const delay = Math.min(
      this.baseDelay * Math.pow(2, this.reconnectAttempts),
      30000 // Max 30 seconds
    ) + Math.random() * 1000;

    await new Promise(resolve => setTimeout(resolve, delay));
    this.reconnectAttempts++;

    return this.connect();
  }

  onConnect() {
    this.reconnectAttempts = 0; // Reset on successful connection
  }
}
```

### Heartbeat Configuration
```typescript
const HEARTBEAT_CONFIG = {
  '/wss': { interval: 30000, timeout: 60000 },
  '/ws/speech': { interval: 5000, timeout: 10000 },
  '/ws/mcp-relay': { interval: 30000, timeout: 60000 },
  '/ws/bots_visualization': { interval: 15000, timeout: 30000 }
};
```

### Connection Health Monitoring
```typescript
interface ConnectionMetrics {
  messagesReceived: number;
  messagesSent: number;
  bytesReceived: number;
  bytesSent: number;
  lastActivity: number;
  averageLatency: number;
  connectionQuality: 'excellent' | 'good' | 'poor' | 'critical';
}
```

## Rate Limiting & Security

### Rate Limits by Endpoint

| Endpoint | JSON Messages/min | Binary Data | Burst |
|----------|-------------------|-------------|-------|
| `/wss` | 1000 | Server-controlled (5-60 Hz) | 100 |
| `/ws/speech` | 500 | 10 MB/min | 50 |
| `/ws/mcp-relay` | 100 | N/A | 20 |
| `/ws/bots_visualization` | 1000 | 1 MB/min | 100 |

### Security Features

#### Message Validation
```typescript
interface MessageValidator {
  validateJSON(message: string): ValidationResult;
  validateBinary(data: ArrayBuffer): ValidationResult;
  checkRateLimit(clientId: string, endpoint: string): boolean;
  sanitizeInput(data: any): any;
}

interface ValidationResult {
  valid: boolean;
  errors?: string[];
  sanitized?: any;
}
```

#### Authentication
- **Session-based**: WebSocket inherits HTTP session authentication
- **Token-based**: Optional query parameter for token-based auth
- **Origin validation**: CORS and origin header verification
- **Rate limiting**: Per-client connection and message limits

## Integration Examples

### React Hook for Position Updates
```typescript
function useVisionFlowWebSocket() {
  const [positions, setPositions] = useState(new Map());
  const [connectionState, setConnectionState] = useState('disconnected');
  const [agents, setAgents] = useState([]);

  useEffect(() => {
    // Primary position stream
    const positionWs = new WebSocket('ws://localhost:3001/wss');

    positionWs.onopen = () => {
      setConnectionState('connected');
      positionWs.send(JSON.stringify({ type: 'requestInitialData' }));
    };

    positionWs.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const updates = decodeBinaryPositions(event.data);
        setPositions(prev => new Map([...prev, ...updates]));
      } else {
        const message = JSON.parse(event.data);
        handleControlMessage(message);
      }
    };

    // Agent visualization stream
    const agentWs = new WebSocket('ws://localhost:3001/ws/bots_visualization');

    agentWs.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'bots-full-update') {
        setAgents(message.payload.agents);
      } else if (message.type === 'agent-state-update') {
        setAgents(prev => prev.map(agent =>
          agent.id === message.payload.agentId
            ? { ...agent, ...message.payload }
            : agent
        ));
      }
    };

    return () => {
      positionWs.close();
      agentWs.close();
    };
  }, []);

  return { positions, agents, connectionState };
}
```

### Three.js Scene Integration
```typescript
class VisionFlowRenderer {
  private scene: THREE.Scene;
  private positions = new Map<number, PositionData>();

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.initWebSocket();
  }

  private initWebSocket() {
    const ws = new WebSocket('ws://localhost:3001/wss');

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const updates = decodeBinaryPositions(event.data);
        this.updatePositions(updates);
      }
    };
  }

  private updatePositions(updates: Map<number, PositionData>) {
    updates.forEach((data, nodeId) => {
      const mesh = this.scene.getObjectByName(`node-${nodeId}`);
      if (mesh) {
        // Update position
        mesh.position.set(data.position.x, data.position.y, data.position.z);

        // Store velocity for interpolation
        mesh.userData.velocity = data.velocity;
        mesh.userData.ssspDistance = data.ssspDistance;

        // Apply type-specific styling
        if (data.type === 'agent') {
          this.styleAsAgent(mesh);
        } else if (data.type === 'knowledge') {
          this.styleAsKnowledge(mesh);
        }
      }
    });
  }

  private styleAsAgent(mesh: THREE.Object3D) {
    const material = mesh.material as THREE.MeshBasicMaterial;
    material.color.setHex(0xff6b6b); // Red for agents
    mesh.scale.setScalar(1.2); // Slightly larger
  }

  private styleAsKnowledge(mesh: THREE.Object3D) {
    const material = mesh.material as THREE.MeshBasicMaterial;
    material.color.setHex(0x4ecdc4); // Teal for knowledge
    mesh.scale.setScalar(1.0); // Standard size
  }
}
```

### MCP Agent Control
```typescript
class MCPController {
  private ws: WebSocket;
  private requestId = 1;

  constructor() {
    this.ws = new WebSocket('ws://localhost:3001/ws/mcp-relay');
  }

  async initializeSwarm(topology: string, maxAgents: number): Promise<string> {
    const request = {
      jsonrpc: '2.0',
      id: this.requestId++,
      method: 'tools/call',
      params: {
        name: 'swarm_init',
        arguments: { topology, maxAgents }
      }
    };

    return new Promise((resolve, reject) => {
      const handleResponse = (event: MessageEvent) => {
        const response = JSON.parse(event.data);
        if (response.id === request.id) {
          this.ws.removeEventListener('message', handleResponse);
          if (response.result) {
            resolve(response.result.content[0].text);
          } else {
            reject(new Error(response.error.message));
          }
        }
      };

      this.ws.addEventListener('message', handleResponse);
      this.ws.send(JSON.stringify(request));
    });
  }

  async executeAgentTask(swarmId: string, agentId: string, task: string) {
    const command = {
      type: 'execute-command',
      payload: {
        swarmId,
        agentId,
        command: 'execute_task',
        parameters: { task },
        priority: 'normal',
        timeout: 30000
      }
    };

    this.ws.send(JSON.stringify(command));
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Binary Data Corruption
```typescript
// Validate message size
if (buffer.byteLength % 34 !== 0) {
  console.error('Invalid binary message size:', buffer.byteLength);
  return;
}

// Check for valid node IDs
const nodeId = view.getUint16(0, true) & 0x3FFF;
if (nodeId === 0 || nodeId > 65535) {
  console.error('Invalid node ID:', nodeId);
}
```

#### 2. Connection Drops
```typescript
// Monitor connection health
class ConnectionMonitor {
  private lastMessage = Date.now();

  onMessage() {
    this.lastMessage = Date.now();
  }

  checkHealth() {
    const timeSinceLastMessage = Date.now() - this.lastMessage;
    if (timeSinceLastMessage > 60000) {
      console.warn('Connection appears stale, reconnecting...');
      this.reconnect();
    }
  }
}
```

#### 3. High Memory Usage
```typescript
// Implement object pooling for frequent updates
class PositionPool {
  private pool: PositionData[] = [];

  acquire(): PositionData {
    return this.pool.pop() || { position: {x:0,y:0,z:0}, velocity: {x:0,y:0,z:0} };
  }

  release(data: PositionData) {
    this.pool.push(data);
  }
}
```

### Debug Configuration
```typescript
// Enable comprehensive logging
const DEBUG_CONFIG = {
  logBinaryMessages: true,
  logJSONMessages: true,
  logPerformanceMetrics: true,
  logConnectionEvents: true
};

if (DEBUG_CONFIG.logBinaryMessages) {
  ws.addEventListener('message', (event) => {
    if (event.data instanceof ArrayBuffer) {
      console.log(`Binary message: ${event.data.byteLength} bytes, ${event.data.byteLength / 34} nodes`);

      // Log first few bytes for inspection
      const bytes = new Uint8Array(event.data.slice(0, 34));
      console.log('First node data:', Array.from(bytes).map(b => b.toString(16)).join(' '));
    }
  });
}
```

## Performance Monitoring

### Metrics Collection
```typescript
class WebSocketMetrics {
  private metrics = {
    messagesReceived: 0,
    messagesSent: 0,
    bytesReceived: 0,
    bytesSent: 0,
    reconnections: 0,
    averageLatency: 0,
    compressionRatio: 0
  };

  onMessage(event: MessageEvent) {
    this.metrics.messagesReceived++;

    if (event.data instanceof ArrayBuffer) {
      this.metrics.bytesReceived += event.data.byteLength;
    } else {
      this.metrics.bytesReceived += new Blob([event.data]).size;
    }

    // Calculate compression ratio
    this.updateCompressionRatio();
  }

  getMetrics() {
    return { ...this.metrics };
  }
}
```

---

*For binary protocol details, see [binary-protocol.md](binary-protocol.md). For REST API endpoints, see [rest-endpoints.md](rest-endpoints.md).*