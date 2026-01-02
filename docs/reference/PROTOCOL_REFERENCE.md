---
layout: default
title: "Protocol Reference"
parent: Reference
nav_order: 6
---

# Protocol Reference

**Version**: 2.0
**Last Updated**: December 18, 2025

Complete technical specification for all VisionFlow communication protocols.

---

## Table of Contents

1. [Binary WebSocket Protocol](#binary-websocket-protocol)
2. [REST HTTP Protocol](#rest-http-protocol)
3. [MCP Protocol](#mcp-protocol)
4. [Authentication Protocols](#authentication-protocols)
5. [Protocol Comparison](#protocol-comparison)

---

## Binary WebSocket Protocol

### Protocol Versions

| Version | Status | Byte Size | Use Case |
|---------|--------|-----------|----------|
| **V2** | **Current** | 36 bytes/node | Production standard |
| V3 | Stable | 48 bytes/node | Analytics extension |
| V4 | Experimental | 16 bytes/change | Delta encoding |
| V1 | Deprecated | 34 bytes/node | Legacy (ID limit: 16383) |

---

### Protocol V2 Specification

#### Wire Format

**Total Message Size**: 1 + (36 × node_count) bytes

```
Byte Layout (Little-Endian):
┌─────────┬────────────────────────────────────────────┐
│ Offset  │ Field (Type, Bytes)                        │
├─────────┼────────────────────────────────────────────┤
│ [0]     │ Protocol Version (u8) = 2                  │
├─────────┼────────────────────────────────────────────┤
│ [1-4]   │ Node ID (u32) with type flags             │
│ [5-8]   │ Position X (f32)                          │
│ [9-12]  │ Position Y (f32)                          │
│ [13-16] │ Position Z (f32)                          │
│ [17-20] │ Velocity X (f32)                          │
│ [21-24] │ Velocity Y (f32)                          │
│ [25-28] │ Velocity Z (f32)                          │
│ [29-32] │ SSSP Distance (f32)                       │
│ [33-36] │ SSSP Parent (i32)                         │
└─────────┴────────────────────────────────────────────┘
```

#### Field Specifications

| Field | Type | Offset | Endian | Range | Description |
|-------|------|--------|--------|-------|-------------|
| **Version** | u8 | 0 | N/A | 2 | Protocol version |
| **Node ID** | u32 | 1-4 | Little | 0-1,073,741,823 | Bits 0-29: ID<br>Bits 30-31: Type flags |
| **Position X** | f32 | 5-8 | Little | -∞ to +∞ | X coordinate |
| **Position Y** | f32 | 9-12 | Little | -∞ to +∞ | Y coordinate |
| **Position Z** | f32 | 13-16 | Little | -∞ to +∞ | Z coordinate |
| **Velocity X** | f32 | 17-20 | Little | -∞ to +∞ | X velocity |
| **Velocity Y** | f32 | 21-24 | Little | -∞ to +∞ | Y velocity |
| **Velocity Z** | f32 | 25-28 | Little | -∞ to +∞ | Z velocity |
| **SSSP Distance** | f32 | 29-32 | Little | 0.0 to +∞ | Shortest path distance |
| **SSSP Parent** | i32 | 33-36 | Little | -1 to max | Parent in path tree |

#### Type Flags Encoding

Encoded in high bits of Node ID field:

```rust
// Flag constants (bits 30-31)
const AGENT_NODE_FLAG: u32 = 0x80000000;      // Bit 31
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000;  // Bit 30
const NODE_ID_MASK: u32 = 0x3FFFFFFF;         // Bits 0-29

// Ontology type flags (bits 26-28, only for GraphType::Ontology)
const ONTOLOGY_CLASS_FLAG: u32 = 0x04000000;      // Bit 26
const ONTOLOGY_INDIVIDUAL_FLAG: u32 = 0x08000000; // Bit 27
const ONTOLOGY_PROPERTY_FLAG: u32 = 0x10000000;   // Bit 28
```

**Decoding Example (TypeScript)**:
```typescript
const nodeIdRaw = view.getUint32(offset, true); // Little-endian
const actualId = nodeIdRaw & 0x3FFFFFFF;
const isAgent = (nodeIdRaw & 0x80000000) !== 0;
const isKnowledge = (nodeIdRaw & 0x40000000) !== 0;

// For ontology nodes
const isClass = (nodeIdRaw & 0x04000000) !== 0;
const isIndividual = (nodeIdRaw & 0x08000000) !== 0;
const isProperty = (nodeIdRaw & 0x10000000) !== 0;
```

**Encoding Example (Rust)**:
```rust
fn encode_node_id(id: u32, is_agent: bool, is_knowledge: bool) -> u32 {
    let mut encoded = id & NODE_ID_MASK;
    if is_agent {
        encoded |= AGENT_NODE_FLAG;
    }
    if is_knowledge {
        encoded |= KNOWLEDGE_NODE_FLAG;
    }
    encoded
}
```

---

### Protocol V3 (Analytics Extension)

Extends V2 with 12 additional bytes for machine learning analytics.

**Total Message Size**: 1 + (48 × node_count) bytes

**Additional Fields**:

| Field | Type | Offset | Description |
|-------|------|--------|-------------|
| **Cluster ID** | u32 | 37-40 | K-means cluster assignment (0 = unassigned) |
| **Anomaly Score** | f32 | 41-44 | LOF score: 0.0 = normal, 1.0 = anomaly |
| **Community ID** | u32 | 45-48 | Louvain community (0 = unassigned) |

---

### Protocol V4 (Delta Encoding - Experimental)

Sends only changed nodes for 60-80% bandwidth reduction.

**Wire Format**: 16 bytes per changed node

```
┌─────────┬────────────────────────────────────────────┐
│ [0]     │ Protocol Version (u8) = 4                  │
│ [1-4]   │ Node ID (u32) with type flags             │
│ [5]     │ Change Flags (u8, bit field)              │
│ [6-8]   │ Padding (3 bytes)                         │
│ [9-10]  │ Delta Position X (i16, scaled)            │
│ [11-12] │ Delta Position Y (i16, scaled)            │
│ [13-14] │ Delta Position Z (i16, scaled)            │
│ [15-16] │ Delta Velocity X (i16, scaled)            │
│ [17-18] │ Delta Velocity Y (i16, scaled)            │
│ [19-20] │ Delta Velocity Z (i16, scaled)            │
└─────────┴────────────────────────────────────────────┘
```

**Change Flags**:
```rust
const DELTA_POSITION_CHANGED: u8 = 0x01;
const DELTA_VELOCITY_CHANGED: u8 = 0x02;
```

**Scale Factor**: 100.0 (converts f32 to i16 with 0.01 precision)

**Resync**: Full V2 state sent on frame 0 and every 60 frames

---

### JSON Control Messages

#### Client → Server

##### Subscribe to Position Updates

```json
{
  "type": "subscribe_position_updates",
  "data": {
    "rate": 60,
    "nodeFilter": "all",
    "protocol": "binary-v2"
  }
}
```

##### Filter Update

```json
{
  "type": "filter_update",
  "data": {
    "quality": 0.7,
    "maxNodes": 10000,
    "types": ["concept", "entity", "class"]
  }
}
```

##### Heartbeat (Keep-alive)

```json
{
  "type": "heartbeat",
  "timestamp": 1702915200000
}
```

#### Server → Client

##### Subscription Confirmed

```json
{
  "type": "subscription_confirmed",
  "data": {
    "rate": 60,
    "protocol": "binary-v2",
    "nodeCount": 50000
  }
}
```

##### State Sync

```json
{
  "type": "state_sync",
  "data": {
    "nodes": 50000,
    "edges": 120000,
    "graphType": "knowledge-graph",
    "timestamp": 1702915200000
  }
}
```

---

### Connection Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server

    C->>S: WebSocket Handshake (ws://host:port/ws?token=JWT)
    S->>C: HTTP 101 Switching Protocols

    C->>S: JSON: {"type":"subscribe_position_updates","data":{...}}
    S->>C: JSON: {"type":"subscription_confirmed","data":{...}}

    S->>C: JSON: InitialGraphLoad (metadata + nodes/edges)

    loop Real-time Updates (60 FPS)
        S-->>C: Binary: Protocol V2 (36 bytes × node_count)
    end

    C->>S: JSON: {"type":"filter_update","data":{"quality":0.8}}
    S->>C: JSON: {"type":"filter_update_success"}
    S->>C: Binary: Filtered graph state (V2)

    C->>S: Close Frame
    S->>C: Close Ack
```

---

### Performance Characteristics

#### Bandwidth (100K nodes @ 60 FPS)

| Protocol | Message Size | Parse Time | Latency | vs JSON |
|----------|--------------|------------|---------|---------|
| **Binary V2** | 3.6 MB | 0.8 ms | 10 ms | **80% smaller** |
| Binary V3 | 4.8 MB | 1.1 ms | 13 ms | 73% smaller |
| Binary V4 (delta) | 0.7-1.4 MB | 0.2 ms | 5 ms | 92-96% smaller |
| JSON (deprecated) | 18 MB | 12 ms | 69 ms | Baseline |

#### CPU Usage (100K nodes @ 60 FPS)

| Operation | Binary V2 | JSON V1 |
|-----------|-----------|---------|
| Server Encode | 1.2 ms | 15 ms |
| Client Decode | 0.8 ms | 12 ms |
| Server CPU | 5% | 28% |
| Client CPU | 3% | 18% |

---

## REST HTTP Protocol

### HTTP Methods

| Method | Idempotent | Safe | Use Case |
|--------|------------|------|----------|
| **GET** | ✓ | ✓ | Retrieve resources |
| **POST** | ✗ | ✗ | Create resources, actions |
| **PUT** | ✓ | ✗ | Update/replace resources |
| **PATCH** | ✗ | ✗ | Partial updates |
| **DELETE** | ✓ | ✗ | Remove resources |

### Request Format

```http
POST /api/ontology/load HTTP/1.1
Host: localhost:9090
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Accept: application/json
User-Agent: VisionFlow-Client/1.0

{
  "source": "https://example.org/ontology.owl",
  "format": "rdf/xml",
  "validateImmediately": true
}
```

### Response Format

```http
HTTP/1.1 201 Created
Content-Type: application/json; charset=utf-8
X-Request-ID: req-550e8400-e29b-41d4-a716-446655440000
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1702915200

{
  "success": true,
  "data": {
    "ontologyId": "onto-550e8400-e29b-41d4-a716-446655440000",
    "loadedAt": "2025-12-18T12:00:00Z",
    "axiomCount": 150
  },
  "meta": {
    "requestId": "req-550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-12-18T12:00:00.123Z"
  }
}
```

### Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful GET, PUT, PATCH |
| 201 | Created | Successful POST |
| 202 | Accepted | Async processing started |
| 204 | No Content | Successful DELETE |
| 301 | Moved Permanently | Resource relocated |
| 304 | Not Modified | Cached version still valid |
| 400 | Bad Request | Invalid input |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Resource conflict (e.g., duplicate) |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Content Negotiation

#### Accept Header

```http
GET /api/graph/data HTTP/1.1
Accept: application/json                    # JSON response (default)
Accept: application/ld+json                 # JSON-LD (RDF)
Accept: text/turtle                         # Turtle (RDF)
Accept: application/rdf+xml                 # RDF/XML
Accept: text/csv                            # CSV export
```

#### Compression

```http
GET /api/graph/data HTTP/1.1
Accept-Encoding: gzip, deflate, br          # Brotli preferred
```

**Response**:
```http
HTTP/1.1 200 OK
Content-Encoding: br
Content-Length: 1234567
```

---

## MCP Protocol

### Connection

**Transport**: TCP
**Port**: 9500 (configurable via `MCP_TCP_PORT`)

```bash
# Connect via telnet
telnet localhost 9500

# Connect via netcat
nc localhost 9500
```

### Message Format

**Structure**: JSON-RPC 2.0

```json
{
  "jsonrpc": "2.0",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "method_name",
  "params": {
    "key": "value"
  }
}
```

### Methods

#### swarm_init

Initialize multi-agent swarm.

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "swarm_init",
  "params": {
    "topology": "hierarchical",
    "maxAgents": 10,
    "strategy": "adaptive"
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "swarmId": "swarm-550e8400-e29b-41d4-a716-446655440000",
    "topology": "hierarchical",
    "maxAgents": 10
  }
}
```

#### agent_spawn

Spawn agent in swarm.

```json
{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "agent_spawn",
  "params": {
    "swarmId": "swarm-550e8400-e29b-41d4-a716-446655440000",
    "type": "researcher",
    "capabilities": ["analysis", "synthesis"]
  }
}
```

#### task_orchestrate

Orchestrate task across swarm.

```json
{
  "jsonrpc": "2.0",
  "id": "3",
  "method": "task_orchestrate",
  "params": {
    "swarmId": "swarm-550e8400-e29b-41d4-a716-446655440000",
    "task": "Analyze knowledge graph for communities",
    "strategy": "parallel",
    "priority": "high"
  }
}
```

### Error Responses

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "field": "topology",
      "message": "Must be one of: hierarchical, mesh, ring, star"
    }
  }
}
```

**Standard Error Codes**:

| Code | Meaning |
|------|---------|
| -32700 | Parse error |
| -32600 | Invalid Request |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |

---

## Solid/LDP Protocol

### Overview

VisionFlow integrates with Solid (Social Linked Data) pods via the JSON Solid Server (JSS) sidecar, enabling decentralized data ownership and Linked Data Platform (LDP) compliance.

### Protocol Version

| Version | Status | Transport | Use Case |
|---------|--------|-----------|----------|
| **solid-0.1** | **Current** | WebSocket | Real-time notifications |
| HTTP/LDP | Stable | REST | Resource CRUD operations |

---

### Solid WebSocket Protocol (solid-0.1)

#### Connection Handshake

```
WebSocket Upgrade Request:
GET /solid/ws HTTP/1.1
Host: localhost:3000
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Protocol: solid-0.1
```

#### Wire Format

All messages are JSON-formatted text frames:

```json
{
  "type": "message_type",
  "data": { ... },
  "timestamp": 1702915200000
}
```

#### Message Types

##### Client to Server

**Subscribe to Resource Changes**:
```json
{
  "type": "subscribe",
  "data": {
    "resource": "/pods/user123/graph/",
    "recursive": true
  }
}
```

**Unsubscribe**:
```json
{
  "type": "unsubscribe",
  "data": {
    "resource": "/pods/user123/graph/"
  }
}
```

##### Server to Client

**Subscription Confirmed**:
```json
{
  "type": "subscribed",
  "data": {
    "resource": "/pods/user123/graph/",
    "id": "sub-550e8400-e29b-41d4-a716-446655440000"
  }
}
```

**Resource Notification**:
```json
{
  "type": "notification",
  "data": {
    "resource": "/pods/user123/graph/node-123.ttl",
    "action": "update",
    "actor": "npub1abc...",
    "timestamp": 1702915200000
  }
}
```

**Notification Actions**:
| Action | Description |
|--------|-------------|
| `create` | New resource created |
| `update` | Resource modified |
| `delete` | Resource removed |
| `move` | Resource relocated |

---

### LDP Container Operations

#### Container Types

| Type | Content-Type | Description |
|------|--------------|-------------|
| Basic Container | `text/turtle` | RDF resources |
| Direct Container | `text/turtle` | Member resources |
| Indirect Container | `text/turtle` | Derived membership |

#### Request Headers

```http
GET /pods/user123/graph/ HTTP/1.1
Host: localhost:3000
Accept: text/turtle, application/ld+json
Authorization: DPoP <access_token>
DPoP: <dpop_proof>
```

#### Response Headers

```http
HTTP/1.1 200 OK
Content-Type: text/turtle
Link: <http://www.w3.org/ns/ldp#BasicContainer>; rel="type"
Link: <http://www.w3.org/ns/ldp#Resource>; rel="type"
Accept-Post: text/turtle, application/ld+json
Allow: GET, HEAD, POST, PUT, PATCH, DELETE
```

---

### Data Flow: Neo4j to JSS to Client

```mermaid
sequenceDiagram
    participant N as Neo4j
    participant B as Backend (Rust)
    participant J as JSS (Solid)
    participant C as Client

    Note over N,C: Initial Data Sync
    B->>N: Query graph data (Cypher)
    N-->>B: Return nodes/edges
    B->>J: PUT /pods/{user}/graph/node-{id}.ttl
    J-->>B: 201 Created

    Note over N,C: Real-time Updates
    C->>J: WebSocket: subscribe /pods/{user}/graph/
    J-->>C: subscribed confirmation

    B->>N: Update node
    N-->>B: Confirm update
    B->>J: PATCH /pods/{user}/graph/node-{id}.ttl
    J->>C: WebSocket: notification (update)
    C->>J: GET /pods/{user}/graph/node-{id}.ttl
    J-->>C: Return updated RDF
```

---

## Authentication Protocols

### JWT (JSON Web Token)

**Algorithm**: HS256 (HMAC with SHA-256)

**Token Structure**:
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiJ1c2VyLWlkIiwiZW1haWwiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNzAyOTE1MjAwfQ.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

├── Header (Base64)
│   {"alg":"HS256","type":"JWT"}
├── Payload (Base64)
│   {"sub":"user-id","email":"user@example.com","exp":1702915200}
└── Signature (HMAC-SHA256)
    HMAC-SHA256(base64(header) + "." + base64(payload), secret)
```

**Payload Claims**:

| Claim | Type | Description |
|-------|------|-------------|
| `sub` | string | Subject (user ID) |
| `email` | string | User email |
| `role` | string | User role (admin, editor, viewer) |
| `exp` | integer | Expiration timestamp (Unix) |
| `iat` | integer | Issued at timestamp |
| `iss` | string | Issuer (visionflow) |

**Usage**:
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

### Nostr (NIP-98)

**Protocol**: Nostr Event Signing (kind 27235)

NIP-98 provides HTTP authentication using Nostr event signatures, enabling decentralized identity verification.

**Event Structure**:
```json
{
  "kind": 27235,
  "created_at": 1702915200,
  "tags": [
    ["u", "https://visionflow.example.com/api/auth/nostr"],
    ["method", "POST"],
    ["payload", "sha256_hash_of_body"]
  ],
  "content": "",
  "pubkey": "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d",
  "id": "d42069d...",
  "sig": "908a15e..."
}
```

**Required Tags**:

| Tag | Description | Required |
|-----|-------------|----------|
| `u` | Full URL of the request | Yes |
| `method` | HTTP method (GET, POST, etc.) | Yes |
| `payload` | SHA-256 hash of request body | For POST/PUT |

**Authentication Flow**:

```mermaid
sequenceDiagram
    participant C as Client
    participant S as VisionFlow Server
    participant R as Nostr Relay

    C->>C: Create NIP-98 event (kind 27235)
    C->>C: Sign event with private key
    C->>S: Request with Authorization header
    Note over C,S: Authorization: Nostr base64(event)
    S->>S: Decode and verify event signature
    S->>S: Validate URL, method, timestamp
    S->>R: (Optional) Verify pubkey reputation
    R-->>S: Pubkey metadata
    S-->>C: 200 OK + JWT session token
```

**Client Implementation**:
```typescript
import { generatePrivateKey, getPublicKey, finishEvent } from 'nostr-tools';

const sk = generatePrivateKey();
const pk = getPublicKey(sk);

const authEvent = finishEvent({
  kind: 27235,
  created_at: Math.floor(Date.now() / 1000),
  tags: [
    ['u', window.location.href],
    ['method', 'POST']
  ],
  content: ''
}, sk);

// Base64 encode the event for HTTP header
const authHeader = `Nostr ${btoa(JSON.stringify(authEvent))}`;

const response = await fetch('/api/auth/nostr', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': authHeader
  },
  body: JSON.stringify({ pubkey: pk })
});
```

**Server Validation Requirements**:

1. **Signature Verification**: Validate Schnorr signature against pubkey
2. **Timestamp Check**: Event `created_at` within 60 seconds of current time
3. **URL Match**: Tag `u` must match request URL exactly
4. **Method Match**: Tag `method` must match HTTP method
5. **Payload Hash**: For POST/PUT, verify payload hash if present

**Error Responses**:

| Code | Reason |
|------|--------|
| 401 | Invalid or missing Authorization header |
| 401 | Signature verification failed |
| 401 | Event timestamp expired |
| 403 | Pubkey not authorized |

---

## Protocol Comparison

### WebSocket vs REST

| Aspect | WebSocket | REST |
|--------|-----------|------|
| **Connection** | Persistent | Stateless |
| **Direction** | Bidirectional | Request/Response |
| **Overhead** | Low (no headers after handshake) | High (headers every request) |
| **Real-time** | Excellent | Poor (polling required) |
| **Caching** | Not applicable | HTTP caching |
| **Load Balancing** | Complex (sticky sessions) | Simple |
| **Use Case** | Real-time updates | CRUD operations |

### Binary vs JSON

| Aspect | Binary | JSON |
|--------|--------|------|
| **Size** | 36 bytes/node | 180+ bytes/node |
| **Parse Time** | 0.8 ms (100K nodes) | 12 ms (100K nodes) |
| **Human Readable** | No | Yes |
| **Debugging** | Difficult | Easy |
| **Bandwidth** | **80% less** | Baseline |
| **Use Case** | High-frequency updates | Metadata, control |

### MCP vs REST

| Aspect | MCP | REST |
|--------|-----|------|
| **Transport** | TCP (raw sockets) | HTTP |
| **Format** | JSON-RPC 2.0 | JSON |
| **Overhead** | Very low | Moderate (HTTP headers) |
| **Firewall** | May be blocked | Usually allowed |
| **Use Case** | Agent orchestration | General API |

---

## Cross-Reference Index

### Related Documentation

| Topic | Documentation | Link |
|-------|---------------|------|
| API Reference | REST Endpoints | [API_REFERENCE.md](./API_REFERENCE.md) |
| Binary Protocol | Detailed Specification | [protocols/binary-websocket.md](./protocols/binary-websocket.md) |
| WebSocket API | WebSocket Guide | [api/03-websocket.md](./api/03-websocket.md) |
| Error Codes | Error Reference | [ERROR_REFERENCE.md](./ERROR_REFERENCE.md) |
| Solid Architecture | Sidecar Design | [architecture/solid-sidecar-architecture.md](../architecture/solid-sidecar-architecture.md) |
| Configuration | JSS Settings | [CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md#solid-integration) |

---

**Protocol Reference Version**: 2.0
**VisionFlow Version**: v0.1.0
**Maintainer**: VisionFlow Protocol Team
**Last Updated**: December 18, 2025
