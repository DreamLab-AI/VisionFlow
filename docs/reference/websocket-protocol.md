---
title: WebSocket Binary Protocol Reference
description: VisionFlow uses a custom binary WebSocket protocol optimized for real-time XR collaboration, semantic graph synchronization, and low-latency node updates. The protocol achieves 36 bytes per node up...
category: reference
tags:
  - api
  - api
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# WebSocket Binary Protocol Reference

## Overview

VisionFlow uses a custom binary WebSocket protocol optimized for real-time XR collaboration, semantic graph synchronization, and low-latency node updates. The protocol achieves 36 bytes per node update at 90 Hz, enabling smooth multi-user immersive experiences.

## Connection Management

### Handshake

**Client → Server**:
```
MESSAGE-TYPE: 0x00 (HELLO)
PROTOCOL-VERSION: u32 (current: 1)
CLIENT-ID: UUID (128 bits)
CAPABILITIES: u32 (bitmask)
  bit 0: hand-tracking
  bit 1: eye-tracking
  bit 2: voice-enabled
  bit 3: ar-supported
  bit 4: vr-supported
PLATFORM: u8
  0: WebXR
  1: Meta Quest
  2: Apple Vision Pro
  3: SteamVR
  4: Desktop/Fallback

Total: 1 + 4 + 16 + 4 + 1 = 26 bytes
```

**Server → Client**:
```
MESSAGE-TYPE: 0x01 (WELCOME)
SESSION-ID: UUID (128 bits)
WORLD-ID: UUID (128 bits)
PROTOCOL-VERSION: u32
CAPABILITY-FLAGS: u32 (server capabilities)
TIMESTAMP: u64 (server time in milliseconds)
STATE-SNAPSHOT-SIZE: u32
[STATE-SNAPSHOT] (variable, gzip compressed)

Total: 1 + 16 + 16 + 4 + 4 + 8 + 4 + variable
```

### Connection Keepalive

**Heartbeat (bidirectional, 30-second interval)**:
```
MESSAGE-TYPE: 0x02 (PING)
TIMESTAMP: u64
SEQUENCE: u32

Response:
MESSAGE-TYPE: 0x03 (PONG)
TIMESTAMP: u64
SEQUENCE: u32

Total: 13 bytes
```

## Message Frame Structure

### Header (fixed 8 bytes)

```
┌─────────────────┬──────────────────┬────────────┬─────────────┐
│ Message Type    │ User ID          │ Timestamp  │ Data Length │
│ (1 byte)        │ (4 bytes)        │ (4 bytes)  │ (2 bytes)   │
├─────────────────┼──────────────────┼────────────┼─────────────┤
│ u8              │ u32 (hash)       │ u32 (delta)│ u16         │
├─────────────────┴──────────────────┴────────────┴─────────────┤
│ Payload (variable, up to 512 bytes)                            │
└────────────────────────────────────────────────────────────────┘
```

### Frame Wrapper

```python
class WebSocketFrame:
    def --init--(self):
        self.message-type: u8
        self.user-id: u32        # Hash of UUID for compactness
        self.timestamp: u32      # Delta from last frame (4-byte window)
        self.data-length: u16    # 0-512 bytes
        self.payload: bytes
```

## Message Types

### 0x01-0x0F: Control Messages

| Type | Name | Purpose | Response |
|------|------|---------|----------|
| 0x01 | WELCOME | Server greeting + snapshot | None |
| 0x02 | PING | Connection check | PONG (0x03) |
| 0x03 | PONG | Ping response | None |
| 0x04 | SYNC-REQUEST | Request full sync | SYNC-RESPONSE |
| 0x05 | SYNC-RESPONSE | Full world state | None |

### 0x10-0x1F: Presence & Avatar

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x10 | POSE-UPDATE | User head/hand transforms | 90 Hz |
| 0x11 | AVATAR-STATE | Avatar appearance/status | On change |
| 0x12 | USER-JOIN | New user entered space | On event |
| 0x13 | USER-LEAVE | User left space | On event |
| 0x14 | VOICE-DATA | Audio stream | ~50 Hz (16kHz mono) |

### 0x20-0x2F: Interaction

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x20 | GESTURE-EVENT | Hand gesture recognized | On gesture |
| 0x21 | VOICE-COMMAND | Voice command | On speech |
| 0x22 | OBJECT-SELECT | Object interaction | On action |
| 0x23 | OBJECT-GRAB | Object grabbed | On action |
| 0x24 | OBJECT-RELEASE | Object released | On action |

### 0x30-0x3F: Graph Updates

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x30 | NODE-CREATE | New ontology node | On creation |
| 0x31 | NODE-UPDATE | Update node properties | On change |
| 0x32 | NODE-DELETE | Remove node | On deletion |
| 0x33 | EDGE-CREATE | New relationship | On creation |
| 0x34 | EDGE-DELETE | Remove relationship | On deletion |
| 0x35 | CONSTRAINT-APPLY | Physics constraint | On change |

### 0x40-0x4F: Agent Actions

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x40 | AGENT-ACTION | Agent-initiated action | On action |
| 0x41 | AGENT-RESPONSE | Agent response data | On response |
| 0x42 | AGENT-STATUS | Agent status update | 1 Hz |

### 0x50-0x5F: Errors & Acknowledgments

| Type | Name | Purpose | Frequency |
|------|------|---------|-----------|
| 0x50 | ERROR | Error notification | On error |
| 0x51 | ACK | Message acknowledgment | On receipt |
| 0x52 | NACK | Negative acknowledgment | On reject |

## Payload Specifications

### POSE-UPDATE (0x10) - 36 bytes

Optimized transform update for user pose (head + hands):

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Position X   │ Position Y   │ Position Z   │ Rotation X   │
│ float16 (2)  │ float16 (2)  │ float16 (2)  │ float16 (2)  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Rotation Y   │ Rotation Z   │ Rotation W   │ Hand State   │
│ float16 (2)  │ float16 (2)  │ float16 (2)  │ u16 (2)      │
├──────────────┴──────────────┴──────────────┴──────────────┤
│ Velocity (velocity estimation for smooth interpolation)   │
│ float16 x3 (6 bytes) = [vx, vy, vz]                       │
├─────────────────────────────────────────────────────────────┤
│ Hand State: 16-bit packed                                  │
│  Left Hand: 4 bits (open, pinch, point, fist)             │
│  Right Hand: 4 bits (open, pinch, point, fist)            │
│  Head Rotation Confidence: 4 bits (0-15)                  │
│  Tracking State: 4 bits (calibrated, tracking, lost, etc)│
└─────────────────────────────────────────────────────────────┘

Total: 8 + 8 + 12 + 2 + 6 = 36 bytes (efficient!)
```

**Typescript Example**:
```typescript
class PoseUpdate {
  position: Vector3;      // XYZ coords (float16 each = 6 bytes)
  rotation: Quaternion;   // XYZW (float16 each = 8 bytes)
  handState: u16;         // Packed hand gesture + tracking state
  velocity: Vector3;      // Movement direction (float16 each = 6 bytes)

  serialize(): Buffer {
    const buf = Buffer.alloc(36);
    // Compress position to float16
    buf.writeUInt16LE(this.position.x, 0);
    buf.writeUInt16LE(this.position.y, 2);
    buf.writeUInt16LE(this.position.z, 4);
    // ... etc for rotation and velocity
    return buf;
  }
}
```

### NODE-UPDATE (0x31) - Variable

Update ontology node in shared graph:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Node ID      │ Property ID  │ Value Type   │ Value Data   │
│ UUID (16)    │ u16          │ u8           │ variable     │
├──────────────┴──────────────┴──────────────┴──────────────┤
│ Value Data Types:                                         │
│  0x00: null (0 bytes)                                     │
│  0x01: boolean (1 byte)                                   │
│  0x02: u32 (4 bytes)                                      │
│  0x03: f32 (4 bytes)                                      │
│  0x04: string (1 + length bytes)                          │
│  0x05: vector3 (12 bytes)                                 │
│  0x06: uri (variable)                                     │
└────────────────────────────────────────────────────────────┘

Minimum: 16 + 2 + 1 = 19 bytes
```

### VOICE-DATA (0x14) - 160 bytes per frame

Opus-encoded audio at 16kHz mono:

```
┌──────────────┬──────────────┬──────────────────────────────┐
│ Sequence     │ Frame Type   │ Opus Payload                  │
│ u16          │ u8           │ (variable, ~160 bytes)        │
├──────────────┼──────────────┼──────────────────────────────┤
│ Frame Types: │              │                              │
│ 0: speech    │ 1: noise     │ 2: silence                   │
│ 3: end-frame │              │                              │
└──────────────┴──────────────┴──────────────────────────────┘

Total: ~160 bytes at 50 fps (20 ms frames) = 8 KB/s per user
```

### ERROR (0x50) - Variable

Error reporting:

```
┌─────────────┬──────────────┬──────────────┬──────────────┐
│ Error Code  │ Severity     │ Message Len  │ Message      │
│ u16         │ u8 (0-3)     │ u8           │ ASCII string │
├─────────────┼──────────────┼──────────────┴──────────────┤
│ Severity Codes:          │                              │
│ 0: Info    1: Warning  2: Error  3: Fatal             │
└─────────────────────────────────────────────────────────┘

Error codes reference: [Error Codes Reference](./error-codes.md)
```

## Compression & Delta Encoding

### Transform Delta Encoding

```typescript
// Only send changed fields
class DeltaPose {
  flags: u8;  // Bitmask of which fields changed
  // bit 0: position changed
  // bit 1: rotation changed
  // bit 2: velocity changed
  // bits 3-7: reserved

  payload: Buffer;  // Only includes changed fields

  // If position changed: 6 bytes (3x float16)
  // If rotation changed: 8 bytes (4x float16)
  // If velocity changed: 6 bytes (3x float16)

  // Example: position + rotation = 1 + 6 + 8 = 15 bytes
  // vs full update = 36 bytes (58% reduction!)
}
```

### Graph Delta Compression

Uses gzip for graph updates:

```typescript
// On server
const graphDelta = computeChanges(previousState, currentState);
const compressed = gzip(serialize(graphDelta));

// Threshold: send full state if delta > 80% of full size
if (compressed.length > fullState.length * 0.8) {
  sendFullState();
} else {
  sendDeltaUpdate(compressed);
}
```

## Bandwidth Estimation

### Per-User Bandwidth

| Content | Message Type | Frequency | Bandwidth |
|---------|--------------|-----------|-----------|
| **Pose** | POSE-UPDATE | 90 Hz | 36 bytes × 90 = 3.24 MB/s |
| **Voice** | VOICE-DATA | 50 Hz (20ms frames) | ~160 bytes × 50 = 8 KB/s |
| **Gestures** | GESTURE-EVENT | ~5-10 per sec | ~50 bytes × 10 = 500 B/s |
| **Graph** | NODE-UPDATE | Variable | ~1-10 KB/s |
| **Overhead** | Headers + keepalive | Constant | ~1 KB/s |
| **TOTAL** | - | - | **~13-15 KB/s per user** |

### Scaling Example

- **10 concurrent users**: 130-150 KB/s (1 Mb/s bandwidth)
- **100 concurrent users**: 1.3-1.5 MB/s (10 Mb/s bandwidth)
- **1000 concurrent users**: 13-15 MB/s (100 Mb/s bandwidth)

## Error Handling

### Automatic Reconnection

```typescript
class WebSocketClient {
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000; // 1 second

  onDisconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      setTimeout(() => {
        this.connect();
        this.reconnectAttempts++;
      }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
    } else {
      this.showFatalError("Unable to reconnect to server");
    }
  }
}
```

### Message Validation

```typescript
// All incoming messages validate:
function validateMessage(frame: WebSocketFrame): boolean {
  // 1. Check message type is valid
  if (frame.messageType > 0x5F) return false;

  // 2. Check payload size
  if (frame.dataLength > 512) return false;

  // 3. Check timestamp is reasonable (max 30 second skew)
  const timeDelta = Math.abs(Date.now() - frame.timestamp);
  if (timeDelta > 30000) return false;

  // 4. Check sequence numbers (prevent duplicates/reordering)
  if (frame.sequence <= this.lastSequence) return false;

  return true;
}
```

## Conflict Resolution

### Last-Write-Wins (LWW)

For concurrent edits:

```typescript
// Both users edit node simultaneously
User1: NODE-UPDATE { id: 'node-1', value: 100, timestamp: 1000 }
User2: NODE-UPDATE { id: 'node-1', value: 200, timestamp: 1001 }

// Server resolves with later timestamp
Result: value = 200 (User2 wins)

// User1 receives NACK + corrected value
Server: NACK { reason: "concurrent-edit", correctValue: 200 }
```

### CRDT-Based Conflict Resolution (Optional)

For concurrent graph modifications:

```typescript
// Conflict-free replicated data type strategy
// Each user has unique ID prefix
User1-ID: "user-a"
User2-ID: "user-b"

// Create operations get unique identifiers
Operation: { id: "user-a-1000", timestamp: 1000, ... }
Operation: { id: "user-b-1000", timestamp: 1000, ... }

// Server merges using CRDT rules (commutative, idempotent)
// Both operations can be applied in any order with same result
```

## Security Considerations

### Message Validation

- All messages validated against schema
- Payload sizes capped at 512 bytes
- User IDs verified against authentication context

### Data Encryption

- All traffic uses WSS (WebSocket Secure = TLS)
- Sensitive data (voice, positioning) encrypted end-to-end
- Eye gaze data encrypted per-frame

### Rate Limiting

```typescript
// Per-user rate limiting
const LIMITS = {
  POSE-UPDATE: 100,    // max per second
  NODE-UPDATE: 10,
  GESTURE-EVENT: 20,
  VOICE-DATA: 60
};

function checkRateLimit(userId: string, msgType: u8): boolean {
  const key = `${userId}:${msgType}`;
  const count = this.rateLimitMap.get(key) || 0;

  if (count >= LIMITS[msgType]) {
    return false; // Drop message
  }

  this.rateLimitMap.set(key, count + 1);
  return true;
}
```

---

---

## Related Documentation

- [Database Schema Reference](database/README.md)
- [VisionFlow Binary WebSocket Protocol](protocols/binary-websocket.md)
- [Pathfinding API Examples](api/pathfinding-examples.md)
- [Error Reference and Troubleshooting](error-codes.md)
- [Complete API Reference](api/README.md)

## Performance Tuning

### Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Max Connections** | 1000 | Per server instance |
| **Pose Update Rate** | 90 Hz | Match HMD refresh rate |
| **Heartbeat Interval** | 30 sec | Keep-alive |
| **Max Message Size** | 512 B | Prevents flooding |
| **Compression** | gzip | For graph updates |
| **Voice Codec** | Opus 16kHz | High quality, low latency |
| **Buffer Size** | 64 KB | Per-connection |
