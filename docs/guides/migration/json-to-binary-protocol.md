---
title: Migration Guide: JSON to Binary WebSocket Protocol
description: **Version:** 2.0 **Last Updated:** November 3, 2025 **Status:** Required Migration
type: guide
status: stable
---

# Migration Guide: JSON to Binary WebSocket Protocol

**Version:** 2.0
**Last Updated:** November 3, 2025
**Status:** Required Migration
**Audience:** Developers migrating from legacy JSON WebSocket protocol

---

## Executive Summary

**The JSON WebSocket protocol is DEPRECATED and will be removed in a future release.**

All VisionFlow clients **MUST** migrate to the **Binary V2 protocol** (36-byte format) to continue receiving real-time graph updates. The binary protocol provides:

- **80% bandwidth reduction** (18 MB → 3.6 MB per frame at 100k nodes)
- **15x faster parsing** (12ms → 0.8ms)
- **5x lower network latency** (45ms → <10ms)
- **84% less memory allocation**

---

## Table of Contents

1. [Why Migrate?](#why-migrate)
2. [Protocol Comparison](#protocol-comparison)
3. [Migration Steps](#migration-steps)
4. [Code Examples](#code-examples)
5. 
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Why Migrate?

### Performance Impact

At 100,000 nodes @ 60 FPS:

| Metric | JSON V1 (Deprecated) | Binary V2 (Current) | Improvement |
|--------|----------------------|---------------------|-------------|
| **Message Size** | 18 MB/frame | 3.6 MB/frame | 80% smaller |
| **Parse Time** | 12 ms | 0.8 ms | 15x faster |
| **Network Latency** | 45 ms | <10 ms | 4.5x faster |
| **CPU Usage** | 28% | 5% | 5.6x lower |
| **Memory Allocation** | 22 MB | 3.6 MB | 84% less |

### Timeline

- **Now**: JSON protocol deprecated, binary protocol recommended
- **Q1 2026**: JSON protocol will show deprecation warnings
- **Q2 2026**: JSON protocol support will be removed

---

## Protocol Comparison

### JSON Protocol (Deprecated)

```json
{
  "type": "node-update",
  "data": {
    "nodes": [
      {
        "id": 42,
        "position": {"x": 1.5, "y": 2.3, "z": -0.5},
        "velocity": {"x": 0.1, "y": -0.2, "z": 0.0},
        "mass": 1.0,
        "charge": 0.5
      }
    ]
  }
}
```

**Size**: ~200 bytes per node (with formatting)

### Binary V2 Protocol (Current)

```
Byte Layout (Little-Endian, 36 bytes):
┌──────────┬───────────────────────────────────────────┐
│ Offset   │ Field                                     │
├──────────┼───────────────────────────────────────────┤
│ [0-3]    │ Node ID (u32)                             │
│ [4-7]    │ X position (f32)                          │
│ [8-11]   │ Y position (f32)                          │
│ [12-15]  │ Z position (f32)                          │
│ [16-19]  │ VX velocity (f32)                         │
│ [20-23]  │ VY velocity (f32)                         │
│ [24-27]  │ VZ velocity (f32)                         │
│ [28-31]  │ Mass (f32)                                │
│ [32-35]  │ Charge (f32)                              │
└──────────┴───────────────────────────────────────────┘
```

**Size**: Exactly 36 bytes per node (zero parsing overhead)

---

## Migration Steps

### Step 1: Update WebSocket Configuration

**Before (JSON):**
```typescript
const ws = new WebSocket('ws://localhost:9090/ws?token=JWT-TOKEN&protocol=json');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleMessage(message);
};
```

**After (Binary V2):**
```typescript
const ws = new WebSocket('ws://localhost:9090/ws?token=JWT-TOKEN');
ws.binaryType = 'arraybuffer';  // CRITICAL: Must set to arraybuffer

ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
        handleBinaryMessage(event.data);
    }
};
```

### Step 2: Implement Binary Parser

Create a binary protocol parser class:

```typescript
interface NodeUpdate {
    id: number;
    position: [number, number, number];
    velocity: [number, number, number];
    mass: number;
    charge: number;
}

class BinaryProtocolParser {
    private view: DataView;

    constructor(buffer: ArrayBuffer) {
        this.view = new DataView(buffer);
    }

    parseNodeUpdates(): NodeUpdate[] {
        const nodeCount = this.view.byteLength / 36;
        const updates: NodeUpdate[] = [];

        for (let i = 0; i < nodeCount; i++) {
            const offset = i * 36;

            updates.push({
                id: this.view.getUint32(offset + 0, true),  // Little-endian
                position: [
                    this.view.getFloat32(offset + 4, true),
                    this.view.getFloat32(offset + 8, true),
                    this.view.getFloat32(offset + 12, true),
                ],
                velocity: [
                    this.view.getFloat32(offset + 16, true),
                    this.view.getFloat32(offset + 20, true),
                    this.view.getFloat32(offset + 24, true),
                ],
                mass: this.view.getFloat32(offset + 28, true),
                charge: this.view.getFloat32(offset + 32, true),
            });
        }

        return updates;
    }
}
```

### Step 3: Update Message Handlers

**Before (JSON):**
```typescript
function handleMessage(message: any) {
    if (message.type === 'node-update') {
        message.data.nodes.forEach(node => {
            updateNodePosition(node.id, node.position);
            updateNodeVelocity(node.id, node.velocity);
        });
    }
}
```

**After (Binary V2):**
```typescript
function handleBinaryMessage(data: ArrayBuffer) {
    const parser = new BinaryProtocolParser(data);
    const updates = parser.parseNodeUpdates();

    updates.forEach(node => {
        updateNodePosition(node.id, node.position);
        updateNodeVelocity(node.id, node.velocity);
    });
}
```

### Step 4: Update Data Structures

**Before (JSON):**
```typescript
interface Node {
    id: number;
    position: { x: number; y: number; z: number };
    velocity: { x: number; y: number; z: number };
    mass: number;
    charge: number;
}
```

**After (Binary V2):**
```typescript
interface Node {
    id: number;
    position: [number, number, number];  // Tuple for efficiency
    velocity: [number, number, number];
    mass: number;
    charge: number;
}
```

### Step 5: Remove JSON Serialization

**Remove:**
- All `JSON.parse()` calls for WebSocket messages
- All `JSON.stringify()` calls for WebSocket messages
- Protocol parameter from WebSocket URL (`?protocol=json`)

---

## Code Examples

### Complete Migration Example

```typescript
// ❌ BEFORE: JSON Protocol (Deprecated)
class OldWebSocketService {
    private ws: WebSocket;

    connect(url: string, token: string) {
        this.ws = new WebSocket(`${url}?token=${token}&protocol=json`);

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);

            if (message.type === 'node-update') {
                this.handleNodeUpdate(message.data);
            }
        };
    }

    private handleNodeUpdate(data: any) {
        data.nodes.forEach((node: any) => {
            console.log(`Node ${node.id} at (${node.position.x}, ${node.position.y}, ${node.position.z})`);
        });
    }
}

// ✅ AFTER: Binary V2 Protocol (Required)
class NewWebSocketService {
    private ws: WebSocket;

    connect(url: string, token: string) {
        this.ws = new WebSocket(`${url}?token=${token}`);
        this.ws.binaryType = 'arraybuffer';  // CRITICAL

        this.ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                this.handleBinaryMessage(event.data);
            }
        };
    }

    private handleBinaryMessage(data: ArrayBuffer) {
        const parser = new BinaryProtocolParser(data);
        const updates = parser.parseNodeUpdates();

        updates.forEach((node) => {
            console.log(`Node ${node.id} at (${node.position[0]}, ${node.position[1]}, ${node.position[2]})`);
        });
    }
}
```

### React Three Fiber Integration

```typescript
import { useEffect, useRef } from 'react';
import * as THREE from 'three';

export function usePhysicsStream() {
    const wsRef = useRef<WebSocket | null>(null);
    const nodesRef = useRef<Map<number, THREE.Mesh>>(new Map());

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:9090/ws?token=' + getToken());
        ws.binaryType = 'arraybuffer';

        ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                const parser = new BinaryProtocolParser(event.data);
                const updates = parser.parseNodeUpdates();

                updates.forEach(update => {
                    const mesh = nodesRef.current.get(update.id);
                    if (mesh) {
                        // Direct TypedArray to Three.js Vector3
                        mesh.position.set(...update.position);

                        // Optional: Use velocity for motion blur
                        mesh.userData.velocity = update.velocity;
                    }
                });
            }
        };

        wsRef.current = ws;
        return () => ws.close();
    }, []);

    return { nodes: nodesRef.current };
}
```

---

## Testing & Validation

### Validation Checklist

- [ ] WebSocket connection established without `protocol=json` parameter
- [ ] `ws.binaryType = 'arraybuffer'` is set before opening connection
- [ ] Message handler checks `event.data instanceof ArrayBuffer`
- [ ] Binary parser correctly extracts all 9 fields (ID, 3D position, 3D velocity, mass, charge)
- [ ] Little-endian byte order used in DataView
- [ ] Node positions update correctly in 3D visualization
- [ ] Performance metrics show 80% bandwidth reduction
- [ ] No JSON parsing errors in console

### Testing Tools

```typescript
// Validate binary message format
function validateBinaryMessage(data: ArrayBuffer): boolean {
    // Check message size is multiple of 36 bytes
    if (data.byteLength % 36 !== 0) {
        console.error('Invalid message size:', data.byteLength);
        return false;
    }

    // Parse and validate node data
    const parser = new BinaryProtocolParser(data);
    const updates = parser.parseNodeUpdates();

    updates.forEach((node, index) => {
        // Validate node ID
        if (node.id < 0 || node.id > 4294967295) {
            console.error(`Invalid node ID at index ${index}:`, node.id);
            return false;
        }

        // Validate positions (should be finite numbers)
        if (!isFinite(node.position[0]) || !isFinite(node.position[1]) || !isFinite(node.position[2])) {
            console.error(`Invalid position for node ${node.id}`);
            return false;
        }

        // Validate velocities
        if (!isFinite(node.velocity[0]) || !isFinite(node.velocity[1]) || !isFinite(node.velocity[2])) {
            console.error(`Invalid velocity for node ${node.id}`);
            return false;
        }
    });

    return true;
}
```

### Performance Testing

```typescript
// Measure parsing performance
function benchmarkBinaryParsing(data: ArrayBuffer, iterations: number = 1000) {
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
        const parser = new BinaryProtocolParser(data);
        parser.parseNodeUpdates();
    }

    const elapsed = performance.now() - start;
    const avgTime = elapsed / iterations;

    console.log(`Average parse time: ${avgTime.toFixed(2)}ms`);
    console.log(`Throughput: ${(1000 / avgTime).toFixed(0)} messages/sec`);
}
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Cannot read property 'byteLength' of undefined"

**Cause**: WebSocket `binaryType` not set to `'arraybuffer'`

**Solution**:
```typescript
ws.binaryType = 'arraybuffer';  // Must be set BEFORE connection opens
```

#### Issue 2: Incorrect node positions

**Cause**: Wrong byte order (big-endian vs little-endian)

**Solution**:
```typescript
// CORRECT: Use little-endian (second parameter = true)
this.view.getFloat32(offset, true);

// WRONG: Default is big-endian
this.view.getFloat32(offset);
```

#### Issue 3: "Message size not multiple of 36"

**Cause**: Mixing JSON and binary protocols

**Solution**: Remove `protocol=json` from WebSocket URL and ensure all messages are binary

#### Issue 4: Performance not improving

**Cause**: Still parsing JSON somewhere in the pipeline

**Solution**: Search codebase for `JSON.parse` and remove all WebSocket-related calls

---

## FAQ

### Q: Can I use both JSON and Binary protocols simultaneously?

**A:** No. Choose one protocol per WebSocket connection. We recommend Binary V2 for all new connections.

### Q: What happens if I don't migrate?

**A:** JSON protocol support will be removed in Q2 2026. Your application will stop receiving real-time updates.

### Q: How do I know if migration was successful?

**A:** Check browser DevTools Network tab:
- WebSocket frames should show "Binary Message" (not "Text Message")
- Frame size should be multiples of 36 bytes
- Bandwidth usage should drop by ~80%

### Q: Does this affect REST API calls?

**A:** No. This migration only affects WebSocket real-time updates. REST API continues to use JSON.

### Q: Can I rollback to JSON if there are issues?

**A:** Yes, temporarily. Add `protocol=json` parameter back to WebSocket URL. However, this is not a long-term solution.

### Q: What about mobile clients?

**A:** Binary protocol works on all platforms (iOS, Android, web). Use platform-specific binary parsing (e.g., Swift's `Data`, Kotlin's `ByteBuffer`).

---

## Related Documentation

- **[WebSocket API Reference](../../reference/api/03-websocket.md)** - Complete protocol specification
- **** - Technical details
- **[Performance Benchmarks](../../reference/performance-benchmarks.md)** - Before/after metrics
- **** - System design

---

## Support

**Need help migrating?**

- **Documentation**: 
- **GitHub Issues**: [Report migration issues](https://github.com/yourusername/VisionFlow/issues)
- **Email**: support@visionflow.io (Enterprise customers)

---

**Last Updated**: November 3, 2025
**Migration Deadline**: Q2 2026
**Protocol Version**: Binary V2 (Current), JSON V1 (Deprecated)
