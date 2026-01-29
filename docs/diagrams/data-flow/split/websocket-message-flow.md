---
title: WebSocket Message Data Flow
description: Detailed data flow for WebSocket binary protocol messages between server and clients
category: explanation
tags:
  - architecture
  - data-flow
  - websocket
  - binary-protocol
updated-date: 2026-01-29
difficulty-level: intermediate
---

# WebSocket Message Data Flow

This document details the WebSocket communication protocol and message flow between the VisionFlow server and connected clients.

## Overview

VisionFlow uses a binary WebSocket protocol (V4) optimized for high-frequency graph updates at 60Hz. The protocol minimizes bandwidth while maintaining real-time synchronization across all connected clients.

## Connection Lifecycle

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
stateDiagram-v2
    [*] --> Disconnected

    Disconnected --> Connecting: connect()
    Connecting --> Connected: onopen
    Connecting --> Disconnected: onerror/timeout

    Connected --> Connected: onmessage
    Connected --> Reconnecting: onerror/onclose

    Reconnecting --> Connecting: backoff elapsed
    Reconnecting --> Disconnected: max_retries exceeded

    Connected --> Disconnected: disconnect()
```

## Binary Protocol V4 Format

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph TB
    subgraph "Message Header (4 bytes)"
        H1[Version: u8<br/>Value: 4]
        H2[Type: u8<br/>0x01-0x0F]
        H3[Flags: u8<br/>Compression, etc]
        H4[Reserved: u8]
    end

    subgraph "Node Data (28 bytes per node)"
        N1[node_id: u32<br/>4 bytes]
        N2[x: f32<br/>4 bytes]
        N3[y: f32<br/>4 bytes]
        N4[z: f32<br/>4 bytes]
        N5[vx: f32<br/>4 bytes]
        N6[vy: f32<br/>4 bytes]
        N7[vz: f32<br/>4 bytes]
    end

    subgraph "Full Frame"
        HEADER[Header: 4 bytes]
        COUNT[Node Count: u32<br/>4 bytes]
        NODES[Node Data: 28n bytes]
    end

    H1 --> HEADER
    H2 --> HEADER
    H3 --> HEADER
    H4 --> HEADER
    HEADER --> COUNT --> NODES

    style HEADER fill:#e1f5ff
    style COUNT fill:#ffe1e1
    style NODES fill:#e1ffe1
```

## Message Types

| Type ID | Name | Direction | Payload | Frequency |
|---------|------|-----------|---------|-----------|
| 0x01 | FullGraph | Server->Client | All nodes + edges | On connect |
| 0x02 | PositionUpdate | Server->Client | Node positions | 60Hz |
| 0x03 | NodeSelect | Client->Server | Selected node ID | On click |
| 0x04 | NodeDrag | Client->Server | Node ID + position | On drag |
| 0x05 | Heartbeat | Bidirectional | Timestamp | Every 30s |
| 0x06 | FilterUpdate | Client->Server | Filter criteria | On change |
| 0x07 | VoiceData | Bidirectional | PCM audio | Streaming |

## Broadcast Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
sequenceDiagram
    participant PO as PhysicsOrchestrator
    participant CC as ClientCoordinator
    participant CM as ClientManager
    participant WS1 as Client 1
    participant WS2 as Client 2
    participant WSN as Client N

    PO->>CC: UpdateNodePositions(positions)
    CC->>CC: Check throttle (50ms interval)

    alt Should broadcast
        CC->>CC: Serialize to binary (28n bytes)
        CC->>CM: Get active clients

        par Parallel broadcast
            CM->>WS1: Binary frame
            CM->>WS2: Binary frame
            CM->>WSN: Binary frame
        end

        CC->>CC: Update last_broadcast timestamp
    else Throttled
        CC->>CC: Skip (too recent)
    end
```

## Client Filter System

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph TB
    subgraph "Server-Side Filtering"
        FULL[Full Graph<br/>All 316 nodes]
        FILTER[Client Filter<br/>Per-connection]
        FILTERED[Filtered Subset<br/>50-100 nodes]
    end

    subgraph "Filter Criteria"
        TYPE[Node Type<br/>Class/Individual]
        DEPTH[Hierarchy Depth<br/>Level 0-5]
        CLUSTER[Cluster ID<br/>Community]
        SEARCH[Search Query<br/>Label match]
    end

    FULL --> FILTER
    TYPE --> FILTER
    DEPTH --> FILTER
    CLUSTER --> FILTER
    SEARCH --> FILTER
    FILTER --> FILTERED

    style FULL fill:#ffe1e1
    style FILTER fill:#ffe66d
    style FILTERED fill:#e1ffe1
```

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Message size (10K nodes) | 280 KB | 4 + 4 + (28 * 10000) |
| Broadcast interval | 50ms active / 1000ms stable | Adaptive |
| Max concurrent clients | 50,000 | Connection limit |
| Latency P50 | 10ms | Server to client |
| Latency P95 | 30ms | Including processing |
| Bandwidth per client | 5.6 MB/s | At 60Hz, 10K nodes |

## Reconnection Strategy

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph LR
    subgraph "Exponential Backoff"
        R1[Retry 1<br/>500ms]
        R2[Retry 2<br/>1000ms]
        R3[Retry 3<br/>2000ms]
        R4[Retry 4<br/>4000ms]
        R5[Retry 5<br/>8000ms]
        FAIL[Max retries<br/>Disconnect]
    end

    R1 -->|Fail| R2
    R2 -->|Fail| R3
    R3 -->|Fail| R4
    R4 -->|Fail| R5
    R5 -->|Fail| FAIL

    R1 -->|Success| CONNECTED[Connected]
    R2 -->|Success| CONNECTED
    R3 -->|Success| CONNECTED
    R4 -->|Success| CONNECTED
    R5 -->|Success| CONNECTED

    style FAIL fill:#ffe1e1
    style CONNECTED fill:#e1ffe1
```

## Data Transformations

| Stage | Input | Output | Size Change |
|-------|-------|--------|-------------|
| GPU -> Host | Device positions | Vec<BinaryNodeData> | 0% |
| Host -> Serialize | Vec<BinaryNodeData> | [u8] binary | +8 bytes header |
| Serialize -> Compress | [u8] binary | [u8] compressed | -20-40% |
| Compress -> Send | [u8] compressed | WebSocket frame | +2-14 bytes framing |

## Related Documentation

- [Binary Protocol Specification](../../infrastructure/websocket/binary-protocol-complete.md)
- [Client Coordinator Actor](../../server/actors/actor-system-complete.md)
- [Client State Management](../../client/state/state-management-complete.md)
