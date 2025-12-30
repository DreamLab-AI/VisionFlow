# VisionFlow Protocol Matrix

## Transport Protocols

| Transport | Protocol | Format | Use Case |
|-----------|----------|--------|----------|
| WebSocket | Binary V3 | 48 bytes/node | Real-time physics updates |
| QUIC | Postcard | Variable | High-throughput analytics batch |
| HTTP/LDP | JSON-LD | Semantic | Solid integration, crawler access |

## Binary Protocol V3 Specification

- Header: 1 byte (protocol version)
- Per-node payload: 48 bytes
  - node_id: u32 (4 bytes)
  - x, y, z: f32 (12 bytes)
  - vx, vy, vz: f32 (12 bytes)
  - sssp_distance: f32 (4 bytes)
  - sssp_parent: i32 (4 bytes)
  - cluster_id: u32 (4 bytes)
  - anomaly_score: f32 (4 bytes)
  - community_id: u32 (4 bytes)

## Node Type Flags (bits 26-31 of node_id)

| Bit | Flag | Description |
|-----|------|-------------|
| 31 | Agent | Agent node (0x80000000) |
| 30 | Knowledge | Knowledge node (0x40000000) |
| 28 | Property | Ontology property (0x10000000) |
| 27 | Individual | Ontology individual (0x08000000) |
| 26 | Class | Ontology class (0x04000000) |

## Deprecated

- **V1**: REMOVED (truncated IDs >16383, causing collisions)
- **V2**: Supported but not preferred (36 bytes/node, no analytics)

## Content Negotiation

JSS handles format conversion automatically:
- `Accept: application/ld+json` -> Native JSON-LD
- `Accept: text/turtle` -> Converted on demand
- `Accept: text/html` -> React SPA

## Protocol Version Detection

The first byte of every binary message indicates the protocol version:
- `0x01` = V1 (legacy, 34 bytes/node)
- `0x02` = V2 (stable, 36 bytes/node)
- `0x03` = V3 (current, 48 bytes/node)

Clients should gracefully handle all versions for backwards compatibility,
but new connections should negotiate V3 by default.
