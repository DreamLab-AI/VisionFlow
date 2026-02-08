# API Reference

REST and WebSocket API reference for the VisionFlow backend server.

Last updated: 2026-02-08

---

## Authentication

All mutation endpoints (POST, PUT, DELETE) require authentication.
Read-only endpoints (GET) are public unless noted.

### Headers

| Header | Required For | Format |
|:-------|:-------------|:-------|
| `Authorization` | All mutations | `Bearer <session-token>` |
| `X-Nostr-Pubkey` | All mutations | Hex-encoded Nostr public key |

### Dev Bypass

Set `SETTINGS_AUTH_BYPASS=true` to skip authentication (development only).
All requests are treated as power user `dev-user`.

### Error Response (401)

```json
{
  "error": "Missing authorization token"
}
```

---

## Endpoint Groups

### WebSocket Endpoints (No Auth -- connection-level)

| Path | Handler | Protocol |
|:-----|:--------|:---------|
| `/wss` | `socket_flow_handler` | JSON + Binary V3 (graph positions, agent state) |
| `/ws/speech` | `speech_socket_handler` | Binary audio (voice) |
| `/ws/mcp-relay` | `mcp_relay_handler` | JSON (MCP protocol relay) |
| `/ws/client-messages` | `client_messages_handler` | JSON (client-to-client messaging) |

### Settings -- `/api/settings/*` (Auth: mutations only)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/settings` | No | Get all settings |
| GET | `/api/settings/:key` | No | Get single setting |
| PUT | `/api/settings/:key` | Yes | Update setting value |
| POST | `/api/settings/bulk` | Yes | Bulk update settings |
| DELETE | `/api/settings/:key` | Yes | Reset setting to default |

Configured in `settings/api/settings_routes.rs`.

### Graph -- `/api/graph/*`

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/graph/data` | No | Get full graph (nodes + edges) |
| GET | `/api/graph/stats` | No | Graph statistics (node/edge counts) |
| GET | `/api/graph/node/:id` | No | Get single node |
| POST | `/api/graph/node` | No | Add node |
| DELETE | `/api/graph/node/:id` | No | Remove node |

Configured in `api_handler/graph/mod.rs`.

### Analytics -- `/api/analytics/*` (Auth: 16 mutation handlers)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/analytics/metrics` | No | Graph metrics (BFS path length, clustering coefficient, centralization, modularity, efficiency) |
| POST | `/api/analytics/pagerank` | Yes | Run PageRank computation |
| POST | `/api/analytics/clustering` | Yes | Run K-means clustering |
| POST | `/api/analytics/community` | Yes | Run Louvain community detection |
| POST | `/api/analytics/anomaly` | Yes | Run LOF anomaly detection |
| POST | `/api/analytics/sssp` | Yes | Run single-source shortest path |
| POST | `/api/analytics/centrality` | Yes | Compute centrality measures |
| GET | `/api/analytics/pathfinding/:source/:target` | No | Find path between nodes |
| POST | `/api/analytics/layout/force` | Yes | Trigger force-directed layout |
| POST | `/api/analytics/layout/stress` | Yes | Trigger stress majorization layout |
| GET | `/api/analytics/clusters` | No | Get cluster assignments |
| GET | `/api/analytics/communities` | No | Get community assignments |
| POST | `/api/analytics/embedding` | Yes | Compute graph embeddings |
| POST | `/api/analytics/similarity` | Yes | Compute node similarity |
| POST | `/api/analytics/filter` | Yes | Apply analytics filter |
| GET | `/api/analytics/summary` | No | Analytics computation summary |

Configured in `api_handler/analytics/mod.rs`.

### Semantic Forces -- `/api/semantic-forces/*` (Auth: 7 mutation handlers)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/semantic-forces/config` | No | Get semantic force configuration |
| POST | `/api/semantic-forces/config` | Yes | Update semantic force config |
| POST | `/api/semantic-forces/compute` | Yes | Trigger semantic force computation |
| GET | `/api/semantic-forces/hierarchy` | No | Get hierarchy levels |
| POST | `/api/semantic-forces/hierarchy/recalculate` | Yes | Recalculate hierarchy |
| POST | `/api/semantic-forces/weights` | Yes | Update force weights |
| POST | `/api/semantic-forces/reset` | Yes | Reset to defaults |

Note: `hierarchy` (GET), `config` (GET), and `hierarchy/recalculate` (POST) return 501 --
these require backend actor messages not yet defined.

Configured in `api_handler/semantic_forces.rs`.

### Ontology Physics -- `/api/ontology-physics/*` (Auth: 3 mutation handlers)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/ontology-physics/constraints` | No | Get active constraints |
| POST | `/api/ontology-physics/constraints` | Yes | Add constraint |
| PUT | `/api/ontology-physics/weights` | Yes | Adjust constraint weights |
| POST | `/api/ontology-physics/reset` | Yes | Reset physics config |

Note: `PUT /weights` returns 501 -- weight adjustment actor message not yet defined.

Configured in `api_handler/ontology_physics/mod.rs`.

### Bots -- `/api/bots/*` (Auth: 4 mutation handlers)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/bots` | No | List registered bots |
| GET | `/api/bots/:id` | No | Get bot details |
| POST | `/api/bots/register` | Yes | Register new bot |
| PUT | `/api/bots/:id` | Yes | Update bot config |
| DELETE | `/api/bots/:id` | Yes | Unregister bot |
| POST | `/api/bots/update` | Yes | Push bot telemetry (triggers agent pipeline) |

Configured in `bots_handler.rs`.

### Constraints -- `/api/constraints/*` (Auth: 4 mutation handlers)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/constraints` | No | List all constraints |
| GET | `/api/constraints/stats` | No | Constraint statistics |
| POST | `/api/constraints` | Yes | Create constraint |
| PUT | `/api/constraints/:id` | Yes | Update constraint |
| DELETE | `/api/constraints/:id` | Yes | Delete constraint |
| POST | `/api/constraints/validate` | Yes | Validate constraint set |

Configured in `constraints_handler.rs`.

### Workspace -- `/api/workspace/*` (Auth: 5 mutation handlers)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/workspace` | No | Get workspace state |
| POST | `/api/workspace/save` | Yes | Save workspace |
| POST | `/api/workspace/load` | Yes | Load workspace |
| POST | `/api/workspace/export` | Yes | Export workspace |
| POST | `/api/workspace/import` | Yes | Import workspace |
| DELETE | `/api/workspace/:id` | Yes | Delete saved workspace |

Configured in `workspace_handler.rs`.

### RAGFlow -- `/api/ragflow/*` (Auth: 3 mutation handlers)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/ragflow/status` | No | RAGFlow integration status |
| POST | `/api/ragflow/query` | Yes | Submit RAG query |
| POST | `/api/ragflow/index` | Yes | Trigger document indexing |
| POST | `/api/ragflow/config` | Yes | Update RAGFlow config |

Configured in `ragflow_handler.rs`.

### Quest3 -- `/api/quest3/*` (Auth: 1 mutation handler)

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/quest3/config` | No | Get Quest 3 XR config |
| POST | `/api/quest3/config` | Yes | Update Quest 3 XR config |
| GET | `/api/quest3/performance` | No | XR performance metrics |

Configured in `api_handler/quest3/mod.rs`.

### Ontology -- `/api/ontology/*`

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/ontology/classes` | No | List OWL classes |
| GET | `/api/ontology/properties` | No | List OWL properties |
| GET | `/api/ontology/axioms` | No | List axioms |
| POST | `/api/ontology/load` | No | Load ontology file |
| POST | `/api/ontology/classify` | No | Run classification |
| GET | `/api/ontology/hierarchy` | No | Get class hierarchy |
| GET | `/api/ontology/individuals` | No | List individuals |

Configured in `ontology_handler.rs`.

### Health & Monitoring

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| GET | `/api/health` | No | Basic health check |
| GET | `/api/health/detailed` | No | Detailed health (DB, GPU, actors) |
| GET | `/api/health/metrics` | No | Prometheus-compatible metrics |

Configured in `consolidated_health_handler.rs`.

### Additional Endpoints

| Method | Path | Auth | Description |
|:-------|:-----|:-----|:------------|
| POST | `/api/client-logs` | No | Client-side error logging |
| GET | `/api/pages/*` | No | Static page serving |
| * | `/api/export/*` | No | Graph export (JSON, CSV, GEXF) |
| * | `/api/share/*` | No | Graph sharing |
| * | `/api/bots-viz/*` | No | Bot visualization data |
| * | `/api/admin/sync/*` | No | Admin sync operations |
| * | `/api/nostr/*` | No | Nostr authentication endpoints |
| * | `/api/solid/*` | No | Solid pod integration |
| GET | `/swagger-ui/*` | No | OpenAPI documentation UI |

---

## Endpoints Returning 501 (Not Implemented)

These endpoints exist in the router but their backend actor messages or adapters
are not yet defined. They return HTTP 501 with a JSON error body.

| Endpoint | Reason |
|:---------|:-------|
| `GET /api/semantic-forces/hierarchy` | Hierarchy level retrieval not yet implemented |
| `GET /api/semantic-forces/config` (detailed) | Semantic config retrieval not yet implemented |
| `POST /api/semantic-forces/hierarchy/recalculate` | Hierarchy recalculation not yet implemented |
| `PUT /api/ontology-physics/weights` | Weight adjustment actor message not defined |

---

## WebSocket Binary Protocol

### Connection

```
ws://host:port/wss
```

No authentication header required at WebSocket level. Authentication is performed
via the initial JSON handshake message after connection.

### Binary Message Header (4 bytes)

```
Byte 0: Message type (u8)
Byte 1: Protocol version (u8)
Bytes 2-3: Payload length (u16, little-endian)
```

For `GRAPH_UPDATE` (0x01), an additional 5th byte contains the graph type flag.

### Position Update Stream (Server -> Client)

Server sends binary frames containing node positions at the configured tick rate
(default: 60 Hz, throttled per client backpressure).

Frame format: `[version_byte] [node_0 ... node_N]`

Where each node is 36 bytes (V2) or 48 bytes (V3).

### Backpressure

Client sends `BROADCAST_ACK` (0x34) to acknowledge received frames:

```
Bytes 0-3: Sequence ID (u32, little-endian)
Bytes 4-7: Nodes received count (u32, little-endian)
```

Server throttles send rate if ACKs fall behind.

### Graph Type Flags (GRAPH_UPDATE 0x01)

The 5th header byte identifies which graph layer this update targets:

| Value | Graph Type |
|:------|:-----------|
| 0x00 | Default / combined |
| 0x01 | Knowledge graph |
| 0x02 | Ontology graph |
| 0x03 | Agent graph |

---

## Error Format

All API errors follow a consistent JSON format:

```json
{
  "error": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": {}
}
```

### Common HTTP Status Codes

| Code | Meaning |
|:-----|:--------|
| 200 | Success |
| 400 | Bad request (validation error) |
| 401 | Unauthorized (missing/invalid auth) |
| 404 | Resource not found |
| 500 | Internal server error |
| 501 | Not implemented (stub endpoint) |

---

## Rate Limiting

WebSocket binary updates are rate-limited to 60 frames/second per client IP,
enforced by `WEBSOCKET_RATE_LIMITER` in `socket_flow_handler.rs`.

REST API endpoints are not rate-limited at the application level. Use a reverse
proxy (nginx) for production rate limiting.
