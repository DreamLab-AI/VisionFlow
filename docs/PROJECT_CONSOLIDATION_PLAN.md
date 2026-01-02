---
layout: default
title: Consolidation Project Plan
description: Plan to unify VisionFlow architecture around JSON-LD
nav_exclude: true
---

# VisionFlow Consolidation Project Plan

## Executive Summary

This document presents a comprehensive plan to unify VisionFlow's architecture around **JSON-LD as the primary data format**, addressing 8 critical architectural tensions identified through deep code analysis of 1,006+ markdown files, 93 source files, and the complete JSS integration.

**Core Thesis**: JSS is JSON-LD native. The current pipeline works against this by serializing to Turtle unnecessarily. By making JSON-LD the universal interchange format, we eliminate 4 transformation layers, fix the source-of-truth crisis, and enable true decentralized data ownership.

---

## Table of Contents

1. [Architectural Tensions Summary](#1-architectural-tensions-summary)
2. [Phase 0: Critical Fixes (Must-Have)](#2-phase-0-critical-fixes)
3. [Phase 1: JSON-LD Unification](#3-phase-1-json-ld-unification)
4. [Phase 2: Source of Truth Consolidation](#4-phase-2-source-of-truth-consolidation)
5. [Phase 3: Protocol Consolidation](#5-phase-3-protocol-consolidation)
6. [Phase 4: Auth Identity Forwarding](#6-phase-4-auth-identity-forwarding)
7. [Phase 5: GPU Backpressure Implementation](#7-phase-5-gpu-backpressure)
8. [Phase 6: Ontology-Code Decoupling](#8-phase-6-ontology-code-decoupling)
9. [Phase 7: God Actor Decomposition](#9-phase-7-god-actor-decomposition)
10. [Implementation Matrix](#10-implementation-matrix)
11. [Risk Assessment](#11-risk-assessment)

---

## 1. Architectural Tensions Summary

| # | Tension | Severity | Impact | Effort |
|---|---------|----------|--------|--------|
| 1 | Source of Truth Crisis | Critical | Data loss on sync | 48h |
| 2 | Protocol Fragmentation (V1-V4) | High | Silent ID collisions | 40h |
| 3 | Solid/Nostr Auth Hairpin | High | ACLs bypassed | 24h |
| 4 | GPU-to-Network Backpressure | High | Memory exhaustion | 32h |
| 5 | Ontology-to-Code Coupling | High | CUDA recompile on change | 24h |
| 6 | God Actor (GPUManager) | Medium | Single point of failure | 24h |
| 7 | JSON-LD Conversion Waste | Medium | Unnecessary transforms | 16h |
| 8 | Enrichment Service Bottleneck | Medium | Sync slowdown | 16h |

**Total Estimated Technical Debt**: 224 hours (~6 engineering weeks)

---

## 2. Phase 0: Critical Fixes (Must-Have)

### P0.1: Kill Protocol V1

**Rationale**: V1 truncates node IDs >16383, causing silent collisions. Still in decode path despite "BUGGY" comment.

**Files to Modify**:

| File | Change |
|------|--------|
| `src/utils/binary_protocol.rs:27-32` | Remove V1 constants |
| `src/utils/binary_protocol.rs:266-295` | Remove `to_wire_id_v1()`, `from_wire_id_v1()` |
| `src/utils/binary_protocol.rs:583-708` | Remove `decode_node_data_v1()` |
| `client/src/services/BinaryWebSocketProtocol.ts:139-141` | Remove V1 size constants |
| `client/src/services/BinaryWebSocketProtocol.ts:301-323` | Remove V1 decode path |

**Implementation**:
```rust
// binary_protocol.rs - REMOVE these lines
// const WIRE_V1_AGENT_FLAG: u16 = 0x8000;
// const WIRE_V1_KNOWLEDGE_FLAG: u16 = 0x4000;
// const WIRE_V1_NODE_ID_MASK: u16 = 0x3FFF;
```

**Validation**: Run test suite with graph >20,000 nodes. Verify no ID collisions.

---

### P0.2: Add Explicit Protocol Negotiation

**Files to Modify**:

| File | Change |
|------|--------|
| `src/handlers/socket_flow_handler.rs:611+` | Add `protocol_version` to `connection_established` |
| `client/src/services/BinaryWebSocketProtocol.ts:276` | Use version byte, not size heuristic |

**Implementation**:
```rust
// socket_flow_handler.rs - ADD protocol field
let response = serde_json::json!({
    "type": "connection_established",
    "timestamp": chrono::Utc::now().timestamp_millis(),
    "protocol": {
        "supported": [2, 3, 4],
        "preferred": 3,
        "deprecated": []
    }
});
```

```typescript
// BinaryWebSocketProtocol.ts - REPLACE size-based detection
// OLD: const isV2 = (payload.byteLength % AGENT_POSITION_SIZE_V2) === 0;
// NEW:
const version = payload[0]; // First byte is protocol version
switch (version) {
    case PROTOCOL_V2: return this.decodeV2(payload.slice(1));
    case PROTOCOL_V3: return this.decodeV3(payload.slice(1));
    case PROTOCOL_V4: return this.decodeV4(payload.slice(1));
    default: throw new Error(`Unknown protocol version: ${version}`);
}
```

---

## 3. Phase 1: JSON-LD Unification

### P1.1: Direct JSON-LD Output in JssSyncService

**Current Flow** (wasteful):
```
Rust Structs → Turtle → JSS → JSON-LD (conversion) → Storage
```

**Target Flow**:
```
Rust Structs → JSON-LD → JSS → Storage (zero conversion)
```

**File**: `src/services/jss_sync_service.rs`

**Add Method** (parallel to `resource_to_turtle`):
```rust
fn resource_to_jsonld(&self, resource: &OntologyResource) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "@context": "https://visionflow.io/contexts/ontology.jsonld",
        "@id": resource.iri,
        "@type": match resource.resource_type {
            OntologyResourceType::Class => "owl:Class",
            OntologyResourceType::Property => "owl:ObjectProperty",
            OntologyResourceType::Individual => "owl:NamedIndividual",
        },
        "rdfs:label": resource.label,
        "rdfs:comment": resource.description,
        "rdfs:subClassOf": resource.parent_iris.iter()
            .map(|p| serde_json::json!({"@id": p}))
            .collect::<Vec<_>>(),
        "vf:qualityScore": resource.quality_score,
        "vf:sourceDomain": resource.source_domain,
        "vf:physicsX": resource.x,
        "vf:physicsY": resource.y,
        "vf:physicsZ": resource.z,
    }))
}
```

**Modify** `sync_resource_to_jss()`:
```rust
// OLD
let turtle = self.resource_to_turtle(resource)?;
let response = self.http_client
    .put(&url)
    .header("Content-Type", "text/turtle")
    .body(turtle)
    .send().await?;

// NEW
let jsonld = self.resource_to_jsonld(resource)?;
let response = self.http_client
    .put(&url)
    .header("Content-Type", "application/ld+json")
    .body(serde_json::to_string(&jsonld)?)
    .send().await?;
```

**Impact**:
- Eliminates `resource_to_turtle()` serialization
- Eliminates JSS `turtleToJsonLd()` conversion
- Frontend receives native JSON-LD (zero parsing overhead)
- Crawlers still get Turtle via JSS content negotiation

---

### P1.2: Create Canonical JSON-LD Context

**New File**: `public/contexts/visionflow.jsonld`

```json
{
  "@context": {
    "@vocab": "http://visionflow.io/ontology#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dcterms": "http://purl.org/dc/terms/",

    "ai": "http://narrativegoldmine.com/ai#",
    "bc": "http://narrativegoldmine.com/blockchain#",
    "rb": "http://narrativegoldmine.com/robotics#",
    "mv": "http://narrativegoldmine.com/metaverse#",
    "ngm": "http://narrativegoldmine.com/ngm#",

    "label": "rdfs:label",
    "description": "rdfs:comment",
    "subClassOf": {"@id": "rdfs:subClassOf", "@type": "@id"},
    "domain": {"@id": "rdfs:domain", "@type": "@id"},
    "range": {"@id": "rdfs:range", "@type": "@id"},

    "requires": {"@id": "ngm:requires", "@type": "@id"},
    "enables": {"@id": "ngm:enables", "@type": "@id"},
    "hasPart": {"@id": "ngm:has-part", "@type": "@id"},
    "bridgesTo": {"@id": "ngm:bridges-to", "@type": "@id"},

    "physicsX": {"@id": "vf:x", "@type": "xsd:float"},
    "physicsY": {"@id": "vf:y", "@type": "xsd:float"},
    "physicsZ": {"@id": "vf:z", "@type": "xsd:float"},
    "physicality": "owl:physicality",
    "role": "owl:role",
    "qualityScore": {"@id": "vf:qualityScore", "@type": "xsd:float"},
    "sourceDomain": "vf:sourceDomain",
    "termId": "vf:termId"
  }
}
```

---

### P1.3: Frontend Native JSON-LD Consumption

**File**: `client/src/hooks/useUnifiedOntology.ts`

**Modify** to fetch from JSS LDP endpoints:
```typescript
// Add JSON-LD fetching capability
async function fetchOntologyJsonLd(resourcePath: string): Promise<JsonLdDocument> {
  const response = await fetch(`/solid/${resourcePath}`, {
    headers: {
      'Accept': 'application/ld+json',
      'Authorization': `Bearer ${getSessionToken()}`
    }
  });
  return response.json();
}

// Replace static file loading with LDP fetch
export function useUnifiedOntology() {
  const [ontology, setOntology] = useState<OntologyData | null>(null);

  useEffect(() => {
    // Fetch from JSS instead of static file
    fetchOntologyJsonLd('public/ontology/classes/')
      .then(jsonld => {
        // JSON-LD is already the correct format
        setOntology(jsonldToOntologyData(jsonld));
      });
  }, []);
}
```

---

## 4. Phase 2: Source of Truth Consolidation

### P2.1: Position-Preserving Sync

**Problem**: `KnowledgeGraphParser` generates RANDOM positions on every sync, destroying GPU-computed layouts.

**File**: `src/services/parsers/knowledge_graph_parser.rs:80-90`

**Current** (destructive):
```rust
let mut rng = rand::thread_rng();
let data = BinaryNodeData {
    node_id: id,
    x: rng.gen_range(-100.0..100.0),  // DESTROYS existing position
    y: rng.gen_range(-100.0..100.0),
    z: rng.gen_range(-100.0..100.0),
    // ...
};
```

**Fixed**:
```rust
// Pass existing positions as parameter (None for new nodes)
pub fn parse_with_positions(
    &self,
    content: &str,
    existing_positions: Option<&HashMap<u32, (f32, f32, f32)>>
) -> Result<GraphData> {
    // ...
    let data = BinaryNodeData {
        node_id: id,
        x: existing_positions
            .and_then(|p| p.get(&id))
            .map(|(x, _, _)| *x)
            .unwrap_or_else(|| rng.gen_range(-100.0..100.0)),
        y: existing_positions
            .and_then(|p| p.get(&id))
            .map(|(_, y, _)| *y)
            .unwrap_or_else(|| rng.gen_range(-100.0..100.0)),
        z: existing_positions
            .and_then(|p| p.get(&id))
            .map(|(_, _, z)| *z)
            .unwrap_or_else(|| rng.gen_range(-100.0..100.0)),
        // ...
    };
}
```

**File**: `src/services/github_sync_service.rs:252`

**Modify** sync to preserve positions:
```rust
async fn process_single_file(&self, file: &FileInfo) -> Result<()> {
    // 1. Get existing positions from Neo4j BEFORE parsing
    let existing_positions = self.kg_repo.get_all_positions().await?;

    // 2. Parse with position preservation
    let parsed = self.kg_parser.parse_with_positions(&content, Some(&existing_positions))?;

    // 3. Save (positions preserved for existing nodes, random for new)
    self.kg_repo.save_graph(&parsed).await?;
}
```

---

### P2.2: GPU-to-Neo4j Position Persistence

**Problem**: GPU positions are ephemeral. On restart, all physics work is lost.

**File**: `src/actors/gpu/force_compute_actor.rs:401-405`

**Add** periodic persistence callback:
```rust
impl ForceComputeActor {
    const PERSIST_INTERVAL: u32 = 300; // Every 300 iterations (~5 seconds at 60Hz)

    fn maybe_persist_positions(&mut self) {
        if self.iteration_count % Self::PERSIST_INTERVAL == 0 {
            if let Some(positions) = self.get_node_positions() {
                // Send to persistence actor (non-blocking)
                if let Some(ref persist_addr) = self.persistence_addr {
                    persist_addr.do_send(PersistPositions { positions });
                }
            }
        }
    }
}
```

**New Actor**: `src/actors/gpu/position_persistence_actor.rs`
```rust
pub struct PositionPersistenceActor {
    neo4j_repo: Arc<Neo4jGraphRepository>,
    pending_batch: Vec<(u32, f32, f32, f32)>,
    batch_size: usize,
}

impl Handler<PersistPositions> for PositionPersistenceActor {
    fn handle(&mut self, msg: PersistPositions, ctx: &mut Self::Context) {
        self.pending_batch.extend(msg.positions);

        if self.pending_batch.len() >= self.batch_size {
            let batch = std::mem::take(&mut self.pending_batch);
            let repo = Arc::clone(&self.neo4j_repo);

            ctx.spawn(async move {
                repo.batch_update_positions(&batch).await.ok();
            }.into_actor(self));
        }
    }
}
```

---

### P2.3: Neo4j Position-Only Updates

**File**: `src/adapters/neo4j_adapter.rs`

**Add** method for position-only updates:
```rust
pub async fn batch_update_positions(
    &self,
    positions: &[(u32, f32, f32, f32)]
) -> RepoResult<()> {
    let query = Query::new(
        "UNWIND $positions AS p
         MATCH (n:GraphNode {id: p.id})
         SET n.sim_x = p.x, n.sim_y = p.y, n.sim_z = p.z"
    ).param("positions", positions.iter().map(|(id, x, y, z)| {
        serde_json::json!({"id": id, "x": x, "y": y, "z": z})
    }).collect::<Vec<_>>());

    self.execute(query).await?;
    Ok(())
}
```

**Key**: Store simulation positions in separate properties (`sim_x`, `sim_y`, `sim_z`) so content syncs (which update `x`, `y`, `z` from parser) don't conflict with physics positions.

---

## 5. Phase 3: Protocol Consolidation

### Target State

| Transport | Protocol | Format | Use Case |
|-----------|----------|--------|----------|
| WebSocket | V3 | 48 bytes/node | Real-time physics |
| QUIC | Postcard | Variable | High-throughput analytics |
| HTTP/LDP | JSON-LD | Semantic | Solid integration |

### P3.1: Consolidate to V3 Default

**File**: `src/utils/binary_protocol.rs:364-366`

**Already correct** (V2 is default, upgrade to V3):
```rust
let use_v3 = true; // Always use V3 now
let item_size = WIRE_V3_ITEM_SIZE; // 48 bytes with analytics
let protocol_version = PROTOCOL_V3;
```

### P3.2: Document Protocol Separation

**New File**: `docs/architecture/PROTOCOL_MATRIX.md`

Document that:
- WebSocket uses Binary V3 for real-time
- QUIC uses Postcard for batch analytics
- HTTP uses JSON-LD for Solid/semantic operations

---

## 6. Phase 4: Auth Identity Forwarding

### P4.1: Dual-Header Authentication

**Problem**: Server re-signs NIP-98 with server keys, hiding user identity from JSS.

**File**: `src/handlers/solid_proxy_handler.rs:147-175`

**Solution**: Forward BOTH server auth AND user auth:

```rust
// Current: Server signs, user identity lost
if let Some(keys) = &state.server_keys {
    let token = generate_nip98_token(keys, &config)?;
    proxy_req = proxy_req.header("Authorization", build_auth_header(&token));
}

// Fixed: Dual headers
if let Some(keys) = &state.server_keys {
    // Server identity for proxy authorization
    let server_token = generate_nip98_token(keys, &config)?;
    proxy_req = proxy_req.header("X-Proxy-Authorization", build_auth_header(&server_token));
}
// Pass through user's original NIP-98
if let Some(user_auth) = req.headers().get("Authorization") {
    if let Ok(val) = user_auth.to_str() {
        proxy_req = proxy_req.header("X-User-Authorization", val);
    }
}
```

### P4.2: JSS Extension for Dual Auth

**File**: `JavaScriptSolidServer/src/auth/nostr.js`

**Add** support for `X-User-Authorization`:
```javascript
async function verifyNostrAuth(req) {
    // Check X-Proxy-Authorization for trusted proxy
    const proxyAuth = req.headers['x-proxy-authorization'];
    if (proxyAuth) {
        const proxyPubkey = await verifyNip98(proxyAuth);
        if (!isTrustedProxy(proxyPubkey)) {
            throw new Error('Untrusted proxy');
        }
    }

    // Extract actual user from X-User-Authorization
    const userAuth = req.headers['x-user-authorization'];
    if (userAuth) {
        return await verifyNip98(userAuth); // Return USER identity
    }

    // Fallback to standard Authorization
    return await verifyNip98(req.headers['authorization']);
}
```

**Impact**: JSS ACLs now apply to actual user identity, enabling true per-user data ownership.

---

## 7. Phase 5: GPU Backpressure

### P5.1: Credit-Based Flow Control

**File**: `src/actors/gpu/force_compute_actor.rs`

**Add** credit tracking:
```rust
pub struct ForceComputeActor {
    // ... existing fields
    broadcast_credits: AtomicU32,
    max_credits: u32,
}

impl ForceComputeActor {
    fn new() -> Self {
        Self {
            broadcast_credits: AtomicU32::new(100), // Start with 100 credits
            max_credits: 100,
            // ...
        }
    }

    fn broadcast_if_credits_available(&mut self, positions: Vec<NodeUpdate>) {
        if self.broadcast_credits.load(Ordering::Acquire) > 0 {
            self.broadcast_credits.fetch_sub(1, Ordering::Release);
            if let Some(ref graph_addr) = self.graph_service_addr {
                graph_addr.do_send(UpdateNodePositions {
                    positions,
                    correlation_id: Some(self.iteration_count),
                });
            }
        } else {
            self.skipped_due_to_backpressure += 1;
        }
    }
}
```

### P5.2: Acknowledgment Loop

**File**: `src/actors/messages.rs`

**Add** acknowledgment message:
```rust
#[derive(Message)]
#[rtype(result = "()")]
pub struct PositionBroadcastAck {
    pub correlation_id: u64,
    pub clients_delivered: u32,
}
```

**File**: `src/actors/client_coordinator_actor.rs`

**Add** ack sending after broadcast:
```rust
pub fn broadcast_to_all(&self, data: Vec<u8>, correlation_id: Option<u64>) -> usize {
    let mut broadcast_count = 0;
    for (_, client_state) in &self.clients {
        client_state.addr.do_send(SendToClientBinary(data.clone()));
        broadcast_count += 1;
    }

    // Send acknowledgment back to GPU
    if let (Some(gpu_addr), Some(corr_id)) = (&self.gpu_addr, correlation_id) {
        gpu_addr.do_send(PositionBroadcastAck {
            correlation_id: corr_id,
            clients_delivered: broadcast_count as u32,
        });
    }

    broadcast_count
}
```

**File**: `src/actors/gpu/force_compute_actor.rs`

**Add** handler to replenish credits:
```rust
impl Handler<PositionBroadcastAck> for ForceComputeActor {
    type Result = ();

    fn handle(&mut self, msg: PositionBroadcastAck, _ctx: &mut Self::Context) {
        // Replenish credits (capped at max)
        let current = self.broadcast_credits.load(Ordering::Acquire);
        let new_credits = (current + 1).min(self.max_credits);
        self.broadcast_credits.store(new_credits, Ordering::Release);
    }
}
```

---

## 8. Phase 6: Ontology-Code Decoupling

### P6.1: Semantic Type Registry

**Problem**: `edge_type_to_int()` hard-codes mappings in both Rust and CUDA.

**New File**: `src/services/semantic_type_registry.rs`

```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct SemanticTypeRegistry {
    uri_to_id: HashMap<String, u32>,
    id_to_config: Vec<RelationshipForceConfig>,
    next_id: AtomicU32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RelationshipForceConfig {
    pub strength: f32,
    pub rest_length: f32,
    pub is_directional: bool,
    pub force_type: u32, // 0=spring, 1=orbit, 2=repulsion
}

impl SemanticTypeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            uri_to_id: HashMap::new(),
            id_to_config: Vec::new(),
            next_id: AtomicU32::new(0),
        };

        // Register default types
        registry.register("ngm:requires", RelationshipForceConfig {
            strength: 0.7, rest_length: 80.0, is_directional: true, force_type: 0
        });
        registry.register("ngm:enables", RelationshipForceConfig {
            strength: 0.4, rest_length: 120.0, is_directional: false, force_type: 0
        });
        registry.register("ngm:has-part", RelationshipForceConfig {
            strength: 0.9, rest_length: 40.0, is_directional: true, force_type: 1
        });
        registry.register("ngm:bridges-to", RelationshipForceConfig {
            strength: 0.3, rest_length: 200.0, is_directional: false, force_type: 2
        });

        registry
    }

    pub fn register(&mut self, uri: &str, config: RelationshipForceConfig) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.uri_to_id.insert(uri.to_string(), id);
        self.id_to_config.push(config);
        id
    }

    pub fn get_id(&self, uri: &str) -> Option<u32> {
        self.uri_to_id.get(uri).copied()
    }

    pub fn build_gpu_buffer(&self) -> Vec<RelationshipForceConfig> {
        self.id_to_config.clone()
    }
}
```

### P6.2: GPU Configuration Buffer

**File**: `src/utils/semantic_forces.cu`

**Replace** hard-coded switch with lookup table:

```cpp
// NEW: Configuration buffer in constant memory
struct RelationshipForceConfig {
    float strength;
    float rest_length;
    bool is_directional;
    uint32_t force_type;
};

__constant__ RelationshipForceConfig c_relationship_configs[MAX_RELATIONSHIP_TYPES];

// Host function to update config buffer
extern "C" void set_relationship_configs(
    const RelationshipForceConfig* configs,
    size_t count
) {
    cudaMemcpyToSymbol(
        c_relationship_configs,
        configs,
        count * sizeof(RelationshipForceConfig)
    );
}

// MODIFIED: Use lookup instead of switch
__global__ void apply_ontology_relationship_force(
    float3* positions,
    const uint32_t* edge_types,
    // ...
) {
    uint32_t edge_type = edge_types[idx];

    // Dynamic lookup (no recompilation needed)
    if (edge_type < MAX_RELATIONSHIP_TYPES) {
        RelationshipForceConfig config = c_relationship_configs[edge_type];
        float strength = config.strength;
        float rest_length = config.rest_length;
        bool is_directional = config.is_directional;

        // Apply force based on config.force_type
        switch (config.force_type) {
            case 0: apply_spring_force(strength, rest_length, ...); break;
            case 1: apply_orbit_force(strength, rest_length, ...); break;
            case 2: apply_repulsion_force(strength, ...); break;
        }
    }
}
```

### P6.3: Runtime Ontology Reload

**File**: `src/gpu/semantic_forces.rs`

**Modify** `edge_type_to_int` to use registry:
```rust
impl SemanticForces {
    fn edge_type_to_int(&self, edge_type: &Option<String>) -> i32 {
        edge_type.as_deref()
            .and_then(|uri| self.registry.get_id(uri))
            .map(|id| id as i32)
            .unwrap_or(0) // Unknown types get default behavior
    }

    pub fn reload_ontology(&mut self, ontology: &Ontology) {
        // Rebuild registry from ontology
        self.registry = SemanticTypeRegistry::new();

        for property in ontology.object_properties() {
            let config = self.infer_force_config(&property);
            self.registry.register(&property.iri, config);
        }

        // Upload to GPU
        let buffer = self.registry.build_gpu_buffer();
        unsafe {
            set_relationship_configs(buffer.as_ptr(), buffer.len());
        }
    }
}
```

**Impact**: Ontology changes no longer require CUDA recompilation. Force parameters can be updated at runtime.

---

## 9. Phase 7: God Actor Decomposition

### P7.1: Subsystem Architecture

**Current**: Single `GPUManagerActor` with 18 handlers, 10 children.

**Target**: 4 independent subsystems with direct access.

```
┌─────────────────────────────────────────────────────────────────┐
│                      AppState (Direct Access)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ PhysicsSubsystem │  │AnalyticsSubsystem│  │ GraphSubsystem  │  │
│  │   - ForceCompute │  │   - Clustering   │  │   - ShortestPath│  │
│  │   - StressMajor  │  │   - Anomaly      │  │   - Components  │  │
│  │   - Constraint   │  │   - PageRank     │  │                 │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │            │
│           └────────────────────┴────────────────────┘            │
│                               │                                  │
│                    ┌──────────▼──────────┐                      │
│                    │  SharedGPUContext    │                      │
│                    │  (Arc<Mutex<...>>)   │                      │
│                    └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### P7.2: Direct Actor Access in AppState

**File**: `src/app_state.rs`

**Modify** to expose subsystem actors directly:
```rust
pub struct AppState {
    // Subsystem actors (direct access)
    pub physics: PhysicsSubsystem,
    pub analytics: AnalyticsSubsystem,
    pub graph: GraphSubsystem,

    // Shared context
    pub gpu_context: Arc<Mutex<SharedGPUContext>>,

    // Legacy (deprecate gradually)
    #[deprecated]
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>,
}

pub struct PhysicsSubsystem {
    pub force_compute: Addr<ForceComputeActor>,
    pub stress_major: Addr<StressMajorizationActor>,
    pub constraint: Addr<ConstraintActor>,
}

pub struct AnalyticsSubsystem {
    pub clustering: Addr<ClusteringActor>,
    pub anomaly: Addr<AnomalyDetectionActor>,
    pub pagerank: Addr<PageRankActor>,
}

pub struct GraphSubsystem {
    pub shortest_path: Addr<ShortestPathActor>,
    pub components: Addr<ConnectedComponentsActor>,
}
```

### P7.3: Context Distribution via Event Bus

**New File**: `src/actors/gpu/context_bus.rs`

```rust
use tokio::sync::broadcast;

pub struct GPUContextBus {
    sender: broadcast::Sender<GPUContextReady>,
}

#[derive(Clone)]
pub struct GPUContextReady {
    pub context: Arc<SharedGPUContext>,
}

impl GPUContextBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(16);
        Self { sender }
    }

    pub fn publish(&self, context: Arc<SharedGPUContext>) {
        let _ = self.sender.send(GPUContextReady { context });
    }

    pub fn subscribe(&self) -> broadcast::Receiver<GPUContextReady> {
        self.sender.subscribe()
    }
}
```

Each actor subscribes independently, eliminating the 140-line sequential distribution in `GPUManagerActor`.

---

## 10. Implementation Matrix

### Priority Order

| Phase | Priority | Dependencies | Parallel? | Hours |
|-------|----------|--------------|-----------|-------|
| P0.1 | Critical | None | Yes | 8 |
| P0.2 | Critical | P0.1 | Yes | 8 |
| P1.1 | High | None | Yes | 16 |
| P1.2 | High | P1.1 | No | 4 |
| P1.3 | High | P1.2 | No | 8 |
| P2.1 | High | None | Yes | 12 |
| P2.2 | High | P2.1 | No | 16 |
| P2.3 | High | P2.2 | No | 8 |
| P3.1 | Medium | P0.1, P0.2 | Yes | 4 |
| P4.1 | Medium | None | Yes | 8 |
| P4.2 | Medium | P4.1 | No | 8 |
| P5.1 | Medium | None | Yes | 16 |
| P5.2 | Medium | P5.1 | No | 8 |
| P6.1 | Medium | None | Yes | 8 |
| P6.2 | Medium | P6.1 | No | 8 |
| P6.3 | Medium | P6.2 | No | 4 |
| P7.1 | Low | All | No | 16 |
| P7.2 | Low | P7.1 | No | 8 |
| P7.3 | Low | P7.2 | No | 8 |

### Sprint Allocation (2-week sprints)

**Sprint 1** (P0 + P1 start):
- P0.1: Kill V1 protocol
- P0.2: Protocol negotiation
- P1.1: JSON-LD output in JssSyncService

**Sprint 2** (P1 complete + P2 start):
- P1.2: Canonical context
- P1.3: Frontend JSON-LD consumption
- P2.1: Position-preserving sync

**Sprint 3** (P2 complete + P4):
- P2.2: GPU position persistence
- P2.3: Neo4j position-only updates
- P4.1: Dual-header auth
- P4.2: JSS dual auth support

**Sprint 4** (P5 + P6):
- P5.1: Credit-based flow control
- P5.2: Acknowledgment loop
- P6.1: Semantic type registry
- P6.2: GPU configuration buffer

**Sprint 5** (P6 complete + P7):
- P6.3: Runtime ontology reload
- P7.1: Subsystem architecture
- P7.2: Direct actor access

**Sprint 6** (P7 complete + hardening):
- P7.3: Context event bus
- Integration testing
- Performance validation

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing clients | Medium | High | Feature flag protocol negotiation |
| Performance regression | Low | Medium | Benchmark before/after each phase |
| Data loss during migration | Medium | Critical | Backup Neo4j before P2 changes |
| CUDA compilation issues | Low | Medium | Test on dev GPU before deploy |
| JSS compatibility | Medium | Medium | Test dual-auth with JSS fork first |
| Actor system instability | Low | High | Gradual deprecation of GPUManager |

### Rollback Points

1. **After P0**: Can revert to V2-only protocol
2. **After P1**: Can fall back to Turtle serialization
3. **After P2**: Can disable position persistence
4. **After P4**: Can revert to server-only auth
5. **After P6**: Can restore hard-coded mappings
6. **After P7**: Can re-enable GPUManagerActor routing

---

## Appendix A: Files Modified Summary

### Rust Files (28 changes)
- `src/utils/binary_protocol.rs` (3 changes)
- `src/handlers/socket_flow_handler.rs` (1 change)
- `src/services/jss_sync_service.rs` (2 changes)
- `src/services/parsers/knowledge_graph_parser.rs` (1 change)
- `src/services/github_sync_service.rs` (1 change)
- `src/adapters/neo4j_adapter.rs` (1 change)
- `src/actors/gpu/force_compute_actor.rs` (3 changes)
- `src/actors/messages.rs` (1 change)
- `src/actors/client_coordinator_actor.rs` (1 change)
- `src/handlers/solid_proxy_handler.rs` (1 change)
- `src/gpu/semantic_forces.rs` (2 changes)
- `src/app_state.rs` (1 change)

### TypeScript Files (3 changes)
- `client/src/services/BinaryWebSocketProtocol.ts` (2 changes)
- `client/src/hooks/useUnifiedOntology.ts` (1 change)

### CUDA Files (1 change)
- `src/utils/semantic_forces.cu` (1 change)

### JavaScript Files (1 change)
- `JavaScriptSolidServer/src/auth/nostr.js` (1 change)

### New Files (7 files)
- `public/contexts/visionflow.jsonld`
- `docs/architecture/PROTOCOL_MATRIX.md`
- `src/actors/gpu/position_persistence_actor.rs`
- `src/services/semantic_type_registry.rs`
- `src/actors/gpu/context_bus.rs`

---

## Appendix B: Markdown Data Summary

**Source**: 1,006 files in `data/markdown/`

| Property | Count | Notes |
|----------|-------|-------|
| Total Files | 1,006 | All with OntologyBlock |
| AI Domain (AI-*) | 312 | Artificial Intelligence |
| Metaverse (mv-*, 20xxx) | 456 | Combined metaverse terms |
| Blockchain (BC-*) | 120 | Blockchain/crypto |
| Robotics (RB-*) | 105 | Robotics automation |
| Relationships (requires) | 487 | Dependency edges |
| Relationships (enables) | 312 | Capability edges |
| Relationships (has-part) | 189 | Compositional edges |
| Relationships (bridges-to) | 78 | Cross-domain edges |

**Key Insight**: The OntologyBlock structure is consistent across all files, enabling uniform JSON-LD transformation.

---

**Document Version**: 1.0
**Created**: 2025-12-30
**Author**: VisionFlow Consolidation Analysis Swarm
**Status**: Ready for Implementation
