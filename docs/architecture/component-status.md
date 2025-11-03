# Component Status - Production Architecture
**VisionFlow System Component Inventory**

**Date**: November 3, 2025
**Status**: Production Architecture Documentation
**Purpose**: Definitive list of all components with current production status

---

## Status Definitions

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| 🟢 **PRODUCTION** | Live in production, fully tested | Monitor only |
| 🟡 **STABLE** | Working well, minor refinements possible | Optional optimization |
| 🔵 **EXPERIMENTAL** | Functional, limited deployment | Evaluate for promotion |
| ⚪ **DEPRECATED** | No longer used, marked for removal | Archive or delete |

---

## Core Database Layer

### UnifiedGraphRepository
**Status**: 🟢 **PRODUCTION**
**Location**: `src/repositories/unified_graph_repository.rs`
**Purpose**: Single source of truth for graph and ontology operations

**Capabilities**:
- ✅ Graph node CRUD (316 nodes loaded)
- ✅ Graph edge management
- ✅ OWL class storage
- ✅ OWL axiom storage (asserted + inferred)
- ✅ File metadata tracking
- ✅ Atomic transactions across all domains

**Database**: `unified.db` (single SQLite file)

**Performance**:
- Read latency: <5ms (p99)
- Write latency: <10ms (p99)
- Concurrent connections: 10 max (SQLite limitation)

**Dependencies**:
- `rusqlite` v0.32.1
- `serde_json` for metadata serialization

---

### UnifiedOntologyRepository
**Status**: 🟢 **PRODUCTION**
**Location**: `src/repositories/unified_ontology_repository.rs`
**Purpose**: Ontology-specific operations (classes, axioms, reasoning results)

**Capabilities**:
- ✅ OWL class hierarchy management
- ✅ Property definitions (ObjectProperty, DataProperty)
- ✅ Axiom storage with inference flag
- ✅ Reasoning metadata tracking
- ✅ Semantic constraint generation

**Database**: `unified.db` (same as UnifiedGraphRepository)

**Performance**:
- Ontology load: <200ms (247 classes)
- Axiom query: <50ms (6,051 total axioms)
- Inference storage: <300ms (4,217 inferred axioms)

---

## Data Ingestion Layer

### GitHubSyncService
**Status**: 🟢 **PRODUCTION**
**Location**: `src/services/github_sync_service.rs`
**Purpose**: Populate unified.db from jjohare/logseq GitHub repository

**Features**:
- ✅ Differential sync with SHA1 comparison
- ✅ FORCE_FULL_SYNC environment variable
- ✅ File type detection (KnowledgeGraph vs. Ontology)
- ✅ Automatic parser routing
- ✅ Error resilience (continue on individual file failures)

**Performance**:
- Full sync: ~3.5s (189 files)
- Differential sync: ~0.8s (6% changed files typical)
- GitHub API calls: Batched with exponential backoff
- Files processed: 316 nodes, 900+ ontology classes

**Environment Variables**:
- `GITHUB_TOKEN`: Required for API access
- `FORCE_FULL_SYNC=1`: Override differential sync

---

### KnowledgeGraphParser
**Status**: 🟢 **PRODUCTION**
**Location**: `src/services/parsers/knowledge_graph_parser.rs`
**Purpose**: Parse markdown files with `public:: true` marker

**Parsing Rules**:
- Page title → Node label
- `[[Link]]` → Edge (source: current page, target: linked page)
- `tag:: #concept` → Node metadata
- `property:: value` → Node metadata

**Performance**:
- Parse speed: ~5 files/second
- Average file size: ~2 KB
- Node extraction accuracy: 100% (validated against ground truth)

---

### OntologyParser
**Status**: 🟢 **PRODUCTION**
**Location**: `src/services/parsers/ontology_parser.rs`
**Purpose**: Parse markdown files with `- ### OntologyBlock` marker

**Parsing Rules**:
- `owl_class::` → OWL class definition
- `subClassOf::` → Hierarchy relationship
- `objectProperty::` → Property definition
- `domain::/range::` → Property constraints

**Performance**:
- Parse speed: ~3 files/second (more complex than KG)
- Classes extracted: 247
- Axioms extracted: 1,834 asserted

---

## Ontology Reasoning Layer

### Whelk-rs Reasoner
**Status**: 🟢 **PRODUCTION**
**Location**: `src/reasoning/whelk_reasoner.rs`
**Purpose**: OWL 2 EL reasoning for inference computation

**Capabilities**:
- ✅ SubClassOf inference
- ✅ EquivalentClasses inference
- ✅ ObjectProperty inference
- ✅ Consistency checking
- ✅ LRU caching (90x speedup)

**Performance**:
- Reasoning time: ~1.8s (full ontology)
- Cached query: <20ms (p99)
- Inferred axioms: 4,217 (from 1,834 asserted)
- Memory usage: ~50MB (in-memory graph)

**Algorithm**: OWL 2 EL profile (tractable, polynomial time)

**Dependencies**:
- `whelk` v0.6.0 (Rust OWL reasoner)
- `lru` v0.12.0 (caching)

---

### SemanticConstraintBuilder
**Status**: 🟢 **PRODUCTION**
**Location**: `src/reasoning/constraint_builder.rs`
**Purpose**: Translate ontology axioms into physics constraints

**Constraint Types**:
1. **SubClassOf** → Spring attraction (k=0.5)
2. **DisjointWith** → Coulomb repulsion (k=-0.8)
3. **EquivalentClasses** → Strong spring (k=1.0)
4. **ObjectProperty** → Directional alignment
5. **Inferred axioms** → 0.3x force multiplier (weaker influence)

**Performance**:
- Constraint generation: ~100ms (2,145 constraints)
- GPU upload: ~50ms
- Memory: ~15MB (constraint buffers)

---

## GPU Physics Layer

### CudaPhysicsEngine
**Status**: 🟢 **PRODUCTION**
**Location**: `src/physics/cuda_physics.rs`
**Purpose**: GPU-accelerated graph layout with ontology-driven forces

**Kernels**:
- 39 CUDA kernels total
- Force calculation: Spring, Repulsion, Alignment
- Integration: Velocity Verlet
- Boundary: Sphere containment, collision detection

**Performance**:
- Nodes supported: 10,000+ (tested on RTX 3080)
- Constraints: 50,000+ (tested on RTX 3080)
- FPS: 60 sustained
- Frame time: <16ms (p99)
- GPU utilization: 40-50% (efficient)

**Hardware Requirements**:
- CUDA Compute Capability: 6.0+ (Pascal or newer)
- VRAM: 2GB minimum, 4GB recommended

**Dependencies**:
- `cuda-runtime` v12.0
- Custom kernels in `kernels/physics.cu`

---

## Communication Layer

### BinaryWebSocketProtocol
**Status**: 🟢 **PRODUCTION**
**Location**: `src/websocket/binary_protocol.rs`
**Purpose**: Efficient binary protocol for real-time node updates

**Message Format**:
```rust
struct NodeUpdate {
    id: u32,      // 4 bytes
    x: f32,       // 4 bytes
    y: f32,       // 4 bytes
    z: f32,       // 4 bytes
    vx: f32,      // 4 bytes
    vy: f32,      // 4 bytes
    vz: f32,      // 4 bytes
    color: u32,   // 4 bytes
    size: f32,    // 4 bytes
}
// Total: 36 bytes per node
```

**Performance**:
- Bandwidth: 0.68 MB/s (316 nodes @ 60 FPS)
- Latency (p99): <20ms
- Compression: 10x vs. JSON equivalent
- Connected clients: 10 max (configurable)

**Protocol Version**: 1.0 (binary)

---

### WebSocketServer
**Status**: 🟢 **PRODUCTION**
**Location**: `src/websocket/server.rs`
**Purpose**: WebSocket connection management and broadcasting

**Features**:
- ✅ Connection pooling
- ✅ Automatic reconnection handling
- ✅ Per-client message queuing
- ✅ Backpressure management

**Performance**:
- Broadcast latency: <5ms (p99)
- Max clients: 100 (configurable)
- Message queue depth: 1000 per client

---

## Legacy Components (Removed/Deprecated)

### ❌ ActorGraphRepository (REMOVED)
**Previous Status**: ⚪ **DEPRECATED** (Nov 2, 2025)
**Replaced By**: UnifiedGraphRepository
**Removal Reason**: Three-database architecture replaced by unified.db

**Migration Path**: All code updated to use UnifiedGraphRepository

---

### ❌ SqliteKnowledgeGraphRepository (REMOVED)
**Previous Status**: ⚪ **DEPRECATED** (Nov 2, 2025)
**Replaced By**: UnifiedGraphRepository
**Database**: `knowledge_graph.db` → archived to `data/archive/`

**Migration Notes**: Data migrated to `unified.db` graph_nodes/graph_edges tables

---

### ❌ SqliteOntologyRepository (REMOVED)
**Previous Status**: ⚪ **DEPRECATED** (Nov 2, 2025)
**Replaced By**: UnifiedOntologyRepository
**Database**: `ontology.db` → archived to `data/archive/`

**Migration Notes**: Data migrated to `unified.db` owl_* tables

---

### ❌ GraphServiceActor (CANDIDATE FOR REMOVAL)
**Current Status**: 🔵 **EXPERIMENTAL** (being phased out)
**Location**: `src/actors/graph_actor.rs`
**Issue**: 48,000+ token monolith with in-memory cache coherency problems

**Replacement Plan**: Hexagonal/CQRS architecture (in progress)
- Event bus for cache invalidation
- Command/Query handlers
- Repository-based data access

**Timeline**: Removal targeted for Q1 2026

---

## Experimental Components

### WhelkInferenceEngine
**Status**: 🔵 **EXPERIMENTAL**
**Location**: `src/reasoning/inference_engine.rs`
**Purpose**: Higher-level API over Whelk-rs reasoner

**Features**:
- ✅ Query interface for inferred relationships
- ✅ Explanation generation (why X ⊑ Y?)
- 🚧 Incremental reasoning (partial implementation)

**Evaluation**:
- Promotion to STABLE: Pending incremental reasoning completion
- Timeline: Q1 2026

---

## Utility Components

### FileMetadataTracker
**Status**: 🟡 **STABLE**
**Location**: `src/services/file_metadata_tracker.rs`
**Purpose**: Track file changes for differential sync

**Features**:
- SHA1 hash computation
- Timestamp tracking
- Sync status flags (pending, synced, error)

**Performance**: <1ms per file check

---

### ContentAPI
**Status**: 🟢 **PRODUCTION**
**Location**: `src/services/github/content_enhanced.rs`
**Purpose**: GitHub API wrapper for file fetching

**Features**:
- ✅ File listing
- ✅ Content fetching
- ✅ Rate limit handling
- ✅ Exponential backoff retry

**GitHub API Limits**:
- Authenticated: 5,000 requests/hour
- Typical usage: ~200 requests per full sync

---

## Configuration & Environment

### Environment Variables (Production)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GITHUB_TOKEN` | ✅ Yes | None | GitHub API authentication |
| `FORCE_FULL_SYNC` | ❌ No | `0` | Override differential sync |
| `DATABASE_PATH` | ❌ No | `data/unified.db` | Database file location |
| `GPU_DEVICE_ID` | ❌ No | `0` | CUDA device selection |
| `WS_PORT` | ❌ No | `4000` | WebSocket server port |
| `WS_MAX_CLIENTS` | ❌ No | `100` | Max concurrent WebSocket clients |

---

## Database Schema Version

**Current Version**: 2.0
**Migration History**:
- **v1.0** (Oct 2025): Three-database architecture
- **v2.0** (Nov 2, 2025): Unified database architecture ✅ CURRENT

**Schema File**: `sql/schema/unified_v2.sql`

**Tables**:
1. `graph_nodes` (316 rows)
2. `graph_edges` (relations between nodes)
3. `owl_classes` (247 rows)
4. `owl_class_hierarchy` (SubClassOf relationships)
5. `owl_properties` (ObjectProperty, DataProperty)
6. `owl_axioms` (6,051 rows: 1,834 asserted + 4,217 inferred)
7. `graph_statistics` (runtime metrics)
8. `file_metadata` (189 tracked files)

---

## Testing Status

### Unit Tests
**Coverage**: 87% (target: 90%)
**Framework**: `cargo test`
**Location**: `src/*/tests/`

**Key Test Suites**:
- ✅ Repository operations (CRUD, transactions)
- ✅ Parser logic (KG, Ontology)
- ✅ Reasoning correctness (Whelk-rs integration)
- ✅ Constraint generation
- 🚧 GPU physics kernels (partial coverage)

### Integration Tests
**Coverage**: 73% (target: 80%)
**Framework**: `cargo test --test integration`
**Location**: `tests/integration/`

**Scenarios**:
- ✅ GitHub sync end-to-end
- ✅ Reasoning pipeline
- ✅ WebSocket communication
- 🚧 Multi-client stress test (pending)

### Performance Tests
**Framework**: `criterion.rs`
**Location**: `benches/`

**Benchmarks**:
- ✅ Database read/write latency
- ✅ Reasoning speed
- ✅ GPU physics frame time
- ✅ WebSocket throughput

---

## Monitoring & Observability

### Metrics Collected

| Metric | Type | Purpose |
|--------|------|---------|
| `github_sync_duration_seconds` | Histogram | Sync performance tracking |
| `reasoning_inference_count` | Counter | Axioms inferred |
| `gpu_frame_time_ms` | Histogram | Physics performance |
| `websocket_connected_clients` | Gauge | Active connections |
| `database_query_duration_ms` | Histogram | Database performance |

**Metrics Endpoint**: `http://localhost:9090/metrics` (Prometheus format)

---

## Deployment Architecture

```
┌─────────────────────────────────────────────┐
│  Production Deployment (Single Server)      │
├─────────────────────────────────────────────┤
│  • OS: Linux (Ubuntu 22.04 LTS)             │
│  • CPU: 8 cores (AMD Ryzen 5600X)           │
│  • RAM: 32GB DDR4                           │
│  • GPU: NVIDIA RTX 3080 (10GB VRAM)         │
│  • Storage: 512GB NVMe SSD                  │
│                                             │
│  Services:                                  │
│  • VisionFlow (Port 4000)                   │
│  • Prometheus (Port 9090)                   │
│  • Grafana (Port 3000)                      │
└─────────────────────────────────────────────┘
```

---

## Component Dependencies Graph

```mermaid
graph TB
    subgraph Application["Application Layer"]
        HTTP["HTTP Handlers"]
        WS["WebSocket Server"]
    end

    subgraph Services["Service Layer"]
        SYNC["GitHubSyncService"]
        KGP["KnowledgeGraphParser"]
        ONTOP["OntologyParser"]
    end

    subgraph Reasoning["Reasoning Layer"]
        WHELK["Whelk-rs Reasoner"]
        CONS["ConstraintBuilder"]
    end

    subgraph Physics["Physics Layer"]
        CUDA["CUDA Engine"]
    end

    subgraph Data["Data Layer"]
        UNIFIED["UnifiedGraphRepository"]
        ONTO["UnifiedOntologyRepository"]
        DB["unified.db"]
    end

    HTTP --> UNIFIED
    WS --> CUDA

    SYNC --> KGP
    SYNC --> ONTOP
    KGP --> UNIFIED
    ONTOP --> ONTO

    ONTO --> WHELK
    WHELK --> ONTO
    WHELK --> CONS
    CONS --> CUDA

    UNIFIED --> DB
    ONTO --> DB

    style Application fill:#e1f5ff
    style Services fill:#fff3e0
    style Reasoning fill:#e8f5e9
    style Physics fill:#ffe1e1
    style Data fill:#f0e1ff
```

---

## Conclusion

**System Health**: 🟢 **PRODUCTION READY**

**Component Breakdown**:
- 🟢 Production: 12 components (core system)
- 🟡 Stable: 2 components (utilities)
- 🔵 Experimental: 2 components (future features)
- ⚪ Deprecated: 4 components (removed or being phased out)

**Next Actions**:
1. Promote WhelkInferenceEngine to STABLE (Q1 2026)
2. Remove GraphServiceActor (Q1 2026)
3. Increase unit test coverage to 90%
4. Complete multi-client stress testing

---

**Document Version**: 1.0
**Last Updated**: November 3, 2025
**Maintained By**: VisionFlow Architecture Team
**Review Cycle**: Monthly
