# VisionFlow Architecture Overview

**A comprehensive guide to VisionFlow's technical architecture for new developers and technical decision-makers.**

## System Philosophy

VisionFlow is built on three core architectural principles:

1. **Server-Authoritative Architecture** - Neo4j graph database is the single source of truth
2. **Hexagonal Architecture (Ports & Adapters)** - Clean separation between business logic and infrastructure
3. **Modular Actor System** - Specialized actors for graph state, physics, semantic processing, and client coordination

This architecture enables VisionFlow to handle enterprise-scale knowledge graphs (100k+ nodes) while maintaining sub-10ms latency and 60 FPS rendering.

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Layer (React + Three.js)"
        Browser["Web Browser<br/>(Chrome, Edge, Firefox)"]
        ThreeJS["Three.js WebGL Renderer<br/>60 FPS @ 100k+ nodes"]
        WSClient["WebSocket Client<br/>Binary Protocol (36 bytes/node)"]
        VoiceUI["Voice UI (WebRTC)<br/>Spatial Audio"]
        XRUI["XR/VR Interface<br/>(Meta Quest 3)"]
    end

    subgraph "Server Layer (Rust + Actix Web)"
        direction TB

        subgraph "API Layer"
            REST["REST API<br/>(HTTP/JSON)"]
            WebSocket["WebSocket Handler<br/>(Binary Protocol V2)"]
            VoiceWS["Voice WebSocket<br/>(WebRTC Bridge)"]
        end

        subgraph "Hexagonal Core"
            Ports["Ports (Interfaces)<br/>- KnowledgeGraphRepository<br/>- OntologyRepository<br/>- SettingsRepository<br/>- InferenceEngine<br/>- GPUPhysicsAdapter"]
            BusinessLogic["Business Logic<br/>- Graph Operations<br/>- Ontology Reasoning<br/>- Semantic Analysis<br/>- User Management"]
            Adapters["Adapters (Implementations)<br/>- Neo4jGraphRepository<br/>- Neo4jOntologyRepository<br/>- Neo4jSettingsRepository<br/>- WhelkInferenceEngine<br/>- CUDAPhysicsAdapter"]
        end

        subgraph "Actor System (21 Actors - Actix)"
            Supervisor["GraphServiceSupervisor<br/>(Actor lifecycle & supervision)"]
            GSA["GraphStateActor<br/>(Graph state & Neo4j sync)"]
            POA["PhysicsOrchestratorActor<br/>(GPU coordination + 11 GPU actors)"]
            SPA["SemanticProcessorActor<br/>(Ontology reasoning)"]
            CCA["ClientCoordinatorActor<br/>(Multi-client coordination)"]
            Support["+ 6 support actors<br/>(Ontology, Metadata, etc.)"]
        end

        subgraph "Services Layer"
            GitHubSync["GitHub Sync Service<br/>(Streaming ingestion)"]
            RAGFlow["RAGFlow Service<br/>(AI agent orchestration)"]
            SchemaService["Schema Service<br/>(Graph metadata)"]
            NLQuery["Natural Language Query<br/>(LLM-powered)"]
            Pathfinding["Semantic Pathfinding<br/>(Intelligent traversal)"]
        end
    end

    subgraph "Data Layer"
        Neo4j["Neo4j 5.13<br/>Graph Database<br/>- Knowledge Graph (:Node, :Edge)<br/>- Ontology (:OwlClass, :OwlProperty)<br/>- Settings (User preferences)"]
    end

    subgraph "GPU Compute Layer (CUDA 12.4)"
        Physics["Physics Kernels<br/>(Force-directed layout)"]
        Clustering["Clustering Kernels<br/>(Leiden algorithm)"]
        PathfindingGPU["Pathfinding Kernels<br/>(SSSP, BFS)"]
    end

    subgraph "External Integrations"
        GitHub["GitHub API<br/>(Markdown + OWL sync)"]
        Logseq["Logseq<br/>(Local knowledge base)"]
        AIProviders["AI Providers<br/>(Claude, OpenAI, Perplexity)"]
    end

    Browser --> ThreeJS
    Browser --> WSClient
    Browser --> VoiceUI
    Browser --> XRUI

    ThreeJS --> WebSocket
    WSClient --> WebSocket
    VoiceUI --> VoiceWS
    XRUI --> WebSocket

    REST --> Ports
    WebSocket --> Ports
    VoiceWS --> Ports

    Ports <--> BusinessLogic
    BusinessLogic <--> Adapters

    Adapters <--> Neo4j

    Ports <--> Supervisor
    Supervisor --> GSA
    Supervisor --> POA
    Supervisor --> SPA
    Supervisor --> CCA

    GSA <--> Neo4j
    POA <--> Physics
    POA <--> Clustering
    POA <--> PathfindingGPU
    SPA <--> Neo4j

    GitHubSync --> Neo4j
    GitHubSync --> GitHub
    RAGFlow --> AIProviders
    SchemaService --> Neo4j
    NLQuery --> AIProviders
    Pathfinding --> Neo4j

    style Browser fill:#e1f5ff
    style Neo4j fill:#f0e1ff
    style Physics fill:#e1ffe1
    style Hexagonal Core fill:#fff9e1
    style Actor System fill:#ffe1f5
```

## Core Components Explained

### 1. Client Layer (React + Three.js)

**Technology:** React 18, Three.js 0.175, React Three Fiber, TypeScript

**Purpose:** Immersive 3D visualization and user interaction

**Key Features:**
- **60 FPS Rendering** - Maintains performance even with 100k+ nodes using GPU-based force-directed layout
- **WebGL 3D Graphics** - Hardware-accelerated rendering with Three.js
- **Binary Protocol** - Receives graph updates via 36-byte binary WebSocket messages (80% bandwidth reduction)
- **Multi-User Sync** - Independent camera controls with shared graph state
- **Voice Interface** - WebRTC integration for natural language interaction
- **XR Support** - WebXR implementation for Meta Quest 3 with hand tracking

**File Location:** `/client/src/`

**Critical Files:**
- `client/src/app/App.tsx` - Application entry point with authentication
- `client/src/rendering/` - Three.js rendering engine
- `client/src/services/WebSocketService.ts` - Binary protocol client
- `client/src/xr/` - WebXR VR/AR implementation

### 2. Server Layer (Rust + Actix Web)

**Technology:** Rust 1.75+, Actix Web 4.11, Tokio async runtime

**Purpose:** Server-authoritative graph state management and API

#### 2.1 Hexagonal Architecture (Ports & Adapters)

**Philosophy:** Business logic depends on abstractions (ports), not implementations (adapters)

**Ports (Interfaces):**
```rust
// src/ports/knowledge_graph_repository.rs
pub trait KnowledgeGraphRepository {
    async fn add_node(&self, node: Node) -> Result<()>;
    async fn get_graph(&self) -> Result<Arc<GraphData>>;
    // ... 15+ graph operations
}

// src/ports/ontology_repository.rs
pub trait OntologyRepository {
    async fn load_ontology_graph(&self) -> Result<Arc<GraphData>>;
    async fn save_ontology_class(&self, class: OwlClass) -> Result<()>;
    // ... ontology-specific operations
}
```

**Adapters (Implementations):**
- `adapters/neo4j_adapter.rs` - Primary graph database adapter
- `adapters/neo4j_ontology_repository.rs` - Ontology persistence
- `adapters/neo4j_settings_repository.rs` - User settings storage

**Benefits:**
- **Testability** - Mock implementations for unit tests
- **Flexibility** - Swap databases without changing business logic
- **Clarity** - Clear boundaries between domain and infrastructure

#### 2.2 Modular Actor System (Actix)

**Philosophy:** Specialized actors handle distinct concerns with message passing

**Actor Hierarchy:**

```
GraphServiceSupervisor (lifecycle manager)
├── GraphStateActor (in-memory graph state)
├── PhysicsOrchestratorActor (GPU coordination)
├── SemanticProcessorActor (ontology reasoning)
└── ClientCoordinatorActor (multi-client sync)
```

**GraphStateActor** (`src/actors/graph_state_actor.rs`):
- Maintains in-memory graph state (nodes, edges, node_map)
- Handles graph queries (GetGraphData, GetNode, ComputeShortestPaths)
- Delegates persistence to KnowledgeGraphRepository port
- **NOT responsible for physics or semantic processing**

**PhysicsOrchestratorActor** (`src/actors/physics_orchestrator_actor.rs`):
- Coordinates GPU physics simulation
- Manages force-directed layout computation
- Sends position updates via GPUPhysicsAdapter port
- Runs simulation loop (60 Hz by default)

**SemanticProcessorActor** (`src/actors/semantic_processor_actor.rs`):
- Handles ontology reasoning requests
- Coordinates with OntologyRepository and InferenceEngine
- Generates semantic constraints for physics (SubClassOf → attraction, DisjointWith → repulsion)
- Validates graph consistency against ontologies

**ClientCoordinatorActor** (`src/actors/client_coordinator_actor.rs`):
- Manages WebSocket connections for multiple clients
- Broadcasts graph updates using binary protocol
- Handles client-specific filtering and subscriptions
- Coordinates multi-user state synchronization

**Message Passing:**
```rust
// Example: Adding a node triggers multiple actors
let result = graph_state_actor
    .send(AddNode { node })        // 1. GraphStateActor adds to in-memory state
    .await?;

physics_orchestrator_actor
    .send(AddNodeToPhysics { id, position })  // 2. PhysicsOrchestrator updates GPU
    .await?;

semantic_processor_actor
    .send(ValidateNode { node })     // 3. SemanticProcessor validates against ontology
    .await?;

client_coordinator_actor
    .send(BroadcastNodeAdded { node }) // 4. ClientCoordinator sends to all clients
    .await?;
```

#### 2.3 Services Layer

**Purpose:** High-level business logic coordinating multiple components

**GitHubSyncService** (`src/services/github_sync_service.rs`):
- Streams markdown files from GitHub repository
- Parses graph data (knowledge graph nodes/edges) and ontology blocks (OWL)
- Persists to Neo4j via repositories
- Handles pagination (100 files/request) and authentication

**SchemaService** (`src/services/schema_service.rs`):
- Extracts graph schema (node types, edge types, properties)
- Powers natural language query translation
- Provides metadata for UI components

**NaturalLanguageQueryService** (`src/services/natural_language_query_service.rs`):
- Translates English queries to Cypher (Neo4j query language)
- Uses LLM (Perplexity API) with schema-aware prompts
- Validates generated queries for safety
- Returns confidence scores and explanations

**SemanticPathfindingService** (`src/services/semantic_pathfinding_service.rs`):
- Implements intelligent path algorithms beyond shortest path:
  - **Semantic Path** - Shortest path weighted by edge relevance
  - **Query-Guided Traversal** - BFS prioritizing query-matching nodes
  - **Chunk Traversal** - Local neighborhood exploration by similarity

### 3. Data Layer (Neo4j 5.13)

**Technology:** Neo4j 5.13 graph database, Cypher query language, neo4rs async driver

**Purpose:** Single source of truth for all persistent state

**Schema:**

```cypher
// Knowledge Graph
(:Node {
  id: u32,
  label: String,
  metadata_id: String,
  public: String,  // "true" or "false"
  owl_class_iri: String?,  // Links to ontology class
  ... // Additional properties from markdown frontmatter
})

(:Node)-[:EDGE {
  id: u32,
  source_id: u32,
  target_id: u32,
  edge_type: String
}]->(:Node)

// Ontology (OWL)
(:OwlClass {
  iri: String,
  label: String,
  subclass_of: [String],
  disjoint_with: [String],
  equivalent_classes: [String]
})

(:OwlProperty {
  iri: String,
  label: String,
  domain: [String],
  range: [String],
  property_type: String  // "ObjectProperty", "DataProperty", "AnnotationProperty"
})

// User Settings (Nostr Authentication)
(:UserSettings {
  pubkey: String,  // Nostr public key (primary key)
  is_power_user: Boolean,
  created_at: DateTime,
  updated_at: DateTime
})

(:UserSettings)-[:HAS_VISUALIZATION_SETTINGS]->(:VisualizationSettings {
  pubkey: String,
  enable_bloom: Boolean,
  physics_enabled: Boolean,
  node_size: Float,
  ... // 30+ visualization parameters
})
```

**Why Neo4j?**
- **Native graph storage** - Relationships are first-class citizens, not JOIN tables
- **Cypher query language** - Expressive pattern matching for graph traversals
- **ACID transactions** - Enterprise-grade consistency guarantees
- **Scale** - Handles billions of nodes/edges efficiently
- **Ontology support** - Natural fit for OWL class hierarchies

**Data Flow:**
1. GitHub markdown → GitHubSyncService → Neo4jGraphRepository → Neo4j
2. OWL ontologies → OntologyParser → Neo4jOntologyRepository → Neo4j
3. User preferences → SettingsActor → Neo4jSettingsRepository → Neo4j

### 4. GPU Compute Layer (CUDA 12.4)

**Technology:** CUDA 12.4, cudarc (Rust bindings), 39 custom kernels

**Purpose:** 100x speedup for computationally intensive operations

**39 Production CUDA Kernels:**

1. **Physics Simulation (22 kernels)**
   - Force calculation (Barnes-Hut approximation, O(n log n))
   - Velocity integration (Verlet integration, adaptive timestep)
   - Collision detection (spatial hashing)
   - Constraint solving (distance constraints, angle constraints)
   - Semantic forces (ontology-driven attractions/repulsions)

2. **Graph Clustering (8 kernels)**
   - Leiden algorithm (community detection)
   - Modularity optimization
   - Label propagation
   - Hierarchical clustering

3. **Pathfinding (5 kernels)**
   - SSSP (Single-Source Shortest Path)
   - BFS (Breadth-First Search)
   - A* with semantic heuristics
   - Multi-source pathfinding

4. **Utility Kernels (4 kernels)**
   - Memory transfer optimization
   - Graph topology preprocessing
   - Distance matrix computation

**Performance Impact:**

| Operation | CPU (100k nodes) | GPU (100k nodes) | Speedup |
|-----------|------------------|------------------|---------|
| Physics Simulation | 1,600ms | 16ms | 100x |
| Leiden Clustering | 800ms | 12ms | 67x |
| SSSP Pathfinding | 500ms | 8ms | 62x |
| Force-Directed Layout | 2,000ms | 20ms | 100x |

**File Location:** `src/gpu/` (kernel coordination)

**Note:** GPU compute is optional - VisionFlow falls back to CPU for non-CUDA systems (macOS, AMD GPUs)

### 5. External Integrations

**GitHub API:**
- Syncs markdown files with frontmatter (knowledge graph data)
- Parses OWL ontology blocks embedded in markdown
- Handles authentication via GitHub tokens
- Pagination: 100 files per request with tree API

**AI Providers:**
- **Claude (Anthropic)** - Primary AI agent orchestration via MCP protocol
- **Perplexity API** - Natural language query translation (schema-aware prompts)
- **OpenAI (optional)** - Alternative LLM provider for queries

**Logseq (Future):**
- Markdown-based knowledge base integration
- Block-based organization with bidirectional links
- Local-first architecture for data sovereignty

## Data Flow: End-to-End Example

**Scenario:** User syncs GitHub repository containing markdown files with ontology definitions

```mermaid
sequenceDiagram
    participant User
    participant REST API
    participant GitHubSyncService
    participant Neo4jGraphRepository
    participant Neo4j
    participant GraphStateActor
    participant PhysicsOrchestratorActor
    participant GPU
    participant ClientCoordinatorActor
    participant WebSocket
    participant Browser

    User->>REST API: POST /api/admin/sync/streaming
    REST API->>GitHubSyncService: trigger_sync()

    GitHubSyncService->>GitHub: GET /repos/:owner/:repo/git/trees/:sha
    GitHub-->>GitHubSyncService: File tree (paginated)

    loop For each markdown file
        GitHubSyncService->>GitHub: GET /repos/:owner/:repo/contents/:path
        GitHub-->>GitHubSyncService: File content (base64)

        alt Contains OntologyBlock
            GitHubSyncService->>Neo4jOntologyRepository: save_ontology_class()
            Neo4jOntologyRepository->>Neo4j: CREATE (:OwlClass), (:OwlProperty)
        end

        alt Contains public:: true
            GitHubSyncService->>Neo4jGraphRepository: add_node()
            Neo4jGraphRepository->>Neo4j: CREATE (:Node)-[:EDGE]->(:Node)
        end
    end

    GitHubSyncService->>GraphStateActor: ReloadGraphFromDatabase
    GraphStateActor->>Neo4j: MATCH (n:Node)-[e:EDGE]-(m:Node) RETURN *
    Neo4j-->>GraphStateActor: Graph data
    GraphStateActor->>GraphStateActor: Build in-memory state

    GraphStateActor->>PhysicsOrchestratorActor: InitializePhysics { nodes }
    PhysicsOrchestratorActor->>GPU: Transfer node data to GPU memory
    GPU-->>PhysicsOrchestratorActor: GPU buffers allocated

    loop Simulation Loop (60 Hz)
        PhysicsOrchestratorActor->>GPU: Execute force calculation kernels
        GPU-->>PhysicsOrchestratorActor: Updated positions
        PhysicsOrchestratorActor->>ClientCoordinatorActor: BroadcastPositions
        ClientCoordinatorActor->>WebSocket: Binary protocol (36 bytes/node)
        WebSocket->>Browser: WebSocket frame
        Browser->>Browser: Update Three.js scene
    end

    REST API-->>User: 200 OK { synced_nodes: 1234, synced_ontologies: 56 }
```

## Communication Protocols

### Binary WebSocket Protocol V2

**Format:** Fixed 36-byte structure per node update

```rust
struct BinaryNodeData {
    id: u32,           // 4 bytes
    x: f32,            // 4 bytes
    y: f32,            // 4 bytes
    z: f32,            // 4 bytes
    vx: f32,           // 4 bytes (velocity)
    vy: f32,           // 4 bytes
    vz: f32,           // 4 bytes
    group_id: u32,     // 4 bytes (clustering)
    flags: u32,        // 4 bytes (bit flags)
}
// Total: 36 bytes
```

**Benefits:**
- **80% bandwidth reduction** vs JSON (180 bytes/node → 36 bytes/node)
- **Sub-10ms latency** for 10k node updates
- **Zero parsing overhead** - direct memory mapping
- **Multi-user scalable** - broadcast to 50+ clients simultaneously

**Alternatives:**
- **REST API (JSON)** - Initial graph load, queries, admin operations
- **Voice WebSocket** - WebRTC bridge for voice AI interaction

## Key Architectural Decisions

### 1. Why Rust for Backend?

**Decision:** Use Rust instead of Node.js/Python

**Rationale:**
- **Memory safety** - Zero-cost abstractions prevent memory leaks and race conditions
- **Performance** - Native performance comparable to C++, crucial for 100k+ node graphs
- **Concurrency** - Tokio async runtime handles 1000+ concurrent WebSocket connections efficiently
- **Type safety** - Prevents entire classes of bugs at compile time
- **CUDA integration** - Safe bindings to CUDA kernels via cudarc

**Trade-offs:**
- Steeper learning curve for contributors
- Longer compile times (~1m 42s release build)
- Smaller ecosystem compared to JavaScript/Python

### 2. Why Neo4j Instead of PostgreSQL/MongoDB?

**Decision:** Use Neo4j graph database as primary persistence layer

**Rationale:**
- **Native graph storage** - Relationships are first-class citizens, not JOIN tables
- **Cypher queries** - Expressive pattern matching (e.g., `MATCH (a)-[*1..3]-(b)` for variable-length paths)
- **OWL support** - Natural fit for ontology class hierarchies (SubClassOf, DisjointWith)
- **Performance** - Graph traversals are O(edges from node), not O(total edges)
- **Enterprise features** - ACID transactions, clustering, role-based access

**Trade-offs:**
- Operational complexity (requires Neo4j deployment)
- Limited ecosystem compared to PostgreSQL
- License considerations (Enterprise Edition is proprietary)

### 3. Why Binary WebSocket Protocol Instead of JSON?

**Decision:** Custom 36-byte binary protocol over JSON

**Rationale:**
- **Bandwidth** - 80% reduction (180 bytes → 36 bytes per node update)
- **Performance** - Zero-cost parsing (direct memory mapping)
- **Latency** - Sub-10ms for 10k node updates
- **Scalability** - Broadcast to 50+ clients without bandwidth bottleneck

**Trade-offs:**
- More complex client implementation (binary parsing)
- Debugging is harder (can't inspect with browser dev tools)
- Versioning requires careful protocol evolution

### 4. Why Hexagonal Architecture (Ports & Adapters)?

**Decision:** Use hexagonal architecture instead of traditional layered architecture

**Rationale:**
- **Testability** - Business logic depends on interfaces (ports), easily mocked
- **Flexibility** - Swap Neo4j for another database by implementing port
- **Clarity** - Explicit boundaries between domain and infrastructure
- **Independent development** - Teams can work on adapters without touching business logic

**Trade-offs:**
- More upfront design effort (define ports carefully)
- Additional indirection (repository trait + implementation)
- Potential over-engineering for simple CRUD operations

### 5. Why GPU Acceleration for Physics?

**Decision:** Implement 39 CUDA kernels instead of CPU-only physics

**Rationale:**
- **100x speedup** - Physics simulation runs in 16ms instead of 1,600ms
- **Real-time** - Maintains 60 FPS even with 100k+ nodes
- **Scalability** - Handles enterprise-scale graphs without degradation
- **Semantic forces** - GPU enables complex constraint evaluation at scale

**Trade-offs:**
- NVIDIA GPU dependency (no AMD/Intel GPU support yet)
- Operational complexity (CUDA driver installation, GPU memory management)
- Development complexity (CUDA kernel debugging is harder than CPU code)

## Performance Characteristics

### Scalability Limits

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| **Nodes** | 100,000+ | 150,000 tested | 60 FPS maintained with GPU |
| **Concurrent Users** | 50+ | 100+ tested | Binary protocol scales efficiently |
| **WebSocket Latency** | <20ms | <10ms average | Sub-10ms for position updates |
| **Graph Query** | <100ms | 50ms average | Cypher queries with indexing |
| **Ontology Reasoning** | <500ms | 200ms average | Whelk 10-100x faster than Java |
| **GitHub Sync** | 1000 files/min | 1200 files/min | Streaming ingestion, no batching |

### Resource Requirements

**Minimum (Development):**
- CPU: 4 cores, 2.5GHz
- RAM: 8GB
- Storage: 10GB
- GPU: None (CPU fallback)

**Recommended (Production):**
- CPU: 8 cores, 3.0GHz+
- RAM: 16GB
- Storage: 50GB SSD
- GPU: NVIDIA GTX 1060 (6GB VRAM)

**Enterprise (100k+ nodes):**
- CPU: 16+ cores, 3.5GHz
- RAM: 32GB+
- Storage: 200GB+ NVMe SSD
- GPU: NVIDIA RTX 4080+ (16GB+ VRAM)

## Security Architecture

### Authentication

**Nostr Protocol (Decentralized):**
- User authentication via Nostr public/private key pairs
- No centralized identity provider
- Self-sovereign identity (users control their keys)

**JWT Tokens (Optional):**
- Session-based authentication for enterprise deployments
- Stored in Neo4j (:UserSettings nodes)

### Authorization

**Power User System:**
- `is_power_user` flag in Neo4j (:UserSettings)
- Power users can modify ontologies and global settings
- Regular users have read-only access to shared graphs

**Future (v3.0+):**
- Fine-grained RBAC (role-based access control)
- Per-node and per-edge permissions
- SSO integration (SAML, OAuth2)

### Data Security

- **Encryption at Rest** - Neo4j supports transparent data encryption
- **Encryption in Transit** - TLS/SSL for all WebSocket and HTTP connections
- **Secrets Management** - Environment variables, no hardcoded credentials
- **Audit Trail** - Git version control for all graph changes (future)

## Deployment Architecture

### Docker Compose (Development)

```yaml
services:
  neo4j:
    image: neo4j:5.13.0
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data

  backend:
    build: .
    ports: ["4000:4000"]
    depends_on: [neo4j]
    environment:
      DATABASE_URL: bolt://neo4j:7687
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}

  client:
    build: ./client
    ports: ["3001:3001"]
    depends_on: [backend]
```

### Kubernetes (Enterprise)

**Future Roadmap (v3.0+):**
- Helm chart for deployment
- Auto-scaling based on CPU/GPU utilization
- Multi-region data replication
- Redis caching layer for distributed deployments

## Next Steps

**For New Developers:**
1. Read [Developer Journey](DEVELOPER_JOURNEY.md) to understand codebase navigation
2. Review [Technology Choices](TECHNOLOGY_CHOICES.md) for deeper technical rationale
3. Follow [Development Setup](guides/developer/01-development-setup.md) to build locally

**For System Architects:**
1. Review [Hexagonal CQRS Architecture](explanations/architecture/hexagonal-cqrs.md) for design patterns
2. Study [Data Flow Complete](explanations/architecture/data-flow-complete.md) for pipeline details
3. Evaluate [Performance Benchmarks](reference/performance-benchmarks.md) for capacity planning

**For Product Managers:**
1. Understand [What is VisionFlow?](OVERVIEW.md) for value proposition
2. Review [Roadmap](README.md#roadmap) for feature timeline
3. Explore [Use Cases](OVERVIEW.md#real-world-use-cases) for customer scenarios

---

**Last Updated:** 2025-12-02
**Architecture Version:** v2.0.0 (Neo4j Migration Complete)
**Total Lines of Code:** 153,939 lines of Rust
