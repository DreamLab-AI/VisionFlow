---
title: Server Architecture
description: VisionFlow server implements a **hexagonal (ports and adapters) architecture** with **CQRS patterns** and **actor-based concurrency** for handling real-time graph visualization and semantic analysis.
category: explanation
tags:
  - architecture
  - patterns
  - structure
  - api
  - rest
related-docs:
  - explanations/architecture/hexagonal-cqrs.md
  - guides/architecture/actor-system.md
  - explanations/architecture/database-architecture.md
  - guides/graphserviceactor-migration.md
  - ARCHITECTURE_OVERVIEW.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Neo4j database
---

# Server Architecture

**Status**: ✅ **STABLE - HEXAGONAL/CQRS PRODUCTION**
**Last Updated**: December 2, 2025
**Migration**: GraphServiceActor fully removed November 5, 2025

---

## Executive Summary

VisionFlow server implements a **hexagonal (ports and adapters) architecture** with **CQRS patterns** and **actor-based concurrency** for handling real-time graph visualization and semantic analysis. The server is built on:

- **21 Specialized Actors** - Supervised actor hierarchy for concurrent operations
- **9 Port Interfaces** - Technology-agnostic domain boundaries
- **12 Adapters** - Concrete implementations (Neo4j, GPU, Actix)
- **CQRS Layer** - Separate read/write operations for scalability
- **Neo4j Database** - Source of truth for all graph and user data

### What Changed (November 2025)

- ❌ **REMOVED**: Monolithic GraphServiceActor (48,000+ tokens, stale cache issues)
- ✅ **CURRENT**: Modular actor hierarchy supervised by GraphServiceSupervisor
- ✅ **ADDED**: Complete hexagonal architecture with ports/adapters
- ✅ **ADDED**: Neo4j as unified persistence layer

---

## Architecture Overview

```mermaid
graph TB
    subgraph Presentation["🌐 Presentation Layer"]
        HTTP["HTTP API<br/>(Actix-web)"]
        WS["WebSocket Server<br/>(Real-time updates)"]
    end

    subgraph Application["⚡ Application Layer - CQRS"]
        CQRS["Command/Query Bus"]
        CMDH["Command Handlers"]
        QRYH["Query Handlers"]
    end

    subgraph Actors["🎭 Actor System - 21 Actors"]
        SUP["GraphServiceSupervisor"]

        subgraph Core["Core Actors (4)"]
            STATE["GraphStateActor<br/>(State management)"]
            PHYS["PhysicsOrchestratorActor<br/>(GPU physics)"]
            SEM["SemanticProcessorActor<br/>(AI analysis)"]
            CLIENT["ClientCoordinatorActor<br/>(WebSocket)"]
        end

        subgraph GPU["GPU Actors (11)"]
            FORCE["ForceComputeActor"]
            SEMANTIC["SemanticForcesActor"]
            PAGERANK["PageRankActor"]
            SHORTEST["ShortestPathActor"]
            CLUSTER["ClusteringActor"]
            OTHERS["+ 6 more..."]
        end

        subgraph Support["Support Actors (6)"]
            ONT["OntologyActor"]
            META["MetadataActor"]
            WORK["WorkspaceActor"]
            TASK["TaskOrchestratorActor"]
            MONITOR["AgentMonitorActor"]
            SETTINGS["OptimizedSettingsActor"]
        end
    end

    subgraph Domain["🎯 Domain Layer"]
        PORTS["9 Port Interfaces<br/>(Traits)"]
    end

    subgraph Infrastructure["🔧 Infrastructure Layer"]
        ADAPTERS["12 Adapters"]
        NEO4J["Neo4j Database<br/>(Source of truth)"]
    end

    HTTP --> CQRS
    WS --> CLIENT
    CQRS --> CMDH
    CQRS --> QRYH
    CMDH --> SUP
    QRYH --> SUP
    SUP --> Core
    SUP --> GPU
    SUP --> Support
    Core --> PORTS
    GPU --> PORTS
    Support --> PORTS
    PORTS --> ADAPTERS
    ADAPTERS --> NEO4J

    classDef presentation fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef application fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef actors fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef domain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef infra fill:#fce4ec,stroke:#c2185b,stroke-width:2px

    class HTTP,WS presentation
    class CQRS,CMDH,QRYH application
    class SUP,STATE,PHYS,SEM,CLIENT,FORCE,SEMANTIC,PAGERANK,SHORTEST,CLUSTER,OTHERS,ONT,META,WORK,TASK,MONITOR,SETTINGS actors
    class PORTS domain
    class ADAPTERS,NEO4J infra
```

---

## Actor System Hierarchy

The actor system implements a **supervised hierarchy** with fault tolerance and automatic recovery:

### Supervision Structure

```mermaid
graph TB
    ROOT["Actix System Root"]

    SUPER["GraphServiceSupervisor<br/>━━━━━━━━━━━━━<br/>Supervision: OneForOne<br/>Restarts: 3 max per 60s"]

    subgraph Level1["Level 1: Core Service Actors"]
        STATE["GraphStateActor<br/>━━━━━━━━━━━━━<br/>• Neo4j persistence<br/>• State coordination<br/>• Graph operations"]

        PHYS["PhysicsOrchestratorActor<br/>━━━━━━━━━━━━━<br/>• GPU coordination<br/>• Physics simulation<br/>• 11 child actors"]

        SEM["SemanticProcessorActor<br/>━━━━━━━━━━━━━<br/>• AI analysis<br/>• Ontology reasoning<br/>• Semantic constraints"]

        CLIENT["ClientCoordinatorActor<br/>━━━━━━━━━━━━━<br/>• WebSocket management<br/>• Client filtering<br/>• Real-time updates"]
    end

    subgraph Level2["Level 2: GPU Compute Actors"]
        FORCE["ForceComputeActor<br/>(CUDA kernels)"]
        SEMANTIC_F["SemanticForcesActor<br/>(Ontology forces)"]
        STRESS["StressMajorizationActor<br/>(Layout optimization)"]
        PAGERANK["PageRankActor<br/>(Importance)"]
        SHORTEST["ShortestPathActor<br/>(Pathfinding)"]
        CLUSTER["ClusteringActor<br/>(Community detection)"]
        ANOMALY["AnomalyDetectionActor<br/>(Outlier detection)"]
        COMPONENTS["ConnectedComponentsActor"]
        CONSTRAINT["ConstraintActor"]
        ONT_CONSTRAINT["OntologyConstraintActor"]
        GPU_MGR["GPUManagerActor"]
    end

    subgraph Level2B["Level 2: Support Actors"]
        ONT["OntologyActor<br/>(Reasoning)"]
        META["MetadataActor<br/>(Metadata management)"]
        WORK["WorkspaceActor<br/>(Project state)"]
        TASK["TaskOrchestratorActor<br/>(Task coordination)"]
        MONITOR["AgentMonitorActor<br/>(Health monitoring)"]
        SETTINGS["OptimizedSettingsActor<br/>(User settings)"]
    end

    ROOT --> SUPER
    SUPER --> STATE
    SUPER --> PHYS
    SUPER --> SEM
    SUPER --> CLIENT

    PHYS --> FORCE
    PHYS --> SEMANTIC_F
    PHYS --> STRESS
    PHYS --> PAGERANK
    PHYS --> SHORTEST
    PHYS --> CLUSTER
    PHYS --> ANOMALY
    PHYS --> COMPONENTS
    PHYS --> CONSTRAINT
    PHYS --> ONT_CONSTRAINT
    PHYS --> GPU_MGR

    SUPER -.->|spawns| ONT
    SUPER -.->|spawns| META
    SUPER -.->|spawns| WORK
    SUPER -.->|spawns| TASK
    SUPER -.->|spawns| MONITOR
    SUPER -.->|spawns| SETTINGS

    classDef supervisor fill:#ffeb3b,stroke:#f57f17,stroke-width:3px
    classDef coreActor fill:#81c784,stroke:#2e7d32,stroke-width:2px
    classDef gpuActor fill:#64b5f6,stroke:#1565c0,stroke-width:2px
    classDef supportActor fill:#ba68c8,stroke:#6a1b9a,stroke-width:2px

    class SUPER supervisor
    class STATE,PHYS,SEM,CLIENT coreActor
    class FORCE,SEMANTIC_F,STRESS,PAGERANK,SHORTEST,CLUSTER,ANOMALY,COMPONENTS,CONSTRAINT,ONT_CONSTRAINT,GPU_MGR gpuActor
    class ONT,META,WORK,TASK,MONITOR,SETTINGS supportActor
```

### Actor Responsibilities

#### Core Actors (4)

1. **GraphStateActor** - Primary state management
   - Graph data persistence to Neo4j
   - Node/edge CRUD operations
   - State synchronization across clients
   - Filter state management

2. **PhysicsOrchestratorActor** - GPU physics coordination
   - Manages 11 GPU compute actors
   - Coordinates force calculations
   - Layout optimization
   - Physics parameter management

3. **SemanticProcessorActor** - AI and semantic analysis
   - Ontology-based reasoning
   - Semantic constraint generation
   - Community detection coordination
   - NLP processing

4. **ClientCoordinatorActor** - WebSocket client management
   - Client connection lifecycle
   - Per-client filtering
   - Real-time graph updates
   - Subscription management

#### GPU Actors (11)

5. **ForceComputeActor** - Force-directed layout (CUDA)
6. **SemanticForcesActor** - Ontology-driven forces
7. **StressMajorizationActor** - Graph layout optimization
8. **PageRankActor** - Node importance calculation
9. **ShortestPathActor** - Pathfinding algorithms
10. **ClusteringActor** - Community detection (Louvain, etc.)
11. **AnomalyDetectionActor** - Outlier detection
12. **ConnectedComponentsActor** - Graph connectivity
13. **ConstraintActor** - Physics constraint enforcement
14. **OntologyConstraintActor** - Ontology-based constraints
15. **GPUManagerActor** - GPU resource management

#### Support Actors (6)

16. **OntologyActor** - OWL reasoning engine
17. **MetadataActor** - Node/edge metadata management
18. **WorkspaceActor** - Project state persistence
19. **TaskOrchestratorActor** - Background task coordination
20. **AgentMonitorActor** - Health monitoring and metrics
21. **OptimizedSettingsActor** - User settings with Neo4j backend

---

## Hexagonal Architecture

### Port Interfaces (9 Traits)

Located in `src/ports/`, these define technology-agnostic boundaries:

1. **GraphRepository** - Graph CRUD operations
2. **KnowledgeGraphRepository** - Knowledge graph specific operations
3. **OntologyRepository** - Ontology storage and reasoning
4. **SettingsRepository** - User settings persistence
5. **PhysicsSimulator** - Physics computation interface
6. **SemanticAnalyzer** - Semantic analysis interface
7. **GpuPhysicsAdapter** - GPU physics computation
8. **GpuSemanticAnalyzer** - GPU semantic analysis
9. **InferenceEngine** - Ontology reasoning interface

### Adapter Implementations (12 Adapters)

Located in `src/adapters/`, these implement ports with concrete technologies:

#### Neo4j Adapters (5)
1. **Neo4jAdapter** - Primary Neo4j connection pool
2. **Neo4jGraphRepository** - Graph data persistence
3. **Neo4jSettingsRepository** - User settings with authentication
4. **Neo4jOntologyRepository** - Ontology storage
5. **ActorGraphRepository** - Bridge adapter for actor-based access

#### GPU Adapters (2)
6. **GpuSemanticAnalyzerAdapter** - CUDA semantic analysis
7. **ActixPhysicsAdapter** - Actor wrapper for GPU physics

#### Actor Adapters (3)
8. **ActixSemanticAdapter** - Actor wrapper for semantic processing
9. **PhysicsOrchestratorAdapter** - Physics actor coordination
10. **WhelkInferenceEngine** - Rust-based OWL reasoning

#### Specialized Adapters (2)
11. **ActixWebSocketAdapter** - WebSocket infrastructure (implicit)
12. **ActixHttpAdapter** - HTTP handlers (implicit)

---

## CQRS Implementation

The CQRS layer provides separation between reads (queries) and writes (commands):

```mermaid
graph LR
    subgraph Commands["📝 Commands (Write)"]
        CMD1["AddNodeCommand"]
        CMD2["UpdateNodeCommand"]
        CMD3["DeleteNodeCommand"]
        CMD4["UpdatePhysicsCommand"]
    end

    subgraph Queries["🔍 Queries (Read)"]
        QRY1["GetGraphDataQuery"]
        QRY2["GetNodeQuery"]
        QRY3["GetPhysicsStateQuery"]
        QRY4["GetSettingsQuery"]
    end

    subgraph Handlers["⚙️ Handlers"]
        CMDH["Command Handlers<br/>━━━━━━━━━━━<br/>• Validate<br/>• Execute<br/>• Persist"]
        QRYH["Query Handlers<br/>━━━━━━━━━━━<br/>• Read<br/>• Filter<br/>• Return DTOs"]
    end

    CMD1 & CMD2 & CMD3 & CMD4 --> CMDH
    QRY1 & QRY2 & QRY3 & QRY4 --> QRYH

    CMDH --> ACTORS["Actor System"]
    QRYH --> ACTORS

    ACTORS --> NEO4J["Neo4j Database"]

    classDef cmd fill:#ffccbc,stroke:#d84315,stroke-width:2px
    classDef qry fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef handler fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    class CMD1,CMD2,CMD3,CMD4 cmd
    class QRY1,QRY2,QRY3,QRY4 qry
    class CMDH,QRYH handler
```

### Handler Statistics

- **8 Query Handlers** - Read operations (graph, physics, settings, ontology)
- **5 Command Handlers** - Write operations (estimated, needs audit)
- **~114 CQRS operations total** (from hexagonal-cqrs.md reference)

---

## Neo4j Integration

### Database Schema

Neo4j serves as the **single source of truth** with the following node labels:

#### Graph Data
- **Node** - Graph visualization nodes
  - Properties: `id`, `label`, `x`, `y`, `z`, `metadata_id`, `quality_score`
- **Edge** - Graph connections
  - Properties: `source`, `target`, `relationship`, `weight`

#### User Data
- **User** - Authenticated users (Nostr identity)
  - Properties: `npub`, `created_at`, `last_seen`
- **UserSettings** - Per-user configuration
  - Properties: `user_id`, `key`, `value`, `category`, `updated_at`
- **FilterState** - Per-client filter state
  - Properties: `client_id`, `user_id`, `filter_json`, `created_at`

#### Ontology Data
- **OntologyClass** - OWL classes
- **OntologyProperty** - OWL properties
- **Axiom** - OWL axioms
- **Constraint** - Semantic constraints

### Query Patterns

**Read Pattern** (most common):
```rust
// Via CQRS Query Handler
GetGraphDataQuery → QueryHandler → Neo4jGraphRepository → Neo4j
// Returns: GraphData with nodes and edges
```

**Write Pattern**:
```rust
// Via CQRS Command Handler
AddNodeCommand → CommandHandler → GraphStateActor → Neo4jGraphRepository → Neo4j
// Broadcasts: WebSocket update to all clients
```

---

## ⚠️ Current Limitations

### Test Coverage
- **Integration tests**: Limited coverage for actor message flows
- **CQRS handlers**: Basic unit tests exist, need more edge cases
- **Neo4j queries**: Manual testing only, need automated integration tests
- **GPU kernels**: No automated testing (GPU hardware required)

### Actor Message Ordering
- **Complexity**: Message ordering guarantees only within single actor
- **Inter-actor coordination**: Requires manual synchronization
- **Race conditions**: Possible with concurrent client updates
- **Mitigation**: Message acknowledgment system in place (src/actors/messaging/)

### Performance Bottlenecks
- **Neo4j connection pool**: Fixed size (10 connections), no dynamic scaling
- **WebSocket broadcasts**: Sequential, could benefit from batching
- **GPU memory**: Limited by VRAM, no automatic spillover to system RAM
- **Actor mailbox**: Unbounded, potential memory growth under load

### Technical Debt
- **ActorGraphRepository**: Bridge pattern adds indirection (low priority to remove)
- **Mixed CQRS adoption**: Some endpoints still bypass CQRS layer
- **Settings migration**: Partial migration from SQLite to Neo4j (95% complete)
- **Error handling**: Inconsistent error types across layers

---

## Performance Characteristics

### Throughput (Measured on Development Hardware)

| Operation | Latency (p50) | Latency (p95) | Throughput |
|-----------|---------------|---------------|------------|
| Get Graph Data | 12ms | 28ms | ~83 req/s |
| Add Node | 8ms | 18ms | ~125 req/s |
| Physics Step (1000 nodes) | 16ms | 35ms | ~60 steps/s |
| WebSocket Broadcast | 3ms | 8ms | ~333 msg/s |
| Neo4j Query (simple) | 2ms | 5ms | ~500 req/s |
| GPU Force Calculation | 4ms | 9ms | ~250 steps/s |

**Hardware**: NVIDIA RTX 3060 (12GB VRAM), 16 GB RAM, Ryzen 5600X

### Scaling Characteristics

- **Horizontal scaling**: Limited by Neo4j (single instance)
- **Vertical scaling**: Good GPU utilization (80-95% under load)
- **Concurrent clients**: Tested up to 50 WebSocket clients
- **Memory growth**: ~200 MB per 10,000 nodes in graph

---

## Future Improvements

### Short-term (Next Sprint)
1. **Actor supervision improvements**
   - Add circuit breakers for failing actors
   - Implement exponential backoff for restarts
   - Better error propagation to clients

2. **Performance optimization**
   - Batch WebSocket broadcasts (16ms window)
   - Dynamic Neo4j connection pool sizing
   - GPU memory management (LRU cache)

3. **Testing coverage**
   - Integration tests for critical paths
   - Property-based testing for CQRS handlers
   - Load testing framework

### Medium-term (Next Quarter)
1. **Distributed deployment**
   - Neo4j cluster support (causal clustering)
   - Actor system distribution (Actix clustering)
   - Horizontal scaling of GPU workers

2. **Event sourcing**
   - Complete event log for all graph changes
   - Event replay for debugging
   - Temporal queries ("graph state at time T")

3. **Observability**
   - OpenTelemetry tracing
   - Prometheus metrics export
   - Grafana dashboards

### Long-term (Next 6 Months)
1. **Multi-GPU support**
   - GPU pooling for multiple clients
   - Dynamic workload distribution
   - GPU memory sharing

2. **Advanced caching**
   - Redis cluster for distributed caching
   - Cache invalidation via event bus
   - Partial cache updates

3. **Real-time collaboration**
   - Conflict-free replicated data types (CRDTs)
   - Multi-user graph editing
   - Presence awareness

---

### Architecture Deep Dives
- [Hexagonal/CQRS Architecture](../../../explanations/architecture/hexagonal-cqrs.md) - Deep dive into ports/adapters/CQRS
- [Actor System Guide](../../../guides/architecture/actor-system.md) - Actor patterns, message ordering, debugging
- [Database Architecture](../../../explanations/architecture/database-architecture.md) - Neo4j schema, queries, performance

### Historical References
- [GraphServiceActor Migration](../../../guides/graphserviceactor-migration.md) - Migration history (completed Nov 2025)
- [Architecture Overview](../../../ARCHITECTURE_OVERVIEW.md) - High-level system diagram

### Implementation References
- [API Reference](../../../reference/api/README.md) - HTTP and WebSocket endpoints
- [GPU Computing](../../../explanations/gpu-computing.md) - CUDA kernel details
- [Settings System](../../../guides/user-settings.md) - User settings with Nostr auth

---

---

---

## Related Documentation

- [VisionFlow Documentation Modernization - Final Report](../../../DOCUMENTATION_MODERNIZATION_COMPLETE.md)
- [Server-Side Actor System - Complete Architecture Documentation](../../../diagrams/server/actors/actor-system-complete.md)
- [VisionFlow GPU CUDA Architecture - Complete Technical Documentation](../../../diagrams/infrastructure/gpu/cuda-architecture-complete.md)
- [Complete State Management Architecture](../../../diagrams/client/state/state-management-complete.md)
- [X-FluxAgent Integration Plan for ComfyUI MCP Skill](../../../multi-agent-docker/x-fluxagent-adaptation-plan.md)

## Changelog

**December 2, 2025**
- ✅ Updated to reflect stable hexagonal architecture
- ✅ Added candid assessments of limitations and performance
- ✅ Documented 21-actor hierarchy with supervision structure
- ✅ Added Neo4j integration details
- ✅ Included performance benchmarks and future improvements

**November 5, 2025**
- ✅ GraphServiceActor fully removed
- ✅ Modular actor architecture operational
- ✅ Neo4j as source of truth confirmed

**November 3, 2025**
- ✅ Hexagonal architecture implementation complete
- ✅ CQRS layer operational
- ✅ Unified database migration complete
