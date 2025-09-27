# Rust Server Architecture

**Last Updated**: 2025-09-27
**Status**: Transitional Architecture - Migration from Monolithic to Supervised Actor System
**Analysis Base**: Direct source code inspection of Rust server implementation

> **Note**: This codebase is evolving rapidly. The server is currently in a transitional state, migrating from a monolithic `GraphServiceActor` to a fully supervised architecture. This document reflects the current hybrid implementation.

## High-Level System Overview

```mermaid
graph TB
    %% Client Layer
    Client[Web Clients<br/>Unity/Browser]

    %% Entry Point
    Main[main.rs<br/>HTTP Server Entry Point]

    %% Core Server Infrastructure
    subgraph "HTTP Server Layer"
        HttpServer[Actix HTTP Server<br/>:8080]
        Middleware[CORS + Logger + Compression<br/>Error Recovery Middleware]
        Router[Route Configuration]
    end

    %% Application State
    AppState[AppState<br/>Centralised State Management]

    %% Actor System - CURRENT STATE
    subgraph "Actor System (Actix) - Transitional Architecture"
        subgraph "Graph Supervision (Hybrid)"
            TransitionalSupervisor[TransitionalGraphSupervisor<br/>Bridge Pattern Wrapper]
            GraphActor[GraphServiceActor<br/>Monolithic (Being Refactored)]
            GraphStateActor[GraphStateActor<br/>State Management (Partial)]
            PhysicsOrchestrator[PhysicsOrchestratorActor<br/>Physics (Extracted)]
            SemanticProcessor[SemanticProcessorActor<br/>Semantic Analysis]
        end

        GPUManager[GPUManagerActor<br/>GPU Resource Management]
        ClientCoordinator[ClientCoordinatorActor<br/>WebSocket Connections]
        OptimisedSettings[OptimisedSettingsActor<br/>Configuration Management]
        ProtectedSettings[ProtectedSettingsActor<br/>Secure Configuration]
        MetadataActor[MetadataActor<br/>File Metadata Storage]
        WorkspaceActor[WorkspaceActor<br/>Project Management]
        ClaudeFlowActor[ClaudeFlowActor<br/>MCP Integration]
        TcpConnectionActor[TcpConnectionActor<br/>TCP Management]
        FileSearchActor[FileSearchActor<br/>Content Search]
        CacheActor[CacheActor<br/>Memory Caching]
        OntologyActor[OntologyActor<br/>Ontology Processing]
    end

    %% Non-Actor Services
    subgraph "Utility Services (Non-Actor)"
        DockerHiveMind[DockerHiveMind<br/>Container Orchestration]
        McpTcpClient[McpTcpClient<br/>Direct TCP Client]
        JsonRpcClient[JsonRpcClient<br/>MCP Protocol]
    end

    %% WebSocket Handlers
    subgraph "WebSocket Layer"
        SocketFlow[Socket Flow Handler<br/>Binary Graph Updates (34-byte)]
        SpeechWS[Speech WebSocket<br/>Voice Commands]
        MCPRelay[MCP Relay WebSocket<br/>Multi-Agent Communication]
        HealthWS[Health WebSocket<br/>System Monitoring]
    end

    %% REST API Handlers
    subgraph "REST API Layer"
        APIHandler[API Handler<br/>/api routes]
        GraphAPI[Graph API<br/>CRUD operations]
        FilesAPI[Files API<br/>GitHub integration]
        BotsAPI[Bots API<br/>Task Management via DockerHiveMind]
        HybridAPI[Hybrid API<br/>Docker/MCP Spawning]
        AnalyticsAPI[Analytics API<br/>GPU computations]
        WorkspaceAPI[Workspace API<br/>Project management]
    end

    %% GPU Subsystem - FULLY IMPLEMENTED
    subgraph "GPU Computation Layer (40 CUDA Kernels)"
        GPUResourceActor[GPU Resource Actor<br/>CUDA Device & Memory]
        ForceComputeActor[Force Compute Actor<br/>Physics Kernels]
        ClusteringActor[Clustering Actor<br/>K-means, Louvain]
        ConstraintActor[Constraint Actor<br/>Layout Constraints]
        AnomalyDetectionActor[Anomaly Detection Actor<br/>LOF, Z-score]
        StressMajorizationActor[Stress Majorisation<br/>Graph Layout]
    end

    %% Data Storage
    subgraph "Data Layer"
        FileStorage[File System Storage<br/>Metadata & Graph Data]
        MemoryStore[In-Memory Store<br/>Active Graph State]
        CUDA[CUDA GPU Memory<br/>Compute Buffers]
    end

    %% External Services
    subgraph "External Integrations"
        GitHub[GitHub API<br/>Content Fetching]
        Docker[Docker Services<br/>multi-agent-container]
        ClaudeFlowCLI[Claude Flow CLI<br/>npx claude-flow]
        MCP[MCP Servers<br/>TCP :9500]
        Nostr[Nostr Protocol<br/>Decentralised Identity]
        RAGFlow[RAGFlow API<br/>Chat Integration]
        Speech[Speech Services<br/>Voice Processing]
    end

    %% Connections
    Client --> HttpServer
    HttpServer --> Middleware
    Middleware --> Router
    Router --> AppState

    AppState --> TransitionalSupervisor
    TransitionalSupervisor --> GraphActor
    GraphActor -.-> GraphStateActor
    GraphActor -.-> PhysicsOrchestrator
    GraphActor -.-> SemanticProcessor

    AppState --> GPUManager
    AppState --> ClientCoordinator
    AppState --> OptimisedSettings
    AppState --> ClaudeFlowActor

    ClaudeFlowActor --> TcpConnectionActor
    ClaudeFlowActor --> JsonRpcClient

    BotsAPI --> DockerHiveMind
    DockerHiveMind --> Docker

    Router --> SocketFlow
    SocketFlow --> ClientCoordinator

    GPUManager --> GPUResourceActor
    GPUManager --> ForceComputeActor
    GPUManager --> ClusteringActor
    GPUManager --> ConstraintActor
    GPUManager --> AnomalyDetectionActor
    GPUManager --> StressMajorizationActor

    TcpConnectionActor --> MCP
    DockerHiveMind --> ClaudeFlowCLI
```

## Actor System Architecture - Transitional State

### Current Implementation Status

The server is in **Phase 2 of 3** of a major architectural refactoring:

#### Phase 1: ✅ COMPLETE - Actor Extraction
- Client management → `ClientCoordinatorActor`
- GPU management → `GPUManagerActor` + 6 specialised actors
- Settings → `OptimisedSettingsActor`
- TCP connections → `TcpConnectionActor`

#### Phase 2: 🔄 IN PROGRESS - Supervision Layer
- Implemented `TransitionalGraphSupervisor` as bridge pattern
- `GraphServiceActor` still handles core functionality (35,193 lines)
- Partial extraction of physics and semantic processing
- Message routing through supervisor wrapper

#### Phase 3: ❌ NOT STARTED - Full Decomposition
- Complete breakdown of monolithic `GraphServiceActor`
- Full `GraphServiceSupervisor` implementation
- Pure actor-based message routing

### Transitional Architecture Details

```rust
// Current architecture in app_state.rs
pub struct AppState {
    pub graph_service_addr: Addr<TransitionalGraphSupervisor>, // Wrapper around GraphServiceActor
    pub gpu_manager_addr: Addr<GPUManagerActor>,
    pub client_coordinator_addr: Addr<ClientCoordinatorActor>,
    // ... other actors
}
```

The `TransitionalGraphSupervisor`:
- **Wraps** the existing monolithic `GraphServiceActor`
- **Forwards** all messages to maintain compatibility
- **Manages** actor lifecycle and restarts
- **Bridges** between current and planned architecture

### Planned Final Architecture (Not Yet Implemented)

```
GraphServiceSupervisor (Future)
├── GraphStateActor          (State management & persistence)
├── PhysicsOrchestratorActor (Physics simulation & GPU compute)
├── SemanticProcessorActor   (Semantic analysis & AI features)
└── ClientCoordinatorActor   (WebSocket & client management)
```

## Binary Protocol Specifications

### Wire Protocol Format (34 bytes)

Manual serialisation creates 34-byte packets:

```rust
// Wire format (manually serialised)
Wire Packet {
    node_id: u16,           // 2 bytes (truncated for bandwidth)
    position: [f32; 3],     // 12 bytes
    velocity: [f32; 3],     // 12 bytes
    sssp_distance: f32,     // 4 bytes (default: f32::INFINITY)
    sssp_parent: i32,       // 4 bytes (default: -1)
}
// Total: 34 bytes transmitted
```

### GPU Internal Format (48 bytes)

Server-side GPU computation format:

```rust
pub struct BinaryNodeDataGPU {
    pub node_id: u32,         // 4 bytes
    pub x: f32, y: f32, z: f32,     // 12 bytes - position
    pub vx: f32, vy: f32, vz: f32,  // 12 bytes - velocity
    pub sssp_distance: f32,   // 4 bytes
    pub sssp_parent: i32,     // 4 bytes
    pub cluster_id: i32,      // 4 bytes
    pub centrality: f32,      // 4 bytes
    pub mass: f32,            // 4 bytes
}
// Total: 48 bytes (GPU-only, not sent to clients)
```

## GPU Computation Layer

### CUDA Kernel Implementation

**40 Production CUDA Kernels** across 5 files:

#### Kernel Distribution:
- **visionflow_unified.cu**: 28 kernels (core physics, clustering, anomaly detection)
- **gpu_clustering_kernels.cu**: 8 kernels (specialised clustering algorithms)
- **visionflow_unified_stability.cu**: 2 kernels (stability optimisation)
- **sssp_compact.cu**: 2 kernels (SSSP frontier compaction)
- **dynamic_grid.cu**: Host-side optimisation only

### GPU Actor Hierarchy

All 6 specialised GPU actors are fully implemented:

```
GPUManagerActor (Supervisor)
├── GPUResourceActor        (Memory & device management)
├── ForceComputeActor       (Physics simulation)
├── ClusteringActor         (K-means, Louvain, community detection)
├── AnomalyDetectionActor   (LOF, Z-score analysis)
├── StressMajorizationActor (Graph layout optimisation)
└── ConstraintActor         (Distance, position, semantic constraints)
```

### GPU Capabilities

- **Physics Engine**: Force-directed layout, spring-mass physics, spatial grid optimisation
- **Clustering**: K-means++, Louvain modularity, label propagation
- **Anomaly Detection**: Local Outlier Factor (LOF), statistical Z-score
- **Performance**: Stability gates, kinetic energy monitoring, dynamic grid sizing
- **Memory**: RAII wrappers, stream-based execution, shared context

## External Integration Architecture

### Task & Container Orchestration

**Important**: Task orchestration is implemented via utility services, NOT actors:

#### DockerHiveMind (Utility Service)
- **Location**: `/src/utils/docker_hive_mind.rs`
- **Pattern**: Direct Docker exec commands
- **Target**: `multi-agent-container`
- **Features**: Session caching, health monitoring, cleanup automation

```bash
# Example command executed
docker exec multi-agent-container /app/node_modules/.bin/claude-flow hive-mind spawn
```

#### Task Management Endpoints
- **Implementation**: `/src/handlers/bots_handler.rs`
- **Endpoints**: `/bots/remove-task`, `/pause-task`, `/resume-task`
- **Pattern**: REST handlers calling DockerHiveMind methods

### MCP Communication Stack

Layered actor system for MCP protocol:

```
ClaudeFlowActor (Application Logic)
    ↓
TcpConnectionActor (TCP Stream Management)
    ↓
JsonRpcClient (Protocol Correlation)
    ↓
MCP Server (multi-agent-container:9500)
```

### Integration Patterns

1. **Docker Direct**: DockerHiveMind → Docker exec → claude-flow CLI
2. **MCP TCP**: ClaudeFlowActor → TcpConnectionActor → MCP:9500
3. **Backup Path**: McpTcpClient → Direct TCP → MCP:9500

## Data Models & Configuration

### Field Serialisation

All data models use consistent camelCase conversion:

```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    pub node_id: u32,      // → "nodeId" in JSON
    pub display_name: String, // → "displayName" in JSON
    pub force_x: f32,      // → "forceX" in JSON
}
```

### Settings Management

- **OptimisedSettingsActor**: High-performance configuration management
- **ProtectedSettingsActor**: Secure settings with encryption
- **AppFullSettings**: Comprehensive configuration structure
- **Dynamic Updates**: Real-time settings propagation

## WebSocket Communication

### Binary Protocol (Position Updates)
- **Format**: 34-byte wire protocol
- **Frequency**: Real-time streaming
- **Compression**: Binary encoding (95% smaller than JSON)

### JSON Protocol (Metadata)
- **Settings Updates**: Configuration changes
- **Agent Status**: Task and health information
- **Graph Metadata**: Labels, relationships, properties

## Current Architecture Strengths

✅ **Modular GPU System**: 40 production kernels with proper actor separation
✅ **Transitional Stability**: Bridge pattern allows gradual migration
✅ **Protocol Efficiency**: Optimised binary format for real-time updates
✅ **Resilient Integration**: Multiple paths for external service communication
✅ **Type Safety**: Comprehensive Rust type system with serde serialisation

## Known Architecture Issues

⚠️ **Monolithic Graph Actor**: 35k+ lines need decomposition
⚠️ **Documentation Drift**: Missing actors referenced in docs
⚠️ **Binary Protocol Confusion**: Multiple formats with unclear documentation
⚠️ **Pattern Fragmentation**: Task orchestration split between utilities and handlers

## Migration Roadmap

1. **Current State**: TransitionalGraphSupervisor wrapping monolithic actor
2. **Next Phase**: Extract remaining functionality from GraphServiceActor
3. **Final State**: Pure supervised architecture with specialised actors
4. **Timeline**: Estimated 2-3 sprints for complete migration

---

*This architecture document represents the current transitional state of the VisionFlow Rust server as of 2025-09-27. The system is actively migrating from a monolithic to a supervised actor architecture. Documentation will be updated as the migration progresses.*