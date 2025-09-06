# Server Architecture

*[Server](../index.md)*

## Overview

VisionFlow's backend is an actor-based system built in Rust using the Actix framework. It is designed for high-performance, real-time graph visualisation and computation. The architecture is centered around a message-passing model that eliminates lock contention and enables a high degree of concurrency.

Key architectural pillars include:
- **Actor-Based Concurrency**: State is managed by independent actors that communicate via asynchronous messages, avoiding shared-memory pitfalls like `Arc<RwLock<T>>`.
- **Unified GPU Compute**: A single, powerful CUDA kernel (`visionflow_unified.cu`) handles all GPU-accelerated tasks, from physics simulations to graph analytics.
- **TCP-Only MCP Integration**: A resilient TCP client (`ClaudeFlowActorTcp`) communicates with the MCP server for agent control, featuring robust error handling and connection management.
- **Modular Services**: New features like ontology validation and semantic analysis are encapsulated in dedicated services and actors, integrating cleanly into the existing data flow.

## Core Components & Data Flow

The system is composed of several specialised actors that collaborate to process data and serve it to clients.

```mermaid
graph TD
    subgraph "External Inputs"
        MCP["🤖 Claude Flow MCP (TCP)"]
        API["🌐 REST API Clients"]
        FS["📄 File System (Metadata)"]
    end

    subgraph "Core Actor System"
        CFT[ClaudeFlowActorTcp]
        GSA[GraphServiceActor]
        GCA[GPUComputeActor]
        OA[OntologyActor]
        SA[SemanticAnalyzer]
    end

    subgraph "Services"
        OVS[OwlValidatorService]
        AEG[AdvancedEdgeGenerator]
        SMS[StressMajorizationSolver]
    end

    subgraph "Outputs"
        CMA[ClientManagerActor] --> WS["📡 WebSocket Clients"]
    end

    MCP -- "Line-delimited JSON-RPC" --> CFT
    API -- "HTTP Requests" --> GSA
    FS -- "File Events" --> GSA

    CFT -- "Agent Telemetry" --> GSA
    
    GSA -- "Orchestrates" --> SA
    GSA -- "Orchestrates" --> SMS
    GSA -- "ValidateGraph" --> OA
    GSA -- "PerformGPUClustering, etc." --> GCA

    OA -- "Uses" --> OVS
    SA -- "Uses" --> AEG

    GCA -- "UnifiedGPUCompute Engine" --> GCA
    
    GSA -- "Broadcast Updates" --> CMA

    style GSA fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style GCA fill:#3A3F47,stroke:#76B900,color:#FFFFFF
    style CFT fill:#3A3F47,stroke:#F56565,color:#FFFFFF
    style OA fill:#3A3F47,stroke:#D69E2E,color:#FFFFFF
    style SA fill:#3A3F47,stroke:#9F7AEA,color:#FFFFFF
```

### Data Flow Explained

1.  **Inputs**: The system receives data from three primary sources:
    *   **Claude Flow MCP**: The `ClaudeFlowActorTcp` maintains a persistent TCP connection to the MCP server, receiving agent telemetry and control messages.
    *   **REST API**: Clients can interact with the graph, run analytics, and manage settings through a standard HTTP API, handled primarily by the `GraphServiceActor`.
    *   **File System**: The `GraphServiceActor` monitors the file system for changes to the knowledge graph's source files.

2.  **Orchestration (`GraphServiceActor`)**: The `GraphServiceActor` is the central hub. It manages the dual graph (knowledge vs. agent) and orchestrates all major operations. It delegates tasks to specialised actors and services:
    *   It sends compute-heavy tasks like clustering and physics calculations to the `GPUComputeActor`.
    *   It requests graph validation from the `OntologyActor`.
    *   It uses the `SemanticAnalyzer` to enrich the graph and the `StressMajorizationSolver` for layout optimisation.

3.  **Specialised Actors & Services**:
    *   **`GPUComputeActor`**: Manages the `UnifiedGPUCompute` engine, executing various compute modes as requested.
    *   **`OntologyActor`**: Provides an asynchronous interface to the `OwlValidatorService`, which performs formal OWL/RDF validation and reasoning.
    *   **`SemanticAnalyzer`**: A service that extracts features, generates semantic edges, and creates layout constraints.

4.  **Output**: The `GraphServiceActor` sends the final, computed graph state to the `ClientManagerActor`, which then broadcasts the updates to all connected WebSocket clients.

## Actor-Based State Management

The server avoids traditional shared-state concurrency models. Instead of wrapping data in `Arc<RwLock<T>>`, each piece of state is "owned" by a single actor. To read or modify state, other actors must send a message and `await` a response.

This approach provides several key benefits:
-   **No Lock Contention**: Eliminates performance bottlenecks and deadlocks associated with shared locks.
-   **Clear Ownership**: State ownership is unambiguous, simplifying the design and reducing bugs.
-   **Fault Isolation**: If an actor panics, its state is lost, but it does not corrupt the state of other actors. The supervisor can restart it cleanly.
-   **Asynchronous by Default**: The entire system is built on non-blocking communication, ensuring that no single task can hold up the entire server.

## Related Topics

- [AI Services Documentation](../server/ai-services.md)
- [Actor System](../server/actors.md)
- [Agent Visualisation Architecture](../agent-visualization-architecture.md)
- [Architecture Documentation](../architecture/README.md)
- [Architecture Migration Guide](../architecture/migration-guide.md)
- [Bots Visualisation Architecture](../architecture/bots-visualization.md)
- [Bots/VisionFlow System Architecture](../architecture/bots-visionflow-system.md)
- [Case Conversion Architecture](../architecture/CASE_CONVERSION.md)
- [Claude Flow MCP Integration](../server/features/claude-flow-mcp-integration.md)
- [ClaudeFlowActor Architecture](../architecture/claude-flow-actor.md)
- [Client Architecture](../client/architecture.md)
- [Configuration Architecture](../server/config.md)
- [Decoupled Graph Architecture](../technical/decoupled-graph-architecture.md)
- [Dynamic Agent Architecture (DAA) Setup Guide](../architecture/daa-setup-guide.md)
- [Feature Access Control](../server/feature-access.md)
- [GPU Compute Architecture](../server/gpu-compute.md)
- [GPU Compute Improvements & Troubleshooting Guide](../architecture/gpu-compute-improvements.md)
- [Graph Clustering](../server/features/clustering.md)
- [MCP Connection Architecture](../architecture/mcp_connection.md)
- [MCP Integration Architecture](../architecture/mcp-integration.md)
- [MCP Integration](../server/mcp-integration.md)
- [MCP WebSocket Relay Architecture](../architecture/mcp-websocket-relay.md)
- [Managing the Claude-Flow System](../architecture/managing_claude_flow.md)
- [Multi Agent Orchestration](../server/agent-swarm.md)
- [Ontology Validation](../server/features/ontology.md)
- [Parallel Graph Architecture](../architecture/parallel-graphs.md)
- [Physics Engine](../server/physics-engine.md)
- [Request Handlers Architecture](../server/handlers.md)
- [Semantic Analysis Pipeline](../server/features/semantic-analysis.md)
- [Server Documentation](../server/index.md)
- [Server-Side Data Models](../server/models.md)
- [Services Architecture](../server/services.md)
- [Settings Architecture Analysis Report](../architecture_analysis_report.md)
- [Types Architecture](../server/types.md)
- [Utilities Architecture](../server/utils.md)
- [VisionFlow Component Architecture](../architecture/components.md)
- [VisionFlow Data Flow Architecture](../architecture/data-flow.md)
- [VisionFlow GPU Compute Integration](../architecture/gpu-compute.md)
- [VisionFlow GPU Migration Architecture](../architecture/visionflow-gpu-migration.md)
- [VisionFlow System Architecture Overview](../architecture/index.md)
- [VisionFlow System Architecture](../architecture/system-overview.md)
- [arch-system-design](../reference/agents/architecture/system-design/arch-system-design.md)
- [architecture](../reference/agents/sparc/architecture.md)
