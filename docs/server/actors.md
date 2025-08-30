# Actor System

VisionFlow's backend is built around the Actix actor framework, which allows for highly concurrent and fault-tolerant state management. Each core component of the server is implemented as an actor, an independent entity that communicates with other actors by sending and receiving asynchronous messages.

This model avoids common concurrency problems like race conditions and deadlocks by ensuring that each piece of state is owned by a single actor.

## Core Actors

### `GraphServiceActor`
**The central orchestrator.** This actor is the heart of the server, responsible for managing the state of the dual graph (knowledge graph and agent graph).

**Responsibilities:**
-   Manages the in-memory representation of the graph data.
-   Orchestrates complex workflows by delegating tasks to other actors and services. For example, it sends compute-intensive tasks to the `GPUComputeActor` and validation requests to the `OntologyActor`.
-   Integrates data from multiple sources, such as agent telemetry from `ClaudeFlowActorTcp` and file system updates.
-   Periodically runs the `StressMajorizationSolver` to perform global layout optimizations.
-   Uses the `SemanticAnalyzer` to generate dynamic constraints for the physics engine.
-   Broadcasts graph updates to connected clients via the `ClientManagerActor`.

### `GPUComputeActor`
**The high-performance compute engine.** This actor acts as the sole interface to the `UnifiedGPUCompute` engine, which runs on a dedicated CUDA kernel (`visionflow_unified.cu`).

**Responsibilities:**
-   Orchestrates the execution of GPU-accelerated tasks.
-   Supports multiple compute modes that can be switched at runtime:
    -   `Basic`: A standard force-directed layout.
    -   `DualGraph`: Manages physics for both the knowledge and agent graphs simultaneously.
    -   `Constraints`: Applies semantic constraints to the layout.
    -   `VisualAnalytics`: Runs specialized analytics computations.
-   Handles new message types for offloading specific analytics tasks, such as `PerformGPUClustering` and `ComputeShortestPaths`.
-   Manages the transfer of data between system memory and GPU memory.

### `ClaudeFlowActorTcp`
**The MCP integration gateway.** This actor is responsible for all communication with the external Claude Flow MCP server. It replaces all previous WebSocket or stdio implementations.

**Responsibilities:**
-   Maintains a resilient, TCP-only connection to the MCP server on port 9500.
-   Implements robust network resilience patterns, including:
    -   **Connection Pooling**: Manages a pool of TCP connections for efficiency.
    -   **Exponential Backoff**: Automatically retries failed connections with increasing delays.
    -   **Circuit Breaker**: Halts connection attempts for a period after repeated failures to avoid overwhelming the server.
-   Handles the line-delimited JSON-RPC protocol used for MCP communication.
-   Correlates outgoing requests with incoming responses using a `pending_requests` HashMap.

### `OntologyActor` (New)
**The graph validation and reasoning engine.** This new actor provides a formal validation layer for the knowledge graph, ensuring its logical consistency.

**Responsibilities:**
-   Receives `ValidateGraph` messages from the `GraphServiceActor`.
-   Uses the `OwlValidatorService` to perform validation, which involves:
    1.  Mapping the property graph to a formal RDF graph.
    2.  Using the `horned-owl` library to parse OWL ontologies.
    3.  Using the `whelk-rs` reasoner to check for logical inconsistencies and perform inference.
-   Returns a `ValidationReport` to the `GraphServiceActor` with the results.
-   Operates asynchronously to handle potentially long-running reasoning tasks without blocking the main application threads.

For more details, see the [Ontology Validation documentation](features/ontology.md).