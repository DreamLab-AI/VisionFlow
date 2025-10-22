# Task: Full Architectural Migration to a Database-Backed Hexagonal System

## 1. Objective

This document outlines the comprehensive plan to refactor the application to a fully database-backed, Hexagonal Architecture using the `hexser` crate. The goal is to completely eliminate all file-based data and configuration, resulting in a more robust, scalable, and maintainable system. This is a non-breaking upgrade that will be implemented in a phased approach.

## 2. Core Architectural Principles

*   **Hexagonal Architecture**: All new components will be structured according to the Ports and Adapters pattern, enforced by the `hexser` crate's derive macros. The implementation must be complete, with no stubs or placeholders.
*   **Database as Single Source of Truth**: All application data and configuration will be migrated to a SQLite database. All `.json`, `.yaml`, and `.toml` files will be deprecated and removed.
*   **Separation of Data Domains**: The application will be divided into three distinct data domains, each with its own dedicated database file for clarity and separation of concerns:
    *   `settings.db`: For all application, user (with tiered auth), and developer configurations.
    *   `knowledge_graph.db`: For the main graph structure, parsed from local markdown files.
    *   `ontology.db`: For the ontology graph structure, parsed from GitHub markdown files.
*   **CQRS (Command Query Responsibility Segregation)**: The application layer will be structured using `hexser`'s `Directive` (write) and `Query` (read) patterns.
*   **API Strategy**:
    *   **REST API**: Used for on-demand, rich data loading (e.g., initial graph structure, settings).
    *   **WebSocket API**: Used exclusively for high-frequency, low-latency, bi-directional updates (positions, velocities, voice data) via a simple binary protocol.
*   **Client-Side Simplicity**: The client-side caching and lazy-loading layer will be removed in favor of a direct, on-demand data fetching model from the new, high-performance REST API.

## 3. Scope of Changes: Key Files & Directories

This refactoring will primarily impact the following areas:

### Server-Side (`src/`)

*   **Created**:
    *   `src/ports/`: Directory for all `hexser` port traits.
    *   `src/adapters/`: Directory for all `hexser` adapter implementations.
    *   `src/actors/semantic_processor_actor_new.rs`: The new, dedicated actor for semantic analysis.
*   **Heavily Modified**:
    *   `src/app_state.rs`: To orchestrate the new actor and service landscape.
    *   `src/main.rs`: To handle the new initialization and migration logic.
    *   `src/services/database_service.rs`: To manage connections to the three separate databases.
    *   `src/services/settings_service.rs`: To provide a high-level API for the settings database.
    *   `src/actors/optimized_settings_actor.rs`: To integrate with the new `SettingsService`.
    *   `src/actors/physics_orchestrator_actor.rs`: To focus solely on physics simulation.
    *   `src/actors/gpu/clustering_actor.rs`: To be integrated into the `SemanticProcessorActor`.
    *   `src/utils/binary_protocol.rs`: To implement the simplified, multiplexed WebSocket protocol.
    *   `src/handlers/`: All handlers will be refactored to use the CQRS pattern.
*   **Deleted/Deprecated**:
    *   `src/actors/graph_service_supervisor.rs`
    *   `src/actors/graph_actor.rs` (Monolithic version)
    *   All file-based config files in `data/` (`settings.yaml`, `dev_config.toml`, etc.).

### Client-Side (`client/src/`)

*   **Heavily Modified**:
    *   `client/src/store/settingsStore.ts`: To remove caching and implement direct fetching.
    *   `client/src/services/WebSocketService.ts`: To handle the new binary protocol.
    *   `client/src/services/BinaryWebSocketProtocol.ts`: To decode the new multiplexed messages.
    *   UI components in `client/src/features/` that manage settings or graph display.
*   **Deleted/Deprecated**:
    *   `client/src/client/settings_cache_client.ts`

## 4. Detailed Implementation Plan

### Phase 1: Project Setup & Dependency Integration

1.  **Add `hexser` Dependency**:
    *   Add `hexser = { version = "0.4.7", features = ["full"] }` to `Cargo.toml`.
2.  **Add `whelk-rs` Dependency**:
    *   Add the `whelk-rs` crate to `Cargo.toml` for the ontology inference engine.

### Phase 2: Database Migration & Complete Deprecation of File-Based Config

1.  **Create Unified Database Service**:
    *   Refactor `src/services/database_service.rs` to manage connections to `settings.db`, `knowledge_graph.db`, and `ontology.db`.
    *   Implement a migration utility to populate these databases from all legacy configuration files.
    *   Ensure the service handles `camelCase` to `snake_case` conversion automatically.

2.  **Implement `hexser` Repositories**:
    *   Create `SqliteSettingsRepository`, `SqliteKnowledgeGraphRepository`, and `SqliteOntologyRepository` adapters.

3.  **Complete Deprecation**:
    *   Remove all file I/O logic from the `config` modules and `AppState`.
    *   Delete the legacy configuration and data files from the `data/` directory.

### Phase 3: Full Actor Decomposition & Hexagonal Implementation

1.  **Refactor Actors as `hexser` Adapters**:
    *   **`GraphStateActor`**: Refactor to manage in-memory graph state, loading from and persisting to the `KnowledgeGraphRepository` and `OntologyRepository` ports.
    *   **`PhysicsOrchestratorActor`**: Refactor as the implementation for the `GpuPhysicsAdapter`.
    *   **`SemanticProcessorActor`**: Implement as the adapter for the `GpuSemanticAnalyzer`.

2.  **Implement `hexser` Ports & Application Layer (CQRS)**:
    *   Define all necessary port traits in `src/ports` using `#[derive(HexPort)]`.
    *   Create `Directive`/`Query` and `DirectiveHandler`/`QueryHandler` structs for all domains.

3.  **Complete Removal of Legacy Actors**:
    *   Fully remove `GraphServiceSupervisor` and the monolithic `GraphServiceActor`.

### Phase 4: API and WebSocket Refactoring

1.  **REST API**:
    *   Refactor all HTTP handlers to use the new CQRS handlers.
    *   Create a new endpoint (`/api/ontology/graph`) for on-demand loading of the ontology graph.
    *   Implement tiered authentication for settings endpoints.

2.  **WebSocket Protocol**:
    *   **Simplify Binary Protocol**: The WebSocket will handle two main types of high-frequency, bi-directional data: graph updates and voice data, identified by a 1-byte header (`0x01` for Graph, `0x02` for Voice).
    *   **Graph Update Payload**:
        *   **Server-to-Client**: A flat array of `[graph_type_flag, node_id, x, y, z, vx, vy, vz]`.
        *   **Client-to-Server (User Interaction)**: The client sends updates for dragged nodes in the same format. The server recalculates the physics and broadcasts the new state of the entire graph to all clients.
    *   **Bandwidth Throttling**: Implement dynamic throttling in the `ClientCoordinatorActor` to prioritize voice data over graph updates when necessary.

### Phase 5: Client-Side Refactoring

1.  **Simplify State Management**:
    *   Remove all client-side caching and lazy-loading from `settingsStore.ts`.
    *   Refactor UI components to fetch data directly from the REST API on-demand.

2.  **Integrate Ontology Mode**:
    *   Implement a UI toggle to switch between "Knowledge Graph Mode" and "Ontology Graph Mode".
    *   On mode switch, the client will fetch the appropriate graph structure via REST and then listen for WebSocket updates filtered by the `graph_type_flag`.

3.  **Remove Case Conversion Logic**:
    *   Remove all manual `camelCase` to `snake_case` conversion logic from the client.

### Phase 6: Semantic Analyzer Integration

1.  **Define `SemanticAnalyzer` Port**:
    *   Create a detailed `SemanticAnalyzer` port in `src/ports/semantic_analyzer.rs`.

2.  **Implement `SemanticProcessorActor` as Adapter**:
    *   Implement the `SemanticProcessorActor` to fulfill the `SemanticAnalyzer` contract using `#[derive(HexAdapter)]`.

3.  **Create To-Do List for Semantic Features**:
    *   The coding agent will be tasked with the following:
        *   [ ] **GPU-Accelerated Pathfinding**: Integrate the existing CUDA kernel from `src/utils/sssp_compact.cu` into the `SemanticProcessorActor`.
        *   [ ] **GPU-Accelerated Community Detection**: Integrate the existing `clustering_actor.rs` and `gpu_clustering_kernels.cu` into the `SemanticProcessorActor` to provide a unified interface for graph analytics.
        *   [ ] **Inference Engine (Initial Integration)**: Integrate the `whelk-rs` crate. Implement a basic capability to load an ontology and infer new `SubClassOf` relationships. This will establish the foundation for more advanced reasoning in the future.
        *   [ ] **Caching**: Add a caching layer for the results of expensive analysis operations.