---
title: VisionFlow Architecture Cross-Reference Matrix
description: This matrix shows how different architectural components relate to each other across the documentation corpus.
category: reference
tags:
  - architecture
  - structure
  - api
  - api
  - api
related-docs:
  - diagrams/README.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Neo4j database
---

# VisionFlow Architecture Cross-Reference Matrix

## Component Interaction Map

This matrix shows how different architectural components relate to each other across the documentation corpus.

## Primary Component Dependencies

| Component | Depends On | Referenced In | Key Interactions |
|-----------|------------|---------------|------------------|
| **GraphCanvas** | WebSocketService, GraphManager, BotsVisualization | threejs-pipeline, state-management, binary-protocol | Renders 3D graph, receives position updates |
| **WebSocketService** | BinaryProtocol, BatchQueue, ValidationMiddleware | binary-protocol, data-flows, state-management | Handles real-time communication |
| **GraphStateActor** | Neo4j, KnowledgeGraphRepository | actor-system, neo4j-architecture, data-flows | Maintains in-memory graph state |
| **PhysicsOrchestratorActor** | GPU kernels, GraphStateActor | actor-system, cuda-architecture, data-flows | Coordinates physics simulation |
| **ClientCoordinatorActor** | WebSocketService, GraphStateActor | actor-system, binary-protocol, data-flows | Manages client connections |
| **SemanticProcessorActor** | OntologyRepository, InferenceEngine | actor-system, neo4j-architecture | Handles reasoning |
| **Management API** | ProcessManager, Metrics | backend-api, rest-api, agent-system | Task orchestration |
| **SettingsStore** | AutoSaveManager, SettingsAPI | state-management, data-flows, rest-api | Configuration management |
| **Three.js Renderer** | GraphCanvas, SelectiveBloom | threejs-pipeline, xr-architecture | WebGL rendering |
| **Quest 3 XR** | XRCoreProvider, WebXR API | xr-architecture, threejs-pipeline | VR/AR support |

## Data Flow Dependencies

| Flow Type | Source | Destination | Protocol | Documents |
|-----------|--------|-------------|----------|-----------|
| **Graph Updates** | Neo4j → GraphStateActor → ClientCoordinator → Browser | Binary WebSocket | data-flows, binary-protocol |
| **Settings Changes** | UI → SettingsStore → API → Neo4j | REST/JSON | state-management, rest-api |
| **Physics Simulation** | GraphStateActor → GPU → ClientCoordinator | CUDA/Binary | cuda-architecture, actor-system |
| **Agent State** | Management API → ProcessManager → WebSocket | JSON/Binary | agent-system, data-flows |
| **Voice Commands** | Browser → WebSocket → STT → Command → TTS | Binary Audio | data-flows, binary-protocol |
| **GitHub Sync** | GitHub API → Neo4j → GraphStateActor | REST/Cypher | data-flows, neo4j-architecture |

## Technology Stack Cross-References

| Technology | Primary Docs | Integration Points | Related Docs |
|------------|--------------|-------------------|--------------|
| **React 18** | state-management | Zustand, Three.js | threejs-pipeline, xr-architecture |
| **Three.js** | threejs-pipeline | React Three Fiber, WebGL | state-management, xr-architecture |
| **Rust/Actix** | actor-system | WebSocket, Neo4j | rest-api, binary-protocol |
| **Neo4j 5.13** | neo4j-architecture | Cypher, neo4rs | actor-system, data-flows |
| **CUDA 12.4** | cuda-architecture | cudarc, kernels | actor-system, data-flows |
| **WebSocket** | binary-protocol | Actix-web, client | data-flows, state-management |
| **WebXR** | xr-architecture | Three.js, Quest 3 | threejs-pipeline |
| **Fastify** | backend-api | Management API | rest-api, agent-system |

## Architecture Pattern Usage

| Pattern | Implementation | Location | Related Components |
|---------|----------------|----------|-------------------|
| **Hexagonal Architecture** | Ports & Adapters | actor-system | Repository interfaces, Neo4j adapters |
| **Actor Model** | Actix actors | actor-system | 21 specialized actors |
| **Event Sourcing** | WebSocket streams | binary-protocol | Graph updates, agent state |
| **Observer Pattern** | Zustand subscriptions | state-management | React components |
| **Repository Pattern** | Neo4j repositories | neo4j-architecture | Graph, Ontology, Settings |
| **Strategy Pattern** | Rendering modes | threejs-pipeline | LOD, instancing |
| **Circuit Breaker** | Error recovery | agent-system | Retry logic |
| **Factory Pattern** | Agent creation | agent-system | ProcessManager |

## Performance Optimization Cross-References

| Optimization | Technique | Impact | Documentation |
|--------------|-----------|--------|---------------|
| **Rendering** | Instance rendering | 10,000+ nodes @ 60 FPS | threejs-pipeline |
| **Network** | Binary protocol | 90% bandwidth reduction | binary-protocol |
| **Compute** | GPU acceleration | 100x speedup | cuda-architecture |
| **State** | Lazy loading | Reduced memory | state-management |
| **API** | Request batching | Lower latency | rest-api |
| **Database** | Connection pooling | Higher throughput | neo4j-architecture |

## System Boundaries

| Boundary | Interface | Protocol | Documentation |
|----------|-----------|----------|---------------|
| **Client ↔ Server** | WebSocket, REST API | Binary, JSON | binary-protocol, rest-api |
| **Server ↔ Database** | Neo4j driver | Bolt protocol | neo4j-architecture |
| **Server ↔ GPU** | CUDA runtime | Memory transfer | cuda-architecture |
| **Client ↔ XR Device** | WebXR API | Browser API | xr-architecture |
| **Server ↔ GitHub** | GitHub API | REST/JSON | data-flows |
| **Server ↔ AI** | LLM APIs | REST/JSON | data-flows |

## Message Type Matrix

| Message Category | Type Count | Protocols | Actors Involved | Documentation |
|-----------------|------------|-----------|-----------------|---------------|
| **Graph Operations** | 15+ | Binary, Actor messages | GraphStateActor, ClientCoordinator | actor-system, binary-protocol |
| **Physics Updates** | 8+ | Binary, CUDA | PhysicsOrchestratorActor, GPU actors | cuda-architecture, actor-system |
| **Agent Control** | 12+ | JSON, Binary | Management API, ProcessManager | agent-system, rest-api |
| **Settings** | 10+ | JSON | SettingsStore, API | state-management, rest-api |
| **WebSocket Control** | 6+ | Binary | WebSocketService, ClientCoordinator | binary-protocol |

## Dependency Depth Analysis

| Level | Components | Dependencies |
|-------|------------|--------------|
| **Level 0 (Core)** | Neo4j, GPU | None (external) |
| **Level 1 (Infrastructure)** | Repositories, CUDA runtime | Level 0 |
| **Level 2 (Services)** | Actors, Management API | Level 0-1 |
| **Level 3 (Coordination)** | ClientCoordinator, WebSocket | Level 0-2 |
| **Level 4 (Presentation)** | React, Three.js | Level 0-3 |
| **Level 5 (Features)** | XR, Voice, Analytics | Level 0-4 |

---

---

## Related Documentation

- [ASCII Diagram Deprecation Audit](../audits/ascii-diagram-deprecation-audit.md)
- [ComfyUI Management API Integration - Summary](../comfyui-management-api-integration-summary.md)
- [VisionFlow Client Architecture - Deep Analysis](../archive/analysis/client-architecture-analysis-2025-12.md)
- [VisionFlow Client Architecture](../concepts/architecture/core/client.md)
- [WebSocket Binary Protocol - Complete System Documentation](infrastructure/websocket/binary-protocol-complete.md)

## Critical Path Analysis

| Operation | Path | Latency | Bottleneck |
|-----------|------|---------|------------|
| **Node Update** | User → React → WebSocket → Actor → GPU → Broadcast | 25ms | GPU compute |
| **Settings Save** | UI → Store → API → Neo4j → Broadcast | 80ms | Neo4j write |
| **GitHub Sync** | API → Parse → Neo4j → Actor → GPU → Client | 700ms | GitHub API |
| **Voice Command** | Audio → STT → Process → TTS → Audio | 2000ms | STT processing |
| **XR Render** | Scene → Stereo → Submit → Display | 11ms | GPU submission |

---

*This cross-reference matrix provides navigation between related architectural components across all documentation. Use it to understand system-wide implications of changes and to trace data flows through multiple subsystems.*