---
layout: default
title: "System Architecture"
nav_order: 20
parent: Architecture
has_children: true
permalink: /explanations/architecture/
---

# System Architecture

Deep dives into VisionFlow's architecture, design patterns, and technical decisions.

## Architecture Sections

| Section | Description |
|---------|-------------|
| [Core Components](core/) | Client, Server, and Visualisation architecture |
| [Ports (Hexagonal)](ports/01-overview.md) | Interface definitions and contracts |
| [GPU Acceleration](gpu/) | CUDA kernels and communication flow |
| [Physics Engine](semantic-physics-system.md) | Force-directed layouts and semantic forces |
| [Ontology System](ontology-storage-architecture.md) | OWL storage and reasoning pipeline |
| [Decisions](decisions/) | Architecture Decision Records (ADRs) |

## Key Documents

### Core Architecture

- [Hexagonal CQRS](hexagonal-cqrs.md) - Ports and adapters with CQRS
- [Data Flow Complete](data-flow-complete.md) - End-to-end pipeline
- [Integration Patterns](integration-patterns.md) - System integration strategies
- [Services Architecture](services-architecture.md) - Business logic layer

### Database

- [Database Architecture](database-architecture.md) - Neo4j schema and queries
- [Adapter Patterns](adapter-patterns.md) - Repository implementations
- [RuVector Integration](ruvector-integration.md) - 150x faster vector search

### Physics and Visualisation

- [Semantic Physics System](semantic-physics-system.md) - Force-directed layouts
- [GPU Semantic Forces](gpu-semantic-forces.md) - 39 CUDA kernels
- [Stress Majorization](stress-majorization.md) - Graph layout algorithm

### Advanced Topics

- [Multi-Agent System](multi-agent-system.md) - AI agent coordination
- [XR Immersive System](xr-immersive-system.md) - Quest 3 WebXR
- [Analytics Visualisation](analytics-visualization.md) - UI/UX patterns

## Related Documentation

- [Unified Services Guide](services-layer.md)
- [GPU Architecture Documentation](gpu/README.md)
- [Semantic Physics Architecture](semantic-physics.md)
- [Stress Majorization for GPU-Accelerated Graph Layout](stress-majorization.md)
- [Integration Patterns in VisionFlow](integration-patterns.md)
