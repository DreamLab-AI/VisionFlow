---
layout: default
title: Architecture
nav_order: 20
has_children: true
permalink: /explanations/
---

# Architecture Documentation

Understanding-oriented documentation explaining VisionFlow's design, patterns, and technical decisions.

## Overview

VisionFlow uses a **hexagonal architecture** (ports and adapters) combined with **CQRS** (Command Query Responsibility Segregation) to achieve clean separation of concerns, testability, and flexibility.

## Sections

| Section | Description | Documents |
|---------|-------------|-----------|
| [System Overview](system-overview.md) | Architectural blueprint | 1 |
| [System Architecture](architecture/) | Core patterns and design | 46 |
| [Ontology System](ontology/) | OWL processing and reasoning | 8 |
| [Physics Engine](physics/) | Semantic forces and layouts | 2 |

## Key Concepts

### Hexagonal Architecture

VisionFlow separates business logic from infrastructure through ports (interfaces) and adapters (implementations):

- **Ports**: Interfaces defining contracts
- **Adapters**: Implementations of those contracts
- **Core**: Business logic independent of infrastructure

### CQRS Pattern

Commands and queries are handled separately:

- **Commands**: Write operations that change state
- **Queries**: Read operations that return data

### Actor Model

The backend uses Actix actors for concurrent, isolated processing:

- 21 specialised actors
- Message-based communication
- Fault tolerance through supervision

## Learning Path

### Beginner

1. [System Overview](system-overview.md) - Start here
2. [Hexagonal CQRS](architecture/hexagonal-cqrs.md) - Core pattern
3. [Data Flow Complete](architecture/data-flow-complete.md) - End-to-end flow

### Intermediate

1. [Services Architecture](architecture/services-architecture.md) - Business logic
2. [Database Architecture](architecture/database-architecture.md) - Neo4j design
3. [Integration Patterns](architecture/integration-patterns.md) - System integration

### Advanced

1. [Ports Overview](architecture/ports/01-overview.md) - Interface contracts
2. [GPU Semantic Forces](architecture/gpu-semantic-forces.md) - CUDA kernels
3. [Ontology Reasoning Pipeline](architecture/ontology-reasoning-pipeline.md) - Inference

## Quick Access

| Document | Purpose |
|----------|---------|
| [System Overview](system-overview.md) | Complete architectural blueprint |
| [Hexagonal CQRS](architecture/hexagonal-cqrs.md) | Core architectural pattern |
| [Semantic Forces](architecture/semantic-forces-system.md) | Physics-based layouts |
| [Multi-Agent System](architecture/multi-agent-system.md) | AI agent coordination |
| [XR Immersive System](architecture/xr-immersive-system.md) | Quest 3 WebXR |

## Related Documentation

- [Architecture Overview](/ARCHITECTURE_OVERVIEW/) - High-level summary
- [Technology Choices](/TECHNOLOGY_CHOICES/) - Stack rationale
- [Developer Journey](/DEVELOPER_JOURNEY/) - Codebase learning path
