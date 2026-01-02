---
layout: default
title: Concepts
nav_order: 4
has_children: true
description: Core architectural concepts and design documentation for VisionFlow
---

# Concepts

Core architectural concepts and design documentation for VisionFlow.

## Architecture

Detailed documentation of VisionFlow's client and server architecture.

### Core Architecture

| Document | Description |
|----------|-------------|
| [Client Architecture](architecture/core/client.md) | High-performance 3D visualization platform with React Three Fiber, instanced rendering, and WebSocket synchronization |
| [Server Architecture](architecture/core/server.md) | Hexagonal (ports and adapters) architecture with CQRS patterns and actor-based concurrency |

## Key Architectural Patterns

### Client Architecture Highlights

- **Instanced Rendering**: Single draw call for 10,000+ nodes
- **Binary Protocol**: 34-byte node format (90% bandwidth reduction)
- **Lazy Loading**: 87% faster initial load
- **XR Support**: Quest 3 AR/VR with Babylon.js fallback

### Server Architecture Highlights

- **21 Specialized Actors**: Supervised actor hierarchy for concurrent operations
- **9 Port Interfaces**: Technology-agnostic domain boundaries
- **12 Adapters**: Concrete implementations (Neo4j, GPU, Actix)
- **CQRS Layer**: Separate read/write operations for scalability

## Related Documentation

- [API Reference](/reference/api/) - HTTP and WebSocket endpoints
- [Guides](/guides/) - Implementation and development guides
- [Explanations](/explanations/) - In-depth architectural explanations
