---
layout: default
title: Architecture
parent: Explanations
nav_order: 1
has_children: true
permalink: /explanations/architecture/
---

# Architecture Documentation

System architecture, design decisions, and technical analysis for VisionFlow.

## Overview

This section documents the complete system architecture including:

- Core server and client components
- GPU-accelerated visualization pipelines
- Database and storage patterns
- CQRS and hexagonal architecture implementation

## Subsections

| Subsection | Description |
|------------|-------------|
| [Core](./core/) | Server, client, and visualization core |
| [Components](./components/) | WebSocket and protocol components |
| [Decisions](./decisions/) | Architecture Decision Records (ADRs) |
| [GPU](./gpu/) | GPU acceleration and WASM compute |
| [Ports](./ports/) | Port interfaces for hexagonal architecture |

## Key Documents

- [Hexagonal CQRS](./hexagonal-cqrs.md) - Core architectural pattern
- [Data Flow](./data-flow-complete.md) - Complete data flow documentation
- [Services Layer](./services-layer.md) - Unified services architecture
- [Integration Patterns](./integration-patterns.md) - System integration patterns

## Related Documentation

- [Unified Services Guide](services-layer.md)
- [GPU Architecture Documentation](gpu/)
- [Semantic Physics Architecture](semantic-physics.md)
- [Stress Majorization for GPU-Accelerated Graph Layout](stress-majorization.md)
- [Integration Patterns in VisionFlow](integration-patterns.md)
