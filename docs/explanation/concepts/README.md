---
title: Core Concepts
description: Foundational concepts and mental models for understanding VisionFlow
category: explanation
diataxis: explanation
tags:
  - concepts
  - architecture
  - learning
updated-date: 2025-01-29
---

# Core Concepts

This section explains the foundational concepts and mental models needed to understand VisionFlow's architecture and design decisions.

## Purpose

Understanding these concepts will help you:
- Grasp the "why" behind architectural decisions
- Navigate the codebase more effectively
- Contribute meaningful improvements
- Debug complex issues

## Contents

### Physics and Simulation
- [Physics Engine](physics-engine.md) - Force-directed graph simulation with real-time physics
- [Constraint System](constraint-system.md) - LOD-aware constraint management and GPU acceleration
- [GPU Acceleration](gpu-acceleration.md) - WGPU compute shaders for parallel force calculations

### Knowledge Representation
- [Ontology Reasoning](ontology-reasoning.md) - OWL/RDF parsing and semantic inference
- [Actor Model](actor-model.md) - Concurrent actor-based architecture patterns

### System Architecture
- [Hexagonal Architecture](hexagonal-architecture.md) - Ports and adapters pattern for clean separation
- [Multi-Agent System](multi-agent-system.md) - Client coordination and consensus
- [Real-Time Sync](real-time-sync.md) - WebSocket binary protocols and delta encoding

## Related Sections

- [Architecture](../architecture/README.md) - Technical architecture and ADRs
- [Reference](../../reference/README.md) - API and configuration reference
- [How-To Guides](../../how-to/README.md) - How-to guides for common tasks
