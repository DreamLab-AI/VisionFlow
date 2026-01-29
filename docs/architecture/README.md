---
title: Architecture Documentation
description: Comprehensive architecture documentation and ADRs for VisionFlow
category: explanation
diataxis: explanation
tags:
  - architecture
  - adr
  - design
updated-date: 2025-01-29
---

# Architecture Documentation

Comprehensive architecture documentation for VisionFlow/TurboFlow.

## Architectural Decision Records (ADRs)

See the [ADR Index](adr/README.md) for all architectural decisions:

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-0001](adr/ADR-0001-neo4j-persistent-with-filesystem-sync.md) | Neo4j Persistent with Filesystem Sync | Accepted |

## Overview

- **[Architecture Overview](overview.md)** - Complete system architecture
- **[Technology Choices](technology-choices.md)** - Technology stack decisions
- **[Developer Journey](developer-journey.md)** - Getting started with the codebase
- **[Data Flow](data-flow.md)** - Complete data flow through the system
- **[Database Architecture](database.md)** - Database design and patterns
- **[Services Architecture](services.md)** - Service layer design

## Core Components

### Client
- **[Client Overview](client/overview.md)** - Client-side architecture

### Server
- **[Server Overview](server/overview.md)** - Server-side architecture

## Patterns & Design

### Architectural Patterns
- **[Hexagonal CQRS](patterns/hexagonal-cqrs.md)** - Hexagonal architecture with CQRS
- **[Hexagonal Status](HEXAGONAL_ARCHITECTURE_STATUS.md)** - Implementation status

### Ports & Adapters
- **[Ports Overview](ports/01-overview.md)** - Ports and adapters overview
- **[Settings Repository](ports/02-settings-repository.md)** - Settings port
- **[Knowledge Graph Repository](ports/03-knowledge-graph-repository.md)** - Graph port
- **[Ontology Repository](ports/04-ontology-repository.md)** - Ontology port
- **[Inference Engine](ports/05-inference-engine.md)** - Inference port
- **[GPU Physics Adapter](ports/06-gpu-physics-adapter.md)** - GPU physics port
- **[GPU Semantic Analyzer](ports/07-gpu-semantic-analyzer.md)** - GPU semantic port

## Domain-Specific Architecture

### Agents
- **[Multi-Agent System](agents/multi-agent.md)** - Multi-agent coordination

### GPU Computing
- **[GPU Overview](gpu/README.md)** - GPU architecture overview
- **[Communication Flow](gpu/communication-flow.md)** - GPU communication patterns
- **[Optimizations](gpu/optimizations.md)** - GPU performance optimizations

### Ontology
- **[Client-Side Hierarchical LOD](ontology/client-side-hierarchical-lod.md)** - Level of detail
- **[Enhanced Parser](ontology/enhanced-parser.md)** - Ontology parsing
- **[Hierarchical Visualization](ontology/hierarchical-visualization.md)** - Hierarchy display
- **[Intelligent Pathfinding](ontology/intelligent-pathfinding-system.md)** - Pathfinding
- **[Neo4j Integration](ontology/neo4j-integration.md)** - Graph database
- **[Pipeline Integration](ontology/ontology-pipeline-integration.md)** - Processing pipeline
- **[Typed System](ontology/ontology-typed-system.md)** - Type system
- **[Reasoning Engine](ontology/reasoning-engine.md)** - Inference reasoning

### Physics Simulation
- **[Semantic Forces](physics/semantic-forces.md)** - Force-based layout
- **[Semantic Forces Actor](physics/semantic-forces-actor.md)** - Actor-based physics

## Integration & Infrastructure

### Integrations
- **[Blender MCP](blender-mcp-unified-architecture.md)** - Blender integration
- **[SOLID Sidecar](solid-sidecar-architecture.md)** - SOLID pod architecture
- **[User Agent Pod Design](user-agent-pod-design.md)** - User agent design

### Protocols
See [protocols/](protocols/) for protocol specifications.

### Pipelines
See [pipelines/](pipelines/) for data pipeline documentation.

## Architecture Decisions

- **[Decisions](decisions/)** - Architecture Decision Records (ADRs)
- **[Protocol Matrix](PROTOCOL_MATRIX.md)** - Communication protocols

## Visualization

- **[Visualization](visualization/)** - Visualization architecture
- **[XR](xr/)** - Extended reality components

## Research & Analysis

- **[Research](research/)** - Architecture research
- **[VisionFlow Assessment](visionflow-distributed-systems-assessment.md)** - System assessment
- **[Vircadia Analysis](VIRCADIA_BABYLON_CONSOLIDATION_ANALYSIS.md)** - Engine analysis

## Skills & Classification

- **[Skill MCP Classification](skill-mcp-classification.md)** - Skill categorization
- **[Skills Refactoring Plan](skills-refactoring-plan.md)** - Refactoring roadmap

## Migrations

See [migrations/](migrations/) for migration documentation.

## Diagrams

See [diagrams/](diagrams/) for architecture diagrams.

---

## Quick Links

| Document | Description |
|----------|-------------|
| [Overview](overview.md) | Start here for system overview |
| [Technology Choices](technology-choices.md) | Tech stack rationale |
| [Hexagonal CQRS](patterns/hexagonal-cqrs.md) | Core architecture pattern |
| [Services](services.md) | Service layer details |
| [Data Flow](data-flow.md) | End-to-end data flow |
