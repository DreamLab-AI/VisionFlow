---
layout: default
title: For Architects
nav_order: 61
permalink: /architect/
---

# Architect Hub

System design, architecture decisions, and technical deep dives.

## Quick Start

1. [Architecture Overview]($1.md) - Complete system architecture
2. [Technology Choices]($1.md) - Stack rationale
3. [System Overview]($1.md) - Architectural blueprint
4. [Hexagonal CQRS]($1.md) - Core pattern

## Architecture Deep Dives

### System Design

| Document | Focus |
|----------|-------|
| [Data Flow Complete]($1.md) | End-to-end pipeline |
| [Integration Patterns]($1.md) | System integration |
| [Services Architecture]($1.md) | Business logic layer |
| [Adapter Patterns]($1.md) | Repository implementations |

### Hexagonal Architecture

| Document | Focus |
|----------|-------|
| [Ports Overview]($1.md) | Interface contracts |
| [Knowledge Graph Repository]($1.md) | Graph operations |
| [Ontology Repository]($1.md) | OWL storage |
| [GPU Physics Adapter]($1.md) | Physics computation |

### Component Architecture

| Component | Documents |
|-----------|-----------|
| **Server** | [Server Architecture]($1.md), [Actor System]($1.md) |
| **Client** | [Client Architecture]($1.md), [State Management]($1.md) |
| **Database** | [Database Architecture]($1.md), [Schemas]($1.md) |
| **GPU** | [GPU Semantic Forces]($1.md), [Optimisations]($1.md) |

### Domain Architecture

| Domain | Documents |
|--------|-----------|
| **Ontology** | [Storage Architecture]($1.md), [Reasoning Pipeline]($1.md) |
| **Physics** | [Semantic Physics System]($1.md), [Stress Majorisation]($1.md) |
| **Multi-Agent** | [Multi-Agent System]($1.md), [Agent Orchestration]($1.md) |
| **XR** | [XR Immersive System]($1.md), [XR Integration]($1.md) |

## Architecture Decisions

### ADRs

- [ADR-0001: Neo4j Persistence]($1.md) - Database strategy

### Key Decisions

| Decision | Rationale |
|----------|-----------|
| Hexagonal + CQRS | Clean separation, testability, flexibility |
| Actor Model (Actix) | Concurrency, isolation, fault tolerance |
| Neo4j Graph Database | Native graph queries, relationship-first design |
| CUDA GPU Acceleration | Real-time physics for large graphs |
| WebXR | Cross-platform VR without app stores |

## Quality Attributes

### Performance

- [Performance Benchmarks]($1.md)
- [GPU Optimisations]($1.md)
- [RuVector Integration]($1.md) (150x faster search)

### Security

- [Security Guide]($1.md)
- [Authentication]($1.md)

### Scalability

- [Multi-Agent System]($1.md)
- [Pipeline Integration]($1.md)

## Evaluation Matrix

### Technology Stack

| Layer | Choice | Alternatives Considered |
|-------|--------|------------------------|
| Backend | Rust + Actix | Go, Node.js |
| Frontend | React + Three.js | Vue, Babylon.js |
| Database | Neo4j | PostgreSQL, MongoDB |
| GPU | CUDA | OpenCL, WebGPU |
| Protocol | Binary WebSocket | JSON, gRPC |
