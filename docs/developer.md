---
layout: default
title: For Developers
nav_order: 60
permalink: /developer/
---

# Developer Hub

Everything you need to build features and contribute to VisionFlow.

## Quick Start

1. [Development Setup](/guides/developer/01-development-setup.md) - Environment setup
2. [Project Structure](/guides/developer/02-project-structure.md) - Code organisation
3. [Adding Features](/guides/developer/04-adding-features.md) - Build your first feature
4. [Testing Guide](/guides/testing-guide.md) - Write and run tests

## Essential Reading

| Document | Purpose |
|----------|---------|
| [Developer Journey](/DEVELOPER_JOURNEY.md) | Complete codebase learning path |
| [Architecture Overview](/ARCHITECTURE_OVERVIEW.md) | System design and patterns |
| [Hexagonal CQRS](/explanations/architecture/hexagonal-cqrs.md) | Ports and adapters pattern |

## By Technology

### Rust Backend

- [Server Architecture](/explanations/architecture/core/server.md) - 21 actors, ports/adapters
- [Actor System Guide](/guides/architecture/actor-system.md) - Actor patterns
- [Services Architecture](/explanations/architecture/services-architecture.md) - Business logic

### React Frontend

- [Client Architecture](/explanations/architecture/core/client.md) - React Three.js
- [State Management](/guides/client/state-management.md) - Zustand patterns
- [Three.js Rendering](/guides/client/three-js-rendering.md) - 3D visualisation

### Neo4j Database

- [Database Architecture](/explanations/architecture/database-architecture.md) - Schema design
- [Neo4j Integration](/guides/neo4j-integration.md) - CRUD operations
- [Database Schemas](/reference/database/schemas.md) - Data models

### GPU/CUDA

- [GPU Semantic Forces](/explanations/architecture/gpu-semantic-forces.md) - 39 kernels
- [GPU Optimisations](/explanations/architecture/gpu/optimizations.md) - Performance tuning
- [Performance Benchmarks](/reference/performance-benchmarks.md) - Metrics

## API Reference

- [REST API Complete](/reference/api/rest-api-complete.md) - HTTP endpoints
- [WebSocket API](/reference/api/03-websocket.md) - Real-time protocol
- [Binary WebSocket](/reference/protocols/binary-websocket.md) - 36-byte format

## Contributing

- [Contributing Guide](/guides/developer/06-contributing.md) - Code style, PRs
- [Development Workflow](/guides/development-workflow.md) - Git, CI/CD
- [Test Execution](/guides/developer/test-execution.md) - Running tests

## Learning Paths

### Week 1-2: Foundations

1. Set up development environment
2. Understand project structure
3. Run the full test suite
4. Read the architecture overview

### Week 3-4: First Contribution

1. Pick a good first issue
2. Implement following patterns
3. Write comprehensive tests
4. Submit pull request

### Month 2+: Deep Expertise

1. Master hexagonal architecture
2. Understand actor system
3. Contribute to core features
4. Review others' contributions
