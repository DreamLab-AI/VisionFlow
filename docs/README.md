---
title: VisionFlow Documentation
description: Complete documentation for VisionFlow - enterprise-grade multi-agent knowledge graphing
category: reference
updated-date: 2026-01-29
---

# VisionFlow Documentation

Enterprise-grade multi-agent knowledge graphing with 3D visualization, semantic reasoning, and GPU-accelerated physics. This documentation follows the [Diataxis framework](https://diataxis.fr/) for maximum discoverability.

## Quick Start

Get running in 5 minutes:

1. **[Installation](tutorials/installation.md)** - Docker or native setup
2. **[First Graph](tutorials/creating-first-graph.md)** - Create your first visualization
3. **[Navigation Guide](how-to/navigation-guide.md)** - Master the 3D interface

## Documentation by Role

<details>
<summary><strong>New Users</strong> - Getting started with VisionFlow</summary>

### Your Learning Path

| Step | Document | Time |
|------|----------|------|
| 1 | [What is VisionFlow?](tutorials/overview.md) | 10 min |
| 2 | [Installation](tutorials/installation.md) | 15 min |
| 3 | [First Graph](tutorials/creating-first-graph.md) | 20 min |
| 4 | [Navigation Guide](how-to/navigation-guide.md) | 15 min |
| 5 | [Configuration](how-to/operations/configuration.md) | 10 min |

### Next Steps

- [Neo4j Quick Start](tutorials/neo4j-basics.md) - Query the graph database
- [Natural Language Queries](how-to/features/natural-language-queries.md) - Ask questions in plain English
- [Troubleshooting](how-to/operations/troubleshooting.md) - Common issues and solutions

</details>

<details>
<summary><strong>Developers</strong> - Building and extending VisionFlow</summary>

### Onboarding Path

| Priority | Document | Focus |
|----------|----------|-------|
| High | [Developer Journey](explanation/architecture/developer-journey.md) | Codebase learning path |
| High | [Development Setup](how-to/development/01-development-setup.md) | IDE and environment |
| High | [Project Structure](how-to/development/02-project-structure.md) | Code organisation |
| Medium | [Architecture Overview](explanation/architecture/README.md) | System design |
| Medium | [Adding Features](how-to/development/04-adding-features.md) | Development workflow |
| Standard | [Testing Guide](how-to/development/testing-guide.md) | Unit, integration, E2E |

### By Technology

- **Rust Backend** - [Server Architecture](explanation/architecture/server/overview.md), [Hexagonal CQRS](explanation/architecture/patterns/hexagonal-cqrs.md)
- **React Frontend** - [Client Architecture](explanation/architecture/client/overview.md), [State Management](how-to/development/state-management.md)
- **Neo4j** - [Database Architecture](explanation/architecture/database.md), [Schemas](reference/database/schemas.md)
- **GPU/CUDA** - [GPU Overview](explanation/architecture/gpu/README.md), [Optimisations](explanation/architecture/gpu/optimizations.md)
- **WebSocket** - [Binary Protocol](reference/protocols/binary-websocket.md), [Best Practices](how-to/development/websocket-best-practices.md)

### API Reference

- [REST API](reference/api/rest-api-complete.md)
- [WebSocket API](reference/api/03-websocket.md)
- [Authentication](reference/api/01-authentication.md)

</details>

<details>
<summary><strong>Architects</strong> - System design and patterns</summary>

### Architecture Path

| Document | Focus |
|----------|-------|
| [Architecture Overview](explanation/architecture/README.md) | Complete system architecture |
| [Technology Choices](explanation/architecture/technology-choices.md) | Stack rationale |
| [System Overview](explanations/system-overview.md) | Architectural blueprint |
| [Hexagonal CQRS](explanation/architecture/patterns/hexagonal-cqrs.md) | Ports and adapters |
| [Data Flow](explanation/architecture/data-flow.md) | End-to-end pipeline |
| [Integration Patterns](explanations/architecture/integration-patterns.md) | System integration |

### Deep Dives

- **Actor System** - [Actor Guide](how-to/development/actor-system.md), [Server Architecture](explanation/architecture/server/overview.md)
- **Database** - [Database Architecture](explanation/architecture/database.md), [Neo4j ADR](explanation/architecture/adr/ADR-0001-neo4j-persistent-with-filesystem-sync.md)
- **Physics** - [Semantic Physics](explanation/architecture/physics/semantic-forces.md), [GPU Communication](explanation/architecture/gpu/communication-flow.md)
- **Ontology** - [Ontology Storage](explanations/architecture/ontology-storage-architecture.md), [Reasoning Pipeline](explanation/architecture/ontology/reasoning-engine.md)
- **Multi-Agent** - [Multi-Agent System](explanation/architecture/agents/multi-agent.md), [Agent Orchestration](how-to/agents/agent-orchestration.md)

### Hexagonal Architecture Ports

- [Ports Overview](reference/architecture/ports/01-overview.md)
- [Knowledge Graph Repository](reference/architecture/ports/03-knowledge-graph-repository.md)
- [Ontology Repository](reference/architecture/ports/04-ontology-repository.md)
- [Inference Engine](reference/architecture/ports/05-inference-engine.md)
- [GPU Physics Adapter](reference/architecture/ports/06-gpu-physics-adapter.md)

</details>

<details>
<summary><strong>Operators</strong> - Deployment and operations</summary>

### Operations Path

| Document | Purpose |
|----------|---------|
| [Deployment Guide](how-to/deployment/deployment.md) | Production deployment |
| [Docker Compose](how-to/deployment/docker-compose-guide.md) | Container orchestration |
| [Operator Runbook](how-to/operations/pipeline-operator-runbook.md) | Operations playbook |
| [Configuration](how-to/operations/configuration.md) | Environment variables |
| [Security](how-to/operations/security.md) | Authentication and secrets |
| [Telemetry](how-to/operations/telemetry-logging.md) | Observability |

### Infrastructure

- [Infrastructure Architecture](how-to/infrastructure/architecture.md)
- [Docker Environment](how-to/infrastructure/docker-environment.md)
- [Port Configuration](how-to/infrastructure/port-configuration.md)
- [Infrastructure Troubleshooting](how-to/infrastructure/troubleshooting.md)

### Data Operations

- [Neo4j Migration](how-to/integration/neo4j-migration.md)
- [Pipeline Admin API](how-to/operations/pipeline-admin-api.md)
- [GitHub Sync Service](explanations/architecture/github-sync-service-design.md)

</details>

## Documentation Structure

```mermaid
graph TB
    subgraph Entry["Entry Points"]
        README["README.md"]
        OVERVIEW["getting-started/overview.md"]
    end

    subgraph Learning["Learning (Tutorials)"]
        T1["Installation"]
        T2["First Graph"]
        T3["Neo4j Quick Start"]
    end

    subgraph Tasks["Task-Oriented (Guides)"]
        G1["Features"]
        G2["Developer"]
        G3["Infrastructure"]
        G4["Operations"]
    end

    subgraph Understanding["Understanding (Explanations)"]
        E1["Architecture"]
        E2["Ontology"]
        E3["Physics"]
    end

    subgraph Lookup["Lookup (Reference)"]
        R1["API"]
        R2["Database"]
        R3["Protocols"]
    end

    README --> Learning
    README --> Tasks
    README --> Understanding
    README --> Lookup
    OVERVIEW --> Learning

    Learning --> Tasks
    Tasks --> Understanding
    Understanding --> Lookup

    style README fill:#4A90E2,color:#fff
    style Learning fill:#7ED321,color:#fff
    style Tasks fill:#F5A623,color:#000
    style Understanding fill:#BD10E0,color:#fff
    style Lookup fill:#9013FE,color:#fff
```

## Quick Links

| Task | Document |
|------|----------|
| **Install VisionFlow** | [Installation](tutorials/installation.md) |
| **Create first graph** | [First Graph](tutorials/creating-first-graph.md) |
| **Deploy AI agents** | [Agent Orchestration](how-to/agents/agent-orchestration.md) |
| **Query Neo4j** | [Neo4j Integration](how-to/integration/neo4j-integration.md) |
| **Add a feature** | [Adding Features](how-to/development/04-adding-features.md) |
| **Set up XR/VR** | [Vircadia XR Guide](how-to/features/vircadia-xr-complete-guide.md) |
| **Understand architecture** | [Architecture Overview](explanation/architecture/README.md) |
| **Learn the codebase** | [Developer Journey](explanation/architecture/developer-journey.md) |
| **Deploy to production** | [Deployment Guide](how-to/deployment/deployment.md) |
| **Configure environment** | [Configuration](how-to/operations/configuration.md) |
| **Fix issues** | [Troubleshooting](how-to/operations/troubleshooting.md) |
| **Write tests** | [Testing Guide](how-to/development/testing-guide.md) |
| **Use REST API** | [REST API](reference/api/rest-api-complete.md) |
| **Use WebSocket API** | [WebSocket API](reference/api/03-websocket.md) |
| **Optimise performance** | [GPU Optimisations](explanation/architecture/gpu/optimizations.md) |
| **Secure the app** | [Security Guide](how-to/operations/security.md) |

## Documentation Categories

### Tutorials (Learning-Oriented)

Step-by-step lessons for beginners.

| Tutorial | Time | Description |
|----------|------|-------------|
| [Installation](tutorials/installation.md) | 10 min | Docker and native setup |
| [First Graph](tutorials/creating-first-graph.md) | 15 min | Create your first visualisation |
| [Neo4j Quick Start](tutorials/neo4j-basics.md) | 20 min | Query the graph database |

### Concepts (Understanding-Oriented)

Core mental models and foundational knowledge.

| Concept | Description |
|---------|-------------|
| [Core Concepts](explanation/concepts/README.md) | Overview of VisionFlow mental models |
| [Physics Engine](explanation/concepts/physics-engine.md) | Force-directed graph simulation |
| [Actor Model](explanation/concepts/actor-model.md) | Concurrent actor-based patterns |
| [Hexagonal Architecture](explanation/concepts/hexagonal-architecture.md) | Ports and adapters design |

### Guides (Task-Oriented)

Practical instructions for specific goals.

<details>
<summary>Core Features (8 guides)</summary>

- [Navigation Guide](how-to/navigation-guide.md) - 3D interface controls
- [Filtering Nodes](how-to/features/filtering-nodes.md) - Graph filtering
- [Intelligent Pathfinding](how-to/features/intelligent-pathfinding.md) - Graph traversal
- [Natural Language Queries](how-to/features/natural-language-queries.md) - Semantic search
- [Semantic Forces](how-to/features/semantic-forces.md) - Physics layouts
- [Configuration](how-to/operations/configuration.md) - Settings
- [Troubleshooting](how-to/operations/troubleshooting.md) - Common issues
- [Extending the System](how-to/development/extending-the-system.md) - Plugins

</details>

<details>
<summary>AI Agent System (4 guides)</summary>

- [Agent Orchestration](how-to/agents/agent-orchestration.md) - Deploy AI agents
- [Orchestrating Agents](how-to/agents/orchestrating-agents.md) - Coordination patterns
- [Multi-Agent Skills](how-to/agents/using-skills.md) - Agent capabilities
- [AI Models](how-to/ai-integration/README.md) - Model integrations

</details>

<details>
<summary>Developer Guides (8 guides)</summary>

- [Development Setup](how-to/development/01-development-setup.md) - Environment
- [Project Structure](how-to/development/02-project-structure.md) - Code organisation
- [Adding Features](how-to/development/04-adding-features.md) - Workflow
- [Contributing](how-to/development/06-contributing.md) - Code standards
- [WebSocket Best Practices](how-to/development/websocket-best-practices.md) - Real-time
- [JSON Serialisation](how-to/development/json-serialization-patterns.md) - Data formats
- [Test Execution](how-to/development/test-execution.md) - Running tests

</details>

<details>
<summary>Infrastructure and Operations (15 guides)</summary>

- [Deployment](how-to/deployment/deployment.md) - Production deployment
- [Docker Compose](how-to/deployment/docker-compose-guide.md) - Container orchestration
- [Docker Environment](how-to/deployment/docker-environment-setup.md) - Container config
- [Security](how-to/operations/security.md) - Auth and secrets
- [Telemetry](how-to/operations/telemetry-logging.md) - Observability
- [Operator Runbook](how-to/operations/pipeline-operator-runbook.md) - Operations
- [Infrastructure Architecture](how-to/infrastructure/architecture.md) - System design
- [Docker Environment](how-to/infrastructure/docker-environment.md) - Containers
- [Port Configuration](how-to/infrastructure/port-configuration.md) - Networking
- [Infrastructure Troubleshooting](how-to/infrastructure/troubleshooting.md) - Issues

</details>

### Explanations (Understanding-Oriented)

Deep dives into architecture and design.

<details>
<summary>System Architecture (20+ documents)</summary>

- [System Overview](explanations/system-overview.md) - Architectural blueprint
- [Hexagonal CQRS](explanation/architecture/patterns/hexagonal-cqrs.md) - Ports and adapters
- [Data Flow](explanation/architecture/data-flow.md) - End-to-end pipeline
- [Services Architecture](explanation/architecture/services.md) - Business logic
- [Multi-Agent System](explanation/architecture/agents/multi-agent.md) - AI coordination
- [Integration Patterns](explanations/architecture/integration-patterns.md) - System integration
- [Database Architecture](explanation/architecture/database.md) - Neo4j design

</details>

<details>
<summary>GPU and Physics (8 documents)</summary>

- [Semantic Physics System](explanations/architecture/semantic-physics-system.md) - Force layout
- [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md) - CUDA kernels
- [GPU Communication](explanation/architecture/gpu/communication-flow.md) - Data transfer
- [GPU Optimisations](explanation/architecture/gpu/optimizations.md) - Performance
- [Stress Majorisation](explanations/architecture/stress-majorization.md) - Layout algorithm

</details>

<details>
<summary>Ontology and Reasoning (11 documents)</summary>

- [Ontology Reasoning Pipeline](explanations/architecture/ontology-reasoning-pipeline.md) - Inference
- [Reasoning Engine](explanation/architecture/ontology/reasoning-engine.md) - Inference concepts
- [Ontology Storage](explanations/architecture/ontology-storage-architecture.md) - Neo4j persistence
- [Hierarchical Visualisation](explanation/architecture/ontology/hierarchical-visualization.md) - Tree layouts
- [Pathfinding System](explanation/architecture/ontology/intelligent-pathfinding-system.md) - Graph traversal

</details>

### Reference (Information-Oriented)

Technical specifications and APIs.

<details>
<summary>API Documentation (8 references)</summary>

- [API Complete](reference/api-complete-reference.md) - All endpoints
- [REST API](reference/api/rest-api-complete.md) - HTTP API spec
- [WebSocket API](reference/api/03-websocket.md) - Real-time protocol
- [Authentication](reference/api/01-authentication.md) - JWT and Nostr
- [Semantic Features API](reference/api/semantic-features-api.md) - NL queries

</details>

<details>
<summary>Database and Protocols (6 references)</summary>

- [Database Schemas](reference/database/schemas.md) - Neo4j schema
- [Ontology Schema V2](reference/database/ontology-schema-v2.md) - OWL schema
- [User Settings Schema](reference/database/user-settings-schema.md) - User data
- [Binary WebSocket](reference/protocols/binary-websocket.md) - 36-byte format
- [WebSocket Protocol](reference/websocket-protocol.md) - V2 spec

</details>

<details>
<summary>System Status (5 references)</summary>

- [Error Codes](reference/error-codes.md) - Error reference
- [Implementation Status](reference/implementation-status.md) - Feature matrix
- [Code Quality](reference/code-quality-status.md) - Build health
- [Performance Benchmarks](reference/performance-benchmarks.md) - GPU metrics
- [Physics Implementation](reference/physics-implementation.md) - Physics details

</details>

## Getting Help

| Issue Type | Resource |
|------------|----------|
| Documentation gaps | [GitHub Issues](https://github.com/DreamLab-AI/VisionFlow/issues) with `documentation` label |
| Technical problems | [Troubleshooting Guide](how-to/operations/troubleshooting.md) |
| Infrastructure issues | [Infrastructure Troubleshooting](how-to/infrastructure/troubleshooting.md) |
| Developer setup | [Development Setup](how-to/development/01-development-setup.md) |
| Feature requests | [GitHub Discussions](https://github.com/DreamLab-AI/VisionFlow/discussions) |

## Documentation Stats

| Category | Count |
|----------|-------|
| **Tutorials** | 5 |
| **How-To Guides** | 81 |
| **Explanation** | 72 |
| **Reference** | 46 |
| **Other (diagrams, research)** | 38 |
| **Total** | 242 markdown files |

- **Framework**: Diataxis (Tutorials, How-To, Explanation, Reference)
- **Last Updated**: 2026-01-29
- **Verified**: Links checked, Mermaid diagrams validated

---

*Maintained by DreamLab AI Documentation Team*
