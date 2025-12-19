---
title: "VisionFlow Documentation - Master Index"
description: "Complete index of all 226+ documentation files with role-based navigation"
category: reference
tags:
  - documentation
updated-date: 2025-12-19
difficulty-level: intermediate
---


# VisionFlow Documentation - Master Index

**226+ documents organized for maximum discoverability** | [Quick Start](#quick-start) | [Search](#search-index) | [By Role](#navigation-by-role)

---

## 🚀 Quick Start

**Get started in 5 minutes:**

1. **[Install VisionFlow](tutorials/01-installation.md)** → Docker or native setup
2. **[Create Your First Graph](tutorials/02-first-graph.md)** → Launch AI agents
3. **[Navigate in 3D](guides/navigation-guide.md)** → Master the interface

**Choose your path:**
- 🆕 [New Users](#new-users) | 👨‍💻 [Developers](#developers) | 🏗️ [Architects](#architects) | 🔧 [DevOps](#devops-operators)

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Navigation by Role](#navigation-by-role)
3. [Documentation Sections](#documentation-sections)
   - [Tutorials](#1-tutorials-3-documents)
   - [How-To Guides](#2-how-to-guides-61-documents)
   - [Explanations](#3-explanations-75-documents)
   - [Reference](#4-reference-22-documents)
   - [Specialized](#specialized-content)
4. [Search Index](#search-index)
5. [Topic Index (A-Z)](#topic-index-a-z)
6. [Learning Paths](#learning-paths)

---

## 🎯 Navigation by Role

### 🆕 New Users

**Your Learning Path:**

| Step | Document | What You'll Learn | Time |
|------|----------|-------------------|------|
| 1 | **[What is VisionFlow?](OVERVIEW.md)** | Value proposition and use cases | 10 min |
| 2 | **[Installation](tutorials/01-installation.md)** | Docker and native setup | 15 min |
| 3 | **[First Graph](tutorials/02-first-graph.md)** | Create visualization with AI | 20 min |
| 4 | **[Navigation Guide](guides/navigation-guide.md)** | Master 3D interface | 15 min |
| 5 | **[Configuration](guides/configuration.md)** | Customize settings | 10 min |

**Next Steps:**
- [Neo4j Quick Start](tutorials/neo4j-quick-start.md) - Query the graph database
- [Troubleshooting](guides/troubleshooting.md) - Common issues
- [Natural Language Queries](guides/features/natural-language-queries.md) - Ask questions

---

### 👨‍💻 Developers

**Your Onboarding Path:**

| Priority | Document | Focus Area |
|----------|----------|-----------|
| ⭐⭐⭐ | **[Developer Journey](DEVELOPER_JOURNEY.md)** | Codebase learning path |
| ⭐⭐⭐ | **[Development Setup](guides/developer/01-development-setup.md)** | IDE, dependencies, environment |
| ⭐⭐⭐ | **[Project Structure](guides/developer/02-project-structure.md)** | Code organization |
| ⭐⭐ | **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** | System design patterns |
| ⭐⭐ | **[Adding Features](guides/developer/04-adding-features.md)** | Development workflow |
| ⭐ | **[Testing Guide](guides/testing-guide.md)** | Unit, integration, E2E tests |

**By Technology:**
- **Rust Backend** → [Server Architecture](concepts/architecture/core/server.md), [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md)
- **React Frontend** → [Client Architecture](explanations/architecture/core/client.md), [State Management](guides/client/state-management.md)
- **Neo4j Database** → [Database Architecture](explanations/architecture/database-architecture.md), [Schemas](reference/database/schemas.md)
- **GPU/CUDA** → [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md), [GPU Optimizations](explanations/architecture/gpu/optimizations.md)
- **WebSocket** → [Binary Protocol](reference/protocols/binary-websocket.md), [Best Practices](guides/developer/websocket-best-practices.md)

**API Reference:**
- [REST API Complete](reference/api/rest-api-complete.md)
- [WebSocket API](reference/api/03-websocket.md)
- [Authentication](reference/api/01-authentication.md)

---

### 🏗️ Architects

**System Design Path:**

| Document | Focus |
|----------|-------|
| **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** | Complete system architecture |
| **[Technology Choices](TECHNOLOGY_CHOICES.md)** | Technology stack rationale |
| **[System Overview](explanations/system-overview.md)** | Architectural blueprint |
| **[Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md)** | Ports & adapters pattern |
| **[Data Flow Complete](explanations/architecture/data-flow-complete.md)** | End-to-end pipeline |
| **[Integration Patterns](explanations/architecture/integration-patterns.md)** | System integration |

**Deep Dives:**
- **Actor System** → [Actor System Guide](guides/architecture/actor-system.md), [Server Architecture](concepts/architecture/core/server.md)
- **Database** → [Database Architecture](explanations/architecture/database-architecture.md), [Neo4j Persistence ADR](explanations/architecture/decisions/0001-neo4j-persistent-with-filesystem-sync.md)
- **Physics** → [Semantic Physics System](explanations/architecture/semantic-physics-system.md), [GPU Communication Flow](explanations/architecture/gpu/communication-flow.md)
- **Ontology** → [Ontology Storage Architecture](explanations/architecture/ontology-storage-architecture.md), [Reasoning Pipeline](explanations/architecture/ontology-reasoning-pipeline.md)
- **Multi-Agent** → [Multi-Agent System](explanations/architecture/multi-agent-system.md), [Agent Orchestration](guides/agent-orchestration.md)

**Ports (Hexagonal Architecture):**
- [Ports Overview](explanations/architecture/ports/01-overview.md)
- [Knowledge Graph Repository](explanations/architecture/ports/03-knowledge-graph-repository.md)
- [Ontology Repository](explanations/architecture/ports/04-ontology-repository.md)
- [Inference Engine](explanations/architecture/ports/05-inference-engine.md)
- [GPU Physics Adapter](explanations/architecture/ports/06-gpu-physics-adapter.md)

---

### 🔧 DevOps Operators

**Deployment & Operations Path:**

| Document | Purpose |
|----------|---------|
| **[Deployment Guide](guides/deployment.md)** | Production deployment |
| **[Docker Compose Guide](guides/docker-compose-guide.md)** | Multi-container orchestration |
| **[Pipeline Operator Runbook](guides/operations/pipeline-operator-runbook.md)** | Operations playbook |
| **[Configuration](guides/configuration.md)** | Environment variables |
| **[Security](guides/security.md)** | Authentication, secrets |
| **[Telemetry Logging](guides/telemetry-logging.md)** | Observability, monitoring |

**Infrastructure:**
- [Infrastructure Architecture](guides/infrastructure/architecture.md)
- [Docker Environment](guides/infrastructure/docker-environment.md)
- [Port Configuration](guides/infrastructure/port-configuration.md)
- [Troubleshooting](guides/infrastructure/troubleshooting.md)

**Data Operations:**
- [Neo4j Migration](guides/neo4j-migration.md)
- [Pipeline Admin API](guides/pipeline-admin-api.md)
- [GitHub Sync Service](explanations/architecture/github-sync-service-design.md)

---

## 📚 Documentation Sections

### 1. Tutorials (3 documents)

*Learning-oriented: Step-by-step lessons for beginners*

| Tutorial | What You'll Learn | Time | Difficulty |
|----------|-------------------|------|------------|
| **[Installation](tutorials/01-installation.md)** | Docker and native setup for all platforms | 10 min | Beginner |
| **[First Graph](tutorials/02-first-graph.md)** | Create your first visualization with AI agents | 15 min | Beginner |
| **[Neo4j Quick Start](tutorials/neo4j-quick-start.md)** | Query and explore the graph database | 20 min | Beginner |

---

### 2. How-To Guides (61 documents)

*Task-oriented: Practical instructions for specific goals*

#### 🎯 Core Features (8 guides)

| Guide | Task |
|-------|------|
| **[Navigation Guide](guides/navigation-guide.md)** | 3D interface controls, camera, selection |
| **[Configuration](guides/configuration.md)** | Environment variables and settings |
| **[Troubleshooting](guides/troubleshooting.md)** | Solve common issues |
| **[Extending the System](guides/extending-the-system.md)** | Plugin patterns and custom components |
| **[Filtering Nodes](guides/features/filtering-nodes.md)** | Client-side graph filtering |
| **[Natural Language Queries](guides/features/natural-language-queries.md)** | Ask questions in plain English |
| **[Intelligent Pathfinding](guides/features/intelligent-pathfinding.md)** | Graph traversal algorithms |
| **[Semantic Forces](guides/features/semantic-forces.md)** | Physics-based layouts |

#### 🔐 Authentication & Security (3 guides)

- [Auth & User Settings](guides/features/auth-user-settings.md)
- [Nostr Authentication](guides/features/nostr-auth.md)
- [Settings Authentication](guides/features/settings-authentication.md)

#### 🤖 AI Agent System (4 guides)

- [Agent Orchestration](guides/agent-orchestration.md)
- [Orchestrating Agents](guides/orchestrating-agents.md)
- [Multi-Agent Skills](guides/multi-agent-skills.md)
- [AI Models & Services](guides/ai-models/README.md)

#### 🗄️ Neo4j & Data (6 guides)

- [Neo4j Integration](guides/neo4j-integration.md)
- [Neo4j Implementation Roadmap](guides/neo4j-implementation-roadmap.md)
- [Neo4j Migration](guides/neo4j-migration.md)
- [Local File Sync Strategy](guides/features/local-file-sync-strategy.md)
- [GitHub Pagination Fix](guides/features/github-pagination-fix.md)
- [Pipeline Admin API](guides/pipeline-admin-api.md)

#### 🦉 Ontology & Reasoning (5 guides)

- [Ontology Parser](guides/ontology-parser.md)
- [Ontology Reasoning Integration](guides/ontology-reasoning-integration.md)
- [Ontology Storage Guide](guides/ontology-storage-guide.md)
- [Ontology Semantic Forces](guides/ontology-semantic-forces.md)
- [Semantic Features Implementation](guides/semantic-features-implementation.md)

#### 🚀 Deployment & Operations (9 guides)

- [Deployment](guides/deployment.md)
- [Docker Compose Guide](guides/docker-compose-guide.md)
- [Docker Environment Setup](guides/docker-environment-setup.md)
- [Development Workflow](guides/development-workflow.md)
- [Testing Guide](guides/testing-guide.md)
- [Security](guides/security.md)
- [Telemetry Logging](guides/telemetry-logging.md)
- [Pipeline Operator Runbook](guides/operations/pipeline-operator-runbook.md)
- [Contributing](guides/developer/06-contributing.md)

#### 🏢 Infrastructure (6 guides)

- [Architecture](guides/infrastructure/architecture.md)
- [Docker Environment](guides/infrastructure/docker-environment.md)
- [Tools](guides/infrastructure/tools.md)
- [Port Configuration](guides/infrastructure/port-configuration.md)
- [Troubleshooting](guides/infrastructure/troubleshooting.md)
- [Goalie Integration](guides/infrastructure/goalie-integration.md)

#### 🥽 XR & Multi-User (2 guides)

- [Vircadia XR Complete Guide](guides/vircadia-xr-complete-guide.md)
- [Vircadia Multi-User Guide](guides/vircadia-multi-user-guide.md)

#### 👨‍💻 Developer (8 guides)

- [Development Setup](guides/developer/01-development-setup.md) ⭐⭐⭐
- [Project Structure](guides/developer/02-project-structure.md) ⭐⭐⭐
- [Adding Features](guides/developer/04-adding-features.md) ⭐⭐
- [Contributing](guides/developer/06-contributing.md)
- [WebSocket Best Practices](guides/developer/websocket-best-practices.md)
- [JSON Serialization Patterns](guides/developer/json-serialization-patterns.md)
- [Test Execution](guides/developer/test-execution.md)

#### 📊 Client Development (3 guides)

- [State Management](guides/client/state-management.md)
- [Three.js Rendering](guides/client/three-js-rendering.md)
- [XR Integration](guides/client/xr-integration.md)

#### 🔄 Advanced Features (6 guides)

- [Hierarchy Integration](guides/hierarchy-integration.md)
- [Stress Majorization Guide](guides/stress-majorization-guide.md)
- [Ontology Sync Enhancement](guides/features/ontology-sync-enhancement.md)
- [DeepSeek Verification](guides/features/deepseek-verification.md)
- [DeepSeek Deployment](guides/features/deepseek-deployment.md)

#### 🔧 Migration (2 guides)

- [GraphServiceActor Migration](guides/graphserviceactor-migration.md)
- [JSON to Binary Protocol](guides/migration/json-to-binary-protocol.md)

---

### 3. Explanations (75 documents)

*Understanding-oriented: Deep dives into architecture and design*

#### 🏛️ High-Level Architecture (4 documents)

| Document | Description |
|----------|-------------|
| **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** | Complete system architecture with diagrams |
| **[Developer Journey](DEVELOPER_JOURNEY.md)** | Step-by-step codebase learning path |
| **[System Overview](OVERVIEW.md)** | What VisionFlow is and why it exists |
| **[Technology Choices](TECHNOLOGY_CHOICES.md)** | Technology stack rationale |

#### 🔷 System Architecture (29 documents)

**⭐ Core Architecture:**
- [Server Architecture](concepts/architecture/core/server.md) - 21 actors, ports/adapters, Neo4j
- [Actor System Guide](guides/architecture/actor-system.md) - Actor patterns, debugging
- [Database Architecture](explanations/architecture/database-architecture.md) - Neo4j schema, queries
- [System Overview](explanations/system-overview.md) - Complete blueprint
- [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md) - Ports & adapters

**Pipeline & Integration:**
- [Data Flow Complete](explanations/architecture/data-flow-complete.md)
- [Pipeline Integration](explanations/architecture/pipeline-integration.md)
- [GitHub Sync Service Design](explanations/architecture/github-sync-service-design.md)
- [Integration Patterns](explanations/architecture/integration-patterns.md)

**Services & Business Logic:**
- [Services Architecture](explanations/architecture/services-architecture.md)
- [Services Layer](explanations/architecture/services-layer.md)
- [API Handlers Reference](explanations/architecture/api-handlers-reference.md)
- [CQRS Directive Template](explanations/architecture/cqrs-directive-template.md)

**Advanced Topics:**
- [Multi-Agent System](explanations/architecture/multi-agent-system.md)
- [Analytics Visualization](explanations/architecture/analytics-visualization.md)
- [XR Immersive System](explanations/architecture/xr-immersive-system.md)
- [Quick Reference](explanations/architecture/quick-reference.md)

#### 🗄️ Database & Persistence (3 documents)

- [Database Architecture](explanations/architecture/database-architecture.md)
- [Adapter Patterns](explanations/architecture/adapter-patterns.md)
- [Ontology Storage Architecture](explanations/architecture/ontology-storage-architecture.md)

#### 🦉 Ontology & Reasoning (11 documents)

**Architecture:**
- [Ontology Reasoning Pipeline](explanations/architecture/ontology-reasoning-pipeline.md)
- [Reasoning Data Flow](explanations/architecture/reasoning-data-flow.md)
- [Reasoning Tests Summary](explanations/architecture/reasoning-tests-summary.md)

**Concepts:**
- [Reasoning Engine](explanations/ontology/reasoning-engine.md)
- [Neo4j Integration](explanations/ontology/neo4j-integration.md)
- [Ontology Pipeline Integration](explanations/ontology/ontology-pipeline-integration.md)
- [Ontology Typed System](explanations/ontology/ontology-typed-system.md)
- [Client-Side Hierarchical LOD](explanations/ontology/client-side-hierarchical-lod.md)
- [Hierarchical Visualization](explanations/ontology/hierarchical-visualization.md)
- [Intelligent Pathfinding System](explanations/ontology/intelligent-pathfinding-system.md)
- [Enhanced Parser](explanations/ontology/enhanced-parser.md)

#### ⚡ Visualization & Physics (8 documents)

- [Semantic Physics System](explanations/architecture/semantic-physics-system.md)
- [Semantic Forces System](explanations/architecture/semantic-forces-system.md)
- [Semantic Physics](explanations/architecture/semantic-physics.md)
- [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md)
- [Stress Majorization](explanations/architecture/stress-majorization.md)
- [Hierarchical Visualization](explanations/architecture/hierarchical-visualization.md)
- [Semantic Forces](explanations/physics/semantic-forces.md)
- [Semantic Forces Actor](explanations/physics/semantic-forces-actor.md)

#### 🎮 GPU Acceleration (3 documents)

- [GPU Communication Flow](explanations/architecture/gpu/communication-flow.md)
- [GPU Optimizations](explanations/architecture/gpu/optimizations.md)
- [GPU README](explanations/architecture/gpu/readme.md)

#### 💻 Client-Server (4 documents)

- [Client](explanations/architecture/core/client.md)
- [Server](explanations/architecture/core/server.md)
- [Visualization](explanations/architecture/core/visualization.md)
- [WebSocket Protocol](explanations/architecture/components/websocket-protocol.md)

#### 🔌 Ports (Hexagonal Architecture) (7 documents)

- [Ports Overview](explanations/architecture/ports/01-overview.md)
- [Settings Repository](explanations/architecture/ports/02-settings-repository.md)
- [Knowledge Graph Repository](explanations/architecture/ports/03-knowledge-graph-repository.md)
- [Ontology Repository](explanations/architecture/ports/04-ontology-repository.md)
- [Inference Engine](explanations/architecture/ports/05-inference-engine.md)
- [GPU Physics Adapter](explanations/architecture/ports/06-gpu-physics-adapter.md)
- [GPU Semantic Analyzer](explanations/architecture/ports/07-gpu-semantic-analyzer.md)

#### 📐 Architecture Decisions (1 document)

- [ADR-0001: Neo4j Persistence](explanations/architecture/decisions/0001-neo4j-persistent-with-filesystem-sync.md)

---

### 4. Reference (22 documents)

*Information-oriented: Technical specifications and APIs*

#### 🌐 API Documentation (8 references)

| Reference | Type |
|-----------|------|
| **[API Complete Reference](reference/api-complete-reference.md)** | All endpoints with examples |
| **[REST API Complete](reference/api/rest-api-complete.md)** | HTTP API specification |
| **[REST API Reference](reference/api/rest-api-reference.md)** | OpenAPI/Swagger format |
| **[Authentication](reference/api/01-authentication.md)** | JWT, sessions, Nostr auth |
| **[WebSocket API](reference/api/03-websocket.md)** | Real-time binary protocol |
| **[Semantic Features API](reference/api/semantic-features-api.md)** | Natural language queries |
| **[Pathfinding Examples](reference/api/pathfinding-examples.md)** | Graph traversal examples |
| **[API README](reference/api/readme.md)** | API documentation index |

#### 📡 Protocols (2 references)

- [Binary WebSocket](reference/protocols/binary-websocket.md) - 36-byte node format
- [WebSocket Protocol](reference/websocket-protocol.md) - V2 protocol specification

#### 🗄️ Database (4 references)

- [Schemas](reference/database/schemas.md)
- [Ontology Schema V2](reference/database/ontology-schema-v2.md)
- [User Settings Schema](reference/database/user-settings-schema.md)
- [Neo4j Persistence Analysis](reference/database/neo4j-persistence-analysis.md)

#### ⚙️ System Status (5 references)

- [Error Codes](reference/error-codes.md)
- [Implementation Status](reference/implementation-status.md)
- [Code Quality Status](reference/code-quality-status.md)
- [Performance Benchmarks](reference/performance-benchmarks.md)
- [Physics Implementation](reference/physics-implementation.md)

---

## 🔍 Search Index

### Common Tasks

| I want to... | Go here → |
|-------------|----------|
| **Install VisionFlow** | [Installation Tutorial](tutorials/01-installation.md) |
| **Create my first graph** | [First Graph Tutorial](tutorials/02-first-graph.md) |
| **Deploy AI agents** | [Agent Orchestration](guides/agent-orchestration.md) |
| **Query Neo4j** | [Neo4j Integration](guides/neo4j-integration.md) |
| **Add a feature** | [Adding Features](guides/developer/04-adding-features.md) |
| **Set up XR/VR** | [Vircadia XR Guide](guides/vircadia-xr-complete-guide.md) |
| **Understand architecture** | [Architecture Overview](ARCHITECTURE_OVERVIEW.md) |
| **Learn the codebase** | [Developer Journey](DEVELOPER_JOURNEY.md) |
| **Deploy to production** | [Deployment Guide](guides/deployment.md) |
| **Configure environment** | [Configuration](guides/configuration.md) |
| **Fix issues** | [Troubleshooting](guides/troubleshooting.md) |
| **Write tests** | [Testing Guide](guides/testing-guide.md) |
| **Use REST API** | [REST API Complete](reference/api/rest-api-complete.md) |
| **Use WebSocket** | [WebSocket API](reference/api/03-websocket.md) |
| **Optimize performance** | [GPU Optimizations](explanations/architecture/gpu/optimizations.md) |
| **Secure the app** | [Security Guide](guides/security.md) |

### By Technology

**Neo4j Graph Database:**
- [Neo4j Integration](guides/neo4j-integration.md)
- [Database Schemas](reference/database/schemas.md)
- [Database Architecture](explanations/architecture/database-architecture.md)
- [Neo4j Migration](guides/neo4j-migration.md)

**Rust / Actix Web:**
- [Server Architecture](concepts/architecture/core/server.md)
- [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md)
- [Project Structure](guides/developer/02-project-structure.md)

**React / Three.js:**
- [Client Architecture](explanations/architecture/core/client.md)
- [Three.js Rendering](guides/client/three-js-rendering.md)
- [State Management](guides/client/state-management.md)

**CUDA / GPU:**
- [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md)
- [GPU Optimizations](explanations/architecture/gpu/optimizations.md)
- [Performance Benchmarks](reference/performance-benchmarks.md)

**OWL / Ontologies:**
- [Ontology Storage Architecture](explanations/architecture/ontology-storage-architecture.md)
- [Ontology Parser](guides/ontology-parser.md)
- [Reasoning Engine](explanations/ontology/reasoning-engine.md)

**WebSocket:**
- [Binary Protocol](reference/protocols/binary-websocket.md)
- [WebSocket API](reference/api/03-websocket.md)
- [WebSocket Best Practices](guides/developer/websocket-best-practices.md)

**AI / MCP:**
- [Multi-Agent System](explanations/architecture/multi-agent-system.md)
- [Agent Orchestration](guides/agent-orchestration.md)
- [Multi-Agent Skills](guides/multi-agent-skills.md)

**XR / WebXR:**
- [XR Immersive System](explanations/architecture/xr-immersive-system.md)
- [Vircadia XR Guide](guides/vircadia-xr-complete-guide.md)
- [XR Integration](guides/client/xr-integration.md)

---

## 📖 Topic Index (A-Z)

**A**
- Actors → [Actor System Guide](guides/architecture/actor-system.md), [Server Architecture](concepts/architecture/core/server.md)
- AI Agents → [Multi-Agent System](explanations/architecture/multi-agent-system.md), [Agent Orchestration](guides/agent-orchestration.md)
- API → [REST API Complete](reference/api/rest-api-complete.md), [API Complete Reference](reference/api-complete-reference.md)
- Architecture → [Architecture Overview](ARCHITECTURE_OVERVIEW.md), [System Overview](explanations/system-overview.md)
- Authentication → [Auth & User Settings](guides/features/auth-user-settings.md), [Nostr Auth](guides/features/nostr-auth.md)

**B**
- Binary Protocol → [Binary WebSocket](reference/protocols/binary-websocket.md)
- Benchmarks → [Performance Benchmarks](reference/performance-benchmarks.md)

**C**
- Client → [Client Architecture](explanations/architecture/core/client.md), [State Management](guides/client/state-management.md)
- Configuration → [Configuration Guide](guides/configuration.md)
- CQRS → [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md)
- CUDA → [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md)

**D**
- Database → [Database Architecture](explanations/architecture/database-architecture.md), [Schemas](reference/database/schemas.md)
- Deployment → [Deployment Guide](guides/deployment.md)
- Developer → [Developer Journey](DEVELOPER_JOURNEY.md), [Development Setup](guides/developer/01-development-setup.md)
- Docker → [Docker Compose Guide](guides/docker-compose-guide.md)

**E**
- Error Codes → [Error Reference](reference/error-codes.md)

**F**
- Features → [Adding Features](guides/developer/04-adding-features.md)
- Filtering → [Filtering Nodes](guides/features/filtering-nodes.md)

**G**
- GitHub → [GitHub Sync Service](explanations/architecture/github-sync-service-design.md)
- GPU → [GPU Optimizations](explanations/architecture/gpu/optimizations.md)
- Graph → [Knowledge Graph Repository](explanations/architecture/ports/03-knowledge-graph-repository.md)

**H**
- Hexagonal Architecture → [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md), [Ports Overview](explanations/architecture/ports/01-overview.md)
- Hierarchy → [Hierarchical Visualization](explanations/ontology/hierarchical-visualization.md)

**I**
- Installation → [Installation Tutorial](tutorials/01-installation.md)
- Integration → [Integration Patterns](explanations/architecture/integration-patterns.md)

**J**
- JSON → [JSON Serialization](guides/developer/json-serialization-patterns.md)

**K**
- Knowledge Graph → [First Graph](tutorials/02-first-graph.md)

**L**
- Layout → [Stress Majorization](explanations/architecture/stress-majorization.md)

**M**
- Migration → [Neo4j Migration](guides/neo4j-migration.md)
- Multi-Agent → [Multi-Agent System](explanations/architecture/multi-agent-system.md)

**N**
- Navigation → [Navigation Guide](guides/navigation-guide.md)
- Neo4j → [Neo4j Integration](guides/neo4j-integration.md), [Neo4j Quick Start](tutorials/neo4j-quick-start.md)
- Nostr → [Nostr Authentication](guides/features/nostr-auth.md)

**O**
- Ontology → [Ontology Parser](guides/ontology-parser.md), [Ontology Storage](guides/ontology-storage-guide.md)
- Operations → [Pipeline Operator Runbook](guides/operations/pipeline-operator-runbook.md)

**P**
- Pathfinding → [Intelligent Pathfinding](guides/features/intelligent-pathfinding.md)
- Performance → [Performance Benchmarks](reference/performance-benchmarks.md)
- Physics → [Semantic Physics System](explanations/architecture/semantic-physics-system.md)
- Ports → [Ports Overview](explanations/architecture/ports/01-overview.md)

**Q**
- Queries → [Natural Language Queries](guides/features/natural-language-queries.md)

**R**
- React → [Client Architecture](explanations/architecture/core/client.md)
- Reasoning → [Reasoning Engine](explanations/ontology/reasoning-engine.md)
- REST → [REST API Complete](reference/api/rest-api-complete.md)
- Rust → [Server Architecture](concepts/architecture/core/server.md)

**S**
- Security → [Security Guide](guides/security.md)
- Semantic Forces → [Semantic Forces](guides/features/semantic-forces.md)
- Server → [Server Architecture](concepts/architecture/core/server.md)
- Settings → [Settings Repository](explanations/architecture/ports/02-settings-repository.md)

**T**
- Testing → [Testing Guide](guides/testing-guide.md)
- Three.js → [Three.js Rendering](guides/client/three-js-rendering.md)
- Troubleshooting → [Troubleshooting Guide](guides/troubleshooting.md)

**V**
- Vector Search → [RuVector Integration](explanations/architecture/ruvector-integration.md)
- Vircadia → [Vircadia XR Guide](guides/vircadia-xr-complete-guide.md)
- Visualization → [Visualization](explanations/architecture/core/visualization.md)

**W**
- WebSocket → [WebSocket Protocol](reference/websocket-protocol.md), [Binary WebSocket](reference/protocols/binary-websocket.md)

**X**
- XR → [XR Immersive System](explanations/architecture/xr-immersive-system.md), [XR Integration](guides/client/xr-integration.md)

---

## 🎓 Learning Paths

### Beginner → Intermediate → Advanced

**Beginner (Week 1):**
1. [What is VisionFlow?](OVERVIEW.md)
2. [Installation](tutorials/01-installation.md)
3. [First Graph](tutorials/02-first-graph.md)
4. [Navigation Guide](guides/navigation-guide.md)
5. [Neo4j Quick Start](tutorials/neo4j-quick-start.md)

**Intermediate (Week 2-3):**
1. [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
2. [Developer Journey](DEVELOPER_JOURNEY.md)
3. [Project Structure](guides/developer/02-project-structure.md)
4. [Adding Features](guides/developer/04-adding-features.md)
5. [Testing Guide](guides/testing-guide.md)

**Advanced (Week 4+):**
1. [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md)
2. [Database Architecture](explanations/architecture/database-architecture.md)
3. [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md)
4. [Ontology Reasoning Pipeline](explanations/architecture/ontology-reasoning-pipeline.md)
5. [Multi-Agent System](explanations/architecture/multi-agent-system.md)

### By Feature Area

**Graph Visualization:**
- [First Graph](tutorials/02-first-graph.md)
- [Navigation Guide](guides/navigation-guide.md)
- [Three.js Rendering](guides/client/three-js-rendering.md)
- [Semantic Physics System](explanations/architecture/semantic-physics-system.md)

**AI Agents:**
- [Agent Orchestration](guides/agent-orchestration.md)
- [Multi-Agent Skills](guides/multi-agent-skills.md)
- [Multi-Agent System](explanations/architecture/multi-agent-system.md)

**Database:**
- [Neo4j Quick Start](tutorials/neo4j-quick-start.md)
- [Neo4j Integration](guides/neo4j-integration.md)
- [Database Architecture](explanations/architecture/database-architecture.md)
- [Database Schemas](reference/database/schemas.md)

**Ontology:**
- [Ontology Parser](guides/ontology-parser.md)
- [Ontology Storage Guide](guides/ontology-storage-guide.md)
- [Reasoning Engine](explanations/ontology/reasoning-engine.md)
- [Ontology Reasoning Pipeline](explanations/architecture/ontology-reasoning-pipeline.md)

---

## 📊 Documentation Statistics

- **Total Documents**: 226+ markdown files
- **Tutorials**: 3 learning-oriented guides
- **How-To Guides**: 61 task-oriented instructions
- **Explanations**: 75+ understanding-oriented deep dives
- **Reference**: 22 information-oriented specifications
- **Last Full Audit**: 2025-12-18
- **Link Health**: 98% valid internal links
- **Diagram Format**: Mermaid (100% valid syntax)

---

## 📞 Getting Help

| Issue Type | Where to Go |
|------------|-------------|
| **Documentation gaps** | [File GitHub Issue](https://github.com/DreamLab-AI/VisionFlow/issues) with `documentation` label |
| **Technical problems** | [Troubleshooting Guide](guides/troubleshooting.md) |
| **Infrastructure issues** | [Infrastructure Troubleshooting](guides/infrastructure/troubleshooting.md) |
| **Developer setup** | [Development Setup](guides/developer/01-development-setup.md) |
| **Feature requests** | [GitHub Discussions](https://github.com/DreamLab-AI/VisionFlow/discussions) |

---

**Last Updated**: 2025-12-18
**Documentation Version**: 2.0
**Framework**: Diátaxis
**Maintainer**: DreamLab AI Documentation Team
