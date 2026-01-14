---
title: VisionFlow Documentation - Quick Navigation
description: > **Search tip**: Use Ctrl+F to find keywords across all docs
category: reference
tags:
  - architecture
  - design
  - patterns
  - structure
  - api
related-docs:
  - README.md
  - OVERVIEW.md
  - ARCHITECTURE_OVERVIEW.md
  - DEVELOPER_JOURNEY.md
  - TECHNOLOGY_CHOICES.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Docker installation
  - Neo4j database
---

# VisionFlow Documentation - Quick Navigation

**Fast reference for all 226 documentation files** - Organized by directory with one-line descriptions.

> **Search tip**: Use Ctrl+F to find keywords across all docs

---

## 📁 Root Level (6 files)

| File | Description |
|------|-------------|
| **[README.md](README.md)** | Master documentation index with Diátaxis organization |
| **[OVERVIEW.md](OVERVIEW.md)** | What VisionFlow is and why it exists - value proposition |
| **[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)** | Complete system architecture with layered diagrams |
| **[DEVELOPER_JOURNEY.md](DEVELOPER_JOURNEY.md)** | Step-by-step codebase learning path for new developers |
| **[TECHNOLOGY_CHOICES.md](TECHNOLOGY_CHOICES.md)** | Technology stack rationale and trade-off analysis |
| **[MERMAID_FIXES_STATS.json](MERMAID_FIXES_STATS.json)** | Mermaid diagram validation statistics |

---

## 📚 Tutorials (3 files)

*Learning-oriented: Step-by-step lessons*

| File | Description |
|------|-------------|
| **[01-installation.md](tutorials/01-installation.md)** | Docker and native installation for all platforms |
| **[02-first-graph.md](tutorials/02-first-graph.md)** | Create your first knowledge graph with AI agents |
| **[neo4j-quick-start.md](tutorials/neo4j-quick-start.md)** | Query and explore the Neo4j graph database |

---

## 🛠️ Guides (61 files)

*Task-oriented: Practical instructions for specific goals*

### Core Guides (8 files)

| File | Description |
|------|-------------|
| **[configuration.md](guides/configuration.md)** | Environment variables and application settings |
| **[contributing.md](guides/contributing.md)** | Code style, pull requests, documentation standards |
| **[deployment.md](guides/deployment.md)** | Production deployment strategies and checklist |
| **[development-workflow.md](guides/development-workflow.md)** | Git workflow, testing, CI/CD process |
| **[extending-the-system.md](guides/extending-the-system.md)** | Plugin patterns and custom component development |
| **[navigation-guide.md](guides/navigation-guide.md)** | 3D interface controls, camera, node selection |
| **[testing-guide.md](guides/testing-guide.md)** | Unit, integration, E2E testing strategies |
| **[troubleshooting.md](guides/troubleshooting.md)** | Common issues and solutions |

### AI Agent Guides (3 files)

| File | Description |
|------|-------------|
| **[agent-orchestration.md](guides/agent-orchestration.md)** | Deploy and manage 50+ concurrent AI agents |
| **[multi-agent-skills.md](guides/multi-agent-skills.md)** | Agent capabilities and specialization matrix |
| **[orchestrating-agents.md](guides/orchestrating-agents.md)** | Advanced coordination patterns and topologies |

### Neo4j & Data (6 files)

| File | Description |
|------|-------------|
| **[neo4j-integration.md](guides/neo4j-integration.md)** | Work with graph database operations |
| **[neo4j-implementation-roadmap.md](guides/neo4j-implementation-roadmap.md)** | Migration planning and timeline |
| **[neo4j-migration.md](guides/neo4j-migration.md)** | Step-by-step migration from in-memory to persistent |
| **[graphserviceactor-migration.md](guides/graphserviceactor-migration.md)** | Actor system migration patterns |
| **[pipeline-admin-api.md](guides/pipeline-admin-api.md)** | Control GitHub sync pipelines via API |
| **[index.md](guides/index.md)** | Guides section index |

### Ontology & Reasoning (5 files)

| File | Description |
|------|-------------|
| **[ontology-parser.md](guides/ontology-parser.md)** | Parse and validate OWL ontologies |
| **[ontology-reasoning-integration.md](guides/ontology-reasoning-integration.md)** | Enable semantic inference with Whelk reasoner |
| **[ontology-semantic-forces.md](guides/ontology-semantic-forces.md)** | Visualize OWL constraints as physics forces |
| **[ontology-storage-guide.md](guides/ontology-storage-guide.md)** | Persist ontologies in Neo4j with typed nodes |
| **[semantic-features-implementation.md](guides/semantic-features-implementation.md)** | Natural language query system |

### Advanced Features (4 files)

| File | Description |
|------|-------------|
| **[hierarchy-integration.md](guides/hierarchy-integration.md)** | Hierarchical graph layout algorithms |
| **[stress-majorization-guide.md](guides/stress-majorization-guide.md)** | Advanced force-directed layout technique |
| **[security.md](guides/security.md)** | Authentication, authorization, secrets management |
| **[telemetry-logging.md](guides/telemetry-logging.md)** | Observability, tracing, monitoring |

### Docker & Infrastructure (3 files)

| File | Description |
|------|-------------|
| **[docker-compose-guide.md](guides/docker-compose-guide.md)** | Multi-container orchestration with profiles |
| **[docker-environment-setup.md](guides/docker-environment-setup.md)** | Container configuration and networking |
| **[readme.md](guides/readme.md)** | Guides overview |

### XR & VR (2 files)

| File | Description |
|------|-------------|
| **[vircadia-multi-user-guide.md](guides/vircadia-multi-user-guide.md)** | Collaborative VR experiences with Vircadia |
| **[vircadia-xr-complete-guide.md](guides/vircadia-xr-complete-guide.md)** | Meta Quest 3 WebXR implementation |

### guides/client/ (3 files)

| File | Description |
|------|-------------|
| **[state-management.md](guides/client/state-management.md)** | React state patterns with Zustand stores |
| **[three-js-rendering.md](guides/client/three-js-rendering.md)** | 3D visualization pipeline with React Three Fiber |
| **[xr-integration.md](guides/client/xr-integration.md)** | WebXR implementation details |

### guides/developer/ (8 files)

| File | Description |
|------|-------------|
| **[01-development-setup.md](guides/developer/01-development-setup.md)** | IDE, dependencies, local environment setup |
| **[02-project-structure.md](guides/developer/02-project-structure.md)** | Codebase organization and module layout |
| **[04-adding-features.md](guides/developer/04-adding-features.md)** | Feature development workflow with SPARC |
| **[06-contributing.md](guides/developer/06-contributing.md)** | Code style, testing, documentation standards |
| **[json-serialization-patterns.md](guides/developer/json-serialization-patterns.md)** | Data serialization strategies for WebSocket |
| **[readme.md](guides/developer/readme.md)** | Developer guides overview |
| **[test-execution.md](guides/developer/test-execution.md)** | Running and debugging test suites |
| **[websocket-best-practices.md](guides/developer/websocket-best-practices.md)** | Real-time communication patterns |

### guides/features/ (11 files)

| File | Description |
|------|-------------|
| **[auth-user-settings.md](guides/features/auth-user-settings.md)** | User authentication system implementation |
| **[deepseek-deployment.md](guides/features/deepseek-deployment.md)** | Deploy DeepSeek LLM skill container |
| **[deepseek-verification.md](guides/features/deepseek-verification.md)** | LLM API integration testing |
| **[filtering-nodes.md](guides/features/filtering-nodes.md)** | Client-side graph filtering with quality scores |
| **[github-pagination-fix.md](guides/features/github-pagination-fix.md)** | Handle large GitHub API responses |
| **[intelligent-pathfinding.md](guides/features/intelligent-pathfinding.md)** | A* and Dijkstra graph traversal algorithms |
| **[local-file-sync-strategy.md](guides/features/local-file-sync-strategy.md)** | File synchronization patterns and strategies |
| **[natural-language-queries.md](guides/features/natural-language-queries.md)** | Semantic search with natural language |
| **[nostr-auth.md](guides/features/nostr-auth.md)** | Decentralized authentication with Nostr protocol |
| **[ontology-sync-enhancement.md](guides/features/ontology-sync-enhancement.md)** | GitHub ontology sync with HNSW vector search |
| **[semantic-forces.md](guides/features/semantic-forces.md)** | Physics-based visualization with GPU acceleration |
| **[settings-authentication.md](guides/features/settings-authentication.md)** | Settings API authentication with JWT |

### guides/infrastructure/ (6 files)

| File | Description |
|------|-------------|
| **[architecture.md](guides/infrastructure/architecture.md)** | Multi-agent Docker system design |
| **[docker-environment.md](guides/infrastructure/docker-environment.md)** | Container setup and management |
| **[goalie-integration.md](guides/infrastructure/goalie-integration.md)** | Quality gates and automated testing |
| **[port-configuration.md](guides/infrastructure/port-configuration.md)** | Network and service port mappings |
| **[readme.md](guides/infrastructure/readme.md)** | Infrastructure guides overview |
| **[tools.md](guides/infrastructure/tools.md)** | Available MCP tools and integrations |
| **[troubleshooting.md](guides/infrastructure/troubleshooting.md)** | Infrastructure-specific issues |

### guides/migration/ (1 file)

| File | Description |
|------|-------------|
| **[json-to-binary-protocol.md](guides/migration/json-to-binary-protocol.md)** | WebSocket protocol upgrade to binary format |

### guides/operations/ (1 file)

| File | Description |
|------|-------------|
| **[pipeline-operator-runbook.md](guides/operations/pipeline-operator-runbook.md)** | Operations playbook for production systems |

---

## 🧠 Explanations (75 files)

*Understanding-oriented: Deep dives into architecture and design*

### explanations/ Root (1 file)

| File | Description |
|------|-------------|
| **[system-overview.md](explanations/system-overview.md)** | Complete architectural blueprint and component interaction |

### explanations/architecture/ (44 files)

| File | Description |
|------|-------------|
| **[README.md](explanations/architecture/README.md)** | Architecture documentation index |
| **[adapter-patterns.md](explanations/architecture/adapter-patterns.md)** | Repository implementation patterns for hexagonal architecture |
| **[analytics-visualization.md](explanations/architecture/analytics-visualization.md)** | UI/UX design patterns for analytics dashboard |
| **[api-handlers-reference.md](explanations/architecture/api-handlers-reference.md)** | Handler patterns with code examples |
| **[cqrs-directive-template.md](explanations/architecture/cqrs-directive-template.md)** | Command/query separation templates |
| **[data-flow-complete.md](explanations/architecture/data-flow-complete.md)** | End-to-end data pipeline from GitHub to GPU |
| **[github-sync-service-design.md](explanations/architecture/github-sync-service-design.md)** | Streaming ontology sync with backpressure |
| **[gpu-semantic-forces.md](explanations/architecture/gpu-semantic-forces.md)** | 39 CUDA kernels for physics simulation |
| **[hexagonal-cqrs.md](explanations/architecture/hexagonal-cqrs.md)** | Ports & adapters with command/query separation |
| **[hierarchical-visualization.md](explanations/architecture/hierarchical-visualization.md)** | DAG and tree layout algorithms |
| **[integration-patterns.md](explanations/architecture/integration-patterns.md)** | System integration strategies and patterns |
| **[multi-agent-system.md](explanations/architecture/multi-agent-system.md)** | AI agent coordination with MCP protocol |
| **[ontology-analysis.md](explanations/architecture/ontology-analysis.md)** | Architecture decision analysis for ontology system |
| **[ontology-physics-integration.md](explanations/architecture/ontology-physics-integration.md)** | Wire OWL constraints to GPU physics forces |
| **[ontology-reasoning-pipeline.md](explanations/architecture/ontology-reasoning-pipeline.md)** | Whelk inference engine integration |
| **[ontology-storage-architecture.md](explanations/architecture/ontology-storage-architecture.md)** | OWL persistence in Neo4j with typed nodes |
| **[pipeline-integration.md](explanations/architecture/pipeline-integration.md)** | GitHub → Neo4j → GPU complete flow |
| **[pipeline-sequence-diagrams.md](explanations/architecture/pipeline-sequence-diagrams.md)** | Visual interaction flows with Mermaid |
| **[quick-reference.md](explanations/architecture/quick-reference.md)** | Architecture cheat sheet for developers |
| **[reasoning-data-flow.md](explanations/architecture/reasoning-data-flow.md)** | Inference pipeline stages and transformations |
| **[reasoning-tests-summary.md](explanations/architecture/reasoning-tests-summary.md)** | Test coverage report for reasoning system |
| **[ruvector-integration.md](explanations/architecture/ruvector-integration.md)** | 150x faster HNSW vector search integration |
| **[semantic-forces-system.md](explanations/architecture/semantic-forces-system.md)** | Physics constraint generation from semantic rules |
| **[semantic-physics-system.md](explanations/architecture/semantic-physics-system.md)** | Force-directed layout engine architecture |
| **[semantic-physics.md](explanations/architecture/semantic-physics.md)** | Physics simulation theory and algorithms |
| **[services-architecture.md](explanations/architecture/services-architecture.md)** | Business logic layer design with clean architecture |
| **[services-layer.md](explanations/architecture/services-layer.md)** | Business logic refactoring patterns |
| **[stress-majorization.md](explanations/architecture/stress-majorization.md)** | Graph layout optimization technique |
| **[xr-immersive-system.md](explanations/architecture/xr-immersive-system.md)** | Quest 3 WebXR architecture with hand tracking |

#### explanations/architecture/components/ (1 file)

| File | Description |
|------|-------------|
| **[websocket-protocol.md](explanations/architecture/components/websocket-protocol.md)** | Binary protocol design with 36-byte node format |

#### explanations/architecture/core/ (3 files)

| File | Description |
|------|-------------|
| **[client.md](explanations/architecture/core/client.md)** | React Three.js frontend architecture |
| **[server.md](explanations/architecture/core/server.md)** | Rust Actix backend with actor model |
| **[visualization.md](explanations/architecture/core/visualization.md)** | 3D rendering pipeline with WebGL |

#### explanations/architecture/decisions/ (1 file)

| File | Description |
|------|-------------|
| **[0001-neo4j-persistent-with-filesystem-sync.md](explanations/architecture/decisions/0001-neo4j-persistent-with-filesystem-sync.md)** | ADR: Database persistence strategy rationale |

#### explanations/architecture/gpu/ (3 files)

| File | Description |
|------|-------------|
| **[communication-flow.md](explanations/architecture/gpu/communication-flow.md)** | CPU-GPU data transfer optimization |
| **[optimizations.md](explanations/architecture/gpu/optimizations.md)** | Performance tuning strategies for CUDA |
| **[readme.md](explanations/architecture/gpu/readme.md)** | GPU subsystem overview and architecture |

#### explanations/architecture/ports/ (7 files)

*Hexagonal architecture interface definitions*

| File | Description |
|------|-------------|
| **[01-overview.md](explanations/architecture/ports/01-overview.md)** | Ports & adapters pattern overview |
| **[02-settings-repository.md](explanations/architecture/ports/02-settings-repository.md)** | User settings persistence port interface |
| **[03-knowledge-graph-repository.md](explanations/architecture/ports/03-knowledge-graph-repository.md)** | Graph operations port with Neo4j adapter |
| **[04-ontology-repository.md](explanations/architecture/ports/04-ontology-repository.md)** | Ontology storage port interface |
| **[05-inference-engine.md](explanations/architecture/ports/05-inference-engine.md)** | Reasoning engine port with Whelk adapter |
| **[06-gpu-physics-adapter.md](explanations/architecture/ports/06-gpu-physics-adapter.md)** | Physics computation port interface |
| **[07-gpu-semantic-analyzer.md](explanations/architecture/ports/07-gpu-semantic-analyzer.md)** | Semantic processing port for NLP |

### explanations/ontology/ (8 files)

| File | Description |
|------|-------------|
| **[client-side-hierarchical-lod.md](explanations/ontology/client-side-hierarchical-lod.md)** | Level-of-detail optimization for large graphs |
| **[enhanced-parser.md](explanations/ontology/enhanced-parser.md)** | OWL parser v2 with improved error handling |
| **[hierarchical-visualization.md](explanations/ontology/hierarchical-visualization.md)** | Tree and DAG layout algorithms |
| **[intelligent-pathfinding-system.md](explanations/ontology/intelligent-pathfinding-system.md)** | Graph traversal theory and implementation |
| **[neo4j-integration.md](explanations/ontology/neo4j-integration.md)** | Graph database integration patterns |
| **[ontology-pipeline-integration.md](explanations/ontology/ontology-pipeline-integration.md)** | End-to-end OWL processing pipeline |
| **[ontology-typed-system.md](explanations/ontology/ontology-typed-system.md)** | Type system design for ontology nodes |
| **[reasoning-engine.md](explanations/ontology/reasoning-engine.md)** | Inference engine concepts and algorithms |

### explanations/physics/ (2 files)

| File | Description |
|------|-------------|
| **[semantic-forces-actor.md](explanations/physics/semantic-forces-actor.md)** | Actor system integration for physics engine |
| **[semantic-forces.md](explanations/physics/semantic-forces.md)** | Physics constraint generation from semantics |

---

## 📖 Reference (22 files)

*Information-oriented: Technical specifications and APIs*

### reference/ Root (5 files)

| File | Description |
|------|-------------|
| **[api-complete-reference.md](reference/api-complete-reference.md)** | All REST endpoints with examples |
| **[code-quality-status.md](reference/code-quality-status.md)** | Build and test health dashboard |
| **[error-codes.md](reference/error-codes.md)** | Complete error code reference |
| **[implementation-status.md](reference/implementation-status.md)** | Feature completion matrix |
| **[performance-benchmarks.md](reference/performance-benchmarks.md)** | GPU performance metrics and analysis |
| **[physics-implementation.md](reference/physics-implementation.md)** | Physics system implementation details |
| **[websocket-protocol.md](reference/websocket-protocol.md)** | V2 binary protocol specification |

### reference/api/ (7 files)

| File | Description |
|------|-------------|
| **[01-authentication.md](reference/api/01-authentication.md)** | JWT, sessions, Nostr authentication |
| **[03-websocket.md](reference/api/03-websocket.md)** | Real-time binary protocol API |
| **[pathfinding-examples.md](reference/api/pathfinding-examples.md)** | Graph traversal API examples |
| **[readme.md](reference/api/readme.md)** | API documentation index |
| **[rest-api-complete.md](reference/api/rest-api-complete.md)** | HTTP API complete specification |
| **[rest-api-reference.md](reference/api/rest-api-reference.md)** | OpenAPI/Swagger format reference |
| **[semantic-features-api.md](reference/api/semantic-features-api.md)** | Natural language query API |

### reference/database/ (4 files)

| File | Description |
|------|-------------|
| **[neo4j-persistence-analysis.md](reference/database/neo4j-persistence-analysis.md)** | Migration analysis and recommendations |
| **[ontology-schema-v2.md](reference/database/ontology-schema-v2.md)** | Advanced ontology schema with types |
| **[schemas.md](reference/database/schemas.md)** | Neo4j graph schema definitions |
| **[user-settings-schema.md](reference/database/user-settings-schema.md)** | User data model and relationships |

### reference/protocols/ (1 file)

| File | Description |
|------|-------------|
| **[binary-websocket.md](reference/protocols/binary-websocket.md)** | 36-byte node format specification |

---

## 🗄️ Architecture (1 file)

| File | Description |
|------|-------------|
| **[HEXAGONAL_ARCHITECTURE_STATUS.md](architecture/HEXAGONAL_ARCHITECTURE_STATUS.md)** | Hexagonal architecture implementation status |

---

## 📊 Audits (4 files)

| File | Description |
|------|-------------|
| **[README.md](audits/README.md)** | Audits section overview |
| **[neo4j-migration-action-plan.md](audits/neo4j-migration-action-plan.md)** | Step-by-step migration action items |
| **[neo4j-migration-summary.md](audits/neo4j-migration-summary.md)** | Migration completion summary |
| **[neo4j-settings-migration-audit.md](audits/neo4j-settings-migration-audit.md)** | Settings migration audit results |

---

## 📦 Archive (50+ files)

*Historical documentation, completion reports, deprecated content*

See **[archive/README.md](archive/README.md)** for archive organization.

Key archive sections:
- **reports/** - Completion reports and summaries
- **sprint-logs/** - Sprint retrospectives and logs
- **fixes/** - Bug fix documentation
- **deprecated-patterns/** - Superseded patterns
- **working/** - Temporary working documents

---

## 🔧 Working (5 files)

*Work-in-progress documentation (not linked from main index)*

| File | Description |
|------|-------------|
| **[CLIENT_ARCHITECTURE_ANALYSIS.md](working/CLIENT_ARCHITECTURE_ANALYSIS.md)** | Client architecture analysis notes |
| **[CLIENT_DOCS_SUMMARY.md](working/CLIENT_DOCS_SUMMARY.md)** | Client documentation summary |
| **[DEPRECATION_PURGE.md](working/DEPRECATION_PURGE.md)** | Deprecation cleanup tracking |
| **[DEPRECATION_PURGE_COMPLETE.md](working/DEPRECATION_PURGE_COMPLETE.md)** | Deprecation cleanup completion report |
| **[DOCS_ROOT_CLEANUP.md](working/DOCS_ROOT_CLEANUP.md)** | Root directory cleanup notes |
| **[HISTORICAL_CONTEXT_RECOVERY.md](working/HISTORICAL_CONTEXT_RECOVERY.md)** | Historical context recovery process |

---

## 📐 Assets (1 file)

### assets/diagrams/

| File | Description |
|------|-------------|
| **[sparc-turboflow-architecture.md](assets/diagrams/sparc-turboflow-architecture.md)** | SPARC methodology system visualization |

---

## 🔍 Search Tips

**Find documentation by keyword:**
1. Use Ctrl+F on this page to search all file descriptions
2. Common search terms:
   - "install", "setup" → Installation guides
   - "API", "endpoint" → API references
   - "Neo4j", "database" → Database docs
   - "GPU", "CUDA" → GPU acceleration
   - "ontology", "OWL" → Semantic web
   - "agent", "MCP" → AI systems
   - "XR", "VR" → Immersive experiences
   - "test" → Testing documentation
   - "migrate", "migration" → Migration guides

**Find by role:**
- New users → tutorials/
- Developers → guides/developer/
- Architects → explanations/architecture/
- DevOps → guides/infrastructure/, guides/operations/

**Find by technology:**
- Neo4j → Search "neo4j", "database", "graph"
- Rust → Search "rust", "actix", "server"
- React → Search "react", "client", "three"
- CUDA → Search "gpu", "cuda", "physics"
- OWL → Search "ontology", "owl", "reasoning"

---

## 📊 Documentation Statistics

- **Total Files**: 226 markdown documents
- **Tutorials**: 3 learning-oriented guides
- **How-To Guides**: 61 task-oriented instructions
- **Explanations**: 75 understanding-oriented deep dives
- **Reference**: 22 information-oriented specifications
- **Archive**: 50+ historical documents
- **Last Updated**: 2025-12-02

---

**Last Updated**: 2025-12-02
**Purpose**: Fast keyword-searchable index of all documentation
**Maintained by**: DreamLab AI Documentation Team
