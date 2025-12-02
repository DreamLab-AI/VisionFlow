# VisionFlow Documentation Hub

**Complete documentation for VisionFlow's enterprise-grade multi-agent knowledge graphing system.**

This documentation uses the **Diátaxis** framework, organising content into four distinct categories:

- **🎓 Tutorials** - Learning-oriented, step-by-step guides for beginners
- **📘 How-To Guides** - Goal-oriented practical instructions for specific tasks
- **📕 Concepts** - Understanding-oriented explanations of architecture and design
- **📗 Reference** - Information-oriented technical specifications and APIs

---

## 🚀 Quick Start

**New to VisionFlow?** Start here:

1. **[Installation Guide](getting-started/01-installation.md)** - Set up VisionFlow in under 5 minutes
2. **[First Graph & Agents](getting-started/02-first-graph-and-agents.md)** - Create your first knowledge graph with AI agents
3. **[Navigation Guide](guides/navigation-guide.md)** - Master the 3D interface and controls

**Looking for something specific?** See the comprehensive index sections below.

---

## 📚 Documentation Index

### 🎓 Getting Started (Tutorials)

Learn by doing with step-by-step tutorials:

| Document | Description | Audience |
|----------|-------------|----------|
| **[Installation Guide](getting-started/01-installation.md)** | Docker and native installation for all platforms | All users |
| **[First Graph & Agents](getting-started/02-first-graph-and-agents.md)** | Create your first visualization with AI agents | Beginners |

### 📘 User Guides (How-To)

Accomplish specific goals:

#### Core Usage
| Document | Description |
|----------|-------------|
| **[Navigation Guide](guides/navigation-guide.md)** | 3D interface controls and navigation |
| **[Configuration](guides/configuration.md)** | Environment variables and settings |
| **[Troubleshooting](guides/troubleshooting.md)** | Common issues and solutions |
| **[XR Setup](archive/docs/guides/xr-setup.md)** | Quick XR/VR device configuration |

#### AI Agent System
| Document | Description |
|----------|-------------|
| **[Agent Orchestration](guides/agent-orchestration.md)** | Deploy and manage AI agent teams |
| **[Orchestrating Agents](guides/orchestrating-agents.md)** | Advanced coordination patterns |
| **[Multi-Agent Skills](guides/multi-agent-skills.md)** | Agent capabilities and specializations |

#### Neo4j & Data Integration
| Document | Description |
|----------|-------------|
| **[Neo4j Integration](guides/neo4j-integration.md)** | Work with graph database directly |
| **[Neo4j Implementation Roadmap](guides/neo4j-implementation-roadmap.md)** | Migration path and timeline |
| **[Neo4j Migration](guides/neo4j-migration.md)** | Step-by-step migration guide |

#### Ontology & Reasoning
| Document | Description |
|----------|-------------|
| **[Ontology Parser](guides/ontology-parser.md)** | Parse and validate OWL ontologies |
| **[Ontology Reasoning Integration](guides/ontology-reasoning-integration.md)** | Enable semantic inference |
| **[Ontology Storage Guide](guides/ontology-storage-guide.md)** | Persist ontologies in Neo4j |
| **[Ontology Semantic Forces](guides/ontology-semantic-forces.md)** | Visualize constraints as physics |

#### Advanced Features
| Document | Description |
|----------|-------------|
| **[Semantic Features Implementation](guides/semantic-features-implementation.md)** | Natural language queries and pathfinding |
| **[Hierarchy Integration](guides/hierarchy-integration.md)** | Hierarchical layouts and DAG rendering |
| **[Stress Majorization Guide](guides/stress-majorization-guide.md)** | Advanced layout algorithm |
| **[Pipeline Admin API](guides/pipeline-admin-api.md)** | GitHub sync and data pipeline control |

#### Deployment & Operations
| Document | Description |
|----------|-------------|
| **[Deployment](guides/deployment.md)** | Production deployment strategies |
| **[Docker Compose Guide](guides/docker-compose-guide.md)** | Multi-container orchestration |
| **[Docker Environment Setup](guides/docker-environment-setup.md)** | Container configuration |
| **[Pipeline Operator Runbook](guides/operations/pipeline-operator-runbook.md)** | Operations playbook |

#### Immersive XR & Multi-User
| Document | Description |
|----------|-------------|
| **[Vircadia XR Complete Guide](guides/vircadia-xr-complete-guide.md)** | Full Quest 3 implementation guide |
| **[Vircadia Multi-User Guide](guides/vircadia-multi-user-guide.md)** | Collaborative VR sessions |

#### Migration Guides
| Document | Description |
|----------|-------------|
| **[GraphServiceActor Migration](guides/graphserviceactor-migration.md)** | Modular actor system migration |
| **[JSON to Binary Protocol](guides/migration/json-to-binary-protocol.md)** | WebSocket protocol upgrade |

### 📕 Developer Guides

Build and extend VisionFlow:

#### Essential Reading
| Document | Description | Priority |
|----------|-------------|----------|
| **[Development Setup](guides/developer/01-development-setup.md)** | Configure development environment | ⭐⭐⭐ |
| **[Project Structure](guides/developer/02-project-structure.md)** | Navigate the codebase | ⭐⭐⭐ |
| **[Architecture](guides/developer/03-architecture.md)** | Understand system design | ⭐⭐⭐ |
| **[Adding Features](guides/developer/04-adding-features.md)** | Extend the system | ⭐⭐ |
| **[Contributing](guides/developer/06-contributing.md)** | Contribution guidelines | ⭐⭐ |

#### Development Workflow
| Document | Description |
|----------|-------------|
| **[Development Workflow](guides/development-workflow.md)** | Git branching, testing, CI/CD |
| **[Testing Guide](guides/testing-guide.md)** | Comprehensive testing strategies |
| **[Test Execution](guides/developer/test-execution.md)** | Run and debug tests |
| **[Extending the System](guides/extending-the-system.md)** | Plugin and extension patterns |

#### Technical Patterns
| Document | Description |
|----------|-------------|
| **[WebSocket Best Practices](guides/developer/websocket-best-practices.md)** | Binary protocol implementation |
| **[JSON Serialization Patterns](guides/developer/json-serialization-patterns.md)** | Rust serde patterns |

#### Security & Monitoring
| Document | Description |
|----------|-------------|
| **[Security](guides/security.md)** | Authentication, authorization, secrets |
| **[Telemetry Logging](guides/telemetry-logging.md)** | Observability and monitoring |

### 📙 Concepts (Understanding)

Learn the underlying architecture and design:

#### System Architecture
| Document | Description |
|----------|-------------|
| **[Architecture Overview](concepts/architecture/00-architecture-overview.md)** | Complete system design |
| **[Hexagonal CQRS Architecture](concepts/architecture/hexagonal-cqrs-architecture.md)** | Ports & adapters pattern |
| **[Services Architecture](concepts/architecture/services-architecture.md)** | Business logic layer |
| **[Multi-Agent System](concepts/architecture/multi-agent-system.md)** | AI agent coordination |
| **[Quick Reference](concepts/architecture/quick-reference.md)** | Architecture cheat sheet |

#### Data Flow & Integration
| Document | Description |
|----------|-------------|
| **[Data Flow Complete](concepts/architecture/data-flow-complete.md)** | End-to-end data pipeline |
| **[Pipeline Integration](concepts/architecture/pipeline-integration.md)** | GitHub sync to GPU rendering |
| **[Pipeline Sequence Diagrams](concepts/architecture/pipeline-sequence-diagrams.md)** | Visual data flow |
| **[Integration Patterns](concepts/architecture/integration-patterns.md)** | System integration approaches |
| **[GitHub Sync Service Design](concepts/architecture/github-sync-service-design.md)** | Streaming sync architecture |

#### Database & Persistence
| Document | Description |
|----------|-------------|
| **[Database Schemas](concepts/architecture/04-database-schemas.md)** | Neo4j schema design |
| **[Neo4j Integration](concepts/neo4j-integration.md)** | Graph database concepts |
| **[Adapter Patterns](concepts/architecture/adapter-patterns.md)** | Repository implementations |

#### Ontology & Reasoning
| Document | Description |
|----------|-------------|
| **[Ontology Storage Architecture](concepts/architecture/ontology-storage-architecture.md)** | OWL persistence in Neo4j |
| **[Ontology Reasoning Pipeline](concepts/architecture/ontology-reasoning-pipeline.md)** | Whelk inference engine |
| **[Reasoning Data Flow](concepts/architecture/reasoning-data-flow.md)** | Inference pipeline stages |
| **[Reasoning Tests Summary](concepts/architecture/reasoning-tests-summary.md)** | Test coverage and results |
| **[Ontology Pipeline Integration](concepts/ontology-pipeline-integration.md)** | End-to-end ontology processing |

#### Visualization & Physics
| Document | Description |
|----------|-------------|
| **[Semantic Physics System](concepts/architecture/semantic-physics-system.md)** | Force-directed layout engine |
| **[Semantic Forces System](concepts/architecture/semantic-forces-system.md)** | Physics constraint generation |
| **[Semantic Physics](concepts/architecture/semantic-physics.md)** | Physics theory and algorithms |
| **[Stress Majorization](concepts/architecture/stress-majorization.md)** | Advanced layout technique |
| **[Hierarchical Visualization](concepts/architecture/hierarchical-visualization.md)** | DAG and tree layouts |
| **[Client-Side Hierarchical LOD](concepts/client-side-hierarchical-lod.md)** | Level-of-detail optimization |

#### GPU Acceleration
| Document | Description |
|----------|-------------|
| **[GPU Semantic Forces](concepts/architecture/gpu-semantic-forces.md)** | CUDA kernel architecture |
| **[GPU Communication Flow](concepts/architecture/gpu/communication-flow.md)** | CPU-GPU data transfer |
| **[GPU Optimizations](concepts/architecture/gpu/optimizations.md)** | Performance tuning |
| **[GPU README](concepts/architecture/gpu/readme.md)** | GPU subsystem overview |

#### XR & Immersive
| Document | Description |
|----------|-------------|
| **[XR Immersive System](concepts/architecture/xr-immersive-system.md)** | Quest 3 WebXR architecture |

#### Client-Server Architecture
| Document | Description |
|----------|-------------|
| **[Client Architecture](concepts/architecture/core/client.md)** | React Three.js frontend |
| **[Server Architecture](concepts/architecture/core/server.md)** | Rust Actix backend |
| **[Visualization](concepts/architecture/core/visualization.md)** | 3D rendering pipeline |

#### Communication Protocols
| Document | Description |
|----------|-------------|
| **[WebSocket Protocol](concepts/architecture/components/websocket-protocol.md)** | Binary protocol design |

#### Ports (Hexagonal Architecture)
| Document | Description |
|----------|-------------|
| **[Ports Overview](concepts/architecture/ports/01-overview.md)** | Interface definitions |
| **[Settings Repository](concepts/architecture/ports/02-settings-repository.md)** | Settings persistence |
| **[Knowledge Graph Repository](concepts/architecture/ports/03-knowledge-graph-repository.md)** | Graph operations |
| **[Ontology Repository](concepts/architecture/ports/04-ontology-repository.md)** | Ontology storage |
| **[Inference Engine](concepts/architecture/ports/05-inference-engine.md)** | Reasoning interface |
| **[GPU Physics Adapter](concepts/architecture/ports/06-gpu-physics-adapter.md)** | Physics computation |
| **[GPU Semantic Analyzer](concepts/architecture/ports/07-gpu-semantic-analyzer.md)** | Semantic processing |

#### Architectural Patterns
| Document | Description |
|----------|-------------|
| **[CQRS Directive Template](concepts/architecture/cqrs-directive-template.md)** | Command/query separation |
| **[API Handlers Reference](concepts/architecture/api-handlers-reference.md)** | HTTP handler patterns |

### 📗 Reference (Technical Details)

Complete technical specifications:

#### API Documentation
| Document | Description |
|----------|-------------|
| **[REST API Reference](reference/api/rest-api-reference.md)** | Complete HTTP API |
| **[REST API Complete](reference/api/rest-api-complete.md)** | Extended API documentation |
| **[Authentication](reference/api/01-authentication.md)** | JWT and session management |
| **[WebSocket](reference/api/03-websocket.md)** | Real-time communication |
| **[Neo4j Quick Start](reference/api/neo4j-quick-start.md)** | Graph database queries |
| **[API Complete Reference](reference/api-complete-reference.md)** | All endpoints and schemas |

#### Protocols & Specifications
| Document | Description |
|----------|-------------|
| **[WebSocket Protocol](reference/websocket-protocol.md)** | Binary protocol V2 specification |
| **[Binary Protocol Specification](reference/binary-protocol-specification.md)** | 36-byte message format |

#### Error Handling
| Document | Description |
|----------|-------------|
| **[Error Codes](reference/error-codes.md)** | Complete error reference |

#### Performance & Benchmarks
| Document | Description |
|----------|-------------|
| **[Performance Benchmarks](reference/performance-benchmarks.md)** | GPU acceleration metrics |
| **[Semantic Physics Implementation](reference/semantic-physics-implementation.md)** | Physics performance analysis |

#### Implementation Status
| Document | Description |
|----------|-------------|
| **[Implementation Status](reference/implementation-status.md)** | Feature completion matrix |
| **[Code Quality Status](reference/code-quality-status.md)** | Build health and metrics |

---

## 🐳 Multi-Agent Docker System

AI agent orchestration container:

| Document | Description |
|----------|-------------|
| **[Architecture](multi-agent-docker/architecture.md)** | 54+ agent system design |
| **[Tools](multi-agent-docker/tools.md)** | Available MCP tools |
| **[Docker Environment](multi-agent-docker/docker-environment.md)** | Container configuration |
| **[Troubleshooting](multi-agent-docker/troubleshooting.md)** | Common agent issues |
| **[Port Configuration](multi-agent-docker/port-configuration.md)** | Network configuration |
| **[Goalie Integration](multi-agent-docker/goalie-integration.md)** | Quality gates and verification |

---

## 🔧 Analysis & Implementation

### Analysis Documents
| Document | Description |
|----------|-------------|
| **[Documentation Features](analysis/documentation-features.md)** | Documentation system analysis |
| **[GPU Implementation Audit](analysis/gpu-implementation-audit.md)** | CUDA kernel audit |
| **[Markdown to Graph Pipeline](analysis/markdown-to-graph-pipeline.md)** | Data ingestion analysis |

### Implementation Records
| Document | Description |
|----------|-------------|
| **[Enhanced Ontology Parser Implementation](enhanced-ontology-parser-implementation.md)** | OWL parser v2 |
| **[Hive Mind Integration Complete](HIVE_MIND_INTEGRATION_COMPLETE.md)** | Multi-agent coordination |
| **[Implementation Complete](IMPLEMENTATION_COMPLETE.md)** | Milestone delivery |
| **[Ontology Sync Enhancement](ONTOLOGY_SYNC_ENHANCEMENT.md)** | GitHub sync improvements |
| **[Services Layer Complete](services-layer-complete.md)** | Business logic refactor |

### Specialized Implementations
| Document | Description |
|----------|-------------|
| **[Neo4j Persistence Analysis](neo4j-persistence-analysis.md)** | Database migration analysis |
| **[Neo4j Rich Ontology Schema V2](neo4j-rich-ontology-schema-v2.md)** | Advanced schema design |
| **[Neo4j User Settings Schema](neo4j-user-settings-schema.md)** | User data model |
| **[Ruvector Integration Analysis](ruvector-integration-analysis.md)** | Vector database integration |
| **[Ontology Physics Integration Analysis](ontology-physics-integration-analysis.md)** | Physics-ontology coupling |
| **[Semantic Forces Actor Design](semantic-forces-actor-design.md)** | Actor system design |
| **[Quality Gates API Audit](quality-gates-api-audit.md)** | API quality assessment |
| **[Analytics Visualization Design](analytics-visualization-design.md)** | UI/UX design |

### Authentication & Settings
| Document | Description |
|----------|-------------|
| **[Nostr Auth Implementation](nostr-auth-implementation.md)** | Decentralized authentication |
| **[Auth User Settings](auth-user-settings.md)** | User authentication system |
| **[Settings Authentication](settings-authentication.md)** | Settings security |
| **[User Settings Implementation Summary](user-settings-implementation-summary.md)** | Settings system overview |

---

## 🛠️ Fixes & Known Issues

### Fix Documentation
| Document | Description |
|----------|-------------|
| **[Fixes README](fixes/README.md)** | Overview of all fixes |
| **[Quick Reference](fixes/quick-reference.md)** | Fix lookup table |
| **[Before After Comparison](fixes/before-after-comparison.md)** | Code transformation examples |
| **[Technical Details](fixes/technical-details.md)** | In-depth fix explanations |

### Rust Compilation Fixes
| Document | Description |
|----------|-------------|
| **[Type Corrections](fixes/type-corrections.md)** | Type system fixes |
| **[Type Corrections Progress](fixes/type-corrections-progress.md)** | Migration tracking |
| **[Type Corrections Final Summary](fixes/type-corrections-final-summary.md)** | Complete resolution |
| **[Rust Type Correction Guide](fixes/rust-type-correction-guide.md)** | Developer guide |
| **[Borrow Checker](fixes/borrow-checker.md)** | Lifetime issues |
| **[Borrow Checker Summary](fixes/borrow-checker-summary.md)** | Resolution summary |
| **[Actor Handlers](fixes/actor-handlers.md)** | Actor system fixes |
| **[PageRank Fix](fixes/pagerank-fix.md)** | Algorithm correction |

---

## 🎯 Features & Capabilities

### Feature Documentation
| Document | Description |
|----------|-------------|
| **[Client-Side Filtering](features/client-side-filtering.md)** | Real-time graph filtering |

### API Features
| Document | Description |
|----------|-------------|
| **[Semantic Features API](api/semantic-features-api.md)** | Natural language queries |
| **[Pathfinding Examples](api/pathfinding-examples.md)** | Graph traversal patterns |

---

## 🏛️ Architectural Decisions

### Architecture Decision Records
| Document | Description |
|----------|-------------|
| **[ADR-001: Neo4j Persistent with Filesystem Sync](architecture/decisions/ADR-001-neo4j-persistent-with-filesystem-sync.md)** | Database persistence strategy |

---

## 📦 Specialized Topics

### Client Architecture
| Document | Description |
|----------|-------------|
| **[Client TypeScript Architecture](specialized/client-typescript-architecture.md)** | Frontend architecture |
| **[Client Components Reference](specialized/client-components-reference.md)** | React component library |

### Extension System
| Document | Description |
|----------|-------------|
| **[Extension Guide](specialized/extension-guide.md)** | Plugin development |

---

## 🎓 Audits & Reviews

### System Audits
| Document | Description |
|----------|-------------|
| **[Audits README](audits/README.md)** | Audit overview |
| **[Neo4j Settings Migration Audit](audits/neo4j-settings-migration-audit.md)** | Settings migration review |
| **[Neo4j Migration Action Plan](audits/neo4j-migration-action-plan.md)** | Migration roadmap |
| **[Neo4j Migration Summary](audits/neo4j-migration-summary.md)** | Migration results |

---

## 📊 Diagrams & Assets

### Visual Documentation
| Location | Description |
|----------|-------------|
| **[Diagrams](diagrams/)** | System architecture diagrams |
| **[Assets](assets/)** | Screenshots, videos, graphics |
| **[SPARC Architecture Diagram](assets/diagrams/sparc-turboflow-architecture.md)** | SPARC methodology visualization |

---

## 🔍 Finding Documentation

### By Task
- **Install VisionFlow** → [Installation Guide](getting-started/01-installation.md)
- **Deploy AI Agents** → [Agent Orchestration](guides/agent-orchestration.md)
- **Query Neo4j** → [Neo4j Integration](guides/neo4j-integration.md)
- **Add New Feature** → [Adding Features](guides/developer/04-adding-features.md)
- **Fix Compilation Error** → [Type Corrections](fixes/type-corrections.md)
- **Set Up XR/VR** → [Vircadia XR Complete Guide](guides/vircadia-xr-complete-guide.md)

### By Role
- **End User** → Start with [Getting Started](getting-started/), then [User Guides](guides/)
- **Operator** → See [Deployment](guides/deployment.md) and [Operations](guides/operations/)
- **Developer** → Begin with [Developer Guides](guides/developer/) and [Concepts](concepts/)
- **Architect** → Read [Architecture Overview](concepts/architecture/00-architecture-overview.md) and [Concepts](concepts/)

### By Technology
- **Neo4j** → [Neo4j Integration](guides/neo4j-integration.md), [Database Schemas](concepts/architecture/04-database-schemas.md)
- **Rust/Actix** → [Server Architecture](concepts/architecture/core/server.md), [Project Structure](guides/developer/02-project-structure.md)
- **React/Three.js** → [Client Architecture](concepts/architecture/core/client.md), [Visualization](concepts/architecture/core/visualization.md)
- **CUDA/GPU** → [GPU Semantic Forces](concepts/architecture/gpu-semantic-forces.md), [GPU Optimizations](concepts/architecture/gpu/optimizations.md)
- **OWL/Ontologies** → [Ontology Storage Architecture](concepts/architecture/ontology-storage-architecture.md), [Ontology Parser](guides/ontology-parser.md)

---

## 📝 Documentation Issues

For known documentation problems and improvement tracking:

- **[Documentation Issues](DOCUMENTATION_ISSUES.md)** - Issue tracker and backlog

---

## 🤝 Contributing to Documentation

We welcome documentation improvements! Guidelines:

1. Follow the **Diátaxis** framework:
   - **Tutorials** - Learning by doing
   - **How-To Guides** - Problem solving
   - **Concepts** - Understanding
   - **Reference** - Information lookup

2. Use **UK English** spelling throughout
3. Include code examples with syntax highlighting
4. Add diagrams for complex concepts (Mermaid preferred)
5. Cross-reference related documents
6. Update this index when adding new documents

See **[Contributing Guide](guides/developer/06-contributing.md)** for complete guidelines.

---

## 📞 Getting Help

- **Documentation Issues**: File in GitHub Issues with `documentation` label
- **Technical Support**: See [Troubleshooting](guides/troubleshooting.md)
- **Community**: Join GitHub Discussions

---

**Last Updated**: 2025-12-02
**Documentation Version**: v2.0.0
**Total Documents**: 172 markdown files
