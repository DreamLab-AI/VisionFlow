# VisionFlow Architecture Diagrams - Complete Reference

**Version:** 1.0.0
**Last Updated:** 2025-10-27
**Status:** Production-Ready Documentation

---

## Overview

This directory contains comprehensive Mermaid diagrams documenting the **ACTUAL** VisionFlow architecture, deployment infrastructure, and SPARC development methodology. All diagrams are based on the current production codebase and verified against implementation.

---

## Document Index

### 📐 [System Architecture](./system-architecture.md)
**8 Core Architecture Diagrams**

Hexagonal architecture, CQRS pattern, and three-database design.

| Diagram | Description | Key Components |
|---------|-------------|----------------|
| **1. System Architecture Overview** | Complete system layers from client to GPU | React Client, Actix-Web, CQRS, Ports/Adapters, 3 Databases, CUDA Kernels |
| **2. Hexagonal Architecture** | Ports and Adapters pattern | Inbound/Outbound Adapters, Domain Ports, Repository Pattern |
| **3. Component Interaction** | Data flow with CQRS | Sequence diagram: Client → Handler → Directive → Port → Adapter → Database |
| **4. Binary WebSocket Protocol** | Real-time communication flow | 36-byte binary protocol, GPU updates, <10ms latency |
| **5. Three-Database Architecture** | Database separation rationale | settings.db, knowledge_graph.db, ontology.db |
| **6. Actor System Integration** | Legacy actor system wrapper | Graph Actor, Physics Actor, GPU Actors, Supervisor |
| **7. Deployment Architecture** | Docker container topology | VisionFlow container, External services, GPU access |
| **8. API Endpoint Architecture** | REST and WebSocket APIs | /api/*, /ws/*, Handler modules |

**Focus:** Database-first, server-authoritative, hexagonal design
**Status:** ✅ Verified against production code

---

### 🔄 [Data Flow & Deployment](./data-flow-deployment.md)
**9 Data Pipeline and Deployment Diagrams**

File processing, real-time synchronization, and container infrastructure.

| Diagram | Description | Key Components |
|---------|-------------|----------------|
| **1. File Processing Pipeline** | Markdown and OWL ingestion | Logseq files, GitHub sync, Metadata extraction, Vector embeddings |
| **2. Real-Time Data Sync** | Multi-user collaboration flow | WebSocket broadcasting, Physics loop, Binary protocol |
| **3. GPU Processing Pipeline** | CUDA compute streams | 4 concurrent streams, 39 production kernels, DMA transfers |
| **4. External Service Integration** | Third-party API connections | RAGFlow, MCP Orchestrator, GitHub, Nostr, Qdrant |
| **5. Docker Compose Topology** | Multi-container architecture | VisionFlow dev/prod, Cloudflare tunnel, Network discovery |
| **6. Network Port Mapping** | Port exposure and routing | 3001 (dev), 4000 (API), Internal services |
| **7. Volume Mount Strategy** | Persistent storage design | Named volumes, Bind mounts, Hot reload |
| **8. Development vs Production** | Deployment profiles | Dev (hot reload), Prod (optimized build) |
| **9. CI/CD & Build Pipeline** | Build and deployment flow | Multi-stage Docker build, Cargo + Vite + CUDA |

**Focus:** Real-time data pipelines, GPU acceleration, container orchestration
**Status:** ✅ Verified against docker-compose.yml

---

### 🚀 [SPARC & Turbo Flow](./sparc-turboflow-architecture.md)
**8 Development Workflow and Agent Orchestration Diagrams**

SPARC methodology, agent coordination, and Turbo Flow integration.

| Diagram | Description | Key Components |
|---------|-------------|----------------|
| **1. SPARC Development Workflow** | 5-phase development cycle | Specification → Pseudocode → Architecture → Refinement → Completion |
| **2. Agent Orchestration Topology** | Multi-agent coordination | Claude Code Task tool, MCP coordination, 6 concurrent agents |
| **3. Turbo Flow Container Architecture** | Unified development environment | 4-user isolation, Supervisord, Claude Code skills, 610+ agents |
| **4. VisionFlow + Turbo Flow Integration** | Cross-container communication | docker_ragflow network, MCP relay, Service discovery |
| **5. Work Chunking Protocol (WCP)** | Task decomposition strategy | EPIC → Features → 10-minute tasks, 100% CI pass |
| **6. Agent Hook Integration** | Pre/post task automation | Hook system, Memory store, Neural pattern training |
| **7. SPARC + TDD Integration** | Test-Driven Development cycle | Red → Green → Refactor, CI pipeline validation |
| **8. Performance Metrics** | Optimization and monitoring | Token usage, Bottleneck analysis, 84.8% SWE-Bench score |

**Focus:** Agent coordination, SPARC methodology, performance optimization
**Status:** ✅ Verified against CLAUDE.md and claude-flow

---

## Quick Navigation

### By Role

**🎨 Frontend Developer**
- System Architecture Overview (diagram 1)
- Binary WebSocket Protocol (diagram 4)
- API Endpoint Architecture (diagram 8)
- Real-Time Data Sync (data-flow diagram 2)

**⚙️ Backend Developer**
- Hexagonal Architecture (diagram 2)
- Component Interaction (diagram 3)
- Three-Database Architecture (diagram 5)
- File Processing Pipeline (data-flow diagram 1)

**🔧 DevOps Engineer**
- Deployment Architecture (diagram 7)
- Docker Compose Topology (data-flow diagram 5)
- Network Port Mapping (data-flow diagram 6)
- Volume Mount Strategy (data-flow diagram 7)
- CI/CD Pipeline (data-flow diagram 9)

**🧠 AI/ML Developer**
- GPU Processing Pipeline (data-flow diagram 3)
- Agent Orchestration Topology (SPARC diagram 2)
- Agent Hook Integration (SPARC diagram 6)
- Performance Metrics (SPARC diagram 8)

**📐 System Architect**
- System Architecture Overview (diagram 1)
- Hexagonal Architecture (diagram 2)
- SPARC Development Workflow (SPARC diagram 1)
- VisionFlow + Turbo Flow Integration (SPARC diagram 4)

**🧪 QA Engineer**
- Component Interaction (diagram 3)
- SPARC + TDD Integration (SPARC diagram 7)
- Work Chunking Protocol (SPARC diagram 5)
- CI/CD Pipeline (data-flow diagram 9)

---

## Architecture Principles

### Database-First Design
All diagrams reflect the **three-database architecture**:
- `settings.db` - Configuration and preferences
- `knowledge_graph.db` - Nodes, edges, file metadata
- `ontology.db` - OWL classes, properties, axioms

**Server-Authoritative:** Single source of truth, no client-side caching.

### Hexagonal Architecture
**Ports and Adapters pattern** throughout:
- **Ports** define domain interfaces (traits)
- **Adapters** implement infrastructure (SQLite, GPU, Actors)
- **CQRS** separates Directives (write) and Queries (read)

**Clean separation** between business logic and frameworks.

### GPU-First Performance
**39 production CUDA kernels** across 4 concurrent streams:
1. **Physics Stream:** Force computation, collision detection, integration
2. **Clustering Stream:** Leiden algorithm, community detection
3. **Pathfinding Stream:** SSSP shortest paths, multi-hop reasoning
4. **Constraint Stream:** Ontology validation, physics forces

**Performance metrics:**
- 100x speedup over CPU
- 60 FPS at 100k+ nodes
- Sub-10ms WebSocket latency

### Binary Protocol Optimization
**Protocol V2:** 36 bytes per node update
- 80% bandwidth reduction vs 200-byte JSON
- Header (4) + node_id (4) + xyz (12) + metadata (16)
- Real-time broadcasting to all connected clients

### SPARC Methodology
**5-phase systematic development:**
1. **Specification** - Requirements analysis, microtask breakdown
2. **Pseudocode** - Algorithm design, language-agnostic logic
3. **Architecture** - System design, ports & adapters
4. **Refinement** - TDD cycle (Red → Green → Refactor)
5. **Completion** - Integration, deployment, documentation

**Work Chunking Protocol (WCP):**
- EPIC → Features (1-3 days) → Tasks (10 minutes)
- Require 100% CI pass before next feature
- Spawn swarm for complex features (2+ issues)

---

## Performance Characteristics

### System Performance
| Metric | Value | Diagram Reference |
|--------|-------|-------------------|
| **Frame Rate** | 60 FPS @ 100k nodes | GPU Processing Pipeline |
| **WebSocket Latency** | <10ms | Binary WebSocket Protocol |
| **GPU Speedup** | 100x vs CPU | GPU Processing Pipeline |
| **Bandwidth Reduction** | 80% (binary vs JSON) | Binary WebSocket Protocol |
| **Concurrent Users** | 50+ real-time | Real-Time Data Sync |

### Development Performance
| Metric | Value | Diagram Reference |
|--------|-------|-------------------|
| **SWE-Bench Solve Rate** | 84.8% | Performance Metrics |
| **Development Speed** | 2.8-4.4x faster | Performance Metrics |
| **Token Reduction** | 32.3% via batching | Performance Metrics |
| **Truth Verification** | >95% accuracy | SPARC + TDD Integration |
| **CI/CD Success Rate** | >90% | CI/CD Pipeline |

---

## Technology Stack

### Frontend
- **React** + **Three.js** (React Three Fiber)
- **WebGL** 3D rendering
- **Binary WebSocket Protocol V2**
- **Vite** for hot module reload

### Backend
- **Rust** + **Actix-Web**
- **Hexagonal Architecture** (hexser crate)
- **CQRS Pattern** (Directives & Queries)
- **Three SQLite Databases** (WAL mode)

### GPU Compute
- **CUDA Toolkit** (39 production kernels)
- **cuDNN** for deep learning
- **4 Concurrent Streams** for parallelism
- **Pinned Host Memory** for DMA

### AI Orchestration
- **Claude Code** primary development
- **MCP Protocol** for agent coordination
- **610+ Agent Templates** in Markdown
- **Claude Flow** hooks and memory system

### Ontology & Semantic
- **OWL/RDF** definitions
- **Whelk Reasoner** (Rust implementation)
- **Horned-OWL** parser
- **Logical inference** and validation

### Container Infrastructure
- **Docker Compose** multi-container
- **Supervisord** process management
- **Nginx** reverse proxy
- **Cloudflare Tunnel** for external access

---

## Diagram Rendering

### Mermaid Live Editor
All diagrams can be rendered at: https://mermaid.live/

### VS Code Extensions
- **Markdown Preview Mermaid Support**
- **Mermaid Markdown Syntax Highlighting**

### GitHub Rendering
GitHub natively renders Mermaid diagrams in Markdown files.

### Export Options
- **PNG/SVG:** Use Mermaid CLI or Live Editor
- **PDF:** Export via browser print to PDF
- **Draw.io:** Import Mermaid syntax

---

## Migration Status

**Current Architecture Version:** 3.1.0
**Migration Phase:** ✅ Completed
**Last Verified:** 2025-10-25

All legacy file-based configuration has been migrated to database-backed storage. The hexagonal architecture is fully implemented with CQRS pattern and three-database separation.

**Migration Highlights:**
- ✅ Settings: YAML → settings.db
- ✅ Graph: In-memory → knowledge_graph.db
- ✅ Ontology: File parsing → ontology.db
- ✅ Architecture: Monolithic → Hexagonal (ports & adapters)
- ✅ API: Direct actor calls → CQRS pattern
- ✅ Client: Cached state → Server-authoritative

**See:** [../MIGRATION_PLAN.md](../MIGRATION_PLAN.md) for complete history.

---

## Related Documentation

### Core Architecture
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Hexagonal design details
- [DATABASE.md](../DATABASE.md) - Database schemas and migrations
- [API.md](../API.md) - REST and WebSocket API reference
- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) - Development workflows

### Turbo Flow Integration
- [multi-agent-docker/CLAUDE.md](../../multi-agent-docker/CLAUDE.md) - Container configuration
- [multi-agent-docker/docs/developer/architecture.md](../../multi-agent-docker/docs/developer/architecture.md) - Multi-user system
- [multi-agent-docker/devpods/claude-flow-quick-reference.md](../../multi-agent-docker/devpods/claude-flow-quick-reference.md) - CLI commands

### SPARC Methodology
- [multi-agent-docker/devpods/SPARC_Methodology_Example.md](../../multi-agent-docker/devpods/SPARC_Methodology_Example.md) - Complete workflow
- [agents/doc-planner.md](../../agents/doc-planner.md) - Planning methodology
- [agents/microtask-breakdown.md](../../agents/microtask-breakdown.md) - Task decomposition

---

## Changelog

### Version 1.0.0 (2025-10-27)
- ✅ Initial comprehensive diagram set
- ✅ 25 total Mermaid diagrams across 3 documents
- ✅ Verified against production codebase
- ✅ Covers system architecture, data flow, SPARC workflow
- ✅ Includes Turbo Flow multi-agent integration
- ✅ Performance metrics and optimization patterns

**Created by:** Diagram Specialist Agent
**Verified by:** System Architect Agent
**Approved for:** Production documentation

---

## Contributing

When updating diagrams:

1. **Verify against code:** All diagrams must reflect ACTUAL implementation
2. **Use Mermaid syntax:** Keep diagrams as code for version control
3. **Update metrics:** Performance numbers must be current
4. **Cross-reference docs:** Link to relevant architecture documents
5. **Test rendering:** Verify on GitHub, VS Code, Mermaid Live

**Style Guide:**
- Use consistent color palette (defined in diagrams)
- Include legends for complex diagrams
- Add notes for key architectural decisions
- Keep diagram complexity manageable (split if needed)

---

## Questions & Support

**For diagram-related questions:**
- Check the relevant architecture document first
- Review the diagram legend and notes
- Consult the developer guide for implementation details

**For technical implementation:**
- See [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md)
- Review source code in `/src/` directory
- Check database schemas in `/schema/` directory

**For SPARC methodology:**
- Load `agents/doc-planner.md`
- Load `agents/microtask-breakdown.md`
- Review `SPARC_Methodology_Example.md`

---

**VisionFlow** - Immersive Multi-User Multi-Agent Knowledge Graphing
**Turbo Flow** - Comprehensive Agentic Development Environment

*Architecture diagrams are living documentation. Keep them updated as the system evolves.*
