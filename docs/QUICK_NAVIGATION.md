# Quick Navigation Guide

**Fast access to common documentation tasks**

**Last Updated:** November 3, 2025

---

## 🚀 I Want To...

### Get Started

| Task | Document | Time |
|------|----------|------|
| **Install VisionFlow** | [Installation Guide](getting-started/01-installation.md) | 15 min |
| **Create my first graph** | [First Graph & Agents](getting-started/02-first-graph-and-agents.md) | 20 min |
| **Set up development environment** | [Development Setup](guides/developer/01-development-setup.md) | 30 min |
| **Deploy with Docker** | [Docker Compose Usage](DOCKER_COMPOSE_UNIFIED_USAGE.md) | 10 min |

---

### Understand the Architecture

| Task | Document | Time |
|------|----------|------|
| **Get architecture overview** | [Architecture Overview](architecture/00-ARCHITECTURE-OVERVIEW.md) | 15 min |
| **Understand CQRS pattern** | [Hexagonal CQRS](architecture/hexagonal-cqrs-architecture.md) | 20 min |
| **Learn about GPU physics** | [Semantic Physics](semantic-physics-architecture.md) | 25 min |
| **Understand ontology system** | [Ontology Pipeline](ONTOLOGY_PIPELINE_INTEGRATION.md) | 30 min |
| **See database schemas** | [Database Schemas](database-schema-diagrams.md) | 10 min |

---

### Work with APIs

| Task | Document | Time |
|------|----------|------|
| **Use REST API** | [REST API Reference](api/rest-api-reference.md) | 15 min |
| **Connect via WebSocket** | [WebSocket API](api/03-websocket.md) | 10 min |
| **Understand binary protocol** | [Binary Protocol](architecture/components/websocket-protocol.md) | 15 min |
| **Authenticate requests** | [Authentication](api/01-authentication.md) | 5 min |

---

### Develop Features

| Task | Document | Time |
|------|----------|------|
| **Add a new feature** | [Adding Features](guides/developer/04-adding-features.md) | 20 min |
| **Understand project structure** | [Project Structure](guides/developer/02-project-structure.md) | 10 min |
| **Write tests** | [Testing Guide](guides/developer/05-testing.md) | 15 min |
| **Run test suite** | [Test Execution](TEST_EXECUTION_GUIDE.md) | 10 min |
| **Contribute code** | [Contributing](guides/developer/06-contributing.md) | 10 min |

---

### Configure & Deploy

| Task | Document | Time |
|------|----------|------|
| **Configure system** | [Configuration Guide](guides/configuration.md) | 15 min |
| **Deploy to production** | [Deployment Guide](guides/deployment.md) | 25 min |
| **Set up Neo4j** | [Neo4j Quick Start](NEO4J_QUICK_START.md) | 10 min |
| **Configure Docker environment** | [Docker Environment](multi-agent-docker/DOCKER-ENVIRONMENT.md) | 15 min |

---

### Troubleshoot Issues

| Task | Document | Time |
|------|----------|------|
| **General troubleshooting** | [Troubleshooting](guides/troubleshooting.md) | Variable |
| **Docker issues** | [Docker Troubleshooting](multi-agent-docker/TROUBLESHOOTING.md) | Variable |
| **Test failures** | [Testing Status](guides/developer/04-testing-status.md) | 5 min |
| **Pipeline operations** | [Pipeline Runbook](operations/PIPELINE_OPERATOR_RUNBOOK.md) | 20 min |

---

### Extend the System

| Task | Document | Time |
|------|----------|------|
| **Add ontology reasoning** | [Ontology Integration](ontology_reasoning_integration_guide.md) | 30 min |
| **Extend GPU physics** | [GPU Architecture](architecture/gpu/README.md) | 20 min |
| **Add new parsers** | [Ontology Parser](guides/ontology-parser.md) | 15 min |
| **Build custom agents** | [Agent Orchestration](guides/agent-orchestration.md) | 25 min |

---

## 📚 By Role

### I'm a **New User**

**Start Here:**
1. [README.md](README.md) - Project overview
2. [Installation](getting-started/01-installation.md) - Get it running
3. [First Graph](getting-started/02-first-graph-and-agents.md) - Learn basics

**Next Steps:**
- [Working with Agents](guides/user/working-with-agents.md)
- [XR Setup](guides/user/xr-setup.md)
- [Troubleshooting](guides/troubleshooting.md)

---

### I'm a **Developer**

**Start Here:**
1. [Development Setup](guides/developer/01-development-setup.md) - Environment setup
2. [Project Structure](guides/developer/02-project-structure.md) - Code organization
3. [Architecture](guides/developer/03-architecture.md) - System design

**Deep Dives:**
- [Adding Features](guides/developer/04-adding-features.md)
- [Testing Guide](guides/developer/05-testing.md)
- [CQRS Architecture](architecture/hexagonal-cqrs-architecture.md)
- [GPU Development](architecture/gpu/README.md)

---

### I'm an **Operator**

**Start Here:**
1. [Deployment Guide](guides/deployment.md) - Production deployment
2. [Pipeline Runbook](operations/PIPELINE_OPERATOR_RUNBOOK.md) - Operations guide
3. [Test Execution](TEST_EXECUTION_GUIDE.md) - Validation procedures

**Operations:**
- [Configuration](guides/configuration.md)
- [Docker Compose](DOCKER_COMPOSE_UNIFIED_USAGE.md)
- [Neo4j Setup](NEO4J_QUICK_START.md)
- [Troubleshooting](guides/troubleshooting.md)

---

### I'm a **Contributor**

**Start Here:**
1. [Contributing Docs](CONTRIBUTING_DOCS.md) - Documentation guidelines
2. [Developer Guide](guides/developer/06-contributing.md) - Code contribution
3. [Roadmap](ROADMAP.md) - Project status and plans

**Reference:**
- [Test Execution](TEST_EXECUTION_GUIDE.md)
- [Architecture Overview](architecture/00-ARCHITECTURE-OVERVIEW.md)
- [API Reference](api/rest-api-reference.md)

---

## 🔍 By Technology

### Ontology & Reasoning

| Document | Purpose |
|----------|---------|
| [Ontology Pipeline Integration](ONTOLOGY_PIPELINE_INTEGRATION.md) | Complete pipeline documentation |
| [Ontology Reasoning Guide](ontology_reasoning_integration_guide.md) | Integration guide |
| [Ontology Parser](guides/ontology-parser.md) | Parser implementation |
| [Ontology Storage](guides/ontology-storage-guide.md) | Storage architecture |

### GPU & Physics

| Document | Purpose |
|----------|---------|
| [Semantic Physics Architecture](semantic-physics-architecture.md) | Physics system design |
| [GPU Architecture](architecture/gpu/README.md) | GPU subsystem overview |
| [GPU Communication](architecture/gpu/communication-flow.md) | Communication patterns |
| [GPU Optimizations](architecture/gpu/optimizations.md) | Performance tuning |

### Database

| Document | Purpose |
|----------|---------|
| [Database Schemas](database-schema-diagrams.md) | Complete schema diagrams |
| [Database Architecture](architecture/04-database-schemas.md) | Architecture details |
| [Neo4j Integration](guides/neo4j-integration.md) | Graph database setup |
| [Neo4j Quick Start](NEO4J_QUICK_START.md) | Quick setup guide |

### Visualization

| Document | Purpose |
|----------|---------|
| [Client-Side LOD](CLIENT_SIDE_HIERARCHICAL_LOD.md) | Hierarchical rendering |
| [Visualization System](architecture/core/visualization.md) | Client architecture |
| [XR Setup](guides/xr-setup.md) | VR/AR configuration |
| [Vircadia Integration](guides/vircadia-xr-complete-guide.md) | Multi-user VR |

---

## 🗂️ By Category

### Core Documentation

- [README.md](README.md) - Main documentation
- [INDEX.md](INDEX.md) - Complete file index
- [ROADMAP.md](ROADMAP.md) - Project roadmap
- [REFACTORING_NOTES.md](REFACTORING_NOTES.md) - Documentation changes

### Architecture

- [Architecture Overview](architecture/00-ARCHITECTURE-OVERVIEW.md)
- [Hexagonal CQRS](architecture/hexagonal-cqrs-architecture.md)
- [Data Flow](architecture/data-flow-complete.md)
- [Pipeline Integration](architecture/PIPELINE_INTEGRATION.md)

### API Documentation

- [REST API](api/rest-api-reference.md)
- [WebSocket API](api/03-websocket.md)
- [Complete API](api/rest-api-complete.md)
- [Authentication](api/01-authentication.md)

### Guides

- [Developer Guides](guides/developer/)
- [User Guides](guides/user/)
- [Configuration](guides/configuration.md)
- [Deployment](guides/deployment.md)

---

## 🔧 Common Tasks

### Installation & Setup
```bash
# Quick start
git clone <repository>
cd visionflow
cargo build --release

# See full guide
→ getting-started/01-installation.md
```

### Run Tests
```bash
# All tests
cargo test

# Specific module
cargo test --package visionflow --lib ontology

# See full guide
→ TEST_EXECUTION_GUIDE.md
```

### Start Development
```bash
# Backend
cargo run

# Frontend
cd client && npm run dev

# See full guide
→ guides/developer/01-development-setup.md
```

### Deploy with Docker
```bash
docker-compose up -d

# See full guide
→ DOCKER_COMPOSE_UNIFIED_USAGE.md
```

---

## 🔎 Search Tips

### Find by Keyword

**Architecture:** Search in `architecture/` directory
```bash
grep -r "keyword" docs/architecture/
```

**APIs:** Search in `api/` directory
```bash
grep -r "endpoint" docs/api/
```

**Guides:** Search in `guides/` directory
```bash
grep -r "how to" docs/guides/
```

### Find by File Type

**All markdown files:**
```bash
find docs/ -name "*.md"
```

**Implementation guides:**
```bash
find docs/guides/ -name "*.md"
```

**Architecture docs:**
```bash
find docs/architecture/ -name "*.md"
```

---

## 📖 Full Navigation

For complete documentation navigation, see:

**[📖 Master Index (INDEX.md)](INDEX.md)**

The master index provides:
- Complete file listing (162 documents)
- Hierarchical organization
- Topic-based grouping
- Role-based paths
- Full archive access

---

## 🆘 Need Help?

### Can't Find What You're Looking For?

1. **Check the Master Index:** [INDEX.md](INDEX.md)
2. **Search by role:** See "By Role" section above
3. **Search by technology:** See "By Technology" section above
4. **Browse guides:** [guides/](guides/)
5. **Check archive:** [archive/historical-reports/](archive/historical-reports/)

### Document Issues

- **Missing information?** Check [ROADMAP.md](ROADMAP.md) for planned features
- **Broken links?** See [REFACTORING_NOTES.md](REFACTORING_NOTES.md)
- **Need to contribute?** See [CONTRIBUTING_DOCS.md](CONTRIBUTING_DOCS.md)

---

**Navigation:**
- [📖 Main README](README.md)
- [📚 Master Index](INDEX.md)
- [🗺️ Roadmap](ROADMAP.md)
- [🔧 Refactoring Notes](REFACTORING_NOTES.md)

**Last Updated:** November 3, 2025
**Maintained By:** Documentation Team
