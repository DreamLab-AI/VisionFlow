# VisionFlow Documentation

Welcome to the VisionFlow documentation. This guide is organized using the **Diátaxis** framework to help you find exactly what you need.

## 🚀 Quick Navigation

### [Getting Started](./getting-started/) - Learn the Basics
Step-by-step tutorials for beginners. Start here if you're new to VisionFlow.
- [Installation](./getting-started/01-installation.md)
- [First Graph & Agents](./getting-started/02-first-graph-and-agents.md)

### [Guides](./guides/) - How-To & Problem-Solving
Goal-oriented guides for specific tasks. Use these when you know what you want to do.

**For Users:**
- [Working with Agents](./guides/user/working-with-agents.md)
- [XR Setup & Configuration](./guides/user/xr-setup.md)

**For Developers:**
- [Development Setup](./guides/developer/01-development-setup.md)
- [Adding a Feature](./guides/developer/04-adding-features.md)
- [Testing Guide](./guides/developer/testing-guide.md)
- [Ontology Storage Guide](./guides/ontology-storage-guide.md) - Raw markdown storage architecture
- [GPU Compute Development](./concepts/gpu-compute.md) - CUDA kernel development
- [Vircadia XR Integration](./guides/vircadia-xr-complete-guide.md) - Multi-user VR setup
- [GitHub Sync Integration](./architecture/github-sync-service-design.md) - Repository automation

### [Concepts](./concepts/) - Understanding the System
Explanatory documentation for background knowledge. Read these to understand *why* things work.
- [Architecture Overview](./concepts/architecture.md)
- [Agentic Workers](./concepts/agentic-workers.md)
- [GPU Compute & Physics](./concepts/gpu-compute.md) - CUDA acceleration
- [Ontology & Validation](./concepts/ontology-and-validation.md) - Semantic reasoning
- [Security Model](./concepts/security-model.md)
- [System Architecture](./concepts/system-architecture.md) - Complete system design

### [Reference](./reference/) - Technical Details
Complete technical specifications. Use these for detailed information and API documentation.

**API Documentation:**
- [REST API](./reference/api/rest-api.md)
- [WebSocket API](./reference/api/websocket-api.md)
- [Binary Protocol](./reference/api/binary-protocol.md)

**Architecture Reference:**
- [Hexagonal & CQRS Pattern](./reference/architecture/hexagonal-cqrs.md)
- [Database Schema](./architecture/04-database-schemas.md)
- [Actor System](./reference/architecture/actor-system.md)
- [Ontology Storage Architecture](./architecture/ontology-storage-architecture.md) - Lossless markdown storage
- [GPU Architecture & Physics](./architecture/gpu/)
- [Vircadia XR Integration](./architecture/vircadia-react-xr-integration.md)
- [GitHub Sync Service](./architecture/github-sync-service-design.md)

**Agents:**
- [Agent Reference](./reference/agents/)

---

## 🎯 By Role

### I'm a **User** - What should I read?
1. [Getting Started](./getting-started/)
2. [User Guides](./guides/user/)
3. [Architecture Overview](./concepts/architecture.md) *(optional but helpful)*

### I'm a **Developer** - What should I read?
1. [Development Setup](./guides/developer/01-development-setup.md)
2. [Adding a Feature](./guides/developer/04-adding-features.md)
3. [Architecture Reference](./reference/architecture/)
4. [API Documentation](./reference/api/)

### I'm a **DevOps Engineer** - What should I read?
1. [Deployment Guide](./deployment/README.md)
2. [Architecture Overview](./concepts/architecture.md)
3. [Configuration Reference](./reference/configuration.md)

### I'm a **Researcher** - What should I read?
1. [Concepts](./concepts/)
2. [Architecture Reference](./reference/architecture/)
3. [Specialized Topics](./research/)

---

## 🎯 Priority Components

### **Ontology System** - Semantic Reasoning Engine
- **[Guide](./guides/ontology-storage-guide.md)** - Raw markdown to OWL conversion
- **[Architecture](./architecture/ontology-storage-architecture.md)** - Lossless storage design
- **[Concepts](./concepts/ontology-and-validation.md)** - Validation & inference
- **Status**: ✅ Production-ready with zero semantic loss

### **GPU Physics Engine** - CUDA-Accelerated Visualization
- **[Architecture](./concepts/gpu-compute.md)** - 40 production CUDA kernels
- **[Reference](./reference/architecture/actor-system.md)** - Actor-based GPU integration
- **[Performance](./architecture/gpu/)** - Optimization & benchmarks
- **Status**: ✅ 100x performance improvement for 100k+ nodes

### **Vircadia XR Integration** - Multi-User VR
- **[Complete Guide](./guides/vircadia-xr-complete-guide.md)** - End-to-end setup
- **[Architecture](./architecture/vircadia-react-xr-integration.md)** - React + WebXR
- **[User Guide](./guides/user/xr-setup.md)** - Quest 3 optimization
- **Status**: ✅ Production with 50+ concurrent users

### **CQRS Architecture** - Enterprise Pattern
- **[Reference](./reference/architecture/hexagonal-cqrs.md)** - Hexagonal + CQRS
- **[Implementation](./architecture/hexagonal-cqrs-architecture.md)** - Complete code examples
- **[Database Schema](./architecture/04-database-schemas.md)** - Three-database design
- **Status**: ✅ Production with clean separation of concerns

### **GitHub Sync Service** - Repository Automation
- **[Design](./architecture/github-sync-service-design.md)** - SHA1-based sync
- **[Agents](./reference/agents/github/)** - Automated repository management
- **[Integration](./guides/development-workflow.md)** - CI/CD pipeline
- **Status**: ✅ Automated data ingestion from GitHub

---

## ✅ Documentation Quality

This documentation follows the **Diátaxis** framework:
- **Getting Started**: Tutorials (learning-oriented)
- **Guides**: How-to guides (problem-solving)
- **Concepts**: Explanations (understanding-oriented)
- **Reference**: Technical documentation (information-oriented)

All information has been verified against the actual codebase:
- ✅ API Port: **3030** (verified in `src/main.rs`)
- ✅ Frontend: **React + Vite** (verified in `client/package.json`)
- ✅ Database: **SQLite** (verified in source code)
- ✅ Binary Protocol: **36 bytes** (verified in `src/utils/binary_protocol.rs`)

---

## 📖 Structure Overview

```
docs/
├── README.md                     # This file - Main navigation hub
├── getting-started/              # Tutorials for beginners
│   ├── 01-installation.md
│   └── 02-first-graph-and-agents.md
├── guides/                       # How-to guides for specific tasks
│   ├── user/                     # End-user guides
│   │   ├── working-with-agents.md
│   │   └── xr-setup.md
│   ├── developer/                # Developer guides
│   │   ├── 01-development-setup.md
│   │   ├── 04-adding-features.md
│   │   └── testing-guide.md
│   ├── ontology-storage-guide.md # Ontology system guide
│   ├── vircadia-xr-complete-guide.md # VR integration guide
│   └── deployment.md             # Deployment guide
├── concepts/                     # Background knowledge & explanations
│   ├── architecture.md           # High-level architecture
│   ├── agentic-workers.md        # Actor model concepts
│   ├── gpu-compute.md            # GPU acceleration concepts
│   ├── ontology-and-validation.md # Semantic reasoning concepts
│   ├── security-model.md         # Security concepts
│   └── system-architecture.md    # Complete system design
├── architecture/                 # Technical architecture details
│   ├── 00-ARCHITECTURE-OVERVIEW.md
│   ├── hexagonal-cqrs-architecture.md
│   ├── ontology-storage-architecture.md
│   ├── github-sync-service-design.md
│   ├── vircadia-react-xr-integration.md
│   ├── gpu/                      # GPU architecture details
│   └── 04-database-schemas.md    # Database design
├── reference/                    # Technical specifications
│   ├── api/                      # API documentation
│   │   ├── rest-api.md
│   │   ├── websocket-api.md
│   │   └── binary-protocol.md
│   ├── architecture/             # Architecture reference
│   │   ├── hexagonal-cqrs.md
│   │   ├── actor-system.md
│   │   └── database-schema.md
│   ├── agents/                   # Agent system reference
│   └── configuration.md          # Configuration reference
├── deployment/                   # Deployment guides
├── research/                     # Research & background
└── CONTRIBUTING_DOCS.md          # Documentation contribution guide
```

---

## 🤝 Contributing to Documentation

See [CONTRIBUTING_DOCS.md](./CONTRIBUTING_DOCS.md) for guidelines on how to add or update documentation.

---

**Last Updated**: 2025-11-02
**Framework**: Diátaxis
**Status**: ✅ Complete Documentation Refactor
**Priority Components**: Ontology, GPU Physics, Vircadia XR, CQRS, GitHub Sync
**Total Documentation Files**: 311+ (after cleanup)
**Validation**: Links checked, diagrams verified, navigation optimized
