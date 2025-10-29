# VisionFlow Documentation

Welcome to the VisionFlow documentation. This guide is organized using the **Diátaxis** framework to help you find exactly what you need.

## 🚀 Quick Navigation

### [Getting Started](./getting-started/) - Learn the Basics
Step-by-step tutorials for beginners. Start here if you're new to VisionFlow.
- [Installation](./getting-started/01-installation.md)
- [First Graph & Agents](./getting-started/02-first-graph.md)

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

### [Concepts](./concepts/) - Understanding the System
Explanatory documentation for background knowledge. Read these to understand *why* things work.
- [Architecture Overview](./concepts/architecture.md)
- [Agentic Workers](./concepts/agentic-workers.md)
- [GPU Compute](./concepts/gpu-compute.md)
- [Security Model](./concepts/security-model.md)

### [Reference](./reference/) - Technical Details
Complete technical specifications. Use these for detailed information and API documentation.

**API Documentation:**
- [REST API](./reference/api/rest-api.md)
- [WebSocket API](./reference/api/websocket-api.md)
- [Binary Protocol](./reference/api/binary-protocol.md)

**Architecture Reference:**
- [Hexagonal & CQRS Pattern](./reference/architecture/hexagonal-cqrs.md)
- [Database Schema](./reference/architecture/database-schema.md)
- [Actor System](./reference/architecture/actor-system.md)
- [Ontology Storage Architecture](./architecture/ontology-storage-architecture.md) - Lossless markdown storage

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
├── README.md  (This file)
├── getting-started/     (Tutorials)
│   ├── 01-installation.md
│   └── 02-first-graph.md
├── guides/             (How-To Guides)
│   ├── user/
│   │   ├── working-with-agents.md
│   │   └── xr-setup.md
│   └── developer/
│       ├── 01-development-setup.md
│       ├── 04-adding-features.md
│       └── testing-guide.md
├── concepts/           (Explanations)
│   ├── architecture.md
│   ├── agentic-workers.md
│   ├── gpu-compute.md
│   └── security-model.md
├── reference/          (Technical Details)
│   ├── api/
│   │   ├── rest-api.md
│   │   ├── websocket-api.md
│   │   └── binary-protocol.md
│   ├── architecture/
│   │   ├── hexagonal-cqrs.md
│   │   ├── database-schema.md
│   │   └── actor-system.md
│   └── agents/
├── deployment/         (Deployment Guides)
├── archive/            (Historical Documents)
└── CONTRIBUTING_DOCS.md (How to contribute)
```

---

## 🤝 Contributing to Documentation

See [CONTRIBUTING_DOCS.md](./CONTRIBUTING_DOCS.md) for guidelines on how to add or update documentation.

---

**Last Updated**: 2025-10-27  
**Framework**: Diátaxis  
**Status**: ✅ Refactored & Organized
