# VisionFlow Quick Navigation Guide

**Fast access to essential VisionFlow documentation**

---

## 🎯 Most Important Links

### Start Here
- **[📖 Master Documentation Index](index.md)** - Complete catalog of all 311+ docs
- **[📚 Documentation Hub](readme.md)** - Organized by Diátaxis framework
- **[🚀 Main Project README](../readme.md)** - Project overview and quick start

---

## 🏃 Quick Paths by Goal

### I Want To...

#### **Get Started with VisionFlow**
1. [Install VisionFlow](getting-started/01-installation.md)
2. [Create First Graph](getting-started/02-first-graph-and-agents.md)
3. [Basic Usage Guide](user-guide/03-basic-usage.md)

#### **Deploy to Production**
1. [Docker Deployment](deployment/01-docker-deployment.md)
2. [Configuration Guide](deployment/02-configuration.md)
3. [Monitoring Setup](deployment/03-monitoring.md)

#### **Develop a Feature**
1. [Development Setup](developer/01-development-setup.md)
2. [Architecture Overview](../concepts/architecture/00-ARCHITECTURE-overview.md)
3. [Adding Features](developer/04-adding-features.md)
4. [Testing Guide](developer/05-testing-guide.md)

#### **Work with Ontologies**
1. [Ontology Fundamentals](specialized/ontology/ontology-fundamentals.md)
2. [Quick Start](specialized/ontology/quickstart.md)
3. [User Guide](specialized/ontology/ontology-user-guide.md)
4. [API Reference](specialized/ontology/ontology-api-reference.md)

#### **Setup XR/VR**
1. [XR Setup Guide](user/xr-setup.md)
2. [Vircadia Complete Guide](vircadia-xr-complete-guide.md)
3. [XR Architecture](../concepts/architecture/xr-immersive-system.md)

#### **Understand the Architecture**
1. [Architecture Overview](../concepts/architecture/00-ARCHITECTURE-overview.md)
2. [Hexagonal CQRS](../concepts/architecture/hexagonal-cqrs-architecture.md)
3. [System Concepts](../concepts/system-architecture.md)
4. [Database Schema](../concepts/architecture/04-database-schemas.md)

#### **Use the API**
1. [API Quick Reference](../api/quick-reference.md)
2. [REST API](../reference/api/rest-api-complete.md)
3. [WebSocket API](../reference/api/03-websocket.md)
4. [Binary Protocol](../reference/binary-protocol-specification.md) - 36-byte binary format specification

#### **Debug Issues**
1. [Troubleshooting Guide](guides/troubleshooting.md)
2. [Common Issues](user-guide/05-troubleshooting.md)
3. [Debug Task](tasks/task-debug.md)

---

## 📚 Documentation by Topic

### Core Systems
| System | Documentation |
|--------|---------------|
| **Ontology** | [Fundamentals](../specialized/ontology/ontology-fundamentals.md) • [Reasoning](../ontology-reasoning.md) • [API](../specialized/ontology/ontology-api-reference.md) |
| **GPU Acceleration** | [Concepts](../concepts/gpu-compute.md) • [Architecture](../concepts/architecture/gpu/readme.md) • [Optimizations](../concepts/architecture/gpu/optimizations.md) |
| **AI Agents** | [Concepts](../concepts/agentic-workers.md) • [User Guide](user/working-with-agents.md) • [Reference](../reference/agents/readme.md) |
| **XR/VR** | [Setup](user/xr-setup.md) • Architecture (TODO) • [Vircadia](vircadia-xr-complete-guide.md) |
| **Database** | [Schema](../concepts/architecture/04-database-schemas.md) • [Reference](../reference/architecture/database-schema.md) |
| **Binary Protocol** | [WebSocket](../reference/api/03-websocket.md) • Binary Format (TODO) |

### Key Features
| Feature | Quick Link | Complete Docs |
|---------|------------|---------------|
| **Ontology Reasoning** | [Overview](../ontology-reasoning.md) | [Complete Guide](../specialized/ontology/) |
| **Semantic Physics** | [Architecture](../semantic-physics-architecture.md) | [GPU Forces](../gpu-semantic-forces.md) |
| **Multi-User XR** | [Quick Setup](user/xr-setup.md) | [Complete Guide](vircadia-xr-complete-guide.md) |
| **CQRS Pattern** | [Architecture](../concepts/architecture/hexagonal-cqrs-architecture.md) | [Reference](../reference/architecture/hexagonal-cqrs.md) |
| **GPU Compute** | [Concepts](../concepts/gpu-compute.md) | [Architecture](../concepts/architecture/gpu/) |

---

## 👥 Paths by Role

### New User
1. [Installation](getting-started/01-installation.md)
2. [First Graph](getting-started/02-first-graph-and-agents.md)
3. [Basic Usage](user-guide/03-basic-usage.md)
4. [Features Overview](user-guide/04-features-overview.md)

### Developer
1. [Development Setup](guides/developer/01-development-setup.md)
2. [Project Structure](guides/developer/02-project-structure.md)
3. [Architecture](guides/developer/03-architecture.md)
4. [Adding Features](guides/developer/04-adding-features.md)
5. [Testing](guides/developer/05-testing-guide.md)

### DevOps Engineer
1. [Docker Deployment](deployment/01-docker-deployment.md)
2. [Configuration](deployment/02-configuration.md)
3. [Monitoring](deployment/03-monitoring.md)
4. [Backup & Restore](deployment/04-backup-restore.md)

### Researcher
1. [System Architecture](../concepts/system-architecture.md)
2. [Architecture Overview](../concepts/architecture/00-ARCHITECTURE-overview.md)
3. [Research Documents](../research/)
4. [Academic Survey](../research/Academic-Research-Survey.md)

### Ontology Expert
1. [Fundamentals](specialized/ontology/ontology-fundamentals.md)
2. [Quick Start](specialized/ontology/quickstart.md)
3. [API Reference](specialized/ontology/ontology-api-reference.md)
4. [Physics Integration](specialized/ontology/physics-integration.md)

---

## 🔍 Search Tips

### Find by Keyword
```bash
# From project root
grep -r "keyword" docs/ --include="*.md"

# Find specific topic
grep -r "ontology" docs/ --include="*.md" -l
```

### Browse by Directory
```bash
# Architecture docs
ls docs/architecture/

# API reference
ls docs/api/

# User guides
ls docs/guides/user/

# Developer guides
ls docs/guides/developer/
```

---

## 📊 Documentation Statistics

- **Total Files**: 311+
- **Categories**: 9 major sections
- **Architecture Docs**: 45+
- **API Docs**: 12+
- **Guides**: 50+
- **Agent Specs**: 50+
- **Research Papers**: 40+

---

## 🆘 Need Help?

### Common Questions
- **Can't find something?** → Check [Master Index](index.md)
- **New to VisionFlow?** → Start with [Installation](getting-started/01-installation.md)
- **Need API docs?** → See [API Quick Reference](api/quick-reference.md)
- **Want to contribute?** → Read [Contributing Guide](guides/developer/06-contributing.md)

### Support Resources
- [GitHub Issues](https://github.com/yourusername/VisionFlow/issues)
- [GitHub Discussions](https://github.com/yourusername/VisionFlow/discussions)
- [Documentation Hub](readme.md)
- [Master Index](index.md)

---

## 🔗 External Resources

- **Main Repository**: https://github.com/yourusername/VisionFlow
- **Vircadia Platform**: https://vircadia.com
- **Vircadia Docs**: https://docs.vircadia.com
- **Diátaxis Framework**: https://diataxis.fr/

---

## 🎯 Bookmark These

**Essential Documentation:**
- [Master Index](../index.md) - Complete catalog
- [Architecture Overview](../concepts/architecture/00-ARCHITECTURE-overview.md) - System design
- [API Quick Reference](../api/quick-reference.md) - API lookup
- [Troubleshooting](troubleshooting.md) - Problem solving
- [Roadmap](../roadmap.md) - Future plans

**Quick References:**
- [Configuration](reference/configuration.md) - All settings
- [Glossary](reference/glossary.md) - Terms & definitions
- [CUDA Parameters](reference/cuda-parameters.md) - GPU config

---

**Last Updated**: 2025-11-03
**Framework**: Diátaxis
**Total Documentation**: 311+ files

---

**Navigation:** [📖 Master Index](index.md) | [📚 Documentation Hub](readme.md) | [🚀 Main README](../readme.md)
