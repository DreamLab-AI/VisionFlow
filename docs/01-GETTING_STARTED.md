---
layout: default
title: Getting Started
description: Entry points for new users, developers, architects, and operators
nav_order: 8
parent: VisionFlow Documentation
---


# Getting Started with VisionFlow

**Your role-specific entry point to VisionFlow** | [New User](#new-user) | [Developer](#developer) | [Architect](#architect) | [DevOps](#devops)

---

## 🎯 Choose Your Path

Select your role to get started:

### 🆕 New User
**I want to use VisionFlow to visualize knowledge graphs**
→ [Start Here: New User Guide](guides/getting-started/GETTING_STARTED_USER.md)

### 👨‍💻 Developer
**I want to build features and contribute code**
→ [Start Here: Developer Guide](guides/getting-started/GETTING_STARTED_DEVELOPER.md)

### 🏗️ Architect
**I want to understand system design and make architectural decisions**
→ [Start Here: Architect Guide](guides/getting-started/GETTING_STARTED_ARCHITECT.md)

### 🔧 DevOps / Operator
**I want to deploy and operate VisionFlow in production**
→ [Start Here: Operator Guide](guides/getting-started/GETTING_STARTED_OPERATOR.md)

---

## ⚡ 5-Minute Quick Start

**For everyone - get VisionFlow running in 5 minutes:**

### Step 1: Install (2 minutes)

**Docker (Recommended):**
```bash
docker-compose up -d
```

**Native:**
```bash
# See full installation guide
```

→ **[Full Installation Guide](tutorials/01-installation.md)**

### Step 2: Create Graph (2 minutes)

1. Open http://localhost:3001
2. Click "Create Graph"
3. Select GitHub repository or upload files
4. Click "Visualize"

→ **[First Graph Tutorial](tutorials/02-first-graph.md)**

### Step 3: Explore (1 minute)

- **Rotate**: Left mouse drag
- **Pan**: Right mouse drag / arrow keys
- **Zoom**: Mouse wheel / +/- keys
- **Select**: Left click on node

→ **[Navigation Guide](guides/navigation-guide.md)**

---

## 📚 Essential Documentation

### Everyone Should Read

1. **[What is VisionFlow?](OVERVIEW.md)** - Value proposition and use cases
2. **[Installation](tutorials/01-installation.md)** - Docker and native setup
3. **[First Graph](tutorials/02-first-graph.md)** - Create your first visualization
4. **[Navigation Guide](guides/navigation-guide.md)** - Master the 3D interface

### Role-Specific Next Steps

**New Users:**
- [Configuration](guides/configuration.md)
- [Natural Language Queries](guides/features/natural-language-queries.md)
- [Troubleshooting](guides/troubleshooting.md)

**Developers:**
- [Developer Journey](DEVELOPER_JOURNEY.md)
- [Development Setup](guides/developer/01-development-setup.md)
- [Project Structure](guides/developer/02-project-structure.md)

**Architects:**
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Technology Choices](TECHNOLOGY_CHOICES.md)
- [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md)

**DevOps:**
- [Deployment Guide](guides/deployment.md)
- [Docker Compose](guides/docker-compose-guide.md)
- [Pipeline Operator Runbook](guides/operations/pipeline-operator-runbook.md)

---

## 🎓 Learning Paths

### Beginner Path (Week 1)

| Day | Focus | Documents |
|-----|-------|-----------|
| **1** | Installation & First Graph | [Installation](tutorials/01-installation.md), [First Graph](tutorials/02-first-graph.md) |
| **2** | Navigation & Interface | [Navigation Guide](guides/navigation-guide.md) |
| **3** | Database Basics | [Neo4j Quick Start](tutorials/neo4j-quick-start.md) |
| **4** | Configuration | [Configuration](guides/configuration.md) |
| **5** | Advanced Features | [Natural Language Queries](guides/features/natural-language-queries.md) |

### Intermediate Path (Week 2-3)

- **Architecture Understanding** → [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- **Development Setup** → [Development Setup](guides/developer/01-development-setup.md)
- **Feature Development** → [Adding Features](guides/developer/04-adding-features.md)

### Advanced Path (Week 4+)

- **System Design** → [Hexagonal CQRS](explanations/architecture/hexagonal-cqrs.md)
- **GPU Optimization** → [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md)
- **Multi-Agent AI** → [Multi-Agent System](explanations/architecture/multi-agent-system.md)

---

## 🔑 Key Concepts

### Knowledge Graph
**What**: Network of interconnected concepts and relationships
**Why**: Enables semantic understanding and reasoning
**Learn**: [First Graph Tutorial](tutorials/02-first-graph.md)

### Neo4j Database
**What**: Graph database storing nodes and edges
**Why**: Native graph storage with powerful query language (Cypher)
**Learn**: [Neo4j Quick Start](tutorials/neo4j-quick-start.md)

### AI Agents
**What**: 50+ concurrent AI agents for graph analysis
**Why**: Automate knowledge extraction and analysis
**Learn**: [Agent Orchestration](guides/agent-orchestration.md)

### Semantic Physics
**What**: Physics-based layout using ontology constraints
**Why**: Meaningful spatial arrangement of concepts
**Learn**: [Semantic Forces](guides/features/semantic-forces.md)

### OWL Ontologies
**What**: Formal knowledge representation with reasoning
**Why**: Enable semantic inference and validation
**Learn**: [Ontology Parser](guides/ontology-parser.md)

### GPU Acceleration
**What**: 39 CUDA kernels for computation
**Why**: 100x speedup for large graphs (100k+ nodes)
**Learn**: [GPU Semantic Forces](explanations/architecture/gpu-semantic-forces.md)

---

## 📊 System Requirements

### Minimum (Development)
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 10GB
- **GPU**: None (CPU fallback)
- **OS**: Linux, macOS, Windows (Docker)

### Recommended (Production)
- **CPU**: 8 cores, 3.0GHz+
- **RAM**: 16GB
- **Storage**: 50GB SSD
- **GPU**: NVIDIA GTX 1060 (6GB VRAM)
- **OS**: Linux (Ubuntu 22.04+ recommended)

### Enterprise (100k+ nodes)
- **CPU**: 16+ cores, 3.5GHz
- **RAM**: 32GB+
- **Storage**: 200GB+ NVMe SSD
- **GPU**: NVIDIA RTX 4080+ (16GB+ VRAM)
- **OS**: Linux with CUDA 12.4+

---

## 🆘 Common Questions

### "How do I install VisionFlow?"
→ [Installation Tutorial](tutorials/01-installation.md)

### "How do I create my first graph?"
→ [First Graph Tutorial](tutorials/02-first-graph.md)

### "How do I navigate the 3D interface?"
→ [Navigation Guide](guides/navigation-guide.md)

### "What technologies does VisionFlow use?"
→ [Technology Choices](TECHNOLOGY_CHOICES.md)

### "How do I deploy to production?"
→ [Deployment Guide](guides/deployment.md)

### "How do I contribute?"
→ [Contributing Guide](guides/developer/06-contributing.md)

### "Where can I get help?"
→ [Troubleshooting](guides/troubleshooting.md), [GitHub Issues](https://github.com/DreamLab-AI/VisionFlow/issues)

---

## 📞 Getting Help

| Issue Type | Where to Go |
|------------|-------------|
| **Installation problems** | [Troubleshooting Guide](guides/troubleshooting.md) |
| **Bug reports** | [GitHub Issues](https://github.com/DreamLab-AI/VisionFlow/issues) |
| **Feature requests** | [GitHub Discussions](https://github.com/DreamLab-AI/VisionFlow/discussions) |
| **Documentation gaps** | File issue with `documentation` label |
| **Infrastructure issues** | [Infrastructure Troubleshooting](guides/infrastructure/troubleshooting.md) |

---

## 🚀 Next Steps

**After completing getting started:**

1. **Choose your learning path** (Beginner / Intermediate / Advanced)
2. **Follow role-specific guide** (User / Developer / Architect / DevOps)
3. **Join the community** (GitHub Discussions, Issues)
4. **Contribute** (Code, documentation, feedback)

---

**Last Updated**: 2025-12-18
**Audience**: All users - entry point to VisionFlow
**Next**: Choose your role-specific guide above
