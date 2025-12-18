# Navigation Design Specification

**Version:** 1.0
**Date:** 2025-12-18
**Parent Specification:** `UNIFIED_ARCHITECTURE_SPEC.md`
**Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

This specification defines the complete navigation system for the VisionFlow documentation, ensuring seamless information discovery through multiple complementary navigation patterns:

1. **Hierarchical Navigation** - Category-based browsing
2. **Sequential Navigation** - Learning path progression
3. **Contextual Navigation** - Related content discovery
4. **Tag-Based Navigation** - Topic-based exploration
5. **Search-Optimized Navigation** - Keyword discovery

---

## 1. User Journey Maps

### 1.1 Journey #1: New Developer (Beginner)

**Goal:** Get up and running with first knowledge graph

**Entry Point:** `README.md` → "Getting Started" section

**Path:**
```
README.md
  → Section: "Quick Start"
    → getting-started/01-installation.md
      ├─ Prerequisites: Docker, Node.js, Rust
      ├─ Installation steps
      └─ Verification tests
        → getting-started/02-first-graph.md
          ├─ Create first nodes
          ├─ Add relationships
          └─ Visualize graph
            → getting-started/03-neo4j-quickstart.md
              ├─ Neo4j setup
              ├─ Cypher basics
              └─ Query examples
                → guides/features/semantic-forces.md
                  ├─ Enable physics
                  ├─ Adjust parameters
                  └─ See results
                    → explanations/physics/semantic-forces.md
                      ├─ Theory: Force-directed layout
                      ├─ Barnes-Hut algorithm
                      └─ GPU acceleration
```

**Navigation Elements:**
- **Breadcrumbs:** Home > Getting Started > Installation
- **Previous/Next:** ← None | First Graph →
- **Progress Indicator:** Step 1 of 5
- **Related Docs:** Neo4j Quick Start, Docker Environment Guide
- **See Also:** Architecture Overview, Developer Journey

**Exit Points:**
- Advanced tutorial: Agent Orchestration
- Deep dive: Hexagonal Architecture
- Reference: REST API Complete

---

### 1.2 Journey #2: System Architect (Advanced)

**Goal:** Understand system architecture and design decisions

**Entry Point:** `ARCHITECTURE_OVERVIEW.md`

**Path:**
```
ARCHITECTURE_OVERVIEW.md
  → Section: "Architectural Patterns"
    → explanations/architecture/hexagonal-cqrs.md
      ├─ Hexagonal architecture principles
      ├─ CQRS pattern implementation
      └─ Port-adapter separation
        → diagrams/mermaid-library/01-system-architecture.md
          ├─ Full system architecture
          ├─ Hexagonal architecture diagram
          └─ CQRS pattern diagram
            → architecture/decisions/ADR-001-hexagonal.md
              ├─ Context: Why hexagonal?
              ├─ Decision: Ports & adapters
              └─ Consequences: Trade-offs
                → explanations/architecture/core/server.md
                  ├─ Server architecture details
                  ├─ Actor system design
                  └─ Component interactions
                    → reference/api/rest-api-complete.md
                      ├─ API endpoint specifications
                      ├─ Request/response formats
                      └─ Error handling
```

**Navigation Elements:**
- **Breadcrumbs:** Home > Architecture > Hexagonal CQRS
- **Related ADRs:** ADR-002-CQRS, ADR-003-Actor-Model
- **Related Diagrams:** System Architecture, CQRS Pattern
- **Related Explanations:** Adapter Patterns, Services Architecture
- **Referenced By:** Developer Journey, Deployment Guide

**Exit Points:**
- Implementation: Guides > Developer > Testing
- Deep dive: Actor System Complete
- Visual: Diagrams > Architecture

---

### 1.3 Journey #3: DevOps Engineer (Operations Focus)

**Goal:** Deploy and operate VisionFlow in production

**Entry Point:** `QUICK_NAVIGATION.md` → "Deployment" section

**Path:**
```
QUICK_NAVIGATION.md
  → Section: "Deployment & Operations"
    → guides/infrastructure/deployment.md
      ├─ Docker setup
      ├─ Environment configuration
      └─ Production checklist
        → diagrams/mermaid-library/03-deployment.md
          ├─ Deployment topology
          ├─ Docker container architecture
          └─ Network architecture
            → guides/infrastructure/docker-environment.md
              ├─ Container configuration
              ├─ Multi-user setup
              └─ Volume management
                → guides/infrastructure/troubleshooting.md
                  ├─ Common issues
                  ├─ Debugging steps
                  └─ Performance tuning
                    → explanations/architecture/subsystems/
                      ├─ Infrastructure components
                      └─ Service dependencies
```

**Navigation Elements:**
- **Breadcrumbs:** Home > Guides > Infrastructure > Deployment
- **Related Guides:** Docker Environment, Troubleshooting
- **Related Diagrams:** Deployment Topology, Network Architecture
- **Related Reference:** Docker Configuration, Environment Variables
- **Tools:** Deployment checklist, Health check scripts

**Exit Points:**
- Monitoring: Telemetry & Logging Guide
- Scaling: GPU Cluster Setup
- Security: Authentication Guide

---

### 1.4 Journey #4: API Consumer (Integration Focus)

**Goal:** Integrate VisionFlow into external application

**Entry Point:** `README.md` → "API Documentation" section

**Path:**
```
README.md
  → Section: "API Documentation"
    → reference/api/rest-api-complete.md
      ├─ Authentication
      ├─ Graph endpoints
      └─ Error handling
        → diagrams/server/api/rest-api-architecture.md
          ├─ API architecture diagram
          └─ Request flow
            → explanations/architecture/api-handlers-reference.md
              ├─ Handler implementation
              ├─ Validation logic
              └─ Business rules
                → guides/developer/websocket-best-practices.md
                  ├─ Binary protocol usage
                  ├─ Connection management
                  └─ Error recovery
                    → reference/protocols/binary-websocket.md
                      ├─ Message format
                      ├─ Message types
                      └─ Performance characteristics
```

**Navigation Elements:**
- **Breadcrumbs:** Home > Reference > API > REST API
- **Related Reference:** WebSocket Protocol, Error Codes
- **Related Guides:** Authentication, API Best Practices
- **Related Diagrams:** API Architecture, Data Flow
- **Code Examples:** Request samples, Response examples

**Exit Points:**
- Advanced: Agent Orchestration API
- Deep dive: Hexagonal Architecture
- Examples: Sample Applications

---

### 1.5 Journey #5: Contributor (Code Contribution)

**Goal:** Understand codebase and contribute features

**Entry Point:** `DEVELOPER_JOURNEY.md`

**Path:**
```
DEVELOPER_JOURNEY.md
  → Section: "Contributing"
    → guides/developer/testing-guide.md
      ├─ Test setup
      ├─ Writing tests
      └─ CI/CD pipeline
        → explanations/architecture/hexagonal-cqrs.md
          ├─ Understand patterns
          └─ Code organization
            → architecture/decisions/
              ├─ Read ADRs
              └─ Understand rationale
                → guides/developer/websocket-best-practices.md
                  ├─ Coding standards
                  └─ Best practices
                    → diagrams/architecture/
                      ├─ Visual reference
                      └─ Component relationships
```

**Navigation Elements:**
- **Breadcrumbs:** Home > Developer Journey > Testing
- **Related Guides:** Debugging, Telemetry & Logging
- **Related Architecture:** Hexagonal CQRS, Actor System
- **Related Decisions:** ADRs relevant to changes
- **Tools:** Test templates, Code review checklist

**Exit Points:**
- Submit PR: GitHub workflow
- Get help: Troubleshooting guide
- Understand more: Architecture deep dives

---

## 2. Breadcrumb Navigation System

### 2.1 Breadcrumb Template

**Implementation in Front Matter:**
```yaml
---
breadcrumbs:
  - text: "Home"
    href: "../../README.md"
  - text: "Guides"
    href: "../README.md"
  - text: "Ontology"
    href: "README.md"
  - text: "Neo4j Setup"
    href: "neo4j-setup.md"
    current: true
---
```

**Rendered Output:**
```
Home > Guides > Ontology > Neo4j Setup
```

### 2.2 Breadcrumb Generation Rules

**Rule 1: Depth-Based Breadcrumbs**
```python
def generate_breadcrumbs(file_path: Path, docs_root: Path) -> List[Dict]:
    """Generate breadcrumbs from file path"""
    relative = file_path.relative_to(docs_root)
    parts = relative.parts[:-1]  # Exclude filename

    breadcrumbs = [{
        'text': 'Home',
        'href': '../' * len(parts) + 'README.md'
    }]

    for i, part in enumerate(parts):
        breadcrumbs.append({
            'text': part.replace('-', ' ').title(),
            'href': '../' * (len(parts) - i - 1) + 'README.md'
        })

    # Add current file
    breadcrumbs.append({
        'text': file_path.stem.replace('-', ' ').title(),
        'href': file_path.name,
        'current': True
    })

    return breadcrumbs
```

**Rule 2: Maximum Breadcrumb Depth**
- Max depth: 5 levels (Home + 4 directories)
- If deeper, show: Home > ... > Parent > Current

**Rule 3: Breadcrumb Styling**
```markdown
<nav aria-label="Breadcrumb">
  <a href="../../README.md">Home</a> >
  <a href="../README.md">Guides</a> >
  <a href="README.md">Ontology</a> >
  <strong>Neo4j Setup</strong>
</nav>
```

---

## 3. Previous/Next Article Chains

### 3.1 Sequential Navigation Template

**Front Matter:**
```yaml
---
previous:
  title: "01 - Installation"
  href: "01-installation.md"
  description: "Set up VisionFlow development environment"
next:
  title: "03 - Neo4j Quick Start"
  href: "03-neo4j-quickstart.md"
  description: "Learn Neo4j basics for knowledge graphs"
---
```

**Rendered Output:**
```markdown
---

## Navigation

### ← Previous: Installation
Set up VisionFlow development environment

[← Previous](01-installation.md) | [Next →](03-neo4j-quickstart.md)

### Next →: Neo4j Quick Start
Learn Neo4j basics for knowledge graphs

---
```

### 3.2 Sequential Navigation Rules

**Rule 1: Tutorial Sequences**
- All tutorials MUST have prev/next
- First tutorial: previous = null, next = 02-*
- Last tutorial: previous = second-to-last, next = "What's Next?" section

**Rule 2: Guide Sequences**
- Guides within same topic: prev/next within topic
- Cross-topic guides: prev/next based on learning path

**Rule 3: Explanation Sequences**
- Hierarchical: parent → child progression
- Lateral: related concepts at same level

---

## 4. Sidebar Navigation Maps

### 4.1 Category Sidebar Template

**For Tutorials (`getting-started/README.md`)**:
```markdown
# Getting Started

## Tutorial Path

### Core Tutorials
1. [Installation](01-installation.md) - Set up environment
2. [First Graph](02-first-graph.md) - Create your first knowledge graph
3. [Neo4j Quick Start](03-neo4j-quickstart.md) - Database basics
4. [Agent Basics](04-agent-basics.md) - Multi-agent introduction
5. [Deployment Basics](05-deployment-basics.md) - Deployment fundamentals

### Next Steps
- **Features:** [Semantic Forces](../guides/features/semantic-forces.md)
- **Architecture:** [Hexagonal CQRS](../explanations/architecture/hexagonal-cqrs.md)
- **API:** [REST API Reference](../reference/api/rest-api-complete.md)

### Related Resources
- [Developer Journey](../DEVELOPER_JOURNEY.md) - Complete learning path
- [Quick Navigation](../QUICK_NAVIGATION.md) - Fast access index
- [Troubleshooting](../guides/troubleshooting.md) - Common issues

---

**Progress Tracking:**
- [ ] Environment setup complete
- [ ] First graph created
- [ ] Neo4j basics understood
- [ ] Agent system basics learned
- [ ] Deployment tested
```

### 4.2 Guides Sidebar Template

**For Guides (`guides/README.md`)**:
```markdown
# How-To Guides

## By Category

### 🎨 Features
- [Semantic Forces](features/semantic-forces.md) - Force-directed layout
- [Physics Simulation](features/physics-simulation.md) - GPU-accelerated physics
- [XR Immersive](features/xr-immersive.md) - VR/AR experiences
- [Voice Interaction](features/voice-interaction.md) - Speech integration
- [Multi-Workspace](features/multi-workspace.md) - Workspace isolation

### 🧠 Ontology & Knowledge Graphs
- [Neo4j Setup](ontology/neo4j-setup.md) - Database installation
- [Semantic Forces Guide](ontology/semantic-forces-guide.md) - Ontology physics
- [Reasoning Integration](ontology/reasoning-integration.md) - Inference engine
- [Visualization](ontology/visualization.md) - Graph rendering

### 🏗️ Infrastructure & Deployment
- [Docker Environment](infrastructure/docker-environment.md) - Container setup
- [Deployment](infrastructure/deployment.md) - Production deployment
- [Troubleshooting](troubleshooting.md) - Common issues

### 👨‍💻 Developer Workflows
- [WebSocket Best Practices](developer/websocket-best-practices.md) - Binary protocol
- [Testing Guide](developer/testing-guide.md) - Test strategy
- [Telemetry & Logging](developer/telemetry-logging.md) - Observability
- [Debugging](developer/debugging.md) - Debug techniques

### 🤖 AI Models & Agents
- [ComfyUI Integration](ai-models/comfyui-integration.md) - Image generation
- [Blender MCP](ai-models/blender-mcp.md) - 3D model integration
- [DeepSeek Deployment](ai-models/deepseek-deployment.md) - LLM setup

### 💻 Client Development
- [State Management](client/state-management.md) - Zustand patterns
- [Three.js Rendering](client/three-js-rendering.md) - 3D visualization
- [XR Integration](client/xr-integration.md) - WebXR implementation

---

## Popular Guides
1. 🔥 [Neo4j Setup](ontology/neo4j-setup.md) - Most viewed
2. 🔥 [Deployment](infrastructure/deployment.md) - Essential
3. 🔥 [Testing Guide](developer/testing-guide.md) - Best practices

## By Difficulty
- **Beginner:** Neo4j Setup, Docker Environment
- **Intermediate:** Testing Guide, WebSocket Best Practices
- **Advanced:** Deployment, Telemetry & Logging
```

### 4.3 Explanations Sidebar Template

**For Explanations (`explanations/README.md`)**:
```markdown
# Conceptual Explanations

## Architecture

### Core Concepts
- [Hexagonal CQRS](architecture/hexagonal-cqrs.md) - Pattern overview
- [Adapter Patterns](architecture/adapter-patterns.md) - Port-adapter design
- [Services Architecture](architecture/services-architecture.md) - Service layer
- [Integration Patterns](architecture/integration-patterns.md) - Integration layer

### Core Architecture
- [Client](architecture/core/client.md) - Client architecture
- [Client TypeScript](architecture/core/client-typescript.md) - TypeScript details
- [Server](architecture/core/server.md) - Server architecture
- [Database](architecture/core/database.md) - Data persistence

### Components
- [WebSocket Protocol](architecture/components/websocket-protocol.md) - Binary protocol
- [Actor System](architecture/components/actor-system.md) - Actor model
- [GPU Pipeline](architecture/components/gpu-pipeline.md) - GPU acceleration

### Subsystems
- [Blender MCP](architecture/subsystems/blender-mcp.md) - 3D integration
- [Ontology Storage](architecture/subsystems/ontology-storage.md) - Graph storage
- [Analytics](architecture/subsystems/analytics.md) - Metrics system

## Ontology & Knowledge Graphs
- [Ontology Overview](ontology/ontology-overview.md) - Knowledge graph concepts
- [Neo4j Integration](ontology/neo4j-integration.md) - Database architecture
- [Hierarchical Visualization](ontology/hierarchical-visualization.md) - Graph layout
- [Semantic Forces](ontology/semantic-forces.md) - Force-directed theory
- [Graph Algorithms](ontology/graph-algorithms.md) - Pathfinding, clustering

## Physics & Simulation
- [Semantic Forces](physics/semantic-forces.md) - Physics theory
- [Barnes-Hut Algorithm](physics/barnes-hut-algorithm.md) - Force calculation
- [GPU Acceleration](physics/gpu-acceleration.md) - CUDA optimization

## Design Patterns
- [Hexagonal Architecture](patterns/hexagonal-architecture.md) - Ports & adapters
- [CQRS Pattern](patterns/cqrs-pattern.md) - Command-query separation
- [Actor Model](patterns/actor-model.md) - Concurrency pattern

---

## Learning Paths

### 🎯 Understanding Architecture
1. Hexagonal CQRS
2. Core Client/Server
3. Actor System
4. Integration Patterns

### 🧠 Mastering Ontologies
1. Ontology Overview
2. Neo4j Integration
3. Semantic Forces
4. Graph Algorithms

### ⚡ Performance Deep Dive
1. GPU Acceleration
2. Barnes-Hut Algorithm
3. WebSocket Protocol
4. Actor System
```

### 4.4 Reference Sidebar Template

**For Reference (`reference/README.md`)**:
```markdown
# Technical Reference

## API Specifications

### REST API
- [REST API Complete](api/rest-api-complete.md) - **PRIMARY:** All REST endpoints
- [GraphQL API](api/graphql-api.md) - GraphQL schema & queries
- [Error Codes](api/error-codes.md) - Error code reference

### Protocols
- [Binary WebSocket](protocols/binary-websocket.md) - **PRIMARY:** WebSocket protocol

## Database

### Neo4j
- [Schemas](database/schemas.md) - Database schemas
- [Cypher Queries](database/cypher-queries.md) - Common queries

## Ontology

### Specifications
- [Schema](ontology/schema.md) - Ontology schema definition
- [API](ontology/api.md) - Ontology API reference
- [Data Model](ontology/data-model.md) - Data model specification

## Client

### TypeScript/React
- [Components](client/components.md) - Component API reference
- [State API](client/state-api.md) - State management API

---

## Quick Access

### Most Referenced
- [REST API Complete](api/rest-api-complete.md)
- [Binary WebSocket](protocols/binary-websocket.md)
- [Database Schemas](database/schemas.md)

### By Type
- **APIs:** REST, GraphQL, WebSocket
- **Schemas:** Database, Ontology, Data Model
- **Protocols:** WebSocket
```

---

## 5. Related Articles Sections

### 5.1 Related Documentation Template

**Standard Footer Section:**
```markdown
---

## Related Documentation

### Prerequisites
- [Installation Tutorial](../../getting-started/01-installation.md) - Required before this guide
- [Neo4j Quick Start](../../getting-started/03-neo4j-quickstart.md) - Database basics needed

### See Also
- [Neo4j Integration Guide](../guides/ontology/neo4j-setup.md) - Database setup
- [Database Schemas Reference](../reference/database/schemas.md) - Schema specification
- [Ontology Overview](../explanations/ontology/ontology-overview.md) - Conceptual background

### Deep Dive
- [Hexagonal CQRS Architecture](../explanations/architecture/hexagonal-cqrs.md) - Architectural patterns
- [Database Architecture](../explanations/architecture/core/database.md) - Persistence layer design
- [Actor System Complete](../diagrams/server/actors/actor-system-complete.md) - Visual architecture

### Referenced By
- [Developer Journey](../DEVELOPER_JOURNEY.md) - Learning path
- [Architecture Overview](../ARCHITECTURE_OVERVIEW.md) - System architecture
- [Deployment Guide](../guides/infrastructure/deployment.md) - Production deployment

### Tools & Resources
- [Neo4j Browser](http://localhost:7474) - Database interface
- [Cypher Cheat Sheet](https://neo4j.com/docs/cypher-cheat-sheet/) - Query reference
- [Graph Algorithms Library](https://neo4j.com/docs/graph-algorithms/) - Algorithm docs

---

**Last Updated:** 2025-12-18
**Category:** Guide
**Tags:** neo4j, database, integration, ontology
**Difficulty:** Intermediate
**Estimated Time:** 30 minutes
```

### 5.2 Relationship Types & Rules

**Relationship Type Rules:**

| Relationship | Direction | Purpose | Example |
|--------------|-----------|---------|---------|
| **Prerequisites** | Backward | Required prior knowledge | Tutorial 02 requires Tutorial 01 |
| **See Also** | Lateral | Related same-level content | Guide links to related guide |
| **Deep Dive** | Forward | More detailed explanation | Guide links to explanation |
| **Referenced By** | Backward | Who links to this | Shows usage context |
| **Next Steps** | Forward | Natural progression | Tutorial to guide to explanation |
| **Tools & Resources** | External | Related tools/docs | External references |

**Automatic Relationship Generation:**
```python
def generate_related_docs(doc: Document, all_docs: Dict) -> Dict:
    """Generate related documentation sections"""
    related = {
        'prerequisites': [],
        'see_also': [],
        'deep_dive': [],
        'referenced_by': [],
        'next_steps': [],
    }

    # Prerequisites: Inferred from category and sequence
    if doc.category == 'tutorial' and doc.sequence > 1:
        prev_tutorial = find_previous_tutorial(doc)
        related['prerequisites'].append(prev_tutorial)

    # See Also: Tag-based similarity
    similar_docs = find_similar_by_tags(doc, all_docs, threshold=0.3)
    related['see_also'] = similar_docs[:5]

    # Deep Dive: Guide → Explanation
    if doc.category == 'guide':
        explanation_docs = find_explanations_for_guide(doc, all_docs)
        related['deep_dive'] = explanation_docs[:3]

    # Referenced By: Inbound links
    related['referenced_by'] = doc.inbound_links[:5]

    # Next Steps: Category progression
    if doc.category == 'tutorial':
        next_guides = find_related_guides(doc, all_docs)
        related['next_steps'] = next_guides[:3]

    return related
```

---

## 6. Tag-Based Navigation

### 6.1 Tag Taxonomy

**Technology Tags:**
```yaml
technology:
  - rust
  - actix
  - neo4j
  - cuda
  - three-js
  - react
  - typescript
  - websocket
  - graphql
```

**Feature Tags:**
```yaml
features:
  - agent-system
  - multi-agent
  - orchestration
  - graph-visualization
  - 3d-rendering
  - xr
  - webxr
  - physics
  - semantic-forces
```

**Domain Tags:**
```yaml
domain:
  - knowledge-graph
  - semantic-web
  - ontology
  - owl
  - reasoning
  - inference
  - linked-data
```

**Architecture Tags:**
```yaml
architecture:
  - hexagonal
  - cqrs
  - actor-model
  - ports-adapters
  - event-driven
```

### 6.2 Tag-Based Discovery

**Tag Index Page (`docs/tags/README.md`):**
```markdown
# Documentation by Tag

## Technology Tags

### 🦀 Rust
Documents: 45
- [Server Architecture](../explanations/architecture/core/server.md)
- [Actor System](../explanations/architecture/components/actor-system.md)
- [GPU Pipeline](../explanations/architecture/components/gpu-pipeline.md)
- [Binary WebSocket Protocol](../reference/protocols/binary-websocket.md)

### 🗄️ Neo4j
Documents: 32
- [Neo4j Setup](../guides/ontology/neo4j-setup.md)
- [Database Schemas](../reference/database/schemas.md)
- [Ontology Storage](../explanations/architecture/subsystems/ontology-storage.md)

### ⚡ CUDA
Documents: 18
- [GPU Acceleration](../explanations/physics/gpu-acceleration.md)
- [Barnes-Hut Algorithm](../explanations/physics/barnes-hut-algorithm.md)
- [Physics Simulation Guide](../guides/features/physics-simulation.md)

## Feature Tags

### 🤖 Agent System
Documents: 28
- [Agent Basics](../getting-started/04-agent-basics.md)
- [Agent Orchestration](../diagrams/mermaid-library/04-agent-orchestration.md)
- [Multi-Agent Guide](../guides/ai-models/multi-agent-orchestration.md)

## Domain Tags

### 🧠 Knowledge Graph
Documents: 56
- [Ontology Overview](../explanations/ontology/ontology-overview.md)
- [Graph Algorithms](../explanations/ontology/graph-algorithms.md)
- [Hierarchical Visualization](../explanations/ontology/hierarchical-visualization.md)
```

---

## 7. Search-Optimized Navigation

### 7.1 Search Keywords in Front Matter

```yaml
---
title: "Neo4j Setup Guide"
keywords:
  - neo4j installation
  - graph database setup
  - cypher queries
  - knowledge graph database
  - docker neo4j
  - neo4j configuration
search_boost: 1.5  # Boost in search results
---
```

### 7.2 Search Landing Page

**`docs/search/README.md`:**
```markdown
# Search Documentation

## Popular Searches

### Getting Started
- [Installation](../getting-started/01-installation.md) - Environment setup
- [First Graph](../getting-started/02-first-graph.md) - Quick start
- [Neo4j Setup](../guides/ontology/neo4j-setup.md) - Database setup

### Architecture
- [System Overview](../ARCHITECTURE_OVERVIEW.md) - High-level architecture
- [Hexagonal CQRS](../explanations/architecture/hexagonal-cqrs.md) - Design patterns
- [Actor System](../explanations/architecture/components/actor-system.md) - Concurrency

### API & Integration
- [REST API](../reference/api/rest-api-complete.md) - API reference
- [WebSocket Protocol](../reference/protocols/binary-websocket.md) - Binary protocol
- [Authentication](../guides/security/authentication.md) - Security

### Deployment & Operations
- [Deployment](../guides/infrastructure/deployment.md) - Production deployment
- [Docker Environment](../guides/infrastructure/docker-environment.md) - Containers
- [Troubleshooting](../guides/troubleshooting.md) - Common issues

## Search by Category
- [Tutorials](../getting-started/README.md)
- [Guides](../guides/README.md)
- [Explanations](../explanations/README.md)
- [Reference](../reference/README.md)

## Search by Tag
- [By Technology](../tags/README.md#technology)
- [By Feature](../tags/README.md#features)
- [By Domain](../tags/README.md#domain)
```

---

## 8. Implementation Scripts

### 8.1 Generate Navigation Script

```python
#!/usr/bin/env python3
# docs/scripts/generate-navigation.py

from pathlib import Path
import yaml

DOCS_ROOT = Path("/home/devuser/workspace/project/docs")

def generate_breadcrumbs(file_path: Path) -> List[Dict]:
    """Generate breadcrumbs from file path"""
    # (Implementation from section 2.2)
    pass

def generate_prev_next(doc: Document, all_docs: Dict) -> Dict:
    """Generate previous/next navigation"""
    # (Implementation from section 3.2)
    pass

def generate_sidebar(category: str) -> str:
    """Generate sidebar navigation for category"""
    # (Implementation from section 4)
    pass

def generate_related_docs(doc: Document, all_docs: Dict) -> Dict:
    """Generate related documentation sections"""
    # (Implementation from section 5.2)
    pass

def inject_navigation(doc_path: Path, navigation: Dict):
    """Inject navigation into document"""
    # Read document
    # Update front matter
    # Inject sidebar
    # Inject related docs footer
    # Write back
    pass

def main():
    # Scan all documents
    # Generate navigation for each
    # Inject into documents
    pass

if __name__ == "__main__":
    main()
```

---

## 9. Validation & Testing

### 9.1 Navigation Validation

**Test User Journeys:**
```bash
#!/bin/bash
# Test that all user journeys are navigable

# Journey 1: New Developer
python3 /home/devuser/workspace/project/docs/scripts/test-user-journey.py \
  --journey new-developer \
  --start README.md \
  --end guides/features/semantic-forces.md

# Journey 2: System Architect
python3 /home/devuser/workspace/project/docs/scripts/test-user-journey.py \
  --journey architect \
  --start ARCHITECTURE_OVERVIEW.md \
  --end reference/api/rest-api-complete.md
```

**Test Breadcrumb Integrity:**
```bash
# Verify all breadcrumbs resolve
python3 /home/devuser/workspace/project/docs/scripts/validate-breadcrumbs.py
```

**Test Prev/Next Chains:**
```bash
# Verify sequential navigation
python3 /home/devuser/workspace/project/docs/scripts/validate-sequences.py
```

---

## 10. Success Metrics

### Navigation Quality Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| Avg clicks to any doc from root | ≤3 clicks | User journey tests |
| Breadcrumb coverage | 100% | Breadcrumb validation |
| Prev/next coverage (tutorials) | 100% | Sequence validation |
| Related docs avg count | 5-10 | Link analysis |
| Tag coverage | 95%+ | Front matter check |
| Orphan documents | 0 | Link graph analysis |
| Sidebar index coverage | 100% | Directory scan |

---

**Document Status:** READY FOR IMPLEMENTATION
**Next Action:** Run navigation generation scripts
**Integration:** Combine with directory restructure and link generation
