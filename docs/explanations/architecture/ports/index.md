---
layout: default
title: Port Interfaces
parent: Architecture
grand_parent: Explanations
nav_order: 5
has_children: true
permalink: /explanations/architecture/ports/
---

# Port Interfaces (Hexagonal Architecture)

Port definitions for VisionFlow's hexagonal architecture implementation.

## What are Ports?

In hexagonal architecture, ports define the interfaces through which the application core communicates with the outside world. They provide abstraction boundaries that enable:

- Swappable adapters (database, GPU, external services)
- Testability through mock implementations
- Clear separation of concerns

## Port Definitions

| Document | Port | Description |
|----------|------|-------------|
| [01-overview](./01-overview.md) | Overview | Port system architecture overview |
| [02-settings-repository](./02-settings-repository.md) | SettingsRepository | User and application settings persistence |
| [03-knowledge-graph-repository](./03-knowledge-graph-repository.md) | KnowledgeGraphRepository | Graph data storage and retrieval |
| [04-ontology-repository](./04-ontology-repository.md) | OntologyRepository | Ontology schema management |
| [05-inference-engine](./05-inference-engine.md) | InferenceEngine | Semantic reasoning and inference |
| [06-gpu-physics-adapter](./06-gpu-physics-adapter.md) | GpuPhysicsAdapter | GPU physics simulation interface |
| [07-gpu-semantic-analyzer](./07-gpu-semantic-analyzer.md) | GpuSemanticAnalyzer | GPU semantic analysis interface |
