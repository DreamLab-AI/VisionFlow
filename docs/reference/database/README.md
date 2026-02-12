---
title: Database Schema Reference
description: Complete database schema documentation for VisionFlow
category: reference
difficulty-level: intermediate
tags:
  - database
  - schema
  - reference
updated-date: 2025-01-29
---

# Database Schema Reference

Complete database schema documentation for VisionFlow including SQLite, Neo4j, and Solid pod schemas.

---

## Database Architecture

### Dual Database Strategy

VisionFlow uses two complementary databases:

| Database | Technology | Purpose | Performance |
|----------|------------|---------|-------------|
| **Graph DB** | Neo4j | Graph data, relationships, analytics | 1M+ nodes, <20ms queries |
| **Ontology Store** | In-Memory | OWL classes, axioms, reasoning | Sub-millisecond reads |

---

## Documentation Index

| Topic | File | Description |
|-------|------|-------------|
| **Unified Schema** | [schemas.md](./schemas.md) | Legacy SQLite schema reference |
| **Neo4j Schema** | [neo4j-schema.md](./neo4j-schema.md) | Neo4j graph database schema |
| **Ontology Schema** | [ontology-schema-v2.md](./ontology-schema-v2.md) | OWL ontology storage |
| **Solid Pod Schema** | [solid-pod-schema.md](./solid-pod-schema.md) | Decentralized data storage |
| **User Settings** | [user-settings-schema.md](./user-settings-schema.md) | User preferences |

---

## Quick Reference

### Core SQLite Tables

| Table | Purpose |
|-------|---------|
| `graph_nodes` | Knowledge graph nodes with 3D positions |
| `graph_edges` | Relationships between nodes |
| `owl_classes` | OWL ontology class definitions |
| `owl_class_hierarchy` | SubClassOf relationships |
| `owl_properties` | OWL property definitions |
| `owl_axioms` | Complete OWL axiom storage |
| `file_metadata` | Source file tracking |
| `graph_statistics` | Runtime metrics |

### Neo4j Node Labels

| Label | Purpose |
|-------|---------|
| `GraphNode` | Primary knowledge graph nodes |
| `OWLClass` | Ontology class definitions |

### Neo4j Relationship Types

| Type | Purpose |
|------|---------|
| `RELATES_TO` | Generic relationships from graph_edges |
| `SUBCLASS_OF` | OWL SubClassOf relationships |
| `INSTANCE_OF` | Class membership |

---

## Data Flow

```
Source Files --> File Metadata Table
     |
     v
  Sync Process
     |
     +--> Neo4j: Graph DB
     |         |
     |         +--> GraphNode labels
     |         +--> Relationships
     |         +--> owl_classes
     |         +--> owl_axioms
     |
     +--> In-Memory: OntologyRepository
               |
               +--> GraphNode labels
               +--> Relationships
```

---

## Related Documentation

- [ADR-0001](../../architecture/adr/ADR-0001-neo4j-persistent-with-filesystem-sync.md) - Database architecture decision
- [API Reference](../api/README.md)
- [Configuration Reference](../configuration/README.md)
- [Error Codes](../error-codes.md)
