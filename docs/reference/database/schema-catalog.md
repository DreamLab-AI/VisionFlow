---
title: VisionFlow Database Schema Catalog
description: Complete database schema reference for VisionFlow including SQLite unified schema, Neo4j graph database, and Solid pod structure.
category: reference
tags:
  - database
  - schema
  - sqlite
  - neo4j
  - visionflow
updated-date: 2025-01-29
difficulty-level: intermediate
---

# VisionFlow Database Schema Catalog

This document provides the complete database schema reference for VisionFlow, covering the SQLite unified storage, Neo4j graph database for analytics, and Solid pod structure for decentralized storage.

---

## Architecture Overview

VisionFlow uses a dual-database architecture:

| Database | Purpose | Use Case |
|----------|---------|----------|
| **SQLite** | Primary storage | CRUD operations, file metadata, ontology classes |
| **Neo4j** | Graph analytics | Traversal queries, community detection, PageRank |

Data synchronizes from SQLite to Neo4j for analytics workloads.

---

## SQLite Unified Schema

### Core Tables

#### graph_nodes

Primary knowledge graph nodes.

```sql
CREATE TABLE graph_nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    metadata_id     TEXT UNIQUE NOT NULL,      -- UUID for cross-system refs
    label           TEXT NOT NULL,              -- Display name
    type            TEXT DEFAULT 'concept',     -- concept, entity, class, individual
    color           TEXT DEFAULT '#3498db',     -- Hex color code
    size            REAL DEFAULT 1.0,           -- Visual size multiplier
    metadata        TEXT,                       -- JSON string
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_nodes_metadata_id ON graph_nodes(metadata_id);
CREATE INDEX idx_nodes_type ON graph_nodes(type);
CREATE INDEX idx_nodes_label ON graph_nodes(label);
```

**Node Types**:
| Type | Description |
|------|-------------|
| `concept` | Abstract concept from markdown |
| `entity` | Named entity (person, org, etc.) |
| `class` | OWL class definition |
| `individual` | OWL individual instance |

#### graph_edges

Relationships between nodes.

```sql
CREATE TABLE graph_edges (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id           INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    target_id           INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    relationship_type   TEXT DEFAULT 'related-to',
    weight              REAL DEFAULT 1.0,
    metadata            TEXT,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id, relationship_type)
);

CREATE INDEX idx_edges_source ON graph_edges(source_id);
CREATE INDEX idx_edges_target ON graph_edges(target_id);
CREATE INDEX idx_edges_type ON graph_edges(relationship_type);
```

**Relationship Types**:
| Type | Description |
|------|-------------|
| `related-to` | Generic semantic relationship |
| `subclass-of` | OWL SubClassOf |
| `instance-of` | OWL ClassAssertion |
| `property-assertion` | OWL ObjectPropertyAssertion |
| `hyperlink` | Markdown/Wiki link |

### OWL Ontology Tables

#### owl_classes

OWL class definitions.

```sql
CREATE TABLE owl_classes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    iri         TEXT UNIQUE NOT NULL,          -- Full IRI
    label       TEXT,                          -- rdfs:label
    description TEXT,                          -- rdfs:comment
    source_file TEXT,                          -- Origin ontology file
    deprecated  INTEGER DEFAULT 0,             -- owl:deprecated
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_owl_classes_iri ON owl_classes(iri);
CREATE INDEX idx_owl_classes_label ON owl_classes(label);
```

#### owl_class_hierarchy

Class hierarchy relationships (precomputed for performance).

```sql
CREATE TABLE owl_class_hierarchy (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    child_iri   TEXT NOT NULL,
    parent_iri  TEXT NOT NULL,
    depth       INTEGER DEFAULT 1,             -- Hierarchy depth
    source      TEXT DEFAULT 'subClassOf',     -- axiom source
    UNIQUE(child_iri, parent_iri)
);

CREATE INDEX idx_hierarchy_child ON owl_class_hierarchy(child_iri);
CREATE INDEX idx_hierarchy_parent ON owl_class_hierarchy(parent_iri);
```

#### owl_properties

OWL property definitions.

```sql
CREATE TABLE owl_properties (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    iri             TEXT UNIQUE NOT NULL,
    type            TEXT NOT NULL,             -- ObjectProperty, DataProperty, AnnotationProperty
    domain_iri      TEXT,
    range_iri       TEXT,
    label           TEXT,
    functional      INTEGER DEFAULT 0,
    inverse_of      TEXT,
    source_file     TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_owl_props_iri ON owl_properties(iri);
CREATE INDEX idx_owl_props_type ON owl_properties(type);
```

#### owl_axioms

Raw OWL axioms for reasoning.

```sql
CREATE TABLE owl_axioms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    axiom_type  TEXT NOT NULL,                 -- SubClassOf, EquivalentClasses, etc.
    subject_iri TEXT NOT NULL,
    predicate   TEXT,
    object_iri  TEXT,
    literal     TEXT,                          -- For data property assertions
    source_file TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_axioms_subject ON owl_axioms(subject_iri);
CREATE INDEX idx_axioms_type ON owl_axioms(axiom_type);
```

**Axiom Types**:
| Type | Description |
|------|-------------|
| `SubClassOf` | Class subsumption |
| `EquivalentClasses` | Class equivalence |
| `DisjointClasses` | Class disjointness |
| `ClassAssertion` | Individual membership |
| `ObjectPropertyAssertion` | Object property instance |
| `DataPropertyAssertion` | Data property instance |

### Metadata Tables

#### graph_statistics

Cached graph analytics.

```sql
CREATE TABLE graph_statistics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name     TEXT UNIQUE NOT NULL,
    metric_value    REAL,
    computed_at     TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**Metrics**:
- `node_count`, `edge_count`, `density`
- `avg_degree`, `max_degree`, `diameter`
- `clustering_coefficient`, `modularity`

#### file_metadata

Source file tracking for incremental updates.

```sql
CREATE TABLE file_metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT UNIQUE NOT NULL,
    sha1_hash   TEXT NOT NULL,
    file_size   INTEGER,
    mime_type   TEXT,
    processed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    node_count  INTEGER DEFAULT 0
);

CREATE INDEX idx_file_sha1 ON file_metadata(sha1_hash);
```

---

## Neo4j Graph Schema

Neo4j is used for graph analytics and traversal queries.

### Node Labels

#### GraphNode

Primary knowledge graph nodes synchronised from SQLite.

**Constraints**:
```cypher
CREATE CONSTRAINT graph_node_id IF NOT EXISTS
FOR (n:GraphNode) REQUIRE n.id IS UNIQUE;

CREATE INDEX graph_node_label IF NOT EXISTS
FOR (n:GraphNode) ON (n.label);

CREATE INDEX graph_node_type IF NOT EXISTS
FOR (n:GraphNode) ON (n.type);
```

**Properties**:
```cypher
(:GraphNode {
  id: INTEGER,           // Maps to SQLite graph_nodes.id
  metadataId: STRING,    // UUID for cross-system references
  label: STRING,         // Display name
  type: STRING,          // concept, entity, class, individual
  color: STRING,         // Hex color code
  size: FLOAT,           // Visual size multiplier
  metadata: STRING       // JSON string
})
```

#### OWLClass

Ontology class definitions.

**Constraints**:
```cypher
CREATE CONSTRAINT owl_class_iri IF NOT EXISTS
FOR (c:OWLClass) REQUIRE c.iri IS UNIQUE;

CREATE INDEX owl_class_label IF NOT EXISTS
FOR (c:OWLClass) ON (c.label);
```

**Properties**:
```cypher
(:OWLClass {
  iri: STRING,           // IRI (Internationalized Resource Identifier)
  label: STRING,         // Human-readable label
  description: STRING,   // Class description
  sourceFile: STRING     // Source ontology file
})
```

### Relationship Types

#### RELATES_TO

Generic relationships from graph_edges.

```cypher
(:GraphNode)-[:RELATES_TO {
  edgeId: STRING,        // UUID edge identifier
  type: STRING,          // relationship_type from SQLite
  weight: FLOAT          // Edge weight
}]->(:GraphNode)
```

**Common Types**:
- `related-to`: Generic relationship
- `subclass-of`: OWL SubClassOf
- `instance-of`: OWL ClassAssertion
- `property-assertion`: OWL PropertyAssertion
- `hyperlink`: Markdown/Wiki link

#### SUBCLASS_OF

OWL SubClassOf relationships.

```cypher
(:OWLClass)-[:SUBCLASS_OF]->(:OWLClass)
```

#### INSTANCE_OF

Class membership.

```cypher
(:GraphNode)-[:INSTANCE_OF]->(:OWLClass)
```

### Indexes

```cypher
// Full-text search on labels
CREATE INDEX graph_node_label_fulltext IF NOT EXISTS
FOR (n:GraphNode) ON (n.label);

// Full-text search on OWL classes
CREATE FULLTEXT INDEX owl_class_search IF NOT EXISTS
FOR (c:OWLClass) ON EACH [c.label, c.description];

// Composite index on edge type and weight
CREATE INDEX edge_type_weight IF NOT EXISTS
FOR ()-[r:RELATES_TO]-() ON (r.type, r.weight);
```

### Common Query Patterns

#### Get Node Neighbors

```cypher
MATCH (n:GraphNode {id: $nodeId})-[r:RELATES_TO]-(neighbor)
RETURN neighbor.id, neighbor.label, r.type, r.weight
LIMIT 50;
```

#### Shortest Path

```cypher
MATCH path = shortestPath(
    (start:GraphNode {id: $startId})-[*]-(end:GraphNode {id: $endId})
)
RETURN [node IN nodes(path) | node.id] AS path,
       length(path) AS pathLength;
```

#### Community Detection (Louvain)

```cypher
CALL gds.louvain.stream('graph-projection')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).id AS nodeId, communityId
ORDER BY communityId;
```

#### PageRank

```cypher
CALL gds.pageRank.stream('graph-projection')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id AS nodeId, score
ORDER BY score DESC
LIMIT 100;
```

#### Class Hierarchy Traversal

```cypher
MATCH path = (child:OWLClass)-[:SUBCLASS_OF*1..5]->(parent:OWLClass)
WHERE child.iri = $classIri
RETURN [node IN nodes(path) | node.label] AS hierarchy;
```

#### Get All Instances of Class

```cypher
MATCH (instance:GraphNode)-[:INSTANCE_OF]->(class:OWLClass {iri: $classIri})
RETURN instance.id, instance.label, instance.type;
```

### Performance Characteristics

| Query Type | Typical Time |
|------------|--------------|
| Get node by ID | 1.2 ms |
| Get neighbors (depth=1) | 1.8 ms |
| Shortest path | 15 ms |
| Community detection | 450 ms |
| Full-text search | 3 ms |

### Graph Data Science (GDS) Projections

#### Create Projection

```cypher
CALL gds.graph.project(
    'graph-projection',
    'GraphNode',
    {
        RELATES_TO: {
            orientation: 'UNDIRECTED',
            properties: ['weight']
        }
    }
);
```

#### Drop Projection

```cypher
CALL gds.graph.drop('graph-projection');
```

### Synchronisation with SQLite

#### Node Sync

```cypher
// Upsert node from SQLite
MERGE (n:GraphNode {id: $id})
SET n.metadataId = $metadataId,
    n.label = $label,
    n.type = $type,
    n.color = $color,
    n.size = $size,
    n.metadata = $metadata;
```

#### Edge Sync

```cypher
// Create edge from SQLite
MATCH (source:GraphNode {id: $sourceId})
MATCH (target:GraphNode {id: $targetId})
MERGE (source)-[r:RELATES_TO {edgeId: $edgeId}]->(target)
SET r.type = $type,
    r.weight = $weight;
```

---

## Solid Pod Structure

VisionFlow supports decentralized storage via Solid pods.

### Directory Layout

```
/pods/{userId}/
  profile/
    card                    # WebID Profile (Turtle)
  graph/
    nodes/
      {nodeId}.ttl         # Individual node resources
    edges/
      {edgeId}.ttl         # Edge resources
    ontology/
      classes.ttl          # OWL class definitions
      properties.ttl       # OWL property definitions
  settings/
    preferences.ttl        # User preferences
    acl.ttl               # Access control
```

### JSON-LD Contexts

#### Node Context

```json
{
  "@context": {
    "@vocab": "https://visionflow.example.com/ontology#",
    "vf": "https://visionflow.example.com/ontology#",
    "ldp": "http://www.w3.org/ns/ldp#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "label": "vf:label",
    "type": "vf:nodeType",
    "position": {
      "@id": "vf:position",
      "@type": "@json"
    },
    "metadata": {
      "@id": "vf:metadata",
      "@type": "@json"
    },
    "createdAt": {
      "@id": "vf:createdAt",
      "@type": "xsd:dateTime"
    }
  }
}
```

#### Edge Context

```json
{
  "@context": {
    "@vocab": "https://visionflow.example.com/ontology#",
    "vf": "https://visionflow.example.com/ontology#",
    "source": {
      "@id": "vf:sourceNode",
      "@type": "@id"
    },
    "target": {
      "@id": "vf:targetNode",
      "@type": "@id"
    },
    "relationshipType": "vf:relationshipType",
    "weight": {
      "@id": "vf:weight",
      "@type": "xsd:decimal"
    }
  }
}
```

---

## Entity-Relationship Diagram

```
+----------------+       +----------------+       +----------------+
|  graph_nodes   |       |  graph_edges   |       |  owl_classes   |
+----------------+       +----------------+       +----------------+
| id (PK)        |<----->| source_id (FK) |       | id (PK)        |
| metadata_id    |<----->| target_id (FK) |       | iri            |
| label          |       | relationship   |       | label          |
| type           |       | weight         |       | description    |
| color          |       | metadata       |       | source_file    |
| size           |       +----------------+       | deprecated     |
| metadata       |                                +----------------+
+----------------+                                        |
        |                                                 |
        v                                                 v
+------------------+                           +----------------------+
| file_metadata    |                           | owl_class_hierarchy  |
+------------------+                           +----------------------+
| id (PK)          |                           | id (PK)              |
| file_path        |                           | child_iri            |
| sha1_hash        |                           | parent_iri           |
| processed_at     |                           | depth                |
| node_count       |                           +----------------------+
+------------------+
        |
        v
+------------------+
| owl_properties   |
+------------------+
| id (PK)          |
| iri              |
| type             |
| domain_iri       |
| range_iri        |
| label            |
+------------------+
        |
        v
+------------------+
| owl_axioms       |
+------------------+
| id (PK)          |
| axiom_type       |
| subject_iri      |
| predicate        |
| object_iri       |
| literal          |
+------------------+
```

---

## Migration Strategy

### Phase 1: SQLite Primary

- All CRUD operations use SQLite
- Neo4j populated via batch sync
- Sync runs every 5 minutes or on-demand

### Phase 2: Hybrid Operations

- Read queries routed by complexity
- Simple lookups: SQLite
- Graph traversals: Neo4j
- Write operations: SQLite with async Neo4j sync

### Phase 3: Full Integration

- Real-time sync via change data capture
- Neo4j becomes read replica
- SQLite remains source of truth

---

## Related Documentation

- [REST API Reference](../api/rest-api.md)
- [WebSocket Protocol](../protocols/websocket-binary-v2.md)
- [Physics Implementation](../physics-implementation.md)

---

**Last Updated**: January 29, 2025
**Maintainer**: VisionFlow Database Team
