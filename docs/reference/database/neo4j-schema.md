---
title: Neo4j Schema Reference
description: Neo4j graph database schema for VisionFlow
category: reference
difficulty-level: intermediate
tags:
  - database
  - neo4j
  - graph
updated-date: 2025-01-29
---

# Neo4j Schema Reference

Neo4j graph database schema for VisionFlow knowledge graph traversal and analytics.

---

## Node Labels

### GraphNode

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

### OWLClass

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

---

## Relationship Types

### RELATES_TO

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

### SUBCLASS_OF

OWL SubClassOf relationships.

```cypher
(:OWLClass)-[:SUBCLASS_OF]->(:OWLClass)
```

### INSTANCE_OF

Class membership.

```cypher
(:GraphNode)-[:INSTANCE_OF]->(:OWLClass)
```

---

## Indexes

### Node Indexes

```cypher
// Full-text search on labels
CREATE INDEX graph_node_label_fulltext IF NOT EXISTS
FOR (n:GraphNode) ON (n.label);

// Full-text search on OWL classes
CREATE FULLTEXT INDEX owl_class_search IF NOT EXISTS
FOR (c:OWLClass) ON EACH [c.label, c.description];
```

### Relationship Indexes

```cypher
// Composite index on edge type and weight
CREATE INDEX edge_type_weight IF NOT EXISTS
FOR ()-[r:RELATES_TO]-() ON (r.type, r.weight);
```

---

## Common Query Patterns

### Get Node Neighbors

```cypher
MATCH (n:GraphNode {id: $nodeId})-[r:RELATES_TO]-(neighbor)
RETURN neighbor.id, neighbor.label, r.type, r.weight
LIMIT 50;
```

### Shortest Path

```cypher
MATCH path = shortestPath(
    (start:GraphNode {id: $startId})-[*]-(end:GraphNode {id: $endId})
)
RETURN [node IN nodes(path) | node.id] AS path,
       length(path) AS pathLength;
```

### Community Detection (Louvain)

```cypher
CALL gds.louvain.stream('graph-projection')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).id AS nodeId, communityId
ORDER BY communityId;
```

### PageRank

```cypher
CALL gds.pageRank.stream('graph-projection')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id AS nodeId, score
ORDER BY score DESC
LIMIT 100;
```

### Class Hierarchy Traversal

```cypher
MATCH path = (child:OWLClass)-[:SUBCLASS_OF*1..5]->(parent:OWLClass)
WHERE child.iri = $classIri
RETURN [node IN nodes(path) | node.label] AS hierarchy;
```

### Get All Instances of Class

```cypher
MATCH (instance:GraphNode)-[:INSTANCE_OF]->(class:OWLClass {iri: $classIri})
RETURN instance.id, instance.label, instance.type;
```

---

## Performance Characteristics

| Query Type | Typical Time |
|------------|--------------|
| Get node by ID | 1.2 ms |
| Get neighbors (depth=1) | 1.8 ms |
| Shortest path | 15 ms |
| Community detection | 450 ms |
| Full-text search | 3 ms |

---

## Graph Data Science (GDS) Projections

### Create Projection

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

### Drop Projection

```cypher
CALL gds.graph.drop('graph-projection');
```

---

## Synchronisation with SQLite

### Node Sync

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

### Edge Sync

```cypher
// Create edge from SQLite
MATCH (source:GraphNode {id: $sourceId})
MATCH (target:GraphNode {id: $targetId})
MERGE (source)-[r:RELATES_TO {edgeId: $edgeId}]->(target)
SET r.type = $type,
    r.weight = $weight;
```

---

## Related Documentation

- [Database Schema Reference](./README.md)
- [Unified Schema (SQLite)](./schemas.md)
- [API Reference](../api/README.md)
