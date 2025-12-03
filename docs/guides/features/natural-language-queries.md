---
title: Natural Language Queries Tutorial
description: VisionFlow translates natural language questions into Cypher queries using LLM-powered semantic understanding.
type: guide
status: stable
---

# Natural Language Queries Tutorial

## Overview

VisionFlow translates natural language questions into Cypher queries using LLM-powered semantic understanding.

## Quick Start

### Simple Query

```bash
POST /api/nl-query/translate
Content-Type: application/json

{
  "query": "Show me all person nodes",
  "suggestAlternatives": false
}
```

Response:
```json
{
  "translations": [{
    "originalQuery": "Show me all person nodes",
    "cypherQuery": "MATCH (n:GraphNode {node_type: 'person'}) RETURN n LIMIT 50",
    "explanation": "Finds all nodes with type 'person'",
    "confidence": 0.95,
    "warnings": []
  }]
}
```

## Query Patterns

### Find Relationships

```
"What are the dependencies of Project X?"
```

Generates:
```cypher
MATCH (p:GraphNode {label: 'Project X'})-[r:EDGE {relation_type: 'dependency'}]->(dep:GraphNode)
RETURN dep
```

### Path Queries

```
"Show me the shortest path between Alice and Bob"
```

Generates:
```cypher
MATCH path = shortestPath(
  (a:GraphNode {label: 'Alice'})-[*]-(b:GraphNode {label: 'Bob'})
)
RETURN path
```

### Neighborhood Queries

```
"Find all nodes within 2 hops of Node X"
```

Generates:
```cypher
MATCH (start:GraphNode {label: 'Node X'})-[*1..2]-(connected:GraphNode)
RETURN DISTINCT connected LIMIT 100
```

## Advanced Features

### Multiple Suggestions

Request alternative interpretations:

```json
{
  "query": "connected nodes",
  "suggestAlternatives": true
}
```

Returns 3 different Cypher interpretations.

### Explain Cypher

Reverse translation - explain what a Cypher query does:

```bash
POST /api/nl-query/explain
Content-Type: application/json

{
  "cypher": "MATCH (n)-[r*1..3]-(m) RETURN n, m LIMIT 10"
}
```

### Validate Syntax

Check Cypher before execution:

```bash
POST /api/nl-query/validate
Content-Type: application/json

{
  "cypher": "MATCH (n:GraphNode) RETURN n"
}
```

## Examples

Get curated examples:

```bash
GET /api/nl-query/examples
```

Returns:
- Person node queries
- Dependency relationship queries
- Hierarchy queries
- Path queries
- Neighborhood queries

## Tips

1. **Be specific**: "person nodes" vs "nodes"
2. **Use labels**: Mention specific node labels for better results
3. **Mention relationships**: "connected by dependency" vs "connected"
4. **Check confidence**: Low confidence? Try rephrasing
5. **Review before execution**: Always validate generated queries

## Schema Context

The service automatically uses your graph schema to generate appropriate queries. It knows:
- Available node types
- Available edge types
- Common properties
- OWL classes and properties
