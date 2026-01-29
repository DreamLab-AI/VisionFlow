---
title: Semantic Features API Reference
description: Get complete graph schema.
category: reference
tags:
  - api
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Semantic Features API Reference

## Schema Endpoints

### GET /api/schema
Get complete graph schema.

### GET /api/schema/node-types
List node types with counts.

### GET /api/schema/edge-types
List edge types with counts.

## Natural Language Query Endpoints

### POST /api/nl-query/translate
Translate natural language to Cypher.

### GET /api/nl-query/examples
Get example queries.

### POST /api/nl-query/explain
Explain Cypher query.

### POST /api/nl-query/validate
Validate Cypher syntax.

---

## Related Documentation

- [Authentication (DEPRECATED - JWT NOT USED)](01-authentication.md)
- [Pathfinding API Examples](pathfinding-examples.md)
- [Database Schema Reference](../database/README.md)
- [VisionFlow Binary WebSocket Protocol](../protocols/binary-websocket.md)
- [WebSocket Binary Protocol Reference](../websocket-protocol.md)

## Pathfinding Endpoints

### POST /api/pathfinding/semantic-path
Find shortest semantic path.

### POST /api/pathfinding/query-traversal
Explore graph by query.

### POST /api/pathfinding/chunk-traversal
Explore local neighborhood.

See feature guides for detailed examples.
