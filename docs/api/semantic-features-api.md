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

## Pathfinding Endpoints

### POST /api/pathfinding/semantic-path
Find shortest semantic path.

### POST /api/pathfinding/query-traversal
Explore graph by query.

### POST /api/pathfinding/chunk-traversal
Explore local neighborhood.

See feature guides for detailed examples.
