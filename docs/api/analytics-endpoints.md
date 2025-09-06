# Analytics API Endpoints

*[Api](../index.md)*

This document describes the REST API endpoints for accessing the server's advanced analytics capabilities, including graph clustering and ontology validation.

## Clustering Endpoints

These endpoints provide an interface to the GPU-accelerated graph clustering features. For a more detailed explanation of the concepts and algorithms, see the [Graph Clustering documentation](../../server/features/clustering.md).

### `POST /api/analytics/clustering/run`

Initiates an asynchronous clustering task.

**Request Body:**
```json
{
  "algorithm": "louvain",
  "parameters": {
    "resolution": 1.0,
    "tolerance": 0.0001
  }
}
```
-   `algorithm` (string, required): The clustering algorithm to use. Supported values are `spectral`, `kmeans`, and `louvain`.
-   `parameters` (object, optional): Algorithm-specific parameters.

**Response:**
```json
{
  "taskId": "a-unique-task-id"
}
```
-   `taskId` (string): A unique identifier for the clustering task, which can be used to poll for results.

### `GET /api/analytics/clustering/status?task_id=<id>`

Retrieves the status and results of a clustering task.

**Query Parameters:**
-   `task_id` (string, required): The ID of the task to query.

**Response (Success):**
```json
{
  "status": "completed",
  "results": {
    "node_1": 0,
    "node_2": 1,
    "node_3": 0
  }
}
```
-   `status` (string): The current status of the task (`pending`, `running`, `completed`, or `failed`).
-   `results` (object): If the task is `completed`, this object maps each node ID to its assigned cluster ID.

## Ontology Validation Endpoint

This endpoint provides an interface to the ontology validation and reasoning engine. For a more detailed explanation of the concepts, see the [Ontology Validation documentation](../../server/features/ontology.md).

### `POST /api/analytics/validate` (Planned)

Initiates a formal validation of the knowledge graph against the loaded ontology.

**Request Body:**
An empty request body is sufficient to trigger the validation of the current graph state.

**Response:**
```json
{
  "isValid": false,
  "inconsistencies": [
    {
      "type": "DisjointClass",
      "node": "NodeA",
      "message": "NodeA cannot be an instance of both :Person and :Company, as they are disjoint classes."
    }
  ],
  "inferredStatements": [
    {
      "subject": "NodeC",
      "predicate": "worksFor",
      "object": "NodeD"
    }
  ]
}
```
-   `isValid` (boolean): `true` if the graph is logically consistent with the ontology, `false` otherwise.
-   `inconsistencies` (array): A list of any logical contradictions found in the graph.
-   `inferredStatements` (array): A list of new relationships that were inferred by the reasoner.



## See Also

- [Request Handlers Architecture](../server/handlers.md) - Server implementation

## Related Topics

- [AI Services Documentation](../server/ai-services.md) - Implementation
- [Actor System](../server/actors.md) - Implementation
- [Graph API Reference](../api/rest/graph.md)
- [Modern Settings API - Path-Based Architecture](../MODERN_SETTINGS_API.md)
- [Multi-MCP Agent Visualisation API Reference](../api/multi-mcp-visualization-api.md)
- [REST API Bloom/Glow Field Validation Fix](../REST_API_BLOOM_GLOW_VALIDATION_FIX.md)
- [REST API Reference](../api/rest/index.md)
- [Request Handlers Architecture](../server/handlers.md) - Implementation
- [Services Architecture](../server/services.md) - Implementation
- [Settings API Reference](../api/rest/settings.md)
- [Single-Source Shortest Path (SSSP) API](../api/shortest-path-api.md)
- [VisionFlow API Documentation](../api/index.md)
- [VisionFlow MCP Integration Documentation](../api/mcp/index.md)
- [VisionFlow WebSocket API Documentation](../api/websocket/index.md)
- [WebSocket API Reference](../api/websocket.md)
- [WebSocket Protocols](../api/websocket-protocols.md)
- [dev-backend-api](../reference/agents/development/backend/dev-backend-api.md)
- [docs-api-openapi](../reference/agents/documentation/api-docs/docs-api-openapi.md)
