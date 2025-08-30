# Analytics API Endpoints

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