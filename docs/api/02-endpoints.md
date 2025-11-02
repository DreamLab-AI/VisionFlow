# API Endpoints Reference

## Base URL

```
http://localhost:9090/api
```

## Authentication

All endpoints except `/auth/*` require authentication.

## Endpoints

### Projects

#### List Projects

```
GET /api/projects
```

**Query Parameters**:
- `limit` (number): Max results (default: 20)
- `offset` (number): Pagination offset
- `status` (string): Filter by status

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid",
      "name": "My Project",
      "status": "active",
      "createdAt": "2025-01-23T10:00:00Z"
    }
  ],
  "pagination": {
    "total": 100,
    "limit": 20,
    "offset": 0
  }
}
```

#### Create Project

```
POST /api/projects
```

**Request Body**:
```json
{
  "name": "New Project",
  "description": "Project description",
  "config": {}
}
```

**Response**: `201 Created`

#### Get Project

```
GET /api/projects/:id
```

#### Update Project

```
PUT /api/projects/:id
```

#### Delete Project

```
DELETE /api/projects/:id
```

### Assets

#### Upload Asset

```
POST /api/assets
Content-Type: multipart/form-data
```

**Form Data**:
- `file`: File to upload
- `projectId`: Project UUID
- `name`: Asset name (optional)

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "name": "file.jpg",
    "size": 1024000,
    "type": "image/jpeg",
    "url": "/assets/uuid"
  }
}
```

#### List Assets

```
GET /api/assets?projectId=uuid
```

#### Download Asset

```
GET /api/assets/:id/download
```

### Users

#### Get Current User

```
GET /api/users/me
```

#### Update User Profile

```
PUT /api/users/me
```

### Graph Data

#### Get Complete Graph Data

```
GET /api/graph/data
```

**Description**: Returns complete graph data with nodes, edges, and metadata.

**Authentication**: Required

**Response Structure**:
```json
{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
      "metadata": {
        "owl_class_iri": "string",
        "file_source": "string"
      },
      "visual": {
        "color": "#hexcolor",
        "size": 1.0
      }
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id",
      "relationship_type": "string"
    }
  ],
  "metadata": {
    "node_count": 50,
    "edge_count": 12,
    "last_updated": "2025-11-02T13:00:00Z"
  }
}
```

**Example**: Typical response contains 50 nodes, 12 edges (~17KB JSON)

**Response**: `200 OK`

### Admin Operations

#### Trigger GitHub Synchronization

```
POST /api/admin/sync
```

**Description**: Triggers GitHub repository synchronization to import ontology files.

**Authentication**: Required (Admin)

**Environment Variables**:
- `FORCE_FULL_SYNC=1` - Bypass SHA1 filtering, process all files

**Response**:
```json
{
  "status": "success",
  "files_processed": 50,
  "nodes_created": 45,
  "edges_created": 12
}
```

**Response**: `200 OK` on success

## Database

The API uses a **unified database architecture** with `unified.db` containing all domain tables:
- `nodes` - Knowledge graph nodes
- `edges` - Relationships between nodes
- `owl_classes` - OWL ontology classes
- `owl_properties` - OWL ontology properties
- `github_sync_state` - Synchronization tracking

## Error Responses

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input",
    "details": [
      {
        "field": "name",
        "message": "Name is required"
      }
    ]
  }
}
```

## Status Codes

- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing/invalid auth
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
