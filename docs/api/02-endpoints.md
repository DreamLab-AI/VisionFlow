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
