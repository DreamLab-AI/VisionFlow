# Authentication

## Overview

VisionFlow API uses JWT (JSON Web Tokens) for authentication.

## Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API

    Client->>API: POST /auth/login
    API->>Client: JWT Token

    Client->>API: Request + JWT
    API->>Client: Response
```

## Login

**Endpoint**: `POST /api/auth/login`

**Request**:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "uuid",
      "email": "user@example.com",
      "role": "user"
    }
  }
}
```

## Using Tokens

Include JWT in Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:9090/api/projects
```

## API Keys

Generate API keys for programmatic access:

```bash
curl -X POST http://localhost:9090/api/auth/api-keys \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

Use API keys:
```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:9090/api/projects
```

## Token Expiration

- JWT tokens expire in 24 hours
- Refresh tokens expire in 30 days
- Use `/auth/refresh` to get new tokens

## Security Best Practices

1. Store tokens securely
2. Use HTTPS in production
3. Rotate API keys regularly
4. Never commit tokens to git
