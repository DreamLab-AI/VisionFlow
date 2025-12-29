---
title: Solid API Reference
description: Complete API reference for VisionFlow Solid Pod integration including endpoints, authentication, WebSocket protocol, and examples.
category: reference
tags:
  - api
  - solid
  - ldp
  - websocket
  - nostr
  - authentication
updated-date: 2025-12-29
difficulty-level: intermediate
---

# Solid API Reference

**Version**: 1.0
**Base URL**: `http://localhost:9090/solid`
**WebSocket URL**: `ws://localhost:9090/solid/ws`

Complete API reference for VisionFlow's Solid Pod integration.

---

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Pod Management](#pod-management)
  - [LDP Resource Operations](#ldp-resource-operations)
- [WebSocket Protocol](#websocket-protocol)
- [Error Responses](#error-responses)
- [Examples](#examples)

---

## Authentication

All Solid API requests require Nostr NIP-98 authentication.

### Bearer Token (Nostr Session)

Use the session token obtained from Nostr login:

```http
Authorization: Bearer <nostr_session_token>
```

The session token is obtained during the Nostr authentication flow and stored in `localStorage` as `nostr_session_token`.

### NIP-98 Header Format

For operations requiring fresh signatures, use NIP-98 HTTP Auth:

```http
Authorization: Nostr <base64_encoded_signed_event>
```

The signed event must include:

```json
{
  "kind": 27235,
  "created_at": 1703859000,
  "tags": [
    ["u", "http://localhost:9090/solid/pods"],
    ["method", "POST"],
    ["payload", "sha256_hash_of_body"]
  ],
  "content": "",
  "pubkey": "npub1...",
  "sig": "signature..."
}
```

**Event Fields**:

| Tag | Description | Required |
|-----|-------------|----------|
| `u` | Full request URL | Yes |
| `method` | HTTP method (GET, POST, PUT, DELETE) | Yes |
| `payload` | SHA-256 hash of request body (hex) | If body present |

**Timing Requirements**:
- `created_at` must be within 60 seconds of server time
- Events are single-use (replay protection)

---

## Endpoints

### Pod Management

#### POST /solid/pods - Create Pod

Create a new Solid Pod for the authenticated user.

**Request**:

```http
POST /solid/pods HTTP/1.1
Authorization: Nostr <signed_event>
Content-Type: application/json

{
  "name": "my-knowledge-base",
  "template": "visionflow-default"
}
```

**Request Body**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Pod identifier (alphanumeric, hyphens) |
| `template` | string | No | Pod template to use |

**Available Templates**:

| Template | Description |
|----------|-------------|
| `visionflow-default` | Full VisionFlow structure with memories, ontologies |
| `minimal` | Basic profile and preferences only |
| `agent-focused` | Optimised for agent memory storage |
| `ontology-contributor` | Focus on ontology proposals |

**Response** (201 Created):

```json
{
  "url": "/pods/npub1abc.../my-knowledge-base/",
  "webId": "https://visionflow.example/id/npub1abc.../profile/card#me",
  "created": "2025-12-29T10:30:00Z",
  "template": "visionflow-default",
  "containers": [
    "/pods/npub1abc.../my-knowledge-base/profile/",
    "/pods/npub1abc.../my-knowledge-base/agent-memories/",
    "/pods/npub1abc.../my-knowledge-base/ontologies/",
    "/pods/npub1abc.../my-knowledge-base/graphs/"
  ]
}
```

**Error Responses**:

| Status | Code | Description |
|--------|------|-------------|
| 400 | `INVALID_POD_NAME` | Pod name contains invalid characters |
| 401 | `UNAUTHORIZED` | Invalid or expired authentication |
| 409 | `POD_EXISTS` | Pod with this name already exists |
| 500 | `POD_CREATION_FAILED` | Internal server error during creation |

---

#### GET /solid/pods/check - Check Pod Exists

Check if a Pod exists for the authenticated user.

**Request**:

```http
GET /solid/pods/check?name=my-knowledge-base HTTP/1.1
Authorization: Bearer <session_token>
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Pod name to check |

**Response** (200 OK):

```json
{
  "exists": true,
  "url": "/pods/npub1abc.../my-knowledge-base/",
  "created": "2025-12-29T10:30:00Z",
  "size": 102400,
  "resourceCount": 42
}
```

**Response if not found** (200 OK):

```json
{
  "exists": false
}
```

---

#### GET /solid/pods - List User Pods

List all Pods owned by the authenticated user.

**Request**:

```http
GET /solid/pods HTTP/1.1
Authorization: Bearer <session_token>
```

**Response** (200 OK):

```json
{
  "pods": [
    {
      "name": "my-knowledge-base",
      "url": "/pods/npub1abc.../my-knowledge-base/",
      "created": "2025-12-29T10:30:00Z",
      "template": "visionflow-default"
    },
    {
      "name": "work-ontologies",
      "url": "/pods/npub1abc.../work-ontologies/",
      "created": "2025-12-28T14:20:00Z",
      "template": "ontology-contributor"
    }
  ],
  "totalCount": 2,
  "storageUsed": 512000,
  "storageQuota": 104857600
}
```

---

#### DELETE /solid/pods/{name} - Delete Pod

Delete a Pod and all its contents.

**Request**:

```http
DELETE /solid/pods/my-knowledge-base HTTP/1.1
Authorization: Nostr <signed_event>
```

**Response** (204 No Content): Empty body on success.

**Error Responses**:

| Status | Code | Description |
|--------|------|-------------|
| 401 | `UNAUTHORIZED` | Invalid authentication |
| 403 | `FORBIDDEN` | Cannot delete another user's Pod |
| 404 | `POD_NOT_FOUND` | Pod does not exist |

---

### LDP Resource Operations

All LDP operations follow the [Linked Data Platform](https://www.w3.org/TR/ldp/) specification.

#### GET /solid/{path} - Read Resource

Retrieve a resource or container listing.

**Request**:

```http
GET /solid/pods/npub1abc.../agent-memories/episodic/memory-001 HTTP/1.1
Authorization: Bearer <session_token>
Accept: application/ld+json
```

**Headers**:

| Header | Value | Description |
|--------|-------|-------------|
| `Accept` | `application/ld+json` | JSON-LD format (default) |
| `Accept` | `text/turtle` | Turtle RDF format |
| `Accept` | `application/n-triples` | N-Triples format |

**Response** (200 OK):

```http
HTTP/1.1 200 OK
Content-Type: application/ld+json
ETag: "abc123"
Link: <http://www.w3.org/ns/ldp#Resource>; rel="type"
```

```json
{
  "@context": {
    "@vocab": "https://visionflow.example/ns/memory#"
  },
  "@id": "/pods/npub1abc.../agent-memories/episodic/memory-001",
  "@type": "EpisodicMemory",
  "timestamp": "2025-12-29T10:30:00Z",
  "content": "User explored design patterns in the software ontology"
}
```

**Container Listing** (for directories):

```json
{
  "@context": {
    "ldp": "http://www.w3.org/ns/ldp#"
  },
  "@id": "/pods/npub1abc.../agent-memories/episodic/",
  "@type": "ldp:Container",
  "ldp:contains": [
    { "@id": "memory-001" },
    { "@id": "memory-002" },
    { "@id": "memory-003" }
  ]
}
```

---

#### PUT /solid/{path} - Update Resource

Replace a resource with new content.

**Request**:

```http
PUT /solid/pods/npub1abc.../profile/preferences HTTP/1.1
Authorization: Nostr <signed_event>
Content-Type: application/ld+json
If-Match: "abc123"
```

```json
{
  "@context": { "@vocab": "https://visionflow.example/ns/prefs#" },
  "@type": "UserPreferences",
  "theme": "dark",
  "language": "en-GB",
  "memoryRetention": "1y"
}
```

**Headers**:

| Header | Description |
|--------|-------------|
| `If-Match` | ETag for optimistic concurrency (recommended) |
| `If-None-Match` | `*` to only create if not exists |

**Response** (200 OK or 201 Created):

```json
{
  "@id": "/pods/npub1abc.../profile/preferences",
  "updated": "2025-12-29T11:00:00Z"
}
```

---

#### POST /solid/{path} - Create Resource

Create a new resource in a container.

**Request**:

```http
POST /solid/pods/npub1abc.../agent-memories/episodic/ HTTP/1.1
Authorization: Nostr <signed_event>
Content-Type: application/ld+json
Slug: memory-2025-12-29-001
```

```json
{
  "@context": { "@vocab": "https://visionflow.example/ns/memory#" },
  "@type": "EpisodicMemory",
  "timestamp": "2025-12-29T10:30:00Z",
  "content": "Session started - user authenticated via Nostr"
}
```

**Headers**:

| Header | Description |
|--------|-------------|
| `Slug` | Suggested resource name (server may modify) |
| `Link` | `<http://www.w3.org/ns/ldp#Resource>; rel="type"` for RDF resources |

**Response** (201 Created):

```http
HTTP/1.1 201 Created
Location: /pods/npub1abc.../agent-memories/episodic/memory-2025-12-29-001
```

---

#### DELETE /solid/{path} - Delete Resource

Remove a resource.

**Request**:

```http
DELETE /solid/pods/npub1abc.../agent-memories/episodic/old-memory HTTP/1.1
Authorization: Nostr <signed_event>
```

**Response** (204 No Content): Empty body on success.

---

#### PATCH /solid/{path} - Partial Update

Apply partial updates using SPARQL UPDATE or N3 Patch.

**Request (SPARQL UPDATE)**:

```http
PATCH /solid/pods/npub1abc.../profile/card HTTP/1.1
Authorization: Nostr <signed_event>
Content-Type: application/sparql-update
```

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
DELETE { <#me> foaf:name ?old }
INSERT { <#me> foaf:name "New Display Name" }
WHERE { <#me> foaf:name ?old }
```

**Response** (200 OK):

```json
{
  "success": true,
  "triples": {
    "added": 1,
    "removed": 1
  }
}
```

---

## WebSocket Protocol

### Connection

Connect to the Solid WebSocket endpoint for real-time notifications.

**Connection URL**:

```
ws://localhost:9090/solid/ws?token=<session_token>
```

**Handshake**:

```
GET /solid/ws HTTP/1.1
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Protocol: solid-0.1
```

### Commands

#### Subscribe (sub)

Subscribe to changes on a resource or container.

**Request**:

```json
{
  "type": "sub",
  "resource": "/pods/npub1abc.../agent-memories/",
  "recursive": true
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"sub"` |
| `resource` | string | Resource or container URL |
| `recursive` | boolean | Include child resources (containers only) |

**Response** (ack):

```json
{
  "type": "ack",
  "resource": "/pods/npub1abc.../agent-memories/",
  "subscriptionId": "sub-001"
}
```

#### Unsubscribe (unsub)

Cancel a subscription.

**Request**:

```json
{
  "type": "unsub",
  "resource": "/pods/npub1abc.../agent-memories/"
}
```

**Response**:

```json
{
  "type": "ack",
  "unsubscribed": "/pods/npub1abc.../agent-memories/"
}
```

### Notifications

#### Resource Changed (pub)

Notification when a subscribed resource changes.

**Message**:

```json
{
  "type": "pub",
  "resource": "/pods/npub1abc.../agent-memories/episodic/memory-005",
  "activity": "created",
  "actor": "npub1abc...",
  "timestamp": "2025-12-29T11:30:00Z",
  "container": "/pods/npub1abc.../agent-memories/episodic/"
}
```

**Activity Types**:

| Activity | Description |
|----------|-------------|
| `created` | New resource created |
| `updated` | Resource content modified |
| `deleted` | Resource removed |
| `moved` | Resource relocated |

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error description",
    "details": {
      "field": "Additional context"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource does not exist |
| `CONFLICT` | 409 | Resource already exists or version conflict |
| `INVALID_RDF` | 400 | Malformed RDF content |
| `INVALID_CONTENT_TYPE` | 415 | Unsupported media type |
| `PRECONDITION_FAILED` | 412 | If-Match/If-None-Match failed |
| `QUOTA_EXCEEDED` | 507 | Storage quota exceeded |
| `SERVER_ERROR` | 500 | Internal server error |

---

## Examples

### Complete Flow: Create Memory

```typescript
import { nostrAuth } from './nostrAuthService';

// 1. Get authentication
const token = nostrAuth.getSessionToken();
const user = nostrAuth.getCurrentUser();

// 2. Create the memory object
const memory = {
  "@context": {
    "@vocab": "https://visionflow.example/ns/memory#",
    "xsd": "http://www.w3.org/2001/XMLSchema#"
  },
  "@type": "EpisodicMemory",
  "timestamp": new Date().toISOString(),
  "sessionId": "session-" + Date.now(),
  "content": "User navigated to the design patterns section",
  "entities": [
    { "@id": "http://example.org/ontology#FactoryPattern" }
  ],
  "importance": 0.6
};

// 3. POST to create
const response = await fetch('/solid/pods/' + user.npub + '/agent-memories/episodic/', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/ld+json',
    'Slug': 'memory-' + Date.now()
  },
  body: JSON.stringify(memory)
});

if (response.status === 201) {
  const location = response.headers.get('Location');
  console.log('Memory created at:', location);
}
```

### Subscribe to Memory Updates

```typescript
const ws = new WebSocket(`ws://localhost:9090/solid/ws?token=${token}`);

ws.onopen = () => {
  // Subscribe to all memory changes
  ws.send(JSON.stringify({
    type: 'sub',
    resource: `/pods/${user.npub}/agent-memories/`,
    recursive: true
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'pub') {
    console.log(`Memory ${msg.activity}: ${msg.resource}`);
    // Trigger UI update
    refreshMemoryList();
  }
};
```

### Query Memories with SPARQL

```typescript
const query = `
  PREFIX vf: <https://visionflow.example/ns/memory#>
  SELECT ?memory ?content ?importance
  WHERE {
    ?memory a vf:EpisodicMemory ;
            vf:content ?content ;
            vf:importance ?importance .
    FILTER(?importance > 0.7)
  }
  ORDER BY DESC(?importance)
  LIMIT 5
`;

const response = await fetch(`/solid/pods/${user.npub}/agent-memories/`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/sparql-query',
    'Accept': 'application/json'
  },
  body: query
});

const results = await response.json();
console.log('Important memories:', results.results.bindings);
```

---

## Related Documentation

- [Solid Integration Guide](../../guides/solid-integration.md) - Getting started guide
- [Nostr Authentication](../../guides/features/nostr-auth.md) - Authentication details
- [REST API Complete](rest-api-complete.md) - Full REST API reference
- [WebSocket API](03-websocket.md) - WebSocket protocol reference

---

**Last Updated**: 2025-12-29
**Version**: 1.0
**Maintainer**: VisionFlow Documentation Team
