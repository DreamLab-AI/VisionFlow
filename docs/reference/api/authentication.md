---
title: Authentication Reference
description: Complete reference for VisionFlow authentication methods including JWT, API keys, and Nostr NIP-98
category: reference
difficulty-level: intermediate
tags:
  - api
  - authentication
  - security
updated-date: 2025-01-29
---

# Authentication Reference

Complete reference for all VisionFlow authentication methods.

---

## Authentication Methods

VisionFlow supports three authentication methods:

| Method | Use Case | Token Lifetime | Best For |
|--------|----------|----------------|----------|
| **JWT** | User sessions | 24 hours | Web applications |
| **API Keys** | Programmatic access | Never expires | Integrations, scripts |
| **Nostr NIP-98** | Decentralized identity | Per-request | Decentralized apps |

---

## JWT Authentication

### Login Endpoint

```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure-password"
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

### Using JWT Tokens

Include in Authorization header:
```bash
curl -H "Authorization: Bearer YOUR-JWT-TOKEN" \
  http://localhost:9090/api/graph/data
```

### Token Structure

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiJ1c2VyLWlkIiwiZW1haWwiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNzAyOTE1MjAwfQ.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

**Payload Claims**:

| Claim | Type | Description |
|-------|------|-------------|
| `sub` | string | Subject (user ID) |
| `email` | string | User email |
| `role` | string | User role (admin, editor, viewer) |
| `exp` | integer | Expiration timestamp (Unix) |
| `iat` | integer | Issued at timestamp |
| `iss` | string | Issuer (visionflow) |

### Token Refresh

```http
POST /api/auth/refresh
Authorization: Bearer YOUR-EXPIRED-TOKEN
```

---

## API Keys

### Generate API Key

```bash
curl -X POST http://localhost:9090/api/auth/api-keys \
  -H "Authorization: Bearer YOUR-JWT-TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "apiKey": "vf_live_xxxxxxxxxxxx",
    "createdAt": "2025-12-18T12:00:00Z"
  }
}
```

### Using API Keys

```bash
curl -H "X-API-Key: YOUR-API-KEY" \
  http://localhost:9090/api/graph/data
```

### API Key Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/api-keys` | GET | List API keys |
| `/api/auth/api-keys` | POST | Create new key |
| `/api/auth/api-keys/{id}` | DELETE | Revoke key |

---

## Nostr NIP-98 Authentication

### Protocol Overview

NIP-98 provides HTTP authentication using Nostr event signatures (kind 27235).

### Event Structure

```json
{
  "kind": 27235,
  "created_at": 1702915200,
  "tags": [
    ["u", "https://visionflow.example.com/api/auth/nostr"],
    ["method", "POST"],
    ["payload", "sha256_hash_of_body"]
  ],
  "content": "",
  "pubkey": "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d",
  "id": "d42069d...",
  "sig": "908a15e..."
}
```

### Required Tags

| Tag | Description | Required |
|-----|-------------|----------|
| `u` | Full URL of the request | Yes |
| `method` | HTTP method (GET, POST, etc.) | Yes |
| `payload` | SHA-256 hash of request body | For POST/PUT |

### Client Implementation

```typescript
import { generatePrivateKey, getPublicKey, finishEvent } from 'nostr-tools';

const sk = generatePrivateKey();
const pk = getPublicKey(sk);

const authEvent = finishEvent({
  kind: 27235,
  created_at: Math.floor(Date.now() / 1000),
  tags: [
    ['u', window.location.href],
    ['method', 'POST']
  ],
  content: ''
}, sk);

// Base64 encode the event for HTTP header
const authHeader = `Nostr ${btoa(JSON.stringify(authEvent))}`;

const response = await fetch('/api/auth/nostr', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': authHeader
  },
  body: JSON.stringify({ pubkey: pk })
});
```

### Server Validation

1. **Signature Verification**: Validate Schnorr signature against pubkey
2. **Timestamp Check**: Event `created_at` within 60 seconds
3. **URL Match**: Tag `u` must match request URL exactly
4. **Method Match**: Tag `method` must match HTTP method
5. **Payload Hash**: For POST/PUT, verify payload hash if present

---

## Token Expiration Summary

| Token Type | Expiration | Refresh Method |
|------------|------------|----------------|
| JWT Access Token | 24 hours | `/auth/refresh` endpoint |
| Refresh Token | 30 days | Re-authenticate |
| API Keys | Never | Revoke and regenerate |
| Nostr Events | 60 seconds | Sign new event |

---

## Related Documentation

- [REST API Reference](./rest-api.md)
- [Error Codes](../error-codes.md)
- [Security Configuration](../configuration/environment-variables.md#authentication--security)
