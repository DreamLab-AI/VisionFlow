---
layout: default
title: Settings API Authentication
parent: Features
grand_parent: Guides
nav_order: 10
description: Nostr-based session token authentication for settings API endpoints
---


# Settings API Authentication

## Overview

The settings API endpoints now support Nostr-based authentication using session tokens. This provides secure access control while maintaining backward compatibility with read-only anonymous access.

## Authentication Architecture

### Components

1. **NostrService** (`src/services/nostr_service.rs`)
   - Manages user sessions and tokens
   - Validates Nostr authentication events
   - Tracks power user status

2. **AuthExtractor** (`src/settings/auth_extractor.rs`)
   - Actix-web `FromRequest` implementation
   - Extracts and validates session tokens
   - Provides `AuthenticatedUser` and `OptionalAuth` types

3. **Settings Routes** (`src/settings/api/settings_routes.rs`)
   - Protected endpoints requiring authentication for writes
   - Read-only access for anonymous users

## Authentication Types

### AuthenticatedUser

Required authentication - returns 401 if not authenticated.

```rust
pub struct AuthenticatedUser {
    pub pubkey: String,
    pub is_power_user: bool,
}
```

**Usage in handlers:**
```rust
pub async fn update_settings(
    auth: AuthenticatedUser,  // Will fail if not authenticated
    body: web::Json<Settings>,
) -> impl Responder {
    info!("User {} updating settings", auth.pubkey);
    // Handle update
}
```

### OptionalAuth

Optional authentication - allows both authenticated and anonymous access.

```rust
pub struct OptionalAuth(pub Option<AuthenticatedUser>);
```

**Usage in handlers:**
```rust
pub async fn get_settings(
    auth: OptionalAuth,
) -> impl Responder {
    match auth.0 {
        Some(user) => {
            // Authenticated user - return personalized settings
            info!("Authenticated user: {}", user.pubkey);
        }
        None => {
            // Anonymous user - return global settings
            info!("Anonymous access");
        }
    }
}
```

## Request Headers

Authenticated requests must include:

1. **Authorization**: `Bearer <session_token>`
   - Session token obtained from `/auth/nostr` login endpoint

2. **X-Nostr-Pubkey**: `<user_pubkey>`
   - User's Nostr public key (hex format)

### Example Request

```bash
curl -X PUT http://localhost:4000/api/settings/physics \
  -H "Authorization: Bearer 550e8400-e29b-41d4-a716-446655440000" \
  -H "X-Nostr-Pubkey: a1b2c3d4e5f6..." \
  -H "Content-Type: application/json" \
  -d '{"gravity": -9.81, "timeStep": 0.016}'
```

## Endpoint Access Levels

### Read Endpoints (OptionalAuth)

Allow both authenticated and anonymous access:

- `GET /api/settings/all`
- `GET /api/settings/physics`
- `GET /api/settings/constraints`
- `GET /api/settings/rendering`
- `GET /api/settings/node-filter`
- `GET /api/settings/quality-gates`
- `GET /api/settings/profiles`
- `GET /api/settings/profiles/{id}`

**Behavior:**
- Anonymous users: Read-only access to global settings
- Authenticated users: Access to user-specific settings (future enhancement)

### Write Endpoints (AuthenticatedUser)

Require valid authentication:

- `PUT /api/settings/physics`
- `PUT /api/settings/constraints`
- `PUT /api/settings/rendering`
- `PUT /api/settings/node-filter`
- `PUT /api/settings/quality-gates`
- `POST /api/settings/profiles`
- `DELETE /api/settings/profiles/{id}`

**Behavior:**
- Returns 401 Unauthorized if no valid session
- Logs pubkey with all modifications

## Authentication Flow

### 1. Login

```typescript
// Client sends Nostr authentication event
const response = await fetch('/auth/nostr', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    id: eventId,
    pubkey: userPubkey,
    content: "Login to WebXR",
    sig: signature,
    created_at: timestamp,
    kind: 22242,
    tags: []
  })
});

const { user, token, expires_at } = await response.json();
// Store token and pubkey for subsequent requests
```

### 2. Authenticated Request

```typescript
// Use stored token and pubkey
const response = await fetch('/api/settings/physics', {
  method: 'PUT',
  headers: {
    'Authorization': `Bearer ${token}`,
    'X-Nostr-Pubkey': pubkey,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(settings)
});
```

### 3. Session Validation

The `AuthenticatedUser` extractor automatically:
1. Extracts token from Authorization header
2. Extracts pubkey from X-Nostr-Pubkey header
3. Validates session with NostrService
4. Returns user info or 401 error

### 4. Session Refresh

```typescript
// Refresh session before expiry
const response = await fetch('/auth/nostr/refresh', {
  method: 'POST',
  body: JSON.stringify({ pubkey, token })
});

const { token: newToken, expires_at } = await response.json();
// Update stored token
```

## Error Responses

### 401 Unauthorized

Returned when:
- Authorization header missing
- Token format invalid
- Session expired or invalid
- User not found

```json
{
  "error": "Invalid or expired session"
}
```

### 403 Forbidden

Returned when:
- Power user access required but user is not power user

```json
{
  "error": "This operation requires power user access"
}
```

## Power User Features

Power users have elevated privileges for:
- Global settings modifications (future)
- Administrative operations
- System configuration

Check power user status:
```rust
auth.require_power_user()?;  // Returns error if not power user
```

## Future Enhancements

### User-Specific Settings

Currently planned but not implemented:

1. **Per-User Settings Storage**
   - Store user preferences in Neo4j indexed by pubkey
   - Fall back to global settings if user settings not found

2. **Settings Inheritance**
   ```
   User Settings > Global Settings > Defaults
   ```

3. **User Filter Management**
   - Per-user node filters
   - Personal quality gate thresholds
   - Custom rendering preferences

### Implementation Pattern

```rust
pub async fn get_user_settings_or_global(
    pubkey: &str,
    settings_actor: &Addr<SettingsActor>,
) -> Result<AllSettings> {
    // Try user-specific settings first
    match settings_actor.send(GetUserSettings(pubkey)).await {
        Ok(Some(settings)) => Ok(settings),
        _ => {
            // Fall back to global settings
            settings_actor.send(GetAllSettings).await
        }
    }
}
```

## Security Considerations

### Session Management

- Tokens expire after configured duration (default: 3600 seconds)
- Sessions cleaned up automatically (hourly)
- Maximum session age: 24 hours

### Token Storage

Clients should:
- Store tokens securely (httpOnly cookies or secure storage)
- Never expose tokens in URLs or logs
- Clear tokens on logout

### Rate Limiting

Future enhancement:
- Per-user rate limits
- Anonymous access throttling
- Brute force protection

## Testing

### Manual Testing

```bash
# 1. Login
TOKEN=$(curl -X POST http://localhost:4000/auth/nostr \
  -H "Content-Type: application/json" \
  -d '{"id":"...","pubkey":"...","sig":"..."}' \
  | jq -r '.token')

PUBKEY="your_pubkey_here"

# 2. Test authenticated endpoint
curl -X PUT http://localhost:4000/api/settings/physics \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Nostr-Pubkey: $PUBKEY" \
  -H "Content-Type: application/json" \
  -d '{"gravity": -9.81}'

# 3. Test anonymous read
curl http://localhost:4000/api/settings/all

# 4. Test unauthorized write (should fail)
curl -X PUT http://localhost:4000/api/settings/physics \
  -H "Content-Type: application/json" \
  -d '{"gravity": -9.81}'
```

### Integration Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test;

    #[actix_rt::test]
    async fn test_authenticated_write_requires_token() {
        let app = test::init_service(/* ... */).await;

        // Without auth headers - should fail
        let req = test::TestRequest::put()
            .uri("/api/settings/physics")
            .set_json(&physics_settings)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 401);
    }

    #[actix_rt::test]
    async fn test_anonymous_read_allowed() {
        let app = test::init_service(/* ... */).await;

        let req = test::TestRequest::get()
            .uri("/api/settings/all")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }
}
```

## Client Integration

### React/TypeScript Example

```typescript
// auth.service.ts
export class AuthService {
  private token: string | null = null;
  private pubkey: string | null = null;

  async login(authEvent: NostrAuthEvent) {
    const response = await fetch('/auth/nostr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(authEvent)
    });

    const { token, user } = await response.json();
    this.token = token;
    this.pubkey = user.pubkey;

    // Store securely
    localStorage.setItem('auth_token', token);
    localStorage.setItem('pubkey', user.pubkey);
  }

  getAuthHeaders(): Record<string, string> {
    if (!this.token || !this.pubkey) {
      return {};
    }

    return {
      'Authorization': `Bearer ${this.token}`,
      'X-Nostr-Pubkey': this.pubkey
    };
  }

  async updateSettings(category: string, settings: any) {
    const response = await fetch(`/api/settings/${category}`, {
      method: 'PUT',
      headers: {
        ...this.getAuthHeaders(),
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(settings)
    });

    if (response.status === 401) {
      // Token expired - trigger re-login
      await this.refreshSession();
      // Retry request
      return this.updateSettings(category, settings);
    }

    return response.json();
  }
}
```

## Migration Guide

### For Existing Clients

No breaking changes for read operations:

```typescript
// Still works without authentication
const settings = await fetch('/api/settings/all').then(r => r.json());
```

Write operations now require authentication:

```typescript
// Before (no longer works)
await fetch('/api/settings/physics', {
  method: 'PUT',
  body: JSON.stringify(settings)
});

// After (requires auth)
await fetch('/api/settings/physics', {
  method: 'PUT',
  headers: {
    'Authorization': `Bearer ${token}`,
    'X-Nostr-Pubkey': pubkey,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(settings)
});
```

---

---

## Related Documentation

- [Ontology Sync Service Enhancement](ontology-sync-enhancement.md)
- [Deprecated Patterns Archive](../../archive/deprecated-patterns/README.md)
- [Architecture Overview (OBSOLETE - WRONG STACK)](../../archive/deprecated-patterns/03-architecture-WRONG-STACK.md)
- [User Settings Implementation Summary](../../archive/reports/2025-12-02-user-settings-summary.md)
- [Hexagonal Architecture Ports - Overview](../../explanations/architecture/ports/01-overview.md)

## Summary

The settings API now provides:

- Secure Nostr-based authentication for write operations
- Backward-compatible read-only anonymous access
- Session management with automatic cleanup
- Power user privilege system
- Foundation for user-specific settings

All write operations are logged with user pubkey for auditing and debugging.
