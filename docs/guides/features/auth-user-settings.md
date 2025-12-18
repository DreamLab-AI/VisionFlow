---
title: Per-User Settings Implementation
description: Implemented server-side authentication middleware and per-user settings lookup for VisionFlow.
category: howto
tags:
  - guide
  - neo4j
  - rust
updated-date: 2025-12-18
difficulty-level: advanced
---


# Per-User Settings Implementation

## Overview
Implemented server-side authentication middleware and per-user settings lookup for VisionFlow.

## Architecture

### Authentication Flow
1. Client sends `Authorization: Bearer {token}` header
2. `AuthenticatedUser` extractor validates session via NostrService
3. Routes receive authenticated user or return 401 Unauthorized

### Components

#### Auth Extractor (`src/settings/auth_extractor.rs`)
- `AuthenticatedUser` - Required authentication extractor
- `OptionalAuth` - Optional authentication (allows anonymous)
- Validates sessions using NostrService
- Extracts pubkey and power_user status

#### Neo4j Repository (`src/adapters/neo4j_settings_repository.rs`)
- `User` node - Nostr pubkey-based user identity
- `UserSettings` node - Full AppFullSettings per user
- `UserFilter` node - Graph filter preferences per user
- Methods:
  - `get_user_settings(&pubkey)` - Retrieve user's personal settings
  - `save_user_settings(&pubkey, settings)` - Save user's settings
  - `get_user_filter(&pubkey)` - Retrieve user's filter
  - `save_user_filter(&pubkey, filter)` - Save user's filter

### API Endpoints

#### Settings Endpoints (with auth)
- `GET /api/settings/all` - Returns user settings if authenticated, global otherwise
- `PUT /api/settings/physics` - Requires authentication
- `PUT /api/settings/constraints` - Requires authentication
- `PUT /api/settings/rendering` - Requires authentication
- `PUT /api/settings/node-filter` - Requires authentication
- `PUT /api/settings/quality-gates` - Requires authentication

#### User Filter Endpoints (authentication required)
- `GET /api/settings/user/filter` - Get user's personal filter settings
- `PUT /api/settings/user/filter` - Update user's filter settings

### UserFilter Schema
```rust
pub struct UserFilter {
    pub pubkey: String,
    pub enabled: bool,
    pub quality_threshold: f64,
    pub authority_threshold: f64,
    pub filter_by_quality: bool,
    pub filter_by_authority: bool,
    pub filter_mode: String,  // "and" | "or"
    pub max_nodes: Option<i32>,
    pub updated_at: DateTime<Utc>,
}
```

### Backward Compatibility
- Anonymous users can read global settings (read-only)
- Authenticated users get their personal settings if available
- Falls back to global settings if no user settings exist
- All existing functionality preserved

### Session Validation
Sessions are validated through NostrService:
1. Extract `Authorization: Bearer {token}` header
2. Extract `X-Nostr-Pubkey` header
3. Call `nostr_service.validate_session(&pubkey, &token)`
4. Check expiry against `AUTH_TOKEN_EXPIRY` env var (default 3600s)

### Neo4j Schema
```cypher
// User node
CREATE (u:User {
  pubkey: string,
  is_power_user: boolean,
  created_at: datetime,
  last_seen: datetime,
  display_name: string?
})

// User settings
CREATE (u:User)-[:HAS_SETTINGS]->(us:UserSettings {
  pubkey: string,
  settings_json: string,  // JSON-serialized AppFullSettings
  updated_at: datetime
})

// User filter
CREATE (u:User)-[:HAS_FILTER]->(uf:UserFilter {
  pubkey: string,
  enabled: boolean,
  quality_threshold: float,
  authority_threshold: float,
  filter_by_quality: boolean,
  filter_by_authority: boolean,
  filter_mode: string,
  max_nodes: integer?,
  updated_at: datetime
})
```

## Client Integration

### Example: Fetch User Settings
```typescript
const token = getAuthToken();
const pubkey = getUserPubkey();

const response = await fetch('/api/settings/all', {
  headers: {
    'Authorization': `Bearer ${token}`,
    'X-Nostr-Pubkey': pubkey
  }
});

const settings = await response.json();
// Returns user-specific settings if authenticated
// Falls back to global settings if not found
```

### Example: Update User Filter
```typescript
const filter = {
  enabled: true,
  quality_threshold: 0.8,
  authority_threshold: 0.6,
  filter_by_quality: true,
  filter_by_authority: false,
  filter_mode: 'or',
  max_nodes: 5000
};

const response = await fetch('/api/settings/user/filter', {
  method: 'PUT',
  headers: {
    'Authorization': `Bearer ${token}`,
    'X-Nostr-Pubkey': pubkey,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(filter)
});
```

## Testing
```bash
# Compile check
cargo check

# Run tests (requires Neo4j instance)
cargo test neo4j_settings_repository -- --ignored

# Test filter endpoints
cargo test user_filter -- --ignored
```

## Environment Variables
- `NEO4J_URI` - Neo4j connection URI (default: bolt://localhost:7687)
- `NEO4J_USER` - Neo4j username (default: neo4j)
- `NEO4J_PASSWORD` - Neo4j password (default: password)
- `AUTH_TOKEN_EXPIRY` - Token expiry in seconds (default: 3600)
- `POWER_USER_PUBKEYS` - Comma-separated list of power user pubkeys

## Files Modified
- `/src/settings/api/settings_routes.rs` - Added filter endpoints, per-user settings lookup
- `/src/main.rs` - Added Neo4jSettingsRepository to app data
- `/src/adapters/neo4j_settings_repository.rs` - Already had user support (no changes needed)
- `/src/settings/auth_extractor.rs` - Already existed (no changes needed)

---

---

## Related Documentation

- [Client-Side Filtering Implementation](filtering-nodes.md)
- [GraphServiceActor Migration Guide (HISTORICAL REFERENCE)](../graphserviceactor-migration.md)
- [Ontology Storage Guide](../ontology-storage-guide.md)
- [Docker Compose Unified Configuration - Usage Guide](../docker-compose-guide.md)
- [Developer Guides](../developer/readme.md)

## Future Enhancements
- Add endpoint to copy global settings to user settings
- Add endpoint to reset user settings to defaults
- Add bulk user management endpoints for power users
- Add settings versioning and migration support
