---
layout: default
title: "Neo4j User Settings Schema"
parent: Database
grand_parent: Reference
nav_order: 99
---

# Neo4j User Settings Schema

## Overview

VisionFlow's Neo4j settings repository has been extended with per-user settings support, enabling:
- User-specific settings storage (Nostr pubkey-based authentication)
- Individual graph filter preferences per user
- Power user access controls
- Persistent user session tracking

## Schema Components

### 1. User Node

**Label**: `:User`

**Properties**:
```cypher
(:User {
  pubkey: String,           // Nostr public key (unique identifier)
  is_power_user: Boolean,   // Can access debug settings, write global settings
  created_at: DateTime,     // User registration timestamp
  last_seen: DateTime,      // Last activity timestamp
  display_name: String?     // Optional NIP-05 or custom name
})
```

**Constraints**:
```cypher
CREATE CONSTRAINT user_pubkey_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.pubkey IS UNIQUE;
```

### 2. UserSettings Node

**Label**: `:UserSettings`

**Properties**:
```cypher
(:UserSettings {
  pubkey: String,           // Links to User (indexed)
  settings_json: String,    // Full AppFullSettings object as JSON
  updated_at: DateTime      // Last update timestamp
})
```

**Indices**:
```cypher
CREATE INDEX user_settings_pubkey_idx IF NOT EXISTS
FOR (us:UserSettings) ON (us.pubkey);
```

### 3. UserFilter Node

**Label**: `:UserFilter`

**Properties**:
```cypher
(:UserFilter {
  pubkey: String,              // Links to User (indexed)
  enabled: Boolean,            // Enable/disable filtering
  quality_threshold: Float,    // Min quality score (0.0-1.0), default 0.7
  authority_threshold: Float,  // Min authority score (0.0-1.0), default 0.5
  filter_by_quality: Boolean,  // Use quality for filtering
  filter_by_authority: Boolean,// Use authority for filtering
  filter_mode: String,         // 'or' | 'and' - how to combine filters
  max_nodes: Integer?,         // Optional max nodes limit, default 10000
  updated_at: DateTime         // Last update timestamp
})
```

**Indices**:
```cypher
CREATE INDEX user_filter_pubkey_idx IF NOT EXISTS
FOR (uf:UserFilter) ON (uf.pubkey);
```

### 4. Relationships

```cypher
(:User)-[:HAS_SETTINGS]->(:UserSettings)
(:User)-[:HAS_FILTER]->(:UserFilter)
```

## API Methods

### User Management

#### `get_or_create_user(pubkey: &str) -> Result<User>`
Creates or retrieves a user by Nostr pubkey. Automatically updates `last_seen` timestamp.

**Cypher**:
```cypher
MERGE (u:User {pubkey: $pubkey})
ON CREATE SET
  u.is_power_user = false,
  u.created_at = datetime(),
  u.last_seen = datetime()
ON MATCH SET
  u.last_seen = datetime()
RETURN u
```

#### `update_user_last_seen(pubkey: &str) -> Result<()>`
Updates the user's last seen timestamp.

#### `is_power_user(pubkey: &str) -> Result<bool>`
Checks if user has power user privileges.

#### `set_power_user(pubkey: &str, is_power: bool) -> Result<()>`
Grants or revokes power user status.

### User Settings

#### `get_user_settings(pubkey: &str) -> Result<Option<AppFullSettings>>`
Retrieves the user's complete settings object.

**Cypher**:
```cypher
MATCH (u:User {pubkey: $pubkey})-[:HAS_SETTINGS]->(us:UserSettings)
RETURN us.settings_json
```

#### `save_user_settings(pubkey: &str, settings: &AppFullSettings) -> Result<()>`
Saves the user's complete settings object as JSON.

**Cypher**:
```cypher
MATCH (u:User {pubkey: $pubkey})
MERGE (u)-[:HAS_SETTINGS]->(us:UserSettings {pubkey: $pubkey})
ON CREATE SET
  us.settings_json = $settings_json,
  us.updated_at = datetime()
ON MATCH SET
  us.settings_json = $settings_json,
  us.updated_at = datetime()
```

### User Filters

#### `get_user_filter(pubkey: &str) -> Result<Option<UserFilter>>`
Retrieves the user's graph filter preferences.

**Returns default values if not found**:
- `enabled: true`
- `quality_threshold: 0.7`
- `authority_threshold: 0.5`
- `filter_by_quality: true`
- `filter_by_authority: false`
- `filter_mode: "or"`
- `max_nodes: 10000`

#### `save_user_filter(pubkey: &str, filter: &UserFilter) -> Result<()>`
Saves the user's filter preferences.

## Rust Structs

### User
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub pubkey: String,
    pub is_power_user: bool,
    pub created_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub display_name: Option<String>,
}
```

### UserFilter
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFilter {
    pub pubkey: String,
    pub enabled: bool,
    pub quality_threshold: f64,
    pub authority_threshold: f64,
    pub filter_by_quality: bool,
    pub filter_by_authority: bool,
    pub filter_mode: String,
    pub max_nodes: Option<i32>,
    pub updated_at: DateTime<Utc>,
}
```

### UserSettingsNode
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettingsNode {
    pub pubkey: String,
    pub settings_json: String,
    pub updated_at: DateTime<Utc>,
}
```

## Usage Examples

### Create User and Save Settings

```rust
use crate::adapters::neo4j_settings_repository::{Neo4jSettingsRepository, Neo4jSettingsConfig};
use crate::config::AppFullSettings;

let config = Neo4jSettingsConfig::default();
let repo = Neo4jSettingsRepository::new(config).await?;

// Get or create user
let user = repo.get_or_create_user("user_pubkey_123").await?;

// Save user's settings
let settings = AppFullSettings::default();
repo.save_user_settings(&user.pubkey, &settings).await?;

// Retrieve user's settings
let loaded = repo.get_user_settings(&user.pubkey).await?;
```

### Configure User Filters

```rust
use crate::adapters::neo4j_settings_repository::UserFilter;

// Create user first
repo.get_or_create_user("user_pubkey_123").await?;

// Configure filter
let filter = UserFilter {
    pubkey: "user_pubkey_123".to_string(),
    enabled: true,
    quality_threshold: 0.8,
    authority_threshold: 0.6,
    filter_by_quality: true,
    filter_by_authority: true,
    filter_mode: "and".to_string(),
    max_nodes: Some(5000),
    updated_at: Utc::now(),
};

repo.save_user_filter("user_pubkey_123", &filter).await?;

// Retrieve filter
let loaded_filter = repo.get_user_filter("user_pubkey_123").await?;
```

### Power User Management

```rust
// Grant power user status
repo.set_power_user("user_pubkey_123", true).await?;

// Check power user status
if repo.is_power_user("user_pubkey_123").await? {
    // Allow access to debug settings
}
```

## Testing

Tests are included in `/home/devuser/workspace/project/src/adapters/neo4j_settings_repository.rs`:

```bash
# Run tests (requires Neo4j instance)
cargo test --lib neo4j_settings_repository -- --ignored

# Individual tests
cargo test --lib test_user_management -- --ignored
cargo test --lib test_user_settings -- --ignored
cargo test --lib test_user_filter -- --ignored
```

## Schema Migration

The schema is automatically initialized when `Neo4jSettingsRepository::new()` is called. The `initialize_schema()` method creates:

1. Constraints for unique user pubkeys
2. Indices for fast lookups on pubkey fields
3. Root settings node (existing functionality)

No manual migration is required. The schema is backward-compatible with existing settings.

## Performance Considerations

- **Indexed Lookups**: All user lookups use indexed `pubkey` fields
- **JSON Storage**: Full settings stored as JSON for flexibility
- **Caching**: User settings should be cached at application layer
- **Batch Operations**: Not yet implemented for user operations

## Security Notes

1. **Power Users**: Only power users should be able to write global settings
2. **Pubkey Validation**: Validate Nostr pubkeys before storage
3. **Settings Validation**: Validate settings before saving
4. **Rate Limiting**: Implement rate limits on settings updates

## Future Enhancements

- [ ] Batch user operations
- [ ] User settings history/versioning
- [ ] Settings inheritance (global -> user overrides)
- [ ] User groups and shared settings
- [ ] Settings export/import per user
- [ ] Audit log for settings changes
