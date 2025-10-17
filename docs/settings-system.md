# Settings System Architecture

## Overview

The VisionFlow settings system is a high-performance, SQLite-backed configuration management system designed for multi-user environments with fine-grained permission control. The system migrated from YAML-based file storage to a database-backed approach to enable real-time updates, user-specific overrides, and comprehensive audit logging.

## Key Features

### 1. SQLite-Backed Storage
- All settings stored in SQLite database
- ACID guarantees for configuration changes
- High-performance query optimization with strategic indexes
- Support for concurrent access with proper locking

### 2. Multi-User Support
- Global default settings
- Per-user setting overrides
- User-specific configuration isolation
- Settings inheritance (user settings override global defaults)

### 3. Permission System
- Power user role for settings modification
- Nostr-based authentication integration
- Per-user permission grants/revokes
- Audit logging for all permission changes

### 4. Dual Key Format
The system supports both `camelCase` (frontend/API) and `snake_case` (backend/database) naming conventions:

```
Frontend/API:    { "ambientLightIntensity": 0.5 }
Backend/DB:      { "ambient_light_intensity": 0.5 }
```

Automatic conversion ensures compatibility across the entire stack.

### 5. Real-Time Updates
- WebSocket-based settings broadcast
- Immediate propagation to all connected clients
- Subscription model for settings changes
- Automatic physics engine parameter updates

### 6. Comprehensive Validation
- Type checking for all settings
- Range validation for numeric values
- Format validation for strings (colors, URLs, domains)
- Cross-field validation for complex constraints
- GPU parameter safety checks

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                     │
│  (Browser, WebSocket clients, API consumers)                │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/WebSocket
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (Actix-Web)                    │
│  • GET/PUT /api/settings - Global settings CRUD             │
│  • GET/PUT /api/settings/user/:pubkey - User overrides      │
│  • POST /api/settings/validate - Validation endpoint        │
│  • WebSocket /ws - Real-time settings subscription          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              Settings Handler & Validation Layer             │
│  • SettingsHandler - Request routing & responses            │
│  • ValidationService - Comprehensive validation             │
│  • Field name normalization (camelCase ↔ snake_case)       │
│  • Permission checks & audit logging                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  Actor System (Message Passing)              │
│  • OptimizedSettingsActor - Settings state management       │
│  • ProtectedSettingsActor - Permission management           │
│  • PhysicsOrchestratorActor - Physics parameter updates     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   SQLite Database Layer                      │
│  Tables:                                                     │
│  • settings - Global default settings                        │
│  • user_settings - User-specific overrides                   │
│  • users - User accounts & authentication                    │
│  • physics_settings - Physics engine configuration           │
│  • settings_audit_log - Change tracking                      │
└─────────────────────────────────────────────────────────────┘
```

## Migration from YAML

### Why We Migrated

**Previous System (YAML Files):**
- Settings stored in `/app/settings.yaml` and `/app/user_settings/{pubkey}.yaml`
- File I/O for every settings change
- No transactional guarantees
- Limited query capabilities
- No built-in audit logging
- Race conditions with concurrent access

**Current System (SQLite Database):**
- Single source of truth in `settings.db`
- Transactional updates with ACID guarantees
- Fast indexed queries
- Built-in audit logging
- Concurrent access with proper locking
- User-specific overrides with inheritance

### Migration Path

The system performs automatic migration on first startup:

1. Check for existence of `/app/settings.yaml`
2. If found, parse YAML and import to SQLite
3. Create default user settings if needed
4. Preserve backward compatibility during transition

## Settings Hierarchy

Settings are resolved using the following priority order:

```
1. User-specific override (highest priority)
   ↓
2. Global default setting
   ↓
3. Hardcoded fallback (lowest priority)
```

### Example:

```json
// Global default
{
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 0.5
    }
  }
}

// User "pubkey123" override
{
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 0.8
    }
  }
}

// Result for user "pubkey123"
// ambientLightIntensity = 0.8 (user override)

// Result for other users
// ambientLightIntensity = 0.5 (global default)
```

## Configuration Categories

### Visualisation Settings
- **Rendering**: Lights, shadows, post-processing
- **Animations**: Motion blur, node animations, effects
- **Glow/Bloom**: GPU-accelerated visual effects
- **Hologram**: Geometric overlays and decorations
- **Graphs**: Per-graph-type configurations (Logseq, VisionFlow, Ontology)
- **Camera**: FOV, position, clipping planes

### Physics Settings (Per-Graph Type)
- **Forces**: Spring, repulsion, gravity, damping
- **Boundaries**: Spatial constraints, collision handling
- **Auto-balance**: Adaptive parameter tuning
- **Auto-pause**: Energy-based simulation control
- **Clustering**: Community detection algorithms
- **Performance**: Iterations, timestep, compute mode

### System Settings
- **Network**: Port, domain, TLS, rate limiting
- **WebSocket**: Update rates, compression, heartbeat
- **Security**: CORS, authentication, session management
- **Debug**: Logging, profiling, diagnostics

### Integration Settings
- **XR**: WebXR configuration for VR/AR
- **Auth**: Nostr authentication provider
- **RagFlow**: AI chat integration
- **Perplexity**: AI search integration
- **OpenAI**: LLM API configuration
- **Kokoro**: TTS service settings
- **Whisper**: STT service settings

## Performance Optimizations

### 1. Connection Pooling
- Shared SQLite connection pool
- Configurable pool size
- Automatic connection recycling

### 2. Caching Strategy
- In-memory cache for frequently accessed settings
- Cache invalidation on updates
- TTL-based expiration

### 3. Indexed Queries
```sql
CREATE INDEX idx_settings_category ON settings(category);
CREATE INDEX idx_settings_key ON settings(key);
CREATE INDEX idx_user_settings_user_key ON user_settings(user_id, key);
CREATE INDEX idx_settings_audit_user ON settings_audit_log(user_id);
CREATE INDEX idx_settings_audit_timestamp ON settings_audit_log(timestamp);
```

### 4. Batch Operations
- Bulk insert/update support
- Transaction batching for related changes
- Reduced round-trips to database

## Security Features

### 1. Permission Model
- Only "power users" can modify settings
- Permission grant tracked in `users` table
- Audit log for permission changes

### 2. Nostr Authentication
- Public key based identity
- Cryptographic signature verification
- No password storage

### 3. Audit Logging
Every settings change records:
- User ID (Nostr pubkey)
- Timestamp
- Setting key modified
- Old value
- New value
- Change reason (optional)

### 4. Input Validation
- SQL injection prevention (parameterized queries)
- Type validation
- Range checks
- Format validation (regex patterns)

## Error Handling

### Validation Errors
```json
{
  "error": "ValidationError",
  "field": "visualisation.rendering.ambientLightIntensity",
  "message": "Value must be between 0.0 and 10.0",
  "receivedValue": 15.0
}
```

### Permission Errors
```json
{
  "error": "PermissionDenied",
  "message": "User does not have power user permissions",
  "requiredPermission": "power_user"
}
```

### Database Errors
- Automatic retry on lock contention
- Graceful degradation on connection failure
- Fallback to cached values when possible

## Best Practices

### For Developers

1. **Always validate settings before persisting**
   ```rust
   let validation_result = ValidationService::validate_settings(&settings)?;
   ```

2. **Use transactions for related changes**
   ```rust
   conn.execute("BEGIN TRANSACTION")?;
   // Multiple updates
   conn.execute("COMMIT")?;
   ```

3. **Prefer batch operations**
   ```rust
   update_settings_batch(&updates)?;
   ```

4. **Check permissions before modification**
   ```rust
   if !user.is_power_user() {
       return Err(PermissionError::NotPowerUser);
   }
   ```

### For Operators

1. **Regular database backups**
   ```bash
   sqlite3 /app/data/settings.db ".backup /backup/settings-$(date +%Y%m%d).db"
   ```

2. **Monitor audit log for suspicious activity**
   ```sql
   SELECT * FROM settings_audit_log
   WHERE timestamp > datetime('now', '-1 hour')
   ORDER BY timestamp DESC;
   ```

3. **Periodically vacuum database**
   ```bash
   sqlite3 /app/data/settings.db "VACUUM;"
   ```

## Troubleshooting

### Settings not updating
1. Check WebSocket connection
2. Verify user permissions
3. Check validation errors in logs
4. Ensure database is not locked

### Performance issues
1. Check database size (vacuum if needed)
2. Verify index usage (EXPLAIN QUERY PLAN)
3. Monitor connection pool exhaustion
4. Review cache hit rates

### Data corruption
1. Restore from backup
2. Run integrity check: `PRAGMA integrity_check;`
3. Rebuild indexes if needed

## Related Documentation

- [Settings API Reference](./settings-api.md)
- [Validation Rules](./settings-validation.md)
- [Database Schema](./settings-schema.md)
- [User Permissions](./user-permissions.md)
- [Migration Guide](./settings-migration-guide.md)
