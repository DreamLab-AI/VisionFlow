# Multi-User Settings Integration Guide

This document explains how to integrate the newly implemented multi-user settings system with Nostr authentication into the main application.

## Overview

The following components have been implemented:

1. **Database Schema** (`schema/ontology_db.sql`):
   - `users` table with Nostr pubkey authentication
   - `user_settings` table for per-user setting overrides
   - `settings_audit_log` for tracking all setting changes

2. **UserService** (`src/services/user_service.rs`):
   - User CRUD operations
   - Per-user settings management
   - Power user permission management
   - Audit log queries

3. **Permission Middleware** (`src/middleware/permissions.rs`):
   - Extract and validate Nostr authentication
   - Inject AuthContext into requests
   - Enforce power user restrictions

4. **Admin Handler** (`src/handlers/admin_handler.rs`):
   - GET `/api/admin/users` - List all users
   - GET `/api/admin/users/{pubkey}` - Get user by pubkey
   - PUT `/api/admin/users/{pubkey}/power-user` - Grant power user
   - DELETE `/api/admin/users/{pubkey}/power-user` - Revoke power user
   - GET `/api/admin/settings/audit` - View audit log

5. **User Settings Handler** (`src/handlers/user_settings_handler.rs`):
   - GET `/api/user-settings` - Get current user's settings
   - POST `/api/user-settings` - Set user setting (power users only)
   - DELETE `/api/user-settings/{key}` - Delete user setting (power users only)

## Integration Steps

### Step 1: Initialize UserService in main.rs

Add this after initializing NostrService (around line 274):

```rust
#[cfg(feature = "ontology")]
use webxr::services::user_service::UserService;

// Initialize UserService with the ontology database
#[cfg(feature = "ontology")]
let user_service = {
    let db_path = std::env::var("ONTOLOGY_DB_PATH")
        .unwrap_or_else(|_| "/app/data/ontology.db".to_string());

    match UserService::new(db_path).await {
        Ok(service) => {
            info!("UserService initialized successfully");
            Some(web::Data::new(service))
        }
        Err(e) => {
            error!("Failed to initialize UserService: {}", e);
            None
        }
    }
};

#[cfg(not(feature = "ontology"))]
let user_service = None;
```

### Step 2: Register Services in HttpServer

Add UserService to app_data in the HttpServer configuration (around line 568):

```rust
let app = App::new()
    .wrap(middleware::Logger::default())
    .wrap(cors)
    .wrap(middleware::Compress::default())
    // ... existing app_data ...
    .app_data(app_state_data.nostr_service.clone().unwrap_or_else(|| web::Data::new(NostrService::default())))
    .app_data(app_state_data.feature_access.clone());

// Add UserService if ontology feature is enabled
#[cfg(feature = "ontology")]
if let Some(ref user_svc) = user_service {
    app = app.app_data(user_svc.clone());
}
```

### Step 3: Register Admin Routes with Middleware

Add admin routes protected by power user middleware:

```rust
use webxr::handlers::admin_handler;
use webxr::handlers::user_settings_handler;
use webxr::middleware::permissions::PermissionMiddleware;
use std::rc::Rc;

// In the HttpServer App::new closure, add:

#[cfg(feature = "ontology")]
if let (Some(ref nostr_svc), Some(ref user_svc)) = (&app_state_data.nostr_service, &user_service) {
    app = app.service(
        web::scope("/api/admin")
            .wrap(PermissionMiddleware::power_user(
                Rc::new((**nostr_svc).clone()),
                Rc::new((**user_svc).clone()),
            ))
            .configure(admin_handler::configure_routes)
    );

    app = app.service(
        web::scope("/api/user-settings")
            .wrap(PermissionMiddleware::authenticated(
                Rc::new((**nostr_svc).clone()),
                Rc::new((**user_svc).clone()),
            ))
            .configure(user_settings_handler::configure_routes)
    );
}
```

### Step 4: Update NostrService Integration

The NostrService should create users in the database when they authenticate. Update the `nostr_handler::init_nostr_service` function:

```rust
// In nostr_handler.rs, after verifying auth event:

#[cfg(feature = "ontology")]
if let Some(user_service) = app_state.user_service.as_ref() {
    // Create or update user in database
    match user_service.create_or_update_user(&pubkey, username).await {
        Ok(user) => {
            info!("User registered in database: user_id={}, pubkey={}", user.id, user.nostr_pubkey);
        }
        Err(e) => {
            warn!("Failed to register user in database: {:?}", e);
        }
    }
}
```

### Step 5: Environment Variables

Add these environment variables to your `.env` file:

```bash
# Ontology Database Path
ONTOLOGY_DB_PATH=/app/data/ontology.db

# Power User Pubkeys (comma-separated)
POWER_USER_PUBKEYS=your_nostr_pubkey_here

# Authentication Token Expiry (seconds)
AUTH_TOKEN_EXPIRY=3600
```

### Step 6: Database Initialization

The database schema will be automatically created when the ontology system initializes. The user tables are now part of `schema/ontology_db.sql` with schema version 2.

## API Usage Examples

### Admin Operations (Power Users Only)

```bash
# List all users
curl -H "X-Nostr-Pubkey: $PUBKEY" \
     -H "X-Nostr-Token: $TOKEN" \
     http://localhost:8080/api/admin/users

# Grant power user status
curl -X PUT \
     -H "X-Nostr-Pubkey: $ADMIN_PUBKEY" \
     -H "X-Nostr-Token: $ADMIN_TOKEN" \
     http://localhost:8080/api/admin/users/$TARGET_PUBKEY/power-user

# View audit log
curl -H "X-Nostr-Pubkey: $ADMIN_PUBKEY" \
     -H "X-Nostr-Token: $ADMIN_TOKEN" \
     "http://localhost:8080/api/admin/settings/audit?limit=100"
```

### User Settings Operations

```bash
# Get user's settings
curl -H "X-Nostr-Pubkey: $PUBKEY" \
     -H "X-Nostr-Token: $TOKEN" \
     http://localhost:8080/api/user-settings

# Set a user setting (power users only)
curl -X POST \
     -H "X-Nostr-Pubkey: $POWER_USER_PUBKEY" \
     -H "X-Nostr-Token: $POWER_USER_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"key": "theme", "value": "dark"}' \
     http://localhost:8080/api/user-settings

# Delete a user setting (power users only)
curl -X DELETE \
     -H "X-Nostr-Pubkey: $POWER_USER_PUBKEY" \
     -H "X-Nostr-Token: $POWER_USER_TOKEN" \
     http://localhost:8080/api/user-settings/theme
```

## Architecture

### Authentication Flow

1. User authenticates via Nostr (existing system)
2. NostrService validates signature and creates session
3. UserService creates/updates user record in database
4. PermissionMiddleware extracts auth context on each request
5. AuthContext (user_id, pubkey, is_power_user) injected into request

### Settings Hierarchy

1. **Global Settings** (from settings.yaml/database)
   - Default values for all users
   - Modified by admins through existing settings API

2. **User Settings** (from user_settings table)
   - Per-user overrides of specific settings
   - Only power users can modify their settings
   - Regular users read-only access to their settings

3. **Audit Trail**
   - All setting changes logged to settings_audit_log
   - Includes old_value, new_value, timestamp, user_id
   - Power users can query audit log

### Permission Model

- **Authenticated Users**: Can view their own settings
- **Power Users**: Can modify settings and perform admin operations
- **Power User Grant/Revoke**: Only existing power users can grant/revoke

## Testing

After integration, test the following:

1. User registration on Nostr authentication
2. Permission middleware authentication checks
3. Admin API endpoints (power user only)
4. User settings CRUD operations
5. Audit log tracking
6. Database schema version upgrade

## Troubleshooting

### Issue: "User not found" errors

**Solution**: Ensure NostrService integration creates users in database on authentication.

### Issue: Permission denied errors

**Solution**: Check POWER_USER_PUBKEYS environment variable and verify user exists in database.

### Issue: Database errors

**Solution**: Verify ONTOLOGY_DB_PATH points to writable location and schema is initialized.

### Issue: Middleware not working

**Solution**: Ensure middleware is wrapped around routes in correct order (auth before handler).

## Next Steps

After integration:

1. Add per-user default settings on registration
2. Implement settings sync across devices for same user
3. Add bulk user management tools
4. Create admin dashboard UI
5. Add user activity analytics

## Files Modified/Created

### Created
- `/home/devuser/workspace/project/src/services/user_service.rs`
- `/home/devuser/workspace/project/src/middleware/permissions.rs`
- `/home/devuser/workspace/project/src/middleware/mod.rs`
- `/home/devuser/workspace/project/src/handlers/admin_handler.rs`
- `/home/devuser/workspace/project/src/handlers/user_settings_handler.rs`

### Modified
- `/home/devuser/workspace/project/schema/ontology_db.sql` - Added user tables
- `/home/devuser/workspace/project/src/services/mod.rs` - Added user_service
- `/home/devuser/workspace/project/src/handlers/mod.rs` - Added admin and user settings handlers
- `/home/devuser/workspace/project/src/lib.rs` - Added middleware module

### To Modify
- `/home/devuser/workspace/project/src/main.rs` - Integration steps above
- `/home/devuser/workspace/project/src/handlers/nostr_handler.rs` - User DB sync
- `/home/devuser/workspace/project/src/app_state.rs` - Add user_service field (optional)
