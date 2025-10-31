# Settings Management Integration Guide

## Overview
This guide explains how to integrate the new persistent settings management system into the main application.

## Files Created

### Core Module Files
- `/src/settings/mod.rs` - Module entry point
- `/src/settings/models.rs` - Data models (ConstraintSettings, AllSettings, SettingsProfile)
- `/src/settings/settings_repository.rs` - Database persistence layer
- `/src/settings/settings_actor.rs` - Actix actor for runtime settings
- `/src/settings/api/settings_routes.rs` - REST API endpoints
- `/src/settings/api/mod.rs` - API module exports

### Database Migration
- `/src/migrations/006_settings_tables.sql` - SQLite tables for settings storage

### Tests
- `/tests/settings_integration_test.rs` - Comprehensive integration tests

## Integration Steps

### 1. Module Declaration (COMPLETED)
The settings module has been added to `/src/lib.rs`:
```rust
pub mod settings; // Persistent settings management (Week 8-9)
```

### 2. Initialize Settings Actor in main.rs

Add after AppState initialization (around line 320):

```rust
use webxr::settings::{SettingsRepository, SettingsActor};

// Initialize Settings Actor for persistent settings management
info!("[main] Initializing Settings Actor...");
let settings_repository = Arc::new(SettingsRepository::new(
    app_state.settings_repository.clone() // Assuming AppState has SQLite pool access
));
let settings_actor_addr = SettingsActor::new(settings_repository).start();
info!("[main] Settings Actor initialized");
```

### 3. Configure Routes in main.rs

Add to the HTTP server configuration (around line 480):

```rust
use webxr::settings::api::configure_routes as configure_settings_routes;

// Inside App::new()
.service(
    web::scope("/api")
        .configure(api_handler::config)
        .configure(workspace_handler::config)
        .configure(admin_sync_handler::configure_routes)
        .configure(admin_bridge_handler::configure_routes)
        .configure(configure_settings_routes) // NEW: Settings routes
        // ... rest of routes
)
```

### 4. Add Settings Actor to App Data

Add before `.service()` calls:

```rust
.app_data(web::Data::new(settings_actor_addr.clone()))
```

## API Endpoints

Once integrated, the following REST endpoints will be available:

### Physics Settings
- `GET /api/settings/physics` - Get current physics settings
- `PUT /api/settings/physics` - Update physics settings

### Constraint Settings
- `GET /api/settings/constraints` - Get current constraint settings
- `PUT /api/settings/constraints` - Update constraint settings

### Rendering Settings
- `GET /api/settings/rendering` - Get current rendering settings
- `PUT /api/settings/rendering` - Update rendering settings

### All Settings
- `GET /api/settings/all` - Get all current settings

### Profile Management
- `POST /api/settings/profiles` - Save current settings as named profile
- `GET /api/settings/profiles` - List all profiles
- `GET /api/settings/profiles/:id` - Load specific profile
- `DELETE /api/settings/profiles/:id` - Delete profile

## Example Usage

### Frontend Integration

```typescript
// Update physics settings
const response = await fetch('/api/settings/physics', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    damping: 0.85,
    springK: 0.01,
    repelK: 75.0,
    gravity: 0.0,
    massScale: 1.0
  })
});

// Save current settings as profile
const saveResponse = await fetch('/api/settings/profiles', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'My Custom Layout'
  })
});

const { id } = await saveResponse.json();

// Load profile later
const loadResponse = await fetch(`/api/settings/profiles/${id}`);
const settings = await loadResponse.json();
```

### Rust Usage

```rust
use actix::Addr;
use webxr::settings::{
    SettingsActor, UpdatePhysicsSettings, GetPhysicsSettings,
    SaveProfile, LoadProfile
};
use webxr::config::PhysicsSettings;

async fn update_settings(settings_actor: Addr<SettingsActor>) -> Result<()> {
    let mut physics = PhysicsSettings::default();
    physics.damping = 0.85;

    settings_actor
        .send(UpdatePhysicsSettings(physics))
        .await??;

    Ok(())
}

async fn save_and_load_profile(settings_actor: Addr<SettingsActor>) -> Result<()> {
    // Save current settings
    let profile_id = settings_actor
        .send(SaveProfile { name: "test".to_string() })
        .await??;

    // Load profile
    let settings = settings_actor
        .send(LoadProfile(profile_id))
        .await??;

    Ok(())
}
```

## Default Settings

### PhysicsSettings
Defined in `/src/config/mod.rs` - already exists with comprehensive defaults including:
- Force-directed layout parameters (damping, spring_k, repel_k)
- Boundary settings
- GPU compute parameters
- Auto-balance configuration

### ConstraintSettings (NEW)
Defined in `/src/settings/models.rs`:
```rust
ConstraintSettings {
    lod_enabled: true,
    far_threshold: 1000.0,    // Only priority 1-3 constraints
    medium_threshold: 100.0,  // Priority 1-5 constraints
    near_threshold: 10.0,     // All constraints
    priority_weighting: PriorityWeighting::Exponential,
    progressive_activation: true,
    activation_frames: 60,    // 1 second at 60 FPS
}
```

### RenderingSettings
Defined in `/src/config/mod.rs` - already exists with:
- Lighting parameters
- Post-processing effects
- Shadow settings

## Database Schema

The migration creates 4 tables:

1. **physics_settings** - Single row (id=1) for current physics settings
2. **constraint_settings** - Single row (id=1) for current constraint settings
3. **rendering_settings** - Single row (id=1) for current rendering settings
4. **settings_profiles** - Multiple rows for saved profiles

All settings are stored as JSON for flexibility and forward compatibility.

## Testing

Run integration tests:
```bash
cargo test --test settings_integration_test
```

Tests cover:
- Physics settings persistence
- Constraint settings persistence
- Profile save/load/delete
- Default settings on empty database
- All settings save/load

## Coordination with GPU Physics Actor

The SettingsActor is designed to notify the GpuPhysicsActor when physics settings change. To enable this:

1. Pass the GpuPhysicsActor address to SettingsActor during initialization
2. Modify the `UpdatePhysicsSettings` handler to send a notification message
3. Example:
```rust
// In UpdatePhysicsSettings handler
if let Some(gpu_physics_addr) = &self.gpu_physics_addr {
    gpu_physics_addr.do_send(PhysicsParamsChanged(settings.clone()));
}
```

## Migration Notes

This system is designed to work alongside the existing OptimizedSettingsActor and does not replace it. The two systems serve different purposes:

- **OptimizedSettingsActor**: In-memory caching and real-time access
- **SettingsActor**: Persistent storage and profile management

They can work together, with SettingsActor providing the persistence layer.

## Next Steps

1. Run `cargo check` to verify compilation
2. Add SettingsActor initialization to main.rs
3. Configure routes in HTTP server
4. Test endpoints with curl or frontend
5. Integrate with control center UI

## Troubleshooting

### Compilation Errors
- Ensure all imports are correct in main.rs
- Check that sqlx database URL is configured
- Verify migration has been applied

### Runtime Issues
- Check logs for actor initialization errors
- Verify database file permissions
- Ensure settings tables exist

### API Issues
- Verify routes are configured correctly
- Check actor address is passed to app_data
- Validate request JSON format matches structs
