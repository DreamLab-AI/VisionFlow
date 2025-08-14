# Settings API Implementation - Complete REST Flow

## Overview
This document describes the complete REST API flow for settings management in the VisionFlow application, including the critical case conversion between client (camelCase) and server (snake_case).

## API Flow Architecture

### Complete Request/Response Flow
```
Client (React/TypeScript) → Nginx Proxy → Rust Backend → Settings Actor → YAML File
         ↓                      ↓              ↓              ↓            ↓
    camelCase JSON        Port 3001      Port 4000     snake_case    settings.yaml
```

### Key Components

#### 1. Client-Side (TypeScript/React)
- **settingsStore.ts**: Zustand store managing client state
- **apiService.ts**: Generic API service for HTTP requests  
- **settingsApi.ts**: Settings-specific API client
- **settings.ts**: TypeScript interfaces (camelCase)

#### 2. Proxy Layer
- **nginx.dev.conf**: Routes `/api/*` from port 3001 to Rust backend on port 4000

#### 3. Server-Side (Rust)
- **main.rs**: Initializes app with AppFullSettings
- **api_handler/mod.rs**: Routes API endpoints
- **settings_handler.rs**: REST endpoint handlers
- **settings_actor.rs**: Actor managing settings state
- **config/mod.rs**: AppFullSettings struct and conversion logic

#### 4. Persistence
- **settings.yaml**: Snake_case YAML configuration file

## Critical Case Conversion Logic

### The Problem
- Client uses **camelCase** (JavaScript convention)
- Server uses **snake_case** (Rust/YAML convention)
- Validation must handle both formats

### The Solution

#### 1. Conversion Functions (config/mod.rs)
```rust
// Convert JSON keys from camelCase to snake_case
fn keys_to_snake_case(value: Value) -> Value

// Convert JSON keys from snake_case to camelCase  
fn keys_to_camel_case(value: Value) -> Value
```

#### 2. Request Flow (Client → Server)
1. Client sends camelCase JSON: `{ "springStrength": 0.5 }`
2. Server receives in `update_settings` handler
3. `AppFullSettings::merge_update()` converts to snake_case
4. Validation now accepts BOTH formats (fixed)
5. Settings saved to YAML in snake_case

#### 3. Response Flow (Server → Client)
1. Server loads snake_case from AppFullSettings
2. `to_camel_case_json()` converts to camelCase
3. Client receives familiar camelCase JSON

## Fixed Issues

### 1. Validation Bug (CRITICAL)
**Problem**: Validation functions expected camelCase but received snake_case after conversion.

**Fix**: Modified all validation functions to check both formats:
```rust
// OLD (broken)
if let Some(spring) = physics.get("springStrength") { 

// NEW (fixed)
let spring = physics.get("springStrength")
    .or_else(|| physics.get("spring_strength"));
```

**Files Updated**:
- `/workspace/ext/src/handlers/settings_handler.rs`
  - `validate_physics_settings()` - Lines 228-341
  - `validate_node_settings()` - Lines 343-390
  - `validate_rendering_settings()` - Lines 392-402
  - `validate_xr_settings()` - Lines 404-413

### 2. Multi-Graph Support
The system supports multiple graph configurations:
- `logseq` graph settings
- `visionflow` graph settings

Each has independent physics, nodes, edges, and labels settings.

## API Endpoints

### GET /api/settings
- Returns complete settings in camelCase
- Source: AppFullSettings from YAML/memory

### POST /api/settings  
- Accepts partial updates in camelCase
- Validates all fields with proper ranges
- Merges updates into existing settings
- Saves to YAML if `persist_settings: true`
- Returns updated settings in camelCase

### POST /api/settings/reset
- Resets to default settings from YAML
- Returns defaults in camelCase

## Settings Structure

### Client (camelCase)
```typescript
interface Settings {
  visualisation: {
    graphs: {
      logseq: {
        physics: {
          springStrength: number;
          repulsionStrength: number;
          // ...
        }
      }
    }
  }
}
```

### Server (snake_case)
```rust
struct AppFullSettings {
  visualisation: VisualisationSettings {
    graphs: GraphsSettings {
      logseq: GraphSettings {
        physics: PhysicsSettings {
          spring_strength: f32,
          repulsion_strength: f32,
          // ...
        }
      }
    }
  }
}
```

## Testing Checklist

- [x] Client can GET settings (camelCase response)
- [x] Client can POST updates (camelCase request)
- [x] Validation accepts both camelCase and snake_case
- [x] Settings persist to YAML in snake_case
- [x] Physics updates propagate to GPU compute
- [x] Multi-graph support (logseq/visionflow)

## Implementation Status

✅ **COMPLETED**:
1. Analyzed existing client/server components
2. Examined Rust handlers and actors
3. Investigated case conversion logic
4. Verified YAML persistence
5. Identified validation bug
6. Fixed validation to accept both formats
7. Documented complete flow

## Future Improvements

1. Add comprehensive error messages for validation
2. Implement settings versioning/migration
3. Add settings import/export functionality
4. Create settings diff/merge UI
5. Add real-time settings sync across clients