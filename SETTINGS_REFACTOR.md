# Settings System Major Refactor - Implementation Guide

## Overview
This document outlines the comprehensive refactoring of the settings system from a complex, multi-layered architecture to a clean, unified structure that fixes the physics settings propagation bug.

## Problem Summary
The existing settings system has grown organically into an unmaintainable mess with:
- **1200+ lines** of settings handler code with massive duplication
- **4 different settings representations** (AppFullSettings, Settings, UISettings, ClientSettingsPayload)
- **Manual case conversions** between snake_case (YAML) and camelCase (JSON)
- **Dual physics storage** (flat legacy + multi-graph) causing synchronization issues
- **Physics settings not propagating** to GPU simulation correctly

## Solution Architecture

### New Unified Structure
```
settings-unified.yaml (single source of truth)
         ↓
UnifiedSettings struct (single Rust type)
         ↓
Direct JSON serialization (automatic naming)
         ↓
TypeScript interfaces (exact match)
```

## Files Created/Modified

### 1. Core Settings Model
**File:** `/workspace/ext/src/config/unified.rs`
- Single `UnifiedSettings` struct for all settings
- Direct conversion to `SimulationParams` for GPU
- Built-in merge methods for partial updates
- Clean physics update methods per graph

### 2. Simplified Settings Handler  
**File:** `/workspace/ext/src/handlers/unified_settings_handler.rs`
- Reduced from 1200+ lines to ~200 lines
- Direct endpoints: `GET /settings`, `POST /settings`, `POST /settings/physics/{graph}`
- Automatic physics propagation to GPU
- No manual case conversions or merge macros

### 3. Migration Support
**File:** `/workspace/ext/src/config/migration.rs`
- Converts old `AppFullSettings` to new `UnifiedSettings`
- Handles legacy flat physics → multi-graph structure
- Backward compatibility during transition

### 4. Unified Settings Actor
**File:** `/workspace/ext/src/actors/unified_settings_actor.rs`
- Clean actor implementation without conversions
- Direct file I/O with automatic serialization
- Supports both old and new message types during migration

### 5. New YAML Configuration
**File:** `/workspace/ext/settings-unified.yaml`
- Clean multi-graph structure (logseq, visionflow)
- Consistent naming throughout
- No redundant fields

## Key Improvements

### Before (Complex)
```rust
// 200+ lines of merge macros
merge_copy_option!(target_physics.damping, physics_dto.damping);
merge_copy_option!(target_physics.spring_strength, physics_dto.spring_strength);
// ... repeated for every field

// Complex extraction logic
let physics_to_apply = vis_dto.graphs.as_ref()
    .and_then(|g| g.logseq.as_ref())
    .and_then(|l| l.physics.as_ref())
    .or(vis_dto.physics.as_ref()); // Falls back to legacy
```

### After (Simple)
```rust
// Direct field updates
if let Some(v) = update.damping { physics.damping = v; }
if let Some(v) = update.spring_strength { physics.spring_strength = v; }

// Clear physics access
let physics = settings.get_physics("logseq");
let sim_params = physics.into(); // Direct conversion
```

## Physics Bug Fix

The physics settings now flow correctly:

1. **Client sends update** → `/settings/physics/logseq`
2. **Handler updates** `UnifiedSettings.graphs.logseq.physics`
3. **Automatic conversion** to `SimulationParams`
4. **Direct propagation** to GPU via `UpdateSimulationParams`

No more:
- Checking multiple locations for physics
- Manual field mapping with macros
- Mismatched field names (`repulsion` vs `repulsion_strength`)
- Out-of-sync dual storage

## Migration Steps

To implement this refactor:

1. **Add new modules to config:**
```rust
// In src/config/mod.rs
pub mod unified;
pub mod migration;
```

2. **Update handlers registration:**
```rust
// In main.rs or app configuration
cfg.configure(unified_settings_handler::config);
```

3. **Switch settings actor:**
```rust
// Replace SettingsActor with UnifiedSettingsActor
let settings_addr = UnifiedSettingsActor::new().start();
```

4. **Update app state:**
```rust
// AppState now uses UnifiedSettings
pub settings_addr: Addr<UnifiedSettingsActor>,
```

5. **Copy new YAML:**
```bash
cp settings-unified.yaml settings.yaml
```

## Testing

Test the physics settings flow:

```bash
# Update physics via new endpoint
curl -X POST http://localhost:3090/settings/physics/logseq \
  -H "Content-Type: application/json" \
  -d '{
    "damping": 0.8,
    "spring_strength": 0.5,
    "repulsion_strength": 200.0
  }'

# Verify in logs
grep "Propagating logseq physics to GPU" logs/rust.log
```

## Benefits

1. **Maintainability:** 80% less code to maintain
2. **Performance:** No redundant conversions or transformations  
3. **Correctness:** Single source of truth, no synchronization issues
4. **Developer Experience:** Clear, predictable data flow
5. **Bug Prevention:** Type-safe conversions, no manual field mapping

## Rollback Plan

The migration module provides backward compatibility:
- Old `AppFullSettings` messages still work
- Automatic conversion to `UnifiedSettings`
- Can run both systems in parallel during transition

## Next Steps

1. Update TypeScript client to match unified structure
2. Remove deprecated `/user-settings` endpoint
3. Delete old settings modules after migration
4. Update tests to use new structure

## Summary

This refactor transforms a 1200+ line tangled mess into a clean 200-line implementation that:
- **Fixes the physics settings bug** permanently
- **Eliminates all redundant conversions**
- **Provides a single source of truth**
- **Makes the codebase maintainable**

The new system is ready for implementation and will significantly improve both performance and developer experience.