# Settings Migration Utility

⚠️ **CRITICAL DUAL STORE ISSUE** ⚠️

There are currently **TWO separate settings stores** causing conflicts:
1. **Main Store**: `/client/src/store/settingsStore.ts` (functional, Zustand-based)
2. **Legacy Store**: `/client/src/features/settings/store/settingsStore.ts` (placeholder, non-functional)

This dual store architecture creates:
- Import conflicts when components use the wrong store
- Settings that don't persist correctly
- Race conditions during initialization
- Broken settings panel functionality

**IMMEDIATE ACTION REQUIRED**: Consolidate to single store architecture.

## Migration from Legacy to Multi-Graph Structure

This document describes the settings migration utility that handles the transition from the legacy flat settings structure to the new multi-graph architecture.

## Overview

The settings migration utility (`client/src/features/settings/utils/settingsMigration.ts`) provides automatic migration and compatibility functions for transitioning between settings structures.

## Migration Functions

### `migrateToMultiGraphSettings(settings: Settings): Settings`

Automatically migrates settings from the old flat structure to the new multi-graph structure.

**Migration Process:**
1. **Detection**: Checks if settings are already migrated by looking for `visualisation.graphs.logseq` and `visualisation.graphs.visionflow`
2. **Migration**: If not migrated:
   - Creates the `graphs` namespace structure
   - Copies legacy settings to the `logseq` graph
   - Initializes `visionflow` graph with default green theme
   - Removes deprecated flat structure fields
3. **Cleanup**: Removes legacy fields to prevent confusion

**Example Migration:**
```typescript
// Before migration
const oldSettings = {
  visualisation: {
    nodes: { baseColor: '#4B5EFF', nodeSize: 8 },
    edges: { color: '#F59E0B', baseWidth: 2 },
    physics: { enabled: true, springStrength: 0.1 },
    labels: { fontSize: 14, textColor: '#FFFFFF' }
  }
};

// After migration
const newSettings = {
  visualisation: {
    graphs: {
      logseq: {
        nodes: { baseColor: '#4B5EFF', nodeSize: 8 },
        edges: { color: '#F59E0B', baseWidth: 2 },
        physics: { enabled: true, springStrength: 0.1 },
        labels: { fontSize: 14, textColor: '#FFFFFF' }
      },
      visionflow: {
        nodes: { baseColor: '#10B981', nodeSize: 10 },
        edges: { color: '#34D399', baseWidth: 3 },
        physics: { enabled: true, springStrength: 0.15 },
        labels: { fontSize: 16, textColor: '#ECFDF5' }
      }
    },
    // Global settings remain unchanged
    rendering: { ... },
    animations: { ... },
    bloom: { ... },
    hologram: { ... }
  }
};
```

### `getGraphSettings(settings: Settings, graphName: 'logseq' | 'visionflow'): GraphSettings`

Retrieves settings for a specific graph with fallback to legacy settings for backward compatibility.

**Fallback Logic:**
1. First checks for settings in the new multi-graph structure
2. Falls back to legacy flat structure if new structure doesn't exist
3. Finally falls back to default settings for the specified graph

**Usage Example:**
```typescript
// Get settings for the logseq graph
const logseqSettings = getGraphSettings(settings, 'logseq');

// Access specific settings
const nodeColor = logseqSettings.nodes.baseColor;
const physicsEnabled = logseqSettings.physics.enabled;
```

## Integration with Settings Store

The migration utility is integrated into the settings store to ensure automatic migration when settings are loaded:

```typescript
// In settingsStore.ts
const loadedSettings = await settingsService.getSettings();
const migratedSettings = migrateToMultiGraphSettings(loadedSettings);
```

## Viewport Settings Support

The migration works seamlessly with viewport settings patterns defined in `viewportSettings.ts`. This configuration determines which settings require immediate viewport updates without waiting for the debounced save.

### Viewport Settings Patterns

```typescript
// From client/src/features/settings/config/viewportSettings.ts
export const VIEWPORT_SETTINGS_PATTERNS = [
  // Legacy visualization settings
  'visualisation.nodes',
  'visualisation.edges',
  'visualisation.physics',
  'visualisation.rendering',
  'visualisation.animations',
  'visualisation.labels',
  'visualisation.bloom',
  'visualisation.hologram',
  'visualisation.camera',
  
  // NEW: Graph-specific visualization settings (post-migration)
  'visualisation.graphs.*.nodes',
  'visualisation.graphs.*.edges',
  'visualisation.graphs.*.physics',
  'visualisation.graphs.*.rendering',
  'visualisation.graphs.*.animations',
  'visualisation.graphs.*.labels',
  'visualisation.graphs.*.bloom',
  'visualisation.graphs.*.hologram',
  'visualisation.graphs.*.camera',
  
  // XR settings that affect viewport
  'xr.mode',
  'xr.quality',
  'xr.render_scale',
  // ... other XR settings
];
```

### Viewport Settings Functions

The viewport settings module provides utilities for detecting real-time settings:

```typescript
// Check if a settings path requires immediate viewport update
function isViewportSetting(path: string): boolean {
  // Returns true for patterns like:
  // - 'visualisation.graphs.logseq.nodes.baseColor'
  // - 'visualisation.graphs.visionflow.physics.enabled'
  // - 'visualisation.rendering.backgroundColor'
}

// Extract viewport-related paths from settings update
function getViewportPaths(paths: string[]): string[] {
  // Filters array to only viewport-affecting paths
}
```

### Migration Impact on Viewport Settings

The migration from flat to multi-graph structure affects viewport settings in several ways:

1. **Pattern Expansion**: New wildcard patterns support graph-specific settings
2. **Legacy Support**: Old patterns remain for backward compatibility
3. **Performance**: Same real-time update behavior for both structures
4. **Consistency**: Both legacy and new paths trigger immediate viewport updates

**Example Usage:**
```typescript
// Both of these trigger immediate viewport updates:
settingsStore.set('visualisation.nodes.baseColor', '#FF0000');           // Legacy
settingsStore.set('visualisation.graphs.logseq.nodes.baseColor', '#0000FF'); // New
```

## Critical Issues to Address

### Dual Store Architecture Problem

The current architecture has two conflicting stores:

**Main Store** (`/client/src/store/settingsStore.ts`):
- ✅ Functional Zustand implementation with persistence
- ✅ Multi-graph migration logic implemented
- ✅ Real-time viewport updates working
- ✅ Server synchronization functional

**Legacy Store** (`/client/src/features/settings/store/settingsStore.ts`):
- ❌ Placeholder implementation (non-functional)
- ❌ Commented-out Zustand code
- ❌ Missing dependencies
- ❌ Breaks components that import it

### Migration Path to Single Store

**Phase 1 - Immediate (Critical)**:
1. Audit all components using `/features/settings/store/settingsStore`
2. Update imports to use main store: `/store/settingsStore`
3. Remove or deprecate the legacy store file
4. Fix broken settings panel functionality

**Phase 2 - Consolidation**:
1. Move any valid logic from features store to main store
2. Update all import paths across codebase
3. Test settings persistence and real-time updates
4. Verify multi-graph migration works correctly

**Phase 3 - Cleanup**:
1. Remove legacy store file entirely
2. Update documentation to reflect single store architecture
3. Add guards to prevent dual store issues in future

## Best Practices (Updated)

1. **Single Store Only**: Always use `/client/src/store/settingsStore.ts`
2. **Never Import Legacy Store**: Avoid `/features/settings/store/settingsStore.ts`
3. **Always Use Migration**: When loading settings, always run them through `migrateToMultiGraphSettings`
4. **Use Helper Functions**: Use `getGraphSettings` during the transition period for backward compatibility
5. **Update Components Gradually**: Components can be updated to use the new structure while maintaining compatibility
6. **Monitor for Import Conflicts**: Watch for accidental imports of the wrong store

## Migration Timeline

### Store Consolidation (CRITICAL - Do First)
1. **Phase 1**: Audit and fix all store imports (IMMEDIATE)
2. **Phase 2**: Remove legacy store file
3. **Phase 3**: Verify settings panel functionality restored

### Settings Structure Migration (Current)
1. **Phase 1**: Automatic migration on load (✅ implemented)
2. **Phase 2**: Update all components to use new structure
3. **Phase 3**: Remove legacy field support
4. **Phase 4**: Remove migration utility (once all users have migrated)

## URGENT: Complete Migration Action Plan

### Step 1: Identify Affected Components (IMMEDIATE)

**Audit these files for wrong store imports**:
```bash
# Search for problematic imports
grep -r "features/settings/store/settingsStore" client/src/
grep -r "from.*settingsStore" client/src/ | grep features

# Expected problematic files:
# - client/src/features/settings/components/panels/SettingsPanelRedesign.tsx
# - client/src/features/settings/components/SettingControlComponent.tsx  
# - client/src/features/settings/components/*.tsx
```

### Step 2: Fix Store Imports (CRITICAL - 30 min task)

**Replace ALL instances of**:
```typescript
// ❌ WRONG - causes settings to not work
import { useSettingsStore } from '../store/settingsStore';
import { useSettingsStore } from '@/features/settings/store/settingsStore';
import { useSettingsStore } from '../../features/settings/store/settingsStore';

// ✅ CORRECT - functional store
import { useSettingsStore } from '@/store/settingsStore';
import { useSettingsStore } from '../../../store/settingsStore';
```

### Step 3: Remove Legacy Store File

**After fixing all imports, delete**:
```bash
rm client/src/features/settings/store/settingsStore.ts
```

This eliminates the source of confusion and prevents future import errors.

### Step 4: Verify Settings Panel Works

**Test these functions after fixes**:
1. Settings panel opens and displays controls
2. Sliders respond to mouse interaction  
3. Color pickers update visualization in real-time
4. Changes persist after page refresh
5. Multi-graph switching works (logseq ↔ visionflow)

### Step 5: Multi-Graph Migration Testing

**Test the automatic migration logic**:
```typescript
import { migrateToMultiGraphSettings, getGraphSettings } from './settingsMigration';

// Test with legacy settings
const legacySettings = { /* ... old structure ... */ };
const migrated = migrateToMultiGraphSettings(legacySettings);

// Verify migration worked
console.assert(migrated.visualisation.graphs.logseq !== undefined);
console.assert(migrated.visualisation.graphs.visionflow !== undefined);
console.assert(migrated.visualisation.nodes === undefined);

// Test helper function
const logseqSettings = getGraphSettings(migrated, 'logseq');
console.assert(logseqSettings.nodes.baseColor === '#4B5EFF');
```

## Component-Specific Fix Examples

### Settings Panel Redesign
```typescript
// In SettingsPanelRedesign.tsx
// WRONG:
import { useSettingsStore } from '../store/settingsStore';

// CORRECT:
import { useSettingsStore } from '@/store/settingsStore';

// Usage (multi-graph aware):
const settings = useSettingsStore(state => state.settings);
const updateSettings = useSettingsStore(state => state.updateSettings);

// Update graph-specific setting:
updateSettings(draft => {
  draft.visualisation.graphs.logseq.nodes.baseColor = newColor;
});
```

### Setting Control Component  
```typescript
// In SettingControlComponent.tsx
// Ensure proper path resolution:
const value = useSettingsStore(state => {
  // Handle both legacy and new structure
  if (path.startsWith('visualisation.graphs.')) {
    return state.get(path); // New multi-graph structure
  }
  return state.get(path); // Will be auto-migrated
});
```

## Recovery Checklist

**After implementing fixes, verify**:
- [ ] Settings panel opens without errors
- [ ] All control types render correctly (sliders, toggles, color pickers)
- [ ] Real-time updates work (viewport changes immediately)
- [ ] Settings persist to server (check Network tab)
- [ ] Multi-graph themes work (blue logseq, green visionflow)
- [ ] No console errors related to settings store
- [ ] Page refresh preserves settings
- [ ] Authentication-gated settings work correctly

**If issues persist**:
1. Check browser console for import errors
2. Verify store initialization completed
3. Check Network tab for failed settings API calls
4. Test with fresh localStorage (clear browser storage)

This dual store bug is the #1 blocker for settings functionality. Fix store imports first, then all other features will work correctly.