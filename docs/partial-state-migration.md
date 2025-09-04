# Partial State Management System Migration

## Overview

Successfully ported the partial state management system from the codestore implementation to the client settingsStore. This system provides significant performance improvements by loading only essential settings at startup and lazy-loading others on demand.

## Key Changes Implemented

### 1. Interface Changes
- Changed `settings: Settings` → `partialSettings: DeepPartial<Settings>`
- Added `loadedPaths: Set<string>` to track loaded setting paths
- Added `loadingSections: Set<string>` for concurrent section loading
- Updated `initialize()` return type from `Promise<Settings>` → `Promise<void>`

### 2. Essential Paths System
Added `ESSENTIAL_PATHS` array with 8 critical settings loaded at startup:
- `system.debug.enabled`
- `system.websocket.updateRate`
- `system.websocket.reconnectAttempts`
- `auth.enabled`
- `auth.required`
- `visualisation.rendering.context`
- `xr.enabled`
- `xr.mode`

### 3. Lazy Loading Methods
Implemented four new methods for on-demand loading:

#### `ensureLoaded(paths: string[]): Promise<void>`
- Loads unloaded paths from server
- Skips already loaded paths for performance
- Updates `loadedPaths` Set automatically

#### `loadSection(section: string): Promise<void>`
- Loads entire sections (physics, rendering, xr, etc.)
- Prevents concurrent loading with `loadingSections` tracking
- Supports 9 predefined sections with mapped paths

#### `isLoaded(path: SettingsPath): boolean`
- Checks if a specific path has been loaded
- Simple Set membership check for O(1) performance

#### `getSectionPaths(section: string): string[]`
- Maps section names to arrays of setting paths
- Supports: physics, rendering, xr, glow, hologram, nodes, edges, labels

### 4. Updated Core Methods

#### `get<T>(path: SettingsPath): T | undefined`
- Returns `undefined` for unloaded paths instead of triggering loads
- Checks `loadedPaths` before accessing data
- Includes parent/child path loading logic
- Logs warnings for unloaded path access in debug mode

#### `set<T>(path: SettingsPath, value: T): void`
- Updates `partialSettings` with new values
- Automatically marks paths as loaded in `loadedPaths`
- Uses batch update scheduling for server persistence

#### `updateSettings(updater: (draft: DeepPartial<Settings>) => void): Promise<void>`
- Works with partial settings using immer
- Finds changed paths and updates tracking
- Maintains backward compatibility with existing code

### 5. Storage Changes
- Updated storage key to `graph-viz-settings-v2`
- Only persists essential paths to avoid stale data
- Auth state persistence maintained
- Fresh server loading for all non-essential settings

## Migration Path for Components

### Before (Full Settings)
```typescript
const useSettingsStore = create<SettingsState>()(...);

// Component usage
function MyComponent() {
  const value = useSettingsStore(state => state.get('some.deep.path'));
  // value was always available
}
```

### After (Partial Settings)
```typescript
const useSettingsStore = create<SettingsState>()(...);

// Component usage  
function MyComponent() {
  const { ensureLoaded, get } = useSettingsStore();
  
  useEffect(() => {
    // Ensure required paths are loaded
    ensureLoaded(['some.deep.path']);
  }, [ensureLoaded]);
  
  const value = get('some.deep.path');
  // value might be undefined until loaded
  
  if (value === undefined) {
    return <LoadingSpinner />;
  }
  
  return <div>{value}</div>;
}
```

### Section Loading Pattern
```typescript
function PhysicsPanel() {
  const { loadSection, get } = useSettingsStore();
  
  useEffect(() => {
    // Load all physics-related settings
    loadSection('physics');
  }, [loadSection]);
  
  const springK = get('visualisation.graphs.logseq.physics.springK');
  // Will be available after section loads
}
```

## Performance Benefits

1. **Startup Time**: Only loads 8 essential paths instead of entire settings tree
2. **Memory Usage**: Holds only loaded settings in memory
3. **Network**: Batch API calls for multiple paths
4. **Caching**: `loadedPaths` Set provides O(1) lookup
5. **Concurrent Loading**: `loadingSections` prevents duplicate requests

## Backward Compatibility

- All existing `get()` and `set()` calls work unchanged
- `updateSettings()` maintains same signature
- GPU-specific methods (`updatePhysics`, etc.) work without changes
- Viewport update notifications preserved
- Auto-save batching system maintained

## API Support

The implementation leverages existing settingsApi methods:
- `getSettingsByPaths(paths: string[]): Promise<Record<string, any>>`
- `updateSettingsByPaths(updates: BatchOperation[]): Promise<void>`
- All path-based operations work with current backend

## Testing Requirements

Components using deep settings paths should:
1. Test loading states (when `get()` returns `undefined`)
2. Verify `ensureLoaded()` is called in `useEffect`
3. Check section loading for settings panels
4. Validate essential paths are available immediately after `initialize()`

## Future Optimizations

1. **Smart Prefetching**: Load related paths based on usage patterns
2. **Path Compression**: Store common path prefixes more efficiently  
3. **TTL System**: Add expiration for cached settings
4. **Background Sync**: Periodically refresh loaded settings