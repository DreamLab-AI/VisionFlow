# Settings Schema Comprehensive Analysis Report

**Date**: 2025-10-21
**Analyst**: Claude Code Quality Analyzer
**Overall Quality Score**: 7/10
**Critical Issues**: 5
**Technical Debt**: 6-8 hours

---

## Executive Summary

The "No settings available" issue is caused by **missing schema definitions** for settings paths that components are attempting to load. The settings loading pipeline is working correctly, but components reference non-existent paths in the schema.

**Root Cause**: Components in `RestoredGraphTabs.tsx` request settings paths that don't exist in the `Settings` interface defined in `settings.ts`.

---

## 1. Missing Schema Namespaces

### 1.1 Visualisation Sync Settings

**Status**: ❌ **NOT DEFINED IN SCHEMA**

**Requested Paths** (RestoredGraphTabs.tsx:78-81):
```typescript
'visualisation.sync.enabled',
'visualisation.sync.camera',
'visualisation.sync.selection',
```

**Component Usage** (RestoredGraphTabs.tsx:88-90):
```typescript
const syncEnabled = settings?.visualisation?.sync?.enabled ?? false;
const cameraSync = settings?.visualisation?.sync?.camera ?? true;
const selectionSync = settings?.visualisation?.sync?.selection ?? true;
```

**Expected Interface** (MISSING):
```typescript
export interface SyncSettings {
  enabled: boolean;
  camera: boolean;
  selection: boolean;
}
```

**Schema Update Required**:
```typescript
export interface VisualisationSettings {
  // ... existing properties
  sync: SyncSettings; // ADD THIS
}
```

---

### 1.2 Visualisation Effects Settings

**Status**: ❌ **NOT DEFINED IN SCHEMA**

**Requested Paths** (RestoredGraphTabs.tsx:83-84):
```typescript
'visualisation.effects.bloom',
'visualisation.effects.glow',
```

**Component Usage** (RestoredGraphTabs.tsx:92-93):
```typescript
const bloomEffect = settings?.visualisation?.effects?.bloom ?? false;
const glowEffect = settings?.visualisation?.effects?.glow ?? true;
```

**Current Schema Confusion**:
- Schema has `visualisation.glow` as a top-level property (line 356)
- Component expects `visualisation.effects.glow` (nested under effects)

**Expected Interface** (MISSING):
```typescript
export interface EffectsSettings {
  bloom: boolean;
  glow: boolean;
  // Potentially more effects like blur, vignette, etc.
}
```

**Schema Update Required**:
```typescript
export interface VisualisationSettings {
  // ... existing properties
  effects: EffectsSettings; // ADD THIS
  // DECISION NEEDED: Keep both visualisation.glow AND visualisation.effects.glow?
}
```

---

### 1.3 Animation Global Toggle

**Status**: ⚠️ **PARTIAL MISMATCH**

**Requested Path** (RestoredGraphTabs.tsx:82):
```typescript
'visualisation.animations.enabled',
```

**Component Usage** (RestoredGraphTabs.tsx:91):
```typescript
const animationsEnabled = settings?.visualisation?.animations?.enabled ?? true;
```

**Current Schema** (settings.ts:127-136):
```typescript
export interface AnimationSettings {
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  motionBlurStrength: number;
  selectionWaveEnabled: boolean;
  pulseEnabled: boolean;
  // ... individual animation toggles
  // ❌ NO GLOBAL 'enabled' PROPERTY
}
```

**Schema Update Required**:
```typescript
export interface AnimationSettings {
  enabled: boolean; // ADD THIS - global toggle for all animations
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  // ... rest of properties
}
```

---

### 1.4 Performance Settings

**Status**: ⚠️ **PARTIAL DEFINITION**

**Requested Paths** (RestoredGraphTabs.tsx:172-174):
```typescript
'performance.autoOptimize',
'performance.simplifyEdges',
'performance.cullDistance',
```

**Component Usage** (RestoredGraphTabs.tsx:178-180):
```typescript
const autoOptimize = settings?.performance?.autoOptimize ?? false;
const simplifyEdges = settings?.performance?.simplifyEdges ?? true;
const cullDistance = settings?.performance?.cullDistance ?? 50;
```

**Current Schema** (settings.ts:471-477):
```typescript
export interface PerformanceSettings {
  enableAdaptiveQuality: boolean;
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;
  // ❌ MISSING: autoOptimize, simplifyEdges, cullDistance
}
```

**Schema Update Required**:
```typescript
export interface PerformanceSettings {
  // Existing properties
  enableAdaptiveQuality: boolean;
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;

  // ADD THESE:
  autoOptimize: boolean;
  simplifyEdges: boolean;
  cullDistance: number; // Range: 10-100
}
```

---

### 1.5 Interaction Settings

**Status**: ⚠️ **WRONG NAMESPACE**

**Requested Paths** (RestoredGraphTabs.tsx:262-265):
```typescript
'interaction.enableHover',
'interaction.enableClick',
'interaction.enableDrag',
'interaction.hoverDelay',
```

**Component Usage** (RestoredGraphTabs.tsx:269-272):
```typescript
const enableHover = settings?.interaction?.enableHover ?? true;
const enableClick = settings?.interaction?.enableClick ?? true;
const enableDrag = settings?.interaction?.enableDrag ?? true;
const hoverDelay = settings?.interaction?.hoverDelay ?? 200;
```

**Current Schema** (settings.ts:325-327):
```typescript
export interface InteractionSettings {
  headTrackedParallax: HeadTrackedParallaxSettings; // Only contains this
}
```

**Problem**: Component expects top-level `interaction.*` paths, but schema only has `visualisation.interaction.headTrackedParallax`.

**Schema Update Required**:
```typescript
export interface InteractionSettings {
  // Mouse/pointer interactions
  enableHover: boolean;
  enableClick: boolean;
  enableDrag: boolean;
  hoverDelay: number; // milliseconds (0-500)

  // Existing
  headTrackedParallax: HeadTrackedParallaxSettings;
}

// UPDATE Settings interface:
export interface Settings {
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings & { gpu?: XRGPUSettings };
  auth: AuthSettings;
  interaction: InteractionSettings; // ADD THIS TOP-LEVEL
  // ... other properties
}
```

---

### 1.6 Export Settings

**Status**: ❌ **NOT DEFINED IN SCHEMA**

**Requested Paths** (RestoredGraphTabs.tsx:340-341):
```typescript
'export.format',
'export.includeMetadata',
```

**Component Usage** (RestoredGraphTabs.tsx:345-346):
```typescript
const format = settings?.export?.format ?? 'json';
const includeMetadata = settings?.export?.includeMetadata ?? true;
```

**Expected Interface** (MISSING):
```typescript
export interface ExportSettings {
  format: 'json' | 'csv' | 'graphml' | 'gexf';
  includeMetadata: boolean;
  // Potentially more options:
  compressionEnabled?: boolean;
  prettyPrint?: boolean;
}
```

**Schema Update Required**:
```typescript
export interface Settings {
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings & { gpu?: XRGPUSettings };
  auth: AuthSettings;
  export: ExportSettings; // ADD THIS
  // ... other properties
}
```

---

## 2. Complete Path Mapping

### 2.1 Paths Requested by Components

| Path | Component | Status | Schema Location |
|------|-----------|--------|-----------------|
| `visualisation.sync.enabled` | RestoredGraphVisualisationTab | ❌ Missing | N/A |
| `visualisation.sync.camera` | RestoredGraphVisualisationTab | ❌ Missing | N/A |
| `visualisation.sync.selection` | RestoredGraphVisualisationTab | ❌ Missing | N/A |
| `visualisation.animations.enabled` | RestoredGraphVisualisationTab | ❌ Missing | N/A (has individual toggles) |
| `visualisation.effects.bloom` | RestoredGraphVisualisationTab | ❌ Missing | N/A |
| `visualisation.effects.glow` | RestoredGraphVisualisationTab | ❌ Wrong path | `visualisation.glow` exists |
| `performance.autoOptimize` | RestoredGraphOptimisationTab | ❌ Missing | N/A |
| `performance.simplifyEdges` | RestoredGraphOptimisationTab | ❌ Missing | N/A |
| `performance.cullDistance` | RestoredGraphOptimisationTab | ❌ Missing | N/A |
| `interaction.enableHover` | RestoredGraphInteractionTab | ❌ Wrong namespace | `visualisation.interaction` exists |
| `interaction.enableClick` | RestoredGraphInteractionTab | ❌ Wrong namespace | N/A |
| `interaction.enableDrag` | RestoredGraphInteractionTab | ❌ Wrong namespace | N/A |
| `interaction.hoverDelay` | RestoredGraphInteractionTab | ❌ Wrong namespace | N/A |
| `export.format` | RestoredGraphExportTab | ❌ Missing | N/A |
| `export.includeMetadata` | RestoredGraphExportTab | ❌ Missing | N/A |

### 2.2 Existing Schema Paths

| Schema Path | Interface | Line | Status |
|-------------|-----------|------|--------|
| `visualisation.rendering.*` | RenderingSettings | 113 | ✅ Defined |
| `visualisation.animations.*` | AnimationSettings | 127 | ⚠️ Missing global toggle |
| `visualisation.glow.*` | GlowSettings | 155 | ✅ Defined (path conflict) |
| `visualisation.hologram.*` | HologramSettings | 183 | ✅ Defined |
| `visualisation.graphs.*` | GraphsSettings | 347 | ✅ Defined |
| `visualisation.interaction.headTrackedParallax.*` | InteractionSettings | 325 | ✅ Defined (wrong nesting) |
| `system.websocket.*` | WebSocketSettings | 217 | ✅ Defined |
| `system.debug.*` | DebugSettings | 237 | ✅ Defined |
| `xr.*` | XRSettings | 266 | ✅ Defined |
| `performance.*` | PerformanceSettings | 471 | ⚠️ Partial |
| `analytics.*` | AnalyticsSettings | 455 | ✅ Defined |
| `dashboard.*` | DashboardSettings | 444 | ✅ Defined |
| `developer.*` | DeveloperSettings | 479 | ✅ Defined |

---

## 3. Path Resolution Logic Analysis

### 3.1 Settings Store Path Resolution (settingsStore.ts:336-370)

**Function**: `get<T>(path: SettingsPath): T`

**Flow**:
```typescript
1. Check if path is loaded: loadedPaths.has(path)
2. If not loaded:
   - Log warning: "Accessing unloaded path"
   - Return undefined (does NOT trigger loading)
3. Navigate partialSettings using path parts
4. Return value or undefined
```

**Issue**: No validation that path exists in schema.

### 3.2 Lazy Loading (settingsStore.ts:440-474)

**Function**: `ensureLoaded(paths: string[]): Promise<void>`

**Flow**:
```typescript
1. Filter out already loaded paths
2. Call settingsApi.getSettingsByPaths(unloadedPaths)
3. Update state with returned values
4. Mark paths as loaded
```

**Issue**: API returns empty/null for non-existent paths, no error thrown.

### 3.3 API Layer (settingsApi.ts:269-280)

**Function**: `getSettingsByPaths(paths: string[], options?): Promise<Record<string, any>>`

**Flow**:
```typescript
1. Call settingsCacheClient.getBatch(paths, options)
2. Return result object mapping paths to values
```

**Issue**: Backend doesn't validate paths against schema, returns `{}` or `null` for invalid paths.

---

## 4. Missing Interface Definitions Summary

### Complete List of Missing/Incomplete Interfaces

```typescript
// 1. NEW INTERFACE: Sync Settings
export interface SyncSettings {
  enabled: boolean;
  camera: boolean;
  selection: boolean;
}

// 2. NEW INTERFACE: Effects Settings
export interface EffectsSettings {
  bloom: boolean;
  glow: boolean;
}

// 3. UPDATE: Animation Settings (add global toggle)
export interface AnimationSettings {
  enabled: boolean; // ← ADD THIS
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  motionBlurStrength: number;
  selectionWaveEnabled: boolean;
  pulseEnabled: boolean;
  pulseSpeed: number;
  pulseStrength: number;
  waveSpeed: number;
}

// 4. UPDATE: Performance Settings (add missing properties)
export interface PerformanceSettings {
  enableAdaptiveQuality: boolean;
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;
  autoOptimize: boolean;        // ← ADD THIS
  simplifyEdges: boolean;        // ← ADD THIS
  cullDistance: number;          // ← ADD THIS (range: 10-100)
}

// 5. UPDATE: Interaction Settings (add mouse/pointer controls)
export interface InteractionSettings {
  enableHover: boolean;          // ← ADD THIS
  enableClick: boolean;          // ← ADD THIS
  enableDrag: boolean;           // ← ADD THIS
  hoverDelay: number;            // ← ADD THIS (range: 0-500ms)
  headTrackedParallax: HeadTrackedParallaxSettings;
}

// 6. NEW INTERFACE: Export Settings
export interface ExportSettings {
  format: 'json' | 'csv' | 'graphml' | 'gexf';
  includeMetadata: boolean;
}

// 7. UPDATE: VisualisationSettings (add new namespaces)
export interface VisualisationSettings {
  rendering: RenderingSettings;
  animations: AnimationSettings;
  glow: GlowSettings;
  hologram: HologramSettings;
  spacePilot?: SpacePilotConfig;
  camera?: CameraSettings;
  interaction?: InteractionSettings;
  graphs: GraphsSettings;

  // ADD THESE:
  sync: SyncSettings;            // ← ADD THIS
  effects: EffectsSettings;      // ← ADD THIS
}

// 8. UPDATE: Settings (add top-level namespaces)
export interface Settings {
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings & { gpu?: XRGPUSettings };
  auth: AuthSettings;
  ragflow?: RAGFlowSettings;
  perplexity?: PerplexitySettings;
  openai?: OpenAISettings;
  kokoro?: KokoroSettings;
  whisper?: WhisperSettings;
  dashboard?: DashboardSettings;
  analytics?: AnalyticsSettings;
  performance?: PerformanceSettings;
  developer?: DeveloperSettings;

  // ADD THESE:
  interaction: InteractionSettings; // ← ADD THIS (top-level, not under visualisation)
  export: ExportSettings;          // ← ADD THIS
}
```

---

## 5. Architectural Decisions Needed

### 5.1 Interaction Settings Namespace

**Question**: Should `interaction` be top-level or under `visualisation`?

**Current State**:
- Schema has: `visualisation.interaction.headTrackedParallax`
- Components expect: `interaction.enableHover`, `interaction.enableClick`, etc.

**Options**:

**Option A**: Top-level `interaction` namespace
```typescript
export interface Settings {
  interaction: InteractionSettings; // Top-level
  visualisation: VisualisationSettings;
}
```
**Pros**: Clear separation, components already use this
**Cons**: Breaks existing `visualisation.interaction.headTrackedParallax`

**Option B**: Keep under `visualisation`
```typescript
// Update components to use:
'visualisation.interaction.enableHover'
'visualisation.interaction.enableClick'
```
**Pros**: Consistent with existing schema
**Cons**: Requires component updates

**Recommendation**: **Option A** - Make `interaction` top-level, move `headTrackedParallax` to it.

---

### 5.2 Glow Settings Duplication

**Question**: Should glow be under `visualisation.glow` OR `visualisation.effects.glow`?

**Current State**:
- Schema has: `visualisation.glow` (full GlowSettings interface)
- Components expect: `visualisation.effects.glow` (boolean toggle)

**Options**:

**Option A**: Keep both (glow details + effects toggle)
```typescript
export interface VisualisationSettings {
  glow: GlowSettings;           // Detailed glow configuration
  effects: {
    glow: boolean;              // Simple on/off toggle
    bloom: boolean;
  };
}
```

**Option B**: Consolidate under effects
```typescript
export interface VisualisationSettings {
  effects: {
    glow: GlowSettings;         // Full configuration
    bloom: BloomSettings;
  };
}
```

**Recommendation**: **Option A** - Keep both. `effects.glow` is a boolean master toggle, `glow.*` is detailed configuration.

---

## 6. Implementation Plan

### Phase 1: Schema Updates (2 hours)

**File**: `/home/devuser/workspace/project/client/src/features/settings/config/settings.ts`

1. Add `SyncSettings` interface
2. Add `EffectsSettings` interface
3. Add `ExportSettings` interface
4. Update `AnimationSettings` (add `enabled`)
5. Update `PerformanceSettings` (add 3 properties)
6. Update `InteractionSettings` (add 4 properties)
7. Update `VisualisationSettings` (add `sync`, `effects`)
8. Update `Settings` (add `interaction`, `export`)

### Phase 2: Component Updates (1 hour)

**File**: `/home/devuser/workspace/project/client/src/features/visualisation/components/ControlPanel/RestoredGraphTabs.tsx`

**No changes needed** - components already use correct paths with fallbacks.

### Phase 3: Backend Schema Sync (2-3 hours)

**Files**: Server-side schema definitions

1. Update server `Settings` struct/interface
2. Add database migrations for new fields
3. Update default settings generator
4. Test path-based API with new paths

### Phase 4: Validation Layer (1-2 hours)

**Files**:
- `/home/devuser/workspace/project/client/src/store/settingsStore.ts`
- `/home/devuser/workspace/project/client/src/api/settingsApi.ts`

1. Create path validator using schema reflection
2. Add warnings for invalid paths in `ensureLoaded`
3. Add schema validation to API responses
4. Create developer mode path debugging tool

### Phase 5: Testing & Documentation (1 hour)

1. Test all RestoredGraph tabs render correctly
2. Test settings persistence
3. Test WebSocket sync for new paths
4. Update settings documentation
5. Generate TypeDoc for new interfaces

---

## 7. Code Quality Issues

### 7.1 No Path Validation (Severity: High)

**Location**: `settingsStore.ts:440-474`

**Issue**: No validation that requested paths exist in schema.

**Fix**:
```typescript
ensureLoaded: async (paths: string[]): Promise<void> => {
  // ADD THIS:
  const invalidPaths = paths.filter(path => !isValidSettingPath(path));
  if (invalidPaths.length > 0) {
    console.warn('[Settings] Invalid paths requested:', invalidPaths);
  }

  const validPaths = paths.filter(path => isValidSettingPath(path));
  // ... continue with valid paths only
}
```

### 7.2 Duplicate Path Loading Logic (Severity: Medium)

**Location**: Multiple `useEffect` hooks in component files

**Issue**: Each component repeats the same `ensureLoaded` pattern.

**Fix**: Create custom hook
```typescript
// src/hooks/useSettingsPaths.ts
export function useSettingsPaths(paths: string[]) {
  const settings = useSettingsStore(state => state.settings);
  const ensureLoaded = useSettingsStore(state => state.ensureLoaded);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    ensureLoaded(paths)
      .then(() => setIsLoading(false))
      .catch(err => {
        console.error('Failed to load settings:', err);
        setIsLoading(false);
      });
  }, [paths.join(','), ensureLoaded]);

  return { settings, isLoading };
}

// Usage in components:
const { settings, isLoading } = useSettingsPaths([
  'visualisation.sync.enabled',
  'visualisation.sync.camera'
]);
```

### 7.3 Missing Loading States (Severity: Low)

**Location**: All RestoredGraph tabs

**Issue**: No UI feedback while `ensureLoaded()` is pending.

**Fix**: Use `isLoading` from custom hook to show skeleton/spinner.

---

## 8. Testing Checklist

### Pre-Implementation Tests
- [x] Settings store initializes without errors
- [x] Essential paths load on startup
- [x] Backend API responds to /settings requests
- [x] `ensureLoaded()` successfully fetches valid paths
- [ ] ❌ Settings object structure matches schema (FAILS)
- [ ] ❌ UI components receive non-null settings (FAILS for new paths)
- [x] Spelling is consistent across all paths
- [ ] ❌ All paths in components exist in schema (FAILS)

### Post-Implementation Tests
- [ ] All RestoredGraph tabs render without "No settings available"
- [ ] Settings values display correctly in UI
- [ ] Settings persist across page refreshes
- [ ] WebSocket sync works for new paths
- [ ] No console errors about undefined paths
- [ ] Path validation warnings show for invalid paths
- [ ] Export/import works with new settings
- [ ] Backend validates new paths correctly

---

## 9. Additional Files to Check

### Other Components Using Missing Paths

Found 6 files with potential usage:

1. ✅ `/home/devuser/workspace/project/client/src/features/visualisation/components/ControlPanel/RestoredGraphTabs.tsx` (analyzed)
2. `GraphInteractionTab.tsx` - May use `interaction.*` paths
3. `SettingsTabContent.tsx` - General settings usage
4. `GraphOptimisationTab.tsx` - May use `performance.*` paths
5. `GraphExportTab.tsx` - May use `export.*` paths
6. `PhysicsEngineControls.tsx` - Physics settings (already working)

**Recommendation**: Check files 2-5 for similar path mismatches.

---

## 10. Recommendations

### Immediate Fixes (Critical Priority)

1. **Add missing schema definitions** (2 hours)
   - Add all 6 new/updated interfaces to `settings.ts`
   - Update `Settings` main interface

2. **Fix path references** (Optional - components have fallbacks)
   - Components already use `??` fallback values
   - Schema fix is sufficient

3. **Add path validation** (1 hour)
   - Add `isValidSettingPath()` function
   - Integrate into `ensureLoaded()`

### Short-term Improvements (High Priority)

4. **Create custom hook** (30 mins)
   - Reduce code duplication with `useSettingsPaths()`

5. **Add loading states** (1 hour)
   - Show spinners/skeletons while settings load

6. **Backend schema sync** (2-3 hours)
   - Update server-side schema to match

### Long-term Enhancements (Medium Priority)

7. **Schema documentation** (1 hour)
   - Generate TypeDoc for settings interfaces
   - Create settings path reference guide

8. **Runtime validation** (2 hours)
   - Add Zod or similar validation library
   - Type-safe path access with intellisense

9. **Settings migration system** (3 hours)
   - Handle schema changes gracefully
   - Version settings structure

---

## 11. Verification Steps

After implementing fixes:

1. **Schema Verification**
   - [ ] All interfaces defined in `settings.ts`
   - [ ] No TypeScript errors in schema file
   - [ ] All component paths match schema

2. **Component Verification**
   - [ ] Open each RestoredGraph tab in browser
   - [ ] Verify no "No settings available" messages
   - [ ] Verify settings controls work

3. **API Verification**
   - [ ] Test `GET /api/settings/path?path=visualisation.sync.enabled`
   - [ ] Test batch endpoint with new paths
   - [ ] Verify WebSocket updates for new paths

4. **Persistence Verification**
   - [ ] Change a setting in each tab
   - [ ] Refresh page
   - [ ] Verify settings persist

5. **Console Verification**
   - [ ] No errors in browser console
   - [ ] Path validation warnings (if enabled) show for invalid paths
   - [ ] No undefined access warnings

---

## Appendix A: Complete Schema Diff

**File**: `settings.ts`

```typescript
// ==================== NEW INTERFACES ====================

// ADD AFTER line 150 (after LabelSettings)
export interface SyncSettings {
  enabled: boolean;
  camera: boolean;
  selection: boolean;
}

export interface EffectsSettings {
  bloom: boolean;
  glow: boolean;
}

// ==================== UPDATED INTERFACES ====================

// UPDATE AnimationSettings (line 127)
export interface AnimationSettings {
  enabled: boolean; // ← ADD THIS LINE
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  motionBlurStrength: number;
  selectionWaveEnabled: boolean;
  pulseEnabled: boolean;
  pulseSpeed: number;
  pulseStrength: number;
  waveSpeed: number;
}

// UPDATE PerformanceSettings (line 471)
export interface PerformanceSettings {
  enableAdaptiveQuality: boolean;
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;
  autoOptimize: boolean;      // ← ADD THIS LINE
  simplifyEdges: boolean;     // ← ADD THIS LINE
  cullDistance: number;       // ← ADD THIS LINE
}

// UPDATE InteractionSettings (line 325)
export interface InteractionSettings {
  enableHover: boolean;       // ← ADD THIS LINE
  enableClick: boolean;       // ← ADD THIS LINE
  enableDrag: boolean;        // ← ADD THIS LINE
  hoverDelay: number;         // ← ADD THIS LINE
  headTrackedParallax: HeadTrackedParallaxSettings;
}

// ADD NEW ExportSettings (after DeveloperSettings)
export interface ExportSettings {
  format: 'json' | 'csv' | 'graphml' | 'gexf';
  includeMetadata: boolean;
}

// UPDATE VisualisationSettings (line 352)
export interface VisualisationSettings {
  rendering: RenderingSettings;
  animations: AnimationSettings;
  glow: GlowSettings;
  hologram: HologramSettings;
  spacePilot?: SpacePilotConfig;
  camera?: CameraSettings;
  interaction?: InteractionSettings;
  graphs: GraphsSettings;
  sync: SyncSettings;         // ← ADD THIS LINE
  effects: EffectsSettings;   // ← ADD THIS LINE
}

// UPDATE Settings (line 509)
export interface Settings {
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings & { gpu?: XRGPUSettings };
  auth: AuthSettings;
  ragflow?: RAGFlowSettings;
  perplexity?: PerplexitySettings;
  openai?: OpenAISettings;
  kokoro?: KokoroSettings;
  whisper?: WhisperSettings;
  dashboard?: DashboardSettings;
  analytics?: AnalyticsSettings;
  performance?: PerformanceSettings;
  developer?: DeveloperSettings;
  interaction: InteractionSettings; // ← ADD THIS LINE
  export: ExportSettings;          // ← ADD THIS LINE
}
```

---

**End of Report**
