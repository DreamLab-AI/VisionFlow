# Settings Schema Fix - Quick Reference Guide

**Problem**: "No settings available" in control panel tabs
**Root Cause**: Missing schema definitions for paths components are trying to load
**Fix Complexity**: Medium (6-8 hours)
**Files Affected**: 3 main files + backend

---

## TL;DR - What to Do

1. Copy interfaces from Appendix A below
2. Paste into `settings.ts` at specified lines
3. Update backend schema to match
4. Test control panel tabs

---

## Appendix A: Copy-Paste Schema Fix

### Step 1: Add New Interfaces

**File**: `/home/devuser/workspace/project/client/src/features/settings/config/settings.ts`
**Location**: After line 150 (after `LabelSettings`)

```typescript
// ==================== ADD THESE TWO INTERFACES ====================

export interface SyncSettings {
  enabled: boolean;
  camera: boolean;
  selection: boolean;
}

export interface EffectsSettings {
  bloom: boolean;
  glow: boolean;
}

export interface ExportSettings {
  format: 'json' | 'csv' | 'graphml' | 'gexf';
  includeMetadata: boolean;
}
```

### Step 2: Update AnimationSettings

**File**: Same file
**Location**: Line 127

**BEFORE**:
```typescript
export interface AnimationSettings {
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  motionBlurStrength: number;
  // ...
```

**AFTER**:
```typescript
export interface AnimationSettings {
  enabled: boolean; // ← ADD THIS LINE
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  motionBlurStrength: number;
  // ...
```

### Step 3: Update PerformanceSettings

**File**: Same file
**Location**: Line 471

**BEFORE**:
```typescript
export interface PerformanceSettings {
  enableAdaptiveQuality: boolean;
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;
}
```

**AFTER**:
```typescript
export interface PerformanceSettings {
  enableAdaptiveQuality: boolean;
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;
  autoOptimize: boolean;      // ← ADD THIS
  simplifyEdges: boolean;     // ← ADD THIS
  cullDistance: number;       // ← ADD THIS (range: 10-100)
}
```

### Step 4: Update InteractionSettings

**File**: Same file
**Location**: Line 325

**BEFORE**:
```typescript
export interface InteractionSettings {
  headTrackedParallax: HeadTrackedParallaxSettings;
}
```

**AFTER**:
```typescript
export interface InteractionSettings {
  enableHover: boolean;       // ← ADD THIS
  enableClick: boolean;       // ← ADD THIS
  enableDrag: boolean;        // ← ADD THIS
  hoverDelay: number;         // ← ADD THIS (range: 0-500ms)
  headTrackedParallax: HeadTrackedParallaxSettings;
}
```

### Step 5: Update VisualisationSettings

**File**: Same file
**Location**: Line 352

**BEFORE**:
```typescript
export interface VisualisationSettings {
  rendering: RenderingSettings;
  animations: AnimationSettings;
  glow: GlowSettings;
  hologram: HologramSettings;
  spacePilot?: SpacePilotConfig;
  camera?: CameraSettings;
  interaction?: InteractionSettings;
  graphs: GraphsSettings;
}
```

**AFTER**:
```typescript
export interface VisualisationSettings {
  rendering: RenderingSettings;
  animations: AnimationSettings;
  glow: GlowSettings;
  hologram: HologramSettings;
  spacePilot?: SpacePilotConfig;
  camera?: CameraSettings;
  interaction?: InteractionSettings;
  graphs: GraphsSettings;
  sync: SyncSettings;         // ← ADD THIS
  effects: EffectsSettings;   // ← ADD THIS
}
```

### Step 6: Update Main Settings Interface

**File**: Same file
**Location**: Line 509

**BEFORE**:
```typescript
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
}
```

**AFTER**:
```typescript
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
  interaction: InteractionSettings; // ← ADD THIS
  export: ExportSettings;          // ← ADD THIS
}
```

---

## Testing the Fix

### 1. TypeScript Compilation

```bash
cd /home/devuser/workspace/project/client
npm run typecheck
```

Expected: No TypeScript errors

### 2. Browser Testing

1. Open app in browser
2. Navigate to Control Panel
3. Open each tab:
   - **Visualisation** - Check sync toggles appear
   - **Optimisation** - Check performance controls appear
   - **Interaction** - Check hover/click/drag toggles appear
   - **Export** - Check format dropdown appears

Expected: No "No settings available" messages

### 3. Console Verification

Open browser console, look for:
- ❌ No errors about undefined paths
- ❌ No warnings about accessing unloaded paths
- ✅ Settings values log correctly

### 4. Persistence Test

1. Change a setting in each tab
2. Refresh page (F5)
3. Verify settings persisted

---

## Backend Schema Update

**If backend validation fails**, update server-side schema:

### Rust/Go/Java Example Pattern

```rust
// Add to server settings struct
pub struct VisualisationSettings {
    // ... existing fields
    pub sync: SyncSettings,
    pub effects: EffectsSettings,
}

pub struct SyncSettings {
    pub enabled: bool,
    pub camera: bool,
    pub selection: bool,
}

pub struct EffectsSettings {
    pub bloom: bool,
    pub glow: bool,
}

// Add to top-level Settings
pub struct Settings {
    // ... existing fields
    pub interaction: InteractionSettings,
    pub export: ExportSettings,
}
```

---

## Default Values

Add to server's default settings generator:

```json
{
  "visualisation": {
    "sync": {
      "enabled": false,
      "camera": true,
      "selection": true
    },
    "effects": {
      "bloom": false,
      "glow": true
    },
    "animations": {
      "enabled": true
    }
  },
  "interaction": {
    "enableHover": true,
    "enableClick": true,
    "enableDrag": true,
    "hoverDelay": 200
  },
  "performance": {
    "autoOptimize": false,
    "simplifyEdges": true,
    "cullDistance": 50
  },
  "export": {
    "format": "json",
    "includeMetadata": true
  }
}
```

---

## Path Validation (Optional Enhancement)

Add to `settingsStore.ts` after line 440:

```typescript
// Helper to validate paths against schema
function isValidSettingPath(path: string): boolean {
  const validPaths = new Set([
    // Existing essential paths
    ...ESSENTIAL_PATHS,

    // New paths
    'visualisation.sync.enabled',
    'visualisation.sync.camera',
    'visualisation.sync.selection',
    'visualisation.animations.enabled',
    'visualisation.effects.bloom',
    'visualisation.effects.glow',
    'performance.autoOptimize',
    'performance.simplifyEdges',
    'performance.cullDistance',
    'interaction.enableHover',
    'interaction.enableClick',
    'interaction.enableDrag',
    'interaction.hoverDelay',
    'export.format',
    'export.includeMetadata',
  ]);

  return validPaths.has(path) ||
    path.startsWith('visualisation.graphs.') ||
    path.startsWith('system.') ||
    path.startsWith('xr.');
}

// Update ensureLoaded to use validation
ensureLoaded: async (paths: string[]): Promise<void> => {
  const invalidPaths = paths.filter(path => !isValidSettingPath(path));

  if (invalidPaths.length > 0) {
    console.warn('[Settings] Invalid paths requested:', invalidPaths);
  }

  const validPaths = paths.filter(path => isValidSettingPath(path));
  // ... rest of function
}
```

---

## Troubleshooting

### Issue: TypeScript errors after adding interfaces

**Solution**: Run `npm install` to update type definitions

### Issue: Settings still show "No settings available"

**Checklist**:
1. ✅ Cleared browser cache?
2. ✅ Refreshed page (Ctrl+F5)?
3. ✅ Checked browser console for errors?
4. ✅ Backend returning values for new paths?

**Debug command**:
```bash
# Check API response
curl http://localhost:YOUR_PORT/api/settings/path?path=visualisation.sync.enabled
```

### Issue: Settings don't persist

**Solution**: Check that backend has default values for new paths

---

## Rollback Plan

If fix causes issues, revert changes:

```bash
cd /home/devuser/workspace/project/client
git diff src/features/settings/config/settings.ts
git checkout src/features/settings/config/settings.ts
```

---

## Success Criteria

✅ All RestoredGraph tabs render without errors
✅ Settings controls appear and are functional
✅ Settings persist across page refreshes
✅ No console errors about undefined paths
✅ TypeScript compilation succeeds
✅ Backend accepts new setting paths

---

## Time Estimates

| Task | Time |
|------|------|
| Client schema update | 30 mins |
| Backend schema update | 1-2 hours |
| Testing | 30 mins |
| Optional validation | 1 hour |
| **Total** | **3-4 hours** |

(With backend database migrations: 6-8 hours)

---

## Questions?

See full analysis documents:
- `/home/devuser/workspace/project/client/docs/SETTINGS_SCHEMA_COMPREHENSIVE_ANALYSIS.md`
- `/home/devuser/workspace/project/client/docs/SETTINGS_NO_AVAILABLE_ANALYSIS.md`
- `/home/devuser/workspace/project/client/docs/ANALYSIS_COMPARISON.md`

---

**Last Updated**: 2025-10-21
**Version**: 1.0
**Status**: Ready for implementation
