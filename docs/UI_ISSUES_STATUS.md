# UI Issues - Status Update

**Date**: 2025-10-22
**Session**: Post-Crash Fix

## ✅ FIXED

### 1. Backend SIGSEGV Crash ✅
**Status**: RESOLVED
**Fix**: Custom `NetworkSettings::default()` implementation
**File**: `src/config/mod.rs:1239-1261`
**Result**: Backend stable, no more 30s crash cycles

### 2. Settings API Batch Updates ✅
**Status**: RESOLVED
**Fix**: Changed `API_BASE = '/settings'` → `'/api/settings'`
**File**: `client/src/api/settingsApi.ts:7`
**Issue**: Frontend was calling `/settings/path/{path}`, backend expected `/api/settings/path/{path}`
**Result**: Settings updates now work correctly

---

## ⏳ REMAINING ISSUES

### 3. Analytics Tab - "No Settings Dashboard"
**Status**: NOT YET FIXED
**Priority**: 🟡 IMPORTANT

**Problem**: Analytics tab is empty/shows placeholder

**Files to Check**:
- `client/src/features/visualisation/components/ControlPanel/settingsConfig.ts:103` (Analytics settings commented out)
- `client/src/features/settings/components/panels/SettingsPanelRedesign.tsx:128` (Analytics tab registered)
- `client/src/hooks/useAnalytics.ts` (Analytics hook exists)

**Quick Fix**:
1. Uncomment analytics settings in `settingsConfig.ts`
2. Create analytics settings panel component
3. Wire up to analytics store

**Example Location**:
```typescript
// settingsConfig.ts:103
//   title: 'Analytics Settings',  ← UNCOMMENT THIS SECTION
//   settings: [
//     { key: 'computeMode', label: 'Compute Mode', type: 'select', ... }
//   ]
```

---

### 4. Ontology Mode Toggle Missing
**Status**: NOT YET FIXED
**Priority**: 🟡 IMPORTANT

**Problem**: Can't enable ontology features from UI

**Components Available**:
- ✅ `client/src/features/ontology/components/OntologyModeToggle.tsx`
- ✅ `client/src/features/ontology/components/OntologyPanel.tsx`
- ✅ `client/src/features/ontology/store/useOntologyStore.ts`

**Quick Fix**:
1. Import `OntologyModeToggle` into settings panel or control panel
2. Add to appropriate tab (likely System or Advanced)
3. Example:
```typescript
import { OntologyModeToggle } from '@/features/ontology/components/OntologyModeToggle';

// In settings panel render:
<OntologyModeToggle />
```

---

### 5. Raw Telemetry Window Missing
**Status**: NOT YET FIXED
**Priority**: 🟢 NICE-TO-HAVE

**Problem**: Missing debug/monitoring view for agent telemetry

**Components Available**:
- ✅ `client/src/features/bots/components/AgentTelemetryStream.tsx`
- ✅ `client/src/telemetry/AgentTelemetry.ts`
- ✅ `client/src/telemetry/useTelemetry.ts`

**Quick Fix**:
1. Import `AgentTelemetryStream` into control panel or create dedicated panel
2. Add as collapsible/dockable window
3. Example:
```typescript
import { AgentTelemetryStream } from '@/features/bots/components/AgentTelemetryStream';

// In control panel or as floating panel:
<AgentTelemetryStream />
```

---

## Testing Checklist

### ✅ Completed
- [x] Backend starts without crashes
- [x] Graph data loads (185 nodes, 4006 edges)
- [x] WebSocket connections work
- [x] Settings API saves correctly
- [x] SpacePilot connects successfully

### ⏳ To Test
- [ ] Analytics tab shows meaningful dashboard
- [ ] Ontology mode can be toggled
- [ ] Ontology features activate when enabled
- [ ] Telemetry stream displays agent activity
- [ ] All settings persist correctly

---

## Quick Command Reference

```bash
# Monitor frontend hot-reload
tail -f logs/vite.log | grep "page reload\|hmr update"

# Test settings API manually
curl -X PUT http://localhost:3001/api/settings/path/visualisation.nodes.baseColor \
  -H "Content-Type: application/json" \
  -d '"#ff0000"'

# Check backend logs for API calls
tail -f logs/rust-error.log | grep "Settings Handler"

# Monitor for any new errors
tail -f logs/rust-error.log | grep -i error
```

---

## Summary for New Developer

**What Works Now**:
- ✅ Backend stable (no crashes)
- ✅ Settings API functional
- ✅ Graph visualization working
- ✅ WebSocket connections active

**What Needs Attention**:
1. **Analytics Tab**: Uncomment config, create panel component
2. **Ontology Toggle**: Import and add to UI
3. **Telemetry Window**: Import and add to layout

**Estimated Effort**:
- Analytics: ~30 minutes (uncomment + wire up)
- Ontology: ~15 minutes (import + add to UI)
- Telemetry: ~20 minutes (add to layout + styling)

**Total**: ~1 hour to complete all remaining UI issues
