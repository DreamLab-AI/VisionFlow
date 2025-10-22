# UI Issues - Current Status and Fixes

**Date**: 2025-10-22
**Status**: Backend ✅ WORKING | Frontend ❌ PARTIAL ISSUES

## ✅ Good News
- **Backend is stable** - No more SIGSEGV crashes!
- **Graph loads successfully** - 185 nodes, 4006 edges loaded
- **WebSocket connected** - SpacePilot showing "Connected - Hybrid control mode active"
- **Network fix working** - Backend running on `0.0.0.0:8080`

## ❌ Current UI Issues

### 1. Settings API Batch Updates Failing

**Error**:
```
Error: X out of X individual updates failed
```

**Location**: `client/src/api/settingsApi.ts:175`

**Problem**:
```typescript
// Line 175 - Sends value as raw body
await unifiedApiClient.putData(`${API_BASE}/path/${encodedPath}`, value);
```

The backend endpoint `/api/user-settings/path/{path}` might expect:
- Different body format
- Wrapped value object: `{ value: ... }`
- JSON content type issue

**Investigation Needed**:
- Check backend endpoint: `src/handlers/settings_handler.rs`
- Verify expected request format
- Check if endpoint exists and is properly registered

---

### 2. Analytics Tab - "No Settings Dashboard"

**Location**:
- Tab defined: `client/src/features/settings/components/panels/SettingsPanelRedesign.tsx:128`
- Config: `client/src/features/visualisation/components/ControlPanel/settingsConfig.ts:103` (commented out)

**Problem**:
```typescript
// settingsConfig.ts:103 - Analytics settings are commented out!
//   title: 'Analytics Settings',
```

**Files Involved**:
- `client/src/features/settings/components/panels/SettingsPanelRedesign.tsx`
- `client/src/features/visualisation/components/ControlPanel/settingsConfig.ts`
- `client/src/hooks/useAnalytics.ts`

**Fix Required**:
1. Uncomment Analytics settings in `settingsConfig.ts`
2. Add Analytics settings panel content
3. Wire up to Analytics store/hooks

---

### 3. Ontology Mode Toggle Missing

**Components Found**:
- `client/src/features/ontology/components/OntologyModeToggle.tsx` ✅ EXISTS
- `client/src/features/ontology/components/OntologyPanel.tsx` ✅ EXISTS
- `client/src/features/ontology/store/useOntologyStore.ts` ✅ EXISTS

**Problem**: Component exists but not integrated into UI

**Likely Location**: Should be in:
- Settings panel (System/Advanced tab)
- Control panel (as a toggle button)
- Physics controls (as an option)

**Fix Required**:
1. Find where to add ontology toggle in UI
2. Add import and component to appropriate panel
3. Wire up to ontology store

---

### 4. Raw Telemetry Window Missing

**Components Found**:
- `client/src/features/bots/components/AgentTelemetryStream.tsx` ✅ EXISTS
- `client/src/telemetry/AgentTelemetry.ts` ✅ EXISTS
- `client/src/telemetry/useTelemetry.ts` ✅ EXISTS

**Problem**: Telemetry stream component exists but not displayed in control center

**Likely Removed From**:
- Control panel tabs
- Side panel
- Bots/Agents visualization area

**Fix Required**:
1. Find where telemetry window should be displayed
2. Add AgentTelemetryStream component back to layout
3. Ensure telemetry data is flowing

---

## Investigation Priority

### 🔴 CRITICAL - Settings API Errors
**Impact**: Users can't save any settings
**Action**:
1. Check backend endpoint implementation
2. Verify request/response format
3. Fix API call or backend handler

### 🟡 IMPORTANT - Analytics Dashboard
**Impact**: Analytics tab is empty
**Action**:
1. Uncomment analytics settings configuration
2. Create analytics panel component
3. Wire up analytics hooks

### 🟡 IMPORTANT - Ontology Toggle
**Impact**: Can't enable ontology features
**Action**:
1. Add OntologyModeToggle to settings UI
2. Document where it should appear

### 🟢 NICE-TO-HAVE - Telemetry Window
**Impact**: Missing debug/monitoring view
**Action**:
1. Add AgentTelemetryStream back to layout
2. Make it toggleable/dockable

---

## Quick Diagnostic Commands

```bash
# Check backend settings endpoint
grep -rn "user-settings/path" src/handlers/

# Check frontend settings config
cat client/src/features/visualisation/components/ControlPanel/settingsConfig.ts | grep -A 20 "Analytics"

# Check ontology components
ls -la client/src/features/ontology/components/

# Check telemetry integration
grep -rn "AgentTelemetryStream" client/src/
```

---

## Next Steps for Developer

1. **Investigate settings API error** (most critical)
   - Check backend logs for 400/500 errors
   - Test API endpoint manually with curl
   - Fix request format mismatch

2. **Restore Analytics dashboard**
   - Uncomment analytics config
   - Create panel component
   - Test analytics features

3. **Add Ontology toggle**
   - Find appropriate UI location
   - Import and add component
   - Test ontology mode switching

4. **Restore Telemetry window**
   - Add to control panel or side panel
   - Make it collapsible/dockable
   - Test telemetry stream

---

## Success Criteria

- ✅ Settings updates save without errors
- ✅ Analytics tab shows meaningful dashboard
- ✅ Ontology mode can be toggled from UI
- ✅ Telemetry window displays agent activity
