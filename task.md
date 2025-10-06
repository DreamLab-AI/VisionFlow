# Technical Debt Cleanup - Task Status

## ✅ Completed Tasks

### Development Artifacts Removed
- ✅ Removed backup file: `client/src/features/visualisation/components/IntegratedControlPanel.tsx.backup`
- ✅ Removed mock data files: `mockAgentData.ts`, `mockDataAdapter.ts`
- ✅ Removed test component: `GraphCanvasTestMode.tsx` and updated `GraphCanvasWrapper.tsx`
- ✅ Removed console.log statements from `client/vite.config.ts`

### Authentication Cleanup
- ✅ Removed unused `AuthenticationManager` from `client/src/features/auth/initializeAuthentication.ts`
- ✅ System now uses Nostr authentication exclusively via `authInterceptor.ts`

### Security & UX Improvements
- ✅ Replaced iframe system in `NarrativeGoldminePanel.tsx` with new tab opening (security improvement)
- ✅ Removed `client/src/utils/iframeCommunication.ts` utility
- ✅ Fixed broken imports in `useGraphEventHandlers.ts` and `GraphManager_EventHandlers.ts`
- ✅ Updated Narrative Goldmine URLs to use `#/page/` with proper `encodeURIComponent()`

### Protocol Cleanup
- ✅ Removed legacy PROTOCOL_V1 support from `client/src/types/binaryProtocol.ts`
- ✅ Now only supports V2 protocol (36 bytes per node with u32 IDs)

### Code Quality Improvements
- ✅ Flagged all magic numbers in `HolographicDataSphere.tsx` with TODO comments for settings mapping
- ✅ Flagged magic numbers in `BotsVisualizationFixed.tsx` (lerp factors, pulse speeds, opacity values)
- ✅ Updated `polling-system.md` to reflect current hybrid REST/WebSocket architecture
- ✅ Documented deprecated methods and legacy code removal

### Validation
- ✅ Ran `cargo check` - passes with only warnings (no errors)
- ✅ Fixed Vite build errors from removed iframeCommunication module

---

## 🔄 Remaining Tasks

### 1. Architectural Anomalies & Duplicate Systems

**Multiple WebSocket Services:**
- `client/src/services/WebSocketService.ts` - Legacy service, needs careful dependency analysis before removal
- `client/src/services/EnhancedWebSocketService.ts` - Newer implementation, migration incomplete
- `client/src/services/VoiceWebSocketService.ts` - Specialized for audio, needs architecture review
- `client/src/features/bots/services/BotsWebSocketIntegration.ts` - Contains deprecated methods still in use

**Recommendation:** Consolidate into a single, modular WebSocket service (like EnhancedWebSocketService) that can be extended with plugins or handlers for different features (bots, voice, etc.).

**Deprecated Methods (Still In Use):**
The following deprecated methods in `BotsWebSocketIntegration.ts` are still being called and cannot be removed yet:
- `requestInitialData()` - used in `useBotsWebSocketIntegration.ts:54`
- `sendBotsUpdate()` - used in `programmaticMonitor.ts:22, 183`
- `restartPolling()` - used in `MultiAgentInitializationPrompt.tsx:121`

**Action Required:** Refactor calling code to use new patterns (REST API via BotsDataContext), then remove these methods.

---

### 2. Vircadia Multi-User XR Integration

**Status:** Needs investigation and planning

The `client/src/services/vircadia/` directory contains a complete, parallel system for multi-user XR functionality:
- `VircadiaClientCore.ts` - Client core
- `EntitySyncManager.ts` - Entity management
- `AvatarManager.ts` - Avatar system

**Integration unclear** with primary "Bots" and "Graph" visualization systems.

**Action Required:**
- Research Vircadia architecture and capabilities (web search required)
- Map integration points with existing Bots and Graph systems
- Document findings in relevant docs section
- Create detailed integration plan
- Test with running Vircadia dockers on ragflow network

---

### 3. Hardcoded Values & Magic Numbers

**File: `client/src/features/bots/components/BotsVisualizationFixed.tsx`**
- Contains numerous magic numbers for scaling, animation speeds, and color calculations
- Examples: `lerpFactor = 0.15`, `pulseSpeed = 2 * tokenMultiplier * healthMultiplier`
- **Action:** Flag in code for future refactoring, extract to configuration

**File: `client/src/features/visualisation/components/HolographicDataSphere.tsx`**
- Filled with hardcoded numeric values for geometry, colors, opacity, and animation parameters
- **Action:** Map each hardcoded value to corresponding setting in settings system
- **Note:** Settings management system is brittle - don't update UX names yet

---

### 4. Critical Issues

**Disabled Testing Framework:**
- **File:** `client/package.json`
- **Issue:** All test scripts disabled due to supply chain attack concerns
- **Status:** ⚠️ HIGHEST PRIORITY - Security and quality assurance issue
- **Action Required:**
  - Address underlying security concerns
  - Update dependencies
  - Re-enable testing framework
  - Remove `block-test-packages.cjs` workaround

---

### 5. Documentation Updates

**File: `client/src/features/bots/docs/polling-system.md`**
- **Issue:** Describes REST polling system but architecture has evolved to hybrid REST/WebSocket model
- **Action:** Update documentation to reflect current state
- Remove references to legacy code patterns

---

## Notes

- Cargo check passes successfully (only warnings, no errors)
- All client-side TypeScript changes made without breaking imports
- Deprecated methods kept with warnings where still actively used
- Security improvements implemented (iframe → new tab)
- Legacy protocol support removed (V1 → V2 only)
