# WebSocket Service Consolidation Plan

## Current State Analysis

### Overview
The project currently has **4 separate WebSocket implementations**:

1. **WebSocketService.ts** (Legacy)
   - Core WebSocket functionality
   - Binary protocol support (V2, 36 bytes/node)
   - Message queuing and reconnection logic
   - Used by: BotsWebSocketIntegration, AppInitializer, graphDataManager, interactionApi
   - Status: **Active, widely used**

2. **EnhancedWebSocketService.ts**
   - Extends WebSocketService with typed message system
   - Advanced subscription management
   - Comprehensive event handling
   - Statistics and monitoring
   - Status: **Wrapper around legacy service, incomplete migration**

3. **VoiceWebSocketService.ts**
   - Specialized for voice/audio WebSocket
   - Handles transcription results
   - Voice status updates
   - Used by: VoiceButton, VoiceIndicator, useVoiceInteraction
   - Status: **Specialized, may need WebRTC instead**

4. **BotsWebSocketIntegration.ts**
   - Agent swarm-specific WebSocket handling
   - Binary position updates (34-byte format)
   - Contains deprecated polling methods
   - Status: **Partially migrated to REST/WebSocket hybrid**

### Dependency Map

```
WebSocketService (Legacy Base)
    ↓
    ├── EnhancedWebSocketService (Wrapper)
    ├── BotsWebSocketIntegration (Uses base)
    └── VoiceWebSocketService (Independent)
```

**Files Using WebSocketService (14 imports):**
- `AppInitializer.tsx`
- `GraphInteractionTab.tsx`
- `BackendUrlSetting.tsx`
- `graphDataManager.ts`
- `BotsWebSocketIntegration.ts`
- `EnhancedWebSocketService.ts`
- `interactionApi.ts`
- `ConnectionWarning.tsx`
- `useErrorHandler.tsx`
- `useWebSocketErrorHandler.ts`

**Files Using VoiceWebSocketService (5 imports):**
- `VoiceButton.tsx`
- `VoiceIndicator.tsx`
- `VoiceStatusIndicator.tsx`
- `useVoiceInteraction.ts`

**Files Using EnhancedWebSocketService:**
- None currently (not actively used!)

---

## Problems Identified

### 1. EnhancedWebSocketService Not Used
- Created as a wrapper but **never adopted**
- All code still uses legacy WebSocketService
- Duplicates functionality without adding value

### 2. VoiceWebSocketService Architecture Question
- Uses WebSocket for voice/audio data
- **Should use WebRTC instead** (lower latency, better for real-time audio)
- Current WebSocket approach creates throttling issues
- Confusion between voice data stream and voice interaction throttling

### 3. BotsWebSocketIntegration Partial Migration
- Still has deprecated methods in use:
  - `requestInitialData()` (useBotsWebSocketIntegration.ts:54)
  - `sendBotsUpdate()` (programmaticMonitor.ts:22, 183)
  - `restartPolling()` (MultiAgentInitializationPrompt.tsx:121)
- Should handle ONLY binary position updates
- REST polling for metadata handled by BotsDataContext

### 4. Code Duplication
- Reconnection logic duplicated across services
- Event handling patterns inconsistent
- Statistics tracking in multiple places

---

## Consolidation Strategy

### Phase 1: Remove Unused EnhancedWebSocketService ✅
**Action:** Delete EnhancedWebSocketService.ts (not being used)
**Impact:** None - no files import it
**Timeline:** Immediate

### Phase 2: VoiceWebSocketService Architecture Decision
**Options:**

**Option A: Migrate to WebRTC (RECOMMENDED)**
- Replace VoiceWebSocketService with WebRTC data channels
- Lower latency for real-time audio
- Native browser support
- Separate signaling from media

**Option B: Keep as WebSocket Module**
- Refactor as a plugin/module for WebSocketService
- Keep for signaling, use WebRTC for audio data
- Maintain current architecture

**Recommendation:** Option A - Use WebSocket for signaling only, WebRTC for actual audio/voice data

### Phase 3: Consolidate BotsWebSocketIntegration
**Actions:**
1. Remove deprecated methods after refactoring callers:
   - Replace `requestInitialData()` with REST polling
   - Replace `sendBotsUpdate()` with REST API calls
   - Remove `restartPolling()` (handled by BotsDataContext)
2. Keep only binary position update handling
3. Document that metadata comes from REST, positions from WebSocket

### Phase 4: Modularize WebSocketService
**Create Plugin Architecture:**
```typescript
// Core WebSocket service (base)
WebSocketService (singleton)
  ├── BinaryProtocolHandler (plugin for 36-byte position data)
  ├── VoiceSignalingHandler (plugin for voice signaling)
  └── EventStreamHandler (plugin for generic events)
```

**Benefits:**
- Single connection manager
- Pluggable message handlers
- Consistent reconnection/heartbeat logic
- Centralized statistics

---

## Detailed Migration Steps

### Step 1: Delete EnhancedWebSocketService
```bash
rm client/src/services/EnhancedWebSocketService.ts
```

### Step 2: Refactor Voice System
**Create new files:**
- `services/voice/VoiceSignalingService.ts` (WebSocket signaling)
- `services/voice/VoiceRTCService.ts` (WebRTC audio data)

**Update:**
- `VoiceButton.tsx`
- `VoiceIndicator.tsx`
- `useVoiceInteraction.ts`

### Step 3: Refactor BotsWebSocketIntegration Callers
**File: `useBotsWebSocketIntegration.ts:54`**
```typescript
// OLD: botsWebSocketIntegration.requestInitialData()
// NEW: Use BotsDataContext with REST polling (already available)
```

**File: `programmaticMonitor.ts:22, 183`**
```typescript
// OLD: sendBotsUpdate(payload)
// NEW: POST /api/bots/update via REST API
```

**File: `MultiAgentInitializationPrompt.tsx:121`**
```typescript
// OLD: botsWebSocketIntegration.restartPolling()
// NEW: No-op or trigger REST polling via BotsDataContext
```

### Step 4: Create Plugin Architecture
**Pseudocode:**
```typescript
interface WebSocketPlugin {
  name: string;
  handleMessage(data: any): void;
  handleBinary?(data: ArrayBuffer): void;
  onConnect?(): void;
  onDisconnect?(): void;
}

class WebSocketService {
  private plugins: Map<string, WebSocketPlugin> = new Map();

  registerPlugin(plugin: WebSocketPlugin): void {
    this.plugins.set(plugin.name, plugin);
  }

  private routeMessage(message: any): void {
    for (const plugin of this.plugins.values()) {
      plugin.handleMessage(message);
    }
  }
}

// Usage
const binaryProtocolPlugin: WebSocketPlugin = {
  name: 'binary-protocol',
  handleBinary(data: ArrayBuffer) {
    const nodes = parseBinaryNodeData(data);
    // emit to listeners
  }
};

webSocketService.registerPlugin(binaryProtocolPlugin);
```

---

## Risk Assessment

### Low Risk
- ✅ Removing EnhancedWebSocketService (not used)
- ✅ Documenting current architecture
- ✅ Flagging deprecated methods

### Medium Risk
- ⚠️ Refactoring VoiceWebSocketService to WebRTC
- ⚠️ Removing deprecated BotsWebSocketIntegration methods
- ⚠️ Changing caller code patterns

### High Risk
- 🚨 Complete architectural refactor to plugin system
- 🚨 Breaking existing WebSocket message patterns
- 🚨 Changing reconnection behavior

---

## Recommended Approach

### Immediate Actions (Low Risk)
1. ✅ Delete `EnhancedWebSocketService.ts`
2. ✅ Document current architecture in this file
3. ✅ Add deprecation warnings to methods

### Short-term (1-2 weeks)
1. Research WebRTC migration for voice
2. Refactor BotsWebSocketIntegration callers to use REST
3. Remove deprecated methods from BotsWebSocketIntegration
4. Test thoroughly

### Long-term (1-2 months)
1. Design plugin architecture for WebSocketService
2. Migrate existing functionality to plugins
3. Implement comprehensive testing
4. Deploy incrementally with feature flags

---

## Success Criteria

✅ Single WebSocket connection for all features
✅ Voice uses WebRTC for audio data
✅ Clear separation: WebSocket for signaling, REST for metadata, WebSocket binary for positions
✅ No deprecated methods in active code
✅ Comprehensive documentation
✅ Performance improvement (reduced connections)
✅ Code maintainability improved

---

## Next Steps

1. **Approve this plan** with stakeholders
2. **Delete EnhancedWebSocketService.ts** (immediate)
3. **Research WebRTC** for voice system
4. **Create refactoring tickets** for each step
5. **Set timeline** for implementation
