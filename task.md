# AR-AI-Knowledge-Graph Master Task List

> **Last Updated:** 2026-01-19T18:30:00Z
> **Codebase:** 525,379 LOC | 674,740 total lines | 1,079 analyzed files
> **Architecture:** CQRS + Hexagonal + Actor Model + GPU Compute
> **Test Status:** 77/81 passing (95.1%)
> **Quality Tool:** Qlty CLI v0.606.0 (15 plugins enabled)
> **Sprint Protocol:** AISP 5.1 Platinum + Claude-Flow v3 Hive Mind

---

## ⚠️ ACTIVE SPRINT: Edge Visibility Crisis

### 🚨 P0 - CRITICAL: Graph Edges NOT Rendering

**Status:** 🟡 ROOT CAUSE IDENTIFIED | **Severity:** CRITICAL | **Sprint:** 2026-01-19
**Symbol:** ∎ EDGE_VIZ_001

#### ✅ ROOT CAUSE IDENTIFIED (2026-01-19 Claude-Flow v3 Investigation)

**Primary Issue: setState Called in useFrame (Anti-Pattern)**
- **File:** `GraphManager.tsx:705-714`
- **Line 713:** `setLabelPositions(newLabelPositions)` inside `useFrame` hook
- **Effect:** React re-renders every frame → "Maximum update depth exceeded"

**Secondary Issues:**
1. **Stale edge data** (line 694): Only checks array LENGTH not content
2. **Edge opacity too low** (0.2): Below bloom luminanceThreshold (0.1)
3. **Material memoization unstable**: `edgeBloomStrength` from store causes recreation

#### Investigation Evidence
| Finding | Location | Agent |
|---------|----------|-------|
| setState in useFrame | `GraphManager.tsx:705-714` | a1068b2 |
| labelPositions loop | `GraphManager.tsx:877-889` | a1068b2 |
| Edge length-only check | `GraphManager.tsx:694` | a1068b2 |
| Actix mailbox overflow | `socket_flow_handler.rs:74,149,151` | afbe7f9 |
| LOD not connected | `CameraController.tsx` → LOD | a95c2bc |
| Aggressive fog (6-34 units) | `HolographicDataSphere.tsx:30-34` | a95c2bc |

#### Previous Attempted Fixes (2026-01-13)
| Fix | File | Status | Note |
|-----|------|--------|------|
| Bloom boost formula | `FlowingEdges.tsx:102` | ✅ DEPLOYED | Helpful but not root cause |
| Brighter edge color | `settingsApi.ts:216` | ✅ DEPLOYED | Cosmetic fix |
| State update throttle | `GraphManager.tsx:692-695` | ⚠️ INCOMPLETE | Checks length only |
| Opacity increase | `settingsApi.ts:218` | ✅ DEPLOYED | Still below threshold |

**Result:** Root cause is React state management, not visual settings.

---

### 🔧 DEVELOPMENT WORKFLOW

**CRITICAL: Follow this workflow exactly.**

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL DEVELOPMENT (this container)                             │
│  Path: /home/devuser/workspace/project/client/src               │
│  ↓ Edit files here - mounted into host at:                      │
│  /mnt/mldata/githubs/AR-AI-Knowledge-Graph/client/src           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Vite HMR auto-reload
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  HOST DOCKER (192.168.0.51)                                     │
│  Container: visionflow_container                                │
│  Frontend: http://192.168.0.51:3001                             │
│  SSH only for: restart, logs, health checks                     │
└─────────────────────────────────────────────────────────────────┘
```

**Commands for remote operations ONLY:**
```bash
# Restart vite (if HMR fails)
ssh machinelearn@192.168.0.51 "docker exec visionflow_container pkill -f vite; \
  docker exec -d visionflow_container bash -c 'cd /app/client && npm run dev'"

# Check rust-backend health
ssh machinelearn@192.168.0.51 "curl -s http://localhost:3001/api/health | jq"

# View container logs
ssh machinelearn@192.168.0.51 "docker logs visionflow_container --tail 50"

# Restart rust-backend (if crash-looping)
ssh machinelearn@192.168.0.51 "docker exec visionflow_container supervisorctl restart rust-backend"
```

---

### 🔍 DEBUGGING TOOLKIT

| Tool | Purpose | Command/Access |
|------|---------|----------------|
| **Chrome DevTools** | Console errors, Network, Performance | F12 on http://192.168.0.51:3001 |
| **Playwright MCP** | Automated browser testing | `mcp__playwright__*` tools |
| **Screenshots** | Visual verification | Display :1 (VNC port 5901) |
| **React DevTools** | Component tree, state inspection | Chrome extension |
| **THREE.js Inspector** | Scene graph, materials, geometry | Chrome extension |

**Display :1 Access:**
```bash
# VNC to container display
vncviewer localhost:5901  # password: turboflow

# Take screenshot via Playwright
mcp__playwright__screenshot --name "edge_debug" --fullPage true
```

---

### 📋 SPRINT TODO LIST (Updated 2026-01-19)

**Protocol:** AISP 5.1 Platinum | **Topology:** Hive-Mind Mesh

#### ✅ COMPLETED: Quality Gate Filtering (2026-01-19)

| # | Task | File:Line | Status |
|---|------|-----------|--------|
| ✓ | **Implement maxNodeCount filtering** - Filter nodes by authority+quality score | `graphDataManager.ts:298-336` | ✅ DONE |
| ✓ | **Connect QualityGatePanel to store** - Sync maxNodeCount to Zustand store | `QualityGatePanel.tsx:43,62` | ✅ DONE |
| ✓ | **Adjust slider range** - min=100, max=10000, step=100 | `QualityGatePanel.tsx:157-160` | ✅ DONE |

**Result:** Graph filtering now controlled by Quality Gate settings. Default: 1000 nodes if not set.

#### 🔴 P0 - CRITICAL (Fix Now)

| # | Task | File:Line | Status |
|---|------|-----------|--------|
| 1 | **Remove setState from useFrame** - Move `setLabelPositions` outside render loop | `GraphManager.tsx:705-714` | ⬜ TODO |
| 2 | **Fix edge content comparison** - Compare actual values, not just length | `GraphManager.tsx:694` | ⬜ TODO |
| 3 | **Remove labelPositions from useEffect deps** - Use ref instead | `GraphManager.tsx:877-889` | ⬜ TODO |
| 4 | **Increase edge opacity** - Set to 0.6+ to exceed bloom threshold | `settingsApi.ts:218` | ✅ DONE (set to 0.6) |

#### 🟠 P1 - HIGH (Fix This Sprint)

| # | Task | File:Line | Status |
|---|------|-----------|--------|
| 5 | Fix Actix mailbox backpressure - Add bounded queue | `socket_flow_handler.rs:74,149,151` | ⬜ TODO |
| 6 | Implement BroadcastAck flow control on server | `socket_flow_handler.rs:1443-1484` | ⬜ TODO |
| 7 | Connect CameraController to LOD system | `CameraController.tsx` | ⬜ TODO |
| 8 | Reduce aggressive fog - Increase fogNear from 6 to 30+ | `HolographicDataSphere.tsx:30-34` | ⬜ TODO |
| 9 | Add session state recovery on WebSocket reconnect | `socket_flow_handler.rs:1717-1742` | ⬜ TODO |

#### 🟡 P2 - MEDIUM (Backlog)

| # | Task | File:Line | Status |
|---|------|-----------|--------|
| 10 | Stabilize material memoization - extract bloom strength | `FlowingEdges.tsx:98-118` | ⬜ TODO |
| 11 | Implement desktop LOD system | `graphOptimizations.ts` | ⬜ TODO |
| 12 | Screenshot verification via Playwright | MCP tools | ⬜ TODO |
| 13 | Test with hardcoded 2-point edge | Test file | ⬜ TODO |

#### ✅ COMPLETED (Investigation Phase)

| # | Task | Agent | Date |
|---|------|-------|------|
| ✓ | Investigate edge rendering system | a1068b2 | 2026-01-19 |
| ✓ | Investigate WebSocket connection architecture | afbe7f9 | 2026-01-19 |
| ✓ | Investigate label visibility/camera system | a95c2bc | 2026-01-19 |
| ✓ | webrtc.rs evaluation | researcher | 2026-01-19 |

---

### 📊 INVESTIGATION RESULTS SUMMARY (2026-01-19)

#### Edge Rendering System (Agent a1068b2)
```
ROOT CAUSE: setState in useFrame hook
├── GraphManager.tsx:713 → setLabelPositions() in useFrame
├── GraphManager.tsx:889 → dependencies include nodePositionsRef.current
├── GraphManager.tsx:1048 → labelPositions + camera.position in deps
└── Result: 1000s of JSX objects created per frame → Maximum update depth exceeded

SECONDARY: Edge data staleness
├── GraphManager.tsx:694 → if (newEdgePoints.length !== prev)
├── Only checks LENGTH, not content values
└── Edge positions change every frame but aren't updated
```

#### WebSocket Connection (Agent afbe7f9)
```
CRITICAL: Silent message drops
├── Actix mailbox: 16-message default, no monitoring
├── socket_flow_handler.rs:74,149,151 → ctx.text()/ctx.binary() overflow
├── BroadcastAck (0x34) received but server IGNORES
└── Result: Clients miss position updates with no indication

RECONNECTION: State lost
├── X-Client-Session header detected but not used
├── No message queue playback for reconnecting clients
└── Must re-request via REST API
```

#### Label Visibility/LOD (Agent a95c2bc)
```
DESKTOP LOD: Not connected
├── CameraController.tsx → sets position only
├── LODManager/FrustumCuller in graphOptimizations.ts → UNUSED
└── All desktop nodes render at full complexity

FOG: Too aggressive
├── HolographicDataSphere.tsx → fogNear: 6, fogFar: 34
├── Camera at [20, 15, 20]
└── Most content fogged out
```

---

### 🔄 webrtc.rs Evaluation

**Status:** ❌ NOT RECOMMENDED

**Blog Post:** https://webrtc.rs/blog/2026/01/18/rtc-feature-complete-whats-next.html

**Findings:**
- No benchmarks provided
- No WebSocket comparison data
- No clear performance benefit demonstrated
- Would require significant rewrite of binary protocol

**Current System:**
- Binary Protocol V3: 48 bytes/node, well-designed
- Actix WebSocket: Stable, proven
- Issue is **backpressure**, not protocol

**Recommendation:** Fix Actix mailbox backpressure instead of replacing transport layer.

---

### 🤖 MULTI-MODEL CONSULTATION (Deprioritized)

Research phase complete. Focus now on implementing fixes.

```yaml
# Preserved for future reference if fixes don't resolve
consultation_targets:
  - model: openai-5.2-codex
    query: "React Three Fiber LineSegments not rendering despite valid geometry buffer"
  - model: gemini-3-pro
    query: "THREE.js LineSegments invisible when using selective bloom EffectComposer"
  - model: z-ai
    query: "Debug WebGL edge rendering in CQRS actor-based visualization system"
```

---

### 🐝 HIVE-MIND SWARM CONFIGURATION

```yaml
swarm:
  topology: hierarchical-mesh
  queen: opus-coordinator
  workers:
    - type: qe-visual-tester
      count: 2
      task: screenshot-analysis
    - type: qe-integration-tester
      count: 2
      task: devtools-capture
    - type: researcher
      count: 3
      task: multi-model-consultation
    - type: coder
      count: 2
      task: geometry-inspection
  consensus: proof-of-verification
  memory_namespace: aqe/edge-viz/*
```

**Initialize swarm:**
```bash
npx claude-flow@v3alpha swarm init --topology hierarchical-mesh --max-agents 12
```

---

## Current Quality Metrics

| Metric | Value |
|--------|-------|
| Lines of Code | 525,379 |
| Classes | 6,049 |
| Functions | 21,627 |
| Cyclomatic Complexity | 80,480 |
| Cognitive Complexity | 70,111 |
| Unformatted Files | 909 |
| Auto-fixable Issues | 52+ |

---

## P0 - CRITICAL: Code Formatting

### Issue: 909 Unformatted Files
**Status:** NOT STARTED | **Severity:** High | **Effort:** Low (auto-fixable)

The codebase has 909 files that don't meet formatting standards. This affects code review, maintainability, and consistency.

**Affected Areas:**
- `client/src/**/*.tsx` - React components
- `client/src/**/*.ts` - TypeScript services/hooks
- `docs/**/*.md` - Documentation
- `multi-agent-docker/skills/**/*` - Skill definitions
- `scripts/**/*` - Build/utility scripts

**Fix Command:**
```bash
~/.qlty/bin/qlty fmt --all
```

---

## P1 - HIGH: Complexity Hotspots

### Issue: AppInitializer.tsx Extreme Complexity (196)
**Status:** NOT STARTED | **Severity:** High | **Effort:** High
**Location:** `client/src/app/AppInitializer.tsx`

| Function | Complexity |
|----------|------------|
| `AppInitializer` (component) | 186 |
| `initializeWebSocket` | 93 |
| `initApp` | 44 |

**Problems:**
- Deeply nested control flow (level 5+)
- Single component handles too many responsibilities
- Difficult to test and maintain

**Recommended Refactor:**
- [ ] Extract WebSocket initialization to dedicated hook
- [ ] Extract service loading to separate module
- [ ] Break down into smaller, focused components
- [ ] Add error boundary wrappers

---

### Issue: Control Center Components High Complexity
**Status:** NOT STARTED | **Severity:** High | **Effort:** Medium

| File | Complexity | Returns |
|------|------------|---------|
| `SettingsPanel.tsx` | 91 | 8 |
| `ConstraintPanel.tsx` | 88 | - |
| `ProfileManager.tsx` | 77 | 6 |
| `ControlCenter.tsx` | 27 | 6 |

**Location:** `client/src/components/ControlCenter/`

**Problems:**
- Functions with many return statements (6-8)
- High cognitive complexity
- Duplicated update patterns (26+ line blocks)

**Recommended Refactor:**
- [ ] Extract shared update logic to custom hooks
- [ ] Split large components into sub-components
- [ ] Use composition over monolithic components
- [ ] Standardize loading/error state handling

---

### Issue: API Layer Duplication
**Status:** NOT STARTED | **Severity:** Medium | **Effort:** Medium

| File | Duplicated Lines | Locations |
|------|------------------|-----------|
| `analyticsApi.ts` | 18 lines | 2 |
| `settingsApi.ts` | 41 lines | 2 |
| `exportApi.ts` | 19 lines | 2 |

**Location:** `client/src/api/`

**Problems:**
- Repeated API response handling patterns
- Identical error handling blocks
- Copy-paste code for similar endpoints

**Recommended Refactor:**
- [ ] Create generic `apiRequest<T>()` utility
- [ ] Extract common response transformation
- [ ] Standardize error handling with shared wrapper
- [ ] Use TypeScript generics for type-safe patterns

---

## P2 - MEDIUM: Linting Issues

### Issue: Unused Imports (Python Skills)
**Status:** NOT STARTED | **Severity:** Low | **Effort:** Low (auto-fixable)

**Affected Files:**
| File | Unused Imports |
|------|----------------|
| `docs/scripts/diataxis-phase3-frontmatter.py` | `os`, `datetime` |
| `multi-agent-docker/skills/blender/addon/tools/animation.py` | `typing.List` |
| `multi-agent-docker/skills/blender/addon/tools/modifiers.py` | `typing.Optional` |
| `multi-agent-docker/skills/docs-alignment/scripts/scan_stubs.py` | `typing.Set`, `typing.Tuple` |
| `multi-agent-docker/skills/docs-alignment/scripts/validate_links.py` | `os` |
| `multi-agent-docker/skills/imagemagick/mcp-server/server.py` | `os` |
| `multi-agent-docker/skills/slack-gif-creator/core/color_palettes.py` | `typing.Optional` |
| `multi-agent-docker/skills/slack-gif-creator/core/frame_composer.py` | `numpy` |
| `multi-agent-docker/skills/text-processing/server.py` | `subprocess`, `json`, `Path`, `List` |

**Fix Command:**
```bash
~/.qlty/bin/qlty check --fix
```

---

### Issue: F-strings Without Placeholders
**Status:** NOT STARTED | **Severity:** Low | **Effort:** Low (auto-fixable)

**Affected Files:**
- `docs/scripts/diataxis-phase3-frontmatter.py:131`
- `multi-agent-docker/skills/docs-alignment/scripts/scan_stubs.py:281`

**Problem:** Using f-strings where regular strings would suffice.

---

## P2 - MEDIUM: Code Smells

### Issue: Test Code Duplication
**Status:** NOT STARTED | **Severity:** Low | **Effort:** Medium

| Location | Duplicated Lines |
|----------|------------------|
| `tests/inference/classification_tests.rs` | 40 lines |
| `tests/inference/explanation_tests.rs` | 40 lines |
| `tests/api/cqrs_integration_tests.rs` | 17 lines (2x) |

**Recommended Refactor:**
- [ ] Extract shared test ontology setup to test utilities
- [ ] Create test fixtures for common scenarios
- [ ] Use parameterized tests where applicable

---

### Issue: Deep Nesting in Test Parsing
**Status:** NOT STARTED | **Severity:** Low | **Effort:** Low
**Location:** `tests/mcp_parsing_tests.rs:205-207`

Control flow nested 5 levels deep. Consider early returns or extraction.

---

## P3 - LOW: In-Progress Items (Carry Forward)

### Item 10: Legacy CUDA Kernel Cleanup
**Status:** IN PROGRESS | **Severity:** Low | **Effort:** Medium

| Sub-task | Status |
|----------|--------|
| Remove hardcoded switch/case | PARTIAL |
| Increase MAX_RELATIONSHIP_TYPES | NOT STARTED |
| Device-side dynamic allocation | NOT STARTED |

**Files:**
- `src/utils/semantic_forces.cu:70`
- `src/gpu/semantic_forces.rs:878-940`

---

### Item 11: True Client ACK for Backpressure
**Status:** IN PROGRESS | **Severity:** Low | **Effort:** Medium

| Sub-task | Status |
|----------|--------|
| Application-level ACK types | DONE |
| End-to-end delivery confirmation | PARTIAL |
| Integrate fastwebsockets ACK flow | NOT STARTED |

---

## P3 - LOW: JSS Integration Roadmap

### Phase 2: Multi-User Pods (40% Complete)
| Item | Status |
|------|--------|
| `/pods/{npub}/` URL structure | DOCUMENTED |
| Auto-provision pods on login | DOCUMENTED |
| Nostr -> WebID mapping | DOCUMENTED |
| Actual pod provisioning code | NOT IMPLEMENTED |

### Phase 3: User Ontology Ownership
- [ ] Personal ontology fragments in pods
- [ ] Proposal/merge workflow
- [ ] Reverse sync to GitHub

### Phase 4: Frontend Pod UI
- [ ] Create `SolidPodService.ts`
- [ ] Pod browser component
- [ ] Contribution/proposal UI

### Phase 5: Agent Memory
- [ ] Per-agent pods (54 agent types)
- [ ] Claude-flow hooks for JSS
- [ ] Migrate `.agentdb` to pods

---

## Archived: Completed Items (2026-01-12)

<details>
<summary>Click to expand completed heroic refactor sprint</summary>

### Heroic Refactor Sprint (2026-01-08 - 2026-01-12)

**All 8 Critical Issues: RESOLVED**

| Issue | Resolution |
|-------|------------|
| 1. GPU Concurrency | Uses `spawn_blocking()` correctly |
| 2. Session Persistence | Redis implementation complete |
| 3. FFI Struct Alignment | 14 `const_assert_eq!` in place |
| 4. Data Sync Race | ON MATCH excludes sim_* |
| 5. Singleton Pattern | Added `resetInstance()` |
| 6. NIP-98 Token Window | Already 300s (5 min) |
| 7. Worker Silent Failure | Connected to WorkerErrorModal |
| 8. Hardcoded Secret | Removed insecure fallback |

**Quality Gate Progress:**
| Wave | Score |
|------|-------|
| Baseline | 52/100 |
| Final | 75/100 |

**Additional Completed:**
- Jest → Vitest migration
- Binary Protocol V2 Unification
- Parallel Ontology Processing
- Agent Visualization Feature (AGENT_ACTION 0x23)
- Neo4j adapter tests (49 passing)
- unwrap()/expect() cleanup (368 remaining, below 400 target)

</details>

---

## Verification Commands

```bash
# Run full quality check
~/.qlty/bin/qlty check --all

# Auto-format all files
~/.qlty/bin/qlty fmt --all

# View code metrics
~/.qlty/bin/qlty metrics --max-depth=2 --sort complexity --all

# Check code smells
~/.qlty/bin/qlty smells --all

# Sample lint issues
~/.qlty/bin/qlty check --sample=30
```

---

## Priority Execution Order

| Priority | Issue | Auto-fixable | Estimated Effort |
|----------|-------|--------------|------------------|
| P0 | Format 909 files | YES | 5 min |
| P1 | Refactor AppInitializer.tsx | NO | 4-8 hours |
| P1 | Refactor Control Center | NO | 4-6 hours |
| P1 | Extract API utilities | NO | 2-4 hours |
| P2 | Fix unused imports | YES | 5 min |
| P2 | Fix f-string issues | YES | 5 min |
| P2 | DRY test utilities | NO | 2-4 hours |
| P3 | CUDA kernel cleanup | NO | 4-8 hours |
| P3 | Client ACK integration | NO | 4-6 hours |
| P3 | JSS Phases 2-5 | NO | Multi-sprint |

---

> **Next Step:** Run `~/.qlty/bin/qlty fmt --all` to resolve P0 formatting issues first.
