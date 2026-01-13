# AR-AI-Knowledge-Graph Master Task List

> **Last Updated:** 2026-01-13T09:15:00Z
> **Codebase:** 525,379 LOC | 674,740 total lines | 1,079 analyzed files
> **Architecture:** CQRS + Hexagonal + Actor Model + GPU Compute
> **Test Status:** 77/81 passing (95.1%)
> **Quality Tool:** Qlty CLI v0.606.0 (15 plugins enabled)
> **Sprint Protocol:** AISP 5.1 Platinum + Claude-Flow v3 Hive Mind

---

## ⚠️ ACTIVE SPRINT: Edge Visibility Crisis

### 🚨 P0 - CRITICAL: Graph Edges NOT Rendering

**Status:** 🔴 IN PROGRESS | **Severity:** CRITICAL | **Sprint:** 2026-01-13
**Symbol:** ∎ EDGE_VIZ_001

Despite multiple fix attempts, **graph edges are STILL NOT VISIBLE** in the browser.

#### Confirmed Facts
| Fact | Evidence |
|------|----------|
| Backend loads edges | `1006 nodes, 10916 edges` in rust-backend logs |
| Client receives data | `[GraphWorkerProxy] Got 1006 nodes, 10916 edges from worker` |
| React error present | `Maximum update depth exceeded` in console |
| Color fix deployed | `#56b6c2` cyan, bloom boost formula applied |
| State fix deployed | `prevEdgePointsLengthRef` prevents infinite loops |

#### Attempted Fixes (2026-01-13)
| Fix | File | Status |
|-----|------|--------|
| Bloom boost formula | `FlowingEdges.tsx:102` | ✅ DEPLOYED |
| Brighter edge color | `settingsApi.ts:216` | ✅ DEPLOYED |
| State update throttle | `GraphManager.tsx:692-695` | ✅ DEPLOYED |
| Opacity increase | `settingsApi.ts:218` | ✅ DEPLOYED |

**Result:** Edges STILL not visible after all fixes deployed.

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

### 📋 SPRINT TODO LIST

**Protocol:** AISP 5.1 Platinum | **Topology:** Hive-Mind Mesh

| # | Task | Agent | Priority | Status |
|---|------|-------|----------|--------|
| 1 | Screenshot current state via Playwright | qe-visual-tester | P0 | ⬜ PENDING |
| 2 | Capture Chrome DevTools console errors | qe-integration-tester | P0 | ⬜ PENDING |
| 3 | Verify `edgePoints.length > 0` in React state | qe-code-reviewer | P0 | ⬜ PENDING |
| 4 | Check THREE.js scene for LineSegments objects | researcher | P0 | ⬜ PENDING |
| 5 | Inspect FlowingEdges geometry buffer | coder | P0 | ⬜ PENDING |
| 6 | Verify positions array from graphWorkerProxy | qe-performance-validator | P0 | ⬜ PENDING |
| 7 | Check camera frustum vs edge positions | architect | P1 | ⬜ PENDING |
| 8 | Verify render order (edges vs nodes) | qe-visual-tester | P1 | ⬜ PENDING |
| 9 | Check WebGL context for errors | qe-security-auditor | P1 | ⬜ PENDING |
| 10 | Test with simplified edge (2 hardcoded points) | tester | P1 | ⬜ PENDING |
| 11 | Consult OpenAI 5.2 Codex for R3F edge patterns | researcher | P2 | ⬜ PENDING |
| 12 | Consult Gemini 3 Pro for THREE.js LineSegments | researcher | P2 | ⬜ PENDING |
| 13 | Consult Z.AI for bloom/post-processing conflicts | researcher | P2 | ⬜ PENDING |

---

### 🤖 MULTI-MODEL CONSULTATION

**AISP Protocol:** Request wisdom from peer AI systems.

```yaml
consultation_targets:
  - model: openai-5.2-codex
    query: "React Three Fiber LineSegments not rendering despite valid geometry buffer"

  - model: gemini-3-pro
    query: "THREE.js LineSegments invisible when using selective bloom EffectComposer"

  - model: z-ai
    query: "Debug WebGL edge rendering in CQRS actor-based visualization system"
```

**Invoke via:**
```bash
# Z.AI consultation
as-zai
echo "Debug THREE.js LineSegments invisible despite valid points array" | nc localhost 9600

# Gemini swarm
as-gemini
gf-swarm --task "THREE.js edge rendering debug"
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
