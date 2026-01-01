# AR-AI-Knowledge-Graph Hive-Mind Task List

> **Last Verified:** 2026-01-01 via multi-agent swarm verification
> **Codebase:** 200k+ lines | Rust (382 files) | TypeScript (363 files)
> **Architecture:** CQRS + Hexagonal + Actor Model + GPU Compute

---

## 🔴 CRITICAL ISSUES - VERIFIED PRESENT

### Issue 1: Concurrency & Blocking Hazards
**Status:** ❌ PRESENT | **Severity:** Critical | **Priority:** P0

| Location | Line | Problem |
|----------|------|---------|
| `src/actors/gpu/shared.rs` | 102-103 | `Arc<std::sync::Mutex<UnifiedGPUCompute>>` |
| `src/actors/gpu/force_compute_actor.rs` | 525 | `.lock()` inside `async move` block |
| `src/actors/gpu/force_compute_actor.rs` | 394 | `.lock()` in `apply_ontology_forces()` |
| `src/actors/gpu/force_compute_actor.rs` | 779 | `.lock()` in `UploadPositions` handler |

**Impact:** Tokio thread starvation, heartbeat timeouts, WebSocket frame drops at high FPS

**Fix Options:**
- [ ] Replace `std::sync::Mutex<UnifiedGPUCompute>` with `tokio::sync::Mutex`
- [ ] OR wrap GPU compute in `tokio::task::spawn_blocking`
- [ ] OR use `web::block` for offloading to blocking thread pool

---

### Issue 2: Session Persistence (In-Memory Only)
**Status:** ❌ PRESENT | **Severity:** High | **Priority:** P1

| Location | Evidence |
|----------|----------|
| `src/services/nostr_service.rs:102` | `users: Arc<RwLock<HashMap<String, NostrUser>>>` |
| Redis | Available but only for settings cache, NOT sessions |
| Neo4j | `:User` nodes exist but NO `:Session` nodes |

**Impact:** All users logged out on server restart; horizontal scaling impossible

**Fix Options:**
- [ ] Add `(:User)-[:HAS_SESSION]->(:Session)` to Neo4j
- [ ] OR implement Redis session store with proper TTL
- [ ] Add session restoration on server startup

---

### Issue 3: FFI/GPU Struct Alignment Fragility
**Status:** ⚠️ RISK | **Severity:** High | **Priority:** P2

| Location | Concern |
|----------|---------|
| `src/actors/gpu/semantic_forces_actor.rs` | `#[repr(C)]` structs must match CUDA |
| `src/gpu/types.rs` | No automated layout verification |
| `src/utils/semantic_forces.cu` | C++ structs may drift from Rust |

**Impact:** Silent physics explosions, illegal memory access on GPU

**Fix Options:**
- [ ] Implement `bindgen` in `build.rs` to auto-generate from C++ headers
- [ ] OR add `static_assertions` on struct size/alignment
- [ ] Add CI check comparing Rust vs C++ struct layouts

---

### Issue 4: Data Sync Race Condition
**Status:** ⚠️ RISK | **Severity:** Medium | **Priority:** P2

| Component | Action |
|-----------|--------|
| GitHubSyncService | Writes default `sim_x/y/z` to Neo4j |
| Physics Engine | Writes computed positions at 60fps |
| Conflict | Batch sync may reset physics state |

**Impact:** Jittery node positions, DB locking issues

**Fix Options:**
- [ ] Modify sync MERGE to NEVER update `sim_*` properties
- [ ] Add position read-before-write in GitHubSyncService
- [ ] Implement optimistic locking on position updates

---

### Issue 5: Client-Side Singleton Pattern
**Status:** ❌ NOT FIXED | **Severity:** Medium | **Priority:** P3

| Location | Evidence |
|----------|----------|
| `client/src/services/WebSocketService.ts:79` | `private static instance: WebSocketService` |
| `client/src/services/WebSocketService.ts:223-228` | `getInstance()` method |
| `client/src/services/WebSocketService.ts:1716` | `export const webSocketService = WebSocketService.getInstance()` |

**Impact:** Testing difficulties, SSR incompatibility, state reset issues

**Fix Options:**
- [ ] Migrate to React Context (`WebSocketContext`)
- [ ] OR move to Zustand store (already used for `settingsStore`)
- [ ] Add cleanup mechanism for test isolation

---

### Issue 6: NIP-98 Token Window Too Narrow
**Status:** ❌ NOT FIXED | **Severity:** Medium | **Priority:** P2

| Location | Value |
|----------|-------|
| `src/utils/nip98.rs:163` | `TOKEN_MAX_AGE_SECONDS: i64 = 60` |

**Impact:** Users with clock skew >60s cannot authenticate

**Fix Options:**
- [ ] Increase `TOKEN_MAX_AGE_SECONDS` to 300 (5 minutes)
- [ ] Add explicit error message: "Check your system clock"
- [ ] Consider NTP sync recommendation in client

---

### Issue 7: GraphDataManager Silent Failure
**Status:** ⚠️ PARTIAL | **Severity:** Medium | **Priority:** P2

| Location | Current State |
|----------|---------------|
| `client/src/features/graph/managers/graphDataManager.ts:59-64` | Logs warning but continues |
| `client/src/features/graph/managers/graphDataManager.ts:340` | Returns `{ nodes: [], edges: [] }` silently |

**Impact:** Blank screen of death with no user feedback

**Done:**
- [x] Graceful fallback to empty data (no crash)

**Remaining:**
- [ ] Add `<ErrorModal>` when worker initialization fails
- [ ] Implement main-thread fallback processing
- [ ] Display browser compatibility requirements

---

### Issue 8: Hardcoded Secret Backdoor
**Status:** ⚠️ PARTIAL | **Severity:** Critical | **Priority:** P1

| Location | Current State |
|----------|---------------|
| `src/app_state.rs:633-645` | `ALLOW_INSECURE_DEFAULTS` checks |
| `src/app_state.rs:637` | `"change-this-secret-key"` hardcoded |

**Done:**
- [x] Random key generation when `ALLOW_INSECURE_DEFAULTS` not set

**Remaining:**
- [ ] REMOVE `ALLOW_INSECURE_DEFAULTS` condition entirely
- [ ] Panic on missing `MANAGEMENT_API_KEY` in production
- [ ] Add startup validation for all security-critical env vars

---

## 🟡 TODO - IN PROGRESS

### Item 10: Legacy CUDA Kernel Cleanup
**Status:** 🔄 IN PROGRESS

| Sub-task | Status | Location |
|----------|--------|----------|
| Remove hardcoded switch/case | ⚠️ PARTIAL | CPU fallback uses dynamic registry; GPU kernel still has hardcoding |
| Increase MAX_RELATIONSHIP_TYPES | ❌ NOT STARTED | `src/utils/semantic_forces.cu:70` → still 256 |
| Device-side dynamic allocation | ❌ NOT STARTED | Using static array allocation |

**Files:**
- `src/utils/semantic_forces.cu:70` - `#define MAX_RELATIONSHIP_TYPES 256`
- `src/gpu/semantic_forces.rs:878-940` - CPU fallback with dynamic registry

---

### Item 11: True Client ACK for Backpressure
**Status:** 🔄 IN PROGRESS

| Sub-task | Status | Location |
|----------|--------|----------|
| Application-level ACK types | ✅ DONE | `src/actors/messages.rs:1720-1755` |
| End-to-end delivery confirmation | ⚠️ PARTIAL | Framework exists in `backpressure.rs` |
| Integrate fastwebsockets ACK flow | ❌ NOT STARTED | `fastwebsockets_handler.rs` has no ACK sending |

**Defined Types:**
```rust
pub struct PositionBroadcastAck { correlation_id: u64, clients_delivered: u32 }
pub struct ClientBroadcastAck { sequence_id: u64, nodes_received: u32, timestamp: u64 }
```

---

### Item 12: Parallel Ontology Processing
**Status:** ✅ DONE

| Sub-task | Status | Evidence |
|----------|--------|----------|
| FuturesUnordered for file batches | ✅ DONE | `streaming_sync_service.rs:432-435` |
| Remove 50ms rate limit sleep | ✅ DONE | Using `tokio::task::yield_now()` instead |
| Batch node enrichment | ❌ NOT STARTED | No cross-file inference deduplication |

**Evidence:**
```rust
use futures::stream::{FuturesUnordered, StreamExt};
let mut futures: FuturesUnordered<ProcessFuture> = FuturesUnordered::new();
```

---

## 🔵 JSS Integration Roadmap

### Phase 1: Docker Foundation
**Status:** ✅ DONE

- [x] Add JSS to docker-compose
- [x] Create Dockerfile.jss
- [x] Verify Nostr auth works

### Phase 2: Multi-User Pods
**Status:** ⚠️ 40% IMPLEMENTED (Documentation complete, code partial)

| Item | Status | Notes |
|------|--------|-------|
| `/pods/{npub}/` URL structure | 📝 DOCUMENTED | See `SOLID_POD_CREATION.md` |
| Auto-provision pods on login | 📝 DOCUMENTED | Flow documented, code scaffolded |
| Nostr -> WebID mapping | 📝 DOCUMENTED | Mapping schema defined |
| Actual pod provisioning code | ❌ NOT IMPLEMENTED | Only test scaffolds exist |

### Phase 3: User Ontology Ownership
**Status:** 🔄 PENDING

- [ ] Personal ontology fragments in pods
- [ ] Proposal/merge workflow
- [ ] Reverse sync to GitHub

### Phase 4: Frontend Pod UI
**Status:** 🔄 PENDING

- [ ] Create `SolidPodService.ts`
- [ ] Pod browser component
- [ ] Contribution/proposal UI

### Phase 5: Agent Memory
**Status:** 🔄 PENDING

- [ ] Per-agent pods (54 agent types)
- [ ] Claude-flow hooks for JSS
- [ ] Migrate `.agentdb` to pods

---

## 📊 Summary Matrix

| ID | Issue | Status | Severity | Effort |
|----|-------|--------|----------|--------|
| 1 | Concurrency hazards | ❌ PRESENT | Critical | Medium |
| 2 | Session persistence | ❌ PRESENT | High | Medium |
| 3 | FFI struct alignment | ⚠️ RISK | High | Low |
| 4 | Data sync race | ⚠️ RISK | Medium | Low |
| 5 | Singleton pattern | ❌ NOT FIXED | Medium | Low |
| 6 | NIP-98 60s window | ❌ NOT FIXED | Medium | Trivial |
| 7 | GraphDataManager error | ⚠️ PARTIAL | Medium | Low |
| 8 | Hardcoded secret | ⚠️ PARTIAL | Critical | Trivial |
| 10 | CUDA cleanup | 🔄 IN PROGRESS | Low | Medium |
| 11 | Client ACK | 🔄 IN PROGRESS | Low | Medium |
| 12 | Parallel ontology | ✅ DONE | Low | - |

---

## 🛠️ Verification Commands

```bash
# Issue 1: Check for std::sync::Mutex in GPU actors
grep -n "std::sync::Mutex" src/actors/gpu/*.rs

# Issue 2: Check session storage mechanism
grep -n "users.*HashMap" src/services/nostr_service.rs

# Issue 5: Check WebSocket singleton
grep -n "getInstance\|static instance" client/src/services/WebSocketService.ts

# Issue 6: Check NIP-98 token window
grep -n "TOKEN_MAX_AGE_SECONDS" src/utils/nip98.rs

# Issue 8: Check insecure defaults
grep -n "ALLOW_INSECURE_DEFAULTS\|change-this-secret-key" src/app_state.rs

# Item 10: Check MAX_RELATIONSHIP_TYPES
grep -n "MAX_RELATIONSHIP_TYPES" src/utils/semantic_forces.cu

# Item 12: Verify FuturesUnordered usage
grep -n "FuturesUnordered" src/services/streaming_sync_service.rs

# JSS: Check service status
docker-compose -f docker-compose.unified.yml config | grep -A20 "jss:"
```

---

## 📋 Agent Assignment Recommendations

For hive-mind parallel execution:

| Agent Type | Assigned Issues | Estimated Complexity |
|------------|-----------------|---------------------|
| `backend-dev` | Issues 1, 2, 4, 8 | High |
| `coder` | Issues 3, 6 | Low |
| `mobile-dev` / `coder` | Issues 5, 7 | Medium |
| `cicd-engineer` | Issue 8 (env validation) | Low |
| `sparc-coder` | Items 10, 11 | Medium |

---

> **AWAITING AUTHORIZATION** to proceed with fixes.
> Reply with priority order or specific issues to address.
