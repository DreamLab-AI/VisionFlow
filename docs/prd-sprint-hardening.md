# PRD: VisionFlow Security & Architecture Hardening Sprint

**Model Audit Source**: GPT-5.4 (1M context) | **Date**: 2026-03-07
**Findings**: 1 CRITICAL, 70 HIGH, 124 MEDIUM, 77 LOW, 19 INFO across 291 items
**Sprint Scope**: All CRITICAL + HIGH (71 items), selected MEDIUM (30 items)

---

## Epic 1: Authentication & Authorization Hardening

### E1.1 - Enforce Auth on All WebSocket Upgrades (HIGH x6)
**Files**: `socket_flow_handler`, `client_messages_handler`, `mcp_relay_handler`,
`speech_socket_handler`, `multi_mcp_websocket_handler`, `fastwebsockets_handler`
**AC**: Every WebSocket upgrade rejects unauthenticated connections with 401/403.
No "log and allow" patterns remain.

### E1.2 - Enforce Auth on Mutating REST Endpoints (HIGH x5)
**Files**: `graph_state_handler`, `ontology_physics/mod`, `semantic_forces`,
`files/mod`, `analytics/websocket_integration`
**AC**: All POST/PUT/DELETE routes wrapped with `RequireAuth::authenticated()`.
Rate limiting applied to mutation endpoints.

### E1.3 - Fix Nostr Auth Trust Model (HIGH x2)
**Files**: `nostr_handler`, `auth_extractor`
**AC**: `X-Nostr-Pubkey` header validated against active session.
`SETTINGS_AUTH_BYPASS` guarded by compile-time dev-only check.
API-keys routes fixed with `{pubkey}` path param.

### E1.4 - Secure Voice Command Auth Boundary (HIGH)
**File**: `speech_service`, `speech_voice_integration`
**AC**: Voice-triggered orchestration requires authenticated user context.

---

## Epic 2: Critical Security Fixes

### E2.1 - Remove Client-Side Private Key Persistence (CRITICAL)
**File**: `client/src/services/nostrAuthService.ts`
**AC**: No `sessionStorage.setItem('nostr_passkey_key')` calls exist.
Private keys exist only in module memory. Zeroized on logout.

### E2.2 - Eliminate Cypher Injection Surface (HIGH x2)
**Files**: `neo4j_adapter`, `graph_queries`
**AC**: `query_nodes(cypher_query)` removed or restricted to `pub(crate)`.
Only parameterized query builders exposed. `execute_cypher` removed.

### E2.3 - Fix Insecure Default Credentials (HIGH x2)
**Files**: `neo4j_adapter`, `neo4j_settings_repository`
**AC**: Startup fails unless `ALLOW_INSECURE_DEFAULTS=true` in dev.
No silent `"password"` fallbacks. Centralized credential validation.

### E2.4 - Remove Unsafe Remote Script Loading (HIGH)
**File**: `client/src/features/bots/components/AgentTelemetryStream.tsx`
**AC**: No dynamic external `<script>` injection. Self-host or sandbox.

### E2.5 - Sanitize Markdown Link Protocols (HIGH)
**File**: `client/src/features/design-system/patterns/MarkdownRenderer.tsx`
**AC**: Only `http:`, `https:`, `mailto:` protocols allowed in rendered links.

### E2.6 - Fix FeatureAccess .env Mutation (HIGH)
**File**: `src/config/feature_access.rs`
**AC**: Feature access persisted in database, not `.env` file writes.

---

## Epic 3: Actor System & Data Integrity

### E3.1 - Fix GraphStateActor Fire-and-Forget Persistence (HIGH)
**File**: `src/actors/graph_state_actor.rs`
**AC**: Critical mutations persist-first-then-mutate or rollback on failure.

### E3.2 - Fix Startup Race Conditions (HIGH)
**Files**: `app_state`, `graph_state_actor`
**AC**: Explicit startup phases: sync -> DB load -> actor reload -> physics init.
Startup coordinator prevents partial state.

### E3.3 - Fix Physics Pipeline Fragility (HIGH)
**File**: `src/actors/physics_orchestrator_actor.rs`
**AC**: State machine with step IDs. Stale completions rejected.

### E3.4 - Fix Node Position Field Duplication (HIGH)
**File**: `src/models/node.rs`
**AC**: Single source of truth for position/velocity. No `data.x` vs `x` divergence.

### E3.5 - Unify Data Ingestion Pipeline (HIGH)
**Files**: `file_service`, `github_sync_service`, `graph_state_actor`
**AC**: Single ingestion path. All sync converges through one application service.

---

## Epic 4: Protocol & Communication

### E4.1 - Add Binary Protocol Size Caps (HIGH)
**File**: `src/utils/binary_protocol.rs`
**AC**: Max payload size and max node count enforced before allocation.

### E4.2 - Fix GraphExportHandler State Lifetime (HIGH)
**File**: `src/handlers/graph_export_handler.rs`
**AC**: Single `GraphExportHandler` in `AppState`, reused across requests.

### E4.3 - Replace Thread-Per-Request Pattern (HIGH)
**File**: `src/handlers/utils.rs`
**AC**: `execute_in_thread` uses `spawn_blocking` or bounded thread pool.

### E4.4 - Fix MCPRelayActor Broken Forwarding (HIGH)
**File**: `src/handlers/mcp_relay_handler.rs`
**AC**: `orchestrator_tx` properly stored after connection. Forwarding works.

### E4.5 - Fix WebSocket URL Construction (HIGH - Client)
**File**: `client/src/features/bots/components/BotsControlPanel.tsx`
**AC**: WebSocket URLs use absolute `ws://`/`wss://` protocol.

### E4.6 - Cap WebSocket Update Interval (HIGH)
**File**: `analytics/websocket_integration`
**AC**: `update_interval_ms` clamped to `[100, 60000]` server-side.

---

## Epic 5: Client Architecture

### E5.1 - Fix React Hook Rule Violation (HIGH)
**File**: `client/src/hooks/useErrorHandler.tsx`
**AC**: `withErrorHandler` refactored to proper hook or accepts toast as param.

### E5.2 - Fix Per-Frame Allocations (HIGH x2)
**Files**: `GraphManager.tsx`, `BotsVisualization.tsx`
**AC**: No `slice()` in render loop. Preallocated buffers. No `new Vector3` in JSX.

### E5.3 - Decompose WebSocket Store (HIGH x2)
**File**: `client/src/store/websocketStore.ts`
**AC**: Split into graph WS, binary protocol, message queue, connection lifecycle.
Module-level mutable state moved into store or encapsulated services.

### E5.4 - Consolidate Duplicate Hooks (HIGH x2)
**Files**: `useSolidPod` (2 copies), `useSolidResource` (2 copies)
**AC**: Single canonical implementation per hook. Others re-export.

### E5.5 - Fix AgentDetailPanel Select Misuse (HIGH)
**File**: `client/src/features/bots/components/AgentDetailPanel.tsx`
**AC**: Uses Radix `SelectItem`/`SelectContent` instead of native `<option>`.

### E5.6 - Add Runtime API Validation (HIGH x2)
**Files**: `graph.worker.ts`, multiple API consumers
**AC**: Incoming graph data validated with runtime schemas. `any` casts replaced.

### E5.7 - Fix Polling Singleton Reference Counting (HIGH)
**File**: `AgentPollingService.ts`, `useAgentPolling.ts`
**AC**: Stop only when subscriber count reaches zero.

### E5.8 - Eliminate Unsafe HTTPS Bypass (HIGH)
**File**: `client/src/services/AudioInputService.ts`
**AC**: No developer mode HTTPS bypass in production builds.

### E5.9 - Fix Remote Logger Data Leakage (HIGH)
**File**: `client/src/services/remoteLogger.ts`
**AC**: Sensitive keys redacted. Disabled in production by default.

---

## Epic 6: Server Infrastructure

### E6.1 - Fix Global Mutable Analytics State (HIGH)
**File**: `analytics/state.rs`
**AC**: Feature flags persisted in settings repository, not process-local statics.

### E6.2 - Fix SSSP Split-Brain State (HIGH)
**File**: `analytics/sssp_handlers.rs`
**AC**: GPU update applied first, feature flag committed only on success.

### E6.3 - Consolidate Duplicate Route Surfaces (HIGH x2)
**Files**: `ontology_handler` vs `api_handler/ontology`, `constraints` duplication
**AC**: One canonical API surface per domain. Legacy routes removed.

### E6.4 - Fix Ontology Infinite Recursion (HIGH)
**File**: `api_handler/ontology/mod.rs`
**AC**: `calculate_depth`/`count_descendants` have cycle detection via visiting set.

### E6.5 - Fix Dense Matrix Scaling (HIGH)
**File**: `src/physics/stress_majorization.rs`
**AC**: Sparse representation or landmark-sampled stress for large graphs.

### E6.6 - Fix Auth Extractor Thread/Runtime Creation (HIGH)
**File**: `src/settings/auth_extractor.rs`
**AC**: No `Runtime::new()` inside extractors. Async-native implementation.

### E6.7 - Fix Drag Authorization (HIGH)
**File**: `socket_flow_handler/position_updates.rs`
**AC**: Client can only drag nodes they're authorized to manipulate.

### E6.8 - Fix MCPConnectionPool Incomplete Implementation (HIGH)
**File**: `src/utils/async_improvements.rs`
**AC**: Either implement proper pooling or remove dead code.

### E6.9 - Fix Settings Config Source Consolidation (HIGH)
**Files**: `feature_access.rs`, settings APIs
**AC**: One authoritative settings/config service. Legacy paths removed.

### E6.10 - Fix Query Token Auth (HIGH)
**File**: `socket_flow_handler/http_handler.rs`
**AC**: Query-string token auth disabled in production. Header/cookie only.

---

## Epic 7: Docs Alignment (Selected MEDIUM)

### E7.1 - Unify Binary Protocol Spec
**AC**: One canonical protocol spec generated from code. Stale docs removed.

### E7.2 - Remove False "Backend-Only" Doc
**AC**: `backend-api-architecture-complete.md` removed or corrected.

### E7.3 - Update Actor Architecture Diagrams
**AC**: Diagrams reflect current transitional state with explicit status banners.

---

## Execution Tranches

| Tranche | Focus | Epics | Agents |
|---------|-------|-------|--------|
| T1 | Security | E1, E2 | security-architect, security-auditor, coder x2 |
| T2 | Actors & Data | E3 | architecture, coder x2, tester |
| T3 | Protocol & Comms | E4 | coder x2, tester |
| T4 | Client Arch | E5 | coder x3, reviewer |
| T5 | Server Infra | E6 | coder x2, tester |
| T6 | Docs & QE | E7 + validation | reviewer, api-docs |

**Total**: ~15 specialist agents across 6 tranches
**Topology**: hierarchical-mesh
**Consensus**: raft
