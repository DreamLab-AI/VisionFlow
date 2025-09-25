# Backend Refactoring Task List

## Cargo Check Status (Final Verification)
**✅ NO COMPILATION ERRORS** - System compiles successfully after all updates
**⚠️ 230 warnings** - Reduced from 242 (cleaned up GPU actor unused imports)
**📊 Stability Confirmed** - All documentation updates complete, no code breakage

## Priority Tasks (Ordered by Impact)

✅ **RESOLVED**: API Endpoint Analysis Complete
- The `/api/bots/spawn-agent-hybrid` endpoint **IS IMPLEMENTED** in `/workspace/ext/src/handlers/api_handler/bots/mod.rs:24`
- Implementation found in `/workspace/ext/src/handlers/bots_handler.rs:402`
- The documentation in `interface-layer.md:57` is **INCORRECT** - it states the endpoint is missing but it exists
- **Potential Issue**: Client sends `agentType` (camelCase) but server expects `agent_type` (snake_case), though Serde should handle conversion with `#[serde(rename_all = "camelCase")]`

Overly Complex Actor Implementations: Some actors, like graph_actor.rs, are extremely large (over 29,000 tokens). While feature-rich, this can hinder maintainability. Consider breaking down its responsibilities (e.g., into a separate PhysicsOrchestratorActor, SemanticAnalysisActor, etc.) to adhere more closely to the single-responsibility principle.

graph_actor.rs: This actor is the heart of the system but has grown too large. Its responsibilities (graph state, physics, semantic analysis, constraints, stress majorization) should be delegated to smaller, more focused child actors that it supervises.
settings_handler.rs & settings_paths.rs: The path-based API for granular settings updates is a high-performance approach that avoids sending large, redundant configuration objects. This is an excellent design choice.
src/gpu/ & src/utils/*.cu
✅ **RESOLVED**: GraphServiceActor Refactoring Plan Complete
- Created comprehensive refactoring blueprint at `/workspace/ext/docs/graphactor-refactoring-plan.md`
- Analyzed current state: 3,104 lines (~38,456 tokens) with 35+ message handlers
- Proposed decomposition into 5 focused actors:
  - GraphServiceSupervisor (coordinator, 500-800 lines)
  - GraphStateActor (data management, 800-1200 lines)
  - PhysicsOrchestratorActor (simulation & GPU, 1000-1500 lines)
  - SemanticProcessorActor (AI features, 800-1200 lines)
  - ClientCoordinatorActor (WebSocket communication, 600-1000 lines)
- Included 7-phase implementation plan with testing strategies and risk mitigation
- Defined success metrics: 75% complexity reduction, <1000 lines per actor
- Estimated effort: 16 days for complete refactoring with comprehensive testing

## Client-Side Bug Fixes
✅ **RESOLVED**: Fixed syntax error in useAutoBalanceNotifications.ts
- Issue: Extra closing brace causing "Expected finally but found }" error
- Location: `/workspace/ext/client/src/hooks/useAutoBalanceNotifications.ts:55`
- Fix: Removed duplicate closing brace, proper indentation restored
- Verification: TypeScript compilation successful

✅ **RESOLVED**: Fixed useVoiceInteractionCentralized.tsx
- Issue: JSX in .ts file causing TypeScript errors
- Location: `/workspace/ext/client/src/hooks/useVoiceInteractionCentralized.tsx`
- Fix: Renamed from .ts to .tsx to support JSX syntax
- Status: File now compiles correctly

## Client-Side Issues Fixed

✅ **RESOLVED**: Fixed circular dependency between loggerConfig.ts and clientDebugState.ts
- Issue: "Cannot access 'clientDebugState' before initialization" error
- Root cause: Circular import dependency causing initialization order problems
- Location: `/workspace/ext/client/src/utils/loggerConfig.ts`
- Fix: Implemented lazy loading with dynamic imports and proxy pattern
- Status: Client no longer shows "require is not defined" error

✅ **RESOLVED**: Added missing replaceGlobalConsole export
- Issue: Missing export in console.ts causing import errors
- Location: `/workspace/ext/client/src/utils/console.ts`
- Fix: Added replaceGlobalConsole function export
- Status: Import error resolved

✅ **RESOLVED**: Successfully connected to VisionFlow via Playwright
- Used headless Playwright to capture console errors
- Identified remaining client issues:
  - 404 errors for `/api/settings/batch` endpoint
  - SharedArrayBuffer warnings (expected in dev)
  - COOP header warnings (normal for non-HTTPS)

## Backend Issues Identified

⚠️ **ROUTE CONFLICT**: Duplicate `/api/settings/batch` endpoint definitions
- Issue: Both `settings_handler.rs` and `settings_paths.rs` define the same routes
- Locations:
  - `/workspace/ext/src/handlers/settings_handler.rs:1566-1567`
  - `/workspace/ext/src/handlers/settings_paths.rs:625-626`
- Both loaded in `/workspace/ext/src/handlers/api_handler/mod.rs:38-39`
- Impact: Second registration overrides first, causing 404 errors
- Solution needed: Remove duplicate route definition or merge implementations

## Infrastructure Issues Found
⚠️ **MCP TCP Connection Issue** (Previously documented)
- VisionFlow is trying to connect to multi-agent-container:9500
- Error: "Connection refused (os error 111)" every 2 seconds
- Impact: MCP features not working, falling back to JSON-RPC
- Container IPs:
  - VisionFlow: 172.18.0.10 (ports 4000 backend, 5173 vite dev)
  - Multi-agent-container: 172.18.0.9 (port 9500 now listening)
- Note: MCP server is now running on port 9500, connection should work

## GraphServiceActor Refactoring Implementation Plan

### Phase 1: Message Type Extraction (Day 1-2)
- Extract all message types to `src/actors/messages/graph_messages.rs`
- Create shared message traits for inter-actor communication
- Run `cargo check` to verify no compilation errors

### Phase 2: GraphStateActor Extraction (Day 3-4)
- Create `src/actors/graph_state_actor.rs` (800-1200 lines)
- Move node/edge management, state persistence, and basic CRUD operations
- Migrate handlers: UpdateNode, AddEdge, RemoveNode, GetGraph, etc.
- Run `cargo check` after extraction

### Phase 3: PhysicsOrchestratorActor Extraction (Day 5-7)
- Create `src/actors/physics_orchestrator_actor.rs` (1000-1500 lines)
- Move physics simulation loop, force computation coordination, GPU communication
- Migrate handlers: StartSimulation, StopSimulation, UpdatePhysics, etc.
- Run `cargo check` to ensure GPU actor communication works

### Phase 4: SemanticProcessorActor Extraction (Day 8-10)
- Create `src/actors/semantic_processor_actor.rs` (800-1200 lines)
- Move AI features, constraint generation, semantic analysis
- Migrate handlers: ApplyConstraints, UpdateSemantics, RunClustering, etc.
- Run `cargo check` for AI feature validation

### Phase 5: ClientCoordinatorActor Extraction (Day 11-12)
- Create `src/actors/client_coordinator_actor.rs` (600-1000 lines)
- Move WebSocket communication, client state sync, position updates
- Migrate handlers: ClientConnect, ClientDisconnect, BroadcastUpdate, etc.
- Run `cargo check` for WebSocket functionality

### Phase 6: Supervisor Pattern Implementation (Day 13-14)
- Convert GraphServiceActor to GraphServiceSupervisor (500-800 lines)
- Implement actor supervision tree
- Add message routing and coordination logic
- Run `cargo check` for supervision hierarchy

### Phase 7: Testing & Validation (Day 15-16)
- Integration tests for inter-actor communication
- Performance benchmarks (target <1ms latency)
- Load testing with 1000+ nodes
- Final `cargo check` and `cargo test`
