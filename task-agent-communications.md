# Complete Re-engineering: Agent Telemetry to Hybrid REST/WebSocket System

## Project Context

**Codebase Location:**
- Backend (Rust): `/workspace/ext/src`
- Frontend (TypeScript/React): `/workspace/ext/client/src`

**Development Environment:**
- `cargo check` is available for Rust validation
- **Application cannot be started from within this container**
- All changes must be validated through static analysis only
- Version control is active - no fallback implementation needed

## Architectural Goal

Transform the current brittle REST polling system into a **hybrid REST + WebSocket architecture**:
- **REST**: Initial state load, agent creation, configuration (CRUD operations)
- **WebSocket**: Real-time telemetry streaming via binary protocol (continuous updates)

This is a **complete rewrite** approach - no gradual migration, no fallback code.

---

## Swarm Execution Plan

This task will be executed by a **coordinated agent swarm** in a **single parallel operation**. Each agent has clear responsibilities and dependencies.

### Agent Assignment & Execution Order

**Wave 1: Protocol & Core Infrastructure (Parallel)**
1. **System Architect Agent** - Design complete hybrid architecture
2. **Binary Protocol Agent** - Define and implement binary telemetry structures
3. **Database Agent** - Update schema for WebSocket connection tracking

**Wave 2: Backend Implementation (Parallel, depends on Wave 1)**
4. **WebSocket Handler Agent** - Implement hybrid WebSocket handler
5. **REST API Agent** - Refactor REST endpoints for initial load only
6. **Agent Service Agent** - Update agent spawning to emit binary telemetry
7. **MCP Bridge Agent** - Hook telemetry into MCP session lifecycle

**Wave 3: Frontend Implementation (Parallel, depends on Wave 2)**
8. **WebSocket Client Agent** - Implement binary protocol parser
9. **State Management Agent** - Refactor context to hybrid model
10. **UI Component Agent** - Update telemetry display components
11. **Cleanup Agent** - Remove all polling infrastructure

**Wave 4: Integration & Validation (Sequential, depends on Wave 3)**
12. **Integration Agent** - Connect all components end-to-end
13. **Type Safety Agent** - Ensure Rust/TS type consistency
14. **Code Review Agent** - Final quality check

**Wave 5: Documentation & Deployment (Parallel, depends on Wave 4)**
15. **Documentation Agent** - Update architecture and usage docs integrated into the knowledgebase in /workspace/ext/docs with forward and backward links and mermaid diagrams.


---

## Detailed Task Breakdown

### WAVE 1: Foundation

#### Task 1.1: System Architecture Design
**Agent**: `system-architect`
**Deliverable**: `/workspace/ext/docs/hybrid-architecture.md`

Design decisions:
- Binary protocol specification (64-byte agent telemetry frame)
- WebSocket message types (subscription, binary data, metadata updates)
- REST endpoint reduction strategy (7 endpoints → 2 endpoints)
- Connection lifecycle management
- Error handling and reconnection strategy
- String table for dynamic metadata (agent names, IDs)

#### Task 1.2: Binary Protocol Implementation
**Agent**: `coder` (protocol specialist)
**Files to Create/Modify**:
- `/workspace/ext/src/utils/binary_agent_protocol.rs` (NEW)
- `/workspace/ext/src/utils/socket_flow_messages.rs` (EXTEND)

Define:
```rust
#[repr(C, packed)]
pub struct BinaryAgentTelemetry {
    // Identification (8 bytes)
    agent_id: u32,
    agent_type: u32,

    // Position (24 bytes)
    x: f32, y: f32, z: f32,
    vx: f32, vy: f32, vz: f32,

    // Telemetry (16 bytes)
    health: f32,
    cpu_usage: f32,
    memory_usage: f32,
    workload: f32,

    // Status (8 bytes)
    status: u32,
    age_seconds: u32,

    // References (8 bytes)
    string_id_index: u32,
    name_index: u32,
}

pub struct StringTable {
    entries: HashMap<u32, String>,
}
```

Message types:
- `SUBSCRIBE_AGENT_TELEMETRY` (0x04)
- `AGENT_TELEMETRY_BINARY` (0x05)
- `AGENT_METADATA_UPDATE` (0x06 - JSON)

#### Task 1.3: Database Schema Updates
**Agent**: `coder` (database specialist)
**Files to Modify**:
- `/workspace/ext/migrations/` (NEW migration file)

Add:
- WebSocket session tracking table
- Agent-to-session mapping
- Connection state persistence

---

### WAVE 2: Backend Transformation

#### Task 2.1: Hybrid WebSocket Handler
**Agent**: `backend-dev`
**Files to Create/Modify**:
- `/workspace/ext/src/handlers/hybrid_agent_websocket.rs` (NEW - 800 lines)
- `/workspace/ext/src/handlers/socket_flow_handler.rs` (MODIFY - integrate or deprecate)

Implement:
- Subscription management (`subscribeAgentTelemetry`)
- Binary broadcast loop (5Hz = 200ms interval)
- String table updates (event-driven)
- Connection lifecycle (connect, subscribe, stream, disconnect)
- Integration with existing `botsGraphUpdate` infrastructure

Key functions:
```rust
async fn handle_agent_subscription(session_id: &str, state: &AppState)
async fn broadcast_agent_telemetry(state: &AppState)
async fn send_string_table_update(session_id: &str, agents: &[Agent])
```

#### Task 2.2: REST API Refactoring
**Agent**: `backend-dev`
**Files to Modify**:
- `/workspace/ext/src/handlers/bots_handler.rs` (SIMPLIFY)
- `/workspace/ext/src/routes.rs` (UPDATE)

**REMOVE** (deprecate for telemetry):
- `/api/bots/data` (polling endpoint)
- `/api/bots/agents/telemetry` (polling endpoint)
- `/api/bots/status/poll` (polling endpoint)

**KEEP** (initial load only):
- `GET /api/bots/agents` - Initial agent list
- `POST /api/bots/agents` - Create new agent
- `DELETE /api/bots/agents/:id` - Remove agent

**NEW**:
- `GET /api/bots/agents/initial` - Full state snapshot for new clients

#### Task 2.3: Agent Service Integration
**Agent**: `backend-dev`
**Files to Modify**:
- `/workspace/ext/src/services/bots_service.rs` (EXTEND)
- `/workspace/ext/src/handlers/bots_handler.rs` (HOOK)

Changes:
- `convert_agents_to_nodes()` → also produces `BinaryAgentTelemetry`
- Emit binary telemetry to WebSocket broadcast queue
- Update agent state tracking (health, CPU, memory)
- Hook into agent lifecycle events (spawn, update, terminate)

#### Task 2.4: MCP Session Bridge
**Agent**: `backend-dev`
**Files to Modify**:
- `/workspace/ext/src/services/mcp_session_bridge.rs` (EXTEND)

Integration points:
- Agent discovery callback → emit string table update
- Swarm initialization → broadcast metadata
- Agent status changes → update binary telemetry
- Session termination → cleanup telemetry streams

---

### WAVE 3: Frontend Transformation

#### Task 3.1: Binary Protocol Parser
**Agent**: `coder` (TypeScript specialist)
**Files to Create/Modify**:
- `/workspace/ext/client/src/services/binaryAgentProtocol.ts` (NEW - 200 lines)
- `/workspace/ext/client/src/services/binaryProtocol.ts` (EXTEND)

Implement:
```typescript
interface BinaryAgentTelemetry {
  agentId: number;
  agentType: AgentType;
  position: { x: number; y: number; z: number };
  velocity: { vx: number; vy: number; vz: number };
  telemetry: {
    health: number;
    cpuUsage: number;
    memoryUsage: number;
    workload: number;
  };
  status: AgentStatus;
  ageSeconds: number;
  stringIdIndex: number;
  nameIndex: number;
}

export function parseBinaryAgentTelemetry(buffer: ArrayBuffer): BinaryAgentTelemetry[]
export function mergeWithStringTable(telemetry: BinaryAgentTelemetry[], stringTable: Map<number, string>): AgentWithMetadata[]
```

#### Task 3.2: WebSocket Client Refactoring
**Agent**: `coder` (TypeScript specialist)
**Files to Modify**:
- `/workspace/ext/client/src/services/BotsWebSocketIntegration.ts` (EXTEND)
- `/workspace/ext/client/src/contexts/BotsDataContext.tsx` (MAJOR REFACTOR)

Changes:
- Add `subscribeAgentTelemetry()` function
- Handle binary telemetry messages (type 0x05)
- Handle string table updates (type 0x06)
- Remove all REST polling logic (DELETE ~500 lines)
- Use REST only for initial load on mount

#### Task 3.3: State Management Overhaul
**Agent**: `coder` (React specialist)
**Files to Modify**:
- `/workspace/ext/client/src/contexts/BotsDataContext.tsx` (REWRITE)

New state model:
```typescript
interface BotsDataState {
  // REST initial load
  initialAgents: Agent[];

  // WebSocket real-time
  telemetryData: Map<string, BinaryAgentTelemetry>;
  stringTable: Map<number, string>;

  // Connection state
  wsConnected: boolean;
  lastUpdate: Date;
}
```

Remove:
- All `useAgentPolling` hooks
- Polling intervals
- REST API calls (except initial load)

#### Task 3.4: UI Component Updates
**Agent**: `coder` (React specialist)
**Files to Modify**:
- `/workspace/ext/client/src/components/Bots/AgentTelemetryStream.tsx` (SIMPLIFY)
- `/workspace/ext/client/src/components/Bots/BotsPanel.tsx` (UPDATE)
- `/workspace/ext/client/src/components/Bots/AgentDetails.tsx` (UPDATE)

Changes:
- Read telemetry from WebSocket state (not polling)
- Display real-time updates (5Hz refresh)
- Show connection status indicator
- Handle WebSocket disconnections gracefully

#### Task 3.5: Cleanup & Removal
**Agent**: `cleanup-specialist`
**Files to DELETE**:
- `/workspace/ext/client/src/services/AgentPollingService.ts` (218 lines)
- `/workspace/ext/client/src/hooks/useAgentPolling.ts` (254 lines)
- All polling-related utility functions

**Code to DELETE** (search and destroy):
- All `setInterval` for agent polling
- All `fetch('/api/bots/...')` calls (except initial load)
- All polling state variables (`pollingEnabled`, `pollingInterval`, etc.)

---

### WAVE 4: Integration & Validation

#### Task 4.1: End-to-End Integration
**Agent**: `integration-specialist`
**Deliverable**: `/workspace/ext/docs/integration-test-plan.md`

Validation checklist:
- [ ] WebSocket connects on component mount
- [ ] Subscription message sent
- [ ] Binary telemetry received at 5Hz
- [ ] String table updates on agent spawn
- [ ] UI displays real-time data
- [ ] No REST polling occurs after initial load
- [ ] Reconnection works after disconnect
- [ ] Multiple clients can subscribe simultaneously

#### Task 4.2: Type Safety Verification
**Agent**: `code-analyzer`
**Tasks**:
- Run `cargo check` on all Rust code
- Run `tsc --noEmit` on all TypeScript code
- Verify binary protocol byte alignment
- Validate enum consistency (Rust ↔ TypeScript)
- Check for unsafe transmutes

#### Task 4.3: Final Code Review
**Agent**: `reviewer`
**Focus Areas**:
- Memory safety (binary protocol handling)
- Error handling (WebSocket failures)
- Resource cleanup (connection lifecycle)
- Performance (binary encoding efficiency)
- Code removal completeness (no dead polling code)

---

## Success Criteria

1. **Zero REST polling** after initial page load
2. **5Hz telemetry updates** via binary WebSocket
3. **<5ms latency** from agent update to UI display
4. **~3000 lines of code removed** (polling infrastructure)
5. **Type-safe** end-to-end (Rust structs → TypeScript interfaces)
6. **cargo check passes** with no errors
7. **No fallback code** - clean implementation only

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Binary protocol bugs | High | Extensive unit tests for parser |
| WebSocket disconnections | Medium | Automatic reconnection + state sync |
| Breaking existing clients | Low | No external clients depend on this |
| Performance regression | Low | Binary is faster than JSON/REST |
| Type mismatches | Medium | Shared type definitions + codegen |

---

## File Impact Summary

**NEW FILES** (~1500 lines):
- `src/utils/binary_agent_protocol.rs`
- `src/handlers/hybrid_agent_websocket.rs`
- `client/src/services/binaryAgentProtocol.ts`
- `docs/hybrid-architecture.md`
- `migrations/YYYYMMDD_websocket_sessions.sql`

**MODIFIED FILES** (~2000 lines changed):
- `src/handlers/socket_flow_handler.rs`
- `src/handlers/bots_handler.rs`
- `src/services/bots_service.rs`
- `src/services/mcp_session_bridge.rs`
- `src/routes.rs`
- `client/src/services/BotsWebSocketIntegration.ts`
- `client/src/contexts/BotsDataContext.tsx`
- `client/src/components/Bots/AgentTelemetryStream.tsx`

**DELETED FILES** (~500 lines):
- `client/src/services/AgentPollingService.ts`
- `client/src/hooks/useAgentPolling.ts`

**NET CHANGE**: +1500 new, -3000 removed = **-1500 lines** (20% code reduction)

---

## Execution Command

To execute this plan with a coordinated swarm:

```bash
# Initialize hierarchical swarm (13 agents)
npx claude-flow@alpha swarm init --topology hierarchical --max-agents 13

# Spawn all agents in parallel (Claude Code Task tool)
# Execute in a single message with 13 Task() calls for each agent above
```

---

## Notes

- This is a **one-shot rewrite** - no incremental migration
- Version control protects against errors - commit before execution
- Static validation only - `cargo check` will verify correctness
- Application cannot be tested live in this container
- All coordination via claude-flow hooks and memory
