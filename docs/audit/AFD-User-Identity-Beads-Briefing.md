# AFD: User Identity Propagation, Beads Integration, and Briefing Workflow

**Status**: APPROVED FOR IMPLEMENTATION
**Author**: Claude
**Date**: 2026-02-13
**Scope**: Cross-boundary identity chain, Beads task tracking, briefing/debrief cycle, Claude-Flow swarm coordination

---

## 1. Problem Statement

VisionFlow's multi-agent Docker container has **zero user identity propagation**. When a VisionFlow user triggers agent work (via UI, voice, or API), the agent container cannot determine which user requested the work. This blocks:

1. **Per-user task attribution** — who requested what
2. **Beads integration** — binding structured task tracking to requesting users
3. **Briefing workflow** — voice memo → brief → parallel agent execution → debrief
4. **Claude-Flow swarm/team coordination** — epic beads with dependency-aware sub-tasks
5. **Audit trail** — end-to-end traceability from user intent to agent output

### Current State (Broken)

```
Frontend (Nostr pubkey) → WebSocket AUTH → Rust Backend (NostrUser)
    ↓                                           ↓
    ✅ User known                    ManagementApiClient.create_task(agent, task, provider)
                                                ↓
                                        ❌ NO user_id sent
                                                ↓
                                    Management API (static API key only)
                                                ↓
                                        ❌ NO user context
                                                ↓
                                    ProcessManager.spawnTask() as devuser
                                                ↓
                                        ❌ Agent has no idea who asked
```

### Target State

```
Frontend (Nostr pubkey) → WebSocket AUTH → Rust Backend (NostrUser)
    ↓                                           ↓
    ✅ User known              create_task(agent, task, provider, user_context)
                                                ↓
                                    Management API validates user_context
                                                ↓
                                    Creates user-scoped workspace + beads
                                                ↓
                                    VISIONFLOW_USER_ID + VISIONFLOW_USER_PUBKEY in env
                                                ↓
                                    bd init (if needed) → bd create epic bead
                                                ↓
                                    Claude-Flow swarm/team → sub-beads with deps
                                                ↓
                                    Agent work tracked via bd → updates on completion
                                                ↓
                                    Debrief generated → pushed to user's area
```

---

## 2. Architecture Decisions

### AD-1: User Context Struct

A `UserContext` JSON object propagates across every boundary:

```json
{
  "user_id": "npub1abc...",
  "pubkey": "abc123...",
  "display_name": "dinis_cruz",
  "session_id": "sess-uuid-v4",
  "is_power_user": false
}
```

**Rationale**: Nostr pubkey is the canonical identity (already in `NostrUser`). The `npub` (bech32) serves as human-readable user_id. The `display_name` (derived from Nostr profile or configured) is used for folder paths and bead attribution.

### AD-2: Beads as the Task Tracking Layer

Every agent task in the container is backed by a Beads issue:

- **Epic bead** created per brief or top-level user request
- **Sub-beads** created per agent/role spawned by Claude-Flow swarm
- **Dependencies** automatically set: sub-beads block the epic
- **Claiming** via `bd update --claim` prevents double-work
- **Session end** mandates `bd sync` + bead status updates

**Rationale**: Beads provides dependency-aware scheduling (`bd ready`), atomic claiming (compare-and-swap), and hash-based IDs (collision-free multi-agent). This is exactly what's missing from VisionFlow's current fire-and-forget task spawning.

### AD-3: Briefing Workflow Folder Structure

```
team/
├── humans/
│   └── {display_name}/
│       ├── briefs/{MM}/{DD}/
│       │   └── v{version}__{type}__{seq}__{slug}.md
│       └── debriefs/{MM}/{DD}/
│           └── v{version}__debrief__{brief-ref}.md
├── roles/
│   └── {role_name}/
│       └── reviews/{YY-MM-DD}/
│           └── v{version}__response-to-{brief-slug}.md
└── .beads/
    └── issues.jsonl
```

**Rationale**: Mirrors the briefing workflow document exactly. Briefs flow down (human → team), debriefs flow up (team → human). Git-backed, version-controlled, human-readable.

### AD-4: Claude-Flow Swarm/Team → Beads Orchestration

When a Claude-Flow swarm or team is deployed:

1. **Master agent** creates an epic bead for the overall task
2. **Each sub-agent** gets a child bead with `parent-child` dependency on the epic
3. **Sequential dependencies** use `blocks` edges between sub-beads
4. **Agents claim their bead** via `bd update --claim` before starting
5. **On completion**, agents close their bead with `bd close --reason "..."`
6. **On session end**, agents run `bd sync` to persist state
7. **The epic auto-completes** when all child beads are closed

**Rationale**: The user explicitly noted "they coordinate so well, making sure dependencies are met." This maps directly to Beads' `bd ready` (only shows unblocked work) and atomic claiming.

### AD-5: Voice → Bead Binding

When a voice command triggers agent work:

1. `AudioRouter.route_voice_command(user_id, transcription)` identifies the user
2. The resulting task includes `UserContext` with the speaking user's identity
3. A bead is created with `CreatedBy: entity://hop/visionflow/{pubkey}`
4. Agent responses (TTS) reference the bead ID for traceability

**Rationale**: Closes the loop between voice interaction and structured task tracking.

---

## 3. Implementation Plan

### 3.1 Rust Backend Changes

**File**: `src/types/user_context.rs` (NEW)

```rust
pub struct UserContext {
    pub user_id: String,      // npub1... (bech32)
    pub pubkey: String,       // hex secp256k1
    pub display_name: String, // human-readable folder name
    pub session_id: String,   // UUID v4
    pub is_power_user: bool,
}
```

**File**: `src/services/management_api_client.rs` (MODIFY)

- Add `user_context: &UserContext` parameter to `create_task()`
- Include in JSON body: `{ agent, task, provider, user_context: {...} }`

**File**: `src/handlers/ontology_agent_handler.rs` (MODIFY)

- Add auth middleware to validate `AgentContext.user_id` against Nostr session
- Reject requests where user_id doesn't match an authenticated session

**File**: `src/handlers/briefing_handler.rs` (NEW)

- `POST /api/briefs` — create a brief (from voice transcription or text)
- `GET /api/briefs/{user_id}` — list user's briefs
- `GET /api/debriefs/{user_id}` — list user's debriefs
- `POST /api/briefs/{brief_id}/execute` — trigger agent execution of a brief

### 3.2 Management API Changes

**File**: `management-api/middleware/user-context.js` (NEW)

- Extract `user_context` from request body
- Validate required fields (user_id, pubkey, display_name)
- Attach to `request.userContext` for downstream use

**File**: `management-api/utils/process-manager.js` (MODIFY)

- Accept `userContext` in `spawnTask()`
- Create user-scoped workspace: `/home/devuser/workspace/users/{display_name}/tasks/{taskId}`
- Inject env vars: `VISIONFLOW_USER_ID`, `VISIONFLOW_USER_PUBKEY`, `VISIONFLOW_USER_DISPLAY_NAME`
- Create beads epic if `--with-beads` flag set

**File**: `management-api/routes/tasks.js` (MODIFY)

- Accept `user_context` in POST /v1/tasks body
- Pass through to ProcessManager

**File**: `management-api/routes/briefs.js` (NEW)

- `POST /v1/briefs` — store a brief, create epic bead, trigger execution
- `GET /v1/briefs/:userId` — list briefs for a user
- `POST /v1/briefs/:briefId/debrief` — create consolidated debrief from role responses
- `GET /v1/debriefs/:userId` — list debriefs for a user

### 3.3 Container Setup Changes

**File**: `unified-config/entrypoint-unified.sh` (MODIFY)

- Add Phase 5.5: Install `bd` CLI globally (`npm install -g @beads/bd`)
- Initialize beads in workspace: `bd init --prefix vf`
- Generate AGENTS.md with VisionFlow-specific beads instructions

**File**: `CLAUDE.md` in container (MODIFY via entrypoint)

- Add beads workflow instructions:
  - "Use `bd` for all task tracking"
  - "Create beads with full details for PRDs and features"
  - "Complete beads as dependencies are met"
  - "Before session end: update beads and documents, then `bd sync`"
  - "For swarms: create epic bead first, then child beads per agent"

### 3.4 MCP Server Changes

**File**: `mcp-infrastructure/servers/mcp-server.js` (MODIFY)

- Add `beads_create`, `beads_ready`, `beads_claim`, `beads_close`, `beads_sync` MCP tools
- Add `brief_execute`, `brief_debrief` MCP tools for the briefing workflow
- Bind session creation to user context (from Management API env injection)

### 3.5 Briefing Workflow Implementation

**File**: `management-api/services/briefing-service.js` (NEW)

Core briefing workflow orchestrator:

```
createBrief(userContext, content, metadata)
  → writes brief.md to team/humans/{name}/briefs/{date}/
  → creates epic bead with brief content
  → returns briefId

executeBrief(briefId)
  → reads brief
  → identifies roles from content
  → spawns Claude-Flow team with role-specific prompts
  → each role agent gets a child bead
  → agents write responses to team/roles/{role}/reviews/{date}/
  → on completion: createDebrief()

createDebrief(briefId, roleResponses)
  → reads all role responses
  → generates consolidated debrief with links
  → writes to team/humans/{name}/debriefs/{date}/
  → closes epic bead
```

---

## 4. Data Flow Diagrams

### 4.1 User → Agent Task (with Beads)

```
User clicks "Run Agent" in VisionFlow UI
    │
    ├─ Frontend sends POST /api/agent-task with session token
    │
    ├─ Rust handler extracts NostrUser from session
    │   └─ Builds UserContext { user_id: npub, pubkey, display_name, session_id }
    │
    ├─ ManagementApiClient.create_task(agent, task, provider, user_context)
    │   └─ POST http://agentic-workstation:9090/v1/tasks
    │       Body: { agent, task, provider, user_context, with_beads: true }
    │
    ├─ Management API validates user_context
    │   ├─ Creates workspace: /home/devuser/workspace/users/{name}/tasks/{uuid}
    │   ├─ Runs: bd create "{task}" -p 1 --json
    │   │   └─ Returns: { id: "vf-a3f8" }
    │   └─ Sets env: VISIONFLOW_USER_ID, VISIONFLOW_BEAD_ID=vf-a3f8
    │
    ├─ Claude agent starts, reads CLAUDE.md
    │   ├─ Sees: "You are working on bead vf-a3f8 for user {name}"
    │   ├─ Runs: bd update vf-a3f8 --claim --status in_progress
    │   ├─ If swarm needed: creates child beads with deps
    │   │   ├─ bd create "Sub-task 1" -p 1 --parent vf-a3f8
    │   │   ├─ bd create "Sub-task 2" -p 1 --parent vf-a3f8
    │   │   └─ bd dep add vf-a3f8.2 vf-a3f8.1  (2 blocks on 1)
    │   └─ Works through tasks, closing beads as completed
    │
    ├─ On session end (stop hook):
    │   ├─ Agent updates any documents with learnings
    │   ├─ bd close vf-a3f8 --reason "Completed: {summary}"
    │   └─ bd sync (export JSONL, commit, push)
    │
    └─ Management API returns task result to VisionFlow
        └─ Includes bead_id for frontend tracking
```

### 4.2 Briefing Workflow (Voice → Debrief)

```
User records voice memo
    │
    ├─ Otter.ai transcribes → raw text
    │
    ├─ User pastes into VisionFlow or Claude thread
    │   └─ Refinement: raw → structured brief with role assignments
    │
    ├─ POST /api/briefs { content, roles, metadata }
    │   ├─ Writes: team/humans/{name}/briefs/02/13/v0.2.33__daily-brief.md
    │   ├─ Creates epic bead: bd create "Daily Brief 02/13" -p 0
    │   └─ Returns: { brief_id, bead_id, path }
    │
    ├─ POST /api/briefs/{brief_id}/execute
    │   ├─ Identifies roles: [architect, dev, ciso, designer, dpo, devops]
    │   ├─ Creates child beads per role (with parent-child deps)
    │   ├─ Spawns Claude-Flow team:
    │   │   claude --team architect,dev,ciso,designer,dpo,devops
    │   │     --task "Read brief at {path} and respond from your role"
    │   │     --bead-epic vf-{id}
    │   │
    │   ├─ Each role agent:
    │   │   ├─ Claims their bead: bd update vf-{id}.{n} --claim
    │   │   ├─ Reads brief from their perspective
    │   │   ├─ Writes response to team/roles/{role}/reviews/{date}/
    │   │   └─ Closes bead: bd close vf-{id}.{n} --reason "Response filed"
    │   │
    │   └─ Master agent detects all child beads closed
    │
    ├─ Auto-triggers debrief creation:
    │   ├─ Reads all role responses
    │   ├─ Generates consolidated debrief with summary + links
    │   ├─ Writes: team/humans/{name}/debriefs/02/13/v0.2.33__debrief.md
    │   ├─ Closes epic bead
    │   └─ Pushes to GitHub
    │
    └─ Notifies user: "Debrief ready at {github_link}"
```

### 4.3 Claude-Flow Swarm → Beads Coordination

```
Master agent receives complex task
    │
    ├─ Creates epic bead:
    │   bd create "Implement feature X" -p 0 --type epic
    │   → vf-b7c2
    │
    ├─ Decomposes into sub-tasks:
    │   bd create "Design API schema" -p 1 --parent vf-b7c2     → vf-b7c2.1
    │   bd create "Implement backend" -p 1 --parent vf-b7c2     → vf-b7c2.2
    │   bd create "Write tests" -p 1 --parent vf-b7c2           → vf-b7c2.3
    │   bd create "Update docs" -p 2 --parent vf-b7c2           → vf-b7c2.4
    │
    ├─ Sets dependencies:
    │   bd dep add vf-b7c2.2 vf-b7c2.1   (impl blocks on design)
    │   bd dep add vf-b7c2.3 vf-b7c2.2   (tests block on impl)
    │
    ├─ Deploys swarm:
    │   Each agent runs: bd ready --json
    │   → Only vf-b7c2.1 and vf-b7c2.4 are ready (unblocked)
    │   → Agent A claims vf-b7c2.1 (design)
    │   → Agent B claims vf-b7c2.4 (docs, no deps)
    │
    ├─ Agent A completes design → closes vf-b7c2.1
    │   → vf-b7c2.2 (impl) becomes ready
    │   → Agent C claims vf-b7c2.2
    │
    ├─ Agent C completes impl → closes vf-b7c2.2
    │   → vf-b7c2.3 (tests) becomes ready
    │   → Agent D claims vf-b7c2.3
    │
    ├─ All sub-beads closed → epic vf-b7c2 auto-closes
    │
    └─ bd sync → JSONL committed to git
```

---

## 5. Security Considerations

### 5.1 User Context Validation

The Management API must validate that `user_context.pubkey` corresponds to a real VisionFlow user. Two approaches:

**Option A (Implemented here): Trust-but-verify via signed token**
- Rust backend signs the `user_context` with HMAC-SHA256 using `MANAGEMENT_API_KEY`
- Management API verifies the signature before accepting
- This ensures the backend (trusted) is the only source of user context

**Option B (Future): NIP-26 delegation**
- Already scaffolded in `.env.example` (`AGENT_SECRET_KEY`, `AGENT_PUBLIC_KEY`)
- User delegates signing authority to agent container via Nostr delegation token
- Agent actions are cryptographically attributable to the delegating user

### 5.2 Workspace Isolation

Each user's task workspace is isolated:
- `/home/devuser/workspace/users/{display_name}/tasks/{taskId}/`
- Beads database is per-workspace (SQLite in `.beads/`)
- Agents cannot access other users' workspaces (enforced by `cwd` in spawn)

### 5.3 Bead Provenance

All beads carry `CreatedBy` as an EntityRef:
```
entity://hop/visionflow/{pubkey_short}/{display_name}
```

This provides:
- Platform: `visionflow`
- Org: derived from pubkey
- ID: display_name
- Traceable back to Nostr identity

---

## 6. Files Changed

### New Files
| File | Purpose |
|------|---------|
| `src/types/user_context.rs` | UserContext struct |
| `src/handlers/briefing_handler.rs` | Brief/debrief HTTP endpoints |
| `src/services/briefing_service.rs` | Briefing workflow orchestration |
| `multi-agent-docker/management-api/middleware/user-context.js` | User context extraction/validation |
| `multi-agent-docker/management-api/routes/briefs.js` | Briefing workflow API routes |
| `multi-agent-docker/management-api/services/briefing-service.js` | Container-side briefing orchestrator |
| `multi-agent-docker/management-api/services/beads-service.js` | Beads CLI wrapper service |

### Modified Files
| File | Change |
|------|--------|
| `src/services/management_api_client.rs` | Add user_context to create_task |
| `src/types/mod.rs` | Export user_context module |
| `src/main.rs` | Register briefing routes |
| `multi-agent-docker/management-api/server.js` | Register user-context middleware + brief routes |
| `multi-agent-docker/management-api/routes/tasks.js` | Accept user_context in POST body |
| `multi-agent-docker/management-api/utils/process-manager.js` | User-scoped workspaces + beads env |
| `multi-agent-docker/unified-config/entrypoint-unified.sh` | Install bd, init beads |
| `multi-agent-docker/mcp-infrastructure/servers/mcp-server.js` | Add beads + briefing MCP tools |
