# 🔍 Agent 1: Code Auditor - Mission Brief

**Agent ID:** code-auditor
**Type:** Analyst
**Priority:** High
**Compute Units:** 15
**Memory Quota:** 256 MB

## Mission Statement

Audit ALL dependencies on `src/actors/graph_actor.rs` (4,566 lines). Identify every API route, handler, service, and actor that depends on GraphServiceActor. Create comprehensive dependency map showing what needs migration.

## Specific Tasks

### 1. Direct Dependencies Audit
```bash
# Find all imports of GraphServiceActor
grep -r "use.*graph_actor::" src/
grep -r "GraphServiceActor" src/
```

### 2. API Route Analysis
Identify routes in:
- `src/handlers/api_handler/graph/mod.rs`
- `src/handlers/socket_flow_handler.rs`
- `src/handlers/settings_handler.rs`

### 3. Handler Coupling Analysis
Map dependencies in:
- HTTP handlers calling GraphServiceActor
- WebSocket handlers using graph actor
- Actor-to-actor message passing

### 4. Database/Storage Dependencies
- Knowledge graph repository usage
- Metadata store coupling
- State persistence patterns

### 5. WebSocket Real-time Updates
- Client coordinator interactions
- Binary protocol dependencies
- Real-time event propagation

## Deliverables

Create: `/home/devuser/workspace/project/docs/migration/audit-graph-actor-dependencies.md`

**Required Sections:**
1. **Direct Import Map** - All files importing graph_actor
2. **API Route Dependencies** - Every route using GraphServiceActor
3. **Actor Message Flows** - Inter-actor communication patterns
4. **Data Flow Diagram** - How data flows through monolith
5. **Critical Paths** - Mission-critical functionality
6. **Migration Complexity Score** - Per-file migration difficulty (1-10)
7. **Dependency Priority Matrix** - What to migrate first

## Memory Storage

Store findings under: `hive-coordination/audit/graph_actor_dependencies`

**JSON Structure:**
```json
{
  "total_dependencies": 25,
  "critical_paths": ["websocket_updates", "github_sync", "physics_simulation"],
  "api_routes": [
    {"path": "/api/graph/nodes", "handler": "get_nodes", "complexity": 7},
    ...
  ],
  "actor_messages": [
    {"from": "ClientCoordinator", "to": "GraphService", "type": "BatchUpdate"},
    ...
  ],
  "migration_order": ["reads_first", "writes_second", "websocket_third"]
}
```

## Coordination

### Before Starting
```bash
npx claude-flow@alpha hooks pre-task --description "Code audit: GraphServiceActor dependencies"
npx claude-flow@alpha hooks session-restore --session-id "hive-hexagonal-migration"
```

### During Work
```bash
# After analyzing each subsystem
npx claude-flow@alpha hooks notify --message "Analyzed API routes: 12 dependencies found"
```

### After Completion
```bash
npx claude-flow@alpha hooks post-task --task-id "code-auditor-graph-dependencies"
```

## Success Criteria

✅ All 25+ dependencies documented
✅ API routes mapped to hexagonal layer
✅ Critical paths identified
✅ Migration complexity scored
✅ Findings stored in memory
✅ Report delivered in markdown

## Tools Available

- `grep`, `ripgrep` for code search
- `wc -l` for line counts
- `tree` for directory structure
- `cargo tree` for dependency graph
- Memory storage via MCP tools

## Report to Queen

Upon completion, notify Queen Coordinator:
- Total dependencies found
- Critical path analysis
- Recommended migration order
- Estimated migration effort (story points)

**Expected Duration:** 30-45 minutes
**Blocker Escalation:** Report to Queen immediately

---
*Assigned by Queen Coordinator*
