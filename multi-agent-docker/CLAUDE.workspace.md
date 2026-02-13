# Turbo Flow Workspace

## Discovery
```bash
claude-flow doctor          # System diagnostics
supervisorctl status        # Running services
ls /home/devuser/agents/    # 610 agent templates
```

## Protocols
| Protocol | CLI |
|----------|-----|
| AISP 5.1 | `aisp validate`, `aisp binding A B` |
| Claude Flow v3 | `claude-flow` (global) |

## RuVector Memory
`ruvector-postgres:5432` | DB: `ruvector` | Connection: `$RUVECTOR_PG_CONNINFO`
Always use MCP memory tools (`mcp__claude-flow__memory_*`), never CLI.

## Browser Automation
`agent-browser` v0.7.6: `open <url>`, `snapshot -i`, `click @ref`, `fill @ref "text"`, `screenshot`, `eval`, `close`

## Beads Task Tracking
Beads provides dependency-aware structured task tracking for agentic work.

### Core Workflow
```bash
bd init --prefix vf            # Initialize (already done in workspace)
bd create "Task title"         # Create a task bead
bd create "Epic" --type epic   # Create an epic (parent container)
bd create "Sub" --parent <id>  # Create child under epic
bd dep add <child> <parent>    # Add dependency (child blocked until parent closes)
bd ready --json                # Get unblocked work (dependencies resolved)
bd update <id> --claim         # Atomically claim a bead (prevents double-work)
bd close <id> --reason "Done"  # Close bead (unblocks dependents)
bd sync                        # Persist state — ALWAYS run before ending session
```

### Agent Rules
- **Before starting work**: Run `bd ready` to find unblocked beads
- **Claim before working**: Run `bd update <id> --claim` to prevent conflicts
- **Close when done**: Run `bd close <id> --reason "..."` to unblock dependents
- **Before ending session**: ALWAYS run `bd sync` to persist state
- **User attribution**: Use `BEADS_ACTOR` env var (set automatically from VisionFlow user)

### MCP Tools (preferred over CLI)
Use `beads_create`, `beads_ready`, `beads_claim`, `beads_close`, `beads_show`, `beads_sync`, `beads_dep` MCP tools for structured access.

### Briefing Workflow
Briefs flow down (human → team), debriefs flow up (team → human):
- Briefs: `team/humans/{name}/briefs/{MM}/{DD}/`
- Role responses: `team/roles/{role}/reviews/{YY-MM-DD}/`
- Debriefs: `team/humans/{name}/debriefs/{MM}/{DD}/`

## Per-Project
Each project has its own `CLAUDE.md` with swarm config, agent routing, and operational rules.
