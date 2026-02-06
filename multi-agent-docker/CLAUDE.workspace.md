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

## Per-Project
Each project has its own `CLAUDE.md` with swarm config, agent routing, and operational rules.
