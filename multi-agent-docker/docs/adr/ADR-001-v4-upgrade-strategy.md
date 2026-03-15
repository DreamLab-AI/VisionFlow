# ADR-001: V4 Upstream Upgrade Strategy

## Status
Accepted

## Date
2026-03-12

## Context

The upstream turbo-flow repository has released V4.0, introducing three new systems (Beads, GitNexus, native worktree helpers), consolidated Ruflo plugins, and Agent Teams support. Our multi-agent-docker container has significantly diverged from upstream — we run CachyOS with CUDA/GPU, Blender 5.x, QGIS, KiCAD, 5 isolated users, supervisord orchestration, a management API, Z.AI service, and 62+ MCP skills. Upstream V4 targets devcontainer/DevPod environments with debian/ubuntu.

## Decision

**Selective adoption, not wholesale migration.**

We will cherry-pick V4 innovations into our existing build system without disrupting our divergent architecture. Specifically:

### ADOPT (direct integration)
1. **Beads** (`beads-cli`) — Cross-session project memory via git-native JSONL
2. **GitNexus** (`gitnexus`) — Codebase knowledge graph with blast-radius detection
3. **Native worktree helpers** — `wt-add`, `wt-remove`, `wt-list`, `wt-clean` bash functions
4. **Agent Teams** — `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` env var
5. **6 Ruflo plugins** — agentic-qe, code-intelligence, test-intelligence, perf-optimizer, teammate, gastown-bridge
6. **OpenSpec** — spec-driven development (`@fission-ai/openspec`)

### ADAPT (enhance existing)
7. **V4 aliases** — Merge rf-*, bd-*, wt-*, gnx-* into existing turbo-flow-aliases.sh
8. **Statusline Pro v4.0** — Upgrade existing statusline.sh with V4 powerline design
9. **CLAUDE.md template** — Add 3-tier memory protocol, isolation rules, Agent Teams guardrails
10. **Entrypoint** — Add Beads init, GitNexus indexing, plugin verification phases

### REJECT
- Full V4 setup.sh (designed for devcontainer, not our unified Dockerfile)
- Workspace directory structure changes (we have established layout)
- Platform-specific boot scripts (we have unified approach)
- FR8 compatibility layer (no external consumers)
- Removal of tools we still use (agent-browser CLI, agentic-jujutsu standalone)

## Consequences

- Dockerfile grows by ~30 lines (npm globals + plugin installs)
- Entrypoint gains 2 new phases (Beads init, GitNexus indexing)
- turbo-flow-aliases.sh gains ~80 new aliases/functions
- Version label bumps from 3.0.0 to 4.0.0
- No breaking changes to existing services, ports, or volumes

## Bounded Contexts (DDD)

### BC1: Build System (Dockerfile.unified)
- Add npm globals: beads-cli, gitnexus, @fission-ai/openspec
- Add Ruflo plugin installation phase
- Add CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 env var
- Bump version labels to 4.0.0

### BC2: Runtime Initialization (entrypoint-unified.sh)
- Add Phase 5.6: Beads initialization
- Add Phase 5.7: GitNexus codebase indexing
- Add Phase 6.8: Ruflo plugin verification
- Add Phase 6.9: Agent Teams configuration

### BC3: Shell UX (turbo-flow-aliases.sh)
- Add rf-* aliases (Ruflo)
- Add bd-* aliases (Beads)
- Add wt-* functions (worktrees)
- Add gnx-* aliases (GitNexus)
- Add aqe-* aliases (Agentic QE)
- Add hooks-*, neural-*, mem-* aliases
- Add turbo-status and turbo-help functions
- Preserve existing cf-* as backwards-compat

### BC4: Statusline (statusline.sh)
- Replace with V4 powerline design (3-line, project/model/git/tokens/cost)

### BC5: Workspace Context (CLAUDE.md, CLAUDE.workspace.md)
- Add 3-tier memory protocol (Beads > Native Tasks > AgentDB)
- Add isolation rules (worktrees per agent)
- Add Agent Teams guardrails
- Add GitNexus usage instructions
- Add cost guardrails

### BC6: Compose & Labels (docker-compose.unified.yml)
- Bump version labels to 4.0.0
- Add CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS env var

## Definition of Done

1. `docker compose -f docker-compose.unified.yml up -d --build` succeeds
2. `beads-cli`, `gitnexus`, `@fission-ai/openspec` available in container
3. `bd init` works in a git repo inside container
4. `gnx-analyze` indexes a workspace
5. `wt-add test-agent` creates isolated worktree
6. V4 aliases functional (`rf-*`, `bd-*`, `wt-*`, `gnx-*`)
7. Statusline shows V4 powerline design
8. No regression in existing services (SSH, VNC, management API, Z.AI, skills)
