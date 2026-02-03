# CLAUDE.local.md - Local Experiments & Private Context

> This file overlays CLAUDE.md with local experiments and private context.
> When a local rule measurably improves outcomes, promote it to CLAUDE.md with an ADR.
> When it fails, it stays local.

## Active Experiments

### Experiment: Enhanced Stop Hook (2026-02-03)
**Status**: ✅ VALIDATED - Ready for promotion
**Hypothesis**: Stop hooks returning JSON with `{"ok": boolean}` field prevents schema validation errors
**Result**: Schema validation errors eliminated in rebuilt container
**Metrics**: 0 errors in 5 session stops vs 100% failure rate before
**Promotion**: Already integrated into entrypoint-unified.sh

### Experiment: MCP Memory Over CLI (2026-02-02)
**Status**: ✅ VALIDATED - Promoted to CLAUDE.md
**Hypothesis**: MCP memory tools provide better cross-agent coordination than CLI
**Result**: 19,659+ entries accessible via MCP vs 47 via CLI
**Metrics**: 100% reliability for memory operations across agent boundaries
**Promotion**: Documented in all coordinating CLAUDE.md files

## Pending Experiments

### Experiment: Guidance Control Plane
**Status**: 🔬 TESTING
**Hypothesis**: Typed constitution with task-scoped shards improves long-horizon autonomy
**Metrics to track**:
- Autonomy duration before intervention
- Cost per successful outcome
- Tool/memory operation reliability
- Runaway loop self-termination rate
**Started**: 2026-02-03

## Local Overrides

### Memory Backend
- Use external RuVector PostgreSQL for all agent coordination
- CLI memory commands are for debugging only

### Hook Configuration
- Stop hooks MUST return `{"ok": boolean}` JSON format
- All hooks should use `|| true` fallback for resilience

## Private Context

### Container Environment
- Container: turbo-flow-unified (rebuilt 2026-02-03)
- External memory: ruvector-postgres:5432
- Docker network: docker_ragflow

### Known Issues
- supervisorctl requires sudo in this container
- npx calls are slower than global claude-flow binary
