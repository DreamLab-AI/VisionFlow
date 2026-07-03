# Engineering: Harness Engineering Framework

Framework for systematic regulation of agent behaviour through paired feedforward/feedback controls. Based on analysis of Martin Fowler's *Harness Engineering for Coding Agent Users* (2025) applied to the VisionFlow multi-repo ecosystem.

## Documents

| Document | Purpose | Status |
|---|---|---|
| [PRD-harness-engineering.md](PRD-harness-engineering.md) | Product requirements — goals, milestones, acceptance criteria | Active |
| [ADR-004-harness-engineering-framework.md](ADR-004-harness-engineering-framework.md) | Architecture decisions — guide/sensor model, template schema, execution lifecycle | Accepted (D3 `hooks.validate` Deferred; 2026-07-03 amendment) |
| [DDD-harness-engineering-context.md](DDD-harness-engineering-context.md) | Domain model — aggregates, events, services, invariants | Living |
| [PRD-mandate-at-grant.md](PRD-mandate-at-grant.md) | Mandate-at-grant governance — provisioning-time authorization | Speculative |
| [ADR-005-mandate-at-grant-governance.md](ADR-005-mandate-at-grant-governance.md) | Architecture decisions — three-channel mandates, capability graph, kind 31406 | Speculative |

## Artefacts

| Artefact | Location | Purpose |
|---|---|---|
| Template schema | `schemas/harness-template.schema.json` | JSON Schema for harness template validation |
| Canonical templates | `templates/*.json` | 4 topology-specific guide-sensor bundles |
| Pairing audit | `../../scripts/harness-audit.sh` | Automated pairing ratio **and source-backing** calculation |
| Fitness gates CI | `../../.github/workflows/harness-fitness-gates.yml` | CI workflow for merge-time template JSON validation |

## Quick Start

```bash
# Audit current pairing ratios
bash scripts/harness-audit.sh

# Inspect a template (via MCP)
# harness_inspect --topology governance-decision

# Run fitness gates locally
jq empty docs/engineering/templates/*.json && echo "All templates valid"
```

## Key Concepts

- **Guide** — feedforward control shaping behaviour before execution
- **Sensor** — feedback control observing and correcting after execution
- **Pairing** — binding between a guide and its validating sensor
- **Harness template** — declarative bundle of guides + sensors + pairings per topology
- **Fitness gate** — computational sensor promoted to a blocking CI check
- **`source_status`** — per-control honesty flag: `present` (source resolves to real code today) or `planned` (declared but not yet implemented). `harness-audit.sh` reports paired-ratio and source-backed-ratio separately so a template cannot look complete while citing sources that do not exist.

## Tooling honesty

Two limits are worth stating plainly:

- **`harness_validate` (MCP) is a substrate-coverage linter, not a per-guide constraint checker.** It flags when a summary fails to mention a topology's substrates; it does **not** evaluate whether agent output honoured the individual blocking guides (git-marks, WAC, NIP-98, etc.). Do not cite it as an enforcement gate for guide constraints.
- **The pairing ratio counts declared guide→sensor cross-references, not source-path existence.** `source_status` (above) and the audit's source-backed line exist because a 100% pairing ratio previously masked several controls whose `source` pointed at files that did not exist.
