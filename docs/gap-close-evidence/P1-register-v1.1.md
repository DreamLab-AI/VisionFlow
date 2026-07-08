# P1 — Gap Register v1.1 (RegisterKeeper first supersede)

| Field | Value |
|-------|-------|
| Register item | Register v1.1 — the immutable status register ADR-004 requires at a wave boundary; `RegisterKeeper` stewardship |
| Canon owner docs | `docs/ADR-004-gap-close-sprint-governance.md` §Governance Cadence ("promotion cuts a new register version"); `docs/PRD-gap-close-canon.md` §"Register Stewardship"; `docs/DDD-gap-close-canon-context.md` §9 (`RegisterKeeper`), Invariant 5 |
| Canary | none — a register publication, not a loop item |
| Branch | `gap-close/2026-07` |
| Base commit at work start | `f72d173cd57d48fcf66c1a8c42628e6e6d11511c` |
| Maturity (honest) | `RegisterKeeper` reaches `integrated` on first publish + supersede with a forward-chained correction. v1.0 = the meta-PRD inventory (published); v1.1 = this supersession with forward-chained corrections → the publish+supersede condition is met. |

## What was cut

`docs/registers/gap-register-v1.1.md` — immutable, versioned. It supersedes the
v1.0 meta-PRD inventory **as a status register** (the meta-PRD stays authoritative
for scope/ownership/wave, ADR-004 Decision 7). It carries:

1. **P0 item status** — every P0 item with its closure SHA(s), post-wave maturity,
   and canary state, sourced from the P0 wave record.
2. **Forward-chained corrections** — REC-1a/1b, PATCH, NIP-42 (closed/reconciled
   pre-sprint) and D1/D5 (v1.0 wording superseded by re-derived findings), each with
   its evidence citation, landing in this superseding version and **not** editing
   the v1.0 inventory in place (Invariant 5).
3. **The P1 open set** — the wave the P0 close opens.
4. **A canary ledger** — the PENDING-LIVE batch.

`docs/architecture/compatibility-matrix.md` gains a §"Gap-Close Sprint — P0 Item
Tiers" table reflecting the P0-closed items at their evidenced tiers, authoritative
to this register.

## P0 status recorded (from the wave record, SHAs verified)

| Item | Owner(s) | SHA(s) | Maturity | Verified |
|---|---|---|---|---|
| RES-a liveness + canary harness | VisionClaw | `6f4eb1b0a`, `1492bc17b` | `integrated` | ✓ `git log` |
| RES-b diagram render gate | VisionFlow (canon) | `f72d173cd` | `scaffolded` | ✓ HEAD |
| COM-13 agent disclosure | forum (+ canon norm) | `7157a92`, `fb7826859` | forum `integrated`; canon `standalone` | wave record |
| COM-14 `did:nostr` keying | VisionClaw + agentbox | `4a595cc8f`; `6189f47d` | `integrated` | ✓ `git log` (both) |
| REC-2 broker kernel + case queue | VisionClaw | `c9f2e3539` | `integrated` (P0 slice) | ✓ `git log` |
| D5 MCP-status honesty | VisionClaw | `6f4eb1b0a` | `integrated` | ✓ `git log` |
| REC-1 governed-writeback floor | VisionClaw · solid-pod · forum | `6f4eb1b0a`; `791977a` | `integrated`/`reconciled` | wave record |

## Forward-chained corrections recorded

- **REC-1a/1b** — closed/reconciled pre-sprint, verify-only (`6f4eb1b0a`, `tests/rec1_route_guard.rs`).
- **PATCH** — closed pre-sprint, closure-with-receipt vs a stale entry (solid-pod `791977a`, 3/3 tests).
- **NIP-42** — reconciled pre-sprint: edge relay gates by pubkey whitelist (standalone-first), not NIP-42 AUTH; wording superseded.
- **D1** — v1.0 "channel inert" wording superseded: the beam ships over `/wss/agent-events` (ADR-059); residual narrows to live traffic (P1 canary).
- **D5** — v1.0 "fabricated status" wording superseded and closed: status from `check_mcp_metrics` + real WS (`6f4eb1b0a`).

## Receipt — SHA verification, 2026-07-08

```
$ git -C VisionFlow rev-parse HEAD
f72d173cd57d48fcf66c1a8c42628e6e6d11511c
$ git -C project log --oneline -1 6f4eb1b0a
6f4eb1b0a feat(gap-close): RES-a D5 REC-1a REC-1b — LivenessHarness, KG watchdog, status honesty, ontology route-guard
$ git -C project log --oneline -1 4a595cc8f
4a595cc8f feat(gap-close): COM-14 did:nostr keying of agent nodes (consumer side)
$ git -C project log --oneline -1 c9f2e3539
c9f2e3539 feat(gap-close): REC-2 broker kernel + ACSP case-queue events; supersede crashbug BrokerActor
$ git -C project log --oneline -1 1492bc17b
1492bc17b fix(gap-close): RES-a Nostr-relay tap for Nostr-only canary fires (WP-11 AC3)
$ git -C project/agentbox log --oneline -1 6189f47d
6189f47d feat(gap-close): COM-14 mint per-agent did:nostr at spawn (source side)
$ git -C project/agentbox log --oneline -1 d13f8688
d13f8688 feat(gap-close): RES-d script-queryable skill-count source of truth
```
(forum `7157a92`/`fb7826859` and solid-pod `791977a` are recorded from the P0 wave
record `project-state:gap-close-p0-wave-record-2026-07-08`; the forum/solid-pod
checkouts are not present on this box.)

## Acceptance mapping

| Criterion | State |
|-----------|-------|
| Immutable, versioned register cut at the wave boundary | `gap-register-v1.1.md` published, immutability stated. |
| Supersedes the meta-PRD inventory as a status register (meta-PRD stays) | Recorded in §"What this is". |
| Every P0 item with post-wave status/maturity/canary state | P0 status table. |
| Drift corrections chained forward with evidence citations, not edited in place | Forward-chained-corrections table (Invariant 5 honoured). |
| P1 open set stated | §"P1 open set". |
| `compatibility-matrix.md` rows for P0-closed items at evidenced tiers | §"Gap-Close Sprint — P0 Item Tiers" added. |
| `RegisterKeeper` published v1 and superseded ≥1 with a forward-chained correction | v1.0 → v1.1 with five forward-chained corrections. |

## Falsification exposure

Falsified if any P0 item is claimed above its evidenced tier (RES-b and RES-d held
at `scaffolded`; canaries stated PENDING-LIVE), if a correction edits the v1.0
inventory in place (it does not — corrections chain forward into v1.1), or if a
canon loop item is scored `integrated` without its canary firing live (none is).
