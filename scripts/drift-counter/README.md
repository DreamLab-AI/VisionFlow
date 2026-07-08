# drift-counter (RES-d)

The canon's self-description drift gate. It single-sources four counted axes
from their substrate-exposed sources of truth and fails CI when any canon figure
disagrees, or when a second distinct figure appears at a policed site.

- **PRD:** [`docs/PRD-gap-close-canon.md`](../../docs/PRD-gap-close-canon.md) §"Automated Self-Description Counter (RES-d)"
- **ADR:** [`docs/ADR-005-gap-close-canon-decisions.md`](../../docs/ADR-005-gap-close-canon-decisions.md) §Decision 2
- **DDD:** [`docs/DDD-gap-close-canon-context.md`](../../docs/DDD-gap-close-canon-context.md) §9 (`DriftCounter`)
- **Canary:** `CANARY-CANON-DRIFT`
- **Evidence:** [`docs/gap-close-evidence/P1-RES-d.md`](../../docs/gap-close-evidence/P1-RES-d.md)

## Axes

| Axis | Source of truth | Exposed by | State |
|---|---|---|---|
| `skills` | `scripts/skill-count-check.js` (`.count`) — invoked | agentbox | enforced |
| `mcp-ontology-tools` | `mcp/servers/ontology-bridge.js` `TOOLS` array length | agentbox | enforced |
| `ontology-classes` | Oxigraph `owl:Class` count / committed count file | VisionClaw | unavailable this wave (partial-source) |
| `roster` | forum `agent_registry` | nostr-rust-forum | planned |

## Run

```sh
DRIFT_AGENTBOX_DIR=/path/to/agentbox node scripts/drift-counter/drift-counter.mjs
node scripts/drift-counter/drift-counter.mjs --json     # machine output
node scripts/drift-counter/drift-counter.mjs --quiet    # exit code only
node scripts/drift-counter/drift-counter.mjs --strict   # unavailable axis fails too
```

Exit `1` on any drift on an enforced axis; exit `0` otherwise. In CI, agentbox is
checked out beside the canon and `DRIFT_AGENTBOX_DIR` points at it (see
`.github/workflows/drift-counter.yml`).

## Design: allowlist-anchored, not a blind grep

`allowlist.json` names every policed assertion site. The gate does **not** grep
the whole tree for `\d+ skills`, because the tree legitimately carries distinct
figures about other subjects — a Ramp/Glass case study's "350 skills", the book's
rhetorical "10-15 agent skills" pilot range, VisionClaw's native "7 MCP tools",
agentbox's "180+ MCP tools" total. None of those describe a tracked axis. Two
match modes:

- **`scan`** (skills): every occurrence of the axis pattern in each listed live
  self-description file must equal the queried truth. The book prose under
  `presentation/` is deliberately excluded from the scan (its current-count
  references are reconciled by hand); only the live docs are scanned.
- **`sites`** (mcp-ontology-tools): each pinned, context-anchored site pattern
  must match once and equal the truth — precise enough never to match the native
  or total MCP-tool figures that share the "MCP tools" phrasing.

Adding a new self-description site requires an allowlist entry; that is the review
contract that keeps the axis single-sourced.

## Partial-source failure mode (ADR-005 §Decision 2)

An axis whose source is not exposed this wave is reported `UNAVAILABLE` and is not
enforced — a source being down blocks *that* axis, not the whole gate. A figure
that *disagrees* with an *available* source is a hard failure. `--strict` turns
unavailability into a failure for a wave that requires every source live. Set
`DRIFT_VISIONCLAW_CLASS_COUNT` (integer) or `DRIFT_VISIONCLAW_CLASS_COUNT_FILE`
(path to a committed count) to enforce the ontology-classes axis once VisionClaw
publishes its `ClassCountSource`.
