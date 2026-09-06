---
title: Engineering harness and mandate governance
status: source-and-isolated-probe-verified
date: 2026-09-05
type: explanation
---

# Engineering harness and mandate governance

VisionFlow's engineering ADR-004/005 form a separate decision sequence from the archived canon ADRs bearing the same numbers. The harness framework is an accepted design with partial mechanisms and an explicitly deferred validation phase. Mandate-at-grant remains speculative. Neither should be read as an implemented estate-wide authority system.

## What the fitness gate measures

The [actual audit-script receipts](evidence/harness-audit-probe.json) report the current templates as 100% paired and 33/40 controls “source-backed”. The latter number subtracts controls explicitly marked planned. The script does not resolve the remaining source paths or execute the sensors.

Two temporary fixtures establish this limit directly. A guide and sensor pointing to nonexistent paths still report 2/2 source-backed and pass. Repeating their pairing produces a 200% ratio and passes. This is a declaration-counting mechanism, not an independently verified coverage measure.

The hosted workflow source runs the audit with a 50% threshold, versus the script's default 80%. A separate job checks JSON syntax, presence of selected fields, unique guide/sensor IDs and referential integrity. It does not invoke the JSON Schema or reject repeated pairing edges. The fixture with nonexistent source paths is not a complete schema-valid template; the probe demonstrates the audit script's scope, not a hosted workflow run. Even the workflow's additional checks do not resolve control sources or establish runtime enforcement.

Closeout should count distinct valid coverage edges, resolve source identities and distinguish declared, present, exercised and enforced controls. Schema validity, branch protection and release gating need their own evidence. A topology label alone cannot constrain process privileges or tool effects.

## Framework decisions: section-level disposition

| ADR-004 section | Evidence-qualified disposition | Acceptance obligation |
|---|---|---|
| D1 guide/sensor model | Templates and audit implement declared pairing | Verify source and distinct coverage; retain missing controls explicitly |
| D2 governance templates | JSON templates/schema exist; CI applies narrower hand-written checks | Validate full schema and actual consumer inspection at a declared revision |
| D3 validation lifecycle | Explicitly deferred in the decision; planned controls remain | Implement and prove validation before learning, or retain deferral with consequences |
| D4 blocking/advisory sensors | Specific workflow gates exist; universal blocking policy unproved | Inventory sensor entry points and demonstrate rejection/reporting through each intended gate |
| D5 mesh smoke extension | Cross-repository journey remains an acceptance obligation | Execute the complete pinned topology; protocol text does not establish a successful run |
| D6 precedent feedforward | Agentbox PrecedentService has storage/match helpers and requires an injected store | Trace authorised promotion, durable storage, scoped matching, application and retirement through production consumers |
| D7 topology commitment | Design-scoping principle | Bind declared topology to enforced capabilities and observe denied crossings |
| D8 canon ownership | Canon owns template artefacts in this tree | Confirm maintainer adoption and revision-bound cross-repository references |

PrecedentService searches five results, skips retired records and uses a similarity threshold. Those helpers do not prove that similarity is sufficient authority to apply an action. Preserve the earlier correction that decision application code exists; do not resurrect the withdrawn “open steering loop” premise. Current [governance](forum-decisions.md), [shared-memory](shared-memory.md) and [consultation](agent-grounding-and-governance.md) findings identify additional persistence, identity and acceptance boundaries. This pass did not invoke the precedent service or write any precedent.

## Mandate proposal: section-level disposition

The current harness schema disallows additional root properties and has no mandates property. A bounded search of agentbox's manifest, management-api and services found no 31406, current_mandates or mandate_blast_radius implementation in the searched JS/CJS/TOML files. That is not an estate-wide absence proof. Existing signed pod/voice mandate mechanisms are related primitives, not evidence that this nine-part provisioning proposal is implemented.

| ADR-005 section | Disposition and closeout evidence required |
|---|---|
| D1 grant-time checkpoint | Speculative: prove approval occurs before effective capability grant, including restart and failure |
| D2 account/file/tool inventory | Speculative schema extension: define authoritative issuer, resource scope, lifetime and revision; validate inventory against effective grants |
| D3 composition graph | Speculative analysis: identify actual effects and test transitive reach against executor behaviour |
| D4 tool taxonomy | Proposed classification: assign complete callable-surface coverage and handle operations whose scope depends on arguments |
| D5 31406 grant event | Proposed protocol: reconcile kind admission, signature/issuer checks, correlation, replay and durable acknowledgement |
| D6 mandate routing lanes | Proposed policy: prove conservative fallback, precedent authority and scoped application with revocation |
| D7 mandate sensors | Proposed observation: bind actual use to the granted identity/resource/tool set and demonstrate detection or denial |
| D8 topology-bounded authority | Proposed enforcement: verify negative cross-topology cases; a declaration alone is insufficient |
| D9 current mandates on requests | Proposed context propagation: bind requests to an immutable, current grant reference and reject stale/revoked authority |

The vision is useful: review an agent's effective capabilities before it acts. The gap is the connection from that review to the actual executor, storage authority and revocation lifecycle. The existing [voice mandate review](agent-grounding-and-governance.md#voice-speaker-target-and-mandate-scope) shows why a valid signed document alone cannot close that connection.

CP-01/04/05/07/08/09 requires explicit adoption or deferral of each section, declared owner roles, source-bound control evidence and negative/recovery journeys. Preserve accepted/deferred/speculative distinctions. The isolated audit probes establish measurement limits; they do not certify hosted CI or runtime governance.
