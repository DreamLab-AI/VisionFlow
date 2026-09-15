---
id: ADR-2011
title: Operator-declared task properties set the human–agent boundary, not the requesting agent's self-tier
date: 2026-09-14
decision_status: accepted
implementation_status: complete
activation_status: staged
supersedes: []
superseded_by: []
verified_commit: 03db67106e8d5aab6e6a992095107f14df3fc886
owner: jjohare
review_trigger: nostr-bbs-core publishing TaskProperties, or agentbox authority_class gaining a third class
repo: visionflow
domain: BASELINE-visionflow.md
lineage: applies ADR-2010's lens to the one field that today decides escalation; builds on agentbox authority_class (zero-tolerance | recoverable) as the reversibility seed and on the forum's advertised-but-unenforced ESCALATION_DEFAULT_* (wrangler.toml:48-49).
---

# ADR-2011 — Operator-declared task properties set the human–agent boundary, not the requesting agent's self-tier

## Context

An `ActionRequest` carries one boundary signal, `risk_tier`, which the requesting agent declares (`nostr-bbs-core/src/governance.rs:129-176`, doc comment: "the agent's declaration stands"). No human or relay path re-tiers it. The relay advertises `ESCALATION_DEFAULT_TIER=medium` and `ESCALATION_DEFAULT_POSTURE=escalate_to_human` in NIP-11 but enforces neither. agentbox's `authority_class` (`agentbox.toml:822-849`) is operator-declared per action class, the right shape, but encodes only reversibility. arXiv 2609.12482 sets the boundary by three properties of the *task*: verifiability, reversibility, stakes. Today the party with the strongest incentive to under-tier is the only party that tiers.

## Decision

The escalation posture for a governance case derives from an operator-declared **task-property triple** on the panel, which a request may tighten but never loosen.

1. `nostr-bbs-core` owns `TaskProperties { verifiability: Inspectable | Partial | Opaque, reversibility: Reversible | Compensable | Irreversible, stakes: Bounded | Significant | Critical }`, carried as tags on `PanelDefinition` (31400) as the panel default and optionally on `ActionRequest` (31402).
2. `effective_tier(panel_props, request_props, agent_tier)` is a pure, total function in `nostr-bbs-core`: `Irreversible` or `Critical` ⇒ ≥ `high`; `Opaque` ⇒ ≥ `medium` and never member-suppressed; else `max(panel_default, agent_tier)`. Only the effective tier is stored, rendered, or used for suppression.
3. The relay enforces the advertised default: an unlabelled request receives `ESCALATION_DEFAULT_TIER`; an effective `high` or `critical` case resolves only by a human 31403 (or the ADR-2010 manual-continuation receipt after an Approve).
4. agentbox derives a request's default triple from `authority_class` (`zero-tolerance` ⇒ `Irreversible`, `recoverable` ⇒ `Compensable`; verifiability and stakes from the skill manifest, defaulting to `Partial` / `Significant`) and stamps it on every 31402 it publishes.
5. `risk_tier` remains as the agent's declaration for telemetry (declared vs effective is itself a calibration signal) but carries no authority on its own.

## Consequences

- The relay gains a small amount of policy (a pure function and a default), which the forum's own ADR must record; the schema stays owned by `nostr-bbs-core` and is republished as a crate.
- Legacy panels without a triple behave as today except that unlabelled requests fold to the advertised default rather than to `medium` by accident.
- Agents that habitually under-tier become visible: declared-vs-effective divergence is a reviewer-telemetry column.
- Cost: operators must declare the triple when publishing a panel; the `governance_publish_panel` tool and the forum UI prompt for it.

## Verification

- `cargo test -p nostr-bbs-core effective_tier` exercises the boundary table in the PRD (FR3) including the property-based invariant that a request can only raise the tier.
- Relay integration test: an unlabelled 31402 projects with the NIP-11 default tier; a `high` case cannot reach `Decided` via any non-31403 path.
- agentbox unit test: `governance_request_action` for a `zero-tolerance` skill emits `reversibility=irreversible`.
- `verified_commit` set when the three tests are green on their respective branches; `activation_status` moves to `staged` when the crate is published and to `live` on edge deploy.

### Merged at main, crates published (2026-09-15)

The `feat/augmentation-conditions` branches merged to `main` on all three substrates: nostr-rust-forum `11b674c`, agentbox `b859b37d2`, VisionClaw `8f9affe49`. `nostr-bbs-core` and `nostr-bbs-mesh` are republished at `1.0.0-beta.11` (PRD-augmentation-conditions M3). `effective_tier` (`crates/nostr-bbs-core/src/governance.rs:516`), the reviewer-role `Delegate` admission gate (`crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:234-249`), and the agentbox `reversibilityFor` derivation (`management-api/lib/task-properties.js:111-113`) are all present at these heads — the re-graded compatibility matrix cites them directly. `implementation_status` moves to `complete`: the client-side gap noted at the prior `partial` grading (rendering `effective_tier`, hiding the `probe` tag) is closed — see `docs/architecture/compatibility-matrix.md` §Augmentation conditions, C2/C6 forum cells. `activation_status` moves to `staged`: the crates are published but nothing is deployed to the live edge forum (M4 not started). `verified_commit` is left `pending` for the queen to set against the actual deploy commit.

### Executed evidence (branch `feat/augmentation-conditions`, not merged, not deployed)

- **nostr-rust-forum @ `aa438f3`** (evidence commit `c2e4ef7e54a0df9065c54a5784a433082d08c80d`, `.claude/evidence/EXP-AC-003.evidence.md`): `cargo test -p nostr-bbs-core --lib governance::task_property_tests` — 12 passed, 0 failed, including `effective_tier_table_holds_for_every_triple_and_tier` (`crates/nostr-bbs-core/src/governance.rs:2893`) and the tightening-only property `merge_is_tightening_only_over_all_729_pairs`, which enumerates the full 27×27 triple space rather than sampling it. `effective_tier()` itself is `crates/nostr-bbs-core/src/governance.rs:516-534`; `TaskProperties` is `governance.rs:391-397`.
- **agentbox @ `19463a588`** (`tests/sovereign/task-properties.test.js`): `zero-tolerance ⇒ irreversible` is asserted at `tests/sovereign/task-properties.test.js:53-54` and again as `EXP-AC-003: a zero-tolerance action class derives reversibility=irreversible` at `:68-70`; the derivation function is `management-api/lib/task-properties.js:100-113` (`reversibilityFor`). Jest run recorded in `.claude/evidence/EXP-AC-003.evidence.md`: 20 passed, 20 total.
- **Caveat carried from `WIP-STATUS-augmentation-conditions.md`** (nostr-rust-forum, `aa438f3`): `deepsec-gate` is `BLOCK`, not `PASS` (2 pre-existing HIGH findings, untriaged at pause); the forum-client half (rendering `effective_tier`, hiding the `probe` tag) is not started on this branch. This is the basis for `implementation_status: partial` below.

**`implementation_status: partial`** (client half pending on the forum branch). **`activation_status: inactive`** (nothing deployed on any of the three branches). `verified_commit` is left as `pending` above — set by the queen when the branches merge and the matrix is re-graded, per ADR-2010 §Verification.
