# Roadmap

**Status:** Rewritten 2026-09-06. The 2026-05-22 four-phase plan is superseded; the prior text is at `git show 3db4785:docs/roadmap.md`.
**Governed by:** [ADR-2007 Estate Closeout Evidence Roadmap](adr/ADR-2007-estate-closeout-evidence-roadmap.md) and the [closeout programme](estate-review/closeout/README.md), which now own estate-wide sequencing. This file is the short VisionFlow-side pointer, not a second plan.

## Why this changed

Phases 0 to 3 were written before the estate review existed. The closeout corpus replaced the phase model with nine completion packages (CP-01 to CP-09) that carry named owners, dependencies and exit criteria. Tracking the same work twice, in two shapes, was producing drift rather than progress.

## Phase 0, honesty and traceability: subsumed

Fully absorbed by the estate-review and closeout corpus, committed at `3db4785`. The closeout programme's CP-01 (decision and release identity) and CP-08 (delivery and recovery) carry this intent with stricter criteria than the original table. VisionFlow's own sprint work lands the same goal: the drift counter is repaired and re-pinned to agentbox `eb7794b1`, the release roster grew from 6 to 14 repositories with provenance, and the harness audit was de-duplicated from 82.5% to 79.5%. Do not re-open this section; raise gaps against CP-01 or CP-08.

## Phase 1, mesh contract: still the live gap

The only phase whose substance is genuinely unfinished, and the reason this file still exists.

- **IS-Envelope schema owner.** IS-Envelope v1 types, JCS canonicalisation, validation and LDN/AS2 mapping exist in `nostr-rust-forum` (`crates/nostr-bbs-mesh`, ADR-075). No repo outside the forum routes them, and no cross-repo canonical owner is declared, so the contract is implemented but not adopted.
- **Transport and auth.** `MeshTransport`, `PeerManager`, gift-wrapped kind-1059 send/receive and NIP-42 all exist in the same crate. This is new since the original roadmap and closes the "NIP status per substrate" line for the forum specifically.
- **Consumer routing.** Still unimplemented. agentbox's `[mesh]` keys (`peer_relays`, `federated_kinds`, `allowed_remote_dids`) have no code consumers and relay fan-out defaults off. Tracked as M-4 in the unified register (`docs/TODO-unified.md` in the VisionClaw repo).
- **DID document service fields.** Unchanged. The federation-identifier work that landed (agentbox ADR-2025 and ADR-2061, a protocol and event-kind registry with a CI parity fixture) is adjacent but does not normalise pod, WebID or relay discovery fields.

## Phase 2, end-to-end proof: partial

A real agentbox to agentbox smoke test now exists: the machinelearn to HP-Desktop NIP-98 door was exercised under live traffic (401 `pubkey_not_allowed`, then 200 after allowlisting) with the reverse relay direction correctly rejected while unlisted. That is not the agentbox to relay to forum to VisionClaw chain this phase specified. The cross-substrate fixture sync gate is the one line that did land, as agentbox ADR-2061's schema-file contract test. Exit criteria now live under CP-03 (grounded execution) and CP-05 (human judgement and governance); the original contract is still readable at [Mesh Smoke Test](protocol/mesh-smoke-test.md).

## Phase 3, operational readiness: partial

The versioned release manifest advanced (roster 6 to 14 with provenance; draft at `estate-closeout/2026-09-05/release-manifest.local-draft.json`). Pod tier migration, the unified health dashboard and backup/DR runbooks show no evidence of movement and are unchanged. These now sit under CP-08.

## Where to look instead

All nine closeout packages remain open by the programme's own exit criteria. CP-07 (memory and improvement) is worth singling out: its blocking RuVector recall-gate question was re-measured on 2026-09-06 and passes (self-recall 189/200, true-recall 115/120, gate exit 0), which retires the contradiction but not the package. Read [the closeout README](estate-review/closeout/README.md) for the current gate table, and the VisionClaw repo's `docs/TODO-unified.md` for the row-level board.
