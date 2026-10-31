---
title: Governance ingress, expiry and restart recovery
status: in-progress
date: 2026-10-31
type: explanation
---

# Governance ingress, expiry and restart recovery

Signed decisions can reach the corpus consumer with inconsistent proposal identity or an elapsed proposal deadline. A separate checkpoint defect can suppress an unfinished decision after restart. This pass strengthens the [previous adapter-level findings](2026-10-31-proposal-application.md) with real signature verification, signer-roster enforcement and request-by-ID verification. It also traces forum storage, broadcast, projection, UI controls and expiry at source level.

Sources: Agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf` and forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. Relevant files were clean. [Source hashes](evidence/2026-10-31/governance-ingress-sources.json) distinguish these observations from the diagrams' older pins. No product code, live corpus, relay, signer configuration or deployed panel changed.

## Signed consumer probes

The [probe](evidence/2026-10-31/governance-ingress-probes.cjs) signs synthetic requests and human decisions with fixture keys through `nostr-tools`. It invokes the real `RelayConsumer`, real orchestrator and installed Ontomark. The relay bridge only delivers fixtures, Git is deliberately refused, and attestation is recorded in process. Temporary corpus and pod directories are removed afterwards. [Results](evidence/2026-10-31/governance-ingress-probes.json).

| Fixture | Verified boundary | Result |
|---|---|---|
| Amendment body has a valid semantic digest, while request and decision tags share another digest | Decision signature and human roster; request signature, exact event ID and request/decision tag agreement | Amendment applies. Its returned digest names the proposal; attestation names the other case digest. |
| Correctly bound amendment whose `stale_after` is in 2000 | Same signature, roster and request checks | Amendment applies. This consumer does not check expiry or forum projection status. |
| Creation proposes page A and the same resource IRI, with staged title B | Same checks; valid semantic digest for the staged text | CLI writes B.md; adapter receipt still names page A. |

These results establish that Agentbox's actual cryptographic ingress checks do not prevent the discrepancies. They do **not** demonstrate forgery, an unrostered signer bypass, acceptance by a deployed relay, or what a real reviewer would see. Both request and decision are genuinely signed fixture events. The requester controls its body and tags; the reviewer signs the request ID and case tag. That binds the request's bytes, but does not make contradictory fields within those bytes agree.

`mcp/nostr-bridge/relay-consumer.js:446-538` verifies the decision and roster, then fetches the referenced 31402 and verifies its signature and exact ID. It does not recompute a PatchProposal digest. `local-process-manager.js:419-480` checks the fetched request tag and IRI, then passes the decision digest to mutation/attestation. The specialised `approve` branch at `:379-416` adds a proposal-digest equality check that explicit promote lacks. The installed CLI independently checks amendment integrity, but receives only reconstructed Markdown for creation.

## Relay and UI: what admission proves

`nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1063-1085` checks event structure, NIP-40 expiration and event ID/signature. NIP-40's event `expiration` tag is distinct from the proposal body's `stale_after`. Subsequent source checks include session authentication, author admission, registered-agent publication for requests, human admin/delegated-reviewer authority for responses, consequential-case rationale and supersession authority. This pass read those paths; it did not drive a Cloudflare Durable Object or its authentication stores.

Forum boundary planning derives the extra tier floor from the **request's level tag**, not a comparison with the proposal's body or diff (`plan_request_boundary`, `nip_handlers.rs:352-417`). No semantic proposal-digest recomputation or body/tag level agreement check was found in the inspected relay ingress, storage and projection paths. A successful publication by an authorised agent is therefore not evidence that those semantic fields agree. This remains a source-grounded finding until exercised against the deployed admission stack.

The forum client accepts a flat PatchProposal as a field map (`stores/panel_registry.rs:462`), derives the subject from its IRI (`utils/governance_view.rs:475`), and signs the card's case tag and request event ID (`pages/governance.rs:919-925`). The inspected card does not recompute the semantic digest. Its settled/decidable logic uses the decision chain and panel acknowledgement (`pages/governance.rs:175-216`; `utils/governance_view.rs:400-438`). Expiry closes a case **without** a decision, so these inputs alone do not retire an expired card or disable its controls. No browser or accessibility session was run; the card's rendered behaviour remains to be checked.

## Expiry does not yet establish a mutation veto

The expiry sweep reads cases in `open`, `under_review` or `reopened` with `stale_after < now`, records an expired receipt, then conditionally closes them (`cron.rs:748-838`). It neither edits the corpus nor removes the original signed request. The projection's SQL requires the case to remain in an allowed state and to name the cited request (`relay_do/receipts.rs:185-191`), so a closed case is excluded from that normal decision commit.

However, `handle_event` stores, acknowledges and broadcasts first (`nip_handlers.rs:1552-1554`), and calls response projection later (`:1634`). Projection failure is recorded and logged; it cannot retract an event already broadcast. Agentbox consumes that event directly and does not wait for `projection-committed`, read the case's closed state or enforce `stale_after`. The signed expired-proposal fixture proves the last part; the source establishes the ordering. A live test linking all three stages is still required before claiming a demonstrated deployed expiry bypass.

Acceptance needs a defined authority boundary: either mutation requires a successful, identity-bound projection/decision receipt, or the mutation owner independently enforces the relevant case, expiry and replay rules. A relay storage acknowledgement cannot stand in for either. The UI should also distinguish expiry from a human rejection so a closed proposal does not continue inviting an ineffective or inconsistent decision.

## Expiry parser refuses less safely than its contract states

`ontology_governance.rs:282` promises `None` for unsupported timestamps. A native probe linked against the just-built source library finds two exceptions: `2026-10-31T00:00:0éZ` panics because `s[..19]` splits a UTF-8 character, and `2026-10-31T00:00:00+-1:00` is accepted because the offset components use signed integer parsing without lower-bound/digit checks. A normal UTC timestamp is the passing control. [Probe source](evidence/2026-10-31/expiry-parser-probe.rs), [result](evidence/2026-10-31/expiry-parser-probe.txt).

The probe catches the panic without crashing a service. Projection calls this parser on proposal content, but the precise WASM/worker failure and recovery behaviour remains untested. Use character-safe validation and reject malformed offset grammar before date arithmetic. Reproduction: after the native crate build, compile the probe with `rustc --edition=2021`, `--extern nostr_bbs_core=<forum>/target/debug/deps/libnostr_bbs_core.rlib` and `-L dependency=<forum>/target/debug/deps`, then run it from an executable filesystem.

## Restart checkpoint includes unfinished work

`_onForumGovernance` puts each event in the shared in-memory `seen` map before adding its handler to the serial promise chain (`relay-consumer.js:459-501`). When an earlier handler settles, `_saveForumState` serialises **the entire map** (`:388-395`). It can therefore checkpoint an event whose handler has not finished.

The probe admits two signed decisions, blocks both handlers independently, then allows only the first to finish. At the checkpoint, only event 1 is complete, but event 2 is already stored as seen. A fresh consumer loads exactly that saved state and receives event 2 again: it records one rejected duplicate and makes **zero** dispatches. This simulates restart from the precise checkpoint; no process was killed. The original test instance is then released and cleaned up.

This contradicts the comment that a crash mid-apply leaves a decision redeliverable. A governance record remains on disk, but its payload omits the original event signature (`relay-consumer.js:883-896`). It is therefore not a complete signed envelope that a recovery process can independently reverify. Recovery could fetch the original event or use another authoritative store; this pass found no evidence that a reconciler repairs this queue-specific failure. Checkpoint only completed work, or persist an explicit pending/completed journal with a tested restart policy. Also distinguish handler resolution from successful mutation: this consumer intentionally settles returned or thrown application failures for later reconciliation.

## ADR-2013 section assessment

| Section | Assessment |
|---|---|
| Context | Accurately identifies the need for subject-bearing decisions, schema oversight and proposal expiry. Expiry's intended mutation protection is not established by closing the forum row alone. |
| Decision 1 | Promote/Demote carry an IRI and the client signs it alongside request/case references. The typed projection refuses malformed outcomes; that is later than storage/broadcast. The signed consumer probes preserve these subject checks while exposing independent digest/page disagreement. |
| Decision 2 | The named panel and Promote/Demote/Reject definition exist. Current Agentbox also supports a guarded approve-to-amend path, so “bare approve has no usable subject” describes the original profile rather than every current consumer. Publishing the panel and loaded deployment state were not checked. |
| Decision 3 | Source and focused tests support schema/demotion **tag** floors and tightening-only merge. Unknown tags fall back to content. They do not prove that a schema-shaped body or diff cannot claim a looser tag. |
| Decision 4 | Receipt-before-close and conditional pending-state update are implemented. The diagram's old `state='pending'` wording was wrong even at its declared revision: SQL lists open/under_review/reopened. Mutation protection remains incomplete across broadcast and consumer boundaries. |
| Decision 5 | Expiry extraction is gated by the presence of a level tag, not strictly by the named panel. Ordinary parse failures are ignored as documented, but the UTF-8 fixture panics instead. Other level-tagged requests can therefore enter this expiry policy; “ontology only” is a convention, not an enforced panel identity check. |
| Consequences | The test harness exercises useful components without publishing. It does not establish a live relay-to-application journey. The parser panics on a UTF-8 boundary fixture and accepts a malformed signed offset, contradicting the stated refusal contract; its passing tests do not cover these cases. Demotion's stable-subject policy is explicitly left to the proposer. |
| Verification | Historical counts and real-mode receipts remain dated evidence. Current focused suites pass, with limits below. An application success does not prove a projection succeeded. |
| 30 October close-out | Retains an informative historical harness failure. Current local fixture paths use the flat layout and both binaries, so those earlier failures do not explain the new identity/recovery findings. |
| 31 October close-out | Complete/staged refers to the recorded harness correction. The source-level expiry gap and reproduced consumer recovery defect remain additional obligations. No owner status was changed or production panel published. |

## Validation and coverage

Current focused suites: 14 core ontology tests, 11 relay ontology-boundary tests, 34 relay receipt tests, and 14 Agentbox forum-consumer tests. [Core log](evidence/2026-10-31/forum-core-ontology-suite.txt), [boundary log](evidence/2026-10-31/forum-ontology-boundary-suite.txt), [receipt log](evidence/2026-10-31/forum-receipt-suite.txt), [consumer log](evidence/2026-10-31/forum-consumer-suite.txt). The consumer suite uses recording bridges and injected verifiers; the new signed probes use actual signature verification. Native worker tests do not execute Cloudflare services. Jest emitted pre-existing duplicate-package-name warnings while its selected suite passed.

[ES-13](../diagrams/estate/13-governance-delivery-and-expiry.md) maps the current crossing and checkpoint. NF-06.16 corrects the historical SQL description without advancing unrelated panels' source pins. Remaining estate work includes proposal construction and risk classification, deployed admission and UI validation, recovery policy, source-ingest activation and the wider ADR/diagram queue.

Diagram validation: ES-13 and NF-06 passed strict citation checks with 33 and 216 revision-resolved references respectively; all 19 diagrams rendered. The [full corpus run](evidence/2026-10-31/governance-ingress-citations.txt) resolved 23,751 citations across 160 topics and 1,531 diagrams, with zero fallbacks/unresolved references and nine explicitly allowlisted generated-file citations. The generated files were copied from the primary checkout for validation and remain outside revision-backed evidence. These gates validate citation resolution and rendering, not every diagram’s semantic correctness.
