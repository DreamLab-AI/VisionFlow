---
title: Solid provenance delivery and missing anchor outcomes
status: source-reviewed-open-findings
date: 2026-11-01
type: explanation
---

# Solid provenance delivery and missing anchor outcomes

With Git enabled, the native server records a real commit and returns useful receipts for ordinary writes, Git failures and sidecar failures. Two anchoring paths still hide unmet policy: an absent trail silently reduces an inline request to a local mark, and epoch accumulation failures remain log-only. DELETE removes the resource without a provenance receipt or a new commit, leaving its previous sidecar behind.

Scope: every section of [ADR-2004](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2004-provenance-off-by-default.md), the baseline provenance clauses and invariant 5, and SP-07.1 through SP-07.5. Solid source is `e1d16adbfc096527f82ff16a6bce87ffd6d909f8`; the Agentbox consumer profile is read at `9dfc27b16e82e9786535a55e790710493dd51fdf`. The [source manifest](evidence/2026-11-01/solid-provenance/sources.json), [HTTP probe](evidence/2026-11-01/solid-provenance/http-probe.rs) and [runner](evidence/2026-11-01/solid-provenance/run-probe.cjs) identify the evidence. No product code changed.

## P1 — requested inline anchoring can disappear from the receipt

`resolve_anchor_policy` maps an inline condition to `HighValue` (`server/src/handlers/prov.rs:90-119`). The write hook builds an anchorer only when policy requests anchoring, but sets `high_value` to true only if the anchorer bundle exists (`server/src/lib.rs:3967-3990`). A definitely absent trail produces no bundle. The library consequently sees an unflagged `HighValue` request and does not attempt anchoring or report its missing configuration.

This defeats a distinction already present in the shared API: `record_receipt` reports an anchor error when policy requests inline anchoring and no anchorer is configured (`provenance.rs:542-553`). The library regression passes with stub components, while the server clears the flag needed to reach it.

An HTTP probe backed by real Git confirms PUT 201, stored bytes, a commit containing those bytes, a sidecar and the plain summary `local-mark-committed` when inline anchoring is requested with no trail. An unreadable-trail control instead returns `local-mark-committed; anchor-error=...`: `UnavailableAnchorer` preserves that error. Absence and unreadability therefore have different client-visible outcomes even though neither fulfils the requested anchoring policy.

Acceptance: preserve the requested policy independently of whether its provider can be constructed; carry the missing-configuration outcome through the existing shared receipt API. Keep the successful resource write and local mark. Add a server regression for absent and unreadable configuration, alongside the ordinary unanchored control. No new receipt implementation is needed in a consumer.

## P2 — epoch outcomes are not attached to the write receipt

Before composition, the hook passes epoch policy to `record_receipt` as `Never`, then separately accumulates the commit only when an anchorer bundle exists (`server/src/lib.rs:3987`, `4047-4073`). With no trail, no accumulation occurs: the probe returns the same plain local-mark summary and creates no epoch state file.

A second fixture supplies an unreadable trail so that an `UnavailableAnchorer` bundle exists, then supplies malformed epoch JSON. Batch loading fails before any network call. PUT still returns 201, the local commit and sidecar exist, and the summary contains no anchor error. The malformed batch remains unchanged. The implementation logs the error but does not mutate the receipt. Source also shows that a successful closed epoch is logged without attaching its result to that receipt; successful on-chain epoch closure was not executed here.

Acceptance: distinguish queued, failed-to-queue and anchored batch outcomes, and retain the relationship between each commit, batch root and independently checkable proof. Failure to accumulate must be visible to the caller or a durable reconciliation surface. Keep that contract in the owning provenance crate and consume it in adapters. Interrupted broadcast, restart, batch recovery and reconciliation remain separate unverified obligations.

Policy resolution scans every authorisation's conditions for the first provenance anchor and defaults to `Never` on an ACL read error. That source behaviour is recorded in SP-07.3; policy selection across conflicting authorisations was not adversarially exercised in this batch.

## P3 — feature and failure controls establish the narrower working contract

The [default result](evidence/2026-11-01/solid-provenance/default-probe.json) and [Git-enabled results](evidence/2026-11-01/solid-provenance/git-probe.json) come from separate locked builds of the actual server library. The probe uses `build_app`, disposable filesystem storage and Git 2.56.0. Synthetic ACLs explicitly permit the tested operations. No test replaces the production marker with a stub.

| Case | HTTP status | Receipt / effect |
| --- | ---: | --- |
| Cargo default, Git repository present | 201 | `skipped; skipped=not-configured`; no mark or sidecar |
| Git enabled, ordinary resource | 201 | `local-mark-committed`; commit contains stored bytes |
| Git enabled, no Git repository | 201 | `skipped; skipped=not-git-backed` |
| Successful ACL write | 201 | `skipped; skipped=excluded-path`; no mark or provenance sidecar |
| Synthetic Git index lock | 201 | `resource-stored; mark-error=...`; bytes stored, no mark |
| Sidecar destination is a directory | 201 | `local-mark-committed; mark-error=...`; commit retained |
| Inline request, absent trail | 201 | Plain local-mark receipt; missing anchor not reported |
| Inline request, unreadable trail | 201 | Local mark plus `anchor-error` |
| Epoch request, absent trail | 201 | Plain local-mark receipt; no batch created |
| Epoch request, corrupt batch | 201 | Plain local-mark receipt; batch error not reported |

Four existing server Git tests and fourteen library receipt tests pass. The existing ACL exclusion test ignores its response status; the new successful ACL PUT supplies the missing positive control. The fourteen library tests use stub markers and anchorers. Their “default build” and “git build” group labels in the ADR are not evidence of server feature wiring or real chain confirmation. The new probe separately checks the actual default and Git-enabled HTTP builds.

`stage()` classifies the fields already in a receipt: an anchor with a block height is `AnchorConfirmed`, and one without is `AnchorSubmitted` (`provenance.rs:715-730`). It does not poll a chain or perform an automatic submitted-to-confirmed transition. A sidecar failure adds `mark_error` without discarding the achieved mark or anchor; its stage is not necessarily limited to `LocalMarkCommitted`. The corrected diagrams preserve both distinctions.

The Cargo default is not the Agentbox build profile. Agentbox's `lib/solid-pod-rs.nix:83-90` always includes `git`; `flake.nix:1548` additionally selects `forge-full` by default. That package pins alpha.18 at `76ab7aab3e882f32508b2461b3f51e4c09bdcfa1` and patches the path split to strip an optional `pods/` prefix (`lib/solid-pod-rs.nix:127-137`). These upstream local probes do not include that patch and do not certify the Nix binary or deployed pod. Feature availability, runtime repository configuration and deployment identity need separate evidence.

## P4 — DELETE does not record the mutation

After a successfully marked PUT, the probe deletes the resource. DELETE returns 204 with no `X-Provenance`, removes the file, leaves Git HEAD unchanged and retains the old `.prov.ttl` file. The handler authorises Write and deletes storage, then returns without calling `git_mark_write` (`server/src/lib.rs:2276-2301`).

The ADR context's “every LDP write” wording is broader than this implementation. Its later, explicit PUT/POST/PATCH receipt list is the safer description. A previous creation sidecar does not establish that a later deletion was recorded. Acceptance: define the mutation coverage contract, including deletion/tombstones and sidecar lifecycle, and test it at the HTTP boundary. Do not advertise a complete resource history from this path. COPY and container mutations were not freshly exercised.

## ADR-2004 section dispositions

| Section | Assessment |
| --- | --- |
| Front matter, lineage and trigger | The accepted opt-in feature decision is implemented. Historical complete/live axes and `e093e88` do not establish current end-to-end durability or deployment. Consumer profile changes remain a review trigger. |
| Context | Canonical server composition is now `record_receipt`, not legacy `record`. “Every LDP write” overstates coverage (P4). Build-size and latency motivation is unbenchmarked here. |
| Decision | Cargo `default = []` and the no-op shim remain. The fresh default probe returns an explicit skipped receipt. Agentbox deliberately enables Git, so the Cargo default must not be projected onto its packaged server. |
| Consequences | Feature qualification remains necessary but insufficient: filesystem repository configuration and supported mutation matter too. No default flip is proposed. Size/latency benefits were not measured. |
| Verification | Current source and separate build receipts replace historical line anchors. Selected diagram sources are byte-identical at the retained diagram revision; that does not renew unselected panels. |
| Closeout extension — 4 September | Default/Git, successful excluded write, failed mark and failed sidecar now have actual HTTP evidence. Interrupted anchoring, restore and reconciliation remain open. Storage-before-mark ordering persists. |
| Acceptance progress: stages and receipt | Types and local failure preservation work; block height classification is not a live confirmation observer. Missing-provider handling in the library is bypassed by the server (P1); epoch outcomes are omitted (P2). Error payloads are strings in typed receipt fields, not a separate structured error enum. |
| Acceptance progress: caller surface | PUT is freshly exercised. POST and PATCH call sites remain source evidence. DELETE is outside the hook. Summary sanitisation bounds printable error text; the header remains a local convention. |
| Acceptance progress: tests and receipts | Fourteen stub composition tests and four existing server tests pass, plus one default and ten Git probe scenarios. The historical 1,801-test workspace and all-feature Clippy claims were not rerun. |
| Acceptance progress: remaining and governed paths | Crash between storage and marking can prevent any response receipt from reaching the caller. A receipt does not close or reliably expose that crash window. Live pod, actual chain, interrupted anchoring and recovery remain unverified. No source ADR, baseline or product code changed. |

This is not a whole-baseline, whole-provenance or whole-payment audit. SP-07.6 onwards remains outside this new panel claim. The selected policy panel retains the source-search finding that no non-test server call initiates `GitmarkAnchorer::genesis`; the [search receipt](evidence/2026-11-01/solid-provenance/genesis-search.txt) is bounded to the native server source, not all possible external provisioning tools.

## Comparison and user utility

[OpenTimestamps](https://opentimestamps.org/) describes proofs that data existed before a point in time and can later be independently verified. That is a useful comparison for proof export and verification, separate from whether the underlying industrial observation is true. [Sigstore Rekor](https://docs.sigstore.dev/logging/overview/) records signed supply-chain metadata and supports inclusion proofs and log verification. Its transparency-log contract is a different service from local Git history or a Bitcoin anchor. Neither comparison establishes interchangeability or a security ranking.

For an industrial inspection record, show the achieved evidence explicitly: bytes saved, local revision recorded, anchoring requested/queued/failed, proof available and proof checked. Measure whether each requested policy reaches a durable outcome after restart, and whether another machine can verify an exported proof without trusting the authoring UI. Keep receipt semantics and verification in the existing shared crate; let native, edge and VisionClaw clients consume that result. The immediate utility is a truthful inspection handover whose missing evidence can be found and repaired.

## Verification boundary

Retained [library receipt logs](evidence/2026-11-01/solid-provenance/receipts-tests.txt) and [server Git log](evidence/2026-11-01/solid-provenance/git-tests.txt) record eighteen passing native tests. The standalone probe compiles with warnings denied and links the locked production libraries. Its eleven scenarios use synthetic, explicitly authorised local fixtures. No external chain request, live pod, persistent restart, Nix build or deployment was performed. Product fixes remain outstanding. RuView remains excluded and VisionClaw springs are untouched.
