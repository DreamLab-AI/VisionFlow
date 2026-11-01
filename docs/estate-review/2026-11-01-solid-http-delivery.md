---
title: Solid HTTP policy admission, lockout and delivery
status: source-reviewed-open-findings
date: 2026-11-01
type: explanation
---

# Solid HTTP policy admission, lockout and delivery

Authorised HTTP ACL updates can admit a missing condition type or an empty condition, after which anonymous GET returns the synthetic private body. The server rejects the unterminated Turtle control. This extends the [previous library finding](2026-11-01-solid-policy-parsing.md) to the native HTTP handler stack, while preserving the distinction between invalid syntax and incomplete policy structure.

A second probe shows that the lockout guard accepts an owner Control rule aimed at another resource. The update returns 201, then the owner receives 403 when reading the original ACL. Successful GETs also advertise empty `WAC-Allow` permissions despite granting access. Native private cache headers work in the tested paths; the forum edge still emits public cache directives without classifying the audience.

Scope: every section of [ADR-2002](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2002-wac-access-model.md), relevant baseline access-control/cache clauses and invariant 7, and seven selected diagram panels. Solid is `e1d16adbfc096527f82ff16a6bce87ffd6d909f8`; forum is `c39f41f104b114d6a3fa29013daaf6fa58b45080`. [Sources](evidence/2026-11-01/solid-delivery/sources.json), [probe](evidence/2026-11-01/solid-delivery/http-probe.rs), [runner](evidence/2026-11-01/solid-delivery/run-probe.cjs) and [results](evidence/2026-11-01/solid-delivery/probe.json) make the assessment reproducible. No product code was changed.

## D1 — authorised HTTP writes admit incomplete restrictions

These probes run `build_app` with real NIP-98 signing and verification, synthetic `MemoryBackend` data and a known test owner. Every proposed ACL retains an owner Control rule for `/secret.bin`. The additional rule offers public Read constrained by a client condition. Each request traverses the app's middleware and handlers; no network listener is opened.

| Proposed condition | Owner PUT | Following anonymous GET | Synthetic body returned |
| --- | ---: | ---: | --- |
| Complete client condition | 201 | 401 | no |
| Unterminated blank node | 400 | 401 | no |
| Client field without condition type | 201 | 200 | yes |
| Empty condition | 201 | 200 | yes |
| Explicit unknown type | 201 | 401 | no |

`handle_put` enforces existing Control, runs `proposed_acl_keeps_caller_control`, then `validate_rdf_body` before storage (`server/src/lib.rs:1736-1773`). The RDF validator checks syntax (`1669-1698`); it does not validate the private condition schema. The lockout helper searches for a recognisable Control entry, not general policy validity. The incomplete cases therefore reach the lossy parser described in the previous report. The anonymous re-evaluation uses that same parsed policy (`1285-1304`), classifies the binary as Public and emits no explicit `Cache-Control`.

An explicit unknown-type control is stored but denies Read. That differs from the direct library write validator, which refused the same class of unknown condition in the previous batch: this HTTP path does not invoke that validator. The missing-type and empty cases remain unsafe even if that existing validator is added unchanged, because their conditions have already disappeared.

Acceptance: preserve the distinction between absent and invalid condition objects at parsing; perform policy-shape validation before committing an ACL. Keep the existing RDF syntax check. Reuse the owning crate's parser, validator and typed outcomes across consumers. Add HTTP regressions covering these exact positive and negative controls. This is an authorised-owner configuration failure, not proof that an unauthorised caller can write policy. An anonymous PUT control returned 401. No real private data or deployed pod was involved.

## D2 — the lockout check does not evaluate effective Control

`proposed_acl_keeps_caller_control` scans `acl:mode`, `acl:agent` and recognised agent classes (`server/src/lib.rs:913-941`). It has no protected-resource parameter and does not test the candidate authorisation's target or conditions. An unparseable ACL returns true to defer validation (`906-909`).

Starting from a valid owner ACL, the probe submits a syntactically valid replacement whose owner Control entry names `/elsewhere`. The server accepts the PUT with 201. A subsequent owner GET of `/secret.bin.acl` returns 403. The sidecar gate correctly evaluates Control on `/secret.bin`; the earlier guard had only established that some Control entry mentioned the caller.

This contradicts SP-04.6's previous “you may not ACL yourself out” promise. Acceptance: evaluate the candidate policy for Control on the actual protected resource, using the same relevant identity and condition context as enforcement. Keep policy validation and lockout prevention distinct. Verify that a valid owner retains access and a misdirected or unsatisfied Control rule is refused before storage. This probe covers PUT; other guard call sites were inspected, but POST/PATCH/COPY were not freshly executed.

## D3 — successful GETs advertise no permissions

`handle_get` calls `wac_allow_header(None, agent, path)` after a successful read gate (`server/src/lib.rs:1487`). With no ACL document, the helper's mode evaluations fail (`wac/mod.rs:265-286`), producing `user="", public=""`.

Both positive controls assert that header both on an owner GET returning the private binary and on a public GET after granting anonymous Read. Enforcement has granted access, but discovery tells clients that no access modes are available. This can hide controls or make an otherwise working resource appear unusable to a client that follows discovery. It is a capability-reporting defect, not an extra permission grant.

Acceptance: compute advertised modes from the resolved policy and applicable request context; do not substitute a second simplified permission model. Verify public Read, authenticated Read/Write/Control and condition-bound cases together with the actual operation result. The [WAC access-privileges section](https://solidproject.org/TR/wac#access-privileges) requires GET/HEAD responses to advertise privileges. This batch reproduces GET only; the route table sends HEAD to the same handler, which is source evidence rather than a new HEAD execution.

## D4 — native privacy headers work; edge adoption and cache revocation remain open

Twelve existing integration tests pass: ten cache-policy tests and two container-index tests. The index tests prove that a private `index.html` is omitted for a container-only reader and that an inherited readable index is still served. The cache suite spans private binary/PDF/RDF, public controls, a container listing and the pure media/audience policy.

Existing sidecar coverage is weaker than its test name: it accepts a client error because its fixture grants no Control. It does not require a successful private sidecar response. The new probe supplies that missing control: owner sidecar GET returns 200 with `private, no-store` and `Vary: Authorization`. An owner binary GET has the same privacy headers. The implementation preserves an existing handler cache header, including mashlib's `no-store` (`server/src/lib.rs:1404-1419`).

An authorised update adds anonymous Read, giving GET 200; a second update removes that rule, giving GET 401. A fresh app instance is constructed for each request over the same in-memory storage. This verifies that the next local handler evaluation uses the changed ACL. It is not an operating-system restart, persistent-storage recovery or an intermediary-cache experiment. Previously downloaded or cached public bytes are not recalled by changing the ACL.

Forum edge code still uses `add_cache_control` (`pod-worker/src/lib.rs:325-332`): `/media/` gets `public, max-age=31536000, immutable`; other paths get `public, max-age=300, must-revalidate`. Successful protected GET branches call it after authorisation, without audience input (`1015-1064`, `1251-1267`); ACL reads also call it (`1701`, `1715`). The source claim that every `/media/` path is content-addressed was not established here. Its alpha.18 pin already contains the shared audience API, so the ADR's alpha.7/publication prerequisite is obsolete. Edge adoption, actual cache behaviour and end-to-end revocation remain open.

There is also a standards correction to the historical rationale. Missing `Cache-Control` alone does not permit a compliant shared cache to reuse a response to an `Authorization` request: [RFC 9111 section 3.5](https://www.rfc-editor.org/rfc/rfc9111.html#section-3.5) imposes an additional restriction. `public` and certain other directives can lift it. `no-store` governs storage, while `must-revalidate` acts once a response is stale; it does not require every fresh response to contact the origin. The native hardening remains useful, but the old blanket inference about authenticated binary responses is too broad. The edge's explicit public directive is a different case.

## ADR-2002 section dispositions

| Section | Assessment |
| --- | --- |
| Front matter and trigger | Accepted WAC choice remains implemented. Historical complete/live axes and `e093e88` do not certify current interoperability or deployment. Consumer/profile changes still trigger review. |
| Context | Native enforcement uses WAC with private client/issuer/payment extensions. No ACP vocabulary matches in the scoped Rust search; this supports the implemented-path reading, not universal absence of every equivalent concept. |
| Decision | WAC remains the implemented policy model. The local HTTP gate, sidecar Control mapping and condition controls exercise that path. No ACP evaluator was added. |
| Consequences | ACP-only policy graphs remain outside this implementation. Private condition compatibility must be negotiated; WAC choice alone does not establish standards conformance or safe parsing. The report is not a “ratified” W3C standard. |
| Verification | Current source and native receipts replace historical line numbers. The recorded vocabulary search is reproducible. Successful GET discovery still contradicts actual permissions (D3). |
| Closeout extension — 4 September | Typed reported failures remain distinct from absence; D1 shows the surviving condition-loss case. Sidecar Control has positive and negative local evidence. Grant removal works at the next local evaluation. Delegation across consumed versions, cache invalidation and durable restart remain open. |
| Acceptance progress — 5 September: implementation | Audience and private-cache functions are wired natively. Twelve existing tests plus the successful sidecar control pass. The historical missing-header rationale needs RFC 9111 qualification. Existing-header preservation means handlers remain responsible for any explicit override. |
| Acceptance progress: receipts and WAC choice | The historical 1,801-test workspace result is not rerun or recertified. This batch reports its own twelve tests and HTTP probes. WAC selection is unchanged; ADR-2005 does not fully discharge malformed-policy safety. |
| Acceptance progress: remaining and governed paths | Forum pin is alpha.18; adapter adoption is still missing. No real cache, persistent restart, deployment or cross-service delegation result. Source ADR and baseline were not edited. |

Whole-baseline coverage, all WAC modes and the remaining Solid/edge server surface are still open. Payment, linked-identity and origin paths in the read-gate diagram were inspected but not freshly exercised. The [source parity receipt](evidence/2026-11-01/solid-delivery/revision-parity.json) supports the retained diagram revision; unchanged sibling bodies retain their prior claims without new credit.

## Comparison and user utility

[Community Solid Server's authorisation guide](https://communitysolidserver.github.io/CommunitySolidServer/7.x/usage/authorization-methods/) documents a configuration choice between WAC and ACP, with client-specific restrictions as an ACP motivation. That makes it a useful comparison for the estate's private client-condition dialect. It does not prove either implementation secure or establish extension interoperability. The estate's choice can remain WAC, but clients need an explicit capability and dialect contract.

A useful next delivery is a policy preview that checks effective permissions for owner, named collaborator and anonymous reader before save. Include whether owner Control survives, which condition types are recognised, and the advertised GET capabilities. Use the same shared crate result for enforcement and preview. For revocation, distinguish “next origin request denied” from “previously public copies may still exist”; measure grant/removal across native and edge with a real cache and persistent restart. That gives users a verifiable sharing workflow instead of a success response whose meaning changes between consumers.

## Verification boundary

The [native receipt](evidence/2026-11-01/solid-delivery/native.txt) records `cargo test --locked --offline -p solid-pod-rs-server --test cache_control_policy --test container_index_acl`. The standalone probe links the locked production libraries, compiles with warnings denied and asserts observed behaviour. No development bearer feature is enabled. All data and keys are synthetic; no live HTTP service, external account or cache was contacted by the probes.

Selected panels: SP-03.1/.2/.3/.12 and SP-04.2/.5/.6. Rendered siblings receive no source-review credit. RuView remains excluded and VisionClaw springs are untouched. Product fixes and deployment remain outstanding.
