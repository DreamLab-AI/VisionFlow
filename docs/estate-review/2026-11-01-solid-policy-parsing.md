---
title: Solid policy parsing, typed resolution and consumer adoption
status: source-reviewed-open-findings
date: 2026-11-01
type: explanation
---

# Solid policy parsing, typed resolution and consumer adoption

An incomplete Turtle access condition can disappear during parsing while its surrounding Read grant survives. The production resolver returns `Found`; the document validator accepts the resulting rule; the evaluator grants anonymous Read. This differs from the earlier malformed-policy inheritance defect: the new grant comes from a partially parsed specific policy, without falling back to an ancestor.

Typed missing/invalid/unavailable resolution passes its existing native acceptance tests. It cannot detect a restriction that the parser has already discarded. Separately, the forum now consumes Solid alpha.18 but still implements a local resolver that treats failed or invalid reads as misses. A dependency upgrade has not completed that adapter migration.

This assessment covers every section of [ADR-2005](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2005-fail-closed-untrusted-parsing.md), with the relevant baseline invariants, SP-04.3/.4/.9/.10 and SP-06.7. Solid source is `e1d16adbfc096527f82ff16a6bce87ffd6d909f8`; forum source is `c39f41f104b114d6a3fa29013daaf6fa58b45080`. The [source manifest](evidence/2026-11-01/solid-policy/sources.json) and [revision comparison](evidence/2026-11-01/solid-policy/revision-parity.json) bind the observations. Selected implementation files are byte-identical to the diagrams' declared Solid revision `09ec90f`; no sibling panel receives review credit from that equality.

## P1 — missing condition objects can leave an unconditional rule

`parse_turtle_condition_objects` breaks on an unclosed blank node (`wac/parser.rs:317-320`). A blank node with no type produces `None` in `parse_turtle_condition_body` (`418`); its caller appends only `Some` results (`323-325`). Neither case inserts an unknown/invalid condition. The enclosing authorisation can retain its agent class, resource and Read mode with an empty condition list. `validate_for_write` rejects explicit `Unknown` entries, but an empty list passes (`wac/conditions.rs:382-400`). The evaluator checks only the retained conditions (`wac/evaluator.rs:309-323`).

P1 stores each fixture through the real `MemoryBackend`, resolves it with `resolve_policy_from_storage`, validates the resulting document and evaluates Read. No custom parser or substitute evaluator is used.

| Specific policy fixture | Strict Turtle syntax | Retained conditions | Validator | Anonymous Read |
| --- | --- | ---: | --- | --- |
| Complete client condition | valid | 1 | accepts | denied |
| Unterminated client-condition blank node | invalid | 0 | accepts | granted |
| Blank node with client field but no type | valid | 0 | accepts | granted |
| Empty condition blank node | valid | 0 | accepts | granted |
| Explicit unknown condition type | valid | 1 | refuses | denied |

Independent syntax checking uses the existing `oxttl` dependency. It distinguishes invalid Turtle from valid RDF that lacks the private dialect's required condition structure. The public `evaluate_access` entry point deliberately denies retained conditions through its empty dispatcher; losing the condition changes that result. [Probe results](evidence/2026-11-01/solid-policy/probe.json) preserve all five controls.

This is a library/storage reproduction. HTTP write admission, owner lockout checks, authenticated delivery, a running service and actual private-data exposure were not exercised. The finding does not grant an unauthorised caller permission to replace an ACL. Its trigger is a policy with a truncated or incomplete restriction reaching this resolver.

Acceptance: make syntax errors terminal, and make an explicit but unrecognised or incomplete condition deny rather than vanish. Keep deliberately absent conditions distinct from present empty objects. Reuse the crate's existing `oxttl`-backed `rdf_input::parse_turtle` machinery (`rdf_input.rs:201-218`) where appropriate; apply condition shape validation after RDF parsing. A standards parser alone cannot reject the syntactically valid missing-type cases. Preserve the shared pure API for native and edge consumers. Add authorised HTTP upload/read and restart receipts before claiming end-to-end closure.

## P2 — bounds and typed resolution work within their stated contract

Forty-one existing tests passed: thirteen condition tests, five parser-bound tests, twenty policy-outcome tests and three write-validator tests. The outcome corpus exercises legitimate missing-policy inheritance, invalid specific/ancestor policy, non-UTF-8, size/depth limits, backend faults, policy deletion and legacy adaptation. This revalidates the native repair documented in the ADR's September acceptance progress; it does not make every malformed Turtle document invalid.

P2 additionally checks the public classifier's `Absent → Ascend`, failed-read `Unavailable`, and unrecognised-body `Invalid` results. Both parsers accept a document at its explicit byte limit and reject it one byte over; JSON depth is checked before deserialisation. These are boundary checks, not a general CPU/memory bound or a proof about every Turtle grammar feature. The byte/depth controls remain load-bearing baseline requirements.

`cargo check --lib --no-default-features --features core` passed natively. This confirms that the shared surface compiles without the default runtime features in this target; no fresh WASM compilation or Workers deployment is claimed. The [native receipt](evidence/2026-11-01/solid-policy/native.txt) and [core check](evidence/2026-11-01/solid-policy/core-check.txt) record the actual commands' output. Test dependencies can unify extra features, so the selected test run alone is not the evidence for the isolated core surface.

## P3 — pod names are normalised and unmatched hosts fall back

`SubdomainResolver` lowercases a matching label and repeatedly removes `..` (`multitenant.rs:124`, `199-216`). It returns a pod only if the resulting label is nonempty and contains no remaining dot or slash (`130-138`). Unknown hosts, file-like labels and other unresolved cases return `pod: None` with the original path (`145-152`). This is a normalisation/fallback contract, despite ADR-2005's statement that the choice is not to sanitise and continue.

P3 confirms `al..ice.pods.example` selects the same `alice` pod as the ordinary spelling. An outside host and a multi-label host both return the unchanged `/secret` path with no pod selected. These are resolver outputs, not a demonstrated traversal or authorisation bypass. Acceptance: state the normalisation policy explicitly; if strict host rejection is desired, enforce it at a documented ingress boundary and test it through the server. Do not treat absence of a selected pod as an HTTP refusal.

## Consumer adoption remains open

Forum manifest and lockfile now select registry `solid-pod-rs` alpha.18. Nevertheless `nostr-bbs-pod-worker/src/acl.rs:285-320` still returns `Option<AclDocument>`, skips failed R2 reads and body reads, then falls back to KV. `resolve_r2_sidecar` maps invalid or oversized content to `None` (`330-334`). Its source does not call `classify_policy_read` or the typed resolver. No R2 fault or deployed edge request was run in this batch.

ADR-2005 and the baseline still explain this gap using an alpha.7 pin and a need to publish a consumable release. That dependency premise is stale. The remaining work is to consume the shared classifier and carry its failure outcomes through edge handlers, while retaining the edge's stricter document cap and documented sidecar order. Updating the version alone cannot supply those semantics. Real cache, restart and revocation acceptance remains open; this batch does not reassess ADR-2002's whole delivery contract.

## ADR-2005 section dispositions

| Section | Evidence and disposition |
| --- | --- |
| Front matter and review trigger | Accepted decision remains relevant. Historical implementation/live axes and `e093e88` do not certify the current cross-tier system; parser/resolver findings trigger review. |
| Context | ACL bodies and pod labels reach the inspected boundaries. New fixtures demonstrate lost restrictions; no live service exploit is claimed. |
| Decision | Byte bounds exist; strict rejection wording is contradicted by P1's partial parse and P3's label normalisation. |
| Consequences | Retain parser bounds and explicit failure handling. The claimed refusal of lenient parsing needs correction; the loop is implemented, but this audit does not prove the stated single-pass bypass rationale. |
| Verification | Current source and explicit boundary checks replace reliance on historical line numbers. Shared core compilation passed. |
| Closeout extension — 4 September | Native malformed-policy inheritance is repaired for reported failure outcomes. Edge adoption and deployed denial remain open. Policy deletion legitimately exposes ancestor defaults; it is not synonymous with revoking all access. |
| Acceptance progress — 5 September | Typed outcomes, shared native resolver and twenty existing tests revalidated. The “broken Turtle only loses diagnostics” conclusion is too broad: P1 retains a granting rule while dropping its condition. Legacy method adaptation cannot repair an external implementor that already converts errors to `Ok(None)`. |
| Acceptance progress residuals and remaining work | Historical alpha.7 consumer/publish premise is obsolete; alpha.18 adapter still needs migration. No cache/restart or deployment recertification. “Uncommitted” is historical wording: this audit pins current committed source. |

Baseline review covers its relevant access-control section and invariants 3, 4 and 4b. This is not a whole-baseline or whole-Solid review. No ADR was edited in the source repository, and no product remediation is claimed.

## Comparison and useful acceptance work

The [WAC report](https://solidproject.org/TR/wac#effective-acl-resource) bases inheritance on the absence of a representation and treats the present ACL as effective. It also requires Turtle representations for ACL reads. It is a Draft Community Group Report, not a W3C Standard. [Turtle's grammar](https://www.w3.org/TR/turtle/#grammar-production-blankNodePropertyList) defines the bracketed blank-node syntax; it does not define the estate's additional condition schema. This separation explains why both syntax and policy-shape validation matter.

[Community Solid Server](https://github.com/CommunitySolidServer/CommunitySolidServer) provides pods and identity with configurable modules and persistent filesystem operation. It is a useful interoperability comparison for standard ACL discovery, inheritance and resource sharing. No benchmark or security comparison was run, and its general Solid support does not establish support for this private condition dialect.

For users, the next useful deliverable is an effective-policy preview: show the selected ACL, inherited/direct status, recognised restrictions and any parse failure before saving. Compare standard sharing cases against another Solid server; test private extensions with explicit refusal when unsupported. Measure whether a user can grant, inspect and revoke access through native and edge tiers with the same result. Build that on the shared typed contract rather than a second parser or another per-consumer policy model.

## Reproduction and limits

[The probe](evidence/2026-11-01/solid-policy/policy-probe.rs) and [runner](evidence/2026-11-01/solid-policy/run-probe.cjs) build locked/offline dependencies, compile with warnings denied and run against local library APIs. Native tests used:

```sh
cargo test --locked --offline -p solid-pod-rs --no-default-features \
  --features memory-backend,tokio-runtime --test wac_policy_outcomes \
  --test wac_parser_bounds --test wac_validate_for_write \
  --test wac2_conditions_sprint9
```

Five selected panels are reconciled. Other panels rendered alongside them receive no source-review credit. Whole-corpus checks cover structure and unchanged declared revisions only; selected-topic citations and visual review have their own receipts. RuView remains excluded. VisionClaw springs, live pods and deployment state are untouched.
