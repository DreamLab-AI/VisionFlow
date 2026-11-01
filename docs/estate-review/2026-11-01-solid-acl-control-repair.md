---
title: Solid Pod candidate ACL control and discovery repair
date: 2026-11-01
status: assessed-with-open-findings
type: explanation
---

# Solid Pod candidate ACL control and discovery repair

Candidate ACL writes now use the shared bounded policy reader and evaluate
whether the caller retains Control on the destination. The server no longer
accepts a policy merely because some rule names the caller and contains a
Control mode. JSON PATCH now reaches the same guard, and COPY does not install
an accompanying ACL that would leave the destination outside the caller's
control. Ordinary metadata remains Control-protected without requiring an ACL
body.

ACL discovery also uses the resolver's existing `acl_sidecar_key` helper.
A container `/shared/` advertises `/shared.acl`; the root advertises `/.acl`,
including not-found responses. This aligns new client interactions with the
existing storage contract. It does not move stored inner ACLs or assign them
new precedence. The product migration guide now explains that boundary,
conflicting copies, target-IRI limitations and required access checks.

Producer repair `daff10d610ecb0fb4f1dd0e143c0ea6b95bd7445` is merged into Solid
main, followed by documentation-only `b644e55c76d6e89b0e41ec1c7aa44510b20ccf68`.
The reviewed base is `fe4f2ec338f0c4de75c9769b08d54bbb12e779b2`.

## Evidence and scope

Nine production-authenticated route regressions pass. The fixtures use NIP-98,
in-memory storage and a valid owner read before testing refusal. They cover:

- malformed JSON, empty arrays and plain-text ACL replacements;
- a Control rule aimed at a different resource;
- an unsatisfied client condition and a preserved verified linked identity;
- POST and direct COPY rejection before policy bytes change;
- JSON PATCH refusal when it removes Control, with a valid update accepted;
- resource COPY preserving the destination policy rather than installing an
  inapplicable source policy;
- malformed Turtle COPY despite a recognisable Control entry;
- ordinary metadata writes under existing Control admission;
- following the advertised container ACL link, writing a restrictive policy,
  then observing anonymous child access change from 200 to 401 while the owner
  remains authorised.

Six initial guard regressions failed on the old implementation. A separate
link-discovery regression failed with anonymous access still 200 after a 201
policy write to `/shared/.acl`. The refined tests check stored bytes and
subsequent authorised reads, not just an error status.

Initial fixture attempts included a compile error from an unavailable random
source, a POST request aimed at an unregistered root route, and an RDF PATCH
probe whose seed did not establish its required Control grant. Those are not
product-defect evidence. The corrected six-case baseline and separate
container-link baseline are retained independently. RDF PATCH call sites use
the repaired guard, but this checkpoint does not claim a successful RDF ACL
PATCH interoperability journey.

Running the development-bearer suite exposed an older COPY test that only checked
whether an ACL file was created, even though its rules named the source. That
fixture now grants destination Control too and verifies a successful read
under the copied policy. A separate negative regression ensures a source-only
policy is skipped. This preserves valid propagation without using file
existence as proof of usable authority.

Before the core discovery change, the default server suite passed 268 tests with one ignored. The final nine-case route suite passes with that change.
Formatting passes, with existing stable-toolchain warnings about nightly-only
options. The final development-feature suite passes 311 tests, including seven doctests,
with one ignored. Core library and header tests pass 512. Final Clippy passes
with warnings denied. Warning-denied rustdoc initially failed on a pre-existing
link to a feature-gated type; the documentation-only follow-up makes that
reference ordinary code text and rustdoc passes. Counts describe separate runs
with overlapping coverage, not a total of distinct tests.

## Shared contracts and remaining boundaries

Candidate validation composes `classify_policy_read`, `validate_acl_document`, the
existing RDF syntax parser and `grant_for_identities`. It uses the same linked
identity and payment-balance context as ordinary admission, without introducing
another ACL grammar or evaluator. Syntax checking also applies to COPY, which
has no generic request-body validation. Refusal occurs before quota reservation
and the candidate write. Existing request admission still happens first; this
repair does not make payment collection transactional.

This repair addresses the target/condition part of [HTTP D2](2026-11-01-solid-http-delivery.md)
and the malformed-write portion of [parity row 266](2026-11-01-solid-parity-intake.md).
It does not repair policy information already lost by the historical Turtle
reader, full JSON-LD expansion, absolute target matching, self-service owner
provisioning, all MCP policy writers, or durable cross-process races. Older
inner container ACLs still require deliberate migration. Publication, consumer
adoption and deployment remain open.

## Security review disposition

Deepsec review of the server file returned BLOCK at HIGH, with one HIGH, one
HIGH_BUG and two MEDIUM findings. All concern pre-existing serving paths;
none attributes a defect to the new guard. The raw blocking receipt is
preserved. This is not a clean security gate or a declaration of release safety.

The HIGH finding identifies pod-root Read authorising a complete Git repository,
including child/history data with different ACLs. Source inspection supports
the mismatch, but no fresh clone fixture or deployed-build exposure was tested.
Git support is default-off and is pulled in by forge. Additional findings
concern payment debit before delivery, credentials forwarded across proxy
redirects, and filesystem app-manifest reads without WAC. Payment review wording
is corrected where it suggests the sidecar Control branch itself incurs a debit;
that branch explicitly skips charging. These findings remain in the [repair queue](remediation/solid-acl-control-2026-11-01.json)
with source-versus-runtime limits and acceptance conditions.

A second review analysed `ldp.rs` and added one BUG finding: the existing
SPARQL DELETE/INSERT branch ignores WHERE for ground templates and skips
variable templates. Base-versus-repair source inspection confirms that branch
is unchanged. Its variable case overlaps `SP-PAR-SPARQL-NOOP`; conditional
ground mutation needs a fresh runtime fixture. The second receipt exports all
five findings across both run IDs and remains BLOCK at HIGH. It is not five
new findings from the core-file review. The [disposition receipt](evidence/2026-11-01/solid-acl-control/review-disposition.json)
retains these distinctions alongside both raw reviews.

## Diagram coverage

[SP-13.1 and SP-13.2](../diagrams/solid-pod-rs/13-acl-control-and-discovery.md)
record candidate evaluation and the discovered-link access journey. Both panels
were rendered and visually inspected; sixteen citations resolve at the producer
commit with no warnings, fallback or unresolved references. JEV citation
sampling is advisory. These two panels receive assessment credit; no additional
whole-ADR assessment or deployment credit is claimed.

## Contract comparison and user utility

WAC requires the server to advertise its associated ACL resource and clients
to discover that link rather than manufacture a path. It also defines Control
as authority over the ACL associated with a resource, and discusses protection
against loss of control. The server's existing sibling layout can therefore
remain the association; truthful discovery and successful permission changes
are the user-facing acceptance criteria. [WAC discovery and access modes](https://solidproject.org/TR/2024/wac-20240512)

A practical next migration check is a staged pod with both inner and sibling
ACL copies: preserve the originals, establish the intended policy, install it
at the canonical location, and verify owner, unrelated-user and anonymous
access on the container and a nested child. No automated precedence choice or
file deletion is justified by this repair.
