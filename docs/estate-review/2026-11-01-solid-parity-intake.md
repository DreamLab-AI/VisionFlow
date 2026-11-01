---
title: Solid Pod JSS parity intake and access-control recheck
date: 2026-11-01
status: in-progress
type: explanation
---

# Solid Pod JSS parity intake and access-control recheck

Solid Pod merged a concurrent JSS/plugin parity audit as `fe4f2ec`, after the
OIDC repair `fcc624a`. Its production code and dependency manifests are unchanged;
all files in the OIDC source receipt still match. New ADR-2011 and the parity
findings extend the review queue. They do not invalidate the scoped OIDC evidence
or count as newly completed ADR/diagram assessments.

The source audit is `crates/solid-pod-rs/docs/reference/jss-parity-audit-2026-11-01.md`
in Solid Pod. It compares JSS `7c9b0de` (0.0.224) and plugins `63a16cb` with
Solid Pod `e1d16ad`. Those comparator claims are imported here, not a fresh
upstream or deployment test. Its committed checklist rows 265–299 and three
additional PATCH defects are retained in the [intake queue](remediation/solid-parity-intake-2026-11-01.json).
Thirty-five parity rows are not thirty-five defects: the set includes present
behaviour, deliberate differences and deferred capabilities. Do not add these
records mechanically to the parallel audit's 134 distinct issues.

## Fresh runtime observations

The imported verifier probe ran unchanged in an isolated checkout at `fe4f2ec`,
using the server's `dev-bearer` feature, serial execution and in-memory storage.
All six test functions completed. They print observations rather than assert
desired security behaviour, so the result is evidence to inspect, not six
passing acceptance regressions. [Raw output and source hashes](evidence/2026-11-01/solid-parity-intake/manifest.json)
retain the controls and exact version.

| Contract | Observed result | Scope and next acceptance |
| --- | --- | --- |
| Advertised container ACL governs its children (row 290) | GET advertises `/c/.acl`. With inherited public Read, an owner PUT of a restrictive ACL there returns 201, but anonymous child and container reads remain 200. Writing the same policy to `/c.acl` changes both anonymous reads to 401 while the owner remains 200. | Reconcile advertised, written and resolved ACL locations with an explicit migration policy. Preserve protection for existing pods and reject ambiguous policy states. This local fixture demonstrates continued inherited public access. |
| ACL writes preserve a usable policy (row 266) | Malformed JSON, an empty JSON array and plain text are stored with 201, followed by 403 for owner reads and repair PUT. Malformed Turtle is refused with 409 and the valid policy remains usable. | Validate candidate ACL syntax and shape before storage through the shared policy owner; cover every mutation path. Source confirms the guard returns true when it cannot parse, relying incorrectly on later validation. |
| Provisioning produces an owner-usable pod (row 287) | Open registration returns 201 and a WebID, but owner GET and PUT both receive 403; anonymous profile GET is 401. Storage lists the new container and metadata without an owner ACL. | Provision the authenticated/established owner policy, intended public profile and private defaults at the resolver's actual locations, then test the complete registration journey. No claim that the development bearer is a production login mechanism. |
| ACL IRI targets resolve consistently (row 288) | Root-relative `/me/public/doc.ttl` grants access; absolute and document-relative forms refuse it. An absolute-target ACL PUT returns 201 and prevents the owner's repair. Library provisioning's public type index also refuses anonymous and owner reads. | Resolve ACL targets against a trusted, explicit document base and preserve origin boundaries. Check library consumers separately from server registration. |

Rows 266 and 288 extend the earlier [D2 effective-Control finding](2026-11-01-solid-http-delivery.md)
with additional inputs and causes. Keep their acceptance conditions linked;
counting every manifestation as an independent root cause would inflate the audit.
No production repair is claimed by this intake.

## New decision and remaining review

ADR-2011 proposes compile-time extension crates and all-method ownership of a
mounted prefix. Its activation is live while decision status is proposed and
implementation partial. It records the forge as the only mounted extension,
no generic extension trait yet, and open dotfile, shutdown and boot-failure
conditions. The imported forge probe distinguishes its successful ordinary
file read from dotfile refusal. Full source/lifecycle review of this ADR remains
pending; reading its text earns no whole-document assessment credit.

PATCH data corruption, lost updates and ignored conditional writes deserve
repair alongside the ACL/provisioning findings. The parity audit's concurrency,
UTF-8 and conditional-write observations have not been rerun in this intake.
Notification, relay and ActivityPub routes also need consumer-level acceptance;
existing library APIs do not establish a served product. Feature adoption still
requires the owner's intended protocol scope, rather than copying every JSS
plugin. The imported market comparison is bounded to that comparator and is
not a new estate market-positioning conclusion.

This intake adds one candidate ADR and no assessment credit. The next source
pass should reconcile policy locations and candidate ACL validation, with the
six observation groups providing starting fixtures. Existing assessed diagrams
remain bound to their reports; no rendered or imported panel gets new credit.
