---
id: SP-13
title: Candidate ACL control and truthful discovery
area: solid-pod-rs
governing: [../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2005]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/resolver.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/tests/acl_candidate_control.rs
verified_commit: 27e18995fc20790fa956acb41f7d8420544a13ec
---

## For developers

Candidate validation composes the shared policy reader and evaluator. Historical SP-04 records the earlier syntactic guard; this topic records the repair. Existing parser information loss, target-IRI limitations and concurrent updates remain outside this change.

## For the business

Malformed or inapplicable permission changes are refused before replacing the policy, and the advertised container ACL is the one enforcement reads. Existing misplaced ACL files still need migration; package publication and deployment are separate steps.

## SP-13.1 Evaluate Control on the candidate destination

```mermaid
flowchart TD
    CAND["candidate after request admission<br/>lib.rs:974"]
    META["metadata body is not an ACL<br/>Control admission still applies<br/>lib.rs:982"]
    READ["shared bounded policy reader<br/>require a found policy<br/>lib.rs:988"]
    VALID["validate supported conditions<br/>and RDF syntax for non-JSON ACLs<br/>lib.rs:993<br/>lib.rs:1005"]
    EVAL["same linked identities and balance<br/>evaluate Control on protected destination<br/>lib.rs:1009<br/>lib.rs:1015"]
    PATCH["JSON PATCH checks resulting bytes<br/>before quota and storage<br/>lib.rs:2120"]
    COPY["accompanying COPY policy installed<br/>only if destination Control survives<br/>lib.rs:3332"]
    CAND -->|metadata| META
    CAND -->|ACL| READ --> VALID --> EVAL
    PATCH --> CAND
    COPY --> CAND
```

An invalid candidate returns false to the caller. Direct writes refuse it; resource COPY skips an unsuitable accompanying ACL. Admission may already have collected a payment, so the guard does not establish transactional charging. The [repair assessment](../../estate-review/2026-11-01-solid-acl-control-repair.md) distinguishes executed route tests from unverified RDF ACL PATCH interoperability.

## SP-13.2 Follow the ACL link through to enforced access

```mermaid
flowchart TD
    GET["GET container /shared/<br/>client reads returned ACL link<br/>acl_candidate_control.rs:339"]
    LINK["success and not-found discovery<br/>use shared sidecar mapping<br/>ldp.rs:153<br/>ldp.rs:1805"]
    MAP["existing resolver association<br/>/shared/ maps to /shared.acl<br/>resolver.rs:333"]
    PUT["owner writes restrictive policy<br/>at discovered location<br/>acl_candidate_control.rs:382"]
    CHECK["child read: anonymous denied<br/>owner still authorised<br/>acl_candidate_control.rs:392<br/>acl_candidate_control.rs:396"]
    GET --> LINK --> MAP --> PUT --> CHECK
```

The route test starts with public child access and observes its removal after the advertised-location write. Root discovery remains `/.acl`. The mapping does not relocate old inner policies or choose between conflicting files; the migration guide requires explicit policy selection and access checks.
