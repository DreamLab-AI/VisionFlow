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
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/fs.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/acl_layout_report.rs
verified_commit: a28a74832dd0a6e47a531e34e4adc46a3946dfd3
---

## For developers

Candidate validation composes the shared policy reader and evaluator. Historical SP-04 records the earlier syntactic guard; this topic records the repair. Existing parser information loss, target-IRI limitations and concurrent updates remain outside this change.

## For the business

Malformed or inapplicable permission changes are refused before replacing the policy, and the advertised container ACL is the one enforcement reads. Existing misplaced ACL files still need migration; package publication and deployment are separate steps.

## SP-13.1 Evaluate Control on the candidate destination

```mermaid
flowchart TD
    CAND["check_proposed_acl after request admission<br/>lib.rs:1266"]
    META["not an ACL path: Ok, metadata body is not an ACL<br/>Control admission still applies<br/>lib.rs:1274"]
    NOPROT["no protected resource: WouldLockOut<br/>lib.rs:1277"]
    READ["shared bounded policy reader<br/>require a found policy, else Unparseable<br/>lib.rs:1281"]
    VALID["validate supported conditions, else WouldLockOut<br/>lib.rs:1285"]
    BASE["resolve absolute and relative IRIs<br/>against the configured pod origin<br/>lib.rs:1290"]
    SYN["not JSON-LD: RDF syntax check, else Unparseable<br/>lib.rs:1298<br/>lib.rs:1303"]
    EVAL["same linked identities and balance<br/>evaluate Control on protected destination<br/>lib.rs:1307<br/>lib.rs:1313"]
    REF["AclRefusal: Unparseable is 400, WouldLockOut is 409<br/>lib.rs:1221<br/>lib.rs:1233<br/>derived bytes (PATCH, COPY) also 400 / 409 lib.rs:1250"]
    PUT["PUT and POST refuse the client body<br/>lib.rs:2216<br/>lib.rs:2338"]
    PATCH["PATCH checks resulting bytes before quota and storage<br/>JSON Patch lib.rs:2598, RDF lib.rs:2641, create lib.rs:2698"]
    COPY["COPY destination ACL is checked lib.rs:4021<br/>accompanying sidecar installed only if Control survives lib.rs:4068"]
    PUT --> CAND
    PATCH --> CAND
    COPY --> CAND
    CAND -->|metadata| META
    CAND -->|ACL| NOPROT
    CAND -->|ACL| READ --> VALID --> BASE --> SYN --> EVAL
    READ -. refusal .-> REF
    VALID -. refusal .-> REF
    SYN -. refusal .-> REF
    EVAL -. refusal .-> REF
```

The guard now returns a typed refusal: a malformed body is 400 on every write verb (SP-PAR-266), a lockout or unsupported condition 409. Resource COPY still skips an unsuitable accompanying ACL rather than failing. Admission may already have collected a payment, so the guard does not establish transactional charging; the [repair assessment](../../estate-review/2026-11-01-solid-acl-control-repair.md) distinguishes executed route tests from unverified RDF ACL PATCH interoperability.

## SP-13.2 Follow the ACL link through to enforced access

```mermaid
flowchart TD
    GET["GET container /shared/<br/>client reads returned ACL link<br/>acl_candidate_control.rs:389"]
    HDR["success responses set Link and Accept-Ranges<br/>with the configured layout<br/>lib.rs:1766"]
    LINK["link_headers_with_layout<br/>ldp.rs:150<br/>ldp.rs:160"]
    ADV["acl_advertised_key: inner form for a non-root container<br/>under Inner, sidecar otherwise<br/>resolver.rs:415"]
    MAP["default Sidecar: /shared/ maps to /shared.acl<br/>resolver.rs:333"]
    PUT["owner writes restrictive policy<br/>at discovered location<br/>acl_candidate_control.rs:432"]
    CHECK["child read: anonymous denied<br/>owner still authorised, target is /shared.acl<br/>acl_candidate_control.rs:442<br/>acl_candidate_control.rs:446"]
    NF["DIVERGENCE: library not_found_headers still advertises the sidecar<br/>for every layout; the server does not call it<br/>ldp.rs:2015"]
    GET --> HDR --> LINK --> ADV --> MAP --> PUT --> CHECK
    LINK -.-> NF
```

The route test runs under the default layout only. Root discovery is `/.acl` in every layout.

## SP-13.3 Opt-in container ACL layout and what it guards

```mermaid
flowchart TD
    CFG["AppState.acl_layout, --acl-layout / JSS_ACL_LAYOUT<br/>default Sidecar lib.rs:523<br/>lib.rs:725"]
    ENUM["Sidecar reads c.acl only, Inner reads c/.acl only,<br/>Both reads c.acl then c/.acl when the sidecar is absent<br/>resolver.rs:351"]
    PROBE["acl_probe_keys per walk step<br/>resolver.rs:396<br/>resolver.rs:528"]
    DIR["a directory at an ACL key reads as absent,<br/>any other read failure denies<br/>resolver.rs:500"]
    BENEATH["nothing may be created beneath an ACL or meta name<br/>resolver.rs:429<br/>filesystem put refuses it fs.rs:399"]
    SIDEW["under Inner or Both, writing c.acl also needs Control<br/>on c/.acl, billed once lib.rs:1395"]
    REPORT["read-only acl_layout_report lists inner, both-differ,<br/>sidecar-only, directory-at-acl-key before opting in<br/>acl_layout_report.rs:1"]
    INV["INVARIANT: Both never denies merely because both forms exist,<br/>the sidecar wins; switching to Inner drops every sidecar<br/>container ACL and widens access to the parent grant"]
    CFG --> ENUM --> PROBE --> DIR
    PROBE --> BENEATH
    CFG --> SIDEW
    REPORT -.-> CFG
    ENUM -.-> INV
```

Inner and Both honour every existing inner `.acl`, including ones planted before Control was required, so the report is the migration step.
