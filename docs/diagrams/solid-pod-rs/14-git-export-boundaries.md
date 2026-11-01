---
id: SP-14
title: Git repository selection and export authority
area: solid-pod-rs
governing: [../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2001, ADR-2004, ADR-2009, ADR-2010]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/service.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/guard.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/mark.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/mirror.rs
verified_commit: 1a66c2d9222c52782891938da1b0373a0b41ab28
---

## For developers

The pod adapter and generic Git service both apply the pod path component. This topic separates repository selection from the root-only WAC decision; passing that decision does not prove that the intended repository is served.

## For the business

Usable history export needs both correct routing and explicit authority over the repository's objects. A public container grant does not establish permission to export every private child or previous version. The forge is a separate adapter; its use of the shared export configuration is traced below, without certifying the whole forge.

## SP-14.1 The pod component is applied twice

```mermaid
flowchart TD
    REQUEST["request /alice/info/refs<br/>original path retained<br/>solid-pod-rs-server/src/lib.rs:4724"]
    ROOT["adapter root is data_root/alice<br/>solid-pod-rs-server/src/lib.rs:4740"]
    SERVICE["construct service with that root<br/>solid-pod-rs-server/src/lib.rs:4816"]
    SLUG["service extracts slug alice<br/>service.rs:295"]
    JOIN["join root and slug<br/>data_root/alice/alice<br/>service.rs:299<br/>guard.rs:92"]
    REPO["look for Git repository there<br/>service.rs:338"]
    REQUEST --> ROOT --> SERVICE --> SLUG --> JOIN --> REPO
```

The normal write hook records files under `data_root/alice`, so this is a different repository selection. A missing nested repository prevents export; an existing nested repository changes which data the route serves. The [assessment](../../estate-review/2026-11-01-solid-git-export-and-baseline.md) distinguishes source analysis, normal routing and a transport positive control.

## SP-14.2 Root permission precedes whole-repository export

```mermaid
flowchart TD
    WRITE["LDP private content can be committed<br/>git_mark_write builds pod repository path<br/>solid-pod-rs-server/src/lib.rs:3952"]
    MARK["stage the resource in Git history<br/>mark.rs:168"]
    READ["clone and fetch check Read<br/>only on the pod-root resource<br/>solid-pod-rs-server/src/lib.rs:4792<br/>solid-pod-rs-server/src/lib.rs:4797"]
    SELECT["resolve selected repository<br/>service.rs:338"]
    CGI["serve through Git HTTP backend<br/>no child ACL walk here<br/>service.rs:378"]
    WRITE --> MARK
    READ --> SELECT --> CGI
```

The two paths need a common export-authority contract. Correcting selection alone would connect normal stored history to the existing root-only gate. Git's object transport is not a per-resource WAC response; this diagram does not propose root Control or a filtered ref list as a sufficient replacement policy.

## SP-14.3 Mirror branch approval does not isolate held objects

```mermaid
flowchart TD
    HIDE["hide refs/mirror from discovery<br/>mirror.rs:1211"]
    FETCH["fetch remote branches into refs/mirror<br/>same repository object store<br/>mirror.rs:1442"]
    VERDICT["decide whether each branch may move<br/>mirror.rs:2043"]
    HELD["held branch does not move<br/>objects remain in fetched repository"]
    HTTP["forge smart-HTTP uses shared service<br/>crates/solid-pod-rs-forge/src/lib.rs:626"]
    CONFIG["CGI denies direct unadvertised-object requests<br/>overrides repository settings<br/>service.rs:427-429"]
    WANT["known held-tip request refused<br/>advertised public-tip fetch still works"]
    LIMIT["shared object store still needs<br/>a publication boundary for held content"]
    HIDE --> FETCH --> VERDICT --> HELD
    HTTP --> CONFIG --> WANT
    HELD -. "known held tip" .-> CONFIG
    HELD --> LIMIT
```

Two Rust HTTP-service regressions failed before this repair and pass at the declared revision, including against permissive repository settings. They decode a public-tip pack as a positive control. Deployed mirror acceptance remains open. Complete withholding requires a separate object-publication boundary during fetch and validation, because hidden refs are not a general repository read boundary. Evidence and qualifications are in the [assessment](../../estate-review/2026-11-01-solid-git-export-and-baseline.md#mirror-refs-configuration-contradicts-the-withholding-claim).
