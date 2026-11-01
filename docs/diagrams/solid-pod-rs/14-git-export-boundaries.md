---
id: SP-14
title: Pod Git repository selection and export authority
area: solid-pod-rs
governing: [../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2001, ADR-2004]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/service.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/guard.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/mark.rs
verified_commit: b644e55c76d6e89b0e41ec1c7aa44510b20ccf68
---

## For developers

The pod adapter and generic Git service both apply the pod path component. This topic separates repository selection from the root-only WAC decision; passing that decision does not prove that the intended repository is served.

## For the business

Usable history export needs both correct routing and explicit authority over the repository's objects. A public container grant does not establish permission to export every private child or previous version. The forge is a separate adapter and is not certified by this assessment.

## SP-14.1 The pod component is applied twice

```mermaid
flowchart TD
    REQUEST["request /alice/info/refs<br/>original path retained<br/>lib.rs:4724"]
    ROOT["adapter root is data_root/alice<br/>lib.rs:4740"]
    SERVICE["construct service with that root<br/>lib.rs:4816"]
    SLUG["service extracts slug alice<br/>service.rs:294"]
    JOIN["join root and slug<br/>data_root/alice/alice<br/>service.rs:298<br/>guard.rs:92"]
    REPO["look for Git repository there<br/>service.rs:337"]
    REQUEST --> ROOT --> SERVICE --> SLUG --> JOIN --> REPO
```

The normal write hook records files under `data_root/alice`, so this is a different repository selection. A missing nested repository prevents export; an existing nested repository changes which data the route serves. The [assessment](../../estate-review/2026-11-01-solid-git-export-and-baseline.md) distinguishes source analysis, normal routing and a transport positive control.

## SP-14.2 Root permission precedes whole-repository export

```mermaid
flowchart TD
    WRITE["LDP private content can be committed<br/>git_mark_write builds pod repository path<br/>lib.rs:3952"]
    MARK["stage the resource in Git history<br/>mark.rs:168"]
    READ["clone and fetch check Read<br/>only on the pod-root resource<br/>lib.rs:4792<br/>lib.rs:4797"]
    SELECT["resolve selected repository<br/>service.rs:337"]
    CGI["serve through Git HTTP backend<br/>no child ACL walk here<br/>service.rs:377"]
    WRITE --> MARK
    READ --> SELECT --> CGI
```

The two paths need a common export-authority contract. Correcting selection alone would connect normal stored history to the existing root-only gate. Git's object transport is not a per-resource WAC response; this diagram does not propose root Control or a filtered ref list as a sufficient replacement policy.
