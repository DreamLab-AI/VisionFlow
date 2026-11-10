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
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/migrate.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/migrate_fork.rs
verified_commit: 27e18995fc20790fa956acb41f7d8420544a13ec
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](15-authority-and-payment-repairs.md). This historical topic retains its declared source revision; the successor states the repaired contracts and acceptance boundaries.

## For developers

RESOLVED at solid-pod-rs 27e1899: the pod adapter roots the generic Git service at the data root, so the pod path component is applied once and the selected repository is the pod's own. This topic separates repository selection from the root-only WAC decision; passing that decision no longer risks serving a different repository, but it still gates only the pod root.

## For the business

Usable history export needs both correct routing and explicit authority over the repository's objects. A public container grant does not establish permission to export every private child or previous version. The forge is a separate adapter; its use of the shared export configuration is traced below, without certifying the whole forge.

## SP-14.1 The pod component is applied twice

```mermaid
flowchart TD
    REQUEST["request /alice/info/refs<br/>original path retained<br/>solid-pod-rs-server/src/lib.rs:5057"]
    ROOT["adapter checks that data_root/alice exists<br/>solid-pod-rs-server/src/lib.rs:5073"]
    SERVICE["RESOLVED at solid-pod-rs 27e1899: construct the service<br/>rooted at data_root, not at the pod<br/>solid-pod-rs-server/src/lib.rs:5164"]
    SLUG["service extracts slug alice<br/>service.rs:295"]
    JOIN["join root and slug<br/>data_root/alice<br/>service.rs:299<br/>guard.rs:92"]
    REPO["look for Git repository there<br/>service.rs:338"]
    REQUEST --> ROOT --> SERVICE --> SLUG --> JOIN --> REPO
```

The normal write hook records files under `data_root/alice`, and the service now selects that same repository, so the earlier nested `data_root/alice/alice` selection no longer occurs. The [assessment](../../estate-review/2026-11-01-solid-git-export-and-baseline.md) distinguishes source analysis, normal routing and a transport positive control.

## SP-14.2 Root permission precedes whole-repository export

```mermaid
flowchart TD
    WRITE["LDP private content can be committed<br/>git_mark_write builds pod repository path<br/>solid-pod-rs-server/src/lib.rs:4285"]
    MARK["stage the resource in Git history<br/>mark.rs:168"]
    READ["clone and fetch check Read<br/>only on the pod-root resource<br/>solid-pod-rs-server/src/lib.rs:5125<br/>solid-pod-rs-server/src/lib.rs:5130<br/>then only the pod owner is served<br/>solid-pod-rs-server/src/lib.rs:5153"]
    SELECT["resolve selected repository<br/>service.rs:338"]
    CGI["serve through Git HTTP backend<br/>no child ACL walk here<br/>service.rs:378"]
    WRITE --> MARK
    READ --> SELECT --> CGI
```

The handler now also refuses any caller other than the pod owner before the service runs (`solid-pod-rs-server/src/lib.rs:5149`-`5162`). The two paths still need a common export-authority contract. Correcting selection alone would connect normal stored history to the existing root-only gate. Git's object transport is not a per-resource WAC response; this diagram does not propose root Control or a filtered ref list as a sufficient replacement policy.

## SP-14.3 Mirror fetches stay in a private candidate store

```mermaid
flowchart TD
    FETCH["fetch upstream into a candidate store<br/>under mirror-work, outside repos<br/>mirror.rs:1257<br/>mirror.rs:2085"]
    VERDICT["decide whether each branch may move<br/>mirror.rs:769"]
    PUBLISH["push only approved commits by SHA<br/>one lease per ref, atomic<br/>mirror.rs:2164"]
    RP["fixed hardened receive-pack<br/>migrate.rs:105"]
    SERVED["served store holds the approved<br/>closure only"]
    HELD["held objects never reach<br/>the served store"]
    FETCH --> VERDICT --> PUBLISH --> SERVED
    RP --> PUBLISH
    VERDICT --> HELD
```

**What it shows:** a mirror fetch now lands in a private candidate store, the verdict is taken there, and only approved commits are pushed into the store that Forge serves. A held commit is never copied across, so no reader can find it by SHA.

**Why it is this way:** hiding `refs/mirror` only changed what smart-HTTP advertised; Forge's raw, HTML, JSON and archive readers call local Git with a caller-supplied revision and never consult that advertisement. Isolation by store was chosen over per-request checks (SP-GIT-02, Commit A `30135fec`). The publish uses an explicit lease per ref and no `+` refspec, because a `+` silently overrides the lease. Evidence is in the [publication assessment](../../estate-review/2026-11-02-solid-mirror-publication.md).

**Invariant:** the served store contains only objects reachable from approved branches and tags; `publish` is the only path that writes mirror content into it (mirror.rs:2164).

## SP-14.4 Every reader is gated, and old stores are migrated at boot

```mermaid
flowchart TD
    BOOT["ForgeService::new runs migration<br/>before serving<br/>crates/solid-pod-rs-forge/src/lib.rs:276<br/>crates/solid-pod-rs-forge/src/lib.rs:288"]
    MIG["rebuild each legacy mirror store<br/>migrate.rs:1474"]
    HSET["held set computed from approved roots<br/>migrate.rs:2297"]
    VERIFY["verify the rebuilt store object by object<br/>migrate.rs:2470"]
    FORK["sweep forks found through forge.parent<br/>migrate_fork.rs:133"]
    FTRIG["rebuild a fork holding ancestor-dropped<br/>or unreachable objects<br/>migrate_fork.rs:405<br/>migrate_fork.rs:447"]
    RESOLVE["readers resolve the repository<br/>crates/solid-pod-rs-forge/src/lib.rs:621"]
    GATE["503 while unmarked, recorded or journalled<br/>crates/solid-pod-rs-forge/src/lib.rs:651<br/>migrate.rs:739"]
    SMART["smart-HTTP uses the same gate<br/>bad path segments answer 404<br/>crates/solid-pod-rs-forge/src/lib.rs:786<br/>crates/solid-pod-rs-forge/src/lib.rs:850"]
    CGI["CGI still refuses unadvertised wants<br/>service.rs:427-429"]
    BOOT --> MIG --> HSET --> VERIFY
    MIG --> FORK --> FTRIG
    RESOLVE --> GATE
    SMART --> GATE
    SMART --> CGI
```

**What it shows:** stores written before the repair, and forks cloned from them, still held fetched objects. At boot each one is rebuilt from approved branches and tags only and verified object by object before it is swapped in. Every reader, smart-HTTP included, goes through one gate that refuses a store until migration has published it.

**Why it is this way:** `clone --local` hardlinks the whole object store, so forks inherit held objects, and deleting refs followed by `gc` is not reliable proof of removal. A failed migration must not leave a store open, so any journal gates readers and an incomplete pass gates every mirror owner (B1 `bcaf9cf6`, B2 `ff8048b`, decisions 21 to 39 in the evidence folder). The earlier CGI setting stays as a second layer.

**Invariant:** a store with a migration journal, or a mirror record without the publication marker, is never served (crates/solid-pod-rs-forge/src/lib.rs:651).

**Open:** no operator recourse yet for a store left gated at the attempt limit, over the size cap, or behind an unreadable owner directory; queued as B3 under SP-GIT-02.
