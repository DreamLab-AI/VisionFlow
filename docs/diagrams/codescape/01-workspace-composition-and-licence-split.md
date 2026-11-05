---
id: CS-01
title: Workspace composition, licence split and the isolation fence
area: codescape
governing:
  - ../codescape/docs/BASELINE-codescape.md
  - ../codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
adrs: [codescape:ADR-0001, visionclaw:ADR-2143]
sources:
  - ../codescape/Cargo.toml
  - ../codescape/LICENSE
  - ../codescape/AGENTS.md
  - ../codescape/docs/BASELINE-codescape.md
  - ../codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
  - ../codescape/client/package.json
  - ../codescape/crates/codescape-model/Cargo.toml
  - ../codescape/crates/codescape-model/src/lib.rs
  - ../codescape/crates/codescape-model/src/wire.rs
  - ../codescape/crates/codescape-collab/Cargo.toml
  - ../codescape/crates/codescape-ingest/Cargo.toml
  - ../codescape/crates/codescape-layout/Cargo.toml
  - ../codescape/crates/codescape-server/Cargo.toml
  - ../codescape/crates/codescape-server/tests/isolation.rs
verified_commit: 446f8e386d7fc31e4cf28db25bf976c636b149ce
---

## CS-01.1 The five crates and their dependency direction

```mermaid
flowchart BT
    WIRE["visionclaw-wire 0.1.0 from crates.io<br/>../codescape/Cargo.toml:48"]
    MODEL["codescape-model<br/>MIT OR Apache-2.0, publish = true<br/>codescape-model/Cargo.toml:7,15"]
    COLLAB["codescape-collab<br/>MIT OR Apache-2.0, publish = true<br/>codescape-collab/Cargo.toml:7,15"]
    INGEST["codescape-ingest<br/>AGPL-3.0-only via workspace<br/>codescape-ingest/Cargo.toml:7"]
    LAYOUT["codescape-layout<br/>AGPL-3.0-only via workspace<br/>codescape-layout/Cargo.toml:7"]
    SERVER["codescape-server<br/>AGPL-3.0-only via workspace<br/>codescape-server/Cargo.toml:7"]
    GRAPH["graphicle pinned git rev, default features off<br/>../codescape/Cargo.toml:53"]
    CUST["cust 0.3.2, optional<br/>codescape-layout/Cargo.toml:24"]
    CLIENT["client desktop bundle, AGPL-3.0-only, private<br/>client/package.json:4,6"]
    WIRE --> MODEL
    MODEL --> COLLAB
    MODEL --> INGEST
    GRAPH --> INGEST
    MODEL --> LAYOUT
    CUST -.->|"feature cuda"| LAYOUT
    MODEL --> SERVER
    COLLAB --> SERVER
    INGEST --> SERVER
    LAYOUT --> SERVER
    SERVER -.->|"serves static bundle"| CLIENT
    NOTE1["INVARIANT: ingest is the only graphicle user, model and collab take no AGPL dependency<br/>codescape-ingest/Cargo.toml:14 codescape-model/Cargo.toml:21"]
    NOTE2["see VC-39.1 for the VisionClaw-side view of this DAG"]
    INGEST --- NOTE1
    SERVER --- NOTE2
```

**What it shows.** Arrows run from dependency to dependent; the two I/O-free crates are the permissive pair, everything that does I/O is AGPL. The cuda feature adds only cust to layout (`codescape-server/Cargo.toml:22`).

## CS-01.2 Where each crate gets its licence

```mermaid
flowchart TD
    ROOT["workspace.package licence default AGPL-3.0-only<br/>../codescape/Cargo.toml:34"]
    PUB["workspace.package publish = false<br/>../codescape/Cargo.toml:38"]
    OVR["model and collab override licence and publish in their own manifests<br/>../codescape/Cargo.toml:7,8"]
    INH["ingest, layout, server inherit with license.workspace = true<br/>../codescape/Cargo.toml:4,5,6"]
    TBL["LICENSE table: per directory, client and everything else AGPL-3.0-only<br/>LICENSE:5,6,7,8,9,10"]
    ROOT --> INH
    PUB --> INH
    OVR --> TBL
    INH --> TBL
    DEP["workspace dependency table gives model and collab a version so they can be packaged<br/>../codescape/Cargo.toml:41,44"]
    OVR --> DEP
    EXT["EXTERNAL: crates.io shows model and collab at 0.1.0, not re-checked here"]
    DEP --> EXT
    TENS["TENSION: LICENSE is the only prose statement of the split besides the manifests, there is no REUSE or cargo-deny file<br/>LICENSE:1"]
    TBL --- TENS
```

**What it shows.** AGPL is the default and the two permissive crates opt out in their own manifests (`codescape-model/Cargo.toml:7`). Packaging them needs versioned workspace entries, which is why `codescape-ingest` has none (`../codescape/Cargo.toml:42`).

## CS-01.3 Isolation tests that hold the fence

```mermaid
flowchart TD
    BAN["FORBIDDEN names and prefixes ruvector, solid-pod-rs<br/>isolation.rs:16,37"]
    I1["I1 default build graph has no knowledge-work crate<br/>isolation.rs:106"]
    I1C["I1 cuda resolution, visionclaw-gpu must be absent<br/>isolation.rs:121,128"]
    NEG["negative control injects a forbidden package so the check cannot pass vacuously<br/>isolation.rs:161"]
    I1A["I1a only visionclaw-wire may start with visionclaw-, both feature sets<br/>isolation.rs:139,147"]
    I10["I10 no git source naming VisionClaw, exactly five path packages<br/>isolation.rs:215,230,246"]
    I11["I11 no permissive member ships with an AGPL dependency, dev-dependencies excluded<br/>isolation.rs:324,328,393"]
    I8["I8 spawns the binary with RUVECTOR_PG_CONNINFO set and expects exit 2<br/>isolation.rs:272,291"]
    BAN --> I1
    BAN --> I1C
    I1 --> NEG
    I1C --> I1A
    I1A --> I10
    I10 --> I11
    I8 --> I11
    DEBT1["DEBT: I1 scans package names only, a renamed crate would pass<br/>BASELINE-codescape.md:81"]
    DEBT2["DEBT: egress allowlist and no-corpus-mount are not tested invariants<br/>BASELINE-codescape.md:97"]
    I1 --- DEBT1
    I8 --- DEBT2
```

**What it shows.** Every check runs `cargo metadata --locked`, in the default and the `--all-features` resolution (`isolation.rs:126,143,223`). The boot refusal is the one runtime check, using the real binary rather than the runner's own environment (`isolation.rs:273`).

## CS-01.4 What the extraction removed and what replaced it

```mermaid
flowchart LR
    subgraph BEFORE["Inside VisionClaw before ADR-0001"]
        B1["wire.rs included VisionClaw's fixture file by path attribute<br/>ADR-0001-extraction-from-visionclaw.md:54"]
        B2["cuda feature depended on visionclaw-gpu by path for compiled PTX<br/>ADR-0001-extraction-from-visionclaw.md:54"]
        B3["every Codescape crate AGPL-3.0-only<br/>ADR-0001-extraction-from-visionclaw.md:55"]
    end
    subgraph AFTER["Own repository"]
        A1["visionclaw-wire from crates.io, model re-exports it<br/>codescape-model/src/wire.rs:17<br/>codescape-model/src/lib.rs:13,14"]
        A2["PTX read at run time from CODESCAPE_PTX_DIR, manifest and ABI checked<br/>ADR-0001-extraction-from-visionclaw.md:65"]
        A3["model and collab relicensed MIT OR Apache-2.0, rest AGPL<br/>ADR-0001-extraction-from-visionclaw.md:61,62"]
    end
    B1 --> A1
    B2 --> A2
    B3 --> A3
    DRIFT1["DOC-DRIFT: status line says activation has not happened, frontmatter and close-out say live<br/>ADR-0001-extraction-from-visionclaw.md:50,7,100"]
    DRIFT2["DOC-DRIFT: model crate docs still cite ADR-2143 as the invariant source, not ADR-0001<br/>codescape-model/src/lib.rs:9"]
    A2 --- DRIFT1
    A1 --- DRIFT2
    NOTE["see VC-42.1 for the build-graph versus runtime-artefact split"]
    A2 --- NOTE
```

**What it shows.** ADR-0001 swapped two path couplings for a published crate and a runtime artefact (`ADR-0001-extraction-from-visionclaw.md:64,65`). The ADR keeps a stale status paragraph under later close-out sections.

## CS-01.5 Pinned external inputs and why the lock stays separate

```mermaid
flowchart TD
    WHY["Own workspace and own Cargo.lock so graphicle feature unification cannot change number handling elsewhere<br/>../codescape/Cargo.toml:12,14,15,16"]
    GRAPH["graphicle by git rev, never transcription or bedrock features<br/>../codescape/Cargo.toml:7,9,10"]
    TS["tree-sitter pinned exactly to 0.25.10<br/>../codescape/Cargo.toml:55"]
    WIRE2["visionclaw-wire by version from crates.io, never path or git<br/>../codescape/Cargo.toml:45,46,47,48"]
    RULE["bumping graphicle is its own change with an ingest test re-run<br/>AGENTS.md:76"]
    LOCKED["every gate runs with --locked so the committed lock decides<br/>BASELINE-codescape.md:87"]
    WHY --> GRAPH
    WHY --> TS
    GRAPH --> RULE
    WIRE2 --> LOCKED
    RULE --> LOCKED
    DEBT["DEBT: graphicle is a git dependency, so a cold build needs network access to fetch the pinned rev<br/>../codescape/Cargo.toml:53"]
    GRAPH --- DEBT
    MSRV["rust-version 1.94 is declared for all members<br/>../codescape/Cargo.toml:33"]
    LOCKED --- MSRV
    NOTE["see CS-04.5 for the CI toolchain that has to honour it"]
    MSRV --- NOTE
```

**What it shows.** The separate lock is the structural form of invariant I6 (`BASELINE-codescape.md:87`). The two third-party pins are the places an upgrade changes extraction output.
