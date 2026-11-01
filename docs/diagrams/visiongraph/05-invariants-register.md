---
id: VG-05
title: Publication policies and the still-proposed generation acceptance contract
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-001, ADR-VG-002]
sources:
  - ../project/crates/visionclaw-domain/src/vault/mod.rs
  - ../visionGraph/docs/PUBLICATION-contract.md
  - ../visionGraph/docs/adr/ADR-VG-001-publication-policy-boundaries.md
  - ../visionGraph/docs/adr/ADR-VG-002-generation-and-consumer-identity.md
  - ../visionGraph/.github/workflows/publish.yml
  - ../ontomark/crates/ontomark-compile/src/input/page.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/generation.rs
  - ../ontomark/crates/ontomark-compile/src/build/publish.rs
verified_commit: {visiongraph: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6, visionclaw: 75b72416bac176ca8fde715cac3bd7c77657b3e7, ontomark: 0caae5d4b695cb720942b9b9212017c00a80ad41}
---

## For developers

Keep current consumer policies, compiler evidence and historical acceptance requirements distinct; the [section-by-section ADR assessment](../../estate-review/2026-11-01-visiongraph-publication-and-identity.md) records what these mechanisms do and what remains unproved.

## For the business

A public website, a local graph and a pod can intentionally contain different views, but users need clear policy and freshness information rather than one success label for all of them.

## VG-05.1 Current publication and graph-inclusion policies are distinct

```mermaid
flowchart TB
    PAGE["Ontomark Page.is_public<br/>only YAML boolean true grants publication"]
    SITE["Site publication staging<br/>public pages from knowledge and working<br/>held paths excluded, selected pages scanned"]
    PROJECT["Compiler projects before structural closure<br/>private-link policy comes from options/profile"]
    LOCAL["VisionClaw PageMeta.is_kg_included<br/>public OR a valid owl_class value"]
    WHY["Different consumer predicates<br/>local inclusion does not grant site publication"]
    PAGE --> SITE --> PROJECT --> WHY
    LOCAL --> WHY
    %% Source: crates/ontomark-compile/src/input/page.rs:269
    %% Source: crates/ontomark-compile/src/build/publish.rs:200
    %% Source: crates/ontomark-compile/src/build/mod.rs:615
    %% Source: crates/ontomark-compile/src/build/mod.rs:658
    %% Source: crates/ontomark-compile/src/build/mod.rs:678
    %% Source: crates/visionclaw-domain/src/vault/mod.rs:162
```

**What it shows:** exact predicate differences and the compiler's projection order; these source functions replace the older blanket invariant list (`input/page.rs:269`, `vault/mod.rs:162`, `build/mod.rs:678`).
**Why it is this way:** local structural graph inclusion and public release are different policies; ADR-VG-001 requires those differences and their diagnostics to be intentional and tested (`ADR-VG-001-publication-policy-boundaries.md:32`).

## VG-05.2 Proposed acceptance requirements remain open

```mermaid
flowchart TB
    P1["ADR-VG-001<br/>explicit included, excluded or invalid dispositions<br/>typed flags and intentional consumer policies"]
    P2["Visibility fixtures across every export<br/>private ancestry, namespaces, held paths<br/>conflicting flags and duplicate identity"]
    G1["ADR-VG-002<br/>source, corpus, compiler, explorer<br/>dependency and policy identity plus output hashes"]
    G2["Deletion and equal-count replacement<br/>namespace moves, interrupted activation, rollback<br/>independent browser and consumer acceptance"]
    STATUS["Both proposed, partial, inactive<br/>not a deployment certification"]
    P1 --> P2 --> STATUS
    G1 --> G2 --> STATUS
    %% Source: docs/adr/ADR-VG-001-publication-policy-boundaries.md:5
    %% Source: docs/adr/ADR-VG-001-publication-policy-boundaries.md:32
    %% Source: docs/adr/ADR-VG-001-publication-policy-boundaries.md:36
    %% Source: docs/adr/ADR-VG-002-generation-and-consumer-identity.md:32
    %% Source: docs/adr/ADR-VG-002-generation-and-consumer-identity.md:36
```

**What it shows:** the proposed end-to-end acceptance boundary, not the status of whether the existing site is online (`ADR-VG-001-publication-policy-boundaries.md:36`).
**Why it is this way:** the records explicitly preserve incomplete consumer acceptance despite component implementation and historical fixes (`ADR-VG-002-generation-and-consumer-identity.md:36`).

## VG-05.3 A compiler generation marker is not the final-site release manifest

```mermaid
flowchart TB
    INPUT["Digest source page hashes and vocabulary<br/>track source commit and dirty state"]
    MARKER["Generation fields include content/ontology digests<br/>counts, vocabulary version, scope and artefact hashes"]
    TWO["Root marker plus data-scoped marker<br/>root includes data marker, neither includes itself"]
    SITE["Workflow then overlays static files<br/>compresses selected outputs and adds UI/notices"]
    GAP["OPEN: final bytes and loaded consumers<br/>need their own generation binding"]
    INPUT --> MARKER --> TWO --> SITE --> GAP
    %% Source: crates/ontomark-compile/src/build/mod.rs:968
    %% Source: crates/ontomark-compile/src/build/mod.rs:977
    %% Source: crates/ontomark-compile/src/build/generation.rs:90
    %% Source: crates/ontomark-compile/src/build/mod.rs:1008
    %% Source: .github/workflows/publish.yml:261
    %% Source: .github/workflows/publish.yml:358
    %% Source: .github/workflows/publish.yml:441
    %% Source: docs/adr/ADR-VG-002-generation-and-consumer-identity.md:32
```

**What it shows:** a real compiler manifest now exists; it lacks the complete final-site and loaded-consumer identity contract proposed by the ADR (`generation.rs:90`, `ADR-VG-002-generation-and-consumer-identity.md:32`).
**Why it is this way:** compiler emission precedes publication transformations; the earlier public-receipt audit found one static-context hash mismatch at that boundary without establishing malicious alteration (`build/mod.rs:1015`, `publish.yml:261`).

## VG-05.4 Historical failures, repairs and successor boundaries

```mermaid
flowchart TB
    OLD["September synthetic probes<br/>old Python flag and private-ancestor defects"]
    FIX["September execution qualification<br/>projection before inference and rollback repairs"]
    RETIRE["October historical banner<br/>Python retired, notes now current Obsidian pages"]
    NOW["Current workflow invokes released ontomark<br/>old findings require successor-specific evidence"]
    KEEP["Keep broader acceptance obligations open<br/>historical repair is not live consumer acceptance"]
    OLD --> FIX --> RETIRE --> NOW --> KEEP
    %% Source: docs/adr/ADR-VG-002-generation-and-consumer-identity.md:46
    %% Source: docs/adr/ADR-VG-002-generation-and-consumer-identity.md:54
    %% Source: docs/PUBLICATION-contract.md:3
    %% Source: docs/PUBLICATION-contract.md:8
    %% Source: .github/workflows/publish.yml:48
    %% Source: docs/adr/ADR-VG-001-publication-policy-boundaries.md:36
```

**What it shows:** dated succession instead of presenting retired-parser counterexamples as current behaviour; the historical banner's reference to vault also lags the workflow's ontomark pin (`PUBLICATION-contract.md:3`, `publish.yml:48`).
**Why it is this way:** generation identity cannot certify privacy, and implementation changes do not automatically close the proposed acceptance contract (`ADR-VG-002-generation-and-consumer-identity.md:48`, `ADR-VG-001-publication-policy-boundaries.md:36`).
