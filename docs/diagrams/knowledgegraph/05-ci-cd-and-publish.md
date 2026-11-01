---
id: KG-05
title: Publication — released compiler, assembled site and final-gate limits
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/ci-cd/build-and-gates.md
adrs: [ADR-2003, ADR-2013]
sources:
  - ../knowledgeGraph/docs/ci-cd/build-and-gates.md
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/README.md
  - ../knowledgeGraph/archive/github-workflows/build.yml
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/quartz/scripts/check-secrets.sh
  - ../visionGraph/quartz/scripts/check-urls.sh
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/fetchArtefact.ts
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - docs/adr/ADR-2013-sovereign-corpus-is-ecosystem-canon.md
verified_commit: {knowledgegraph: 4ed9ac159daf402b4fb252ce559dbeb894a8d91e, visiongraph: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6, ontomark: 0caae5d4b695cb720942b9b9212017c00a80ad41, visionflow: e7a9182}
---

## For developers

Follow the assembled publication tree through compression and final checks, rather than treating an earlier copy hash as a final receipt. [The assessment](../../estate-review/2026-11-01-knowledgegraph-publication-gates.md) records executable gate probes and decision dispositions.

## For the business

The publisher combines graph data, readable notes and an explorer. Its checks provide useful safeguards, but some inspect only file presence or counts; a successful step is not proof that every intended page and byte reached the site.

## KG-05.3 Configured publication flow and gate boundaries

```mermaid
sequenceDiagram
    participant V as visionGraph workflow
    participant O as ontomark v0.7.0
    participant C as oo-cert v2.0.1
    participant S as assembled www tree
    participant G as knowledgeGraph gh-pages
    V->>O: fetch release and verify SHA-256<br/>publish.yml:48-51,98-117
    V->>O: validate, collision lint, build both vaults<br/>publish.yml:127-183
    O-->>V: machine outputs, generation markers and staged notes<br/>mod.rs:990-1016, publish.yml:155-165
    V->>C: check derivation certificate<br/>publish.yml:217-234
    V->>V: body residue gate<br/>publish.yml:248-253
    V->>V: overlay tracked static contexts<br/>AFTER marker creation, publish.yml:257-262
    V->>V: required files, minimum counts and domain contracts<br/>publish.yml:264-324
    V->>S: build Quartz notes then copy machine files<br/>publish.yml:329-355
    V->>S: verify copied hashes, then gzip selected files<br/>publish.yml:347-373
    V->>S: size check BEFORE explorer build<br/>publish.yml:375-398,421-490
    V->>S: browser smoke, credential scan, layout gate<br/>publish.yml:492-520
    V->>G: deploy www then dispatch corpus-sync<br/>publish.yml:526-542
    Note over O,S: DEBT: overlay can change a manifest-listed context<br/>Compiler receipt does not inventory the final website
    Note over V,S: DEBT: final scanner skips binary files<br/>check-secrets.sh:150<br/>Actual gzip step hides a synthetic canary from this scan
    Note over S,G: OPEN: verify each deployment<br/>DEBT: presence and count checks<br/>do not verify every output identity<br/>check-urls.sh:90-95,115-126
```

**What it shows:** The configured producer is pinned by tag and digest, but publication continues to transform its outputs after the copy check. The final scan and layout gate have narrower coverage than their broad descriptions suggest.

**Why it is this way:** ADR-2013 establishes one governed corpus; the workflow separates compiler data, Quartz notes and the SPA. These stages have distinct evidence boundaries.

## KG-05.5 Archive, trigger scope and consumer representations

```mermaid
flowchart TB
    OLD["Archived Logseq source and Python pipeline<br/>logseq-era-2026-09-22/README.md:1-4<br/>Historical build workflow is archived too<br/>archive/github-workflows/build.yml:1"]
    TRIG["Push main: selected path filters, or manual run<br/>working/ and root assets/ absent from push list<br/>publish.yml:20-31"]
    BUILD["Build uses knowledge plus public working content<br/>publish.yml:155-183<br/>Root assets linked into Quartz staging<br/>publish.yml:195"]
    MIRROR["Producer emits page-id names and unambiguous title aliases<br/>mod.rs:842-919"]
    FETCH["Publishing viewer tries gzip then raw<br/>checks magic bytes and inflates in browser<br/>fetchArtefact.ts:22-38"]
    OLD -.-> TRIG --> BUILD --> MIRROR --> FETCH
    DRIFT["DRIFT: old build guide still names the retired root workflow<br/>build-and-gates.md:3<br/>Canon still names VisionClaw crates/vault as sole compiler<br/>Canon ADR-2013, decision 3"]
    %% DRIFT source: ADR-2013-sovereign-corpus-is-ecosystem-canon.md:52-58
    OLD -.-> DRIFT
    GAP["DEBT: a working-only edit is outside this push filter<br/>publish.yml:24-30<br/>Manual or matching-path runs can still include it"]
    TRIG --> GAP
```

**What it shows:** The archived pipeline is a historical implementation. Current publication includes sources outside the workflow's push filter and serves compressed representations to a viewer that knows how to inflate them.

**Why it is this way:** The compiler owns machine content, while the browser adapts the stored representation. Neither that adapter nor a later manual run makes omitted push triggers or external-reader compatibility implicit guarantees.
