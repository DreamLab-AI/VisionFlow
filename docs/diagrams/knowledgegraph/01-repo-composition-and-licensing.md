---
id: KG-01
title: Repository composition and recorded licence scopes
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2001, ADR-2002]
sources:
  - ../knowledgeGraph/LICENSING.md
  - ../knowledgeGraph/COMMERCIAL.md
  - ../knowledgeGraph/NOTICE
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/README.md
  - ../knowledgeGraph/archive/github-workflows/build.yml
  - ../visionGraph/.github/workflows/publish.yml
verified_commit: {knowledgegraph: 4ed9ac159daf402b4fb252ce559dbeb894a8d91e, visiongraph: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6}
---

## For developers

Separate the retained main-branch source from the configured publisher and deployment branch; the [dated review](../../estate-review/2026-11-01-knowledgegraph-consumers-and-archive.md) records the tree census and committed licence-path drift.

## For the business

A reader can identify which component a licence statement covers without treating the repository's root licence as a grant over every asset; this map records repository declarations, not an independent determination of ownership or enforceability.

## KG-01.1 Retained source and configured publication

```mermaid
flowchart TB
    %% knowledgeGraph/archive/logseq-era-2026-09-22/README.md:1
    ARC["Logseq-era pages and Python pipeline<br/>Archived for history on 22 September<br/>Never a current build input"]
    %% knowledgeGraph/archive/github-workflows/build.yml:1
    OLD["Archived build workflow<br/>Build-and-verify definition, no deploy steps<br/>Stored outside the active workflow directory"]
    %% visionGraph/.github/workflows/publish.yml:127
    VALID["visionGraph source validation<br/>ontomark validate --vault all"]
    %% visionGraph/.github/workflows/publish.yml:175
    BUILD["Pinned ontomark build<br/>Machine artefacts and publication staging"]
    %% visionGraph/.github/workflows/publish.yml:262
    STATIC["Publisher copies its own static/ tree<br/>into site-data, including contexts"]
    %% visionGraph/.github/workflows/publish.yml:526
    DEST["Configured deployment target<br/>knowledgeGraph gh-pages, www/<br/>narrativegoldmine.com"]
    ARC --- OLD
    VALID --> BUILD --> STATIC --> DEST
    ARC -.->|"Historical source boundary"| VALID
    NOTE["OPEN: source configuration establishes the route<br/>A deployed generation needs its own receipt"]
    DEST --- NOTE
```

**What it shows:** The archived source and workflow are separate from the current publisher's validation, build and deploy configuration.
**Why it is this way:** The archive notice records the authoring migration; the active workflow identifies the producer and destination without proving a particular deployment succeeded.

## KG-01.2 Recorded grants and the unresolved archive-path mapping

```mermaid
flowchart TB
    SCOPE["LICENSING.md governs scope conflicts<br/>LICENSING.md:3"]
    AGPL["AGPL-3.0-or-later declaration<br/>pipeline/, root .github/, static/, docs/,<br/>examples/ and named root prose<br/>LICENSING.md:18"]
    MIT["MIT declaration for explorer/<br/>LICENSING.md:24<br/>WebVOWL attribution retained<br/>NOTICE:25"]
    DATA["ODbL-1.0 declaration<br/>ontology/ and dist/<br/>LICENSING.md:25"]
    GAP["DRIFT: committed table names old paths<br/>Archived pages and pipeline are now under archive/<br/>The archive notice does not restate grants"]
    %% knowledgeGraph/archive/logseq-era-2026-09-22/README.md:1
    OFFER["COMMERCIAL.md records alternative terms<br/>and an interim issue-tracker enquiry route<br/>COMMERCIAL.md:3,14"]
    SCOPE --> AGPL
    SCOPE --> MIT
    SCOPE --> DATA
    AGPL --> GAP
    DATA --> GAP
    GAP --> OFFER
    LIMIT["Scope declarations are documentary evidence<br/>No ownership or jurisdiction conclusion<br/>is established by this diagram"]
    OFFER --- LIMIT
```

**What it shows:** Three documented licence scopes coexist, but the committed table has not mapped the moved archive paths.
**Why it is this way:** The repository retained its historical licence documents through the migration; the audit preserves their declarations and records the path gap without silently importing an uncommitted correction.
