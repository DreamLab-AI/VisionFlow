---
id: VG-04
title: Downstream contracts — VisionClaw's pull model, and knowledgeGraph as deploy target
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-002]
sources:
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/docs/PUBLICATION-contract.md
  - ../visionGraph/README.md
verified_commit: 9e308164c
---

## VG-04.1 VisionClaw's ontology-publish.yml — pulls FROM visionGraph, not from knowledgeGraph

```mermaid
sequenceDiagram
    autonumber
    participant VC as VisionClaw ontology-publish.yml<br/>EXTERNAL — see VC-20/ADR-2106
    participant VG as jjohare/visionGraph<br/>PRIVATE — ONTOLOGY_SOURCE_REPO default
    participant BUILD as python -m pipeline.build<br/>the vault's OWN pipeline, not a bespoke converter
    participant PACK as scripts/ontology/pack-pod-resources.py<br/>EXTERNAL — VisionClaw repo
    participant POD as embedded solid-pod-rs<br/>SOLID_POD_URL — /public/ontology/

    VC->>VG: checkout private repo #40;GITHUB_TOKEN cannot read it —<br/>needs a cross-repo PAT#41;
    VC->>BUILD: run knowledge/pages through pipeline.build
    BUILD-->>VC: 8,434 classes, 265,455 triples<br/>#40;run 34046473943, ADR-2106#41;
    VC->>PACK: pack-pod-resources.py — visionflow.ttl, context.jsonld,<br/>ontology.jsonld, index.jsonld
    PACK->>POD: deploy-jss job PUTs resources to SOLID_POD_URL
    Note over VC,POD: DOC-DRIFT #40;fixed 2026-09-06, ADR-2106#41;: an EARLIER inline<br/>md_to_ttl.py read Logseq key:: value lines and never the json-ld<br/>fence — shipped 0 owl:Class from 380 pages under example.org,<br/>valid syntax, matching digests. Replaced with the vault#39;s own<br/>pipeline.build + pack-pod-resources.py
    Note over POD: pull model, not push — a GitHub-hosted runner cannot reach the<br/>LAN pod — no self-hosted runner is registered #40;ADR-2106 context#41;
```

## VG-04.2 Two distribution paths from one authored corpus

```mermaid
flowchart TB
    VG["visionGraph knowledge/pages<br/>8,671 authored pages — THIS repo"]
    PATH1["Path 1: publish.yml<br/>see VG-03.1"]
    PATH2["Path 2: VisionClaw ontology-publish.yml<br/>EXTERNAL — VC-20, ADR-2106"]
    SITE["narrativegoldmine.com<br/>via gh-pages on DreamLab-AI/knowledgeGraph<br/>EXTERNAL — see KG-05.3"]
    POD["embedded solid pod<br/>/public/ontology/ on the VisionClaw server<br/>SOLID_POD_URL, default http://localhost:4000/solid"]
    VG --> PATH1 --> SITE
    VG --> PATH2 --> POD
    note1["DIVERGENCE: PATH1 deploys a STATIC public website via GitHub Pages;<br/>PATH2 deploys ontology TRIPLES into VisionClaw#39;s OWN identity-scoped<br/>pod. Same source corpus, two independent consumers, two independent<br/>release cadences #40;ADR-VG-002 context#41;"]
```

## VG-04.3 Three distinct identities — source, distribution, deployed artefact

```mermaid
flowchart LR
    S["SOURCE identity<br/>visionGraph @ commit — the authored corpus of record"]
    D["DISTRIBUTION identity<br/>knowledgeGraph @ gh-pages commit —<br/>a separate distribution tree, EXTERNAL: KG-*"]
    A["DEPLOYED ARTEFACT identity<br/>narrativegoldmine.com live response —<br/>what a browser actually fetched"]
    S -->|"publish.yml builds + pushes"| D
    D -->|"GitHub Pages serves"| A
    note1["INVARIANT: source, distribution and deployed artefact identities<br/>must NOT be collapsed into one revision #40;PUBLICATION-contract.md:7#41;.<br/>ADR-VG-002 proposes ONE release manifest binding source/corpus/<br/>pipeline/explorer/dependency/policy identity plus output hashes —<br/>proposed, NOT yet activation_status: live"]
    note2["Archived Logseq history remains the citation target for pre-split<br/>commit hashes #40;PUBLICATION-contract.md:7#41; — a hash from before the<br/>vault split does not resolve inside this repo"]
```
