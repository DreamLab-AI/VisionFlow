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
  - ../project/.github/workflows/ontology-publish.yml
  - ../project/docs/adr/ADR-2106-ontology-pull-model-into-the-embedded-pod.md
verified_commit: {visionclaw: dd82a07b0c54defc469a85e9d57fb15d29dc5e07, visiongraph: 482ba593bcffddca7d564d0b6c4b1fea225154fb}
---

## VG-04.1 VisionClaw's ontology-publish.yml — pulls FROM visionGraph, not from knowledgeGraph

```mermaid
sequenceDiagram
    autonumber
    participant VC as validate-source job<br/>ontology-publish.yml:44
    participant VG as jjohare/visionGraph checkout<br/>ontology-publish.yml:90-92, ONTOLOGY_SOURCE_REPO default — line 30
    participant BUILD as pipeline.build<br/>ontology-publish.yml:171
    participant PACK as pack-pod-resources.py<br/>ontology-publish.yml:181
    participant POD as deploy-jss job<br/>ontology-publish.yml:290, PUT loop 335-350

    VC->>VG: checkout repository: env.ONTOLOGY_SOURCE_REPO<br/>#40;GITHUB_TOKEN cannot read a private cross-account repo —<br/>needs ONTOLOGY_SOURCE_TOKEN, checked lines 60-84#41;
    VG->>BUILD: python -m pipeline.build knowledge/pages<br/>#36;GITHUB_WORKSPACE/output/vault
    BUILD->>PACK: pack-pod-resources.py output/vault output/pod
    Note over BUILD,PACK: 8,434 classes, 265,455 triples, substance floor 4000/100k<br/>#40;run 34046473943 — ADR-2106-ontology-pull-model-into-the-embedded-pod.md:38#41;
    PACK->>POD: curl -X PUT SOLID_POD_URL + JSS_PUBLIC_PATH/visionflow.ttl<br/>+ context.jsonld + ontology.jsonld + index.jsonld
    Note over VC,POD: DOC-DRIFT #40;fixed 2026-09-06, ADR-2106-ontology-pull-model-into-the-embedded-pod.md:16#41;: an EARLIER inline<br/>md_to_ttl.py read Logseq key:: value lines and never the json-ld<br/>fence — shipped 0 owl:Class from 380 pages under example.org,<br/>valid syntax, matching digests. Replaced with the vault#39;s OWN<br/>pipeline.build + pack-pod-resources.py
    Note over POD: pull model, not push — a GitHub-hosted runner cannot reach the<br/>LAN pod — no self-hosted runner is registered<br/>#40;ADR-2106-ontology-pull-model-into-the-embedded-pod.md:40, deploy-jss condition ontology-publish.yml:294#41;
```

## VG-04.2 Two distribution paths from one authored corpus

```mermaid
flowchart TB
    VG["visionGraph knowledge/pages<br/>authored pages — THIS repo"]
    PATH1["Path 1: publish.yml build-and-deploy job<br/>publish.yml:23-24 — see VG-03.1"]
    PATH2["Path 2: VisionClaw ontology-publish.yml<br/>ontology-publish.yml:1,30 ONTOLOGY_SOURCE_REPO<br/>EXTERNAL — VC-20, ADR-2106"]
    SITE["narrativegoldmine.com<br/>external_repository DreamLab-AI/knowledgeGraph<br/>publish.yml:240-246"]
    POD["embedded solid pod<br/>JSS_PUBLIC_PATH=/public/ontology — ontology-publish.yml:39<br/>default SOLID_POD_URL — ontology-publish.yml:38"]
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
    note2["Archived Logseq history remains the citation target for pre-split<br/>commit hashes #40;PUBLICATION-contract.md:7#41; — a hash from before the<br/>vault split does not resolve inside this repo. This repo is a history-<br/>preserving split of jjohare/logseq at its 'obsidian' branch #40;3aeae0d05#41;<br/>README.md:100-104 — the split REWROTE every commit, so the equivalent<br/>commit here is 485977fb6; jjohare/logseq is now read-only and is the<br/>citation target for anything OLDER than the split — README.md:106-109"]
```
