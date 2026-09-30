---
id: VG-04
title: Downstream contracts — VisionClaw's pull model, and knowledgeGraph as deploy target
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-002]
sources:
  - ../visionGraph/docs/PUBLICATION-contract.md
  - ../visionGraph/README.md
  - ../project/.github/workflows/ontology-publish.yml
  - ../project/docs/adr/ADR-2106-ontology-pull-model-into-the-embedded-pod.md
verified_commit: {visionclaw: 58f04f2eb272a2707737f2065f8241b931229e81, visiongraph: ac6274f9f5e12375f92086ccb9adba50c965ecf9}
---

## VG-04.1 VisionClaw's ontology-publish.yml — pulls FROM visionGraph, not from knowledgeGraph

```mermaid
sequenceDiagram
    autonumber
    participant VC as validate-source job<br/>ontology-publish.yml:44
    participant VG as jjohare/visionGraph checkout<br/>ontology-publish.yml:147-152, ONTOLOGY_SOURCE_REPO default — line 30
    participant BUILD as vault build (Rust binary)<br/>ontology-publish.yml:165-172, ADR-2113
    participant PACK as pack-pod-resources.py<br/>ontology-publish.yml:187-192
    participant POD as deploy-jss job<br/>ontology-publish.yml:301, PUT loop 346-365

    VC->>VG: checkout repository: env.ONTOLOGY_SOURCE_REPO<br/>#40;GITHUB_TOKEN cannot read a private cross-account repo —<br/>needs ONTOLOGY_SOURCE_TOKEN, checked lines 65-84#41;
    VG->>BUILD: ./target/release/vault --repo vault-source build --vault knowledge<br/>--out #36;GITHUB_WORKSPACE/output/vault
    BUILD->>PACK: pack-pod-resources.py output/vault output/pod
    Note over BUILD,PACK: 8,434 classes, 265,455 triples, substance floor 4000/100k<br/>#40;run 34046473943 — ADR-2106-ontology-pull-model-into-the-embedded-pod.md:38#41;
    PACK->>POD: curl -X PUT SOLID_POD_URL + JSS_PUBLIC_PATH/visionflow.ttl<br/>+ context.jsonld + ontology.jsonld + index.jsonld
    Note over VC,POD: history #40;ontology-publish.yml:126-131#41;: an EARLIER inline md_to_ttl.py<br/>read Logseq key:: value lines and never the json-ld fence — shipped<br/>0 owl:Class from 380 pages under example.org. First replaced by<br/>Python pipeline.build #40;ADR-2106#41;, itself since DELETED and replaced<br/>by the vault#39;s OWN Rust binary #40;ADR-2112 removed pipeline.build,<br/>ADR-2113 landed vault build — same binary visionGraph#39;s publish.yml uses#41;
    Note over POD: pull model, not push — a GitHub-hosted runner cannot reach the<br/>LAN pod — no self-hosted runner is registered<br/>#40;ADR-2106-ontology-pull-model-into-the-embedded-pod.md:27, deploy-jss condition ontology-publish.yml:305#41;
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
    note2["Archived Logseq history remains the citation target for pre-split<br/>commit hashes #40;PUBLICATION-contract.md:7#41; — a hash from before the<br/>vault split does not resolve inside this repo. This repo is a history-<br/>preserving split of jjohare/logseq at its 'obsidian' branch #40;3aeae0d05#41;<br/>README.md:160-164 — the split REWROTE every commit, so the equivalent<br/>commit here is 485977fb6; jjohare/logseq is now read-only and is the<br/>citation target for anything OLDER than the split — README.md:166"]
```
