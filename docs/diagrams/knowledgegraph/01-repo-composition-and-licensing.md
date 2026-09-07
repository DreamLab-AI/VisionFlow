---
id: KG-01
title: Repo composition and the three-way data/pipeline/explorer licensing boundary
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2001, ADR-2002]
sources:
  - ../knowledgeGraph/README.md
  - ../knowledgeGraph/LICENSING.md
  - ../knowledgeGraph/COMMERCIAL.md
  - ../knowledgeGraph/NOTICE
  - ../knowledgeGraph/CNAME
  - ../knowledgeGraph/docs/ecosystem.md
  - ../knowledgeGraph/docs/ci-cd/build-and-gates.md
  - ../knowledgeGraph/docs/architecture/explorer.md
  - ../knowledgeGraph/.github/workflows/build.yml
verified_commit: 2791111fc
---

## KG-01.1 Repository composition — a corpus, a pipeline and a viewer in one tree

```mermaid
flowchart TB
    subgraph REPO["DreamLab-AI/knowledgeGraph — published extraction, read-only mirror"]
        ONT["ontology/pages/<br/>8,138 Logseq .md pages<br/>README.md:52"]
        PIPE["pipeline/<br/>10 Python modules + tests<br/>README.md:19"]
        EXP["explorer/<br/>WasmVOWL derivative<br/>modern/ + license.txt"]
        DOCS["docs/<br/>BASELINE + architecture + ci-cd<br/>+ methodology + playbook + adr"]
        STATIC["static/ns/v2.jsonld<br/>JSON-LD context"]
        SCRIPTS["scripts/adr-index-gen.js"]
        GH[".github/workflows/build.yml<br/>build.yml:58"]
        LIC["LICENSE · LICENSE-DATA<br/>LICENSE-EXPLORER · NOTICE<br/>LICENSING.md · COMMERCIAL.md"]
        CNAME["CNAME<br/>narrativegoldmine.com"]
    end
    ONT --> PIPE
    PIPE --> EXP
    STATIC -.->|"@context of 7,531 blocks<br/>ecosystem.md:94"| ONT
    GH --> PIPE
    LIC --> ONT
    LIC --> PIPE
    LIC --> EXP
    note1["INVARIANT: this repo builds but never deploys — GH:contents:read is<br/>the whole permission set (build.yml:65-66); CNAME is served by GitHub<br/>Pages branch publishing, written by a DIFFERENT repo's CI (see KG-05.3)"]
```

## KG-01.2 Licensing scope — three licences, one repository, one governing file

```mermaid
flowchart LR
    subgraph SCOPE["LICENSING.md scope table — authoritative on conflict"]
        direction TB
        AGPL["AGPL-3.0-or-later<br/>pipeline/, static/, docs/ (original prose),<br/>examples/, README/LICENSING/COMMERCIAL/<br/>CONTRIBUTING/NOTICE, root .github/<br/>LICENSING.md:19-26"]
        MIT["MIT<br/>explorer/ — WebVOWL derivative<br/>LICENSE-EXPLORER<br/>LICENSING.md:24"]
        ODBL["ODbL-1.0<br/>ontology/, dist/ — UK CDPA 1988 s.9(3)<br/>computer-generated works<br/>LICENSING.md:28-29"]
    end
    GHDETECT["GitHub licence detector reads root LICENSE<br/>and labels the WHOLE repo 'AGPL-3.0' —<br/>wrong for explorer/ and ontology/<br/>LICENSING.md:28-29"]
    GHDETECT -.->|"DOC-DRIFT: detector output, not this table"| SCOPE
    COMM["COMMERCIAL.md — dual-licensing offer<br/>DreamLab AI Consulting Ltd holds rights in<br/>pipeline/static/docs/examples/ontology/dist<br/>COMMERCIAL.md:7-8"]
    AGPL -.->|"proprietary licence negotiable<br/>removes §13 network-copyleft"| COMM
    ODBL -.->|"negotiable data licence<br/>COMMERCIAL.md track 2"| COMM
    note1["INVARIANT: MIT for explorer/ is deliberate, not oversight — the<br/>identical code is MIT one repo away at DreamLab-AI/WasmVOWL<br/>(architecture/explorer.md:302-308); AGPL-ing a WebVOWL fork here<br/>would be hollow"]
```

## KG-01.3 Publication topology — this repo is the deploy TARGET, not the publisher

```mermaid
sequenceDiagram
    autonumber
    participant SRC as jjohare/logseq<br/>PRIVATE source repo — not this tree
    participant GHA as peaceiris/actions-gh-pages@v3<br/>ecosystem.md:228-236
    participant PAGES as gh-pages branch<br/>DreamLab-AI/knowledgeGraph
    participant SITE as narrativegoldmine.com<br/>CNAME
    participant THIS as this repo's own CI<br/>.github/workflows/build.yml:58

    Note over SRC,GHA: EXTERNAL: the ACTUAL publisher is visionGraph's<br/>.github/workflows/publish.yml — see VG-03.1/VG-04
    SRC->>GHA: build www/ (JSON-LD pipeline + React SPA + WASM)
    GHA->>PAGES: push external_repository=DreamLab-AI/knowledgeGraph<br/>publish_dir=www, secrets.ACCESS_TOKEN
    PAGES->>SITE: GitHub Pages serves gh-pages, cname narrativegoldmine.com<br/>ecosystem.md:239-241
    THIS->>THIS: build.yml runs 6 gates against ontology/pages/<br/>permissions: contents: read (build.yml:65-66)
    Note over THIS: INVARIANT: this workflow has no deploy step — no Pages action,<br/>no secrets.* reference anywhere in the file (build.yml:1-8)
```

## KG-01.4 What each path is (file-count summary, verified 2026-07-25)

```mermaid
flowchart TB
    P["pipeline/ — 13 tracked files<br/>Python 3 / rdflib, wholly original<br/>LICENSING.md:39-42"]
    S["static/ — ns/v2.jsonld only<br/>3,371 bytes, original work<br/>LICENSING.md:43-44"]
    E["explorer/ — WasmVOWL derivative<br/>modern/ + docs/ + tests/ + scripts/<br/>+ .github/workflows/wasm-publish.yml<br/>LICENSING.md:45-48"]
    O["ontology/pages/ — 8,138 pages<br/>machine-generated, ODbL-1.0"]
    D["dist/ — 7,874-scale build artefacts<br/>ODbL-1.0, dist/data committed<br/>dist/api gitignored"]
    P --> O
    E --> O
    O --> D
    note1["INVARIANT: dist-ci/ (this repo's CI build target) is never committed —<br/>it exists only inside a GitHub Actions run (LICENSING.md:34-36)"]
```
