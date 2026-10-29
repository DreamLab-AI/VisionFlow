---
id: KG-05
title: CI/CD and publish — build.yml archived, visionGraph's ontomark build is the sole producer
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/ci-cd/build-and-gates.md
adrs: [ADR-2003, ADR-2013]
sources:
  - ../knowledgeGraph/README.md
  - ../knowledgeGraph/docs/ci-cd/build-and-gates.md
  - ../knowledgeGraph/archive/github-workflows/build.yml
  - ../visionGraph/.github/workflows/publish.yml
  - docs/adr/ADR-2013-sovereign-corpus-is-ecosystem-canon.md
verified_commit: {knowledgegraph: 1edcfc93de90ccff564073ef82cb198c422f75e3, visiongraph: 59e7dcc8d05ca447d03c74c819a740b390b93074, visionflow: 1d7183a73c1eae4996a7006067d5047308fab7a5}
---

## KG-05.3 Publication topology — this repo is a pure published export, visionGraph's publish.yml is the sole builder

```mermaid
sequenceDiagram
    autonumber
    participant VG as visionGraph publish.yml<br/>EXTERNAL: see VG-03.1
    participant OM as ontomark CLI (Rust)<br/>EXTERNAL: ontomark release v0.6.0
    participant CERT as oo-cert (Lean 4 checker)<br/>EXTERNAL: open-ontologies v2.0.1
    participant PAGES as gh-pages branch<br/>DreamLab-AI/knowledgeGraph
    participant SITE as narrativegoldmine.com
    participant THIS as this repo, at rest

    Note over THIS: this repo is the PUBLISHED EXPORT only — authoring and the active<br/>publication build live upstream (knowledgeGraph/README.md:13)
    VG->>OM: fetch the released binary, pinned by tag and by sha256,<br/>no build-from-source fallback<br/>(visionGraph/.github/workflows/publish.yml:46-51,98-117)
    VG->>OM: ontomark validate --vault all, then ontomark build --vault all<br/>--out site-data --publish-out quartz/content --with-markdown-mirror<br/>--certificate-out (visionGraph/.github/workflows/publish.yml:127-135,165-171)
    OM-->>VG: api/ + data/ + context/ + okf/ (contract C3) plus the Quartz staging tree<br/>ontomark is the ONLY producer (visionGraph/.github/workflows/publish.yml:91-97,<br/>knowledgeGraph/README.md:56)
    VG->>CERT: re-derive every inferred triple from its premises,<br/>the build is refused if the certificate does not cover the closure<br/>(visionGraph/.github/workflows/publish.yml:158-163,210-227)
    VG->>PAGES: peaceiris/actions-gh-pages, external_repository=knowledgeGraph<br/>(visionGraph/.github/workflows/publish.yml:515-523, EXTERNAL: see VG-03.1)
    PAGES->>SITE: GitHub Pages serves gh-pages at narrativegoldmine.com<br/>(this repo's root CNAME is read by Pages, not written by any workflow here)
    Note over VG,SITE: the same run gates the bundle on the space and Earth and the supply chains<br/>domain contracts before Quartz (visionGraph/.github/workflows/publish.yml:309-313) and,<br/>after the deploy, dispatches corpus-sync to VisionClaw with the source sha<br/>(visionGraph/.github/workflows/publish.yml:525-531) — neither step touches this repo,<br/>see VG-03.1 and VG-04.1
    Note over THIS: this repo's own build.yml (Python pipeline, ontology/pages) is<br/>ARCHIVED and inert under archive/github-workflows/build.yml —<br/>retired because ontology/ itself moved to archive/ first<br/>(knowledgeGraph/README.md:58)
```
- **DIVERGENCE:** the governing doc (`build-and-gates.md:1-3`) still describes `build.yml` as this repo's active "reproducible half" running six gates; the workflow it names has been moved to `archive/github-workflows/` and no longer runs — the doc has not been updated to match the retirement.
- **Tension (ADR-2013 vs publish.yml):** ADR-2013 names `VisionClaw/crates/vault` as the only implementation that parses, reasons over or builds the corpus (`docs/adr/ADR-2013-sovereign-corpus-is-ecosystem-canon.md:52-53`), while the sole producer in the workflow is now the released `ontomark` CLI, which replaced the `vault` build with the same verbs and outputs (`../visionGraph/.github/workflows/publish.yml:91-97`); the ADR is `accepted`, `partial`, `staged` and its review trigger names a second build.

## KG-05.5 Sovereignty transition — from active builder to published export

```mermaid
flowchart TB
    subgraph BEFORE["Before — this repo authored and built"]
        OLDONT["ontology/pages/ — 8,138 Logseq .md<br/>ARCHIVED, kept for history only — knowledgeGraph/README.md:64"]
        OLDPIPE["pipeline/ — seven-stage rdflib build<br/>ARCHIVED — knowledgeGraph/README.md:58"]
        OLDCI["build.yml — six gates, no deploy<br/>ARCHIVED — archive/github-workflows/build.yml"]
        OLDONT --> OLDPIPE --> OLDCI
    end
    subgraph AFTER["After — visionGraph authors and ontomark builds, this repo only serves"]
        VGVAULT["visionGraph — Obsidian vault, typed frontmatter<br/>knowledgeGraph/README.md:46"]
        RUSTVAULT["ontomark — single Rust CLI, pinned release<br/>validates, reasons, builds<br/>visionGraph/.github/workflows/publish.yml:91-97"]
        OKF["machine artefacts + Quartz notes under /notes/<br/>visionGraph/.github/workflows/publish.yml:3-12"]
        VGVAULT --> RUSTVAULT --> OKF
    end
    OKF -->|"peaceiris/actions-gh-pages deploy<br/>visionGraph/.github/workflows/publish.yml:519"| GHPAGES["this repo's gh-pages branch"]
    GHPAGES --> SITE2["narrativegoldmine.com"]
    note1["INVARIANT: one vault, one build — the sole producer is the<br/>pinned ontomark release, a run that cannot fetch it fails<br/>visionGraph/.github/workflows/publish.yml:46-47,100-105"]
    note2["The run is triggered by a push to main or by hand; the owner authorised<br/>estate deployment on 2026-10-11 —<br/>visionGraph/.github/workflows/publish.yml:17-18 (EXTERNAL: see VG-03.1)"]
```
- This repo's published artefact is served from `gh-pages`; nothing here computes it any more, and the committed `dist/data/` snapshot is the 7 August 2026 generation, not the served one (`../knowledgeGraph/README.md:87`).
- **Drift:** this topic's earlier text read "Agents never push to visionGraph main — publish.yml runs only on the owner's push or by hand"; the workflow header now says it runs on a push to `main` or by hand, with estate deployment authorised during the ontomark resumption (`../visionGraph/.github/workflows/publish.yml:17-18`).

Audit qualification — 2026-09-07: earlier evidence in this topic predates the 2026-09-23 retirement of `build.yml` and no longer describes the live system; superseded by the sovereignty transition above. See the [federation audit](../../estate-review/2026-09-07-federation-audit.md) for the historical record.
