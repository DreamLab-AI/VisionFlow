---
id: VG-03
title: publish.yml — the actual publisher, quality gates, and the React/vowl-wasm explorer build
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: []
sources:
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/pipeline/public_projection.py
  - ../visionGraph/pipeline/patch_notes_export.py
  - ../visionGraph/pipeline/build.py
  - ../visionGraph/pipeline/jsonld_parser.py
verified_commit: 482ba593bcffddca7d564d0b6c4b1fea225154fb
---

## VG-03.1 publish.yml — checkout to deploy, one job, no separate release gate

```mermaid
flowchart TD
    CO["Checkout + setup Python 3.12 + Node 20<br/>publish.yml:33-49"]
    HON["Honesty gate — no unevidenced 'production ready'<br/>publish.yml:54-70, scoped to publishing-tools/WasmVOWL"]
    PYT["Pipeline unit tests — pytest pipeline/tests -q<br/>publish.yml:72-76"]
    RM["rm -rf www — deletes stale tracked build FIRST<br/>publish.yml:93 — see VG-03.2 for why"]
    BUILD["python -m pipeline.build knowledge/pages www<br/>publish.yml:95-97"]
    VAL["Validate — python -m pipeline.validate --json<br/>NON-BLOCKING — publish.yml:107-117"]
    CONF["Semantic conflict gate — pipeline.conflicts --severity high<br/>HARD-FAILS on high — publish.yml:119-126"]
    IRI["IRI-integrity gate — pipeline.iri_integrity<br/>baseline-aware, publish.yml:128-137"]
    SPA["React SPA build — publishing-tools/WasmVOWL/modern<br/>publish.yml:149-195 — see VG-03.3"]
    FIN["Finalize — CNAME and generated public markdown — see VG-03.4"]
    SMOKE["Explorer smoke — Playwright/CDP against built www/<br/>publish.yml:217-225"]
    NOTES["Preserve existing /notes directory<br/>clone gh-pages, copy notes/ if present<br/>pipeline/patch_notes_export.py:9"]
    DEPLOY["Deploy — peaceiris/actions-gh-pages@v3<br/>external_repository: DreamLab-AI/knowledgeGraph<br/>publish.yml:240-247"]
    CO --> HON --> PYT --> RM --> BUILD --> VAL --> CONF --> IRI --> SPA --> FIN --> SMOKE --> NOTES --> DEPLOY
    note1["INVARIANT: concurrency group deploy-ontology is SEPARATE from the<br/>notes SPA — must never queue-starve or cancel it — 'different<br/>knowledge bases, don#39;t cross the streams' publish.yml:15-20"]
```

## VG-03.2 The rm -rf www incident — why the build directory is deleted before every run

```mermaid
flowchart LR
    TRACKED["www/ is TRACKED in git — checkout arrives<br/>carrying a STALE build #40;6,144 md files,<br/>7,721 page JSONs from last commit#41;"]
    OVERWRITE["pipeline OVERWRITES what it regenerates<br/>but NEVER DELETES stale files"]
    LEAK["anything dropped from the corpus SURVIVED<br/>in the checkout and got REPUBLISHED —<br/>deployed api/pages held 7,721 files against<br/>a 7,457-page build"]
    MIRROR["markdown mirror accumulated TWO naming<br/>schemes across ~14.5k files"]
    TRACKED --> OVERWRITE --> LEAK
    OVERWRITE --> MIRROR
    FIX["rm -rf www BEFORE the build — publish.yml:93<br/>deploy now reflects the build and nothing else"]
    LEAK --> FIX
    MIRROR --> FIX
    note1["DIVERGENCE #40;acknowledged, not closed#41;: 'the real fix is to stop<br/>tracking www/; this makes the build correct in the meantime'<br/>#40;publish.yml:91-92, comment verbatim#41;"]
```

## VG-03.3 React SPA build — vowl-wasm consumed as a pinned, published package

```mermaid
sequenceDiagram
    autonumber
    participant JOB as publish.yml SPA step<br/>publish.yml:149
    participant STAGE as public/data + public/api<br/>staged for local vite preview
    participant NPM as npm ci --no-audit --no-fund<br/>publish.yml:169
    participant VW as @dreamlab-ai/vowl-wasm<br/>EXTERNAL — see VW-*
    participant TSC as npx tsc -b + npm run test<br/>publish.yml:173-175
    participant VITE as npm run build<br/>publish.yml:178

    JOB->>STAGE: copy ontology.json, ontology.ttl, graph/*, search-index.json<br/>from ../../../www/data — NEVER overwrites www/data itself
    JOB->>NPM: install dependencies incl. @dreamlab-ai/vowl-wasm
    NPM->>VW: pinned exactly in package.json,<br/>integrity-locked in package-lock.json
    Note over VW: ADR-NG-001 §3: physics worker imports the wasm-bindgen --target web<br/>glue and drives NggExplorer. The engine is NOT vendored here — this<br/>workflow needs NO Rust/wasm toolchain at all #40;publish.yml:162-168#41;
    JOB->>TSC: type-check + unit tests — CAPABILITIES.md cites these as evidence
    JOB->>VITE: NODE_OPTIONS=--max-old-space-size=4096
    VITE-->>JOB: dist/ — index.html, 404.html, assets/, coi-serviceworker.js
    Note over JOB: copies SPA assets but NOT api/ or data/ — the pipeline already<br/>wrote authoritative data/api to www/ #40;publish.yml:180-181#41;
```

## VG-03.4 Public markdown and preserved notes have separate boundaries

```mermaid
flowchart TB
    SOURCE["Recursive Page parser and typed input census"] --> PROJECT["Public graph and prose projection"]
    PROJECT --> MIRROR["emit_public_markdown from projected fields; no raw-source copy"]
    MIRROR --> NAME["Flat namespace aliases: slash becomes ___ and %2F"]
    NAME --> STAGED["Generated alongside other public artefacts in staging"]
    ARCHIVE["Existing gh-pages notes directory"] --> PRESERVE["Preserve historical SPA"]
    PRESERVE --> PATCH["patch_notes_export adds scoped mobile table stylesheet"]
    STAGED --> DEPLOY["Publication workflow"]
    PATCH --> DEPLOY
    PATCH --> LIMIT["Compatibility override is not a notes SPA rebuild"]
```

Execution qualification, 2026-09-07: the raw-copy mirror loop has been removed from `publish.yml`; `public_projection.py` emits both title-form aliases from sanitised public data. `patch_notes_export.py` fails on a missing/malformed preserved index and inserts its stylesheet once. Browser candidate verification and any publication are recorded separately in the [execution receipt](../../estate-review/closeout/2026-09-07-execution-federation.md).
