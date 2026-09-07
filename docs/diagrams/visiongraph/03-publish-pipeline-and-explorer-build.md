---
id: VG-03
title: publish.yml — the actual publisher, quality gates, and the React/vowl-wasm explorer build
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: []
sources:
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/pipeline/build.py
  - ../visionGraph/pipeline/jsonld_parser.py
verified_commit: 9e308164c
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
    FIN["Finalize — CNAME, markdown mirror, contract gate<br/>publish.yml:200-255 — see VG-03.4"]
    SMOKE["Explorer smoke — Playwright/CDP against built www/<br/>publish.yml:257-265"]
    NOTES["Preserve existing /notes directory<br/>clone gh-pages, copy notes/ if present<br/>publish.yml:267-277"]
    DEPLOY["Deploy — peaceiris/actions-gh-pages@v3<br/>external_repository: DreamLab-AI/knowledgeGraph<br/>publish.yml:279-286"]
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

## VG-03.4 Markdown mirror contract gate — title-form, namespace-recursive, alias-doubled

```mermaid
flowchart TB
    WALK["find knowledge/pages -name *.md<br/>-not -path .* -not -path _misc/*<br/>publish.yml:238"]
    CHECK["grep -qE vc:public:#91;#91;:space:#93;#93;*true<br/>tolerant of pretty AND compact JSON-LD<br/>publish.yml:224"]
    NAME["mirror name: rel path with / → ___<br/>PLUS a %2F-encoded alias when they differ<br/>publish.yml:230-235"]
    RECURSE["recursion added because Obsidian namespace pages<br/>now live at pages/&lt;Ns&gt;/&lt;Title&gt;.md — the OLD flat<br/>glob stopped seeing 13 pages: ETSI_Domain_*, A/B<br/>Testing, TCP/IP, ISO/IEC 9075 — publish.yml:211-215"]
    GATE["contract: copied count == parser#39;s is_public count<br/>#40;PAGES copied, not files — the %2F aliases can#39;t mask drift#41;<br/>publish.yml:242-251"]
    WALK --> CHECK --> NAME --> GATE
    RECURSE -.-> WALK
    note1["INVARIANT: the SAME lesson as knowledgeGraph#39;s markdown-mirror<br/>incident #40;see KG-05.4#41; — a space-bearing literal match once<br/>silently dropped 653 compact pages #40;publish.yml:219-223#41;"]
```
