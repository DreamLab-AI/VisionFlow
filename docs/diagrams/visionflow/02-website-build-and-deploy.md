---
id: VF-02
title: Website build and deploy — copy-only build, asset inventory, gated Pages publication
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
  - docs/site-verification.md
  - docs/PRD-website.md
adrs: [ADR-2002, ADR-2003, ADR-2004, ADR-2005]
sources:
  - website/build.sh
  - website/assets.manifest.json
  - website/build-receipt.json
  - website/static/index.html
  - website/static/js/main.js
  - website/static/js/mesh-webgl.js
  - website/static/css/styles.css
  - website/static/data/estate-health.json
  - scripts/website-assets.mjs
  - scripts/website-browser-check.mjs
  - scripts/check-cdp-sidecar.mjs
  - scripts/dream-build-check.sh
  - scripts/dream-link-check.sh
  - scripts/dream-meta-tags-scan.sh
  - scripts/dream-structured-data-scan.sh
  - scripts/check-diagram-text.js
  - .github/workflows/deploy.yml
  - package.json
  - playwright.config.js
  - tests/site.spec.js
  - tests/fixtures/cdp.js
  - tests/gates/run-all.sh
  - tests/gates/website-assets.test.sh
  - docs/BASELINE-visionflow.md
  - docs/site-verification.md
  - docs/PRD-website.md
  - docs/adr/ADR-2002-static-copy-only-website.md
  - docs/adr/ADR-2003-pages-artifact-deploy.md
verified_commit: bec06dc3a
---

## VF-02.1 The copy-only build — every step of website/build.sh
```mermaid
sequenceDiagram
    autonumber
    participant OP as "Operator or CI"
    participant B as "website/build.sh"
    participant FS as "website/dist/"
    participant WA as "scripts/website-assets.mjs"
    participant RC as "website/build-receipt.json"

    OP->>B: "cd website, then run ./build.sh"
    Note over B: "set -euo pipefail, then cd to the script's own dir<br/>build.sh:2"
    B->>FS: "rm -rf dist then mkdir -p dist — build.sh:17"
    B->>FS: "cp -r static/* dist/ — build.sh:21"
    Note over B,FS: "INVARIANT: static/* deliberately does not glob dotfiles,<br/>and dist/.claude-flow is removed outright, so tool state<br/>never reaches the published artefact — build.sh:24"
    B->>FS: "echo www.visionflow.info into dist/CNAME — build.sh:27"
    B->>WA: "node ../scripts/website-assets.mjs stage — build.sh:30"
    WA-->>FS: "derived plus optional copies, staging sidecar written"
    B->>WA: "node ../scripts/website-assets.mjs verify — build.sh:33"
    WA-->>RC: "receipt written OUTSIDE dist/ so its own hash stays stable"
    B->>OP: "ls -la dist/ then the sentinel BUILD-COMPLETE bytes N — build.sh:40"
    Note over OP,B: "INVARIANT: no compile step, no bundler, no WASM — build.sh:7<br/>ADR-2002-static-copy-only-website.md:30"
```

## VF-02.2 What is in the artefact, and what compiles it — nothing
```mermaid
flowchart LR
    classDef src fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef out fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef ext fill:#f6efd8,stroke:#8a7020,color:#111

    subgraph STATIC["website/static/ — hand-written source"]
        IDX["index.html — one page, ~97 KB<br/>one module entry at index.html:1157<br/>one local stylesheet at index.html:23"]:::src
        CSS["css/styles.css — the sole stylesheet"]:::src
        MAIN["js/main.js — sole ES module entrypoint<br/>DOMContentLoaded wiring at main.js:844"]:::src
        MESH["js/mesh-webgl.js — hand-written WebGL2 ES module<br/>initMesh returns null without WebGL2<br/>mesh-webgl.js:15"]:::src
        DATA["data/estate-health.json — committed nightly snapshot<br/>estate-health.json:2 schema visionflow.estate-health/1<br/>see VF-03"]:::src
        MEDIA["img/showcase/*.webp and video/*.mp4"]:::src
    end

    subgraph DIST["website/dist/ — the published artefact"]
        D1["dist equals static/ minus dotfiles,<br/>plus CNAME, plus staged repo images"]:::out
        D2["51 files, 60,241,361 bytes at this build<br/>build-receipt.json:18"]:::out
    end

    STATIC ==>|"cp -r, no transform — build.sh:21"| DIST
    ASSETS["repo assets/diagrams, assets/generated,<br/>assets/heroes, assets/screenshots<br/>staged by the manifest, not by static/*"]:::src
    ASSETS ==>|"website-assets.mjs stage"| DIST

    FONTS["EXTERNAL fetch: Google Fonts only<br/>Inter and JetBrains Mono, index.html:22<br/>preconnect at index.html:20"]:::ext
    IDX -.-> FONTS

    NEG["ABSENT BY DECISION: no Cargo.toml, no rs, no wasm,<br/>no bundler, no framework, no CDN CSS<br/>ADR-2002-static-copy-only-website.md:53"]
    DIST -.-> NEG
```

## VF-02.3 The asset inventory — three classes, three failure semantics
```mermaid
flowchart TB
    classDef req fill:#f7dede,stroke:#a33333,color:#111
    classDef der fill:#f9f0d5,stroke:#8a7020,color:#111
    classDef opt fill:#e4ecf8,stroke:#3a5a8a,color:#111

    M["website/assets.manifest.json<br/>manifest_version 1, site_root static, output dist<br/>assets.manifest.json:4"]

    M --> REQ["required — 14 entries<br/>assets.manifest.json:6<br/>index.html, CNAME, css/styles.css, js/main.js,<br/>js/mesh-webgl.js, data/estate-health.json,<br/>img/og-card.png, five showcase webp, two mp4"]:::req
    M --> DER["derived — required destinations sourced from OUTSIDE static/<br/>assets.manifest.json:58 img/og-card.png<br/>from ../assets/screenshots/visionflow-final.png"]:::der
    M --> OPT["optional — four directory groups<br/>repo-diagrams, repo-generated, repo-heroes, repo-screenshots<br/>assets.manifest.json:22"]:::opt

    REQ -->|"absent or zero bytes in dist/"| F1["verify exits 1, naming the file and its 'why'<br/>website-assets.mjs:250"]:::req
    DER -->|"source file missing"| F2["stage exits 1 — a derived entry backs a REQUIRED dest<br/>website-assets.mjs:129"]:::der
    OPT -->|"source directory absent"| F3["recorded as status source-absent, exit 0<br/>website-assets.mjs:147"]:::opt

    HIST["The defect this closes: build.sh staged repo images with<br/>cp -r ... 2 dev null or true, so every optional copy could<br/>fail while the build still printed BUILD-COMPLETE<br/>website-assets.mjs:9"]
    M -.-> HIST

    LINK["data/estate-health.json is REQUIRED specifically because<br/>the estate section links it by href, so the link-integrity<br/>gate must resolve it in dist/ — assets.manifest.json:12"]:::req
    REQ -.-> LINK
```

## VF-02.4 website-assets.mjs — stage, verify, and the receipt handoff
```mermaid
sequenceDiagram
    autonumber
    participant CLI as "node scripts/website-assets.mjs"
    participant MAN as "website/assets.manifest.json"
    participant D as "website/dist/"
    participant SC as "website/.staging-report.json"
    participant R as "website/build-receipt.json"

    Note over CLI: "usage guard: the only accepted commands are stage and verify<br/>website-assets.mjs:64"
    rect rgb(226,236,248)
        CLI->>MAN: "read and JSON.parse, then DIST is website plus manifest.output"
        Note over CLI,MAN: website-assets.mjs:78
    end
    alt command is stage
        CLI->>D: "dist must already exist, else exit 1 — website-assets.mjs:278"
        CLI->>D: "copy each derived entry and hash it — website-assets.mjs:136"
        CLI->>D: "copy each optional group by its include glob — website-assets.mjs:159"
        CLI->>SC: "writeStagingSidecar(report) — website-assets.mjs:268"
    else command is verify
        CLI->>D: "assert every required dest exists and is non-empty — website-assets.mjs:177"
        CLI->>D: "walk dist/, total bytes, per-file sha256"
        Note over CLI,D: "tree_sha256 is a sha256 over the sorted 'sha256 path' lines<br/>two builds agree iff they published byte-identical trees<br/>website-assets.mjs:197"
        CLI->>SC: "readStagingSidecar() folded into the receipt — website-assets.mjs:271"
        CLI->>R: "write the receipt, outside dist/ — website-assets.mjs:242"
        alt any required asset missing
            CLI-->>CLI: "print each dest, reason and why, then exit 1 — website-assets.mjs:252"
        else all present
            CLI-->>CLI: "print ASSET-INVENTORY-OK — website-assets.mjs:260"
        end
    end
```

## VF-02.5 The build receipt — what a published page can be traced back to
```mermaid
erDiagram
    BUILD_RECEIPT {
        int receipt_version "1 — build-receipt.json:2"
        string generated_at "ISO to seconds — website-assets.mjs:204"
        object manifest "path, sha256 of the manifest bytes, required_count, optional_count — build-receipt.json:4"
        object revision "source is local git HEAD; published is GITHUB_SHA or --published-revision; workflow_run; dirty — website-assets.mjs:211"
        object dist "path, file_count, bytes, tree_sha256 — build-receipt.json:18"
        list required_assets "dest plus bytes plus sha256 per required entry — website-assets.mjs:188"
        list missing_required "dest plus why plus reason absent or empty — website-assets.mjs:180"
        object staging "the stage-step sidecar, derived and optional outcomes"
        bool ok "true when missing_required is empty — website-assets.mjs:236"
        object gates "per-gate verdict, written by the deploy workflow — deploy.yml:151"
        object published "target, revision, workflow_run, ref — deploy.yml:160"
    }
    REQUIRED_ASSET {
        string dest "dist-relative path"
        int bytes "non-zero, else the gate fails"
        string sha256 "per-file digest"
    }
    ARTEFACT {
        string placement "the receipt lives OUTSIDE dist/ so it never becomes part of the tree it hashes — website-assets.mjs:36"
        string upload "uploaded as its own workflow artefact named website-build-receipt — deploy.yml:173"
    }
    BUILD_RECEIPT ||--o{ REQUIRED_ASSET : lists
    BUILD_RECEIPT ||--|| ARTEFACT : describes
```

## VF-02.6 deploy.yml — the publication gate graph
```mermaid
flowchart TB
    classDef block fill:#f7dede,stroke:#a33333,color:#111
    classDef report fill:#f9f0d5,stroke:#8a7020,color:#111
    classDef ok fill:#e6f0dc,stroke:#4a7a2a,color:#111

    TRIG["push to main, or workflow_dispatch — deploy.yml:34<br/>concurrency group 'pages', cancel-in-progress false<br/>deploy.yml:44"]
    TRIG --> J1["job build on ubuntu-latest<br/>permissions contents read, pages write, id-token write<br/>deploy.yml:39"]

    J1 --> CO["actions/checkout@v4 plus setup-node 22"]
    CO --> AB["OPTIONAL checkout of DreamLab-AI/agentbox at the pinned ref<br/>continue-on-error true — deploy.yml:64"]:::report
    AB --> BUILD["run website/build.sh — deploy.yml:78"]

    BUILD --> G1["BLOCKING 1 asset inventory<br/>website-assets.mjs verify --published-revision GITHUB_SHA<br/>deploy.yml:83"]:::block
    G1 --> G2["BLOCKING 2 build output, grep BUILD-OK<br/>deploy.yml:92"]:::block
    G2 --> G3["BLOCKING 3 internal link integrity, grep LINK-INTEGRITY-OK<br/>deploy.yml:100"]:::block
    G3 --> G4["BLOCKING 4 meta tags, grep META-SCAN-OK<br/>deploy.yml:109"]:::block
    G4 --> G5["BLOCKING 5 structured data, grep SD-SCAN-OK<br/>deploy.yml:118"]:::block
    G5 --> G6["BLOCKING 6 committed diagram baseline text visibility<br/>node scripts/check-diagram-text.js — deploy.yml:125"]:::block
    G6 --> G7["REPORTED 7 self-description drift counter<br/>blocks only when the agentbox checkout landed<br/>deploy.yml:134"]:::report

    G7 --> REC["write gate verdicts and the published block into the receipt<br/>deploy.yml:147"]:::ok
    REC --> UPR["upload-artifact website-build-receipt — deploy.yml:172"]:::ok
    UPR --> UP["upload-pages-artifact@v3, path website/dist<br/>reached only when every blocking gate passed<br/>deploy.yml:180"]:::ok
    UP --> J2["job deploy, needs build<br/>environment github-pages, deploy-pages@v4<br/>deploy.yml:193"]:::ok

    INV["INVARIANT: no gh-pages branch push anywhere;<br/>the CNAME is emitted by build.sh into the artefact<br/>ADR-2003-pages-artifact-deploy.md:29<br/>BASELINE-visionflow.md:225"]
    J2 -.-> INV
```

## VF-02.7 Why every gate greps a sentinel instead of trusting the exit code
```mermaid
sequenceDiagram
    autonumber
    participant W as "deploy.yml step"
    participant S as "scripts/dream-link-check.sh"
    participant SH as "bash"

    W->>S: "capture stdout of bash scripts/dream-link-check.sh"
    S->>SH: "cd website/dist, or echo NO-DIST and exit 0<br/>dream-link-check.sh:20"
    Note over S: "the evaluators exit 0 even on failure, because the dream<br/>annexe reads a non-zero exit as an infrastructure fault<br/>rather than as a finding"
    S-->>W: "stdout ends LINK-INTEGRITY-OK or LINK-INTEGRITY-FAIL<br/>dream-link-check.sh:63"
    W->>W: "grep -q for the OK sentinel, else exit 1"
    Note over W: deploy.yml:100
    alt sentinel absent
        W-->>W: "::error:: internal links or assets are missing from dist/"
    else sentinel present
        W-->>W: continue to the next gate
    end
    Note over W,S: "INVARIANT: the exit code is NOT the contract — the last<br/>stdout sentinel is. Without the grep the first five gates<br/>would be decorative — ADR-2003-pages-artifact-deploy.md:85"
```

## VF-02.8 The drift counter's two modes — reported vs enforced
```mermaid
stateDiagram-v2
    direction LR
    [*] --> Attempt
    state "checkout DreamLab-AI/agentbox at the pinned ref into _agentbox" as Attempt
    Attempt --> Enforced : steps.agentbox.outcome is success
    Attempt --> Reported : checkout failed, no credentials on the runner
    state "ENFORCED — drift-counter.mjs runs and a mismatch fails the job" as Enforced
    state "REPORTED — drift-counter.mjs runs tolerantly, verdict is reported" as Reported
    Enforced --> Receipt : verdict enforced-pass
    Reported --> Receipt : warning annotation, the job continues
    state "gates.drift_counter recorded in build-receipt.json" as Receipt
    Receipt --> [*]
    note right of Attempt
      deploy.yml:62 — the ref is pinned to the same
      revision as the drift-counter allowlist source_pin
    end note
    note right of Reported
      deploy.yml:135 — the partial-source failure mode:
      an axis whose truth lives in a sibling repo cannot
      be evaluated without that checkout, so it is not
      allowed to block. DIVERGENCE: on a runner without
      the checkout this axis is never actually measured
      on the publication path, yet the page still ships.
    end note
```

## VF-02.9 The local verify chain — npm scripts and what each one proves
```mermaid
flowchart LR
    classDef step fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef needs fill:#f9f0d5,stroke:#8a7020,color:#111

    V["npm run verify — package.json:17"]
    V --> S1["npm run build<br/>cd website and run build.sh — package.json:7"]:::step
    S1 --> S2["npm run check:assets<br/>website-assets.mjs verify — package.json:9"]:::step
    S2 --> S3["npm run test:gates<br/>bash tests/gates/run-all.sh — package.json:15"]:::step
    S3 --> S4["npm run check:sidecar<br/>node scripts/check-cdp-sidecar.mjs — package.json:8"]:::needs
    S4 --> S5["npm run test:site<br/>playwright test — package.json:12"]:::needs

    S3 --> SUITES["four suites, each driving a deliberate defect through its<br/>gate and asserting the gate goes red — harness-audit,<br/>drift-counter, release-manifest, website-assets<br/>run-all.sh:18, sentinel GATE-TESTS-OK at run-all.sh:34"]:::step

    SIDE["EXTERNAL, not in this repo: the browsercontainer Chrome sidecar<br/>in-network port 9223, from the host port 9222<br/>check-cdp-sidecar.mjs:15, probes /json/version at :31<br/>see ES-01 for the sidecar's place in the estate"]:::needs
    S4 --- SIDE

    OTHER["npm run test:a11y and test:perf are grep-selected subsets<br/>of the same spec — package.json:13<br/>npm run health:collect and health:check — package.json:10, see VF-03"]:::step
    V -.-> OTHER
```

## VF-02.10 Browser verification topology — what needs the sidecar and why
```mermaid
sequenceDiagram
    autonumber
    participant T as "playwright test"
    participant C as "playwright.config.js"
    participant WS as "python3 http.server on 4173 over website/dist"
    participant FX as "tests/fixtures/cdp.js"
    participant CH as "EXTERNAL browsercontainer Chrome — see ES-01"
    participant P as "the built page"

    C->>WS: "webServer command, url is loopback port 4173<br/>playwright.config.js:36"
    Note over C: "baseURL is SITE_BASE_URL, else the container's own<br/>172.20.x address when running inside Docker<br/>playwright.config.js:23 and playwright.config.js:20"
    T->>FX: "the page fixture is overridden — fixtures/cdp.js:74"
    FX->>CH: "resolve browsercontainer to an IP, GET /json/version,<br/>then chromium.connectOverCDP — fixtures/cdp.js:76"
    Note over FX,CH: "the sidecar rejects a non-localhost Host header on the<br/>WebSocket upgrade, so the hostname is resolved first<br/>fixtures/cdp.js:25"
    CH->>WS: navigate to baseURL
    WS-->>P: serve website/dist
    P-->>T: "structure assertions: ten named sections, the mesh canvas,<br/>four particle canvases, zero console errors — site.spec.js:31"
    P-->>T: "@a11y — axe-core analyze, violations must be empty<br/>site.spec.js:85"
    P-->>T: "@perf — total transferSize at most 800 KB<br/>site.spec.js:108"
    Note over T,CH: "INVARIANT: no local Chromium is installed in CI — both<br/>projects, chromium and mobile-chrome, drive the sidecar<br/>BASELINE-visionflow.md:222"
```

## VF-02.11 The deeper browser receipt — three scenarios over raw CDP
```mermaid
flowchart TB
    classDef sc fill:#e4ecf8,stroke:#3a5a8a,color:#111
    BC["scripts/website-browser-check.mjs<br/>publication-candidate verification over raw CDP<br/>website-browser-check.mjs:6<br/>npm run test:browser — package.json:16"]

    BC --> S1["baseline — WebGL2 available, motion normal;<br/>expect initMesh to return a controller and a rAF loop to run<br/>website-browser-check.mjs:16"]:::sc
    BC --> S2["reduced-motion — prefers-reduced-motion reduce;<br/>expect one settled frame and no sustained rAF<br/>website-browser-check.mjs:18"]:::sc
    BC --> S3["no-webgl — the webgl2 context forced to null before any<br/>page script runs; expect initMesh null, the page still<br/>renders, nothing throws — website-browser-check.mjs:20"]:::sc

    S1 --> PROOF["paint is proven by comparing canvas-region screenshot BYTES<br/>across scenarios, not by readPixels — a canvas without<br/>preserveDrawingBuffer returns a cleared buffer<br/>website-browser-check.mjs:24"]
    S2 --> PROOF
    S3 --> PROOF

    CDPH["CDP host resolution: the sidecar host is resolved to an IP,<br/>because Chrome's DevTools endpoint rejects any Host header<br/>that is not localhost or a bare IP<br/>website-browser-check.mjs:54"]
    BC --- CDPH

    OUT["receipt and screenshots written under --out,<br/>default docs/estate-closeout/2026-09-05<br/>website-browser-check.mjs:48"]
    PROOF --> OUT

    DRIFT["DIVERGENCE: this script is run by hand, never by deploy.yml.<br/>The publication path has no browser gate at all — every<br/>blocking gate in VF-02.6 is browser-free by design."]
    OUT -.-> DRIFT
```

## VF-02.12 The page at runtime — one module, progressive enhancement, no framework
```mermaid
stateDiagram-v2
    direction TB
    [*] --> Parse
    state "browser parses index.html — one stylesheet, one module" as Parse
    Parse --> DOMReady
    state "DOMContentLoaded handler — main.js:844" as DOMReady
    DOMReady --> Nav
    DOMReady --> Sections
    DOMReady --> Estate
    DOMReady --> Mesh
    state "initNavScroll, initSmoothScroll, initScrollReveal" as Nav
    state "initFigures, initDoors, initReadingSwitch, initIndex, initProgress, initSheet" as Sections
    state "initEstateHealth, async, renders at rest without it — see VF-03" as Estate
    state "initMeshBackdrop" as Mesh
    Mesh --> WebGL2Yes : getContext webgl2 succeeded
    Mesh --> WebGL2No : getContext returned null
    state "initMesh returns null — the page renders every section regardless — mesh-webgl.js:15" as WebGL2No
    state "hero star-field plus scroll-driven ontology figures" as WebGL2Yes
    WebGL2Yes --> Reduced : prefers-reduced-motion reduce
    state "one settled frame, no rAF, no bounce — mesh-webgl.js:16" as Reduced
    Nav --> [*]
    Sections --> [*]
    Estate --> [*]
    note right of Estate
      The background video is skipped entirely under
      reduced motion — main.js:856
    end note
    note right of WebGL2No
      DOC-DRIFT: PRD-website.md:194 still requires a static
      SVG fallback under reduced motion, and PRD-website.md:136
      requires role img plus an aria-label on the canvas
      elements. What ships is a null return and a single
      settled WebGL2 frame.
    end note
```
