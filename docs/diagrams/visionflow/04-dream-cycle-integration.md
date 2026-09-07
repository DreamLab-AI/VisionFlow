---
id: VF-04
title: Dream-cycle integration — rotation slots, evaluator contracts, ledger and the human gate
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
  - docs/adr/README.md
adrs: [ADR-2008, ADR-2009]
sources:
  - dream.config.json
  - docs/dream-cycle/LEDGER.md
  - scripts/dream-build-check.sh
  - scripts/dream-link-check.sh
  - scripts/dream-meta-tags-scan.sh
  - scripts/dream-structured-data-scan.sh
  - scripts/dream-asset-refs-scan.sh
  - scripts/estate-health.mjs
  - scripts/website-assets.mjs
  - website/build.sh
  - website/static/js/mesh-webgl.js
  - website/static/index.html
  - website/assets.manifest.json
  - .github/workflows/deploy.yml
  - .github/workflows/estate-health.yml
  - docs/BASELINE-visionflow.md
  - docs/adr/README.md
  - docs/adr/ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md
  - docs/adr/ADR-2009-webgl-mesh-deep-is-sidecar-only.md
  - ./README.md
  - package.json
verified_commit: bec06dc3a
---

## VF-04.1 dream.config.json — the whole contract this repo offers the engine
```mermaid
flowchart LR
    classDef cfg fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef gate fill:#f9f0d5,stroke:#8a7020,color:#111

    C["dream.config.json"]
    C --> ID["repo DreamLab-AI/VisionFlow, cron 0 3 * * *<br/>dream.config.json:2 and dream.config.json:3"]:::cfg
    C --> SL["slots — 4 rotation surfaces, each a deep plus 2 scans<br/>dream.config.json:4"]:::cfg
    C --> BM["bonusModuli — night 30 copy-and-messaging-review,<br/>night 60 asset-weight-budget<br/>dream.config.json:34"]:::cfg
    C --> CP["controlPlaneProbes — node --version and a listing of<br/>website/static and website/dist<br/>dream.config.json:38"]:::cfg
    C --> BS["buildStep — cd website and run build.sh, tail 8 lines;<br/>degradeOnWasmFailure false because there is no WASM<br/>dream.config.json:43 and dream.config.json:44"]:::cfg
    C --> EE["evaluatorEntrypoints — 6 entries, 4 plain strings and<br/>2 objects carrying required and deeps<br/>dream.config.json:46"]:::gate
    C --> ED["extraDisciplines — 5 guardrails, prose, enforced by the<br/>night's own reasoning rather than by code<br/>dream.config.json:68"]:::gate
    C --> OUT["ledgerPath docs/dream-cycle/LEDGER.md,<br/>branchPrefix dream/, labels dream-cycle and marketing-site,<br/>autoMerge FALSE<br/>dream.config.json:75 and dream.config.json:81"]:::cfg
    C --> MISC["competitors empty, adrConvention 4-digit<br/>dream.config.json:66 and dream.config.json:67"]:::cfg

    ENG["EXTERNAL: the dream ENGINE is not in this repo. DreamLab's<br/>dream-engine, a tracking fork of ruvnet/dream-machine, runs the<br/>night and consumes this file — see AB-23 for the engine, its<br/>gates and its acceptance path. ./README.md:128"]
    C -.->|"read by"| ENG
```

## VF-04.2 The rotation — four slots, one parked, two bonus moduli
```mermaid
stateDiagram-v2
    direction LR
    [*] --> Night
    state "a night at 03:00 UTC draws the next slot — dream.config.json:3" as Night
    Night --> CI
    Night --> BP
    Night --> SEO
    Night --> EH
    state "content-integrity — scans internal-links, asset-refs — dream.config.json:6" as CI
    state "build-pipeline — scans static-build, cname-domain — dream.config.json:13" as BP
    state "seo-and-meta — scans meta-tags, structured-data — dream.config.json:20" as SEO
    state "estate-health — scans snapshot-freshness, persistent-reds — dream.config.json:27" as EH
    CI --> Bonus
    BP --> Bonus
    SEO --> Bonus
    EH --> Bonus
    state "bonus surface on night 30 and night 60 — dream.config.json:34" as Bonus
    Bonus --> [*]
    state "webgl-mesh — PARKED, sidecar-only, absent from slots" as Parked
    Parked --> Night : returns only when a browser runner exists, or CI collects mesh artefacts into the evidence pack
    note right of Parked
      Four nights (08-30, 09-02, 09-06, 09-07) produced zero
      mesh observations: the deep's subject is compiled and
      rendered behaviour, the annexe has no browser sidecar,
      and no evaluator reads mesh-webgl.js
      ADR-2009-webgl-mesh-deep-is-sidecar-only.md:36
      Verified by grep at ADR-2009-webgl-mesh-deep-is-sidecar-only.md:84
    end note
    note right of EH
      DOC-DRIFT: ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:44
      still calls estate-health "a fifth rotation slot".
      ADR-2009 removed webgl-mesh, so it is one of four —
      as BASELINE-visionflow.md:149 correctly records.
    end note
```

## VF-04.3 Evaluator entrypoints — which fire on every night, which are deep-scoped
```mermaid
flowchart TB
    classDef always fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef scoped fill:#f9f0d5,stroke:#8a7020,color:#111

    EE["evaluatorEntrypoints — dream.config.json:46"]

    EE --> A1["build-check — bash scripts/dream-build-check.sh<br/>dream.config.json:47"]:::always
    EE --> A2["links — bash scripts/dream-link-check.sh, tail 6<br/>dream.config.json:48"]:::always
    EE --> A3["meta-tags — bash scripts/dream-meta-tags-scan.sh<br/>dream.config.json:49"]:::always
    EE --> A4["structured-data — bash scripts/dream-structured-data-scan.sh<br/>dream.config.json:50"]:::always

    EE --> S1["asset-refs — object form: required false,<br/>deeps [content-integrity]<br/>dream.config.json:51"]:::scoped
    EE --> S2["estate-health — object form: node scripts/estate-health.mjs check,<br/>required false, deeps [estate-health]<br/>dream.config.json:58"]:::scoped

    S1 -.->|"fires only on the content-integrity deep"| D1["content-integrity night"]
    S2 -.->|"fires only on the estate-health deep"| D2["estate-health night"]

    WHY["Why asset-refs exists at all: the config had always listed<br/>asset-refs as a scan surface with no evaluator behind it, so<br/>2026-09-04 honestly recorded FALLBACK rather than inventing a<br/>measurement — dream-asset-refs-scan.sh:7"]:::scoped
    S1 --- WHY

    WHY2["Why estate-health is required false: declared REQUIRED by the<br/>09-06 add, it vetoed every ACCEPT while any sibling repo's CI<br/>was red — six reds unreachable by a website patch<br/>LEDGER.md:19"]:::scoped
    S2 --- WHY2

    SHELL["INVARIANT: every evaluator is a CHECKED-IN script invoked<br/>quote-free. The annexe ssh dispatch strips one level of nested<br/>double quotes, so inline double-quoted logic in the config was<br/>silently mangled — dream-build-check.sh:3, LEDGER.md:6"]
    EE -.-> SHELL
```

## VF-04.4 Each evaluator's pass and fail contract — the sentinel is the verdict
```mermaid
flowchart LR
    classDef pass fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef fail fill:#f7dede,stroke:#a33333,color:#111
    classDef odd fill:#f9f0d5,stroke:#8a7020,color:#111

    E1["dream-build-check.sh<br/>cd website/dist and test -f index.html<br/>dream-build-check.sh:6"]
    E1 --> E1P["prints page count and byte total, then BUILD-OK<br/>dream-build-check.sh:7"]:::pass
    E1 --> E1F["otherwise BUILD-FAIL — dream-build-check.sh:8<br/>exit status is 0 either way"]:::fail

    E2["dream-link-check.sh<br/>resolves href/src refs AND meta content media URLs<br/>dream-link-check.sh:46 and dream-link-check.sh:54"]
    E2 --> E2P["internal-refs-checked N, missing 0, then LINK-INTEGRITY-OK<br/>dream-link-check.sh:62"]:::pass
    E2 --> E2F["any missing ref prints MISSING and LINK-INTEGRITY-FAIL<br/>dream-link-check.sh:63"]:::fail
    E2 --> E2X["no dist/ prints NO-DIST and exits 0 with NO sentinel<br/>dream-link-check.sh:20"]:::odd

    E3["dream-meta-tags-scan.sh<br/>counts title, description, canonical, viewport as REQUIRED;<br/>og, twitter and robots counted but optional<br/>dream-meta-tags-scan.sh:24"]
    E3 --> E3P["required-missing 0, then META-SCAN-OK<br/>dream-meta-tags-scan.sh:33"]:::pass
    E3 --> E3F["MISSING-REQUIRED per tag, then META-SCAN-FAIL<br/>dream-meta-tags-scan.sh:19"]:::fail

    E4["dream-structured-data-scan.sh<br/>parses every application/ld+json block with node<br/>dream-structured-data-scan.sh:15"]
    E4 --> E4P["json-ld-blocks N, parse-errors 0, then SD-SCAN-OK —<br/>ZERO blocks is an honest measurement, not a failure<br/>dream-structured-data-scan.sh:32"]:::pass
    E4 --> E4F["unparseable JSON-LD prints PARSE-ERROR then SD-SCAN-FAIL<br/>dream-structured-data-scan.sh:27"]:::fail

    E5["dream-asset-refs-scan.sh<br/>reuses the build's own gate rather than a second<br/>drifting implementation — dream-asset-refs-scan.sh:30"]
    E5 --> E5P["inventory lines, orphan count and share, dist breakdown,<br/>then ASSET-SCAN-OK — dream-asset-refs-scan.sh:98"]:::pass
    E5 --> E5F["dist absent, or a required asset missing or empty:<br/>ASSET-SCAN-FAIL and a real exit 1<br/>dream-asset-refs-scan.sh:101"]:::fail
    E5 --> E5X["orphans are a MEASUREMENT, never a failure: optional groups<br/>are staged deliberately for deep links<br/>dream-asset-refs-scan.sh:22"]:::odd

    E6["estate-health.mjs check — offline read of the committed snapshot"]
    E6 --> E6P["ESTATE-HEALTH-OK, exit 0 — estate-health.mjs:869"]:::pass
    E6 --> E6F["ESTATE-HEALTH-RED or ESTATE-HEALTH-STALE, exit 1<br/>estate-health.mjs:858 and estate-health.mjs:865"]:::fail
```

## VF-04.5 A night end to end — probe, build, evaluate, ledger, branch, human gate
```mermaid
sequenceDiagram
    autonumber
    participant ENG as "EXTERNAL dream engine on the HP annexe — see AB-23"
    participant CFG as "dream.config.json"
    participant REPO as "the VisionFlow checkout"
    participant EV as "scripts/dream-*.sh and estate-health.mjs check"
    participant LED as "docs/dream-cycle/LEDGER.md"
    participant BR as "a dream/ branch"
    participant H as Human

    ENG->>CFG: read slots, evaluators, disciplines, ledger path
    ENG->>REPO: "controlPlaneProbes — node --version, list static and dist<br/>dream.config.json:39"
    ENG->>REPO: "buildStep — cd website and run build.sh, tail 8<br/>dream.config.json:43"
    Note over REPO: "the build emits BUILD-COMPLETE bytes N as its own<br/>terminal sentinel — build.sh:40"
    ENG->>ENG: "form ONE falsifiable hypothesis against tonight's deep"
    ENG->>EV: run the always-on evaluators plus any scoped to this deep
    EV-->>ENG: "stdout receipts — the last sentinel line is the verdict"
    ENG->>LED: "append one dated row — verdict ACCEPT, INCONCLUSIVE or OPERATOR<br/>ledgerPath at dream.config.json:75"
    alt a change is proposed
        ENG->>BR: "branch named with prefix dream/ — dream.config.json:76"
        ENG->>H: "open a DRAFT pull request, labels dream-cycle and marketing-site<br/>dream.config.json:77"
        Note over ENG,H: "INVARIANT: autoMerge is false — dream.config.json:81<br/>evaluation is not promotion — an agent proposes, a human signs<br/>./README.md:128"
        H-->>BR: "merge, or refuse"
    else no change earned
        ENG->>LED: "the row stands alone, no branch, no PR"
    end
```

## VF-04.6 The five extraDisciplines — guardrails written as prose, checked by judgement
```mermaid
flowchart TB
    classDef inv fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef warn fill:#f9f0d5,stroke:#8a7020,color:#111

    D["extraDisciplines — dream.config.json:68"]

    D --> G1["INVARIANT static-site-only: hand-written HTML/CSS/JS with a<br/>self-contained WebGL2 mesh — no bundler, no WASM, no compile.<br/>A hypothesis may not propose introducing a build framework.<br/>dream.config.json:69"]:::inv
    D --> G2["INVARIANT browser-checks-out-of-annexe: accessibility,<br/>Lighthouse performance and Playwright visual tests need the<br/>browsercontainer sidecar, which the HP annexe does NOT have.<br/>Never invent a browser evaluator — record such findings as<br/>HANDOFF (browser): notes — dream.config.json:70"]:::inv
    D --> G3["INVARIANT content-is-the-product: copy, messaging and asset<br/>integrity outrank code-structure concerns on a marketing site<br/>dream.config.json:71"]:::inv
    D --> G4["INVARIANT estate-health-is-read-not-collected: the snapshot is<br/>collected by CI at 02:30 UTC and committed as data. A night<br/>READS it, forms hypotheses about persistent reds, stale<br/>snapshots or roster drift, and never re-collects, never adds a<br/>token, never edits it by hand — dream.config.json:72"]:::inv
    D --> G5["INVARIANT adr-citations-must-exist: cite only ids present in<br/>docs/adr/; propose new ADRs by title and leave the id to the<br/>operator. Reports have cited ADR-0057, ADR-056 and ADR-2024 —<br/>none exist, and two are not even this repo's 4-digit scheme.<br/>dream.config.json:73"]:::inv

    G5 --> WHY["a citation to a record that does not exist is a fabricated<br/>authority, and it is WORSE than no citation because the reader<br/>cannot tell the difference"]:::warn
    G2 --> LOCAL["the browser path exists, just not here:<br/>npm run test:a11y and npm run test:perf — package.json:13<br/>see VF-02.10"]:::warn
    G1 --> SRC["ground truth for the discipline: build.sh:7 —<br/>no compile step, no bundler, no WASM"]:::warn
    G4 --> SRC2["ground truth: BASELINE-visionflow.md:234 Invariant 6"]:::warn
```

## VF-04.7 The ledger row — shape and what each column is allowed to say
```mermaid
erDiagram
    LEDGER_ROW {
        string Date "the night, ISO date — LEDGER.md:3"
        string Deep "the rotation surface drawn, or operator-handoff for a human row"
        string Finding "the hypothesis, given-when-then, truncated in the table"
        string Issue "NONE when no issue was opened"
        string PR "NONE when no pull request was opened"
        string Evaluated "yes, or n/a for an operator row"
        string Verdict "ACCEPT, INCONCLUSIVE, or OPERATOR"
        string Effect "what changed, empty when nothing did"
        string Witness "the session or run identifier that produced the row"
        string PriorNightFates "what happened to the previous night's proposals"
    }
    LEDGER_FILE {
        string path "docs/dream-cycle/LEDGER.md, declared at dream.config.json:75"
        string form "a bare markdown table — header, separator, one row per night, append-only"
        int rows "20 lines at this commit: 2 header lines and 18 night or operator rows"
        string firstRow "2026-08-16 content-integrity, INCONCLUSIVE — LEDGER.md:3"
    }
    LEDGER_FILE ||--o{ LEDGER_ROW : contains
```

## VF-04.8 What the ledger actually records — nine months of verdicts and three operator audits
```mermaid
flowchart TB
    classDef acc fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef inc fill:#f9f0d5,stroke:#8a7020,color:#111
    classDef op fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef bad fill:#f7dede,stroke:#a33333,color:#111

    L["docs/dream-cycle/LEDGER.md — LEDGER.md:1"]

    L --> A["ACCEPT nights: 08-17 x2, 08-29, 08-31, 09-01, 09-02,<br/>09-03, 09-04, 09-05"]:::acc
    L --> I["INCONCLUSIVE nights: 08-16 content-integrity, and three<br/>webgl-mesh nights 08-30, 09-06, 09-07 — LEDGER.md:17"]:::inc
    L --> O["OPERATOR rows — human audits, deliberately NOT counted<br/>toward the dry streak that parks a repo"]:::op

    O --> O1["2026-08-28 — evaluators converted to checked-in scripts after<br/>the annexe ssh dispatch mangled nested double quotes<br/>LEDGER.md:6"]:::op
    O --> O2["2026-09-06 — estate-health added as a bound slot, verdicts<br/>OK / STALE over 36 h / RED, governed by ADR-2008<br/>LEDGER.md:16"]:::op
    O --> O3["2026-09-07 — FABRICATED CANDIDATES audit: no branch and no PR<br/>ever existed for the 09-03 or 09-04 candidates; both patches<br/>targeted a path that has never existed in this repo, and the<br/>on-disk receipts contradict the reported counters. The parent<br/>receipts and findings are genuine; the candidate halves are<br/>not — LEDGER.md:18"]:::bad
    O --> O4["2026-09-07 — estate-health demoted to advisory, required false;<br/>the checker is UNTOUCHED because widening the RED band or<br/>relaxing the exit code would be reward hacking — LEDGER.md:19"]:::op
    O --> O5["2026-09-07 — webgl-mesh parked as sidecar-only per ADR-2009<br/>LEDGER.md:20"]:::op

    LESSON["The ledger's own standard, from ADR-2009: a verdict that a<br/>night could not observe its subject should be rare and<br/>informative; repeating INCONCLUSIVE on a schedule for a<br/>structural reason turns it into noise<br/>ADR-2009-webgl-mesh-deep-is-sidecar-only.md:63"]
    I --> LESSON
```

## VF-04.9 Landing a night's output — draft branch, human gate, no auto-merge
```mermaid
stateDiagram-v2
    direction TB
    [*] --> Hypothesis
    state "one falsifiable hypothesis against tonight's deep" as Hypothesis
    Hypothesis --> Measured
    state "measured on the repo's real evaluators — never on a new evaluator invented for the night" as Measured
    Measured --> Accept
    Measured --> Inconclusive
    state "ACCEPT — a change is earned" as Accept
    state "INCONCLUSIVE — the night could not observe its subject" as Inconclusive
    Accept --> Branch
    state "branch dream/<deep>-<date> — prefix at dream.config.json:76" as Branch
    Branch --> Draft
    state "DRAFT pull request, labels dream-cycle and marketing-site — dream.config.json:77" as Draft
    Draft --> Human
    state "a human reads the receipts and merges, or refuses — autoMerge false at dream.config.json:81" as Human
    Human --> Merged
    Human --> Refused
    state "merged to main — deploy.yml then runs every blocking gate like any other push — see VF-02.6" as Merged
    state "refused — the finding stays in the ledger as evidence" as Refused
    Inconclusive --> LedgerOnly
    state "a ledger row only" as LedgerOnly
    Merged --> [*]
    Refused --> [*]
    LedgerOnly --> [*]
    note right of Draft
      The judgment-broker boundary wired into development:
      evaluation is not promotion. ./README.md:128
      DIVERGENCE: nothing in this repo enforces the draft-PR
      shape — the branch prefix, the labels and autoMerge
      false are declarations the external engine honours —
      no workflow here rejects a dream/ branch that skipped
      the gate. The 2026-09-07 operator audit found reports
      claiming candidate branches that never existed.
    end note
```

## VF-04.10 Where the dream evaluators and the publication gates meet
```mermaid
flowchart LR
    classDef sh fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef only fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef gap fill:#f9f0d5,stroke:#8a7020,color:#111

    subgraph SHARED["Same script, two consumers"]
        S1["dream-build-check.sh — BUILD-OK<br/>dream.config.json:47 and deploy.yml:92"]:::sh
        S2["dream-link-check.sh — LINK-INTEGRITY-OK<br/>dream.config.json:48 and deploy.yml:100"]:::sh
        S3["dream-meta-tags-scan.sh — META-SCAN-OK<br/>dream.config.json:49 and deploy.yml:109"]:::sh
        S4["dream-structured-data-scan.sh — SD-SCAN-OK<br/>dream.config.json:50 and deploy.yml:118"]:::sh
    end

    subgraph DREAMONLY["Dream only"]
        D1["dream-asset-refs-scan.sh — ASSET-SCAN-OK<br/>orphan and payload decomposition, no CI consumer<br/>dream.config.json:51"]:::only
        D2["estate-health.mjs check — read-only estate verdict<br/>dream.config.json:59; CI runs the same command but only<br/>as a reported step — estate-health.yml:105"]:::only
    end

    subgraph CIONLY["Publication only"]
        C1["website-assets.mjs verify — ASSET-INVENTORY-OK<br/>deploy.yml:83, reused indirectly by the asset-refs scan<br/>dream-asset-refs-scan.sh:30"]:::gap
        C2["check-diagram-text.js — committed diagram baseline<br/>deploy.yml:125"]:::gap
        C3["drift-counter.mjs — reported or enforced — deploy.yml:134"]:::gap
    end

    SENT["The shared contract that makes the reuse safe: each script<br/>exits 0 even on failure and states its verdict as the last<br/>stdout sentinel, so the annexe can tell a finding from an<br/>infrastructure fault while CI greps the same line<br/>dream-meta-tags-scan.sh:9"]
    SHARED -.-> SENT

    NOBROWSER["COVERAGE GAP, by decision: no browser evaluator on either<br/>path. The mesh module is read by nothing — verified by grep<br/>at ADR-2009-webgl-mesh-deep-is-sidecar-only.md:84 — and the<br/>a11y and perf specs run only by hand — mesh-webgl.js:10"]:::gap
    DREAMONLY -.-> NOBROWSER
```
