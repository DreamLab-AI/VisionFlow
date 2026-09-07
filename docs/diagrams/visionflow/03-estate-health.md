---
id: VF-03
title: Estate health — nightly CI collection, offline check, and the snapshot the site renders
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
  - docs/architecture/repository-map.md
adrs: [ADR-2006, ADR-2008]
sources:
  - scripts/estate-health.mjs
  - scripts/estate-health/roster.json
  - .github/workflows/estate-health.yml
  - .github/workflows/deploy.yml
  - website/static/data/estate-health.json
  - website/static/index.html
  - website/static/js/main.js
  - website/assets.manifest.json
  - dream.config.json
  - package.json
  - docs/BASELINE-visionflow.md
  - docs/adr/ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md
  - docs/adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md
  - docs/architecture/repository-map.md
verified_commit: bec06dc3a
---

## VF-03.1 The roster — the only place the estate is enumerated
```mermaid
flowchart TB
    classDef fp fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef imp fill:#f9f0d5,stroke:#8a7020,color:#111
    classDef priv fill:#f7dede,stroke:#a33333,color:#111

    R["scripts/estate-health/roster.json<br/>roster_version 1 — roster.json:2<br/>INVARIANT: estate-health.mjs holds NO repository names;<br/>roster order fixes the published display order — roster.json:3"]

    R --> REPOS["repos — 14 entries"]
    REPOS --> P1["DreamLab-AI/VisionFlow — canon (this area)<br/>roster.json:5"]:::fp
    REPOS --> P2["DreamLab-AI/VisionClaw — GPU engine, OWL 2 EL<br/>EXTERNAL: area visionclaw, VC-01..VC-37 — roster.json:6"]:::fp
    REPOS --> P3["DreamLab-AI/agentbox — sovereign agent runtime<br/>EXTERNAL: area agentbox, AB-01..AB-28 — roster.json:7"]:::fp
    REPOS --> P4["DreamLab-AI/solid-pod-rs — native Solid pod<br/>EXTERNAL: area solid-pod-rs, SP-nn — roster.json:8"]:::fp
    REPOS --> P5["DreamLab-AI/nostr-rust-forum — governance plane, relay<br/>EXTERNAL: area nostr-rust-forum, NF-nn — roster.json:9"]:::fp
    REPOS --> P6["DreamLab-AI/dreamlab-ai-website — commercial surface<br/>EXTERNAL: area dreamlab-ai-website, DW-nn — roster.json:10"]:::fp
    REPOS --> P7["DreamLab-AI/loom — Ontology Loom grounding facade<br/>EXTERNAL: no diagram area in this tree — roster.json:11"]:::fp
    REPOS --> P8["DreamLab-AI/knowledgeGraph — published corpus mirror<br/>EXTERNAL: area knowledgegraph, KG-nn — roster.json:12"]:::fp
    REPOS --> P9["DreamLab-AI/WasmVOWL — demo shell, default branch master<br/>EXTERNAL: no diagram area; distinct from vowl-wasm — roster.json:13"]:::fp
    REPOS --> P10["jjohare/visionGraph — the vault, PRIVATE and cross-owner<br/>EXTERNAL: area visiongraph, VG-nn — roster.json:14"]:::priv
    REPOS --> P11["DreamLab-AI/vowl-wasm — published crate plus npm bundle<br/>EXTERNAL: area vowl-wasm, VW-nn — roster.json:15"]:::fp
    REPOS --> P12["DreamLab-AI/prose-sanitiser — published crate — roster.json:16"]:::fp
    REPOS --> P13["DreamLab-AI/diagram-ir — published crate — roster.json:17"]:::fp
    REPOS --> P14["DreamLab-AI/dream-engine — provenance IMPORTED,<br/>fork of ruvnet/dream-machine — roster.json:18<br/>EXTERNAL: the dream engine itself, see AB-23"]:::imp

    R --> SURF["surfaces — 6 probes — roster.json:20<br/>www.visionflow.info, narrativegoldmine.com,<br/>its /ns/v2.jsonld and /data/graph/stats.json,<br/>www.dreamlab-ai.com, and the VisionClaw<br/>ontology-latest/index.jsonld release asset"]
    R --> REG["registries — 7 entries — roster.json:55<br/>crates.io: vowl-wasm, prose-sanitiser, diagram-ir,<br/>solid-pod-rs, solid-pod-rs-server, nostr-bbs-core<br/>npm: @dreamlab-ai/vowl-wasm — roster.json:62"]

    NOTE["DIVERGENCE: docs/architecture/repository-map.md:8 lists nine<br/>repositories with local paths; the roster lists fourteen with<br/>provenance. The two enumerations of 'the estate' are<br/>maintained separately and do not reference each other."]
    R -.-> NOTE
```

## VF-03.2 The nightly sequence — collect at 02:30, commit, dispatch, publish
```mermaid
sequenceDiagram
    autonumber
    participant CRON as "schedule 30 2 * * * — estate-health.yml:54"
    participant JOB as "job collect — estate-health.yml:66"
    participant COL as "estate-health.mjs collect"
    participant GH as "api.github.com, public surfaces, crates.io, npm"
    participant SNAP as "website/static/data/estate-health.json"
    participant DEP as "deploy.yml — see VF-02"
    participant DREAM as "03:00 UTC dream night — see VF-04"

    CRON->>JOB: "cron plus workflow_dispatch, permissions contents:write, actions:write<br/>estate-health.yml:58"
    JOB->>JOB: "checkout with persist-credentials true — estate-health.yml:74"
    JOB->>COL: "GITHUB_TOKEN plus optional ESTATE_READ_TOKEN — estate-health.yml:97"
    COL->>GH: "roster repos, surfaces and registries, at most 4 requests in flight<br/>estate-health.mjs:773"
    GH-->>COL: "per-field answers, or a degraded field plus a note"
    COL->>SNAP: "write the whole snapshot — estate-health.mjs:790"
    COL-->>JOB: "summary line plus ESTATE-HEALTH-COLLECTED — estate-health.mjs:799"
    JOB->>COL: "check, REPORTED only — estate-health.yml:105"
    COL-->>JOB: "verdict grepped into the step summary — estate-health.yml:107"
    Note over JOB: "INVARIANT: the job never fails on RED. Failing would abort<br/>before the commit, so the one night the estate breaks would be<br/>the night no snapshot is published — estate-health.yml:21"
    alt snapshot changed
        JOB->>SNAP: "commit as estate-health[bot] then push — estate-health.yml:145"
        JOB->>DEP: "gh workflow run deploy.yml --ref main — estate-health.yml:158"
        Note over JOB,DEP: "a push made with GITHUB_TOKEN fires no on:push trigger,<br/>so the deploy must be dispatched explicitly — estate-health.yml:30"
        DEP-->>SNAP: "published through the ordinary blocking gates, no privileged path"
    else unchanged
        JOB-->>JOB: "nothing to commit, deploy not dispatched — estate-health.yml:137"
    end
    DREAM->>SNAP: "reads the committed file offline, 30 minutes later"
```

## VF-03.3 What is collected per repository — six calls, each degrading alone
```mermaid
flowchart LR
    classDef apicall fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef deg fill:#f9f0d5,stroke:#8a7020,color:#111

    E["one roster entry"] --> ACC["resolveRepoAccess — primary token first,<br/>retry with ESTATE_READ_TOKEN on 404 or 403<br/>estate-health.mjs:272 and :277"]:::apicall
    ACC -->|"neither token can see it"| UNREAD["readable false, ci.state unknown, null counts,<br/>a note explaining which tokens were tried<br/>estate-health.mjs:432"]:::deg
    ACC -->|"repo object resolved"| PAR["five calls issued together — estate-health.mjs:446"]

    PAR --> C1["collectHead — tip commit of the default branch:<br/>sha, short, date, first line of the message, url<br/>estate-health.mjs:480"]:::apicall
    PAR --> C2["collectCi — one /actions/runs page, branch-scoped<br/>estate-health.mjs:499"]:::apicall
    PAR --> C3["openPrCount — one /pulls page with state open and per_page 1, so the<br/>rel=last page number IS the count<br/>estate-health.mjs:382 and :393"]:::apicall
    PAR --> C4["collectRelease — latest release; a 404 simply means<br/>there is none and is not recorded as an error<br/>estate-health.mjs:524"]:::apicall
    PAR --> C5["collectPages — html_url, status, build_type;<br/>status null for a workflow-built site is 'not reported',<br/>never 'errored' — estate-health.mjs:552"]:::apicall

    C3 --> ISSUES["open_issues derived: open_issues_count counts issues AND<br/>PRs, so the difference is the issue count, clamped at zero<br/>estate-health.mjs:458"]:::apicall

    SELF["The observer effect: the collector's own in-progress run is<br/>excluded BY RUN ID, never by workflow name, so a genuinely<br/>failed previous nightly still counts red<br/>estate-health.mjs:150 and :322; the exclusion is itself<br/>recorded as a note — estate-health.mjs:514"]:::deg
    C2 --- SELF
```

## VF-03.4 The CI colour rule — red dominates, unknown is never green
```mermaid
stateDiagram-v2
    direction LR
    [*] --> Filter
    state "one /actions/runs page for the default branch" as Filter
    Filter --> Group : keep only push, schedule, workflow_dispatch — estate-health.mjs:135
    Filter --> Dropped : a pull_request run reports a proposal, not the branch
    state "latest run per workflow_id, newest by created_at, tie-broken by run_number" as Group
    Group --> Fold
    state "ciStateFrom — estate-health.mjs:350" as Fold
    Fold --> Red : any latest run concluded failure, timed_out or startup_failure
    Fold --> Amber : cancelled, action_required, stale, still queued or in progress, or an unrecognised conclusion
    Fold --> Green : every latest run concluded success, skipped or neutral
    Fold --> NoneState : no qualifying runs at all
    state "red — one failing workflow makes the repository red" as Red
    state "amber" as Amber
    state "green" as Green
    state "none" as NoneState
    state "excluded before grouping, so it can never mask a real result" as Dropped
    note right of Fold
      startup_failure counts RED: a workflow that
      could not start did not pass — estate-health.mjs:130
      An unrecognised future conclusion counts AMBER,
      never green — estate-health.mjs:132
    end note
```

## VF-03.5 Degradation discipline — the snapshot says what it does not know
```mermaid
flowchart TB
    classDef rule fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef bad fill:#f7dede,stroke:#a33333,color:#111

    R1["RULE — degrade, never throw. A 403, a slow surface or a repo<br/>made private degrades THAT FIELD and leaves the rest intact;<br/>the only non-zero exit from collect is a usage error or an<br/>unwritable output path — estate-health.mjs:30"]:::rule
    R2["RULE — the snapshot says what it does not know. Unreadable is<br/>readable false plus ci.state unknown plus null counts,<br/>never an optimistic zero — estate-health.mjs:37"]:::rule
    R3["RULE — deterministic ordering. Repos, surfaces and registries<br/>in roster order, CI runs sorted by workflow name, so the<br/>nightly commit is a real diff, not reordering noise<br/>estate-health.mjs:42"]:::rule

    MECH["request() never throws: a transport failure, timeout or abort<br/>returns ok false, status null and an error string<br/>estate-health.mjs:210 and :228"]:::rule
    R1 --> MECH
    NOTOK["no log line interpolates a token — estate-health.mjs:66"]:::rule

    HOLE["THE ONE KNOWN HOLE: jjohare/visionGraph is owned by another<br/>account, so the workflow's repo-scoped GITHUB_TOKEN gets a 404.<br/>Without an ESTATE_READ_TOKEN secret the row lands unreadable<br/>rather than failing the run — estate-health.yml:86"]:::bad
    R2 --> HOLE
    TRAP["TRAP recorded in the workflow itself: a maintainer's full-scope<br/>local PAT reads visionGraph WITHOUT the fallback, so a readable<br/>row in a locally produced snapshot is NOT evidence that the CI<br/>path works — estate-health.yml:92"]:::bad
    HOLE --> TRAP
    LIVE["Confirmed at this commit: summary.unreadable is 1 and<br/>summary.repos is 14 — estate-health.json:15"]:::bad
    HOLE --> LIVE
```

## VF-03.6 The check gate — offline verdict and its precedence
```mermaid
stateDiagram-v2
    direction TB
    [*] --> Load
    state "check() — reads CHECK_PATH, no network, no token — estate-health.mjs:813" as Load
    Load --> Stale1 : file missing — estate-health.mjs:814
    Load --> Stale2 : unparseable JSON
    Load --> Stale3 : schema is not visionflow.estate-health/1 — estate-health.mjs:827
    Load --> Scan
    state "walk repos, then surfaces — estate-health.mjs:836" as Scan
    Scan --> RedRepo : ci.state red — print one RED line per bad run
    Scan --> AmberRepo : ci.state amber, or readable false — estate-health.mjs:846
    Scan --> DownSurface : a surface with ok false — estate-health.mjs:851
    RedRepo --> Verdict
    AmberRepo --> Verdict
    DownSurface --> Verdict
    state "verdict" as Verdict
    Verdict --> RED : any red repo OR any unreachable surface — exit 1
    Verdict --> STALE : otherwise, generated_at older than 36 h — exit 1
    Verdict --> OK : otherwise — exit 0
    state "ESTATE-HEALTH-RED — estate-health.mjs:858" as RED
    state "ESTATE-HEALTH-STALE — estate-health.mjs:865" as STALE
    state "ESTATE-HEALTH-OK — estate-health.mjs:869" as OK
    state "ESTATE-HEALTH-STALE" as Stale1
    state "ESTATE-HEALTH-STALE" as Stale2
    state "ESTATE-HEALTH-STALE" as Stale3
    note right of Verdict
      RED beats STALE: a stale snapshot showing a failure
      still shows a failure, and reporting staleness would
      bury the defect — estate-health.mjs:85
      AMBER prints a line but does NOT change the verdict.
      Freshness window: --max-age-hours, default 36
      estate-health.mjs:172
    end note
```

## VF-03.7 The snapshot schema — visionflow.estate-health/1
```mermaid
erDiagram
    SNAPSHOT {
        string schema "visionflow.estate-health/1 — estate-health.mjs:108"
        string generated_at "ISO timestamp; freshness is part of the payload — estate-health.json:3"
        object generator "revision, workflow_run url, collector path — estate-health.mjs:713"
        object summary "repos, green, red, amber, none, unreadable, open_prs, surfaces_ok, surfaces_total — estate-health.mjs:740"
    }
    REPO_ROW {
        string name "from the roster, in roster order"
        string full_name "owner and repo"
        string provenance "first-party or imported — roster.json:3"
        string role "one line from the roster"
        string visibility "public or private, null when unreadable"
        bool readable "false means no token could see it"
        string default_branch "the branch CI is measured on"
        object head "sha, short, date, message first line, url — estate-health.mjs:489"
        object ci "state plus one runRecord per workflow — estate-health.mjs:363"
        int open_prs "null when the call degraded"
        int open_issues "open_issues_count minus open_prs, clamped"
        object release "tag, date, url; null when there is none"
        object pages "url, status, build_type — estate-health.mjs:560"
        list notes "why any field on this row degraded"
    }
    SURFACE_ROW {
        string name "roster label"
        string url "probed with HEAD, or GET when the body is parsed"
        int status "HTTP status, null on a transport failure"
        bool ok "STATUS ONLY, 2xx — estate-health.mjs:610"
        string content_type "recorded; a mismatch is a note, not a demotion"
        int latency_ms "measured per probe"
        string note "joined notes, or null"
    }
    REGISTRY_ROW {
        string registry "crates.io or npm — estate-health.mjs:639"
        string name "package or crate name"
        string version "crates.io max_version, never max_stable_version — estate-health.mjs:675"
        string published_at "creation time of the version actually reported"
        string url "human-facing registry page"
    }
    SNAPSHOT ||--o{ REPO_ROW : repos
    SNAPSHOT ||--o{ SURFACE_ROW : surfaces
    SNAPSHOT ||--o{ REGISTRY_ROW : registries
```

## VF-03.8 Surface and registry probes — the judgement calls
```mermaid
flowchart TB
    classDef ok fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef warn fill:#f9f0d5,stroke:#8a7020,color:#111

    S["probeSurface — estate-health.mjs:580"]
    S --> H["HEAD first: a reachability probe should not pull an<br/>ontology down the wire"]:::ok
    H -->|"405, 501 or 403"| G["retried with GET — a method restriction is not an outage<br/>estate-health.mjs:593"]:::warn
    S --> J["roster entries with parse json use GET outright and the<br/>body is summarised for classes, pages, nodes, edges<br/>estate-health.mjs:621"]:::ok
    S --> CT["a wrong content-type is NOTED but does not mark the surface<br/>down; served-but-mislabelled is a lesser defect than<br/>unreachable, and conflating them makes the page cry wolf"]:::warn
    S --> DOWN["an unreachable surface DOES count towards ESTATE-HEALTH-RED:<br/>a dead published surface deserves to become a hypothesis"]:::warn

    R["collectRegistry — estate-health.mjs:639"]
    R --> CR["crates.io: max_version is read and never falls back to<br/>max_stable_version, which is null for a crate that has only<br/>shipped pre-releases such as nostr-bbs-core<br/>estate-health.mjs:660"]:::ok
    R --> NP["npm: dist-tags.latest, timestamped from the time map<br/>estate-health.mjs:687"]:::ok
    R --> UA["crates.io REQUIRES a User-Agent, so one identifying this<br/>collector is sent to every host — estate-health.mjs:112"]:::ok
```

## VF-03.9 How the site consumes the snapshot — committed file, no live query
```mermaid
sequenceDiagram
    autonumber
    participant V as Visitor
    participant P as "the deployed page"
    participant M as "js/main.js"
    participant F as "data/estate-health.json in the artefact"

    V->>P: "open www.visionflow.info and scroll to the estate section"
    Note over P: "section id estate — index.html:1041<br/>the lead copy states the file is a committed snapshot,<br/>not a live query — index.html:1045"
    P->>M: "DOMContentLoaded calls initEstateHealth — main.js:854"
    M->>F: "fetch('data/estate-health.json', cache no-cache) — main.js:833"
    alt fetch ok
        F-->>M: the snapshot
        M->>P: "estateMarkup writes into #estate-body — main.js:836"
        Note over M,P: "RULE 1: the section's .container.reveal wrapper is already<br/>observed by initScrollReveal — replacing it would leave the<br/>section permanently at opacity 0, so everything writes into<br/>the plain child #estate-body — main.js:583"
        Note over M,P: "RULE 2: every string comes from the GitHub API by way of the<br/>collector, so nothing reaches innerHTML without esc()<br/>hrefs additionally pass a http/https allowlist — main.js:611"
    else fetch failed
        M->>P: "a note pointing at the committed file — failing loudly<br/>beats a blank section — main.js:839"
    end
    Note over P,F: "Relative times are computed in the browser, so a stale deploy<br/>reads as stale rather than quietly claiming freshness — main.js:623"
    Note over P,F: "The file is a REQUIRED entry in website/assets.manifest.json,<br/>so a missing snapshot fails the asset gate — assets.manifest.json:12"
```

## VF-03.10 What estate health deliberately does not do
```mermaid
flowchart LR
    classDef no fill:#f7dede,stroke:#a33333,color:#111
    classDef yes fill:#e6f0dc,stroke:#4a7a2a,color:#111

    SNAP["the nightly snapshot"]
    SNAP --> N1["does NOT gate publication: no step in deploy.yml consults it,<br/>and a red snapshot still deploys<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:61"]:::no
    SNAP --> N2["does NOT gate a dream verdict: the evaluator is declared<br/>required false, so a red sibling is evidence for tonight's<br/>hypothesis, not a veto over it — dream.config.json:60"]:::no
    SNAP --> N3["does NOT judge substrate maturity: it reports observed signals<br/>only — CI conclusion, release tag, HTTP status — which is how<br/>it stays inside the ADR-2006 boundary<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:32"]:::no
    SNAP --> N4["does NOT prove a CI state is CORRECT: it reports GitHub's<br/>conclusion for the latest run per workflow and nothing more<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:104"]:::no

    SNAP --> Y1["DOES replace the hand-maintained closeout table as the living<br/>view; dated tables stay evidence of a day's audit<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:54"]:::yes
    SNAP --> Y2["DOES make staleness itself a finding: if the workflow stops<br/>running, check goes STALE after 36 hours and the dream night<br/>sees it — the monitor's own reader covers the monitor dying<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:69"]:::yes
    SNAP --> Y3["DOES publish as an ordinary bot data commit under the same<br/>gates as a human commit — estate-health.yml:145, see VF-02"]:::yes

    INV["INVARIANT 6 in the governing doc: collected by CI, read by the<br/>dream cycle; a dream night never collects one, acquires a token,<br/>or edits the snapshot by hand — BASELINE-visionflow.md:234"]
    SNAP -.-> INV
```
