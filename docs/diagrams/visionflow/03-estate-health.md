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
  - docs/BASELINE-visionflow.md
  - docs/adr/ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md
  - docs/adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md
  - docs/architecture/repository-map.md
  - tests/gates/estate-health-ci.test.mjs
  - tests/gates/run-all.sh
  - ./README.md
  - docs/jev-option.md
verified_commit: a4a6c20c34c0e298bc4a6725e71e58e00685f64c
---

## VF-03.1 The roster — the only place the estate is enumerated
```mermaid
flowchart LR
    classDef fp fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef imp fill:#f9f0d5,stroke:#8a7020,color:#111
    classDef priv fill:#f7dede,stroke:#a33333,color:#111

    R["scripts/estate-health/roster.json<br/>roster_version 1 — roster.json:2<br/>INVARIANT: estate-health.mjs holds NO repository names;<br/>roster order fixes the published display order — roster.json:3"]

    R --> REPOS["repos — 21 entries"]
    REPOS --> P1["DreamLab-AI/VisionFlow — canon (this area)<br/>roster.json:5"]:::fp
    REPOS --> P2["DreamLab-AI/VisionClaw — GPU engine, OWL 2 EL<br/>EXTERNAL: area visionclaw, VC-nn — roster.json:6"]:::fp
    REPOS --> P3["DreamLab-AI/agentbox — sovereign agent runtime<br/>EXTERNAL: area agentbox, AB-nn — roster.json:7"]:::fp
    REPOS --> P4["DreamLab-AI/solid-pod-rs — native Solid pod<br/>EXTERNAL: area solid-pod-rs, SP-nn — roster.json:8"]:::fp
    REPOS --> P5["DreamLab-AI/nostr-rust-forum — governance plane, relay<br/>EXTERNAL: area nostr-rust-forum, NF-nn — roster.json:9"]:::fp
    REPOS --> P6["DreamLab-AI/dreamlab-ai-website — commercial surface<br/>EXTERNAL: area dreamlab-ai-website, DW-nn — roster.json:10"]:::fp
    REPOS --> P7["DreamLab-AI/loom — Ontology Loom grounding facade<br/>EXTERNAL: no diagram area in this tree — roster.json:11"]:::fp
    REPOS --> P8["DreamLab-AI/knowledgeGraph — published corpus mirror<br/>ci_by_design: a publish target whose only workflow is the<br/>Pages build — EXTERNAL: area knowledgegraph, KG-nn — roster.json:12"]:::fp
    REPOS --> P15["DreamLab-AI/sidestr-rs — sidechain crates, added 2026-09-23<br/>EXTERNAL: area sidestr-rs, SR-nn — roster.json:13"]:::fp
    REPOS --> P16["DreamLab-AI/ontomark — Obsidian-vault ontology toolchain, crates<br/>EXTERNAL: area ontomark, OM-nn — roster.json:14"]:::fp
    REPOS --> P17["DreamLab-AI/ontomark-obsidian — Obsidian plugin for ontomark<br/>EXTERNAL: no diagram area in this tree — roster.json:15"]:::fp
    REPOS --> P9["DreamLab-AI/WasmVOWL — demo shell, default branch master<br/>ci_by_design: frozen, no workflows<br/>EXTERNAL: no diagram area; distinct from vowl-wasm — roster.json:16"]:::fp
    REPOS --> P10["jjohare/visionGraph — the vault, PRIVATE and cross-owner<br/>EXTERNAL: area visiongraph, VG-nn — roster.json:17"]:::priv
    REPOS --> P11["DreamLab-AI/vowl-wasm — published crate plus npm bundle<br/>EXTERNAL: area vowl-wasm, VW-nn — roster.json:18"]:::fp
    REPOS --> P12["DreamLab-AI/prose-sanitiser — published crate — roster.json:19"]:::fp
    REPOS --> P13["DreamLab-AI/diagram-ir — published crate — roster.json:20"]:::fp
    REPOS --> P14["DreamLab-AI/dream-engine — provenance IMPORTED,<br/>fork of ruvnet/dream-machine — roster.json:21<br/>EXTERNAL: the dream engine itself, see AB-23 and DE-nn"]:::imp
    REPOS --> P18["DreamLab-AI/visiontouch-rs — haptic input device study,<br/>private proof of principle, admitted by ADR-2017 — roster.json:22"]:::fp
    REPOS --> P19["DreamLab-AI/codescape — code-graph 3D view extracted from VisionClaw (ADR-2143)<br/>EXTERNAL: area codescape — roster.json:23"]:::fp
    REPOS --> P20["DreamLab-AI/sealmap — code maps and review evidence for diagrams-as-code corpora<br/>EXTERNAL: area sealmap — roster.json:24"]:::fp
    REPOS --> P21["DreamLab-AI/visioncraft — expectation-driven development method, pinned by agentbox<br/>EXTERNAL: area visioncraft — roster.json:25"]:::fp

    R --> SURF["surfaces — 6 probes — roster.json:27<br/>www.visionflow.info, narrativegoldmine.com,<br/>its /ns/v2.jsonld and /data/graph/stats.json,<br/>www.dreamlab-ai.com, and the VisionClaw<br/>ontology-latest/index.jsonld release asset"]
    R --> REG["registries — 26 entries — roster.json:62<br/>crates.io: vowl-wasm, prose-sanitiser, diagram-ir,<br/>solid-pod-rs, solid-pod-rs-server, nostr-bbs-core — roster.json:63-68,<br/>the six ontomark-family crates (parse, model, reason, compile,<br/>ingest, ngg1) and whelk-el — roster.json:69-75,<br/>all eight sidestr-rs crates: header, core, nostr, wallet,<br/>round, hitch, agent, webledgers-teller — roster.json:76-83,<br/>and codescape-model, codescape-collab, sealmap and visioncraft — roster.json:84-87<br/>npm: @dreamlab-ai/vowl-wasm — roster.json:88"]

    NOTE["The two enumerations of the estate are reconciled in one<br/>direction: repository-map.md:8 lists twelve repositories with local<br/>paths, sidestr-rs among them at repository-map.md:18, and<br/>repository-map.md:25 names the roster as THE authoritative list for<br/>nightly collection, scoping the other six roster repositories as<br/>supporting components — repository-map.md:29-40 — beside<br/>non-roster rows such as RuVector and Codescape. The roster still<br/>does not reference the map back."]
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
    COL->>GH: "roster repos, surfaces and registries — three pools of 4 run at once,<br/>and each repo issues 5 calls together — estate-health.mjs:942-945, estate-health.mjs:564"
    GH-->>COL: "per-field answers, or a degraded field plus a note"
    COL->>SNAP: "write the whole snapshot — estate-health.mjs:959"
    COL-->>JOB: "summary line plus ESTATE-HEALTH-COLLECTED — estate-health.mjs:968"
    JOB->>COL: "check, REPORTED only — estate-health.yml:105"
    COL-->>JOB: "verdict grepped into the step summary — estate-health.yml:107"
    Note over JOB: "INVARIANT: the job never fails on RED. Failing would abort<br/>before the commit, so the one night the estate breaks would be<br/>the night no snapshot is published — estate-health.yml:21"
    alt snapshot changed
        JOB->>SNAP: "commit as estate-health[bot] then push — estate-health.yml:145"
        JOB->>DEP: "gh workflow run deploy.yml --ref main — estate-health.yml:158"
        Note over JOB,DEP: "a push made with GITHUB_TOKEN fires no on:push trigger,<br/>so the deploy must be dispatched explicitly — estate-health.yml:30"
        DEP-->>SNAP: "published through the ordinary blocking gates, no privileged path"
        Note over DEP: "A red snapshot still deploys: no publication gate in deploy.yml<br/>consults estate health, and the workflow deliberately does not<br/>fail on check's non-zero exit<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:61"
    else unchanged
        JOB-->>JOB: "nothing to commit, deploy not dispatched — estate-health.yml:137"
    end
    DREAM->>SNAP: "reads the committed file offline, 30 minutes later"
```

**Debt (concurrency is per pool, not estate-wide):** `CONCURRENCY = 4` is commented "≤ 4 in flight, estate-wide" (`scripts/estate-health.mjs:164`). But `mapPool` bounds only its own items (`scripts/estate-health.mjs:235-244`), the repo, surface and registry pools run together (`scripts/estate-health.mjs:942-945`), and each repo worker fires five calls at once (`scripts/estate-health.mjs:564`). The source therefore allows about 28 requests in flight, 20 of them to the GitHub API. This is a source reading only; no rate-limit failure has been observed.

## VF-03.3 What is collected per repository — six calls, each degrading alone
```mermaid
flowchart LR
    classDef apicall fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef deg fill:#f9f0d5,stroke:#8a7020,color:#111

    E["one roster entry"] --> ACC["resolveRepoAccess — primary token first,<br/>retry with ESTATE_READ_TOKEN on 404 or 403<br/>estate-health.mjs:318 and estate-health.mjs:323"]:::apicall
    ACC -->|"neither token can see it"| UNREAD["readable false, ci.state unknown, null counts,<br/>a note explaining which tokens were tried<br/>estate-health.mjs:550"]:::deg
    ACC -->|"repo object resolved"| PAR["five calls issued together — estate-health.mjs:564"]

    PAR --> C1["collectHead — tip commit of the default branch:<br/>sha, short, date, first line of the message, url<br/>estate-health.mjs:620"]:::apicall
    PAR --> C2["collectCi — one UNFILTERED /actions/runs page of 100,<br/>matched to the default branch on head_branch client-side,<br/>because the API branch filter lagged by a fortnight<br/>estate-health.mjs:639 and estate-health.mjs:641, filtered at estate-health.mjs:411 — judgement call 8, estate-health.mjs:93"]:::apicall
    PAR --> C3["openPrCount — one /pulls page with state open and per_page 1, so the<br/>rel=last page number IS the count<br/>estate-health.mjs:500 and estate-health.mjs:511"]:::apicall
    PAR --> C4["collectRelease — latest release; a 404 simply means<br/>there is none and is not recorded as an error<br/>estate-health.mjs:692"]:::apicall
    PAR --> C5["collectPages — html_url, status, build_type;<br/>status null for a workflow-built site is 'not reported',<br/>never 'errored' — estate-health.mjs:720"]:::apicall

    C3 --> ISSUES["open_issues derived: open_issues_count counts issues AND<br/>PRs, so the difference is the issue count, clamped at zero<br/>estate-health.mjs:598"]:::apicall

    SELF["The observer effect: the collector's own in-progress run is<br/>excluded BY RUN ID, never by workflow name, so a genuinely<br/>failed previous nightly still counts red<br/>estate-health.mjs:192 and estate-health.mjs:368; the exclusion is itself<br/>recorded as a note — estate-health.mjs:662"]:::deg
    C2 --- SELF
```

## VF-03.4 The CI colour rule — red dominates, the verdict must be HEAD's, unknown is never green
```mermaid
stateDiagram-v2
    direction TB
    [*] --> Filter
    state "one unfiltered /actions/runs page, kept where head_branch is the default branch — runsOnBranch, estate-health.mjs:410" as Filter
    Filter --> Group : keep only push, schedule, workflow_dispatch — estate-health.mjs:177
    Filter --> Dropped : a pull_request run reports a proposal, not the branch
    state "latest run per workflow_id, newest by created_at, tie-broken by run_number — estate-health.mjs:363" as Group
    Group --> Fold
    state "ciStateFrom — estate-health.mjs:450" as Fold
    Fold --> AtHead : red, amber or green
    Fold --> NoneState : no qualifying runs at all
    state "ciStateAtHead — is any latest run on the tip commit — estate-health.mjs:474" as AtHead
    AtHead --> Kept : at least one latest run is on HEAD, the whole fold stands
    AtHead --> Older : every latest run is on an older commit
    state "docsOnlySince — compare each base with HEAD, at most three bases — estate-health.mjs:678" as Older
    Older --> Carried : every changed path is md, mdx or under docs/ — estate-health.mjs:414
    Older --> NoneState : any other file, a truncated list, or more than three bases
    state "the fold stands — red dominates, one failing workflow makes the repository red" as Kept
    state "the older verdict is carried, ci.carried_from lists the commits — estate-health.mjs:583" as Carried
    state "none, with a note naming the older commits — estate-health.mjs:586" as NoneState
    NoneState --> Exempt : the roster entry declares ci_by_design — estate-health.mjs:591
    state "exempt, rendered 'no CI by design' — estate-health.mjs:592" as Exempt
    state "excluded before grouping, so it can never mask a real result" as Dropped
    note right of Fold
      startup_failure counts RED: a workflow that
      could not start did not pass — estate-health.mjs:172
      An unrecognised future conclusion counts AMBER,
      never green — estate-health.mjs:174
    end note
    note right of AtHead
      The rule is per repository, not per workflow:
      one green run on HEAD keeps a red run from an
      older commit in the fold — the gate test asserts
      exactly that, estate-health-ci.test.mjs:30-32
    end note
    note right of Exempt
      A red or amber verdict on HEAD is never exempted;
      only a none becomes exempt — estate-health.mjs:112
    end note
```

**Invariant:** the HEAD rule cannot hide a real failure — a red run on the tip commit still reads red, and that is pinned by the gate suite rather than by prose (`tests/gates/estate-health-ci.test.mjs:27-28`, wired at `tests/gates/run-all.sh:25`).

**Debt (one page of runs):** since judgement call 8 the collector reads one unfiltered page of 100 runs across every branch and filters client-side (`scripts/estate-health.mjs:641`, `scripts/estate-health.mjs:646`); a repository whose pull-request and feature-branch runs fill that page pushes its default-branch runs off it, and the row then reads none rather than its real state.

## VF-03.5 Degradation discipline — the snapshot says what it does not know
```mermaid
flowchart TB
    classDef rule fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef bad fill:#f7dede,stroke:#a33333,color:#111

    R1["RULE — degrade, never throw. A 403, a slow surface or a repo<br/>made private degrades THAT FIELD and leaves the rest intact;<br/>the only non-zero exit from collect is a usage error or an<br/>unwritable output path — estate-health.mjs:30"]:::rule
    R2["RULE — the snapshot says what it does not know. Unreadable is<br/>readable false plus ci.state unknown plus null counts,<br/>never an optimistic zero — estate-health.mjs:37"]:::rule
    R3["RULE — deterministic ordering. Repos, surfaces and registries<br/>in roster order, CI runs sorted by workflow name, so the<br/>nightly commit is a real diff, not reordering noise<br/>estate-health.mjs:42"]:::rule

    MECH["request() never throws: a transport failure, timeout or abort<br/>returns ok false, status null and an error string<br/>estate-health.mjs:256 and :268"]:::rule
    R1 --> MECH
    NOTOK["no log line interpolates a token — estate-health.mjs:66"]:::rule

    HOLE["THE KNOWN HOLES: jjohare/visionGraph is owned by another<br/>account, so the workflow's repo-scoped GITHUB_TOKEN gets a 404, and the private<br/>visiontouch-rs row reads the same way. Without an ESTATE_READ_TOKEN secret that reaches them<br/>a row lands unreadable rather than failing the run — estate-health.yml:86"]:::bad
    R2 --> HOLE
    TRAP["TRAP recorded in the workflow itself: a maintainer's full-scope<br/>local PAT reads visionGraph WITHOUT the fallback, so a readable<br/>row in a locally produced snapshot is NOT evidence that the CI<br/>path works — estate-health.yml:92"]:::bad
    HOLE --> TRAP
    LIVE["Confirmed at this commit: summary.unreadable is 2 (visionGraph and visiontouch-rs, each HTTP 404 with both tokens tried)<br/>and summary.repos is 21 — estate-health.json:16, estate-health.json:10, estate-health.json:686 and estate-health.json:861"]:::bad
    HOLE --> LIVE
```

## VF-03.6 The check gate — offline verdict and its precedence
```mermaid
stateDiagram-v2
    direction TB
    [*] --> Load
    state "async check() — reads CHECK_PATH, optional Jev advisory — estate-health.mjs:982" as Load
    Load --> Stale1 : file missing — estate-health.mjs:983
    Load --> Stale2 : unparseable JSON
    Load --> Stale3 : schema is not visionflow.estate-health/1 — estate-health.mjs:996
    Load --> Scan
    state "walk repos, then surfaces — estate-health.mjs:1007" as Scan
    Scan --> RedRepo : ci.state red — print one RED line per bad run
    Scan --> AmberRepo : ci.state amber, or readable false — estate-health.mjs:1018
    Scan --> DownSurface : a surface with ok false — estate-health.mjs:1024
    RedRepo --> Verdict
    AmberRepo --> Verdict
    DownSurface --> Verdict
    state "verdict" as Verdict
    Verdict --> RED : any red repo OR any unreachable surface — exit 1
    Verdict --> STALE : otherwise, invalid age or older than 36 h — exit 1
    Verdict --> OK : otherwise — exit 0
    state "ESTATE-HEALTH-RED — estate-health.mjs:1048" as RED
    state "ESTATE-HEALTH-STALE — estate-health.mjs:1054" as STALE
    state "ESTATE-HEALTH-OK — estate-health.mjs:1058" as OK
    state "ESTATE-HEALTH-STALE" as Stale1
    state "ESTATE-HEALTH-STALE" as Stale2
    state "ESTATE-HEALTH-STALE" as Stale3
    note right of Verdict
      RED beats STALE: a stale snapshot showing a failure
      still shows a failure, and reporting staleness would
      bury the defect — estate-health.mjs:85
      AMBER prints a line but does NOT change the verdict.
      After a valid schema, finite age prints INFO against the limit
      Missing files, bad JSON or schema return earlier — estate-health.mjs:983-1001
      Optional Jev runs before the verdict — estate-health.mjs:1045
      Missing arrays scan as empty, future dates pass the age test
      estate-health.mjs:1007 and estate-health.mjs:1052
      Lines are collected in a buffer and printed immediately before the sentinel;
      an opt-in advisory block may add JEV lines to it but never changes the
      verdict or the exit code — estate-health.mjs:1004 and estate-health.mjs:1045
      Freshness window: --max-age-hours, default 36
      estate-health.mjs:218
    end note
```

## VF-03.8 Surface and registry probes — the judgement calls
```mermaid
flowchart TB
    classDef ok fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef warn fill:#f9f0d5,stroke:#8a7020,color:#111

    S["probeSurface — estate-health.mjs:748"]
    S --> H["HEAD first: a reachability probe should not pull an<br/>ontology down the wire"]:::ok
    H -->|"405, 501 or 403"| G["retried with GET — a method restriction is not an outage<br/>estate-health.mjs:761"]:::warn
    S --> J["roster entries with parse json use GET outright and the<br/>body is summarised for classes, pages, nodes, edges<br/>estate-health.mjs:789"]:::ok
    S --> CT["a wrong content-type is NOTED but does not mark the surface<br/>down; served-but-mislabelled is a lesser defect than<br/>unreachable, and conflating them makes the page cry wolf"]:::warn
    S --> DOWN["an unreachable surface DOES count towards ESTATE-HEALTH-RED:<br/>a dead published surface deserves to become a hypothesis"]:::warn

    R["collectRegistry — estate-health.mjs:807"]
    R --> CR["crates.io: max_version is read and never falls back to<br/>max_stable_version, which is null for a crate that has only<br/>shipped pre-releases such as nostr-bbs-core<br/>estate-health.mjs:828"]:::ok
    R --> NP["npm: dist-tags.latest, timestamped from the time map<br/>estate-health.mjs:855"]:::ok
    R --> UA["crates.io REQUIRES a User-Agent, so one identifying this<br/>collector is sent to every host — estate-health.mjs:154"]:::ok

    OKDEF["a surface's ok field is STATUS ONLY: res.status in the 2xx range,<br/>regardless of content-type or body — estate-health.mjs:778"]:::ok
    S --> OKDEF
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
    Note over P: "section id estate — index.html:1156<br/>the lead copy states the file is a committed snapshot,<br/>not a live query — index.html:1160"
    P->>M: "DOMContentLoaded calls initEstateHealth — main.js:919"
    M->>F: "fetch('data/estate-health.json', cache no-cache) — main.js:838"
    alt fetch ok
        F-->>M: the snapshot
        M->>P: "estateMarkup writes into #estate-body — main.js:841"
        Note over M,P: "RULE 1: the section's .container.reveal wrapper is already<br/>observed by initScrollReveal — replacing it would leave the<br/>section permanently at opacity 0, so everything writes into<br/>the plain child #estate-body — main.js:587"
        Note over M,P: "RULE 2: every string comes from the GitHub API by way of the<br/>collector, so nothing reaches innerHTML without esc()<br/>hrefs additionally pass a http/https allowlist — main.js:620"
    else fetch failed
        M->>P: "a note pointing at the committed file — failing loudly<br/>beats a blank section — main.js:844"
    end
    Note over M,P: "pill words since judgement calls 7 and 10: none reads 'no run on head'<br/>and exempt reads 'no CI by design' — main.js:598 and main.js:599"
    Note over P,F: "Relative times are computed in the browser, so a stale deploy<br/>reads as stale rather than quietly claiming freshness — main.js:628"
    Note over P,F: "The file is a REQUIRED entry in website/assets.manifest.json,<br/>so a missing snapshot fails the asset gate — assets.manifest.json:15"
```

## VF-03.11 The snapshot as it stands — 15 of 21 green, 2 red, 1 none, 1 exempt by design, 2 unreadable
```mermaid
flowchart TB
    classDef red fill:#f7dede,stroke:#a33333,color:#111
    classDef green fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef amb fill:#f9f0d5,stroke:#8a7020,color:#111

    SNAP["schema visionflow.estate-health/1, generated 2026-11-10T10:08Z<br/>collected by the nightly run at revision 8301ca8<br/>estate-health.json:2, estate-health.json:3 and estate-health.json:5"]

    SNAP --> SUM["summary — 21 repos: 15 green, 2 red, 0 amber,<br/>1 none, 1 exempt, 2 unreadable, 7 open PRs,<br/>6 of 6 surfaces reachable<br/>estate-health.json:10 through estate-health.json:19"]

    SUM --> R4["RED — solid-pod-rs (the CI run failed) and nostr-rust-forum (the Security Audit run failed)<br/>estate-health.json:290 and estate-health.json:301,<br/>estate-health.json:339 and estate-health.json:357"]:::red
    SUM --> KG["knowledgeGraph declares ci_by_design and now reads GREEN, estate-health.json:501: a red<br/>verdict on HEAD is never exempted, only a none is —<br/>estate-health.mjs:112 and estate-health.mjs:591"]:::green
    SUM --> A1["AMBER — none this snapshot: summary.amber is 0<br/>estate-health.json:13"]:::amb
    SUM --> N1["NONE — VisionClaw (no qualifying run on HEAD abf3162; its latest runs are on older commits and were red), so its older verdict is reported none<br/>estate-health.json:116 and estate-health.json:157 — estate-health.mjs:474"]:::amb
    SUM --> E2["EXEMPT — WasmVOWL, no workflows by design<br/>estate-health.json:655 and estate-health.json:663"]:::amb
    SUM --> U1["UNREADABLE — jjohare/visionGraph and visiontouch-rs, the known holes<br/>see VF-03.5 — estate-health.json:674, estate-health.json:678 and estate-health.json:849"]:::amb
    SUM --> G1["GREEN — VisionFlow itself, agentbox, dreamlab-ai-website, loom, knowledgeGraph, sidestr-rs,<br/>ontomark, ontomark-obsidian, vowl-wasm, prose-sanitiser, diagram-ir,<br/>dream-engine, codescape, sealmap and visioncraft"]:::green

    VERD["The offline check therefore returns ESTATE-HEALTH-RED:<br/>any red repo is enough, and the verdict beats staleness<br/>estate-health.mjs:1048 — see VF-03.6. check() has no branch for<br/>exempt, so an exempt row prints nothing and moves no verdict<br/>estate-health.mjs:1008 and estate-health.mjs:1018"]:::red
    SUM --> VERD

    INV["INVARIANT: the collector reports GitHub's own conclusion for the<br/>latest run per workflow and asserts nothing about whether that<br/>conclusion is correct<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:104"]
    VERD -.-> INV
```

**Debt:** rows flip without a recorded reason. Between the 2026-11-01 snapshot this topic last drew and the 2026-11-10 one it draws now, agentbox, dreamlab-ai-website and knowledgeGraph moved from red to green, nostr-rust-forum and solid-pod-rs moved from green to red, and visiontouch-rs moved from green to unreadable. VisionClaw remains none by the HEAD rule rather than by a passing run. No mechanism records why a row flips, so any trend is only visible by reading committed snapshots in sequence (`website/static/data/estate-health.json:11`).

**Debt (registry roster vs canon) — RESOLVED 2026-10-03, re-opened and RESOLVED again 2026-10-18:** the 10-03 dream ACCEPT (PR #13, merged as `c1e1498`) brought the roster to the seven sidestr crates canon then named; `webledgers-teller` was published later and went unprobed until `26df6c9` added it, with canon updated to eight in the same commit (`scripts/estate-health/roster.json:76-83`, estate register ER-3); the rewritten README no longer states a crate count (`./README.md:31`). The repository map still describes sidestr-rs as "seven published AGPL crates" (`docs/architecture/repository-map.md:18`), one behind the roster's eight (**Drift**).

**Drift (2026-10-15, dream-engine extraction):** the roster row for `dream-engine` (`provenance: imported`, a fork of ruvnet/dream-machine) is stamped before the extraction. The slug now names the new Rust repository and the fork is renamed dream-machine and archived (checked 2026-10-16). The row has not been changed, because the release manifest and its gate assert two imported rows, see DE-01.

**Since the earlier stamps:** `check` now builds its output in a buffer and prints it directly before the sentinel line, so a Jev advisory block, never able to change the verdict or exit code, can add `JEV priority` and `JEV triage` lines for the failed observations (`scripts/estate-health.mjs:1004`, `scripts/estate-health.mjs:1045`, `scripts/estate-health.mjs:1075`, `docs/jev-option.md:13-14`). The library default is off but agentbox projects `VISIONFLOW_JEV=all` into its shells, where these two items are public data and, since the 2026-10-30 System One unification, take the shared proxy's selected backend (TypeSafe by default) once the new agentbox image is activated, while hosted CI and the dream annexe do not inherit it (`docs/jev-option.md:3`, `docs/jev-option.md:13-14` and `docs/jev-option.md:16`). `collect` never calls it.


**Debt (snapshot shape and clock bounds, reproduced 2026-10-31):** A schema tag and current timestamp without any repository/surface arrays returns `ESTATE-HEALTH-OK`; a 2099 timestamp also returns OK. A non-array `repos` object throws instead of returning a classified verdict. These follow from the optional empty-array scans and one-sided age test at `scripts/estate-health.mjs:1007`, `scripts/estate-health.mjs:1024` and `scripts/estate-health.mjs:1052`. The source is unchanged from this topic's declared revision. See the [estate-health adversarial review](../../estate-review/2026-10-31-diagram-drift-and-health.md). The literal verdict is not proof that the full roster was measured.
