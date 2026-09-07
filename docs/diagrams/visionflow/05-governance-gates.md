---
id: VF-05
title: Governance gates — every CI gate, what it reads, its pass/fail rule and whether it blocks
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
  - docs/architecture/compatibility-matrix.md
adrs: [ADR-2005, ADR-2006]
sources:
  - ./README.md
  - scripts/drift-counter/drift-counter.mjs
  - scripts/drift-counter/allowlist.json
  - scripts/drift-counter/README.md
  - .github/workflows/drift-counter.yml
  - scripts/check-fixture-drift.sh
  - .github/workflows/fixture-drift.yml
  - .github/workflows/harness-fitness-gates.yml
  - scripts/harness-audit.sh
  - .github/workflows/copyright-guard.yml
  - .github/workflows/deploy.yml
  - LICENSES/README.md
  - MAINTAINERS.md
  - scripts/mesh-smoke-preflight.sh
  - scripts/generate-release-manifest.sh
  - tests/gates/run-all.sh
  - tests/gates/drift-counter.test.sh
  - tests/gates/harness-audit.test.sh
  - tests/gates/release-manifest.test.sh
  - tests/gates/website-assets.test.sh
  - docs/architecture/compatibility-matrix.md
  - docs/BASELINE-visionflow.md
  - docs/protocol/mesh-smoke-test.md
  - package.json
verified_commit: bec06dc3a
---

## VF-05.1 Gate route table — trigger, script, verdict, blocking
```mermaid
flowchart TB
    classDef block fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef report fill:#fff4d6,stroke:#aa8833,color:#222
    classDef tool fill:#e4ecf8,stroke:#33559a,color:#222

    subgraph TRIG["Triggers"]
        PR["pull_request<br/>path-filtered"]
        PUSHMAIN["push to main"]
        ANY["push OR pull_request<br/>no path filter"]
        DISP["workflow_dispatch"]
    end

    PR --> WDRIFT["drift-counter.yml<br/>paths README.md, docs/**,<br/>scripts/drift-counter/**<br/>drift-counter.yml:28"]
    DISP --> WDRIFT
    WDRIFT --> SDRIFT["node drift-counter.mjs<br/>drift-counter.yml:80"]:::tool
    SDRIFT --> VDRIFT["exit 1 on any enforced-axis drift<br/>drift-counter.mjs:298"]:::block

    PR --> WFIX["fixture-drift.yml<br/>paths tests/fixtures/**<br/>fixture-drift.yml:39"]
    PUSHMAIN --> WFIX
    WFIX --> SFIX["check-fixture-drift.sh --json<br/>fixture-drift.yml:172"]:::tool
    SFIX --> VFIX["exit 1 on sha256 mismatch;<br/>exit 1 when NO canonical corpus<br/>resolves, unless waived<br/>fixture-drift.yml:149"]:::block

    PR --> WHARN["harness-fitness-gates.yml<br/>paths docs/engineering/templates/**<br/>harness-fitness-gates.yml:6"]
    PUSHMAIN --> WHARN
    WHARN --> J1["job harness-audit<br/>harness-audit.sh --target 50<br/>harness-fitness-gates.yml:56"]:::tool
    WHARN --> J2["job fixture-parity<br/>inline jq template schema check<br/>harness-fitness-gates.yml:108"]:::tool
    WHARN --> J3["job release-manifest<br/>release-manifest.test.sh<br/>harness-fitness-gates.yml:166"]:::tool
    J1 --> VH["exit 1 when pairing ratio &lt; target<br/>harness-audit.sh:430"]:::block
    J2 --> VH2["exit 1 on bad JSON, missing field,<br/>duplicate id or dangling pairing ref<br/>harness-fitness-gates.yml:144"]:::block
    J3 --> VH3["exit 1 when a candidate manifest<br/>can be cut with no fixture comparison"]:::block

    ANY --> WCOPY["copyright-guard.yml<br/>no path filter at all<br/>copyright-guard.yml:9"]
    WCOPY --> SCOPY["git ls-files over root *.txt<br/>copyright-guard.yml:36"]:::tool
    SCOPY --> VCOPY["exit 1 if luddites.txt / doctorow.txt<br/>or any unallowlisted root .txt is tracked<br/>copyright-guard.yml:45"]:::block

    PUSHMAIN --> WDEPL["deploy.yml<br/>publication path"]
    WDEPL --> DG1["BLOCKING gate 6 — committed diagram<br/>baseline text visibility<br/>deploy.yml:124"]:::block
    WDEPL --> DG2["drift counter, REPORTED when the<br/>pinned agentbox checkout failed<br/>deploy.yml:134"]:::report

    MANUAL["operator, by hand — no workflow"] --> SMESH["mesh-smoke-preflight.sh<br/>read-only substrate probe<br/>mesh-smoke-preflight.sh:337"]:::report
    MANUAL --> SREL["generate-release-manifest.sh<br/>exit 3 without a canonical revision set<br/>generate-release-manifest.sh:233"]:::block

    NOTE1["INVARIANT: every count claimed in canon prose has one queryable source;<br/>a second distinct figure for one axis is a failure, not a footnote<br/>BASELINE-visionflow.md:232"]
    NOTE2["DIVERGENCE: mesh-smoke-preflight.sh is wired to NO workflow.<br/>Its output is pasted by hand into mesh-smoke-test.md:71"]
    NOTE3["HISTORICAL SNAPSHOT: compatibility-matrix.md:73 records the July<br/>harness PASS at target 80%. Current CI passes --target 50.<br/>These are different run configurations, not a fresh 80% CI result"]
```

## VF-05.2 drift-counter — axis truth query and the sibling source pin
```mermaid
sequenceDiagram
    autonumber
    participant CI as "drift-counter.yml job"
    participant AB as "_agentbox checkout pinned at ref<br/>drift-counter.yml:61"
    participant DC as "drift-counter.mjs"
    participant AL as "allowlist.json"
    participant SKC as "agentbox scripts/skill-count-check.js"
    participant OB as "agentbox mcp/servers/ontology-bridge.js"

    CI->>AB: actions/checkout DreamLab-AI/agentbox into _agentbox<br/>drift-counter.yml:62
    CI->>DC: DRIFT_AGENTBOX_DIR=_agentbox node drift-counter.mjs<br/>drift-counter.yml:80
    DC->>AL: read DRIFT_ALLOWLIST or the committed allowlist<br/>drift-counter.mjs:58
    DC->>DC: resolveAgentbox — first candidate holding<br/>mcp/servers/ontology-bridge.js wins<br/>drift-counter.mjs:77

    rect rgb(228, 238, 250)
    Note over DC,AB: PIN CHECK runs BEFORE any figure is compared
    DC->>AB: git -C agentbox rev-parse HEAD<br/>drift-counter.mjs:104
    alt HEAD equals allowlist source_pin.revision
        AB-->>DC: pinned-ok<br/>drift-counter.mjs:111
    else HEAD moved
        AB-->>DC: pin-mismatch — hard fail unless --allow-pin-drift<br/>drift-counter.mjs:253
    else checkout absent or git unreadable
        AB-->>DC: pin-unverifiable — hard fail only under --strict<br/>drift-counter.mjs:254
    end
    end

    DC->>SKC: execFileSync node skill-count-check.js, parse .count<br/>drift-counter.mjs:129
    Note over DC,SKC: exit 1 from the agentbox script still carries JSON on stdout —<br/>the count is recovered from it rather than lost<br/>drift-counter.mjs:141
    SKC-->>DC: skills axis truth
    DC->>OB: scan the const TOOLS array, counting only<br/>two-space object literals and bare references<br/>drift-counter.mjs:165
    OB-->>DC: mcp-ontology-tools axis truth
    DC->>DC: ontology-classes — env or committed count file, else UNAVAILABLE<br/>drift-counter.mjs:187
    Note over DC: roster axis is declared planned in the allowlist and is never queried<br/>drift-counter.mjs:258
    DC-->>CI: RESULT line then process.exit(hardFail ? 1 : 0)<br/>drift-counter.mjs:298
```

## VF-05.3 What an allowlist entry actually does — policing, not suppressing
```mermaid
flowchart TB
    classDef policed fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef free fill:#f0f0f0,stroke:#888,color:#222
    classDef fail fill:#ffe0e0,stroke:#aa3333,color:#222

    TREE["Every 'N skills' / 'N MCP tools' string in the tree"]

    TREE --> Q{"Does a site or file in<br/>allowlist.json name it?"}

    Q -->|no| FREE["NOT policed. The counter never reads it.<br/>Legitimately distinct subjects survive:<br/>a case study's 350 skills, VisionClaw's<br/>native 7 MCP tools, agentbox's 180+ total<br/>allowlist.json:2"]:::free

    Q -->|"yes, scan mode"| SCAN["skills axis, match=scan<br/>allowlist.json:21"]:::policed
    SCAN --> SCANR["EVERY occurrence of<br/>pattern in EVERY listed file<br/>must equal the queried truth<br/>allowlist.json:22"]
    SCANR --> SCANF["files: README.md, docs/ecosystem-map.md,<br/>docs/PRD-website.md, website/static/index.html<br/>allowlist.json:24"]
    SCANF --> SCANV{"stated == truth?"}
    SCANV -->|no| DRIFT["DRIFT finding, hardFail<br/>drift-counter.mjs:210"]:::fail
    SCANV -->|"file listed but pattern never matched"| NOMATCH["no-match finding — a policed file<br/>that carries no tracked figure FAILS<br/>drift-counter.mjs:213"]:::fail

    Q -->|"yes, sites mode"| SITES["mcp-ontology-tools axis, match=sites<br/>allowlist.json:41"]:::policed
    SITES --> SITER["Each pinned file+regex must match ONCE<br/>and equal the truth; the regex carries its<br/>own context so it cannot hit a sibling figure<br/>allowlist.json:43"]
    SITER --> SITEV{"regex matched?"}
    SITEV -->|"file gone"| FMISS["file-missing — FAIL<br/>drift-counter.mjs:228"]:::fail
    SITEV -->|"pattern moved"| SMISS["site-missing — FAIL<br/>drift-counter.mjs:231"]:::fail
    SITEV -->|yes| CMP["compare stated vs truth<br/>drift-counter.mjs:234"]

    EXCL["An allowlist entry SUPPRESSES NOTHING.<br/>It is the opposite of a mute list: entry = policed,<br/>absence = invisible. Adding a self-description site<br/>REQUIRES an allowlist edit — that is the review contract<br/>drift-counter/README.md:52"]
    EXCL -.-> Q

    DENOM["Each axis publishes a denominator so a reader can settle<br/>what a policed figure claims without reading the counter:<br/>counts, expression, authority, excludes<br/>allowlist.json:14"]
    DENOM -.-> SCAN
    DENOM -.-> SITES

    HOLE["The archived-ADR-002 hole: two sites pointed at a path moved<br/>into docs/archive/, reported file-missing on every run, and that<br/>looked like coverage. Repointed 2026-09-05<br/>allowlist.json:50"]:::fail
    HOLE -.-> FMISS

    DIV["DIVERGENCE: ./README.md:150 renders '7 Ontology MCP Tools' beside<br/>./README.md:157 '12 MCP Ontology Tools'. The sites regex matches only<br/>the second word order, so the adjacency escapes the gate<br/>BASELINE-visionflow.md:182"]
```

## VF-05.4 drift-counter verdict — per-axis states and the fail-open rule
```mermaid
stateDiagram-v2
    [*] --> PinCheck

    PinCheck --> PinnedOk : "HEAD == source_pin.revision"
    PinCheck --> PinMismatch : "sibling moved off the pin"
    PinCheck --> PinUnverifiable : "no checkout, or git unreadable"
    PinCheck --> NoPin : "allowlist declares no source_pin"

    PinMismatch --> HardFail : "default"
    PinMismatch --> PinnedOk : "--allow-pin-drift downgrades to WARN"
    PinUnverifiable --> HardFail : "--strict only"

    PinnedOk --> PerAxis
    NoPin --> PerAxis
    PinUnverifiable --> PerAxis : "not strict"

    state PerAxis {
        [*] --> Planned : "spec.status == planned (roster)"
        [*] --> Unavailable : "source query returned unavailable"
        [*] --> Enforced : "source available"

        Planned --> Reported : "never queried, never enforced"
        Unavailable --> Reported : "reported, not enforced"
        Unavailable --> AxisFail : "--strict flips unavailability into failure"
        Enforced --> AxisPass : "every finding ok"
        Enforced --> AxisFail : "any drift, file-missing, site-missing or no-match"
    }

    AxisFail --> HardFail
    AxisPass --> Green
    Reported --> Green

    HardFail --> [*] : "exit 1 — RESULT: FAIL"
    Green --> [*] : "exit 0 — RESULT: PASS"

    note right of PerAxis
        INVARIANT: partial-source failure mode. A source being
        down blocks THAT axis, never the whole gate — a figure
        disagreeing with an AVAILABLE source is a hard failure.
        drift-counter.mjs:266 and drift-counter/README.md:57
    end note

    note right of Unavailable
        ontology-classes stays unavailable until VisionClaw
        publishes a script-queryable count — the allowlist records
        the exact switch that turns it on with no code change.
        allowlist.json:74
    end note

    note right of Planned
        EXTERNAL: the roster axis is blocked on nostr-rust-forum
        exposing agent_registry as script-queryable — see NF-NN
        for the forum-side registry. allowlist.json:91
    end note
```

## VF-05.5 fixture-drift.yml — the gate that refuses to pass vacuously
```mermaid
sequenceDiagram
    autonumber
    participant GH as "fixture-drift.yml job"
    participant CO as "actions/checkout DreamLab-AI/VisionClaw<br/>continue-on-error<br/>fixture-drift.yml:81"
    participant LOC as "step locate"
    participant ENF as "step Enforce that a canonical revision set exists"
    participant SH as "check-fixture-drift.sh"

    GH->>CO: check out CANONICAL_REPO into _canonical<br/>fixture-drift.yml:65
    Note over CO: no deploy key or PAT on this runner, so the<br/>checkout is expected to fail and is allowed to
    CO-->>GH: outcome recorded, job continues

    GH->>LOC: resolve a canonical corpus
    alt workflow_dispatch canonical_dir input is a real directory
        LOC-->>GH: has_canonical=true
    else _canonical/tests/fixtures exists AND holds *.json
        LOC-->>GH: has_canonical=true, revision from git rev-parse<br/>fixture-drift.yml:105
    else nothing resolves
        LOC-->>GH: has_canonical=false
    end
    LOC->>LOC: enumerate consumer copies in THIS repo:<br/>find *.json under any */fixtures/*, minus _canonical and node_modules<br/>fixture-drift.yml:121

    alt has_canonical != true
        GH->>ENF: run the enforcement step<br/>fixture-drift.yml:131
        alt waive_reason supplied on workflow_dispatch
            ENF-->>GH: ::warning:: WAIVED, recorded in the log, exit 0<br/>fixture-drift.yml:137
        else no waiver
            ENF-->>GH: ::error:: nothing was compared, exit 1<br/>fixture-drift.yml:149
        end
    else has_canonical == true
        alt copy_count == 0
            GH-->>GH: says so explicitly and exits 0 —<br/>a resolved corpus with no local consumer copies<br/>fixture-drift.yml:163
        else copies present
            GH->>SH: check-fixture-drift.sh --canonical DIR --json COPIES<br/>fixture-drift.yml:172
            SH-->>GH: exit 1 on drift
        end
    end

    Note over GH,ENF: INVARIANT: a gate that cannot fail certifies nothing.<br/>Until 2026-09-05 the locate step always set has_canonical=false<br/>and the job exited 0 through a "No fixtures to check" branch<br/>fixture-drift.yml:5
    Note over GH: DIVERGENCE: option (a), wiring sibling checkouts with a deploy key,<br/>is blocked on credentials that cannot be provisioned from the tree —<br/>so every fixture-touching PR fails this gate deliberately<br/>fixture-drift.yml:19
```

## VF-05.6 check-fixture-drift.sh — checksum comparison and its stale default
```mermaid
flowchart TB
    classDef bad fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef ok fill:#e0f2e4,stroke:#2f7a45,color:#222

    START["check-fixture-drift.sh [OPTIONS] [REPO_PATH ...]"] --> ARGS["parse --canonical, --json, --quiet<br/>check-fixture-drift.sh:50"]
    ARGS --> RESOLVE{"--canonical supplied?"}

    RESOLVE -->|no| PROBE["probe well-known locations, first of which is<br/>project/docs/specs/fixtures<br/>check-fixture-drift.sh:82"]:::bad
    PROBE --> STALE["DOC-DRIFT: that path was REMOVED. The corpus moved to<br/>VisionClaw tests/fixtures/ on 2026-06-29, which<br/>the historical Tests/Ops row at compatibility-matrix.md:31 records,<br/>while the script header<br/>at check-fixture-drift.sh:5 still contradicts.<br/>Unqualified runs therefore exit 2, never 0"]:::bad
    STALE --> EXIT2["ERROR: Canonical fixture directory not found, exit 2<br/>check-fixture-drift.sh:93"]:::bad

    RESOLVE -->|yes| DEFREPOS{"any REPO_PATH given?"}
    DEFREPOS -->|no| DEFAULTS["fall back to fixed workspace paths:<br/>agentbox upstream_vectors, two solid-pod-rs crates,<br/>nostr-rust-forum tests/fixtures<br/>check-fixture-drift.sh:104"]
    DEFREPOS -->|yes| INDEX
    DEFAULTS --> INDEX["build CANONICAL_SUMS — sha256 per *.json,<br/>keyed by path relative to the corpus root<br/>check-fixture-drift.sh:133"]

    INDEX --> LOOP["check_repo per consumer path<br/>check-fixture-drift.sh:162"]
    LOOP --> C1{"canonical file present<br/>in the consumer copy?"}
    C1 -->|no| MISS["MISS — counted, reported, does NOT set DRIFT_DETECTED<br/>check-fixture-drift.sh:215"]
    C1 -->|yes| C2{"sha256 equal?"}
    C2 -->|yes| MATCH["MATCH<br/>check-fixture-drift.sh:225"]:::ok
    C2 -->|no| DRIFT["DRIFT — prints both truncated digests<br/>and sets DRIFT_DETECTED<br/>check-fixture-drift.sh:231"]:::bad

    LOOP --> EXTRA["a consumer *.json with no canonical twin is EXTRA only when it<br/>looks protocol-shaped: schemas/*, *.schema.json, or a basename<br/>a canonical file also uses. Everything else is silently skipped<br/>check-fixture-drift.sh:186"]

    MATCH --> SUM["summary: canonical count, repos checked,<br/>matched / drifted / missing / extra<br/>check-fixture-drift.sh:300"]
    MISS --> SUM
    DRIFT --> SUM
    EXTRA --> SUM
    SUM --> VERD{"DRIFT_DETECTED?"}
    VERD -->|yes| E1["exit 1<br/>check-fixture-drift.sh:335"]:::bad
    VERD -->|no| E0["exit 0 — note that MISSING alone still exits 0<br/>check-fixture-drift.sh:337"]:::ok

    NOTE["EXTERNAL: the canonical corpus and its consumers live in sibling repos —<br/>VisionClaw (VC-NN), agentbox (AB-NN), solid-pod-rs (SP-NN),<br/>nostr-rust-forum (NF-NN). VisionFlow's own tests/fixtures/ holds no JSON"]
```

## VF-05.7 harness-fitness-gates.yml — three independent jobs, three verdicts
```mermaid
flowchart LR
    classDef job fill:#e4ecf8,stroke:#33559a,color:#222
    classDef block fill:#ffe0e0,stroke:#aa3333,color:#222

    TRIG["pull_request or push to main touching<br/>docs/engineering/templates/**, schemas/**,<br/>compatibility-matrix.md, harness-audit.sh,<br/>generate-release-manifest.sh, the release schema,<br/>tests/gates/**<br/>harness-fitness-gates.yml:6"]

    TRIG --> A["job harness-audit<br/>Pairing ratio audit"]:::job
    TRIG --> B["job fixture-parity<br/>Template schema validation"]:::job
    TRIG --> C["job release-manifest<br/>Roster and fixture gate"]:::job

    A --> A1["FIRST prove the audit can fail:<br/>tests/gates/harness-audit.test.sh<br/>harness-fitness-gates.yml:44"]
    A1 --> A2["then bash harness-audit.sh --target 50,<br/>exit code captured, NOT propagated directly<br/>harness-fitness-gates.yml:56"]
    A2 --> A3["separate step turns a non-zero code into<br/>::error:: below the minimum threshold, exit 1<br/>harness-fitness-gates.yml:64"]:::block
    A2 --> A4["--strict-sources deliberately OMITTED: the sibling<br/>substrate checkouts are absent on a hosted runner, so<br/>strict would fail for the environment, not the templates<br/>harness-fitness-gates.yml:53"]

    B --> B1["per docs/engineering/templates/*.json:<br/>jq empty parses the file"]
    B1 --> B2["require topology, version, guides, sensors, pairings<br/>harness-fitness-gates.yml:108"]:::block
    B2 --> B3["guide ids unique<br/>harness-fitness-gates.yml:115"]:::block
    B3 --> B4["sensor ids unique<br/>harness-fitness-gates.yml:122"]:::block
    B4 --> B5["every pairing guide_id and sensor_id<br/>resolves to a declared control<br/>harness-fitness-gates.yml:129"]:::block
    B5 --> B6["errors accumulate, one exit 1 at the end<br/>harness-fitness-gates.yml:144"]

    C --> C1["tests/gates/release-manifest.test.sh only —<br/>the generator itself is not run in CI<br/>harness-fitness-gates.yml:166"]:::block

    NOTE1["TARGET SCOPE: the script default is 80 (harness-audit.sh:57).<br/>Historical Harness Coverage totals at compatibility-matrix.md:73<br/>record July's PASS at target 80%, while this workflow passes --target 50"]
    NOTE2["INVARIANT: the missing template directory is a skip, not a failure —<br/>job B exits 0 when docs/engineering/templates does not exist"]
```

## VF-05.8 harness-audit.sh — the three defects the counting rules close
```mermaid
sequenceDiagram
    autonumber
    participant SH as "harness-audit.sh"
    participant TPL as "docs/engineering/templates/*.json"
    participant FS as "sibling substrate checkouts"

    SH->>TPL: jq guides[].id, sensors[].id, pairings[]

    rect rgb(230, 242, 232)
    Note over SH: DEFECT 2 — edges were counted with duplicates,<br/>so one pairing repeated twice scored 200% and passed any target
    SH->>SH: EDGE_SEEN is keyed on the guide-id and sensor-id pair —<br/>a repeat is counted as duplicate and dropped<br/>harness-audit.sh:257
    SH->>SH: an edge naming an id that does not exist is<br/>DANGLING and cannot evidence coverage<br/>harness-audit.sh:264
    SH->>SH: coverage = DISTINCT paired controls / declared controls,<br/>bounded by 100% by construction<br/>harness-audit.sh:276
    end

    rect rgb(232, 238, 250)
    Note over SH,FS: DEFECT 1 — source paths were never resolved. A control counted<br/>as source-backed whenever source_status was absent or "present"
    SH->>SH: source_status == planned short-circuits to state planned<br/>harness-audit.sh:305
    SH->>FS: otherwise resolve SUBSTRATE:LOCATOR against the<br/>substrate root map<br/>harness-audit.sh:67
    alt no substrate prefix, or an http(s) URL
        FS-->>SH: descriptor — stays IN the backing denominator<br/>harness-audit.sh:134
    else substrate has no configured or existing root
        FS-->>SH: unverifiable — EXCLUDED from the denominator<br/>harness-audit.sh:143
    else locator contains a space
        FS-->>SH: descriptor, a prose artefact no filesystem check settles<br/>harness-audit.sh:175
    else locator is a glob
        FS-->>SH: resolved if any match exists<br/>harness-audit.sh:186
    else plain path
        FS-->>SH: resolved when the path exists, else unresolved<br/>harness-audit.sh:196
    end
    end

    rect rgb(250, 244, 228)
    Note over SH: DEFECT 3 — backing was counted per control, so two controls<br/>naming one file scored as two backed controls
    SH->>SH: SOURCE_STATE keyed by the source STRING —<br/>planned loses only to a resolved sibling<br/>harness-audit.sh:326
    SH->>SH: BACKABLE = distinct sources minus unverifiable<br/>harness-audit.sh:396
    SH->>SH: backed_pct = resolved / BACKABLE<br/>harness-audit.sh:398
    end

    SH->>SH: verdict — pairing ratio vs TARGET_RATIO<br/>harness-audit.sh:430
    SH->>SH: --strict-sources additionally fails on any<br/>unresolved OR unverifiable source<br/>harness-audit.sh:437
    SH-->>SH: exit "$FAIL"<br/>harness-audit.sh:464

    Note over SH,FS: EXTERNAL: substrate roots point at VisionClaw (VC-NN), agentbox (AB-NN),<br/>solid-pod-rs (SP-NN), nostr-rust-forum (NF-NN) and ruvector — VisionFlow<br/>resolves against itself. Their absence is reported, never a silent pass
```

## VF-05.9 copyright-guard.yml — defence in depth beyond .gitignore
```mermaid
flowchart TB
    classDef block fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef ok fill:#e0f2e4,stroke:#2f7a45,color:#222

    T["EVERY push and EVERY pull_request —<br/>this is the only gate with no path filter<br/>copyright-guard.yml:9"] --> S1

    S1["git ls-files for luddites.txt, doctorow.txt<br/>and their nested forms<br/>copyright-guard.yml:25"] --> Q1{"any tracked?"}
    Q1 -->|yes| F1["::error:: copyrighted source text is tracked,<br/>exit 1"]:::block
    Q1 -->|no| S2["enumerate every tracked ROOT-LEVEL *.txt<br/>copyright-guard.yml:36"]

    S2 --> Q2{"on the allowlist<br/>LICENSE.txt requirements.txt<br/>copyright-guard.yml:34"}
    Q2 -->|no| F2["::error:: unallowlisted root .txt —<br/>possible copyrighted source leak, exit 1.<br/>The remedy is naming it in the workflow's<br/>allow list, a reviewed diff<br/>copyright-guard.yml:45"]:::block
    Q2 -->|yes| PASS["OK: no copyrighted source texts tracked<br/>copyright-guard.yml:48"]:::ok

    WHY["INVARIANT: .gitignore alone does not stop a git add -f.<br/>The book's transition chapter reads copyrighted sources that must<br/>never enter this PUBLIC repo, so the check runs on tracked state,<br/>not on the working tree<br/>copyright-guard.yml:3"]
    WHY -.-> T

    NOTE["DIVERGENCE: the guard polices only ROOT-LEVEL *.txt by regex.<br/>A copyrighted .txt committed one directory down passes, and<br/>texput.log at the repo root is a .log, outside the pattern entirely"]
```

## VF-05.10 Licensing and ownership governance surface
```mermaid
flowchart TB
    classDef gap fill:#fff4d6,stroke:#aa8833,color:#222

    LIC["LICENSES/README.md"] --> L1["States the repo documents an ecosystem<br/>mixing MPL-2.0 and AGPL-3.0 components<br/>LICENSES/README.md:3"]
    LIC --> L2["Until a root licence file is added, repo-local docs and<br/>website assets need explicit maintainer confirmation<br/>before reuse outside the ecosystem<br/>LICENSES/README.md:5"]:::gap
    LIC --> L3["Delegates sibling licences to<br/>docs/architecture/licensing.md<br/>LICENSES/README.md:7"]

    MNT["MAINTAINERS.md"] --> M1["Two named maintainers with split focus:<br/>coordination architecture and public site;<br/>upstream Solid/JSS and DID:Nostr alignment<br/>MAINTAINERS.md:7"]
    MNT --> M2["Triage rule — security and protocol issues go to the<br/>OWNING substrate first; cross-repository architecture<br/>issues belong in VisionFlow docs<br/>MAINTAINERS.md:10"]

    L3 --> XREF["licensing split, per repo — see VF-08"]
    M2 --> XREF2["ownership rule and dependency direction — see VF-08"]

    GATE["copyright-guard.yml is the ONLY automated enforcement<br/>on this surface; the licence and maintainer rules are<br/>prose obligations with no CI gate"]:::gap

    DIVN["DIVERGENCE: LICENSES/ contains a README and no licence files,<br/>and the repository still has no root LICENSE — the condition<br/>LICENSES/README.md:5 makes conditional is still unmet.<br/>The README.md badge nevertheless advertises AGPL-3.0"]:::gap
```

## VF-05.11 tests/gates — proving each gate can still fail
```mermaid
flowchart TB
    classDef suite fill:#e4ecf8,stroke:#33559a,color:#222

    WHY["A gate nobody tests is a gate nobody knows is broken:<br/>the harness audit scored duplicate pairings 200% PASS,<br/>the drift counter policed two archived files forever, and<br/>a release candidate could assert fixture parity nothing compared<br/>run-all.sh:5"]

    WHY --> RUN["bash tests/gates/run-all.sh — four suites, one verdict<br/>run-all.sh:19"]
    RUN --> PRINT["GATE-TESTS-OK or GATE-TESTS-FAIL<br/>run-all.sh:34"]

    RUN --> S1["drift-counter.test.sh"]:::suite
    RUN --> S2["harness-audit.test.sh"]:::suite
    RUN --> S3["release-manifest.test.sh"]:::suite
    RUN --> S4["website-assets.test.sh"]:::suite

    S1 --> D1["seeded mismatch at a policed site goes red<br/>drift-counter.test.sh:83"]
    D1 --> D2["a policed file that has MOVED fails, never passes<br/>drift-counter.test.sh:102"]
    D2 --> D3["a policed pattern that no longer matches fails<br/>drift-counter.test.sh:111"]
    D3 --> D4["a sibling checkout off the pin fails<br/>drift-counter.test.sh:120"]
    D4 --> D5["allow-pin-drift downgrades that to a warning<br/>drift-counter.test.sh:129"]
    D5 --> D6["the COMMITTED allowlist passes with no unreadable site<br/>drift-counter.test.sh:137"]

    S2 --> H1["duplicate pairing edges de-duplicated<br/>harness-audit.test.sh:59"]
    H1 --> H2["a fabricated source path is NOT source-backed<br/>harness-audit.test.sh:71"]
    H2 --> H3["a real path resolves<br/>harness-audit.test.sh:88"]
    H3 --> H4["two controls sharing one file are one distinct source<br/>harness-audit.test.sh:98"]
    H4 --> H5["an unchecked-out substrate is unverifiable, not a pass<br/>harness-audit.test.sh:106"]
    H5 --> H6["a dangling edge cannot evidence coverage<br/>harness-audit.test.sh:116"]
    H6 --> H7["the committed templates still pass<br/>harness-audit.test.sh:126"]

    S3 --> R1["the roster covers every inventoried repository<br/>release-manifest.test.sh:27"]
    R1 --> R2["every repository declares provenance<br/>release-manifest.test.sh:46"]
    R2 --> R3["a candidate with no canonical revision set is REFUSED<br/>release-manifest.test.sh:70"]
    R3 --> R4["an unresolvable canonical revision set is refused<br/>release-manifest.test.sh:89"]
    R4 --> R5["output validates against the committed schema<br/>release-manifest.test.sh:125"]

    S4 --> W1["a missing required asset fails the build<br/>website-assets.test.sh:61"]
    W1 --> W2["a zero-byte required asset also fails<br/>website-assets.test.sh:77"]
    W2 --> WX["website asset gate and build receipt — see VF-02"]

    CI1["drift-counter.yml runs its suite BEFORE trusting the counter<br/>drift-counter.yml:75"]
    CI2["harness-fitness-gates.yml runs the audit suite first<br/>harness-fitness-gates.yml:44"]
    CI3["DIVERGENCE: run-all.sh itself is wired to NO workflow — it is<br/>reachable only through package.json's test:gates / verify scripts"]
```

## VF-05.12 mesh-smoke-preflight.sh — the read-only, unwired substrate probe
```mermaid
flowchart TB
    classDef ext fill:#f2eaf6,stroke:#7a4a9a,color:#222

    RUN["scripts/mesh-smoke-preflight.sh — no arguments, no services started,<br/>no Nostr events sent<br/>mesh-smoke-preflight.sh:337"]

    RUN --> H["require_file / require_dir raise PASS or MISSING;<br/>grep probes raise WARN; nothing is ever mutated<br/>mesh-smoke-preflight.sh:23"]

    RUN --> P1["VisionClaw — repo present, ADR-075 IS-Envelope contract present,<br/>did:nostr IRI parser found in src/<br/>mesh-smoke-preflight.sh:82"]:::ext
    RUN --> P2["nostr-rust-forum — governance.rs found and carries ALL of<br/>kinds 31400 to 31405; NIP-42 relay gate references; NIP-05 mode<br/>mesh-smoke-preflight.sh:118"]:::ext
    RUN --> P3["dreamlab-ai-website — forum-config/dreamlab.toml references<br/>governance kinds; standalone vs federated inferred from peer_relays<br/>mesh-smoke-preflight.sh:171"]:::ext
    RUN --> P4["agentbox — agentbox.toml federation mode, embedded relay port 7777,<br/>nostr-bridge/relay-consumer.js carrying governance kinds<br/>mesh-smoke-preflight.sh:253"]:::ext
    RUN --> P5["solid-pod-rs — a NIP-98 auth module or verify_schnorr reference;<br/>CORS support; defaults recorded as standalone<br/>mesh-smoke-preflight.sh:283"]:::ext

    P1 --> TAB["summary table: substrate, mounted, mesh-ready, default-mode<br/>mesh-smoke-preflight.sh:321"]
    P2 --> TAB
    P3 --> TAB
    P4 --> TAB
    P5 --> TAB
    TAB --> TOT["TOTALS: passed / failed / warnings<br/>mesh-smoke-preflight.sh:334"]

    TOT --> HAND["Output is pasted BY HAND into the<br/>Preflight Results section of the protocol doc<br/>mesh-smoke-test.md:71"]

    N1["DIVERGENCE: this script has no workflow, no exit-code contract used by CI,<br/>and no assertion of its own — FAIL counts are printed, then the script ends.<br/>It is an operator instrument, not a gate"]
    N2["EXTERNAL: every probe reads a sibling checkout at a fixed workspace path —<br/>VisionClaw VC-NN, nostr-rust-forum NF-NN, dreamlab-ai-website DW-NN,<br/>agentbox AB-NN, solid-pod-rs SP-NN. An unmounted sibling is MISSING, not fatal"]
```
