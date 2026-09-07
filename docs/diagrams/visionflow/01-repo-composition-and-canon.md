---
id: VF-01
title: Repo composition and the canon — document taxonomy, ADR lifecycle, index gate
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
  - docs/README.md
  - docs/architecture/repository-map.md
adrs: [ADR-2001, ADR-2002, ADR-2005, ADR-2006, ADR-2007]
sources:
  - ./README.md
  - docs/README.md
  - docs/BASELINE-visionflow.md
  - docs/adr/README.md
  - docs/adr/PREAMBLE.md
  - docs/adr/TEMPLATE.md
  - docs/adr/ADR-2001-corpus-consolidation.md
  - docs/adr/ADR-2002-static-copy-only-website.md
  - docs/adr/ADR-2003-pages-artifact-deploy.md
  - docs/adr/ADR-2004-diagram-baseline-vendored-render-gate.md
  - docs/adr/ADR-2005-drift-counter-allowlist-substrate-sourced.md
  - docs/adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md
  - docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md
  - docs/adr/ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md
  - docs/adr/ADR-2009-webgl-mesh-deep-is-sidecar-only.md
  - docs/archive/adr/README.md
  - scripts/adr-index-gen.cjs
  - .github/workflows/adr-index.yml
  - docs/architecture/repository-map.md
  - docs/architecture/compatibility-matrix.md
  - docs/architecture/licensing.md
  - docs/architecture/status-reconciliation.md
  - docs/architecture/pod-tier-matrix.md
  - docs/protocol/identity-spine.md
  - docs/terminology.md
  - docs/roadmap.md
  - docs/ecosystem-map.md
  - docs/releases/README.md
  - docs/releases/ecosystem-release.schema.json
  - docs/releases/candidate-2026-05-22.json
  - docs/registers/gap-register-v1.3.md
  - docs/registers/F9-federation-fork-record.md
  - docs/PRD-website.md
  - docs/site-verification.md
  - docs/closeout/final-design.md
  - docs/estate-review/README.md
  - docs/engineering/README.md
  - docs/engineering/ADR-004-harness-engineering-framework.md
  - docs/engineering/ADR-005-mandate-at-grant-governance.md
  - dream.config.json
verified_commit: bec06dc3a
---

## VF-01.1 Repo composition — two surfaces, and everything the repo deliberately is not
```mermaid
flowchart TB
    classDef ships fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef absent fill:#f7dede,stroke:#a33333,color:#111
    classDef doc fill:#e4ecf8,stroke:#3a5a8a,color:#111

    ROOT["DreamLab-AI/VisionFlow<br/>the ecosystem CANON repo"]

    ROOT --> W["website/ — surface 1, the one shipped artefact<br/>static marketing site for www.visionflow.info<br/>BASELINE-visionflow.md:45 — see VF-02"]:::ships
    ROOT --> D["docs/ — surface 2, governance and coordination canon<br/>registers, compatibility matrix, protocol, PRD/DDD<br/>BASELINE-visionflow.md:46"]:::doc
    ROOT --> S["scripts/ — node and bash generators plus gate evaluators<br/>adr-index-gen.cjs, estate-health.mjs, website-assets.mjs,<br/>diagram-index-gen.cjs, dream-*.sh"]:::ships
    ROOT --> GH[".github/workflows/ — nine workflows<br/>adr-index, deploy, estate-health, diagram-index, diagram-render,<br/>drift-counter, fixture-drift, copyright-guard, harness-fitness-gates"]:::ships
    ROOT --> DREAM["dream.config.json — nightly cycle contract<br/>dream.config.json:2 — see VF-04"]:::ships
    ROOT --> CONTENT["pitch/, presentation/, pdf-reports/, assets/,<br/>the-bubble-is-the-architecture*.md<br/>content the two surfaces publish<br/>BASELINE-visionflow.md:50"]:::doc

    ABSENT["INVARIANT: no server runtime, no database, no Rust code<br/>BASELINE-visionflow.md:52<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:30"]:::absent
    ROOT -.->|"the canon ships words and is graded on their accuracy"| ABSENT

    QUICK["./README.md:192 — 'VisionFlow has no application runtime of its own'<br/>two honest local actions: build the site; run the siblings from their own repos"]:::doc
    ROOT -.-> QUICK
```

## VF-01.2 Document taxonomy — which class a file belongs to, and who owns its truth
```mermaid
flowchart LR
    classDef norm fill:#dfeeda,stroke:#3f7a2f,color:#111
    classDef spec fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef reg fill:#f6efd8,stroke:#8a7020,color:#111
    classDef frozen fill:#ededed,stroke:#777,color:#111
    classDef other fill:#f2e6f5,stroke:#7a4a8a,color:#111

    subgraph LIVING["NORMATIVE — the living decision surface"]
        BASE["docs/BASELINE-visionflow.md<br/>doc_id VF-BASELINE, version 0.3.0<br/>'Invariants' section IS the compliance surface<br/>BASELINE-visionflow.md:216"]:::norm
        LED["docs/adr/ — thin ledger, 9 records ADR-2001..2009<br/>generated index + hand-written PREAMBLE + TEMPLATE<br/>docs/adr/README.md:27"]:::norm
    end

    subgraph SPECS["REQUIREMENT AND DESIGN SET"]
        PRD["docs/PRD-*.md — 5 records<br/>website, ecosystem-alignment, judgment-broker,<br/>gap-close-sprint, gap-close-canon<br/>docs/README.md:36"]:::spec
        DDD["docs/DDD-*.md — 5 bounded-context records<br/>website, ecosystem-alignment, judgment-broker,<br/>gap-close, gap-close-canon"]:::spec
        ARCH["docs/architecture/ — compatibility-matrix.md:1,<br/>repository-map.md:8, licensing.md:1,<br/>pod-tier-matrix.md:1, status-reconciliation.md:1"]:::spec
        PROTO["docs/protocol/identity-spine.md:6<br/>the shared did:nostr contract, plus mesh-smoke-test.md"]:::spec
    end

    subgraph EVID["REGISTERS, RELEASES AND EVIDENCE"]
        REG["docs/registers/ — gap-register v1.1 / v1.2 / v1.3<br/>plus the F9 fork record<br/>gap-register-v1.3.md:3 forward-chains, never edits in place"]:::reg
        REL["docs/releases/ — ./README.md:8 generator invocation,<br/>ecosystem-release.schema.json, candidate-2026-05-22.json<br/>manifest machinery — see VF-07"]:::reg
        CLOSE["docs/closeout/final-design.md:5 and docs/estate-closeout/<br/>dated audits — evidence of a day, not the living view"]:::reg
        REVIEW["docs/estate-review/README.md:2<br/>in-progress estate assessment + closeout programme CP-01..CP-09"]:::reg
    end

    subgraph WORDS["VOCABULARY AND DIRECTION"]
        TERM["docs/terminology.md:19<br/>canon for ontology / knowledge graph / reasoning / grounding"]:::other
        ROAD["docs/roadmap.md:7<br/>a pointer, not a second plan — ADR-2007 owns sequencing"]:::other
        MAP["docs/ecosystem-map.md:3<br/>sibling synthesis + gap register"]:::other
    end

    subgraph FROZEN["FROZEN — rationale only, never authority"]
        ARCHV["docs/archive/adr/ — ADR-001..007, frozen 2026-08-31<br/>archive/adr/README.md:3 'Do not add or edit records here'"]:::frozen
        ENG["docs/engineering/ — its OWN ADR-004 / ADR-005 sequence<br/>engineering/README.md:3 — a distinct namespace,<br/>explicitly out of the 2026-08-31 cut"]:::frozen
    end

    BASE --> LED
    LED -.->|"each record names its governing doc in frontmatter 'domain'"| BASE
    ARCHV -.->|"citable as evidence and history"| BASE
    ENG -.->|"number collision with archived canon ADR-004/005"| ARCHV
    EMPTY["docs/governance/ exists on disk and is EMPTY<br/>no governed content ships from it"]:::frozen
    WORDS -.-> EMPTY
```

## VF-01.3 Lookup and authority order — how a reader resolves a claim
```mermaid
flowchart TB
    classDef step fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef deny fill:#f7dede,stroke:#a33333,color:#111

    Q(["A claim about what VisionFlow is or runs"])
    Q --> S1["1. The governing doc for the domain<br/>docs/BASELINE-visionflow.md<br/>docs/adr/PREAMBLE.md:12"]:::step
    S1 --> S2["2. Its file:line citations into code and config<br/>the doc names them; follow them, do not trust the prose<br/>BASELINE-visionflow.md:38 ground-truth order"]:::step
    S2 --> S3["3. The ledger records that AMEND it<br/>docs/adr/README.md:29 index table"]:::step
    S3 --> S4["4. docs/archive/adr/ — RATIONALE AND HISTORY ONLY<br/>archive/adr/README.md:5 frozen because it drifted"]:::step
    S4 -.->|"NEVER as authority"| DENY["INVARIANT: legacy ADR prose is citable evidence,<br/>never a current build instruction<br/>BASELINE-visionflow.md:242"]:::deny

    ROUTE["Domain routing table — one row today<br/>docs/adr/PREAMBLE.md:10<br/>'What VisionFlow is, the static website,<br/>the canon/governance role, CI gates' -> BASELINE-visionflow.md"]:::step
    S1 --- ROUTE
```

## VF-01.4 ADR record lifecycle — three independent status axes
```mermaid
stateDiagram-v2
    direction LR
    state "decision_status" as DEC {
        [*] --> proposed
        proposed --> accepted
        proposed --> rejected
        accepted --> superseded
    }
    state "implementation_status" as IMP {
        [*] --> none
        none --> partial
        partial --> complete
    }
    state "activation_status" as ACT {
        [*] --> inactive
        inactive --> staged
        staged --> live
    }
    note right of DEC
      Enum enforced at adr-index-gen.cjs:31
      A value outside the set is a hard error,
      not a warning: adr-index-gen.cjs:122
    end note
    note right of IMP
      adr-index-gen.cjs:32
      ADR-2005 is 'partial' on purpose: the
      mechanism is live but two of four axes
      are unenforced
      ADR-2005-drift-counter-allowlist-substrate-sourced.md:46
    end note
    note right of ACT
      adr-index-gen.cjs:33
      ADR-2007 sits proposed/partial/staged
      ADR-2007-estate-closeout-evidence-roadmap.md:7
    end note
```

## VF-01.5 Supersession — reciprocity is checked, but only warned
```mermaid
flowchart LR
    classDef warn fill:#f9f0d5,stroke:#8a7020,color:#111
    classDef ok fill:#e6f0dc,stroke:#4a7a2a,color:#111

    A["ADR-A frontmatter<br/>supersedes: [ADR-B]<br/>TEMPLATE.md:8"]:::ok
    B["ADR-B frontmatter<br/>superseded_by: [ADR-A]<br/>TEMPLATE.md:9"]:::ok
    A -->|"forward edge"| B
    B -->|"back edge, required for a clean graph"| A

    CHK["Reciprocity walk over the real records only<br/>templates with id ADR-NNNN are skipped<br/>adr-index-gen.cjs:135"]
    A --> CHK
    CHK -->|"target absent from tree"| W1["warning: supersedes 'X' not present<br/>adr-index-gen.cjs:144"]:::warn
    CHK -->|"back edge missing"| W2["warning: reciprocity not recorded<br/>adr-index-gen.cjs:146"]:::warn
    CHK -->|"same id twice"| E1["ERROR duplicate id, exit 1<br/>adr-index-gen.cjs:138"]

    DIVERGE["DIVERGENCE: reciprocity is warn-only, so a one-sided<br/>supersession still passes the gate. Today the whole<br/>ledger is supersedes: [] / superseded_by: [] —<br/>docs/adr/README.md:31 shows every row as em-dash,<br/>so the edge case is untested in practice."]:::warn
    W2 -.-> DIVERGE
```

## VF-01.6 What adr-index-gen.cjs walks, validates and emits
```mermaid
flowchart TB
    classDef io fill:#e4ecf8,stroke:#3a5a8a,color:#111
    classDef check fill:#f6efd8,stroke:#8a7020,color:#111
    classDef out fill:#e6f0dc,stroke:#4a7a2a,color:#111

    CLI["node scripts/adr-index-gen.cjs docs/adr [--check]<br/>adr-index-gen.cjs:88"]:::io
    CLI --> WALK["walk(): recursive *.md,<br/>SKIPPING README.md and PREAMBLE.md<br/>adr-index-gen.cjs:78"]:::io
    WALK --> FM["parseFrontmatter(): minimal YAML —<br/>flat scalars plus inline [a, b] lists only<br/>adr-index-gen.cjs:45"]:::io

    FM --> V1["required fields present — 12 of them<br/>adr-index-gen.cjs:24 / :107"]:::check
    V1 --> V2["required scalars non-empty<br/>list fields supersedes/superseded_by exempt<br/>adr-index-gen.cjs:117"]:::check
    V2 --> V3["enum membership: decision / implementation /<br/>activation / repo where repo == visionflow<br/>adr-index-gen.cjs:34 / :122"]:::check
    V3 --> V4["supersedes and superseded_by must be lists<br/>adr-index-gen.cjs:128"]:::check
    V4 --> V5["unique ids across the tree<br/>adr-index-gen.cjs:138"]:::check
    V5 --> V6["reciprocity walk — warnings only<br/>adr-index-gen.cjs:141"]:::check

    V6 -->|"errors > 0"| FAIL["exit 1, README NOT generated<br/>adr-index-gen.cjs:158"]
    V6 -->|"--check and 0 errors"| CHECKOK["print 'ok: N ADR(s) valid', write nothing<br/>adr-index-gen.cjs:188"]:::out
    V6 -->|"no --check and 0 errors"| EMIT["emit docs/adr/README.md<br/>adr-index-gen.cjs:190"]:::out

    EMIT --> P1["header comment: GENERATED, DO NOT EDIT BY HAND<br/>docs/adr/README.md:1"]:::out
    EMIT --> P2["PREAMBLE.md inlined verbatim so the routing prose<br/>survives regeneration<br/>adr-index-gen.cjs:174"]:::out
    EMIT --> P3["record count line + 10-column table,<br/>rows sorted numerically by id<br/>adr-index-gen.cjs:164 / :178"]:::out

    EXEMPT["TEMPLATE.md is id ADR-NNNN: it must carry every key<br/>but is exempt from value-level checks and from the index<br/>adr-index-gen.cjs:39 / :111"]:::check
    V1 -.-> EXEMPT
```

## VF-01.7 The ADR frontmatter schema the generator enforces
```mermaid
erDiagram
    ADR_RECORD {
        string id "ADR-NNNN, unique; ADR-NNNN itself is the exempt template — adr-index-gen.cjs:39"
        string title "imperative one-liner — TEMPLATE.md:3"
        string date "YYYY-MM-DD — TEMPLATE.md:4"
        enum decision_status "proposed / accepted / rejected / superseded — TEMPLATE.md:5"
        enum implementation_status "none / partial / complete — TEMPLATE.md:6"
        enum activation_status "inactive / staged / live — TEMPLATE.md:7"
        list supersedes "must be a list — TEMPLATE.md:8"
        list superseded_by "must be a list — TEMPLATE.md:9"
        string verified_commit "sha at which implementation_status was established — TEMPLATE.md:10"
        string owner "accountable handle — TEMPLATE.md:11"
        string review_trigger "the event that forces re-review — TEMPLATE.md:12"
        enum repo "visionflow only — adr-index-gen.cjs:34"
    }
    ADR_BODY {
        section Context "max 10 lines; longer means it is two ADRs — TEMPLATE.md:19"
        section Decision "present-tense policy, specific enough to test compliance — TEMPLATE.md:22"
        section Consequences "costs and follow-on work, not just upside — TEMPLATE.md:26"
        section Verification "the command or artefact that established the status — TEMPLATE.md:30"
    }
    ADR_OPTIONAL {
        string domain "which governing doc this amends — ADR-2002-static-copy-only-website.md:14"
        string lineage "which legacy record it distils or reverses — ADR-2002-static-copy-only-website.md:15"
    }
    ADR_RECORD ||--|| ADR_BODY : carries
    ADR_RECORD ||--o| ADR_OPTIONAL : may carry
```

## VF-01.8 The adr-index.yml gate — two steps, one of them a drift check
```mermaid
sequenceDiagram
    autonumber
    participant PR as "Pull request touching docs/adr/** or the generator"
    participant GH as "GitHub Actions — adr-index.yml:1"
    participant NODE as "node 22 — adr-index.yml:29"
    participant GEN as "scripts/adr-index-gen.cjs"
    participant IDX as "docs/adr/README.md (committed artefact)"

    PR->>GH: "paths filter: docs/adr/**, scripts/adr-index-gen.cjs, the workflow itself"
    Note over GH: "adr-index.yml:12 — permissions are contents: read only<br/>adr-index.yml:18"
    GH->>NODE: actions/checkout@v4 then setup-node
    NODE->>GEN: "step 1 — adr-index-gen.cjs docs/adr --check"
    Note over GEN: "validates frontmatter, then exits 1 on any invalid record<br/>adr-index.yml:33"
    GEN-->>NODE: exit 0 or 1
    NODE->>GEN: "step 2 — regenerate WITHOUT --check"
    GEN->>IDX: rewrite the index in the working tree
    NODE->>IDX: "git diff --quiet -- docs/adr/README.md"
    alt index drifted from the records
        IDX-->>GH: "::error:: README.md is out of date, diff printed, exit 1"
        Note over GH: adr-index.yml:41
    else in sync
        IDX-->>GH: green
    end
    Note over PR,IDX: "INVARIANT: the index is a build artefact — never hand-edited.<br/>Prose changes go in PREAMBLE.md, which is inlined verbatim.<br/>docs/adr/README.md:1"
```

## VF-01.9 How a new decision lands — the four-part change
```mermaid
sequenceDiagram
    autonumber
    participant A as Author
    participant T as "docs/adr/TEMPLATE.md"
    participant R as "docs/adr/ADR-NNNN-slug.md"
    participant B as "docs/BASELINE-visionflow.md"
    participant G as "scripts/adr-index-gen.cjs"
    participant CI as "adr-index.yml"

    A->>T: "copy TEMPLATE.md to the next free number"
    Note over T: docs/adr/PREAMBLE.md:18
    A->>R: "fill the three-axis status honestly + verified_commit"
    A->>B: "update the governing doc IN THE SAME CHANGE"
    Note over B: "BASELINE change process: new file:line, confirm or amend<br/>the Invariant, bump version, re-record verified_commit<br/>BASELINE-visionflow.md:242"
    A->>G: "node scripts/adr-index-gen.cjs docs/adr"
    G-->>A: "ok: N ADR(s) valid, wrote docs/adr/README.md"
    A->>CI: open the PR
    CI-->>A: "--check plus index-sync diff, both must be green"
    Note over A,CI: "DIVERGENCE: nothing in CI enforces the 'update the governing<br/>doc in the same change' half — adr-index.yml checks only<br/>frontmatter validity and index freshness. The BASELINE<br/>coupling is a convention, not a gate."
```

## VF-01.10 The nine ledger records — domain, lineage and what each forecloses
```mermaid
flowchart TB
    classDef live fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef part fill:#f9f0d5,stroke:#8a7020,color:#111
    BASE["docs/BASELINE-visionflow.md — the governing doc every record names"]

    A1["ADR-2001 corpus consolidation<br/>accepted / complete / live<br/>archives ADR-001..007, creates the 2xxx ledger<br/>ADR-2001-corpus-consolidation.md:30"]:::live
    A2["ADR-2002 copy-only static website<br/>reverses legacy ADR-001 D1/D3 Rust+WASM<br/>ADR-2002-static-copy-only-website.md:30"]:::live
    A3["ADR-2003 Pages artifact deploy<br/>reverses legacy ADR-001 D4 gh-pages branch push<br/>ADR-2003-pages-artifact-deploy.md:29"]:::live
    A4["ADR-2004 committed diagram baseline, vendored Mermaid<br/>diverges from legacy ADR-005 D3<br/>ADR-2004-diagram-baseline-vendored-render-gate.md:30"]:::live
    A5["ADR-2005 drift counter, allowlist-anchored, fail-open per axis<br/>implementation_status PARTIAL — 2 of 4 axes unenforced<br/>ADR-2005-drift-counter-allowlist-substrate-sourced.md:30"]:::part
    A6["ADR-2006 canon-only — cross-repo view, never substrate truth<br/>makes BASELINE Invariant 4 a standing constraint<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:32"]:::live
    A7["ADR-2007 estate closeout evidence roadmap<br/>proposed / partial / staged — the roadmap, not the work<br/>ADR-2007-estate-closeout-evidence-roadmap.md:24"]:::part
    A8["ADR-2008 estate health collected by CI, read by the dream cycle<br/>extends ADR-2006 — see VF-03<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:34"]:::live
    A9["ADR-2009 webgl-mesh deep is sidecar-only<br/>applies the ADR-2008 pattern, parks a rotation slot — see VF-04<br/>ADR-2009-webgl-mesh-deep-is-sidecar-only.md:36"]:::live

    BASE --- A1
    BASE --- A2
    BASE --- A3
    BASE --- A4
    BASE --- A5
    BASE --- A6
    BASE --- A7
    A6 -->|"lineage: extends"| A8
    A8 -->|"lineage: applies the same pattern"| A9
    A2 -->|"the build ADR-2003 publishes"| A3
```

## VF-01.11 The canon boundary — what VisionFlow may and may not assert
```mermaid
flowchart LR
    classDef may fill:#e6f0dc,stroke:#4a7a2a,color:#111
    classDef mayn fill:#f7dede,stroke:#a33333,color:#111

    VF["VisionFlow canon"]
    VF --> M1["MAY own: the compatibility matrix<br/>compatibility-matrix.md:6"]:::may
    VF --> M2["MAY own: the release-evidence manifest<br/>releases/README.md:3 — see VF-07"]:::may
    VF --> M3["MAY own: the shared maturity vocabulary<br/>ADR-002 ladder, read from the substrates' own fields<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:36"]:::may
    VF --> M4["MAY report observed signals: CI conclusion, release tag, HTTP status<br/>ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md:76"]:::may

    VF -.-> N1["MAY NOT hold substrate implementation<br/>no server, DB or Rust here<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:30"]:::mayn
    VF -.-> N2["MAY NOT overwrite a substrate's own status<br/>repo-local docs stay authoritative for their own code"]:::mayn
    VF -.-> N3["MAY NOT publish a maturity claim above its evidence tier<br/>that is a governance DEFECT, not a footnote<br/>BASELINE-visionflow.md:227"]:::mayn
    VF -.-> N4["MAY NOT assert a count without one queryable source<br/>policed by the ADR-2005 drift gate<br/>BASELINE-visionflow.md:231"]:::mayn

    EXT["EXTERNAL: the substrates own their own implementation truth —<br/>VisionClaw VC-01..VC-37, agentbox AB-01..AB-28,<br/>solid-pod-rs SP-nn, nostr-rust-forum NF-nn,<br/>dreamlab-ai-website DW-nn, knowledgeGraph KG-nn,<br/>vowl-wasm VW-nn, visionGraph VG-nn, estate ES-01..ES-10"]
    N2 -.-> EXT
```

## VF-01.12 Known divergences the BASELINE itself records — the canon's own defect list
```mermaid
flowchart TB
    classDef drift fill:#f7dede,stroke:#a33333,color:#111
    classDef settled fill:#e6f0dc,stroke:#4a7a2a,color:#111

    SRC["Ground truth: website/build.sh is copy-only<br/>ADR-2002-static-copy-only-website.md:30"]:::settled

    D1["DOC-DRIFT: ./README.md:194 still says<br/>'The site is a Rust/WASM build under website/'<br/>and ./README.md:198 'wasm-pack builds both WASM crates'<br/>recorded open at BASELINE-visionflow.md:166"]:::drift
    D2["DOC-DRIFT: site-verification.md:10 'builds both WASM crates'<br/>and site-verification.md:17 'wasm-pack release builds complete'"]:::drift
    D3["DOC-DRIFT: PRD-website.md:120 still specifies two required<br/>WASM modules, mesh-hero and particle-field,<br/>with PRD-website.md:128 a 300 KB gzip WASM budget"]:::drift
    SRC -.->|"contradicted by"| D1
    SRC -.->|"contradicted by"| D2
    SRC -.->|"contradicted by"| D3

    S1["SETTLED: Tailwind Play CDN never shipped — local CSS ratified<br/>BASELINE-visionflow.md:175"]:::settled
    S2["SETTLED: gh-pages branch push superseded by the Pages actions<br/>BASELINE-visionflow.md:179"]:::settled

    O1["OPEN: ontology-bridge tool count shows two figures<br/>in one README diagram — the exact re-drift the ADR-2005<br/>gate exists to catch<br/>BASELINE-visionflow.md:182"]:::drift
    O2["OPEN: dual ADR numbering hazard — docs/engineering/<br/>carries its own ADR-004 and ADR-005<br/>BASELINE-visionflow.md:200<br/>engineering/ADR-005-mandate-at-grant-governance.md:3 self-labels speculative"]:::drift
    O3["OPEN: legacy ADR-007 governance-loop closure is specified,<br/>not shipped; cross-repo, tracked here not fixed here<br/>BASELINE-visionflow.md:206"]:::drift
    O4["OPEN: the diagram-render gate shipped under a different design<br/>than legacy ADR-005 D3 named — intent honoured,<br/>named artefact absent<br/>BASELINE-visionflow.md:190"]:::drift

    DOCDRIFT5["DOC-DRIFT: BASELINE-visionflow.md:6 pins verified_commit 2daa995<br/>while its own ADR-2008 subsection at BASELINE-visionflow.md:160<br/>says those facts are verified at a later landing commit —<br/>one document carrying two verification epochs"]:::drift
```
