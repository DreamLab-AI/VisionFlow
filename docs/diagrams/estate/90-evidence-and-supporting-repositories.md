---
id: ES-90
title: Supporting repositories, evidence boundaries and closeout admission
area: estate
governing:
  - docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md
  - docs/architecture/repository-map.md
adrs: [visionflow:ADR-2004, visionflow:ADR-2006, visionflow:ADR-2007, dream-engine:ADR-0003, dream-engine:ADR-0005, WasmVOWL:ADR-001]
sources:
  - scripts/estate-health/roster.json
  - scripts/estate-doc-audit.py
  - scripts/diagram-index-gen.cjs
  - ../dream-machine/packages/compile/src/index.ts
  - ../dream-machine/packages/cli/src/index.ts
  - ../dream-machine/packages/cli/src/darwinBounds.ts
  - ../dream-machine/scripts/darwin-entrypoint.sh
  - ../dream-machine/dream.config.json
  - ../project/agentbox/services/dream-engine/src/engine.rs
  - ../WasmVOWL/modern/src/hooks/useWasmSimulation.ts
  - ../WasmVOWL/modern/package-lock.json
  - docs/estate-review/evidence/execution-2026-09-07/wasmvowl-schema-probe.log
  - ../prose-sanitiser/Cargo.toml
  - ../diagram-ir/Cargo.toml
  - ../loom/Cargo.toml
  - ../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/main.rs
  - docs/estate-review/2026-09-07-estate-audit.md
verified_commit: {visionflow: e5987acc8337ddd64c72f775750d61fef46d8e0b, agentbox: 5ab197a9d49e9721b85b791bf9efe30842c9e047, WasmVOWL: 51a1484301901e817757c4cce17b1fece371652a, RuView: b48ab7dada5002414ec18d36a5d652e0f1bd4a67, loom: 8c618faf24950ad4ef70855308991da56a54af2c, dream-machine: 824993cabdbc02dc1715084e0ba165bba3e2b3d6, prose-sanitiser: b134ddb58b2f2a753244e1a2c3e4b9513d8195d3, diagram-ir: eed5ebde490830fc143a744e299c2ec0ed8b0028}
---

## ES-90.1 Three scopes — health roster, source review and workspace neighbours
```mermaid
flowchart TB
    R["Declared health roster: 15 repositories<br/>scripts/estate-health/roster.json:5-19"] --> P["Ten primary diagram directories:<br/>VisionFlow, VisionClaw, Agentbox, pod, forum,<br/>website, knowledgeGraph, visionGraph, vowl-wasm,<br/>sidestr-rs, rostered at roster.json:13"]
    R --> S["Also on roster: Loom, Dream Engine,<br/>WasmVOWL, prose-sanitiser, diagram-ir"]
    C["Git census and ADR candidate discovery<br/>scripts/estate-doc-audit.py"] --> R
    C --> D["Consumed or historical extension:<br/>RuVector, RuView, archived Logseq"]
    C --> N["Other checkouts: inventory only,<br/>no estate adoption from directory proximity"]
    P --> E["DIVERGENCE: a count of diagrams or repositories<br/>does not measure source coverage or deployment acceptance"]
    S --> E
    D --> E
```

## ES-90.2 Dream compiler, diagnostic and runtime are separate paths
```mermaid
sequenceDiagram
    participant C as compile prompt<br/>compile/src/index.ts:268-269
    participant A as agent session
    participant D as verify-entrypoint<br/>cli/src/index.ts:323
    participant E as darwin evaluator<br/>scripts/darwin-entrypoint.sh:78
    participant B as checkDarwinBounds<br/>darwinBounds.ts:75
    participant R as Agentbox Rust service<br/>services/dream-engine/src/engine.rs
    C-->>A: parent and candidate evaluation rules, human promotion rule
    A->>E: the configured darwin evaluator, dream.config.json:55
    E->>E: refuse exit 64 unless darwin is exact-pinned and<br/>sandboxed mock or agent, scripts/darwin-entrypoint.sh:57-67
    E->>D: verify-entrypoint darwin --passthrough, scripts/darwin-entrypoint.sh:78
    D->>D: run darwin, echo its output, classify liveness, cli/src/index.ts:347-354
    opt live Darwin output
        D->>B: parse rows, pass promotedLineages zero
        B-->>D: violations against 3 generations, 5 candidates,<br/>1 lineage, darwinBounds.ts:43-48
        D-->>E: exit 3 on a breach
    end
    E-->>A: exit status IS the evaluator outcome, so a breach is a<br/>FAILED required evaluator, scripts/darwin-entrypoint.sh:18-24
    Note over D,B: CHANGED 2026-10-02 (ADR-0005) — the check was a hand-run diagnostic<br/>and is now the darwin evaluator itself, and the bound rose from four<br/>candidates per generation to five to match darwin 0.10.2's five-surface<br/>map, darwinBounds.ts:46. Promotion count is still unobserved: the leaderboard<br/>has no promotion line, so zero is passed explicitly
    Note over E,R: EXTERNAL: enforcement still depends on the annexe runner honouring a<br/>failed required evaluator. This repo makes the failure, it does not veto the night
    Note over A,R: EXTERNAL: the Rust service has its own candidate/evaluator path.<br/>Toolkit tests do not identify the loaded Nix binary or prove a nightly run
```

## ES-90.3 WasmVOWL demo is a separate consumer from the published engine
```mermaid
sequenceDiagram
    participant H as useWasmSimulation<br/>useWasmSimulation.ts:36
    participant J as JSON graph
    participant P as Installed vowl-wasm 0.1.1 archive
    participant V as Other published engine consumers
    H->>J: serialise nodes and edges
    H->>P: loadOntology with graph JSON<br/>useWasmSimulation.ts:105
    P->>J: require class and property arrays
    J-->>P: required arrays missing
    P-->>H: parse error
    Note over H,P: DIVERGENCE: remote extraction adds position accessors,<br/>but does not repair this input schema.<br/>Actual installed WASM probe fails nodes/edges and accepts class/property.<br/>wasmvowl-schema-probe.log:1-3
    Note over V: EXTERNAL: visionGraph and knowledgeGraph consumers are separate.<br/>Their package identity and rendering must be traced independently, see ES-11
```

## ES-90.4 Tool packages and sensing remain distinct acceptance surfaces
```mermaid
flowchart TB
    PS["prose-sanitiser workspace<br/>prose-sanitiser/Cargo.toml:18 members<br/>core, unicode, UK, slop, media, CLI, server"] --> PT["Tool capability, not proof every publisher invokes it"]
    DI["diagram-ir Cargo.toml<br/>diagram extraction and self-check binaries"] --> PT
    L["Loom Cargo.toml<br/>sibling RuVector core path dependency"] --> INPUT["Declared dependency input must match<br/>the reproducible build and served generation"]
    RV["RuView sensing server<br/>main.rs:2675"] --> SIM["No hardware detected can select simulation"]
    SIM --> BOUND["INVARIANT: simulation output is not hardware sensing evidence"]
    PT --> BOUND
    INPUT --> BOUND
```

## ES-90.5 Evidence types do not imply one another
```mermaid
flowchart TB
    F["Source paths and ADR metadata"] --> G["Structural/index gate: writeIndexes<br/>diagram-index-gen.cjs:716"]
    F --> C["Citation diagnostics; strict mode refuses warnings<br/>citeCheck, diagram-index-gen.cjs:489"]
    F --> SYM["Symbol-anchor check: warns when a cited line<br/>falls outside the labelled function's body<br/>symbolCheck, diagram-index-gen.cjs:617"]
    F --> R["Optional Mermaid rendering: renderAll<br/>diagram-index-gen.cjs:688"]
    F --> S["Semantic source review with file hashes<br/>scripts/estate-doc-audit.py"]
    G --> DOC["Documentation evidence"]
    C --> DOC
    SYM --> DOC
    R --> DOC
    S --> DOC
    ALLOW["citation-allowlist.json waives a citation<br/>that exists in no commit, with a reason;<br/>a stale entry fails the run<br/>loadAllowlist, diagram-index-gen.cjs:380"] --> DOC
    VER["VERIFICATION.md: declared-revisions anchor,<br/>records what actually resolved at HEAD<br/>writeVerificationReport, diagram-index-gen.cjs:441"] --> DOC
    TEST["Named local test and raw result"] --> LOCAL["Only the exercised contract"]
    RUN["Loaded binary, effective config,<br/>identity and cross-service receipt"] --> ACCEPT["Only the observed system journey"]
    DOC -.->|"does not establish"| ACCEPT
    LOCAL -.->|"does not establish"| ACCEPT
    NOTE["INVARIANT: default citations may read declared revisions;<br/>worktree-citations explicitly checks current source.<br/>Neither mode proves behaviour or deployment.<br/>revisionLines, diagram-index-gen.cjs:343,346"] --> DOC
```
