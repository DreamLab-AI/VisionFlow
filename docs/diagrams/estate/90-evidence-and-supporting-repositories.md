---
id: ES-90
title: Supporting repositories, evidence boundaries and closeout admission
area: estate
governing:
  - docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md
  - docs/architecture/repository-map.md
adrs: [visionflow:ADR-2020, visionflow:ADR-2004, visionflow:ADR-2006, visionflow:ADR-2007, dream-machine:ADR-0003, dream-machine:ADR-0005, dream-machine:ADR-0006, dream-machine:ADR-0007, WasmVOWL:ADR-001]
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
  - ../dream-engine/crates/dream-core/src/readiness.rs
  - ../dream-engine/Cargo.toml
  - ../project/agentbox/services/dream-engine/Cargo.toml
  - ../WasmVOWL/modern/src/hooks/useWasmSimulation.ts
  - ../WasmVOWL/modern/package-lock.json
  - docs/estate-review/evidence/execution-2026-09-07/wasmvowl-schema-probe.log
  - ../prose-sanitiser/Cargo.toml
  - ../diagram-ir/Cargo.toml
  - ../loom/Cargo.toml
  - ../RuView/v2/crates/wifi-densepose-sensing-server/src/main.rs
  - docs/estate-review/2026-09-07-estate-audit.md
verified_commit: {visionflow: ff15faf83d82fef7f23d96da381c8ab78d0206b0, agentbox: 7208f7e0d749b44efcfe24bd503c567c9ff8f9bc, dream-engine: 59a3e215e826aa1e9ebe157d49c1ac504f7f39c9, WasmVOWL: 51a1484301901e817757c4cce17b1fece371652a, RuView: 5728b90949e60e3f908d996997d453147c1794e0, loom: 5ad860863b588c0cdb2290e31c69e39d82755aeb, dream-machine: b77093e564ef689923fb523b0e0b18d2bb4a21a4, prose-sanitiser: b261f65466b12bef2fd7c4957cee463cbe06190f, diagram-ir: 15cba163da7086a3ebbb5a059c8a8d0ab4ba9706}
---

## ES-90.1 Three scopes — health roster, source review and workspace neighbours
```mermaid
flowchart TB
    R["Declared health roster: 18 repositories<br/>scripts/estate-health/roster.json:5-22"] --> P["Primary diagram directories include:<br/>VisionFlow, VisionClaw, Agentbox, pod, forum,<br/>website, knowledgeGraph, visionGraph, vowl-wasm,<br/>sidestr-rs (roster.json:13), ontomark (roster.json:14),<br/>visiontouch-rs (roster.json:22), Dream Engine (roster.json:21)"]
    R --> S["Also on roster: Loom, ontomark-obsidian, WasmVOWL,<br/>prose-sanitiser, diagram-ir. The Dream Engine row still<br/>describes a fork of ruvnet/dream-machine, roster.json:21<br/>(see DE-01)"]
    C["Git census and ADR candidate discovery<br/>scripts/estate-doc-audit.py"] --> R
    C --> D["Consumed or historical extension:<br/>RuVector, RuView, archived Logseq"]
    C --> N["Other checkouts: inventory only,<br/>no estate adoption from directory proximity"]
    P --> E["DIVERGENCE: a count of diagrams or repositories<br/>does not measure source coverage or deployment acceptance"]
    S --> E
    D --> E
```

## ES-90.2 Archived TypeScript fork: compiler, diagnostic and runtime are separate paths
```mermaid
sequenceDiagram
    participant C as compile prompt<br/>compile/src/index.ts:268-270
    participant A as agent session
    participant D as verify-entrypoint<br/>cli/src/index.ts:323
    participant E as darwin evaluator<br/>scripts/darwin-entrypoint.sh:197
    participant B as checkDarwinBounds<br/>darwinBounds.ts:104
    participant R as Agentbox Rust service<br/>services/dream-engine/src/engine.rs
    C-->>A: parent-then-candidate evaluation rule and the<br/>promotion gate, compile/src/index.ts:248-249
    A->>E: the configured darwin evaluator, dream.config.json:55
    E->>E: refuse exit 64 unless the head is exactly npx plus one exact pin, no<br/>dash argument precedes it and no tracked flag repeats, the sandbox is mock<br/>or agent, and the ruvllm mutator names its Loom door,<br/>scripts/darwin-entrypoint.sh:57-135
    E->>D: verify-entrypoint darwin --passthrough, scripts/darwin-entrypoint.sh:197
    D->>D: run darwin, echo its output under --passthrough, classify liveness,<br/>cli/src/index.ts:347-355
    opt live Darwin output
        D->>B: parse rows, pass promotedLineages zero,<br/>cli/src/index.ts:369-383
        B-->>D: violations against 3 generations, 5 candidates,<br/>1 lineage, darwinBounds.ts:61-66, and score uniformity<br/>with realMutation count, darwinBounds.ts:190-197
        D-->>E: exit 3 on a bound breach, an unparsable leaderboard,<br/>or uniform scores with zero real shim edits, cli/src/index.ts:405-412
    end
    E-->>A: exit status IS the evaluator outcome, so a breach is a<br/>FAILED required evaluator, scripts/darwin-entrypoint.sh:26-37,199
    Note over E,R: CHANGED 2026-10-06 (ADR-0006, ADR-0007) — with --mutator ruvllm the<br/>mutator is routed through a loopback Loom shim that outlasts darwin's<br/>30 s fetch abort, scripts/darwin-entrypoint.sh:157-184, and a uniform<br/>leaderboard then fails only when the shim counted zero real model edits
    Note over E: CHANGED 2026-10-08 (ADR-0008, named in the script only) — the command head<br/>must be exactly npx plus the pin, every dash argument before the pin is refused<br/>and a repeated sandbox, mutator or URL flag is refused,<br/>scripts/darwin-entrypoint.sh:56-100,113-122
    Note over D,B: CHANGED 2026-10-02 (ADR-0005) — the check was a hand-run diagnostic<br/>and is now the darwin evaluator itself, and the bound rose from four<br/>candidates per generation to five to match darwin 0.10.2's five-surface<br/>map, darwinBounds.ts:63-64. Promotion count is still unobserved: the leaderboard<br/>has no promotion line, so zero is passed explicitly, cli/src/index.ts:369-374
    Note over E,R: EXTERNAL: enforcement still depends on the annexe runner honouring a<br/>failed required evaluator. This repo makes the failure, it does not veto the night
    Note over A,R: EXTERNAL: the Rust judgement and loop moved to DreamLab-AI/dream-engine<br/>(DE-01, public with five crates at 0.1.0) and the estate adapter is AB-23.<br/>Toolkit tests do not identify the loaded Nix binary or prove a nightly run
```

**Drift:** the toolkit this diagram traces is no longer the estate's dream engine. The previous TypeScript fork is renamed DreamLab-AI/dream-machine and archived, and the slug DreamLab-AI/dream-engine now belongs to the Rust repository that holds the extracted judgement and loop crates (mapped in DE-01; its workspace declares dream-core and dream-cli at 0.4.0, `../dream-engine/Cargo.toml:11`, with dream-gate at 0.3.0, `../dream-engine/crates/dream-gate/Cargo.toml:3`; the agentbox adapter pins `dream-core =0.4.0` and `dream-gate =0.3.0`, `../project/agentbox/services/dream-engine/Cargo.toml:26-27`). The Darwin path drawn above is not carried into it: the Rust core keeps only a readiness refusal for a darwin entrypoint run without a sandbox flag, a safety check that appends a problem to the admission report and mutates nothing (`../dream-engine/crates/dream-core/src/readiness.rs:228-232`). The estate adapter is AB-23.

**Drift:** `darwin-entrypoint.sh` cites ADR-0008 for the exact `npx <pin>` head rule (`../dream-machine/scripts/darwin-entrypoint.sh:36,113`) while the archived repository's decision records stop at ADR-0007 (`../dream-machine/docs/adrs/ADR-0007-darwin-mutator-loom-shim.md:1`); the toolkit paths above were re-read at `b77093e`, where only the entrypoint script among them had changed since `6426583`.

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
    RV["RuView sensing server, now under v2/crates<br/>plan_source main.rs:5415"] --> SIM["auto with no real source serves simulated data and keeps the<br/>UDP receiver bound until the first real frame, main.rs:5441-5450;<br/>explicit simulated binds no UDP, main.rs:5454-5459"]
    SIM --> BOUND["INVARIANT: simulation output is not hardware sensing evidence"]
    PT --> BOUND
    INPUT --> BOUND
```

## ES-90.5 Evidence types do not imply one another
```mermaid
flowchart TB
    F["Source paths and ADR metadata"] --> G["Structural/index gate: writeIndexes<br/>diagram-index-gen.cjs:722"]
    F --> C["Citation diagnostics; strict mode refuses warnings<br/>citeCheck, diagram-index-gen.cjs:500"]
    F --> SYM["Symbol-anchor check: warns when a cited line<br/>falls outside the labelled function's body<br/>symbolCheck, diagram-index-gen.cjs:634"]
    F --> JEV["Opt-in advisory per-citation review (--jev with --cite-check), off by default,<br/>stderr only, never changes an exit code or an index; one Noul per resolved citation,<br/>flagged only for jev-* models at p_defect 0.9 or more<br/>jevCitationReview, diagram-index-gen.cjs:805"]
    F --> R["Optional Mermaid rendering with pinned Pele, no browser: renderAll<br/>diagram-index-gen.cjs:698"]
    F --> S["Semantic source review with file hashes<br/>scripts/estate-doc-audit.py"]
    G --> DOC["Documentation evidence"]
    C --> DOC
    SYM --> DOC
    R --> DOC
    JEV -.->|"a pointer for a human reviewer, not evidence"| DOC
    S --> DOC
    ALLOW["citation-allowlist.json waives a citation<br/>that exists in no commit, with a reason;<br/>a stale entry fails the run<br/>loadAllowlist, diagram-index-gen.cjs:387"] --> DOC
    VER["VERIFICATION.md: declared-revisions anchor,<br/>records what actually resolved at HEAD<br/>writeVerificationReport, diagram-index-gen.cjs:452"] --> DOC
    TEST["Named local test and raw result"] --> LOCAL["Only the exercised contract"]
    RUN["Loaded binary, effective config,<br/>identity and cross-service receipt"] --> ACCEPT["Only the observed system journey"]
    DOC -.->|"does not establish"| ACCEPT
    LOCAL -.->|"does not establish"| ACCEPT
    NOTE["INVARIANT: default citations may read declared revisions;<br/>worktree-citations explicitly checks current source.<br/>Neither mode proves behaviour or deployment.<br/>revisionLines, diagram-index-gen.cjs:350,353"] --> DOC
```
