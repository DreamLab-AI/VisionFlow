---
id: ES-90
title: Supporting repositories, evidence boundaries and closeout admission
area: estate
governing:
  - docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md
  - docs/architecture/repository-map.md
adrs: [visionflow:ADR-2004, visionflow:ADR-2006, visionflow:ADR-2007, dream-engine:ADR-0003, WasmVOWL:ADR-001]
sources:
  - scripts/estate-health/roster.json
  - scripts/estate-doc-audit.py
  - scripts/diagram-index-gen.cjs
  - ../dream-machine/packages/compile/src/index.ts
  - ../dream-machine/packages/cli/src/index.ts
  - ../dream-machine/packages/cli/src/darwinBounds.ts
  - ../project/agentbox/services/dream-engine/src/engine.rs
  - ../WasmVOWL/modern/src/hooks/useWasmSimulation.ts
  - ../WasmVOWL/modern/package-lock.json
  - docs/estate-review/evidence/execution-2026-09-07/wasmvowl-schema-probe.log
  - ../prose-sanitiser/Cargo.toml
  - ../diagram-ir/Cargo.toml
  - ../loom/Cargo.toml
  - ../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/main.rs
  - docs/estate-review/2026-09-07-estate-audit.md
verified_commit: {visionflow: d4e44298646768a4b19af359119e16a6884fa80d, agentbox: 6a4ad132f2dc5ddaedd05c679fdd10066bf30a0f, WasmVOWL: 51a1484301901e817757c4cce17b1fece371652a, RuView: b48ab7dada5002414ec18d36a5d652e0f1bd4a67, loom: e39bb4d2b583040cd91346c3bbaf75411c3913b4, dream-machine: a82dab2bf3927cac5864306817f3c3094fe812d0, prose-sanitiser: be8305ee826e389a63dd72ae4b6994a533aff610, diagram-ir: eed5ebde490830fc143a744e299c2ec0ed8b0028}
---

## ES-90.1 Three scopes — health roster, source review and workspace neighbours
```mermaid
flowchart TB
    R["Declared health roster: 14 repositories<br/>scripts/estate-health/roster.json:5"] --> P["Nine primary diagram directories:<br/>VisionFlow, VisionClaw, Agentbox, pod, forum,<br/>website, knowledgeGraph, visionGraph, vowl-wasm"]
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
    participant C as compile prompt<br/>compile/src/index.ts:242
    participant A as agent session
    participant D as verify-entrypoint<br/>cli/src/index.ts:322
    participant B as checkDarwinBounds<br/>darwinBounds.ts:73
    participant R as Agentbox Rust service<br/>services/dream-engine/src/engine.rs
    C-->>A: parent and candidate evaluation rules, human promotion rule
    A->>D: explicit diagnostic invocation if selected
    D->>D: classify process exit and output liveness
    opt live Darwin output
        D->>B: parse rows, pass promotedLineages zero
        B-->>D: depth and candidate violations or ok
        Note over D,B: DIVERGENCE: CLI promotion count is unobserved,<br/>and this diagnostic is not the external nightly admission gate
    end
    Note over A,R: EXTERNAL: the Rust service has its own candidate/evaluator path.<br/>Toolkit tests do not identify the loaded Nix binary or prove a nightly run
```

## ES-90.3 WasmVOWL demo is a separate consumer from the published engine
```mermaid
sequenceDiagram
    participant H as useWasmSimulation<br/>useWasmSimulation.ts:39
    participant J as JSON graph
    participant P as Installed vowl-wasm 0.1.1 archive
    participant V as Other published engine consumers
    H->>J: serialise nodes and edges
    H->>P: loadOntology with graph JSON<br/>useWasmSimulation.ts:111
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
    F["Source paths and ADR metadata"] --> G["Structural/index gate: writeIndexes<br/>diagram-index-gen.cjs:715"]
    F --> C["Citation diagnostics; strict mode refuses warnings<br/>citeCheck, diagram-index-gen.cjs:488"]
    F --> SYM["Symbol-anchor check: warns when a cited line<br/>falls outside the labelled function's body<br/>symbolCheck, diagram-index-gen.cjs:616"]
    F --> R["Optional Mermaid rendering: renderAll<br/>diagram-index-gen.cjs:687"]
    F --> S["Semantic source review with file hashes<br/>scripts/estate-doc-audit.py"]
    G --> DOC["Documentation evidence"]
    C --> DOC
    SYM --> DOC
    R --> DOC
    S --> DOC
    ALLOW["citation-allowlist.json waives a citation<br/>that exists in no commit, with a reason;<br/>a stale entry fails the run<br/>loadAllowlist, diagram-index-gen.cjs:379"] --> DOC
    VER["VERIFICATION.md: declared-revisions anchor,<br/>records what actually resolved at HEAD<br/>writeVerificationReport, diagram-index-gen.cjs:440"] --> DOC
    TEST["Named local test and raw result"] --> LOCAL["Only the exercised contract"]
    RUN["Loaded binary, effective config,<br/>identity and cross-service receipt"] --> ACCEPT["Only the observed system journey"]
    DOC -.->|"does not establish"| ACCEPT
    LOCAL -.->|"does not establish"| ACCEPT
    NOTE["INVARIANT: default citations may read declared revisions;<br/>worktree-citations explicitly checks current source.<br/>Neither mode proves behaviour or deployment.<br/>revisionLines, diagram-index-gen.cjs:342,345"] --> DOC
```
