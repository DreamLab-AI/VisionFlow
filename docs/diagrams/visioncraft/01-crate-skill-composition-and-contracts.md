---
id: VK-01
title: Crate, skill and contract composition
area: visioncraft
governing:
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
  - ../visioncraft/docs/adr/0002-specialist-upstream-boundaries.md
  - ../visioncraft/skills/visioncraft/references/architecture.md
adrs: [visioncraft:ADR-0001, visioncraft:ADR-0002]
sources:
  - ../visioncraft/Cargo.toml
  - ../visioncraft/src/lib.rs
  - ../visioncraft/src/main.rs
  - ../visioncraft/src/plan.rs
  - ../visioncraft/src/process.rs
  - ../visioncraft/src/deepsec.rs
  - ../visioncraft/skills/visioncraft/SKILL.md
  - ../visioncraft/skills/visioncraft/references/architecture.md
  - ../visioncraft/skills/visioncraft/assets/provenance.json
  - ../visioncraft/contracts/deepsec.json
  - ../visioncraft/contracts/agentic-qe.json
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
  - ../visioncraft/docs/adr/0002-specialist-upstream-boundaries.md
  - ../visioncraft/docs/MIGRATION.md
verified_commit: 1067a5f59afad7e2c8780bce6b29967d1130c19d
---

## VK-01.1 Three surfaces and who owns what

```mermaid
flowchart LR
    SK["Skill: task selection, evidence discipline<br/>architecture.md:12"]
    PL["Quality plan: declared checks, subject, expectations<br/>architecture.md:13"]
    RN["Explicit runner: bounded executions, receipts<br/>architecture.md:14"]
    GT["Gate evaluator: deterministic aggregation<br/>architecture.md:15"]
    SM["Sealmap: finding dispositions, family declarations<br/>architecture.md:16"]
    HC["Host or CI: tools, credentials, spend, deployment<br/>architecture.md:17"]
    SK --> PL
    PL --> RN
    RN --> GT
    SM -. "types reused, not a parallel store<br/>ADR 0001:45-48" .-> GT
    HC -. "supplies tools and policy" .-> RN
    N1["INVARIANT: reading the skill starts no tools and authorises no external call<br/>SKILL.md:17-19"]
    N2["INVARIANT: the core reads no files, clocks, processes or network<br/>lib.rs:3-5"]
    N3["INVARIANT: a pass means the recorded policy was satisfied, not that the program is secure or deployed<br/>lib.rs:4-5"]
    SK --- N1
    GT --- N2
    GT --- N3
```

Three surfaces: guidance, versioned library and CLI, and host integration. The table of owners is the authored contract; each row says what it does not establish.

## VK-01.2 Cargo features layer the core, the runner and the CLI

```mermaid
flowchart TB
    DEF["default = cli + process<br/>Cargo.toml:14"]
    CLI["feature cli: clap, url<br/>Cargo.toml:15"]
    PRC["feature process: tempfile, nix, signal-hook<br/>Cargo.toml:16"]
    CORE["always built: evaluate, evidence, plan<br/>lib.rs:21-23"]
    MP["pub mod process<br/>lib.rs:121"]
    MD["pub mod deepsec<br/>lib.rs:125"]
    MC["pub mod cli<br/>lib.rs:129"]
    MG["mod git_metadata<br/>lib.rs:132"]
    BIN["bin visioncraft requires cli<br/>Cargo.toml:21"]
    SMC["sealmap-corpus 0.4.0, unconditional dependency<br/>Cargo.toml:28"]
    DEF --> CLI
    DEF --> PRC
    CLI --> PRC
    PRC --> MP
    CLI --> MD
    CLI --> MC
    CLI --> MG
    CLI --> BIN
    CORE --- SMC
    N1["INVARIANT: unsafe_code is forbidden crate-wide<br/>Cargo.toml:39"]
    N2["TENSION: non-Linux builds still compile deepsec and cli, but run_command returns Unsupported<br/>process.rs:631"]
    N3["see VK-03.1"]
    MP --- N2
    BIN --- N1
    BIN --- N3
```

The pure library is `default-features = false`; the Deepsec adapter hangs off `cli`, not `process`. Native execution is Linux only; elsewhere the adapter exits 78 with a message (deepsec.rs:662).

## VK-01.3 Typed plan: the one input every other component binds to

```mermaid
flowchart LR
    QP["QualityPlan v1<br/>plan.rs:234"]
    SJ["Subject: repository, revision, configuration, dirty, inputs<br/>plan.rs:20"]
    CS["CheckSpec: id, required, invocation, evaluator, scope<br/>plan.rs:159"]
    EX["Expectation: claim, category, checks, stabilizer, audit<br/>plan.rs:197"]
    GP["GatePolicy: max_age_ms, require_observed_scope<br/>plan.rs:224"]
    IV["Invocation: argv, cwd, env names, timeout, output bound<br/>plan.rs:88"]
    EV["Evaluator: exit-code or reported adapter<br/>plan.rs:143"]
    EC["EvidenceCategory: executable, partial, manual<br/>plan.rs:175"]
    ST["Stabilizer: check_id plus test_id<br/>plan.rs:187"]
    QP --> SJ
    QP --> CS
    QP --> EX
    QP --> GP
    CS --> IV
    CS --> EV
    EX --> EC
    EX --> ST
    A1["Admission ceilings: 1024 checks, one day of deadlines, one GiB of capture<br/>plan.rs:11-15"]
    A2["INVARIANT: at least one required check, scope must name subject inputs<br/>plan.rs:303-306"]
    A3["INVARIANT: positive control must be a distinct declared check<br/>plan.rs:384-389"]
    A4["INVARIANT: partial or manual evidence needs a stated limitation<br/>plan.rs:390-395"]
    A5["digest = sha256 of the validated typed plan, a content binding not a signature<br/>plan.rs:407-411"]
    QP --- A1
    QP --- A2
    EX --- A3
    EX --- A4
    QP --- A5
```

Parsing a plan never runs it. Every receipt and assessment later carries this digest, which is how a changed argv or policy cannot reuse an old pass (see VK-02.3).

## VK-01.4 Versioned upstream contracts for Deepsec and Agentic QE

```mermaid
flowchart TB
    DJ["contracts/deepsec.json<br/>adapter_protocol_version deepsec-cli-2.3.10-v1<br/>deepsec.json:3"]
    DV["supported_versions: 2.3.10 only<br/>deepsec.json:4-6"]
    DU["unsupported version: unavailable, exit 78, never satisfies a required gate<br/>deepsec.json:7-11"]
    IC["include_str of the contract at compile time<br/>deepsec.rs:22-25"]
    SV["supported_version() reads package.version<br/>deepsec.rs:26-30"]
    CK["runtime check: installed version must equal the contract<br/>deepsec.rs:859-864"]
    AJ["contracts/agentic-qe.json<br/>protocol agentic-qe-mcp-3.15.10-v1<br/>agentic-qe.json:3"]
    AB["boundary: direct upstream MCP tools, no VisionCraft AQE executor<br/>agentic-qe.json:7"]
    A2["ADR 0002: AQE results describe observations and grant no gate<br/>0002-specialist-upstream-boundaries.md:34-37"]
    DJ --> DV
    DV --> DU
    DJ --> IC
    IC --> SV
    SV --> CK
    AJ --> AB
    AB --> A2
    N1["INVARIANT: unsupported Deepsec exits 78 and records UNAVAILABLE, not a pass<br/>deepsec.rs:1024"]
    N2["DEBT: no source or test loads agentic-qe.json; it is a documentation-grade contract, unlike deepsec.json<br/>agentic-qe.json:7"]
    N3["EXTERNAL: Deepsec and AQE are separate upstream packages installed by the host<br/>0002-specialist-upstream-boundaries.md:21-25"]
    CK --- N1
    AJ --- N2
    DJ --- N3
```

Deepsec is bound to code by a compiled-in contract; AQE is bound only by prose and a JSON file nothing parses. The line 7 boundary text says the file carries JSON Schemas, which the reader should treat as unverified in the build.

## VK-01.5 Skill payload: SKILL.md routes to references, assets and Rust guidance

```mermaid
flowchart LR
    SKM["SKILL.md: 35-file payload entry<br/>SKILL.md:10"]
    R1["EDD and expectation template<br/>SKILL.md:27"]
    R2["Debugging, static-oracle mode<br/>SKILL.md:28-29"]
    R3["Diagram diagnosis<br/>SKILL.md:30"]
    R4["DDD, ADR, TDD<br/>SKILL.md:31"]
    R5["Traceability<br/>SKILL.md:32"]
    R6["Review evidence<br/>SKILL.md:33"]
    R7["Quality gates, Deepsec adapter<br/>SKILL.md:34"]
    R8["Optional Agentic QE<br/>SKILL.md:35"]
    R9["Rust steering, review, port<br/>SKILL.md:36"]
    R10["Browser targets: Rust to WebAssembly<br/>SKILL.md:37"]
    R11["Roles and worked examples<br/>SKILL.md:38-39"]
    SKM --> R1
    SKM --> R2
    SKM --> R3
    SKM --> R4
    SKM --> R5
    SKM --> R6
    SKM --> R7
    SKM --> R8
    SKM --> R9
    SKM --> R10
    SKM --> R11
    L1["Lineage: skills/build-with-quality at agentbox a9125eaf<br/>provenance.json:4-6"]
    L2["Hard cutover: no old-name skill, redirect or second payload<br/>MIGRATION.md:3-8"]
    L3["see VK-05.3"]
    SKM --- L1
    L1 --- L2
    R10 --- L3
```

Load only the resources a task needs. The payload is lineage-tracked: the predecessor lived in agentbox, and the portable entry point requires no agentbox, model provider or fleet.
