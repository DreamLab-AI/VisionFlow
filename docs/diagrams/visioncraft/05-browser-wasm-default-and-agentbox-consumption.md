---
id: VK-05
title: Browser WebAssembly default and agentbox consumption
area: visioncraft
governing:
  - ../visioncraft/skills/visioncraft/references/rust/browser-wasm.md
  - ../visioncraft/skills/visioncraft/references/ecosystem-agentbox.md
  - ../project/agentbox/docs/adr/ADR-2165-visioncraft-owns-the-portable-quality-method.md
  - ../project/agentbox/docs/developer/visioncraft-consumption.md
adrs: [ADR-2165, visioncraft:ADR-0002]
sources:
  - ../visioncraft/skills/visioncraft/references/rust/browser-wasm.md
  - ../visioncraft/skills/visioncraft/references/rust/steering.md
  - ../visioncraft/skills/visioncraft/references/worked-examples.md
  - ../visioncraft/skills/visioncraft/references/ecosystem-agentbox.md
  - ../visioncraft/skills/visioncraft/SKILL.md
  - ../visioncraft/skills/visioncraft/assets/provenance.json
  - ../visioncraft/src/plan.rs
  - ../visioncraft/src/cli.rs
  - ../visioncraft/src/evaluate.rs
  - ../visioncraft/src/deepsec.rs
  - ../visioncraft/docs/adr/0002-specialist-upstream-boundaries.md
  - ../project/agentbox/config/visioncraft-source.json
  - ../project/agentbox/lib/visioncraft.nix
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/services/agentbox-manifest/src/visioncraft_source.rs
  - ../project/agentbox/services/agentbox-manifest/src/deepsec_gate.rs
  - ../project/agentbox/docs/developer/visioncraft-consumption.md
  - ../project/agentbox/docs/adr/ADR-2165-visioncraft-owns-the-portable-quality-method.md
  - ../project/agentbox/skills/SKILL-DIRECTORY.md
verified_commit: {visioncraft: 2c6627ce01d0e68bb0f48fc55a209b2c674716f1, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93}
---

## VK-05.1 How agentbox adopted VisionCraft, in three commits

```mermaid
sequenceDiagram
    autonumber
    participant VC as visioncraft upstream
    participant AB as agentbox repository
    participant OP as operator
    VC->>AB: 75bd6c7e7 integrates the canonical method, a Nix package and direct AQE tools (lib/visioncraft.nix:4-7)
    Note over AB: hard cutover: build-with-quality is removed, one payload owns the name (visioncraft-consumption.md:3-6)
    VC->>AB: c1b0c700c pins published 0.1.0, commit 9cf450c, 34 payload files (ADR-2165-visioncraft-owns-the-portable-quality-method.md:110-112)
    VC->>AB: 83637b7c3 repins 0.2.0 from upstream b7e579f, 35 files (ADR-2165-visioncraft-owns-the-portable-quality-method.md:116-120)
    VC->>AB: cfa535b8c repins 0.3.0 from upstream 2c6627c, 40 files (visioncraft-source.json:3-4)
    Note over AB: 0.2.0 and 0.3.0 are skill-only, library and CLI unchanged from 0.1.0 (visioncraft-consumption.md:23-33)
    AB-->>OP: source pin complete, image rebuild and runtime acceptance still outstanding (visioncraft-consumption.md:38-40)
    Note over AB: INVARIANT: activation_status stays staged until the rebuild is observed (ADR-2165-visioncraft-owns-the-portable-quality-method.md:7)
    Note over VC,AB: DOC-DRIFT: ADR-2165 still says the final pin is 0.1.0 with 34 files, and later paragraphs move it to 0.2.0 with 35 and then 0.3.0 with 40 (ADR-2165-visioncraft-owns-the-portable-quality-method.md:110-111 vs ADR-2165-visioncraft-owns-the-portable-quality-method.md:116-120 and :123-128)
    Note over VC,AB: INVARIANT: the pinned revision 2c6627c is the upstream commit this topic is stamped at, so the pinned skill is the stamped skill
```

Agentbox is one consumer, not the owner: portable changes are made and released upstream, then re-vendored whole. Upstream carries tags v0.1.0, v0.2.0 and v0.3.0.

## VK-05.2 One immutable release, checked three ways

```mermaid
flowchart TB
    PIN["config/visioncraft-source.json: version, revision, NAR hash, per-file digests<br/>visioncraft-source.json:3-5"]
    FILES["40 skill file digests<br/>visioncraft-source.json:7-48"]
    LOCK["cargo_lock_sha256<br/>visioncraft-source.json:49"]
    NIX["lib/visioncraft.nix reads the same pin<br/>visioncraft.nix:6"]
    FETCH["fetchFromGitHub at pin.revision with pin.source_hash<br/>visioncraft.nix:10-14"]
    CL["cargoLock from lib/lockfiles/visioncraft-VERSION.Cargo.lock<br/>visioncraft.nix:16"]
    DC["doCheck = false: upstream's offline suite is the gate<br/>visioncraft.nix:20-21"]
    FL["flake.nix includes visioncraftPkg in skillToolPackages<br/>flake.nix:1854-1855"]
    DF["image bake: diff -r fetched skill against the vendored copy<br/>flake.nix:2097-2099"]
    LK["lockfile copied into the image<br/>flake.nix:2106"]
    SC["agentbox-manifest visioncraft-source-check<br/>visioncraft_source.rs:74"]
    S1["inventory must equal pinned digests<br/>visioncraft_source.rs:113-117"]
    S2["retired build-with-quality must not exist<br/>visioncraft_source.rs:118-124"]
    S3["lockfile digest must match<br/>visioncraft_source.rs:136-140"]
    PIN --> FILES
    PIN --> LOCK
    PIN --> NIX
    NIX --> FETCH
    NIX --> CL
    NIX --> DC
    FETCH --> FL
    FL --> DF
    FL --> LK
    PIN --> SC
    SC --> S1
    SC --> S2
    SC --> S3
    N1["INVARIANT: a local edit to the vendored skill fails both the source check and the image bake<br/>visioncraft-consumption.md:100-104"]
    N2["EXTERNAL: tests are not run at image build; quality depends on upstream CI (see VK-04.1)<br/>visioncraft.nix:20"]
    DF --- N1
    DC --- N2
```

The Nix hash is computed without Nix by NAR-serialising the unpacked archive (visioncraft-consumption.md:33-35). A pin alone proves source integrity, not that a running image contains the release.

## VK-05.3 Browser targets: Rust to WebAssembly is the default

```mermaid
flowchart TB
    Q0{"Will new or changed code run in a browser?<br/>browser-wasm.md:6-8"}
    Q1{"Is it behaviour worth testing: models, parsing, wire formats, geometry, search, schemas, crypto?<br/>browser-wasm.md:10-13"}
    RUST["Write it in a Rust crate for wasm32-unknown-unknown, cdylib plus rlib<br/>browser-wasm.md:35-36"]
    SHIM["JavaScript host shim only: load module, DOM, events, a single fetch, typed arrays<br/>browser-wasm.md:15-21"]
    Q2{"Existing code, or glue over a JavaScript-only library?<br/>browser-wasm.md:26-30"}
    KEEP["Leave it in JavaScript; the UI framework stays<br/>browser-wasm.md:29-31"]
    PORT["Port when a change touches its logic or a maintenance, correctness or performance need justifies it<br/>browser-wasm.md:26-29"]
    EXC["Record the reason when choosing JS: glue, tiny logic, no wasm build step, deadline fix<br/>browser-wasm.md:89-93"]
    Q0 -- yes --> Q1
    Q0 -- no --> OUT["not covered by this reference"]
    Q1 -- yes --> Q2
    Q1 -- no --> SHIM
    Q2 -- new logic --> RUST
    Q2 -- existing --> PORT
    Q2 -- glue --> KEEP
    RUST --> SHIM
    KEEP --> EXC
    N1["INVARIANT: pure core with no clock, randomness, storage or network, so the same core builds for wasm32 and native<br/>browser-wasm.md:37-39"]
    N2["INVARIANT: if a shim function branches on domain state, that logic belongs in the crate<br/>browser-wasm.md:23-24"]
    N3["TENSION: the default is a priority, not a licence to rewrite; the port guide still applies<br/>browser-wasm.md:26-27"]
    RUST --- N1
    SHIM --- N2
    PORT --- N3
```

This is the PR #2 change (79d9ee9, merged as b7e579f). The decision is a skill rule, not a tool: nothing in the crate enforces it, and familiarity alone is not an accepted reason for JavaScript.

## VK-05.4 Where the wasm evidence gates meet the native runner

```mermaid
flowchart LR
    G1["1 cargo test on the native core<br/>browser-wasm.md:74"]
    G2["2 cargo build --target wasm32-unknown-unknown plus clippy for that target<br/>browser-wasm.md:75-76"]
    G3["3 wasm-bindgen-test headless for browser-API behaviour<br/>browser-wasm.md:77"]
    G4["4 one real browser journey through the actual bundle<br/>browser-wasm.md:78-79"]
    G5["5 .wasm size against budget, load time where it matters<br/>browser-wasm.md:80-81"]
    EC["exit-code check, executable evidence<br/>plan.rs:175-177"]
    PV["partially verifiable or manual, needs a limitation<br/>plan.rs:178-181"]
    GEN["generic run records observed_tests empty<br/>cli.rs:875"]
    SK["missing tool becomes Skipped, a visible non-pass<br/>evaluate.rs:190-195"]
    G1 --> EC
    G2 --> EC
    G3 --> EC
    G4 --> PV
    G5 --> PV
    EC --> GEN
    G1 --- N1
    N1["INVARIANT: a missing wasm build or browser run is a visible non-pass; native tests alone do not show the module loads<br/>browser-wasm.md:83-85"]
    N2["TENSION: gates 1 to 3 are expressible as exit-code checks, but none can be a stabilizer without an adapter that reports executed test ids<br/>evaluate.rs:344-347"]
    N3["see VK-02.6"]
    GEN --- N2
    N2 --- N3
    G2 --- SK
```

The skill promises a specific browser journey that the CLI cannot observe. Gate 4 therefore rests on a named human or browser-tool observation, bound to the plan digest as a ManualObservation.

## VK-05.5 Where the browser rule travels, and the Deepsec host policy path

```mermaid
flowchart TB
    SKD["SKILL.md description names Rust compiled to WebAssembly<br/>SKILL.md:3"]
    ROW["routing row: code that will run in a browser<br/>SKILL.md:37"]
    STE["steering.md restates the default and links the reference<br/>steering.md:33-35"]
    WEX["todo example puts item model and storage schema in a wasm crate<br/>worked-examples.md:15-18"]
    PRV["provenance records browser-wasm.md as newly written guidance<br/>provenance.json:249-251"]
    DIR["agentbox skill directory lists browser targets as Rust compiled to WebAssembly<br/>SKILL-DIRECTORY.md:51"]
    TOML["agentbox.toml toolchain flags: agentic_qe, deepsec<br/>agentbox.toml:2285-2288"]
    MAN["load manifest; only an absent default manifest may use defaults<br/>deepsec_gate.rs:129-140"]
    ENV["DEEPSEC_GATE_* overrides read, non-UTF-8 refused<br/>deepsec_gate.rs:144-154"]
    EXE["exec visioncraft deepsec --policy-json; no fallback review runs<br/>deepsec_gate.rs:155-163"]
    POL["adapter parses the policy and rejects unknown fields<br/>deepsec.rs:666-670"]
    SKD --> ROW
    ROW --> STE
    STE --> WEX
    WEX --> PRV
    PRV --> DIR
    TOML --> MAN --> ENV --> EXE --> POL
    N1["INVARIANT: credential names, never values, enter policy JSON<br/>visioncraft-consumption.md:51-53"]
    N2["INVARIANT: the gate is a caller-controlled local CLI, not a privilege boundary<br/>visioncraft-consumption.md:44-48"]
    N3["see VK-03.3"]
    EXE --- N1
    EXE --- N2
    POL --- N3
```

The browser rule reaches agentbox only through the vendored skill copy, so repinning is what propagates it. The second chain shows that agentbox projects policy and then hands execution to the upstream executable; it keeps no parallel gate.
