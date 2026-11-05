---
id: VK-04
title: CI, release and the Sealmap diagram gate
area: visioncraft
governing:
  - ../visioncraft/docs/diagrams/README.md
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
adrs: [visioncraft:ADR-0001, visioncraft:ADR-0002]
sources:
  - ../visioncraft/.github/workflows/ci.yml
  - ../visioncraft/Cargo.toml
  - ../visioncraft/README.md
  - ../visioncraft/src/deepsec.rs
  - ../visioncraft/skills/visioncraft/SKILL.md
  - ../visioncraft/docs/diagrams/README.md
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
  - ../visioncraft/docs/adr/0002-specialist-upstream-boundaries.md
verified_commit: 1067a5f59afad7e2c8780bce6b29967d1130c19d
---

## VK-04.1 CI jobs on push to main and on every pull request

```mermaid
flowchart TB
    TR["triggers: push to main, pull_request<br/>ci.yml:2-5"]
    PM["permissions: contents read only<br/>ci.yml:6-7"]
    Q["job quality<br/>ci.yml:9"]
    Q1["cargo fmt --check<br/>ci.yml:16"]
    Q2["cargo clippy --locked --all-targets -D warnings<br/>ci.yml:17"]
    Q3["cargo test --locked<br/>ci.yml:18"]
    Q4["cargo test --locked --no-default-features<br/>ci.yml:19"]
    Q5["cargo doc --no-deps with RUSTDOCFLAGS -D warnings<br/>ci.yml:20-22"]
    Q6["cargo package --locked<br/>ci.yml:23"]
    M["job msrv: toolchain 1.88.0<br/>ci.yml:24-28"]
    M1["cargo check --locked --all-targets and --no-default-features<br/>ci.yml:29-30"]
    D["job diagrams<br/>ci.yml:31"]
    TR --> PM
    PM --> Q
    PM --> M
    PM --> D
    Q --> Q1 --> Q2 --> Q3 --> Q4 --> Q5 --> Q6
    M --> M1
    N1["INVARIANT: actions are pinned by commit SHA; the stable toolchain action SHA was inspected 2026-11-03<br/>ci.yml:13"]
    N2["INVARIANT: msrv toolchain equals rust-version 1.88<br/>Cargo.toml:5"]
    N3["see VK-04.2"]
    Q --- N1
    M --- N2
    D --- N3
```

Jobs run independently, in parallel. The no-default-features test and check prove the pure library builds without the process and CLI dependencies (VK-01.2).

## VK-04.2 Sealmap generates the machine diagrams; CI only checks them

```mermaid
sequenceDiagram
    autonumber
    actor Dev as maintainer
    participant SM as sealmap 0.4.0
    participant GN as docs/diagrams/generated
    participant CI as diagrams job<br/>ci.yml:31
    Dev->>SM: sealmap generate . --name visioncraft --public-only --min-calls 4 --no-model (docs/diagrams/README.md:11-12)
    SM->>GN: _overview.md and _index.json with hashes, spans, call confidence (docs/diagrams/README.md:3-5)
    Note over GN: INVARIANT: generated views are read-only, edit code and regenerate (docs/diagrams/README.md:5-7)
    CI->>CI: cargo install sealmap --version =0.4.0 --locked (ci.yml:36)
    CI->>SM: the same generate command plus --check against the candidate source (ci.yml:37)
    SM-->>CI: fails if generated output differs from source
    Note over CI: EXTERNAL: sealmap-corpus 0.4.0 is also a library dependency of the crate (Cargo.toml:28)
    Note over GN: DEBT: the generated overview misclassifies the recorded alias, a path-attribute module, as an external call target, recorded as a known extraction limit (docs/diagrams/README.md:46-49)
    Note over GN: see deepsec.rs:19-20 for the alias declaration
    Note over Dev,CI: TENSION: the pinned CLI version and the library dependency are independent edits, nothing ties ci.yml:36 to Cargo.toml:28
```

Generated output is never copied into the estate corpus. This tree is a separate, hand-authored, citation-verified view of the same repository (see VK-04.4).

## VK-04.3 From merge to a consumable release

```mermaid
sequenceDiagram
    autonumber
    actor M as maintainer
    participant GH as GitHub main
    participant CI as CI workflow<br/>ci.yml:1
    participant CR as crates.io
    participant AB as agentbox pin
    M->>GH: pull request, for example 79d9ee9 browser-wasm skill change
    GH->>CI: quality, msrv, diagrams jobs (ci.yml:3-5)
    CI->>CI: cargo package --locked proves the crate packages (ci.yml:23)
    GH-->>M: merge as b7e579f, PR number 2
    Note over GH: version bumped to 0.2.0 in the crate (Cargo.toml:3) and the skill (SKILL.md:6)
    M->>CR: publish by hand, since no CI job publishes (ci.yml:9-37)
    Note over CR: README points at crates.io and docs.rs (../visioncraft/README.md:25)
    Note over CR: README install pins version 0.2.0 for both CLI and library (../visioncraft/README.md:35, ../visioncraft/README.md:43)
    M->>AB: re-pin by commit, tree hash and lockfile digest
    Note over AB: see VK-05.2
    Note over M,AB: DEBT: publication, adoption and deployment are separate observations, only the first two leave a receipt (0002-specialist-upstream-boundaries.md:62)
```

There is no release workflow, tag job or publish step in ci.yml. The crate version, skill metadata and README install lines must be bumped together by hand, and they agree at this revision.

## VK-04.4 Four records of the architecture and who checks each

```mermaid
flowchart LR
    ADR["Hand-written ADRs 0001 and 0002<br/>0001-portable-quality-contract.md:1"]
    AUTH["Authored overview diagram<br/>docs/diagrams/README.md:27-44"]
    GEN["Sealmap generated corpus, machine output<br/>docs/diagrams/README.md:3-5"]
    EST["This estate corpus, VK-01 to VK-05<br/>docs/diagrams/README.md:1"]
    CHK1["ci.yml diagrams job: regenerate and compare<br/>ci.yml:37"]
    CHK2["VisionFlow gate: every path:line read at the declared sha"]
    ST1["ADR verified_commit e3306776<br/>0001-portable-quality-contract.md:6"]
    ST2["README stamp e3306776<br/>docs/diagrams/README.md:22-25"]
    ADR --- ST1
    AUTH --- ST2
    GEN --> CHK1
    EST --> CHK2
    N1["INVARIANT: src, contracts and Cargo.toml dependencies are unchanged between e3306776 and 1067a5f except the version bump, so both stamps remain true"]
    N2["INVARIANT: ADR 0001 says its inspected revision predates the documentation commit; refresh only after rechecking governed paths<br/>0001-portable-quality-contract.md:68-69"]
    N3["INVARIANT: extraction, Mermaid rendering and runtime verification are separate observations<br/>docs/diagrams/README.md:19-20"]
    ST1 --- N1
    ST2 --- N1
    ADR --- N2
    EST --- N3
```

Only the generated corpus is machine-checked inside the repository. The ADRs and authored diagram rely on their stamps and on this estate gate to catch drift.
