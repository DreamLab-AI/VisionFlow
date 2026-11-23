---
id: LM-06
title: Deploy image, entrypoint, compose profiles, CI gates and the release identity
area: loom
governing:
  - ../loom/docs/design/RUST-ARCHITECTURE.md
  - ../loom/docs/development-source.md
adrs: [loom:ADR-137, loom:ADR-141]
sources:
  - ../loom/deploy/Dockerfile
  - ../loom/deploy/Dockerfile.dockerignore
  - ../loom/deploy/entrypoint.sh
  - ../loom/deploy/compose.profile-a.yml
  - ../loom/deploy/compose.profile-b.yml
  - ../loom/deploy/compose.model-vision.yml
  - ../loom/deploy/ruvector-workspace.trimmed.toml
  - ../loom/model/Dockerfile
  - ../loom/model/serve-qwen38.sh
  - ../loom/.github/workflows/rust-contracts.yml
  - ../loom/justfile
  - ../loom/rust-toolchain.toml
  - ../loom/scripts/check-ruvector-source.sh
  - ../loom/Cargo.toml
  - ../loom/crates/loom-facade/Cargo.toml
  - ../loom/crates/loom-facade/build.rs
  - ../loom/crates/loom-facade/src/build_info.rs
  - ../loom/crates/loom-facade/src/main.rs
  - ../loom/crates/loom-embed-xinference/src/lib.rs
  - ../loom/docs/development-source.md
  - ../loom/docs/design/RUST-ARCHITECTURE.md
verified_commit: {loom: 355946dd2deacfdef21171d53eafd025887e447e}
---

## LM-06.1 The image: parent-directory context, three stages, a baked smoke test

```mermaid
flowchart TB
    CTX["build context is the PARENT of the repo so the sibling ruvector path dep is reachable<br/>deploy/Dockerfile line 5 to 9<br/>dockerignore allowlists loom and ruvector-core only<br/>../loom/deploy/Dockerfile.dockerignore:9,12,13,21"]
    subgraph S1["stage 1 builder, rust 1.97 bookworm, deploy/Dockerfile line 39"]
        DEP["dependency layer: copy manifests, stub sources, build deps<br/>deploy/Dockerfile lines 62 to 70, 79 to 89, 96"]
        TRIM["trimmed ruvector workspace root replaces the real one<br/>../loom/deploy/ruvector-workspace.trimmed.toml:1"]
        SRC["real sources, rust-toolchain.toml deleted, release build of loom-facade<br/>deploy/Dockerfile lines 103, 106, 113, 114"]
        DEP --> SRC
        TRIM --> DEP
    end
    subgraph S2["stage 2 runtime, debian bookworm-slim, deploy/Dockerfile line 119"]
        RUN["uid 65532 nonroot, /run/loom and /var/lib/loom owned by it,<br/>env defaults, curl HEALTHCHECK, entrypoint script<br/>deploy/Dockerfile lines 136 to 139, 152 to 159, 163, 166"]
    end
    subgraph S3["stage 3 smoke, deploy/Dockerfile line 177"]
        SMK["bake the generation in, start, assert index_classes 8146<br/>deploy/Dockerfile lines 181 to 186, 201"]
    end
    CTX --> S1 --> S2 --> S3
    D1["DEBT: the stub layer copies 8 of the 11 member manifests and its build ends in or-true,<br/>so a failed dependency layer is silent<br/>deploy/Dockerfile lines 62 to 70, 96<br/>../loom/Cargo.toml:3"]
    DEP --- D1
    D2["DOC-DRIFT: comments say MSRV 1.88, default features exclude attest, and main ignores argv,<br/>but the workspace says 1.89, attest is default and main honours --build-info<br/>deploy/Dockerfile lines 34, 92, 110<br/>../loom/Cargo.toml:24<br/>loom-facade/Cargo.toml:57<br/>loom-facade/src/main.rs:35"]
    SRC --- D2
    D3["TENSION: the smoke stage bakes from loom/app/data, the directory Profile A<br/>calls legacy and unmounted<br/>deploy/Dockerfile line 181<br/>../loom/deploy/compose.profile-a.yml:99"]
    SMK --- D3
    D4["DOC-DRIFT: the architecture still plans cargo-chef into a scratch or distroless static image<br/>../loom/docs/design/RUST-ARCHITECTURE.md:38"]
    S2 --- D4
```

**What it shows.** RocksDB under Oxigraph and the entrypoint's need for a shell forced a glibc Debian runtime instead of the static image the design planned (deploy/Dockerfile lines 18 to 26). The smoke test's class count is a literal, so it pins one corpus size.

## LM-06.2 The entrypoint: copy the rvdb off the read-only mount before the binary opens it

```mermaid
flowchart TB
    START["entrypoint<br/>../loom/deploy/entrypoint.sh:28,30,31"]
    HAS{"rvdb present at LOOM_HNSW_ARTIFACT?<br/>../loom/deploy/entrypoint.sh:36"}
    SKIP["log and serve the lexical floor<br/>../loom/deploy/entrypoint.sh:37,38"]
    CP["cp rvdb and its sidecar into the writable runtime dir<br/>../loom/deploy/entrypoint.sh:41,46,48"]
    SHA{"sidecar records a 64-hex sha256?<br/>../loom/deploy/entrypoint.sh:55,58"}
    CHK["mismatch: FATAL exit 1<br/>../loom/deploy/entrypoint.sh:60,61,62"]
    NOSHA["no digest: copy anyway, log skip<br/>../loom/deploy/entrypoint.sh:65,66"]
    EXP["repoint LOOM_HNSW_ARTIFACT, exec loom-facade<br/>../loom/deploy/entrypoint.sh:69,76"]
    START --> HAS
    HAS -->|no| SKIP --> EXP
    HAS -->|yes| CP --> SHA
    SHA -->|yes| CHK
    SHA -->|no| NOSHA --> EXP
    CHK -->|match| EXP
    WHY["redb mutates the file even for reads, both profiles mount data read-only<br/>../loom/deploy/entrypoint.sh:5,6,7,8"]
    CP --- WHY
    OPEN["OPEN: the comment says the sidecar carries no sha256 today, so the digest step<br/>is dormant until the producer stamps one<br/>../loom/deploy/entrypoint.sh:17,18,19"]
    NOSHA --- OPEN
    TMPFS["tmpfs at /run/loom with uid and gid 65532 in both profiles<br/>../loom/deploy/compose.profile-a.yml:107,108<br/>../loom/deploy/compose.profile-b.yml:79"]
    CP --- TMPFS
```

**What it shows.** The mitigation sits at the container level because opening the artefact rewrites it. Every other artefact is read straight from the read-only mount (`../loom/deploy/entrypoint.sh:22,23`).

## LM-06.3 Three compose files: the colocated reference, the network sidecar and the model

```mermaid
flowchart TB
    subgraph A["Profile A, loom-facade-a"]
        A1["network_mode host, port 8084, backend on the host loopback port 8085<br/>../loom/deploy/compose.profile-a.yml:21,23,42"]
        A2["gate explicit: injection on, 8.0, 2.0, 0.4<br/>../loom/deploy/compose.profile-a.yml:57,58,59,60"]
        A3["F1 verbatim on at 8.0, F3 no-think on, think floor 1536<br/>../loom/deploy/compose.profile-a.yml:70,71,76,79"]
        A4["data read-only, retained bundles read-only, ledger on a named volume<br/>../loom/deploy/compose.profile-a.yml:27,93,97,98"]
    end
    subgraph B["Profile B, loom-facade-b"]
        B1["hostname loom on the external visionclaw_network, published 8080<br/>../loom/deploy/compose.profile-b.yml:16,18,20,94,95"]
        B2["backend loom-model port 8085, embedder xinference port 9997<br/>../loom/deploy/compose.profile-b.yml:40,42"]
        B3["verbatim and no-think left commented out, full delegation<br/>../loom/deploy/compose.profile-b.yml:47,48,50,53"]
        B4["pg-write variable inert in the default image<br/>../loom/deploy/compose.profile-b.yml:61,62,65"]
    end
    subgraph MOD["loom-model"]
        M1["host network, all GPUs, Hemmingway-1.5 Q8_0, 262K context, MTP speculation<br/>../loom/deploy/compose.model-vision.yml:2,18,20,22,28,30"]
        M2["llama-server listens on port 8085 by default<br/>../loom/model/serve-qwen38.sh:20"]
    end
    A --> MOD
    B --> MOD
    SEM["both: semantic fallback off until the recall bench clears<br/>../loom/deploy/compose.profile-a.yml:44<br/>../loom/deploy/compose.profile-b.yml:43"]
    A --- SEM
    TENS["OPEN: Profile A sets no XINFERENCE_URL, so the embedder defaults to the<br/>docker-network name xinference while the container runs on the host network<br/>../loom/deploy/compose.profile-a.yml:21<br/>loom-embed-xinference/src/lib.rs:38,74"]
    A1 --- TENS
    IMG["OPEN: the model image tag names llama.cpp b11982 while model/Dockerfile pins<br/>a commit, and nothing here records that they match<br/>../loom/deploy/compose.model-vision.yml:15<br/>model/Dockerfile line 22"]
    M1 --- IMG
    LIC["licence CC-BY-NC-4.0, non-commercial<br/>../loom/deploy/compose.model-vision.yml:4"]
    M1 --- LIC
    SEE["see AB-24.1 and ES-06.1 for which consumer holds which door"]
    B1 --- SEE
```

**What it shows.** A is the GPU-colocated reference with every findings-driven switch on; B is the in-network door that keeps plain delegation. The model is a separate compose project, so a model swap never touches either Loom container.

## LM-06.4 What "green" means: just ci against the hosted workflow

```mermaid
flowchart TB
    subgraph JUST["just ci, justfile line 56"]
        J0["fmt --check, justfile lines 25 to 26"]
        J1["build --all-features AND --no-default-features, justfile lines 35 to 37"]
        J2["test --all-features, justfile lines 40 to 41"]
        J3["clippy --all-targets --all-features -D warnings, justfile lines 44 to 45"]
        J4["cargo deny check, justfile lines 48 to 49"]
    end
    subgraph CI["Rust contracts workflow, push to main, PRs, manual"]
        C0["read toolchain and ruvector pins from the repo<br/>../loom/.github/workflows/rust-contracts.yml:2,3,4,5,6,25,26"]
        C1["checkout jjohare/ruvector at the pin, source guard<br/>../loom/.github/workflows/rust-contracts.yml:34,35,41"]
        C2["fmt --check<br/>../loom/.github/workflows/rust-contracts.yml:47"]
        C3["test --workspace --locked, default features<br/>../loom/.github/workflows/rust-contracts.yml:53"]
        C4["clippy --all-features -D warnings<br/>../loom/.github/workflows/rust-contracts.yml:57"]
        C0 --> C1 --> C2 --> C3 --> C4
    end
    JUST --> CI
    GAP["TENSION: CI runs no cargo deny, no no-default-features build and tests on default<br/>features only, so the deny gate and the serving-plane build exist only locally<br/>../loom/.github/workflows/rust-contracts.yml:53,57"]
    C3 --- GAP
    TC["toolchain pinned 1.98.0 for CI and rustup, the image builds on 1.97, MSRV 1.89<br/>../loom/rust-toolchain.toml:12,14<br/>../loom/Cargo.toml:24"]
    C0 --- TC
    DRIFT["DOC-DRIFT: the architecture's gates 6 to 8 name a recall gate, a Nix build and a<br/>profile-parity check, none wired in CI here<br/>../loom/docs/design/RUST-ARCHITECTURE.md:832,833,834"]
    CI --- DRIFT
    BENCH["bench and docker-smoke are separate recipes, not part of ci, justfile lines 52 to 53, 68 to 69"]
    JUST --- BENCH
```

**What it shows.** The workflow takes its versions from in-repo files, so local and hosted runs read the same pins. It covers a subset of the local bar (justfile and Dockerfile citations are line references the checker cannot see, derived by hand).

## LM-06.5 Release identity: a Loom sha alone does not identify the build

```mermaid
flowchart TB
    WHY["ruvector-core is a path dependency with no lockfile revision<br/>loom-facade/build.rs:3,4,5,6,7"]
    BS["build.rs bakes Loom sha, RuVector sha and path, rustc, profile, target, features<br/>loom-facade/build.rs:27,28,29,33,34,35,38,43"]
    GIT["reads .git directly, follows worktree pointers, unknown when absent<br/>loom-facade/build.rs:70,79,80,82,89,90"]
    REC["--build-info prints the receipt plus effective config<br/>loom-facade/src/main.rs:18,19,20,21"]
    EXIT["non-zero when either revision is unknown<br/>loom-facade/src/main.rs:22,25,29<br/>loom-facade/src/build_info.rs:72,73"]
    WHY --> BS --> GIT --> REC --> EXIT
    GUARD["source guard: sibling at the fork commit, no modified or untracked core source<br/>../loom/scripts/check-ruvector-source.sh:6,8,9,10<br/>../loom/docs/development-source.md:3,4"]
    BS --- GUARD
    TENS["TENSION: the image context drops loom/.git and every ruvector entry but the core crate,<br/>so build.rs inside the image would record unknown and --build-info there exits non-zero<br/>../loom/deploy/Dockerfile.dockerignore:18,21,28"]
    GIT --- TENS
    TENS2["TENSION: the image build never runs the source guard, it compiles whatever<br/>ruvector-core sits beside loom in the build host's parent directory<br/>deploy/Dockerfile lines 73, 104"]
    GUARD --- TENS2
```

**What it shows.** The receipt is the reproducibility artefact ADR-137 asked for, and it refuses to pretend when a revision is missing. The guard protects CI and developer builds; the container path has neither the guard nor the git metadata the receipt needs.
