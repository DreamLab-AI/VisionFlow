---
id: CS-04
title: Deploy contract, entrypoint and CI
area: codescape
governing:
  - ../codescape/deploy/README.md
  - ../codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
adrs: [codescape:ADR-0001, visionclaw:ADR-2143]
sources:
  - ../codescape/deploy/README.md
  - ../codescape/deploy/codescape-entrypoint.sh
  - ../codescape/.github/workflows/ci.yml
  - ../codescape/AGENTS.md
  - ../codescape/README.md
  - ../codescape/Cargo.toml
  - ../codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
verified_commit: d66bcab80fc0ce1f57deb6a0b6fa74e56ee8e298
---

## CS-04.1 Entrypoint phases from guard to exec

```mermaid
flowchart TD
    GUARD["refuse RUVECTOR_PG_CONNINFO, exit code 2<br/>codescape-entrypoint.sh:24,25,26"]
    FEAT["CODESCAPE_FEATURES auto: cuda only when nvcc and a device exist<br/>codescape-entrypoint.sh:32,33,34"]
    SKIP{"CODESCAPE_SKIP_BUILD is 1?<br/>codescape-entrypoint.sh:123"}
    PTX["build_ptx only when the cuda feature is selected<br/>codescape-entrypoint.sh:124,126"]
    CARGO["cargo build --release --locked -p codescape-server<br/>codescape-entrypoint.sh:129"]
    CLIENT["build_client into the target volume<br/>codescape-entrypoint.sh:130"]
    ARGS["repos, excludes and token flags from CODESCAPE_REPO, CODESCAPE_EXCLUDES, token file<br/>codescape-entrypoint.sh:134,135,138,141,142"]
    EXEC["exec the release server with data, static and port flags<br/>codescape-entrypoint.sh:146,149,150,151"]
    GUARD --> FEAT --> SKIP
    SKIP -->|no| PTX --> CARGO --> CLIENT --> ARGS
    SKIP -->|yes| ARGS
    ARGS --> EXEC
    TENS["TENSION: the default exclude list names agentbox, which the README says must never be excluded because it removes that member scene<br/>codescape-entrypoint.sh:138<br/>deploy/README.md:166"]
    ARGS --- TENS
    NOTE["see VC-42.1 for the runtime-artefact side of the same entrypoint"]
    EXEC --- NOTE
```

**What it shows.** Every phase is a build step except the final exec, and the skip switch jumps straight to the last build. The compose file overrides the exclude list (`deploy/README.md:33`), so the tension bites only a host that relies on the default.

## CS-04.2 PTX staging between the VisionClaw build and the layout loader

```mermaid
flowchart TD
    SRC["visionclaw-gpu sources mounted read-only, wrapper package on the target volume<br/>codescape-entrypoint.sh:54,59,68<br/>deploy/README.md:56"]
    BUILD["cargo build of the wrapper, JSON messages find the build-script out_dir<br/>codescape-entrypoint.sh:74,75,83,84"]
    NOPAIR["no pair produced: log CPU fallback and return success<br/>codescape-entrypoint.sh:86,87,88"]
    NOSRC["sources missing: log CPU fallback and return success<br/>codescape-entrypoint.sh:55,56,57"]
    STAGE["copy the pair to a temp dir, mode 0444, replace the output dir<br/>codescape-entrypoint.sh:90,92,93,94"]
    LOAD["layout reads CODESCAPE_PTX_DIR and checks tag, ABI, provenance<br/>deploy/README.md:187,188"]
    SRC --> BUILD
    BUILD -->|pair found| STAGE --> LOAD
    BUILD --> NOPAIR
    SRC --> NOSRC
    NOPAIR --> LOAD
    NOSRC --> LOAD
    INV["INVARIANT: the hashed build out dirs are never globbed, one fixed directory is the source of truth<br/>codescape-entrypoint.sh:49,50,51"]
    STAGE --- INV
    DEBT["DEBT: cargo stderr is discarded and both failure branches return success, so an old pair stays and still loads<br/>codescape-entrypoint.sh:75,87,88"]
    NOPAIR --- DEBT
    DRIFT["DOC-DRIFT: README says without the sources the engine falls back to CPU, a previously staged pair is not removed on that branch<br/>deploy/README.md:135<br/>codescape-entrypoint.sh:56,57"]
    NOSRC --- DRIFT
    NOTE["see VC-42.4 and CS-02.4"]
    LOAD --- NOTE
```

**What it shows.** The PTX pair is produced by VisionClaw's own build script and only copied here, never written by hand (`codescape-entrypoint.sh:46,47`). A failed refresh leaves the previous pair in place.

## CS-04.3 Ports, proxy trust and what is published

```mermaid
flowchart LR
    BR["browser on the LAN"]
    NGX["VisionClaw dev nginx, strips the codescape prefix, forwards X-Forwarded headers<br/>deploy/README.md:194,196,198,199"]
    CLI["clients port 4100: REST, positions, events, voice, wss, presence, static bundle<br/>deploy/README.md:146"]
    HOOK["hook ingest port 4101, bearer token, never published<br/>deploy/README.md:147"]
    LOOP["host loopback publish only<br/>deploy/README.md:97,146"]
    TRUST["trust proxy 1, trusted peer is the nginx network alias, prefix /codescape<br/>deploy/README.md:46,47,48<br/>deploy/README.md:206,207,208"]
    RATE["rate limiting reads the last forwarded hop, the one nginx appends<br/>deploy/README.md:210"]
    SEC["no-new-privileges, cap_drop ALL, only DAC_READ_SEARCH added<br/>deploy/README.md:90,91,92,93,94,95"]
    BR --> NGX --> CLI
    LOOP --> CLI
    HOOK -.-> SEC
    TRUST --> RATE
    NGX --> TRUST
    TENS["TENSION: the dev port 3001 is LAN-published, so /codescape/ is a deliberate dev-only exposure of an open-auth service<br/>deploy/README.md:212"]
    NGX --- TENS
    NOTE["see VC-39.2 and VC-39.6"]
    TRUST --- NOTE
```

**What it shows.** Only port 4100 is ever reachable from outside the container network, and trust in forwarded headers is narrowed to one named peer. Production nginx carries no Codescape location until P6 (`deploy/README.md:212`).

## CS-04.4 One service, many scenes: the estate mount rules

```mermaid
flowchart TD
    REPO["CODESCAPE_REPO lists estate first, then each member as its own scene<br/>deploy/README.md:30,31,32"]
    MNT["each repository is a .gitignore plus source directories, read-only, nothing else<br/>deploy/README.md:59,60,80,81,82"]
    HOOKS["a hook path under nested roots reaches every scene that contains it<br/>deploy/README.md:153"]
    MAP["CODESCAPE_PATH_MAP rewrites agent-side prefixes, longest prefix wins<br/>deploy/README.md:34,166"]
    LEFT["deliberately left out: target, dist, archive, corpus directories and key material<br/>deploy/README.md:157,158,159,160,162"]
    ADD["adding a sibling needs mounts, a scene entry and path-map entries<br/>deploy/README.md:166"]
    INV["INVARIANT: repositories are mounted read-only, never a whole checkout, corpus, root .env or secrets<br/>deploy/README.md:171<br/>AGENTS.md:67"]
    REPO --> MNT --> LEFT
    REPO --> HOOKS
    HOOKS --> MAP
    MNT --> INV
    ADD --> REPO
    DEBT["DEBT: the PATH_MAP value in the contract ends with an ellipsis, so the real list lives only in the host compose file<br/>deploy/README.md:34"]
    MAP --- DEBT
    EXT["EXTERNAL: mount paths and CODESCAPE_ESTATE_ROOT are defined by the VisionClaw compose file, not by this repository<br/>deploy/README.md:3,80"]
    MNT --- EXT
```

**What it shows.** The estate view is a nested-root trick: member repositories are ingested once each, and the umbrella scene is another walk over the same mounts. The exclude list is a second fence, not the first (`deploy/README.md:166`).

## CS-04.5 CI jobs and what they do not prove

```mermaid
flowchart TD
    TRIG["push to main and every pull request, one run per ref<br/>ci.yml:10,11,12,17,18,19"]
    RUST["rust job: fmt, clippy -D warnings, tests, rustdoc -Dwarnings, publish dry-run of model and collab<br/>ci.yml:35,36,39,40,42,45"]
    CUDA["rust-cuda job in a CUDA devel container: clippy and test --no-run, compile and link only<br/>ci.yml:54,69,70"]
    CLI["client job on node 22: ci, typecheck, test, build<br/>ci.yml:82,85,86,89,90"]
    TRIG --> RUST
    TRIG --> CUDA
    TRIG --> CLI
    GAP1["OPEN: no GPU on the runner, the device conformance test is a manual P2 gate<br/>ci.yml:51,52,53<br/>AGENTS.md:46,47,48"]
    GAP2["TENSION: toolchain is stable while the workspace declares rust-version 1.94<br/>ci.yml:31<br/>../codescape/Cargo.toml:33"]
    GAP3["OPEN: nothing here builds the deployment image or runs the entrypoint<br/>ci.yml:25,26"]
    CUDA --- GAP1
    RUST --- GAP2
    CLI --- GAP3
    INV["INVARIANT: the isolation and wire-fixture tests run inside the rust job's workspace test<br/>ci.yml:37,38,39"]
    RUST --- INV
```

**What it shows.** CI proves the permissive crates package and the CUDA feature links, nothing about a running GPU or a deployed container. Phase evidence for those is committed separately (`ADR-0001-extraction-from-visionclaw.md:110,114`).

## CS-04.6 Ingest token and the activity hook, with the record that disagrees

```mermaid
flowchart TD
    FILE["file secret named by CODESCAPE_INGEST_TOKEN_FILE<br/>deploy/README.md:35,181"]
    NOFILE["without the file the ingest listener does not start<br/>deploy/README.md:182"]
    HOOK["agentbox activity-sink hook reads the same token and posts to the codescape service on port 4101<br/>deploy/README.md:183"]
    ROLE["hooks speak for agents, humans appear only through client sockets<br/>deploy/README.md:183"]
    FILE --> NOFILE
    FILE --> HOOK
    HOOK --> ROLE
    D1["DOC-DRIFT: the deploy contract cites agentbox ADR-2144 on branch feat/codescape-activity-sink<br/>deploy/README.md:183"]
    D2["DOC-DRIFT: the README cites agentbox ADR-2146 as merged and default-off<br/>../codescape/README.md:99"]
    HOOK --- D1
    HOOK --- D2
    NOTE["see VC-39.2 for the compose side and CS-03.1 for the listener"]
    FILE --- NOTE
```

**What it shows.** Two documents in this repository name different ADR numbers and branch states for the same hook. The estate repository map follows the branch reading, which is the one to verify against agentbox.
