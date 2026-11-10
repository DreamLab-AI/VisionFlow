---
id: ES-09
title: Build, deploy and CI estate — source to running container, every gate
area: estate
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [agentbox:ADR-2153, visionclaw:ADR-2148, visionclaw:ADR-2147, visionclaw:ADR-2143, visionclaw:ADR-2008, visionclaw:ADR-2037, agentbox:ADR-2013, agentbox:ADR-2028, agentbox:ADR-2039, agentbox:ADR-2056, agentbox:ADR-2082, agentbox:ADR-2083, agentbox:ADR-2085, agentbox:ADR-2091, agentbox:ADR-2092, agentbox:ADR-2093, agentbox:ADR-2094, visionclaw:ADR-2086, visionclaw:ADR-2119, agentbox:ADR-2122]
sources:
  - ../project/Dockerfile.unified
  - ../project/Dockerfile.production
  - ../project/supervisord.dev.conf
  - ../project/supervisord.production.conf
  - ../project/docker-compose.unified.yml
  - ../project/docker-compose.cloudflared.yml
  - ../project/nginx.dev.conf
  - ../project/nginx.production.conf
  - ../project/nginx.conf
  - ../project/scripts/dev-entrypoint.sh
  - ../project/scripts/rust-backend-wrapper.sh
  - ../project/scripts/prod-entrypoint.sh
  - ../project/scripts/production-startup.sh
  - ../project/docs/adr/ADR-2037-production-build-excludes-dev-auth.md
  - ../project/docs/adr/ADR-2086-ci-gate-dev-auth-excluded-from-release.md
  - ../project/src/main.rs
  - ../project/src/utils/auth.rs
  - ../project/src/handlers/solid_proxy_handler.rs
  - ../project/docs/adr/ADR-2119-prod-ingress-is-declared-lan-or-tunnel.md
  - ../project/.gitmodules
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/.github/workflows/ci.yml
  - ../project/.github/workflows/docs-ci.yml
  - ../project/.github/workflows/ontology-publish.yml
  - ../project/.github/workflows/xr-godot-ci.yml
  - ../project/agentbox/.github/workflows/invariants.yml
  - ../project/agentbox/.github/workflows/ci.yml
  - ../project/agentbox/.github/workflows/contract-tests.yml
  - ../project/agentbox/.github/workflows/manifest-validate.yml
  - ../project/agentbox/.github/workflows/flake-check.yml
  - ../project/agentbox/.github/workflows/secret-scan.yml
  - ../project/agentbox/.github/workflows/shellcheck.yml
  - ../project/agentbox/.github/workflows/image-scan.yml
  - ../project/agentbox/.github/workflows/deepsec.yml
  - ../project/agentbox/.github/workflows/tui-tests.yml
  - ../project/agentbox/.github/workflows/build-multi-arch.yml
  - ../project/agentbox/.github/workflows/nix-flake-update.yml
  - ../project/agentbox/.github/workflows/release.yml
  - ../project/agentbox/.github/workflows/adr-checker.yml
  - ../project/agentbox/.github/workflows/security-vex.yml
  - ../project/build.rs
  - ../project/env.example
  - ../project/src/uri/mod.rs
  - ../project/agentbox/schema/federation-kinds.json
  - ../project/agentbox/scripts/ci/check-ports-loopback.sh
  - ../project/agentbox/scripts/ci/check-ports-loopback.mjs
  - ../project/agentbox/scripts/ci/check-no-logseq-paths.sh
  - ../project/scripts/adr-index-gen.js
  - ../project/scripts/adr-ratchet.sh
  - ../project/agentbox/scripts/adr-ratchet.sh
  - ../project/scripts/ontology/pack-pod-resources.py
  - ../project/scripts/launch.sh
  - ../project/scripts/start.sh
  - ../project/agentbox/.github/workflows/browsercontainer.yml
  - ../project/agentbox/tests/runtime-contract/README.md
  - ../project/agentbox/tests/sovereign/identity-port.node-test.js
  - ../project/agentbox/tests/sovereign/g5-key-split.node-test.js
  - ../project/agentbox/tests/sovereign/role-secret.node-test.js
  - ../project/agentbox/tests/config/role-isolation-rehearsal.test.sh
  - ../project/agentbox/tests/config/sidechain-producer-baked.test.sh
  - ../project/agentbox/services/nostr-pod-bridge/tests/identity_port.rs
  - ../project/agentbox/services/nostr-pod-bridge/tests/role_isolation.rs
  - ../project/agentbox/scripts/ci/env-secret-inventory.js
  - ../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../project/agentbox/docker-compose.system-one.yml
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93}
---
## ES-09.1 The host-vs-container build trap — wrong path vs sanctioned path
```mermaid
flowchart TB
    classDef wrong fill:#5a1414,stroke:#ff4444,color:#fff
    classDef right fill:#144a1e,stroke:#33cc55,color:#fff
    classDef fact fill:#333,stroke:#999,color:#eee

    ENVFACT["ENV-FACT (operator-verified, not repo-committed):<br/>this container has the HOST Docker socket mounted.<br/>DinD builds LOOK like they work — they do not for source mounts:<br/>bind paths resolve against the HOST filesystem"]:::fact

    subgraph CCBOX["Claude Code container (this session)"]
        EDIT["Edit source<br/>/home/devuser/workspace/project"]
        SSHTRY["NEVER: ssh to the host"]
        LAUNCHTRY["NEVER: ./scripts/launch.sh up dev<br/>run from inside this container"]
    end

    subgraph HOSTFS["Host filesystem<br/>bind: /mnt/mldata/githubs/AR-AI-Knowledge-Graph"]
        HOSTSRC["src/, Cargo.toml, client/src<br/>(same inode as CC edit — content identical)"]
    end

    subgraph HOSTDOCKERD["Host dockerd (reached via forwarded socket)"]
        BUILDREQ["docker compose --profile dev up<br/>issued from inside CC"]
    end

    EDIT ---|"bind mount, always in sync"| HOSTSRC
    LAUNCHTRY -->|"socket-forwarded request"| BUILDREQ:::wrong
    SSHTRY -.->|"refused: no LAN IP path from CC to host shell"| REFUSED(("blocked")):::wrong
    BUILDREQ -->|"resolves HOST_PROJECT_ROOT-relative<br/>bind-mount paths against ITS OWN (host-side)<br/>cwd/view, not the CC container path"| MISBIND["docker-compose.unified.yml:169-206<br/>bind source path mismatch"]:::wrong
    MISBIND --> STALE["Running dev container serves the<br/>image-build-time COPY'd source,<br/>NOT the just-edited host file"]:::wrong
    STALE -.->|"the trap: container starts, health check passes,<br/>edits never take effect"| TRAPEND(("silent stale-code failure")):::wrong

    ENVFACT -.-> LAUNCHTRY

    subgraph HOSTSHELL["Host shell — tmux tab 6"]
        RIGHT1["tmux send-keys -t 6<br/>./scripts/launch.sh up dev Enter<br/>source-only, ~2 min"]
        RIGHT2["tmux send-keys -t 6<br/>./scripts/launch.sh rebuild dev Enter<br/>Dockerfile/deps changed, ~15 min"]
        MONITOR["tmux capture-pane -t 6"]
    end

    HOSTSRC ==>|"host operator, or this session's own<br/>tmux send-keys into tab 6"| RIGHT1
    HOSTSRC ==> RIGHT2
    RIGHT1 ==>|"bind mounts resolve correctly:<br/>host cwd IS the bind source"| GOODBUILD["Correct dev container<br/>live-reload works"]:::right
    RIGHT2 ==> GOODBUILD
    GOODBUILD ==> MONITOR:::right

    DOCKEREXEC["docker exec (safe from CC — no bind-path resolution involved)"]
    CCBOX -.->|"agentbox equivalent: ./agentbox.sh rebuild, host only"| AGENTBOXREBUILD["agentbox rebuild (host tmux tab 6)"]:::right
    CCBOX -.->|"read-only inspection, always fine"| DOCKEREXEC
```

## ES-09.2 Edit-build-verify cycle across the container/host boundary
```mermaid
sequenceDiagram
    autonumber
    participant Dev as Developer
    participant CC as ClaudeCodeContainer<br/>bind:/home/devuser/workspace/project
    participant T6 as HostShellTmux6
    participant DD as HostDockerd
    participant DC as visionclaw_container<br/>docker-compose.unified.yml:69
    participant W as rust-backend-wrapper.sh

    Dev->>CC: edit src/main.rs
    Note over CC: same bind mount as host path<br/>/mnt/mldata/githubs/AR-AI-Knowledge-Graph
    rect rgb(230,235,245)
    Note over T6,DD: container/host process boundary — build MUST cross here, never from CC
    Dev->>T6: tmux send-keys -t 6 ./scripts/launch.sh up dev Enter
    T6->>T6: sync_dev_inputs publishes Cargo.toml, Cargo.lock, build.rs and the Vite root files<br/>into the gitignored .dev-inputs, stamped with the source commit<br/>scripts/launch.sh:640-651
    T6->>DD: docker compose --profile dev up -d
    DD->>DD: resolve HOST_PROJECT_ROOT bind mounts, directories only plus the<br/>stamped .dev-inputs (create_host_path false)<br/>docker-compose.unified.yml:169-206
    DD->>DC: recreate container with correct host-side binds
    end
    DC->>W: supervisord starts program:rust-backend<br/>supervisord.dev.conf:34
    W->>W: dev_inputs_install, refuse to build unless the stamp verifies<br/>scripts/rust-backend-wrapper.sh:91-95
    W->>W: needs_rebuild scripts/rust-backend-wrapper.sh:100
    alt source or Cargo manifest changed
        W->>W: cargo build --locked --profile dev-runtime --features "$BUILD_FEATURES"<br/>scripts/rust-backend-wrapper.sh:66, default gpu,ontology,dev-auth :42
        W->>W: write_build_stamp scripts/rust-backend-wrapper.sh:113
    else stamp up to date
        W->>W: skip cargo scripts/rust-backend-wrapper.sh:107
    end
    Note over W: ADR-2156 (2026-10-29) - a failed build is retried after cleaning the root<br/>package only, never a full cargo clean, so compiled dependencies survive,<br/>scripts/rust-backend-wrapper.sh:119-124. The dev-runtime profile also drops<br/>cross-crate LTO, so the dev loop is no longer a single-core fat-LTO link.<br/>build.rs declares .git/HEAD only when it exists, build.rs:38-39
    W->>DC: exec visionclaw-server
    Dev->>CC: sudo docker exec visionclaw_container curl localhost:4000/api/health
    CC->>DC: docker exec (socket path, no bind resolution — safe from CC)
    DC-->>CC: 200 OK
    Note over Dev,CC: NEVER ssh to the host, NEVER run launch.sh from inside CC (ENV-FACT)
    Note over T6,DD: monitor with tmux capture-pane -t 6, never poll inside CC
```

## ES-09.3 Dockerfile.unified — multi-stage build, dev vs prod target divergence
```mermaid
flowchart LR
    BASE["base<br/>Dockerfile.unified:27<br/>cachyos-v3 pinned digest<br/>ARG CUDA_ARCH=75 (:30) promoted to ENV:39-46"]
    RUSTDEPS["rust-deps<br/>Dockerfile.unified:145<br/>COPY Cargo.toml/crates, cargo fetch:183,<br/>cargo build --release --features gpu:184"]
    RUSTBUILD["rust-builder<br/>Dockerfile.unified:190<br/>COPY src:193, cargo build --release --features gpu:213<br/>strip target/release/visionclaw-server:214"]
    NODEDEPS["node-deps<br/>Dockerfile.unified:221<br/>npm ci --prefer-offline --no-audit:233"]
    NODEBUILD["node-builder<br/>Dockerfile.unified:238<br/>npx vite build:247"]
    DEV["development target<br/>Dockerfile.unified:254<br/>FROM base — NO rust-builder/node-builder<br/>COPY src SOURCE (not binaries):288, COPY client:293<br/>ENTRYPOINT ./dev-entrypoint.sh at Dockerfile.unified:339"]
    PROD["production target<br/>Dockerfile.unified:347<br/>FROM cachyos-v3 fresh, NOT from base<br/>COPY --from=rust-builder binary:406<br/>COPY --from=node-builder dist:409<br/>USER appuser:427, ENTRYPOINT prod-entrypoint.sh at Dockerfile.unified:437"]

    BASE --> RUSTDEPS --> RUSTBUILD
    BASE --> NODEDEPS --> NODEBUILD
    BASE --> DEV
    RUSTBUILD --> PROD
    NODEBUILD --> PROD

    DIVERGE["DIVERGENCE: dev COPYs raw src and compiles at container<br/>start. The entrypoint no longer runs cargo itself: it backgrounds<br/>scripts/rust-backend-wrapper.sh (dev-entrypoint.sh:104), which<br/>owns the rebuild decision and the feature set. Prod COPYs the<br/>pre-compiled rust-builder binary and compiles nothing at start."]
    DEV -.-> DIVERGE
    PROD -.-> DIVERGE
```

## ES-09.4 Dockerfile.production — cache-optimised 5-stage pipeline
```mermaid
flowchart LR
    TOOLCHAIN["toolchain<br/>Dockerfile.production:13<br/>ARG CUDA_ARCH=86:15, ENV CUDA_ARCH promoted:17-22<br/>cachyos-v3 pinned digest, rustup stable, node 20.18.3"]
    DEPS["deps<br/>Dockerfile.production:59<br/>FROM toolchain — writes stub src/main.rs + 4 stub bins<br/>(ADR-2114 renamed sync_local.rs/sync_github.rs to sync_corpus.rs)<br/>at Dockerfile.production:73-84, stub build.rs at :88<br/>cargo fetch --locked:100 MUST pass; crate build :101-102 may fail"]
    CUDAPTX["cuda-ptx<br/>Dockerfile.production:107<br/>FROM toolchain — re-declares ARG CUDA_ARCH=86:109<br/>nvcc -ptx -arch sm_CUDA_ARCH:120"]
    FRONTEND["frontend<br/>Dockerfile.production:127<br/>FROM toolchain — npm ci:138, npx vite build:143"]
    BUILDER["builder<br/>Dockerfile.production:148<br/>FROM deps — COPY real src:158, cargo build --release:164<br/>COPY --from=cuda-ptx ptx:154"]
    RUNTIME["runtime (final, unnamed)<br/>Dockerfile.production:171<br/>fresh cachyos-v3 — NOT from toolchain<br/>COPY --from=builder binary:230, COPY start script :237<br/>USER appuser:243, ENTRYPOINT /app/start.sh at Dockerfile.production:245"]

    TOOLCHAIN --> DEPS
    TOOLCHAIN --> CUDAPTX
    TOOLCHAIN --> FRONTEND
    DEPS --> BUILDER
    CUDAPTX --> BUILDER
    BUILDER --> RUNTIME
    FRONTEND --> RUNTIME
    CUDAPTX --> RUNTIME

    NOTE1["cache-layer rationale: deps layer invalidates only on<br/>Cargo.toml/lock change; cuda-ptx only on .cu file change;<br/>frontend only on client/ change — code-only edits skip<br/>dependency download and PTX recompilation entirely"]
    BUILDER -.-> NOTE1
```

## ES-09.5 CUDA_ARCH ARG-to-ENV promotion, and the ADR-2037 hygiene-stub divergence
```mermaid
sequenceDiagram
    autonumber
    participant U as Dockerfile-unified-base<br/>Dockerfile.unified:27
    participant C1 as rust-deps-child-stage<br/>Dockerfile.unified:145
    participant P as Dockerfile-production-toolchain<br/>Dockerfile.production:13
    participant C2 as cuda-ptx-child-stage<br/>Dockerfile.production:107

    U->>U: ARG CUDA_ARCH=75 (:30, scoped to this stage only)
    U->>U: ENV CUDA_ARCH=CUDA_ARCH (:44, promotes ARG into ENV)
    U->>C1: FROM base AS rust-deps
    Note over C1: INVARIANT: ENV values set in a parent stage ARE<br/>inherited by a child FROM stage, ARG values are NOT
    C1->>C1: nvcc build.rs reads env CUDA_ARCH (inherited, correct)

    P->>P: ARG CUDA_ARCH=86 (:15, scoped to toolchain stage only)
    P->>P: ENV CUDA_ARCH=CUDA_ARCH (:21, promotes ARG into ENV)
    P->>C2: FROM toolchain AS cuda-ptx
    Note over C2: re-declares ARG CUDA_ARCH=86 (:109) redundantly,<br/>ENV already inherited from toolchain — both agree
    C2->>C2: nvcc -ptx -arch sm_CUDA_ARCH (:120)

    Note over U,P: DIVERGENCE (ADR-2008 vs ADR-2037): the dev build is now a<br/>NAMED PROFILE rather than release — cargo build --profile<br/>dev-runtime with BUILD_FEATURES defaulting to gpu,ontology,dev-auth<br/>(scripts/rust-backend-wrapper.sh:42,66). The binary still carries<br/>enforce_release_env_hygiene as a no-op stub (src/main.rs:169), so<br/>the ADR-2037 boundary is the CI gate and the profile name, not the<br/>compiler flag
    Note over U,P: ADR-2037 is proposed, implementation_status partial, activation staged<br/>(ADR-2037-production-build-excludes-dev-auth.md:5-7). Its CI half is ADR-2086,<br/>accepted, complete and live (ADR-2086-ci-gate-dev-auth-excluded-from-release.md:5-7):<br/>a hermetic text gate over the committed Dockerfiles, see ES-09.11. No gate<br/>inspects a BUILT image, so a mis-targeted pipeline that bypasses those files<br/>could still promote the stubbed-hygiene binary
```

**Open (no gate inspects a built image):** ADR-2037 remains proposed and partial (`../project/docs/adr/ADR-2037-production-build-excludes-dev-auth.md:5-7`). Its CI half, ADR-2086, is live (`../project/docs/adr/ADR-2086-ci-gate-dev-auth-excluded-from-release.md:5-7`), but that gate is a text assertion over the committed Dockerfiles (`../project/.github/workflows/ci.yml:183-188`). A production binary built outside those files, still carrying the hygiene stub (`../project/src/main.rs:169`), would not be caught.

## ES-09.6 supervisord.dev.conf — program set and restart policy
```mermaid
stateDiagram-v2
    [*] --> supervisordRoot
    supervisordRoot: supervisord nodaemon supervisord.dev.conf:1
    supervisordRoot --> nginx
    supervisordRoot --> rustBackend
    supervisordRoot --> viteDev
    supervisordRoot --> devInputsSync

    nginx: program nginx supervisord.dev.conf:8 autorestart true
    devInputsSync: program dev-inputs-sync supervisord.dev.conf:21 keeps the host root build files current, ADR-2008 supervisord.dev.conf:19-22
    rustBackend: program rust-backend supervisord.dev.conf:33 command rust-backend-wrapper.sh supervisord.dev.conf:34
    viteDev: program vite-dev supervisord.dev.conf:48 npm run dev supervisord.dev.conf:49

    rustBackend --> rustBackendRetry: crash, startretries 3 supervisord.dev.conf:37
    rustBackendRetry --> rustBackend: restart, startsecs 10
    rustBackendRetry --> rustBackendFatal: exceeds startretries
    rustBackendFatal --> [*]

    nginx --> nginxRestart: crash, autorestart true
    nginxRestart --> nginx

    viteDev --> viteRestart: crash, autorestart true
    viteRestart --> viteDev

    note right of rustBackend
        environment RUST_LOG, MCP_TCP_PORT 9500
        supervisord.dev.conf 46
    end note
    note right of supervisordRoot
        unix_http_server /tmp/supervisor.sock
        supervisord.dev.conf 60
    end note
```

## ES-09.7 supervisord.production.conf — reached only by the Dockerfile.unified production target
```mermaid
stateDiagram-v2
    [*] --> supervisordRootProd
    supervisordRootProd: supervisord declares user root supervisord.production.conf:1-3
    supervisordRootProd --> nginxProd
    supervisordRootProd --> rustBackendProd

    nginxProd: program nginx supervisord.production.conf:18 priority 10
    rustBackendProd: program rust-backend supervisord.production.conf:28 command /app/target/release/visionclaw-server --port 4001 supervisord.production.conf:29

    rustBackendProd --> rustBackendProdRetry: crash, startretries 5 supervisord.production.conf:38
    rustBackendProdRetry --> rustBackendProd: restart, startsecs 10, stopasgroup killasgroup true supervisord.production.conf:39-41
    nginxProd --> nginxProdRetry: crash, startretries 3 supervisord.production.conf:26
    nginxProdRetry --> nginxProd

    note right of rustBackendProd
        environment NVIDIA_VISIBLE_DEVICES
        supervisord.production.conf 32
    end note
    note right of supervisordRootProd
        DIVERGENCE vs dev: no vite-dev program,
        binary already compiled, no wrapper script
    end note
    note left of supervisordRootProd
        cross-estate compare (agentbox flake.nix 2622 and 2684):
        agentbox supervisord also runs as PID1 root,
        but every long-running program declares user devuser
        per-program. Here the image switches to USER appuser
        before the entrypoint (Dockerfile.unified 427), so the
        user=root line cannot raise privilege and every program
        runs as appuser
    end note
    note left of rustBackendProd
        DEBT: the command path does not exist in the image
        that uses this file. Dockerfile.unified copies the
        binary to /app/visionclaw-server (lines 393 and 406),
        not /app/target/release
    end note
```

**Divergence (which production path uses this file):** only `Dockerfile.unified`'s `production` target copies this config (`../project/Dockerfile.unified:414`) and execs it (`../project/scripts/prod-entrypoint.sh:23`). The compose `visionclaw-production` service builds `Dockerfile.production` instead (`../project/docker-compose.unified.yml:256-258`), whose entrypoint is `/app/start.sh` (`../project/Dockerfile.production:237`, `:245`). That script starts the backend on port 4001 in the background and runs nginx in the foreground itself, with no supervisord (`../project/scripts/production-startup.sh:38-41`, `:68-70`). From source, nothing on the compose production path restarts a backend that crashes after startup: nginx keeps the container alive, so `restart: unless-stopped` (`../project/docker-compose.unified.yml:323`) does not fire.

**Debt (binary path):** the `rust-backend` program runs `/app/target/release/visionclaw-server` (`../project/supervisord.production.conf:29`), but the image that reads this file puts the binary at `/app/visionclaw-server` (`../project/Dockerfile.unified:406`, with `WORKDIR /app` at `:393`). Traced in source only and not run: from source, the `Dockerfile.unified` production target would fail its backend start. Closing it means pointing the command at `/app/visionclaw-server`, or retiring the target if `Dockerfile.production` is the only production build.

## ES-09.8 Compose profiles — dev, production, tunnel, loom
```mermaid
flowchart TB
    subgraph PROFILES["docker-compose.unified.yml services block"]
        DEVSVC["visionclaw<br/>docker-compose.unified.yml:80 target development, compose-hash label :75<br/>profiles development and dev, docker-compose.unified.yml:243-245<br/>ports 3001 and 4000, docker-compose.unified.yml:225-226<br/>source-bind volumes docker-compose.unified.yml:169-206,<br/>docker.sock read-only docker-compose.unified.yml:215"]
        VAULTMOUNT["ADR-2114 corpus vault mount<br/>agent-workspace:/vault:ro docker-compose.unified.yml:219-222<br/>external named volume multi-agent-docker_workspace<br/>declared docker-compose.unified.yml:680-682"]
        PRODSVC["visionclaw-production<br/>docker-compose.unified.yml:248<br/>profiles production and prod, docker-compose.unified.yml:332-334<br/>port 3001 only, docker-compose.unified.yml:306-308<br/>NO source mounts and NO docker.sock, docker-compose.unified.yml:303-305"]
        CLOUDFLARED["cloudflared<br/>docker-compose.unified.yml:339, image cloudflare/cloudflared at a pinned<br/>digest :342, its OWN profile tunnel docker-compose.unified.yml:358-359 (ADR-2119)<br/>depends_on visionclaw OR visionclaw-production (optional)"]
        LOOM["loom<br/>docker-compose.unified.yml:382, image loom:rust built outside this repo :384<br/>loom compose profile docker-compose.unified.yml:453-454<br/>host port 8090 to container port 8080 docker-compose.unified.yml:438<br/>hostname loom :385, alias ontology-loom docker-compose.unified.yml:443"]
        CODESCAPE["codescape (ADR-2143, opt-in profile codescape)<br/>the Codescape submodule built and run as a sidecar on port 4100,<br/>ingest token as a compose secret<br/>docker-compose.unified.yml:463-476, docker-compose.unified.yml:659-660,<br/>docker-compose.unified.yml:668-669"]
        SYS1["System One (VisionClaw ADR-2148, accepted / complete / live, close-out of ADR-2148-enable-local-typed-decisions, amended 2026-10-30): the x-system-one-environment anchor, docker-compose.unified.yml:2-11,<br/>is merged into the VisionClaw common environment :14,22, loom :387-395 and codescape :475,505. All three<br/>reach the agentbox proxy agentbox:8094 with JEV_ALLOW_CLOUD default 1, not the laya facade systemone:8097<br/>(still an opt-in local backend, agentbox docker-compose.system-one.yml:3 and :106). Thresholds stay unset and uncalibrated.<br/>Codescape command acceptance is qualified in VC-42. LIVE: both activation conditions (the agentbox rebuild and the host recreate)<br/>happened on 2026-10-30, per the ADR-2148 close-out"]
    end
    subgraph EXTFILE["docker-compose.cloudflared.yml (standalone)"]
        CFSTANDALONE["cloudflared<br/>joins external visionclaw_network<br/>alias visionclaw-server:3001"]
    end
    NET["visionclaw_network (external, pre-created)<br/>docker-compose.unified.yml:671-674"]
    INGRESS["launch.sh up prod reads VISIONCLAW_INGRESS from .env.prod,<br/>lan or tunnel, undeclared means tunnel, scripts/launch.sh:177-190<br/>tunnel activates prod,tunnel, lan activates prod alone :296-300;<br/>dev adds the codescape profile only when the sidecar is chosen :270-275"]
    INGRESS -->|"tunnel"| CLOUDFLARED
    INGRESS -->|"prod in both modes"| PRODSVC

    DEVSVC --> VAULTMOUNT
    DEVSVC --> NET
    PRODSVC --> NET
    CLOUDFLARED --> NET
    LOOM --> NET
    CODESCAPE --> NET
    CFSTANDALONE --> NET

    GATE["Deployment constraint: default dev and production<br/>both publish host port 3001, so concurrent activation collides.<br/>Compose profiles do not enforce mutual exclusion."]
    DEVSVC -.-> GATE
    PRODSVC -.-> GATE
```

## ES-09.9 nginx route tables — dev vs production upstreams
```mermaid
flowchart LR
    subgraph DEVNGINX["nginx.dev.conf — listen 3001 nginx.dev.conf:56"]
        DUPRUST["upstream rust_backend<br/>127.0.0.1:4000 :44-47"]
        DUPVITE["upstream vite_frontend<br/>127.0.0.1:5173 :49-52"]
        DAPI["^~ /api/ -> rust_backend :108-109"]
        DWSS["/wss, /ws/speech, /ws/mcp-relay -> rust_backend :128, :148, :168"]
        DSOLID["^~ /solid/, ^~ /pods/ -> rust_backend/api/solid/ :209, :235"]
        DHMR["/vite-hmr, /@vite, /node_modules -> vite_frontend :296, :308"]
        DROOT["/ -> vite_frontend (dev server, no static build) :332"]
        DCODESCAPE["^~ /codescape/ -> the codescape sidecar on codescape:4100, resolved lazily<br/>through Docker DNS so nginx starts when it is down (ADR-2143 A4)<br/>nginx.dev.conf:59-98"]
        DXFH["X-Forwarded-Host = http_host, the dialled host:port,<br/>on /api/, /wss and /ws/speech only nginx.dev.conf:112, nginx.dev.conf:134, nginx.dev.conf:154<br/>/solid/ and /pods/ still send host without port nginx.dev.conf:236, nginx.dev.conf:262"]
    end
    subgraph PRODNGINX["nginx.production.conf — listen 3001 :85"]
        PUPRUST["upstream rust_backend<br/>127.0.0.1:4001 max_fails=0 :69-72"]
        PAPI["^~ /api/ -> rust_backend :114-115"]
        PWS["wss / ws/speech / ws/mcp-relay / ws/hybrid-status -> rust_backend :235-236"]
        PSOLID["^~ /solid/, ^~ /pods/ -> rust_backend/api/solid/ :170,215"]
        PSTATIC["/, *.html, *.js/css/png -> static /app/client/dist :271,284,297"]
        PHEALTH["/health, /healthz, /readyz -> rust_backend or static :306,315,321"]
        PXFH["X-Forwarded-Host = http_host on /api/ and the ws routes<br/>nginx.production.conf:126,248<br/>/solid/ and /pods/ send host nginx.production.conf:177,222"]
    end
    LEGACY["nginx.conf (root, listen 4000 nginx.conf:82)<br/>generic template, NOT referenced by any<br/>Dockerfile/compose COPY — kept as reference only"]

    DAPI --> DUPRUST
    DWSS --> DUPRUST
    DSOLID --> DUPRUST
    DHMR --> DUPVITE
    DROOT --> DUPVITE
    PAPI --> PUPRUST
    PWS --> PUPRUST
    PSOLID --> PUPRUST
    PHEALTH --> PUPRUST
    DAPI -.-> DXFH
    PAPI -.-> PXFH
    XFHWHY["2026-10-02 (e7e6b61d8): a NIP-98 u tag signs the URL the client<br/>dialled, port included. nginx Host drops the port, so the backend<br/>rebuilds the URL from X-Forwarded-Proto and X-Forwarded-Host,<br/>auth.rs:170-189"]
    DXFH -.-> XFHWHY
    PXFH -.-> XFHWHY
    XFHOPEN["OPEN: the pod NIP-98 check also rebuilds its URL from<br/>X-Forwarded-Host, solid_proxy_handler.rs:207-222, but the /solid/ and<br/>/pods/ blocks still forward host without the port. Whether a LAN<br/>client signing host:3001 for a pod write is refused is untested."]
    DSOLID -.-> XFHOPEN

    DIVNOTE["DIVERGENCE: dev proxies ALL non-API routes to the<br/>Vite dev server (live HMR); prod serves a static<br/>client/dist build directly from nginx root, only<br/>API/WS routes reach the backend upstream"]
    DROOT -.-> DIVNOTE
    PSTATIC -.-> DIVNOTE

    PRECED["RESOLVED 2026-10-01 (648c9c442) — in nginx a regex location<br/>outranks a plain prefix, so the asset-extension regex<br/>(nginx.dev.conf:342, nginx.production.conf:271) captured any<br/>/solid/, /pods/ or /api/ path ending in .png or .js and sent it to<br/>Vite or the static root. Every backend prefix now carries ^~,<br/>which makes the prefix match final: nginx.dev.conf:108, nginx.dev.conf:229, nginx.dev.conf:255<br/>and nginx.production.conf:114,215 (production /solid/ already had it)."]
    DSOLID -.-> PRECED
    PSOLID -.-> PRECED
```

## ES-09.10 agentbox flake rebuild gate, and the submodule pointer-bump flow
```mermaid
flowchart TB
    subgraph FLAKE["agentbox/flake.nix — image composition"]
        NIXPKG["Nix package set<br/>e.g. toolchains.ruflo gate agentbox/flake.nix:490"]
        SUPTEXT["supervisorText string<br/>agentbox/flake.nix:2672, agentbox/flake.nix:2703,<br/>agentbox/flake.nix:2719<br/>program blocks e.g. management-api, bootstrap-seal"]
        SUPWRITE["writeText supervisord.conf, asserting no JSS_ variable leaked<br/>agentbox/flake.nix:4050-4055"]
    end
    subgraph TOML["agentbox/agentbox.toml — RUNNING config, not a template"]
        GATEKEY["gate key e.g. interaction_plane.enabled, now rebuild-class (ADR-2153)<br/>agentbox.toml:2125-2126, catalogue entry system-manifest.js:74-75"]
    end
    subgraph MANIFEST["agentbox/management-api/lib/system-manifest.js"]
        CATALOGUE["CATALOGUE entry<br/>system-manifest.js:39<br/>gate, service, apply_class"]
        APPLYCLASS["APPLY_CLASSES:<br/>live system-manifest.js:28, boot system-manifest.js:29,<br/>rebuild system-manifest.js:30<br/>ADR-039 apply-class taxonomy"]
    end

    GATEKEY -->|"read at eval time"| NIXPKG
    GATEKEY -->|"read at eval time"| SUPTEXT
    NIXPKG --> SUPWRITE
    SUPTEXT --> SUPWRITE
    GATEKEY -.->|"catalogue entry documents the SAME gate"| CATALOGUE
    CATALOGUE --> APPLYCLASS

    RULE["rule (agentbox CLAUDE.md, project CLAUDE.md):<br/>adding a gate means gating BOTH the Nix package<br/>set AND the supervisor block, plus a catalogue<br/>entry with an honest apply-class"]
    NIXPKG -.-> RULE
    SUPTEXT -.-> RULE
    CATALOGUE -.-> RULE

    REBUILD["./agentbox.sh rebuild (host tmux tab 6 only)"]
    SUPWRITE --> REBUILD
    REBUILD --> IMAGE["new agentbox image, apply_class rebuild changes take effect"]

    subgraph SUBMOD["VisionClaw submodule pointer-bump"]
        GITMODULES[".gitmodules:1-3<br/>submodule agentbox<br/>url github.com/DreamLab-AI/agentbox.git<br/>.gitmodules:4-6<br/>submodule codescape (ADR-2143 A5, its own repository)<br/>url github.com/DreamLab-AI/codescape"]
        SUBSTATUS["git submodule status<br/>4f41f77be agentbox and e3b76f63e codescape at the stamped gitlinks<br/>a + prefix means the checkout differs from the index"]
        SUBUPDATE["cd agentbox and git checkout NEW_SHA<br/>then git add agentbox (records gitlink)"]
        SUBCOMMIT["git commit records new gitlink SHA<br/>in the VisionClaw superproject tree"]
    end
    GITMODULES --> SUBSTATUS
    SUBSTATUS --> SUBUPDATE
    SUBUPDATE --> SUBCOMMIT
    IMAGE -.->|"agentbox image built from the checked-out<br/>submodule commit, independent pin"| SUBSTATUS
    SKEW["DRIFT: the gitlink 4f41f77be at the VisionClaw stamp is 16 commits behind<br/>the agentbox stamp 720e0d89 this topic reads agentbox files at, so what VisionClaw<br/>pins is older than what these panels describe"]
    SUBSTATUS -.-> SKEW
```

**Drift (submodule pin skew):** `.gitmodules` names the agentbox and codescape submodules (`../project/.gitmodules:1-6`). At the VisionClaw stamp the agentbox gitlink is `4f41f77be`, 16 commits behind the agentbox stamp `720e0d89` that this topic reads agentbox files at. A claim here about agentbox therefore describes a revision newer than the one VisionClaw checks out.

## ES-09.11 VisionClaw ci.yml — Rust CPU/client blocking gates, GPU excluded
```mermaid
sequenceDiagram
    autonumber
    participant GH as GitHub pull request to main or manual dispatch (the push trigger was removed 2026-11-10)<br/>project/.github/workflows/ci.yml:41-62
    participant FMT as rust-fmt job<br/>project/.github/workflows/ci.yml:77 blocking
    participant CPU as rust-cpu job<br/>project/.github/workflows/ci.yml:92 blocking
    participant CLI as client job<br/>project/.github/workflows/ci.yml:140 blocking
    participant GATE as dev-auth-release-gate job<br/>project/.github/workflows/ci.yml:162 blocking
    participant LINT as client-quality job<br/>project/.github/workflows/ci.yml:254 advisory
    participant PW as playwright job<br/>project/.github/workflows/ci.yml:285 manual only

    GH->>FMT: cargo fmt --all --check :89-90
    GH->>CPU: cargo build CPU_CRATES :123, now built over<br/>vault-core and vault (ADR-2113), replacing vault-migrate :108-109
    CPU->>CPU: cargo clippy CPU_CRATES --all-targets :127
    CPU->>CPU: cargo test CPU_CRATES :129
    CPU->>CPU: cargo clippy/test -p visionclaw-integration-tests, hermetic targets<br/>backup_posture, dev_build_inputs, prod_ingress (ADR-2119) and layer_direction :136,138
    GH->>CLI: npm ci :155-156, then npm run test (vitest) :160
    GH->>GATE: hermetic text assertion over the<br/>committed Dockerfiles/entrypoints :166-175
    GATE->>GATE: no --release cargo line may name dev-auth :183-188
    GH->>LINT: npm run lint (ESLint) :274, npx tsc --noEmit :276
    Note over LINT: continue-on-error true :261, never a required check
    opt workflow_dispatch only :291
        GH->>PW: npx playwright install :307, npm run test:e2e :309
    end
    Note over GATE: INVARIANT: ADR-2037 via ADR-2086 - a production/release<br/>image must never carry the dev-auth cargo feature (it compiles in<br/>the Bearer dev-session-token bypass and stubs<br/>enforce_release_env_hygiene to a no-op).<br/>Hermetic: no cargo, no docker, no network - project/.github/workflows/ci.yml:162,166
    Note over CPU: DIVERGENCE: visionclaw-gpu and root server crate link<br/>CUDA at runtime, removed from hosted CI 2026-07-24 (project/.github/workflows/ci.yml:278-280)
    Note over CPU: GPU crates validated only on the developer CUDA<br/>host via scripts/launch.sh, not by any GitHub runner
```

## ES-09.12 VisionClaw docs-ci.yml — ADR ledger gate and documentation quality score
```mermaid
sequenceDiagram
    autonumber
    participant GH as pull request touching docs/** or the ratchet and its tests, or manual dispatch (the push trigger was removed 2026-11-10)<br/>docs-ci.yml:3-16
    participant ADR as validate-adr-ledger job<br/>docs-ci.yml:23
    participant DOC as validate-documentation job<br/>docs-ci.yml:45

    GH->>ADR: checkout fetch-depth 0 :31 (staleness diffs old commits)
    ADR->>ADR: node scripts/adr-index-gen.js docs/adr --check :33
    Note over ADR: checks frontmatter, supersession reciprocity,<br/>verified_commit staleness, and the evidence_class enum (see ES-09.23)
    ADR->>ADR: ratchet evidence-gate self-test :34-35, evidence_class value check :37-38
    ADR->>ADR: bash scripts/adr-ratchet.sh docs/adr BASE sha :40-43
    Note over ADR: TWO RULES in the ratchet. Backlog: fails when more ADRs were ADDED<br/>undecided than LEFT proposed over the push range, project/scripts/adr-ratchet.sh:115-126.<br/>A record added already decided is reported, not counted (:74), an ADR-Ratchet<br/>trailer exempts the record it names (:65-66, :73), and after ADR_RATCHET_UNTIL,<br/>default 2026-10-20 (:45), it reports only (:119-123). Evidence (VisionFlow ADR-2007):<br/>a record turning complete or live without evidence_class and acceptance_boundary<br/>fails, with no end date (:81-113)

    GH->>DOC: checkout
    DOC->>DOC: validate internal links :54-119
    DOC->>DOC: validate mermaid diagrams :121-174
    DOC->>DOC: check stale references :176-222
    Note over DOC: stale refs are warnings only, never a score penalty :219-222
    DOC->>DOC: validate directory structure :224-253
    Note over DOC: required Diataxis dirs: tutorials, how-to,<br/>explanation, reference, plus docs/README.md :232-244
    DOC->>DOC: score = (links_rate*60 + mermaid_rate*40)/100<br/>minus 10 per structure error, clamp 0-100 :266-272
    alt score below 50
        DOC->>GH: exit 1, quality threshold failed :338-340
    else score at or above 50
        DOC->>GH: pass :342
    end
```

**Open (the backlog rule has lapsed):** `ADR_RATCHET_UNTIL` defaults to 2026-10-20 (`../project/scripts/adr-ratchet.sh:45`), so on any run after that date a push that adds more undecided records than it closes is reported and passes (`../project/scripts/adr-ratchet.sh:119-123`). Only the evidence rule still fails a build. Whether the backlog rule should be extended or retired is the owner's decision; the workflow step's name still describes it as live (`../project/.github/workflows/docs-ci.yml:40`).

## ES-09.13 VisionClaw ontology-publish.yml — vault build now runs in-repo, pull model to pod
```mermaid
sequenceDiagram
    autonumber
    participant GH as push/PR/repository_dispatch<br/>ontology-publish.yml:3-22 corpus-sync
    participant VAL as validate-source job<br/>ontology-publish.yml:56
    participant BLD as build-ontology job<br/>ontology-publish.yml:146
    participant REL as publish-release job<br/>ontology-publish.yml:300
    participant SRV as visionclaw-server boot<br/>src/services/ontology_pull.rs
    participant DEP as deploy-jss job (push path)<br/>ontology-publish.yml:371
    participant WS as notify-websocket job<br/>ontology-publish.yml:482
    participant PRP as pr-preview job<br/>ontology-publish.yml:562
    participant MIS as deploy-target-missing job<br/>ontology-publish.yml:617

    GH->>VAL: preflight gh repo view on the private ontology source :80-86<br/>ONTOLOGY_SOURCE_TOKEN, falling back to GITHUB_TOKEN :80-81
    alt source unreadable
        VAL--xGH: ::error + OPS ACTION naming the fine-grained PAT, exit 1 :87-100
    end
    GH->>VAL: checkout ontology source into vault-source/, same token :102-108,<br/>at the dispatch payload's source_sha or main :106
    VAL->>VAL: detect changed markdown files :110-131
    VAL->>BLD: has_changes true, needs validate-source :150-151
    BLD->>BLD: checkout the source into vault-source/ again :162-168,<br/>pinned to validate-source's source_sha :166
    BLD->>BLD: ADR-2113 — cargo build --release --locked -p vault :182<br/>(crates/vault in THIS repo, cached by Cargo.lock hash :170-178)
    BLD->>BLD: ./target/release/vault --repo vault-source build<br/>--vault knowledge --out output/vault :184-187
    BLD->>BLD: ADR-2147 vault certify writes an oo-cert/1 derivation certificate and<br/>exits 1 if the certified closure stops matching Whelk's :194-203, then a pinned<br/>Lean 4 oo-cert binary re-checks it by digest :205-222, reports and certificate uploaded :224-241
    BLD->>BLD: pack-pod-resources.py output/vault output/pod — public-only<br/>TTL, JSON-LD compacted against the vault context, LDP index<br/>manifest, substance floor :255-261
    BLD->>BLD: re-parse packed resources :263-279, upload<br/>ontology-ttl :284-285 and ontology-jsonld :291-292
    BLD->>REL: main only, needs validate-source + build-ontology :304
    REL->>REL: assemble five assets + SHA256SUMS :324-332,<br/>create or move release tag ontology-latest
    REL->>REL: upload --clobber, verify the public<br/>download path diffs SHA256SUMS :365-368
    Note over SRV: ADR-2106 pull model — at boot and every<br/>ONTOLOGY_PULL_INTERVAL_SECS: GET index.jsonld, compare<br/>visionflow:buildSha and visionflow:sourceSha with the pod's, and if moved GET<br/>SHA256SUMS plus the files, verify every digest, then write<br/>via Storage in order: containers, .acl only if absent,<br/>content, manifest last. Fail-open. see ES-08.11
    SRV->>REL: GET releases/download/ontology-latest/{index.jsonld, SHA256SUMS, ...}
    alt vars.SOLID_POD_URL set (a pod a runner can reach)
        BLD->>DEP: needs build-ontology, ref main :376
        DEP->>DEP: backup current pod index for rollback :399
        DEP->>DEP: PUT visionflow.ttl, context/ontology/index jsonld :417-434, verify :443-459
        alt deployment fails
            DEP->>DEP: rollback to backed-up index :475
        end
        DEP->>WS: deployment_status success
        WS->>WS: POST to SOLID_POD_URL/.notifications :526, PATCH index.jsonld :531
    else unset (default: the in-process pod is not reachable from a hosted runner)
        REL->>MIS: ::notice: push deploy skipped, pull model in effect :622-625
    end
    GH->>PRP: pull_request only: comment with stats and SHAs :562-611
    Note over VAL,WS: RESOLVED ADR-2098 (2026-09-05): SOLID_POD_URL now<br/>defaults to the loopback /solid scope :48-50 — what the embedded<br/>solid-pod-rs serves in-process (ADR-032 M3). The POST to<br/>/.notifications is annotated a best-effort no-op there:<br/>that path is a GET WebSocket upgrade
    Note over BLD: RESOLVED — ADR-2112/ADR-2113 replaced the Python<br/>pipeline.build converter (0 owl:Class from 380 pages on run<br/>34045488066, itself a fix of two earlier Logseq-only jobs) first<br/>with a vault-owned Python step, now with an IN-REPO Rust build:<br/>cargo build -p vault then ./target/release/vault build against<br/>the checked-out vault-source/, no external converter left to own
    Note over GH,BLD: DRIFT resolved — the source repo is still checked out (path<br/>renamed logseq-source to vault-source, trigger renamed logseq-sync<br/>to corpus-sync :20-21) but it no longer runs anyone's pipeline in<br/>place: `vault build` reads it via --repo and writes output/vault.<br/>scripts/ontology/pack-pod-resources.py remains this repo's own<br/>script, unchanged, run from the repo root :245. see VG-04.1, KG-05
    Note over GH,BLD: 2026-10 (805219679) — a corpus-sync dispatch may name the<br/>source repo :42 and sha :106, and the build job checks out exactly<br/>the sha validate-source detected :166, so a build can no longer<br/>publish a later corpus commit than the one it validated
    Note over SRV,MIS: RESOLVED ADR-2106 (2026-09-06): deploy-jss had never<br/>run and could not from a hosted runner — delivery inverted to a<br/>boot pull from the ontology-latest release. see ES-08.11
```

## ES-09.14 VisionClaw xr-godot-ci.yml — gdext + GUT under a GL display, Quest 3 advisory
```mermaid
sequenceDiagram
    autonumber
    participant GH as pull request touching xr-client/** or manual dispatch (the push trigger was removed 2026-11-10)<br/>xr-godot-ci.yml:25-38
    participant RT as xr-rust-tests job<br/>xr-godot-ci.yml:54 blocking
    participant GUT as gut-headless job<br/>xr-godot-ci.yml:78 blocking
    participant Q3 as quest3-android job<br/>xr-godot-ci.yml:137 advisory

    GH->>RT: cargo test -p visionclaw-xr-gdext --all-features :74
    RT->>RT: cargo test -p visionclaw-xr-presence :76
    GH->>GUT: cargo build -p visionclaw-xr-gdext (debug cdylib) :95
    GUT->>GUT: launch-guard tests, then fail if the library Godot loads is older<br/>than its Rust source :96-99
    GUT->>GUT: install GL display deps, xvfb and mesa :100-101
    GUT->>GUT: install Godot 4.6.1-stable, env GODOT_VERSION :50, :102-108
    GUT->>GUT: check the committed GUT pin is 9.6.1, not a fetched tag :109-113
    GUT->>GUT: godot --headless --editor --xr-mode off --import, log kept :119
    GUT->>GUT: xvfb-run godot gl_compatibility -s gut_cmdln.gd, log kept :121-123
    GUT->>GUT: GUT guard fails on any parse or compile error or a short script count :124-127
    Note over GUT: the job is still named gut-headless :78 but the suite runs<br/>under a virtual GL display, because the world-space HUD fit<br/>gate needs real GL font metrics :120
    GH->>Q3: continue-on-error true :141
    Q3->>Q3: cargo ndk build aarch64-linux-android :165
    Q3->>Q3: export Quest 3 arm64 APK :181-189
    Q3->>Q3: APK size gate, fail if greater than 80MB :190-195
    Note over Q3: advisory until first green hosted run,<br/>then promote to blocking (per file header comment)
```

## ES-09.15 agentbox invariants.yml — security invariant gates including loopback-publish
```mermaid
sequenceDiagram
    autonumber
    participant GH as push or pull request touching compose, ADRs, the egress policy, entrypoint,<br/>management-api, lib, scripts, tests/security, flake, skills, mcp,<br/>schema or the manifest invariants.yml:6-118
    participant J as invariants job<br/>invariants.yml:128

    GH->>J: checkout fetch-depth 0 :135
    J->>J: runtime-delivery.test.cjs, non-disruptive preparation<br/>and scoped activation (ADR-2132) :146
    J->>J: check-seccomp.sh :149
    J->>J: check-nnp.sh (no-new-privileges) :152
    J->>J: check-ports-loopback.sh :155
    Note over J: ADR-2013: sweeps EVERY docker-compose*.yml via a real<br/>YAML parser (check-ports-loopback.mjs), replacing an<br/>awk line-walker that missed nested-mapping publishes
    Note over J: SANCTIONED allowlist: 9096 sovereign ingress,<br/>voice 8443/8444, browsercontainer 5903/8931/9222,<br/>gui-tools 5905/9876/9877, xr-runtime 5904
    Note over J: DIVERGENCE: implementation_status partial — the<br/>dated closeout says the scanner does not yet cover<br/>every equivalent publish syntax form
    J->>J: check-listeners.test.mjs :158, the tests/security/**<br/>trigger path guards this gate's own unit tests
    J->>J: session-plane retirement test, AoE removed and no script attaches<br/>the operator client (the file keeps its aoe-launch name) :160-161
    J->>J: session-launch-contract (ADR-2153, live) :163-164, session-plane notify :166-167,<br/>then trust-seed :169-170, every seed resolves to a launch the pinned CLIs accept and every Claude config<br/>store a session reads is trusted
    J->>J: check-db-password.sh :173
    J->>J: check-secret-not-in-env.sh :176
    J->>J: custody W2 - env-secret-inventory --check and its unit<br/>tests, plus compose-role-env (W10) :183, :188, :190 - see ES-09.24
    J->>J: RC-X1-01 to RC-X1-07 runtime contracts, the real entrypoint<br/>functions against a scratch root - :197, :199, :201, :203,<br/>:205, :207, :209 - see ES-09.24
    J->>J: check-single-metrics.js :212
    J->>J: check-no-npx-latest.sh (ratchet) :215
    J->>J: lint-skills.sh :218
    J->>J: VisionCraft release source and host contract, which now also runs the visioncraft-deepsec, cutover and aqe config tests (the build-with-quality deepsec-gate unit test step is gone) :229-238
    J->>J: check-manifest-catalogue.js (ADR-039 gate-path parity) :245
    J->>J: ruflo-console bake + boot projection test :248
    J->>J: check-declared-vs-running against the dated runtime snapshot,<br/>plus its unit tests :255, :257
    J->>J: check-no-logseq-paths.sh :260
    Note over J: ADR-2028: vault.root is the single corpus path<br/>authority, greps for hard-coded workspace/logseq<br/>outside docs/archive and docs/adr exemptions
    J->>J: federation-fixture-check (ADR-2025) :267
    J->>J: protocol-registry-lint and its unit tests, kind rows<br/>owned and no collisions (ADR-2105) :272, :274
    J->>J: adr-index-gen.js docs/adr --check :282
    J->>J: adr-index-gen.js docs/adr --check-index (ADR-2001) :290
    J->>J: N-7 render-egress-register.js --check :297 - see ES-09.24
    J->>J: adr-ratchet self-test :300, evidence-gate self-test (VisionFlow ADR-2007) :303,<br/>then adr-ratchet.sh over the push range :305-308
    Note over J: the ratchet script is byte-identical to the host copy at both stamps<br/>(same sha256, agentbox/scripts/adr-ratchet.sh:45 carries the same 2026-10-20<br/>end date) so both ledgers run one rule. The copy at VisionClaw's agentbox<br/>gitlink 4f41f77be is byte-identical too. see ES-09.12
    J->>J: check-crate-licensing.sh :311
    Note over J: DRIFT resolved — the vault-frontmatter unit test step<br/>(node --test mcp/servers/lib/__tests__/*.test.js) is GONE:<br/>ADR-2107/ADR-2108 deleted the V2 frontmatter writer it gated<br/>along with the ontology write path. The workflow comment now names<br/>ontomark validate as the successor contract check, run against the<br/>corpus not a helper (invariants.yml:276-279).
```

**Open:** ADR-2153 has replaced Agent of Empires and is live, with implementation still recorded partial (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`): the supervisor runs `[program:agentbox-sessions]` and no longer `aoe serve` (`../project/agentbox/flake.nix:2963`), and the invariants job runs the contract and notify steps (`../project/agentbox/.github/workflows/invariants.yml:163-167`) while the surviving regression test keeps its `aoe-launch` filename (`../project/agentbox/.github/workflows/invariants.yml:160-161`).

## ES-09.16 agentbox contract-tests.yml — adapter contract suites
```mermaid
sequenceDiagram
    autonumber
    participant GH as PR touching adapters, the sovereign signer,<br/>role-secret and sidechain paths<br/>contract-tests.yml:5-55
    participant C as contract job<br/>contract-tests.yml:117
    participant CU as custody job (ADR-2122)<br/>contract-tests.yml:197

    GH->>C: setup Node 22 (matches runtime image) :125-130
    C->>C: npm ci in management-api/ contract-tests.yml:134<br/>and repo-root npm ci --ignore-scripts contract-tests.yml:141
    C->>C: npx jest ../tests/contract/ --testPathPatterns contract filename filter<br/>contract-tests.yml:145 (plural since jest 30 dropped the singular flag, 23e5818a6)
    C->>C: node:test - chain identity route, pods signer as the sovereign identity,<br/>the ROLE-SECRET loader and its consumers (custody W2), the sidestr<br/>rail, sidechain tip-age and witness - contract-tests.yml:149-181
    C->>C: upload contract-test-results artifact contract-tests.yml:183-189
    Note over C: every durable-state integration rides one of five<br/>adapter slots (beads, pods, memory, events, orchestrator)<br/>and must pass tests/contract/ for all implementation classes
    GH->>CU: nostr-pod-bridge, custody config, role-accounts, the<br/>rehearsal and the identity-port test are trigger paths :42-55
    CU->>CU: rebuilds the layout lib/nostr-pod-bridge.nix pins: agentbox<br/>plus the nostr-rust-forum and solid-pod-rs siblings at their<br/>forumRev/solidRev revisions :210-229
    CU->>CU: cargo test --locked services/nostr-pod-bridge - the identity port<br/>(transport, SO_PEERCRED ACL, receipts) and the Rust role-isolation<br/>readers - contract-tests.yml:255-257
    CU->>CU: builds the bridge binary and runs the JS identity-port end to end,<br/>failing on any skip :259-270
    CU->>CU: role-isolation rehearsal decides correctly against a scratch root<br/>with fake /proc, supervisorctl, stat, docker, sudo and curl :276-277
    Note over C,CU: RESOLVED 2026-10-03 (by 6466e3931): the role-secret loader suite<br/>that invariants.yml points at now runs here - contract-tests.yml:163-165<br/>- and the custody job executes the identity port and the rehearsal<br/>that previously ran only on a developer's box. see ES-09.24
```

## ES-09.17 agentbox manifest-validate.yml — config validator and TUI round-trip
```mermaid
sequenceDiagram
    autonumber
    participant GH as PR touching agentbox.toml, schema, the manifest crate, tests/config,<br/>config/sidechain, config/role-accounts.json, config/lib, the entrypoint<br/>manifest-validate.yml:18-32
    participant V as validate job<br/>manifest-validate.yml:64

    GH->>V: setup Node 20, Python 3.11, Rust stable :71-89
    V->>V: cargo build --release services/agentbox-manifest :92
    V->>V: node agentbox-config-validate.js agentbox.toml :98
    loop each tests/tui/fixtures/valid-*.toml
        V->>V: agentbox-manifest tui-read fixture -> state.json :109
        V->>V: agentbox-manifest tui-write state.json -> out.toml :110
        V->>V: agentbox-config-validate.js out.toml :111
    end
    loop each tests/tui/fixtures/invalid-*.toml
        V->>V: assert failure with the expected E-code :115-130
    end
    V->>V: assert JSON Schema well-formed :133
    V->>V: ADR-2122 - the two supervisor configs cannot drift, the /run/secrets<br/>delivery tree, at-rest custody migrate/revert, and flag off boots as<br/>today :135-154 - see ES-09.24
    V->>V: sidechain-genesis.test.sh — document invariants, P21<br/>mainnet gate, no-key-in-git (ADR-2103) :156-159
    V->>V: mirror chain-event, producer parent gates, catalogue gates and<br/>mirror retry, the dream-inbox reminder :161-176
    V->>V: assert all W-codes route to warnings :178-190
    V->>V: assert E021 fires when exception block missing :192-210
    Note over V: same validator the TUI runs on every section<br/>transition and the flake evaluator runs at build time
```

## ES-09.18 agentbox ci.yml aggregate gate, and the structurally-identical workflow family
```mermaid
sequenceDiagram
    autonumber
    participant GH as PR / push main<br/>agentbox/.github/workflows/ci.yml:15-20
    participant AGG as ci-passed job<br/>agentbox/.github/workflows/ci.yml:31

    GH->>AGG: wait-on-check-action, poll every 20s :40-44
    AGG->>AGG: require ShellCheck error, gitleaks,<br/>agentbox config validate + TUI round-trip :50
    alt all required checks succeed or skipped
        AGG->>GH: CI passed :53-56
    else any required check fails
        AGG->>GH: aggregate gate fails, branch protection blocks merge
    end
    Note over AGG: structurally-identical family (each is its own workflow<br/>file, one job, checkout+setup+run+upload pattern) —<br/>flake-check.yml (Nix eval x86_64/aarch64, statix lint)<br/>secret-scan.yml (gitleaks), shellcheck.yml (severity matrix)<br/>image-scan.yml (Trivy HIGH/CRITICAL gate + SBOM CycloneDX/SPDX)<br/>deepsec.yml (deepsec-gate against PR diff, Anthropic route)<br/>tui-tests.yml (cargo clippy+test services/agentbox-manifest)<br/>build-multi-arch.yml (Nix image build, GHCR push, manifest list)<br/>nix-flake-update.yml (scheduled flake update, branch always pushed with<br/>GITHUB_TOKEN, PR opened only if NIX_FLAKE_UPDATE_TOKEN is set, else a<br/>tracking issue — org disallows Actions-created PR approval)<br/>release.yml (CHANGELOG-derived GitHub Release body)
    Note over AGG: RESOLVED: the stale copy of the pre-2026-09-06 VisionClaw ontology-publish.yml<br/>is deleted (agentbox 7b2811272) and absent at this stamp. Two workflows<br/>outside this family are new: adr-checker.yml (the canonical ledger checker's<br/>consumer contract) and security-vex.yml (VEX contracts, pull requests only)
```

## ES-09.19 End-to-end artefact flow — source to running container
```mermaid
flowchart LR
    SRC["Source on host bind<br/>/mnt/mldata/githubs/AR-AI-Knowledge-Graph"]
    CIGATE["CI gates (ES-09.11 to ES-09.18)<br/>rust-fmt, rust-cpu, client, docs-ci,<br/>invariants, contract-tests, manifest-validate"]
    DOCKERBUILD["docker compose build<br/>Dockerfile.unified or Dockerfile.production<br/>host tmux tab 6 only (ES-09.1)"]
    IMAGE["Built image<br/>cachyos-v3 base + compiled binary or dev toolchain"]
    COMPOSEUP["docker compose --profile dev|production up<br/>docker-compose.unified.yml"]
    CONTAINER["visionclaw_container: supervisord manages nginx, rust-backend, vite-dev<br/>visionclaw_prod_container: production-startup.sh runs the backend and nginx itself,<br/>no supervisord (ES-09.7)"]
    NGINXROUTE["nginx.dev.conf or nginx.production.conf<br/>route table (ES-09.9)"]
    HEALTH["/api/health, /readyz<br/>docker-compose.unified.yml:236, docker-compose.unified.yml:325"]

    SRC --> CIGATE
    CIGATE --> DOCKERBUILD
    DOCKERBUILD --> IMAGE
    IMAGE --> COMPOSEUP
    COMPOSEUP --> CONTAINER
    CONTAINER --> NGINXROUTE
    CONTAINER --> HEALTH

    subgraph AGENTBOXPARALLEL["Parallel estate path: agentbox"]
        ABGATE["agentbox invariants.yml, contract-tests.yml,<br/>manifest-validate.yml (ES-09.15 to ES-09.17)"]
        ABFLAKE["agentbox.sh rebuild -> flake.nix<br/>(ES-09.10), host tmux tab 6 only"]
        ABIMAGE["agentbox image<br/>Nix-composed, supervisord PID1 root"]
    end
    SRC --> ABGATE --> ABFLAKE --> ABIMAGE
    ABIMAGE -.->|"submodule pointer bump<br/>records the pinned commit (ES-09.10)"| SRC
```

## ES-09.20 The compile-time submodule input — why the dev image mounts agentbox/schema
```mermaid
sequenceDiagram
    autonumber
    participant CB as "cargo build (dev container)"<br/>rust-backend-wrapper.sh:66
    participant URI as visionclaw-server src/uri<br/>src/uri/mod.rs:665
    participant FK as federation-kinds artefact<br/>agentbox/schema/federation-kinds.json:2-4
    participant CMP as dev compose mounts<br/>docker-compose.unified.yml:203
    participant BR as "BC20 bridge (agentbox, JS)"

    Note over URI,FK: ADR-2061 — federation-kinds.json is the SINGLE versioned<br/>authority for which urn:agentbox kinds cross the boundary.<br/>cross_from_agentbox derives its closed map from it — the JS bridge<br/>reads the same bytes at load. Neither side transcribes the list.
    URI->>FK: "const FEDERATION_KINDS_JSON = include_str!(\"../../agentbox/schema/federation-kinds.json\")"<br/>src/uri/mod.rs:665
    Note over URI: include_str! is COMPILE time, so the file must exist under /app<br/>or the lib cannot build at all — not a runtime read.
    CMP->>CB: "bind ${HOST_PROJECT_ROOT}/agentbox/schema -> /app/agentbox/schema:ro"<br/>docker-compose.unified.yml:203
    CB->>URI: compiles, artefact bytes baked into the binary
    FK-->>BR: EXTERNAL — the same file is read at load by the agentbox<br/>management-api bridge. see AB-17 and ES-03.1
    Note over CMP: INVARIANT — mount schema/ ONLY, never the whole submodule.<br/>agentbox carries 262 files ADR-2008 counts as build inputs, so a<br/>full mount would turn every agentbox bump into a spurious ~12 min<br/>visionclaw rebuild — schema/ is JSON alone and matches no input glob<br/>(docker-compose.unified.yml:194-202)
    Note over CB: RESOLVED 2026-09 — before this mount the dev image (which carries<br/>no source of its own) failed every build with<br/>the missing federation-kinds.json include<br/>docker-compose.unified.yml:203
```

## ES-09.21 scripts/launch.sh — env-file resolution and the two divergent code paths
```mermaid
flowchart TB
    START["launch.sh main()<br/>scripts/launch.sh:1555<br/>PROJECT_ROOT = dirname SCRIPT_DIR :15"]

    subgraph SANCTIONED["Sanctioned path — up / rebuild dev|prod"]
        LOADENV["load_env_config<br/>looks for .env.$ENVIRONMENT :193-194"]
        INGR["prod: resolve_prod_ingress reads VISIONCLAW_INGRESS,<br/>lan or tunnel, anything else exits :200, function :177-190"]
        PRODREQ["production: .env.prod REQUIRED<br/>refuses four forbidden dev settings :205-210<br/>demands concrete values :225-232, the tunnel token only<br/>for tunnel, CORS_ALLOWED_ORIGINS for lan :215-223<br/>hard error if absent :241-244"]
        DEVFALL["dev: falls back to plain .env :246-251<br/>hard error if neither exists :253-254"]
        HPR["HOST_PROJECT_ROOT export :360, :368, :376<br/>— the DinD path translation the schema<br/>mount in ES-09.20 depends on"]
    end

    subgraph LEGACY["rebuild_agent_container — legacy MAD path"]
        CD["cd $PROJECT_ROOT/multi-agent-docker :1021"]
        MK[".env absent -> cp env.example .env :1026-1029<br/>then blocks on an interactive read :1031"]
    end

    START --> LOADENV
    LOADENV --> INGR --> PRODREQ
    LOADENV --> DEVFALL
    LOADENV --> HPR
    START -->|"launch.sh rebuild-agent, scripts/launch.sh:1632-1636"| CD --> MK

    FIX["RESOLVED 2026-09-06 — the auto-create branch tested for a dotted<br/>.env.example that has never existed in the tree, so it always fell<br/>through to the error exit. It now copies env.example, the template<br/>that actually ships (scripts/launch.sh:1026-1029, env.example)."]
    DIV["DIVERGENCE — multi-agent-docker/ is not present in this checkout,<br/>so rebuild_agent_container cds into a missing directory before it<br/>reaches the fixed branch. The repaired .env creation is correct but<br/>currently unreachable; the MAD stack is the deprecated predecessor<br/>whose only surviving trace is the mad-workspace volume. see ES-01.4"]
    INV["INVARIANT ADR-2027 — profile selection is env-file driven, and the<br/>production branch fails closed: a missing or dev-contaminated<br/>.env.prod aborts the launch rather than defaulting. see ES-10"]

    MK --> FIX
    CD --> DIV
    PRODREQ --> INV
```

## ES-09.22 The agentbox invariants gate grew six checks, and each one names the record it enforces
```mermaid
flowchart TB
    TRIG["push or PR touching compose, ADRs, entrypoint, management-api,<br/>lib, scripts, tests/security, flake.nix, skills, mcp, schema<br/>or the manifest<br/>agentbox/.github/workflows/invariants.yml:6-118"]
    JOB["the single invariants job<br/>agentbox/.github/workflows/invariants.yml:128"]
    TRIG --> JOB

    subgraph SKILLS["Skill-estate gates — required before a rebuild"]
        S1["lint-skills.sh, the structure gate<br/>agentbox/.github/workflows/invariants.yml:218"]
        S2["skill-count-check, the SINGLE count authority<br/>agentbox/.github/workflows/invariants.yml:219-220"]
        S3["gen-routing-table --check, routing-table freshness<br/>generated from frontmatter<br/>agentbox/.github/workflows/invariants.yml:222"]
    end
    JOB --> SKILLS

    subgraph CONTRACT["Contract gates"]
        C1["research-gates tests, the deep-research quote, citation<br/>and independence contract<br/>agentbox/.github/workflows/invariants.yml:241-242"]
        C2["check-manifest-catalogue, ADR-039 gate-path parity<br/>agentbox/.github/workflows/invariants.yml:245"]
        C3["federation-fixture-check, the cross-repo identifier<br/>contract from the agentbox side<br/>agentbox/.github/workflows/invariants.yml:266-267"]
    end
    JOB --> CONTRACT

    DEBT["DEBT the workflow records against itself — skill-count-check was<br/>RED and UNWIRED, and the federation fixture check was governed by<br/>a record but never run by anything, found in a script audit. The<br/>fixture's whole point is that both repositories assert the SAME<br/>table rather than two tables that happen to agree, which an<br/>ungated check cannot deliver.<br/>agentbox/.github/workflows/invariants.yml:219,<br/>agentbox/.github/workflows/invariants.yml:262-265"]
    S2 --> DEBT
    C3 --> DEBT

    CAT["INVARIANT ADR-039 — a new manifest gate must arrive with a<br/>CATALOGUE entry carrying an honest apply class, and the parity<br/>check above is what enforces it. Recent module entries: Claude Code<br/>permissions (ADR-2116) and instruction tiers (ADR-2118) and the<br/>collaboration floor (ADR-2144) at boot, claude-cred-sync at<br/>rebuild, skill-router-cascade (ADR-2095) and<br/>routing-teacher-labels (ADR-2110, proposed) at boot, vault-cli<br/>(ADR-2107/2108) at rebuild<br/>agentbox/management-api/lib/system-manifest.js:256, :262, :265, :268, :351, :354, :409"]
    HONEST["2026-10-02 — two entries show the honesty rule working. jev-compaction<br/>moved from boot to REBUILD when factrail landed (ADR-2121), because<br/>the binary and plugin are now gated in flake.nix,<br/>agentbox/management-api/lib/system-manifest.js:271-272. A new sidechain<br/>entry is rebuild-class with three gates,<br/>agentbox/management-api/lib/system-manifest.js:388-389, and stateOf<br/>lets a false parent gate dominate its child gates,<br/>agentbox/management-api/lib/system-manifest.js:462-464. 2026-10-29: interaction-plane moved from boot to REBUILD with ADR-2153,<br/>agentbox/management-api/lib/system-manifest.js:74-76, and nineteen System One consumer entries arrived,<br/>sixteen boot and three rebuild (solid pod, ontomark, vault rank), agentbox/management-api/lib/system-manifest.js:291-347. 2026-10-30: system-one-network arrived at BOOT,<br/>agentbox/management-api/lib/system-manifest.js:280-282. 2026-10-03: the<br/>role-isolation entry first arrived at BOOT, because both supervisor configs<br/>ship in every image; it is now rebuild-class, because the devuser sudo route is<br/>removed at build, agentbox/management-api/lib/system-manifest.js:97-99"]
    CAT --> HONEST
    C2 --> CAT

    CLASS["A boot-class entry is projected by the entrypoint, so a flip takes effect<br/>on the next container restart with no image rebuild. A rebuild-class entry,<br/>such as claude-cred-sync or vault-cli above, needs ./agentbox.sh rebuild.<br/>The three classes are defined at system-manifest.js:28-30."]
    CAT --> CLASS
```

## ES-09.23 The host ADR ledger gates its own staleness, and the gate is opt-in per record
```mermaid
sequenceDiagram
    autonumber
    participant CI as docs-ci workflow<br/>project/.github/workflows/docs-ci.yml:33
    participant GEN as adr-index-gen check<br/>project/scripts/adr-index-gen.js:186
    participant REC as one ADR record's frontmatter
    participant GIT as git

    CI->>GEN: node scripts/adr-index-gen.js docs/adr --check
    GEN->>REC: does it declare verified_paths
    alt no verified_paths
        REC-->>GEN: nothing to check, the loop continues, project/scripts/adr-index-gen.js:207
        Note over GEN: The staleness gate is OPT-IN per record. The abbreviated-SHA<br/>warning fires only for a record that HAS verified_paths,<br/>project/scripts/adr-index-gen.js:204-206, so a record without them gets no nudge
    else verified_paths declared
        REC-->>GEN: verified_commit plus the governed paths
        GEN->>GIT: does the commit exist, and is it an ancestor of HEAD
        alt either answer is no
            GIT-->>GEN: no
            GEN-->>CI: fail, project/scripts/adr-index-gen.js:210-213
        else both yes
            GEN->>GIT: diff those paths from the commit to HEAD
            alt anything changed
                GIT-->>GEN: a non-empty path list
                GEN-->>CI: STALE, re-verify and bump verified_commit,<br/>project/scripts/adr-index-gen.js:217-218
            else nothing changed
                GIT-->>GEN: empty
                GEN-->>CI: the claim still holds
            end
        end
    end

    Note over GEN,CI: DEBT (historical) — at visionclaw f223bbd40 FIFTEEN host records failed this<br/>gate, each naming the governed path that moved under it. This audit did not<br/>re-run the gate at the stamp, so the current count is not known,<br/>project/scripts/adr-index-gen.js:217-218.
    Note over CI: INVARIANT — the gate is wired into the docs workflow rather than<br/>left to a habit: invalid frontmatter, asymmetric supersession<br/>edges and stale verification claims all fail the build,<br/>project/.github/workflows/docs-ci.yml:33. see VF-01
    Note over CI: SECOND AND THIRD LEDGER GATES — the same job runs the ratchet's<br/>evidence-gate self-tests and then the ratchet itself, project/.github/workflows/docs-ci.yml:34-43.<br/>Staleness asks whether a claim still holds, the ratchet asks whether the<br/>proposed backlog grew, project/scripts/adr-ratchet.sh:115-126, and whether a<br/>closing record carries its evidence, project/scripts/adr-ratchet.sh:81-113. see ES-09.12
    Note over REC: Presence of verified_paths is what ARMS the gate, which is why<br/>a record can be honestly unverified without failing CI,<br/>project/scripts/adr-index-gen.js:207. A proposed or rejected record may also<br/>leave verified_commit empty, project/scripts/adr-index-gen.js:118-121.
```

**Drift (comment vs code):** the block comment says records without `verified_paths` "only get a soft nudge toward full 40-char SHAs" (`../project/scripts/adr-index-gen.js:190-191`), but the nudge is guarded on `verified_paths` being a non-empty list (`../project/scripts/adr-index-gen.js:204`), and those records skip the block at `:207`. A record with no `verified_paths` therefore gets no message at all. Either the comment or the guard should change; source tracing alone cannot say which one the authors meant.

## ES-09.24 Custody X-1 step 1 in agentbox CI — what is gated, and the two suites nothing runs

```mermaid
flowchart TB
    subgraph INV["invariants.yml, every push touching scripts, lib, the entrypoint or the egress policy"]
        I1["env-secret-inventory --check: every env name the boot path and the<br/>role-secret consumers read sits in exactly one class, names only<br/>agentbox/.github/workflows/invariants.yml:182-183"]
        I2["its unit tests, compose-role-env W10, then the RC-X1-01..07<br/>runtime contracts - the real entrypoint functions against a<br/>scratch root, no Docker, no root<br/>agentbox/.github/workflows/invariants.yml:187-209"]
        I3["N-7: the SECURITY-profiles egress table must match egress-policy.json,<br/>and an unmarked gap fails - agentbox/.github/workflows/invariants.yml:296-297"]
    end
    subgraph MV["manifest-validate.yml, on the role table, config/lib or the entrypoint"]
        M1["the two supervisor configs cannot drift: only role programs' user=<br/>and environment= differ - agentbox/.github/workflows/manifest-validate.yml:135-140"]
        M2["/run/secrets delivery builds the right owner and mode tree, chown stubbed,<br/>plus at-rest custody migrate/revert - agentbox/.github/workflows/manifest-validate.yml:142-151"]
        M3["with role_isolation off the boot is today's<br/>agentbox/.github/workflows/manifest-validate.yml:153-154"]
    end
    subgraph TUI["tui-tests.yml, on services/agentbox-manifest and services/agentbox-sessions"]
        T1["cargo test agentbox-manifest, which carries the role_accounts unit tests<br/>agentbox/.github/workflows/tui-tests.yml:77"]
        T2["NEW with ADR-2153: clippy and cargo test agentbox-sessions on a private tmux server,<br/>then the Node/Rust launch-contract parity<br/>agentbox/.github/workflows/tui-tests.yml:105-122"]
        T3["NEW 2026-10-30: the tab0-bridge follows the session plane, readiness against a fake daemon on a unix socket<br/>agentbox/.github/workflows/tui-tests.yml:124-147"]
        T1 --> T2
        T2 --> T3
    end
    subgraph CT["contract-tests.yml custody job, NEW 2026-10 for ADR-2122"]
        C1["cargo test nostr-pod-bridge: the identity port's transport, SO_PEERCRED<br/>ACL and receipts, and the Rust role-isolation readers<br/>agentbox/.github/workflows/contract-tests.yml:255-257"]
        C2["identity port end to end, JS client against the built bridge binary,<br/>a skip fails the job - agentbox/.github/workflows/contract-tests.yml:265-270"]
        C3["role-isolation rehearsal decides correctly against a scratch root<br/>agentbox/.github/workflows/contract-tests.yml:276-277"]
        C1 --> C2
    end
    subgraph BC["browsercontainer.yml, for W9"]
        B1["best-effort fetch of the pinned Podkey artefact, continue-on-error<br/>agentbox/.github/workflows/browsercontainer.yml:50-54"]
        B2["fetch-podkey pin, sha256 and content, launch flags, podkey-ctl<br/>load-once, the profile volume - agentbox/.github/workflows/browsercontainer.yml:56-66"]
        B1 --> B2
    end
    subgraph NONE["Custody tests no workflow runs at this revision"]
        N1["the G-5 dual-admit key-split suite<br/>agentbox/tests/sovereign/g5-key-split.node-test.js:3"]
        N2["the baked-producer test, the producer runs the bake<br/>agentbox/tests/config/sidechain-producer-baked.test.sh:2"]
    end
    INV ~~~ MV
    MV ~~~ TUI
    TUI ~~~ CT
    CT ~~~ BC
    BC ~~~ NONE
    NONE --> D["RESIDUAL DEBT: the G-5 dual-admit and the baked-producer bake have<br/>no CI step; everything else custody needs - the port's uid admission,<br/>the role-only reads, PATH and the one-shot Stage B - is now executed<br/>by invariants and the custody job"]
```

**Debt (custody tests vs CI), RESOLVED at agentbox 6466e3931 except two suites:** through 2026-10-03 only the env-class inventory, RC-X1-06, the supervisor-drift, delivery and flag-off boot tests, the manifest crate's unit tests and the browsercontainer tests were wired. Now the RC-X1-01..07 runtime contracts run in invariants (`../project/agentbox/.github/workflows/invariants.yml:196-209`), and a `custody` job in contract-tests builds the bridge and runs the identity port's Rust and JS suites plus the rehearsal, failing on skips (`../project/agentbox/.github/workflows/contract-tests.yml:255-277`). The G-5 dual-admit key-split suite (`../project/agentbox/tests/sovereign/g5-key-split.node-test.js:3`) and the baked-producer test (`../project/agentbox/tests/config/sidechain-producer-baked.test.sh:2`) still have no step.

**Drift (invariants.yml comment vs contract-tests.yml), RESOLVED:** the invariants job's env-secret-inventory label used to claim the role-secret loader "runs in contract-tests" with no such step existing — the workflow now records that honestly (`../project/agentbox/.github/workflows/invariants.yml:184-186`), and the step it names is real: contract-tests runs the loader and its consumers (`../project/agentbox/.github/workflows/contract-tests.yml:163-165`).


**Drift (31 October audit):** Unset thresholds do not imply that every consumer accepts nothing. Codescape accepts an offered view command at confidence exactly one; see [VC-42](../visionclaw/42-codescape-command-policy-and-runtime-artefacts.md). The deployment topology retains its historical pins.
