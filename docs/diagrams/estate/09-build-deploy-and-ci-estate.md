---
id: ES-09
title: Build, deploy and CI estate — source to running container, every gate
area: estate
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [visionclaw:ADR-2008, visionclaw:ADR-2037, agentbox:ADR-2013, agentbox:ADR-2028]
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
  - ../project/src/main.rs
  - ../project/docs/adr/ADR-2008-dev-image-recompiles.md
  - ../project/docs/adr/ADR-2037-production-build-excludes-dev-auth.md
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
  - ../project/build.rs
  - ../project/env.example
  - ../project/src/uri/mod.rs
  - ../project/agentbox/schema/federation-kinds.json
  - ../project/agentbox/scripts/ci/check-ports-loopback.sh
  - ../project/agentbox/scripts/ci/check-ports-loopback.mjs
  - ../project/agentbox/scripts/ci/check-no-logseq-paths.sh
  - ../project/agentbox/docs/adr/ADR-2013-loopback-publish-except-9096.md
  - ../project/agentbox/docs/adr/ADR-2028-vault-manifest-path-authority.md
  - ../project/scripts/adr-index-gen.js
  - ../project/scripts/ontology/pack-pod-resources.py
  - ../project/scripts/launch.sh
  - ../project/scripts/start.sh
verified_commit: {visionclaw: dd82a07b0c54defc469a85e9d57fb15d29dc5e07, agentbox: 771d96ed5ac6f5daa1e78a60d109c130b9ef9b99}
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
    BUILDREQ -->|"resolves HOST_PROJECT_ROOT-relative<br/>bind-mount paths against ITS OWN (host-side)<br/>cwd/view, not the CC container path"| MISBIND["docker-compose.unified.yml:121<br/>bind source path mismatch"]:::wrong
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
    participant DC as visionclaw_container<br/>docker-compose.unified.yml:47
    participant W as rust-backend-wrapper.sh

    Dev->>CC: edit src/main.rs
    Note over CC: same bind mount as host path<br/>/mnt/mldata/githubs/AR-AI-Knowledge-Graph
    rect rgb(230,235,245)
    Note over T6,DD: container/host process boundary — build MUST cross here, never from CC
    Dev->>T6: tmux send-keys -t 6 ./scripts/launch.sh up dev Enter
    T6->>DD: docker compose --profile dev up -d
    DD->>DD: resolve HOST_PROJECT_ROOT bind mounts<br/>docker-compose.unified.yml:121-154
    DD->>DC: recreate container with correct host-side binds
    end
    DC->>W: supervisord starts program:rust-backend<br/>supervisord.dev.conf:20
    W->>W: needs_rebuild scripts/rust-backend-wrapper.sh:57
    alt source or Cargo manifest changed
        W->>W: cargo build --release --features gpu,ontology,dev-auth
        W->>W: write_build_stamp scripts/rust-backend-wrapper.sh:71
    else stamp up to date
        W->>W: skip cargo scripts/rust-backend-wrapper.sh:66
    end
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
    RUSTDEPS["rust-deps<br/>Dockerfile.unified:145<br/>COPY Cargo.toml/crates, cargo fetch:184,<br/>cargo build --release --features gpu:185"]
    RUSTBUILD["rust-builder<br/>Dockerfile.unified:191<br/>COPY src:194, cargo build --release --features gpu:214<br/>strip target/release/visionclaw-server:215"]
    NODEDEPS["node-deps<br/>Dockerfile.unified:222<br/>npm ci --prefer-offline --no-audit:234"]
    NODEBUILD["node-builder<br/>Dockerfile.unified:239<br/>npx vite build:248"]
    DEV["development target<br/>Dockerfile.unified:255<br/>FROM base — NO rust-builder/node-builder<br/>COPY src SOURCE (not binaries):289, COPY client:294<br/>ENTRYPOINT ./dev-entrypoint.sh at Dockerfile.unified:340"]
    PROD["production target<br/>Dockerfile.unified:348<br/>FROM cachyos-v3 fresh, NOT from base<br/>COPY --from=rust-builder binary:407<br/>COPY --from=node-builder dist:410<br/>USER appuser:428, ENTRYPOINT prod-entrypoint.sh at Dockerfile.unified:437"]

    BASE --> RUSTDEPS --> RUSTBUILD
    BASE --> NODEDEPS --> NODEBUILD
    BASE --> DEV
    RUSTBUILD --> PROD
    NODEBUILD --> PROD

    DIVERGE["DIVERGENCE: dev COPYs raw src and compiles<br/>at container start (dev-entrypoint.sh runs<br/>cargo build --release --features gpu,dev-auth:108);<br/>prod COPYs the pre-compiled rust-builder binary,<br/>no compile at container start"]
    DEV -.-> DIVERGE
    PROD -.-> DIVERGE
```

## ES-09.4 Dockerfile.production — cache-optimised 5-stage pipeline
```mermaid
flowchart LR
    TOOLCHAIN["toolchain<br/>Dockerfile.production:13<br/>ARG CUDA_ARCH=86:15, ENV CUDA_ARCH promoted:17-22<br/>cachyos-v3 pinned digest, rustup stable, node 20.18.3"]
    DEPS["deps<br/>Dockerfile.production:59<br/>FROM toolchain — writes stub src/main.rs + 6 stub bins<br/>at Dockerfile.production:74-85, stub build.rs at :89<br/>cargo fetch --locked:101 MUST pass; crate build :102-103 may fail"]
    CUDAPTX["cuda-ptx<br/>Dockerfile.production:108<br/>FROM toolchain — re-declares ARG CUDA_ARCH=86:110<br/>nvcc -ptx -arch sm_CUDA_ARCH:121"]
    FRONTEND["frontend<br/>Dockerfile.production:128<br/>FROM toolchain — npm ci:139, npx vite build:144"]
    BUILDER["builder<br/>Dockerfile.production:149<br/>FROM deps — COPY real src:159, cargo build --release:165<br/>COPY --from=cuda-ptx ptx:155"]
    RUNTIME["runtime (final, unnamed)<br/>Dockerfile.production:172<br/>fresh cachyos-v3 — NOT from toolchain<br/>COPY --from=builder binary:231, USER appuser:244<br/>ENTRYPOINT /app/start.sh at Dockerfile.production:246"]

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
    participant C2 as cuda-ptx-child-stage<br/>Dockerfile.production:96

    U->>U: ARG CUDA_ARCH=75 (:30, scoped to this stage only)
    U->>U: ENV CUDA_ARCH=CUDA_ARCH (:44, promotes ARG into ENV)
    U->>C1: FROM base AS rust-deps
    Note over C1: INVARIANT: ENV values set in a parent stage ARE<br/>inherited by a child FROM stage, ARG values are NOT
    C1->>C1: nvcc build.rs reads env CUDA_ARCH (inherited, correct)

    P->>P: ARG CUDA_ARCH=86 (:15, scoped to toolchain stage only)
    P->>P: ENV CUDA_ARCH=CUDA_ARCH (:21, promotes ARG into ENV)
    P->>C2: FROM toolchain AS cuda-ptx
    Note over C2: re-declares ARG CUDA_ARCH=86 (:110) redundantly,<br/>ENV already inherited from toolchain — both agree
    C2->>C2: nvcc -ptx -arch sm_CUDA_ARCH (:121)

    Note over U,P: DIVERGENCE (ADR-2008 vs ADR-2037): dev image builds<br/>cargo build --release --features gpu,dev-auth<br/>(scripts/rust-backend-wrapper.sh:66) — a release binary that<br/>still carries enforce_release_env_hygiene as a no-op stub<br/>(src/main.rs:169, cfg debug_assertions OR feature dev-auth)
    Note over U,P: ADR-2037 (proposed, implementation_status none): no CI or<br/>image-build assertion yet verifies a shipped release binary<br/>omits dev-auth — a mis-targeted pipeline could promote the<br/>stubbed-hygiene binary to production undetected
```

## ES-09.6 supervisord.dev.conf — program set and restart policy
```mermaid
stateDiagram-v2
    [*] --> supervisordRoot
    supervisordRoot: supervisord nodaemon supervisord.dev.conf:1
    supervisordRoot --> nginx
    supervisordRoot --> rustBackend
    supervisordRoot --> viteDev

    nginx: program nginx supervisord.dev.conf:8 autorestart true
    rustBackend: program rust-backend supervisord.dev.conf:19 command rust-backend-wrapper.sh
    viteDev: program vite-dev supervisord.dev.conf:34 npm run dev

    rustBackend --> rustBackendRetry: crash, startretries 3 supervisord.dev.conf:23
    rustBackendRetry --> rustBackend: restart, startsecs 10
    rustBackendRetry --> rustBackendFatal: exceeds startretries
    rustBackendFatal --> [*]

    nginx --> nginxRestart: crash, autorestart true
    nginxRestart --> nginx

    viteDev --> viteRestart: crash, autorestart true
    viteRestart --> viteDev

    note right of rustBackend
        environment RUST_LOG, MCP_TCP_PORT 9500
        supervisord.dev.conf 32
    end note
    note right of supervisordRoot
        unix_http_server /tmp/supervisor.sock
        supervisord.dev.conf 47
    end note
```

## ES-09.7 supervisord.production.conf — root PID1, appuser drop, restart policy
```mermaid
stateDiagram-v2
    [*] --> supervisordRootProd
    supervisordRootProd: supervisord user root supervisord.production.conf:1-3
    supervisordRootProd --> nginxProd
    supervisordRootProd --> rustBackendProd

    nginxProd: program nginx supervisord.production.conf:18 priority 10
    rustBackendProd: program rust-backend supervisord.production.conf:28 priority 20 visionclaw-server --port 4001

    rustBackendProd --> rustBackendProdRetry: crash, startretries 5 supervisord.production.conf:38
    rustBackendProdRetry --> rustBackendProd: restart, startsecs 10, stopasgroup killasgroup true
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
        cross-estate compare (agentbox flake.nix 2004):
        agentbox supervisord also runs as PID1 root,
        but every long-running program declares user devuser
        per-program — this file has no per-program user line
        so rust-backend and nginx run as the image USER (appuser)
    end note
```

## ES-09.8 Compose profiles — dev, production, loom, cloudflared
```mermaid
flowchart TB
    subgraph PROFILES["docker-compose.unified.yml services block"]
        DEVSVC["visionclaw<br/>docker-compose.unified.yml:54 target development<br/>profiles: development, dev docker-compose.unified.yml:178-180<br/>ports 3001,4000 docker-compose.unified.yml:160-161<br/>source-bind volumes docker-compose.unified.yml:117-157, docker.sock ro docker-compose.unified.yml:154"]
        PRODSVC["visionclaw-production<br/>docker-compose.unified.yml:187 Dockerfile.production<br/>profiles: production, prod docker-compose.unified.yml:252-254<br/>ports 3001 only docker-compose.unified.yml:228<br/>NO source mounts, NO docker.sock docker-compose.unified.yml:222-225"]
        CLOUDFLARED["cloudflared<br/>docker-compose.unified.yml:257 image cloudflare/cloudflared pinned digest<br/>profiles: production, prod docker-compose.unified.yml:276<br/>depends_on visionclaw OR visionclaw-production (optional)"]
        LOOM["loom<br/>docker-compose.unified.yml:303 image loom:rust (built outside this repo)<br/>profiles: loom docker-compose.unified.yml:363<br/>port 8090->8080 docker-compose.unified.yml:348<br/>hostname loom, alias ontology-loom docker-compose.unified.yml:349-353"]
    end
    subgraph EXTFILE["docker-compose.cloudflared.yml (standalone)"]
        CFSTANDALONE["cloudflared<br/>joins external visionclaw_network<br/>alias visionclaw-server:3001"]
    end
    NET["visionclaw_network (external, pre-created)<br/>docker-compose.unified.yml:366-369"]

    DEVSVC --> NET
    PRODSVC --> NET
    CLOUDFLARED --> NET
    LOOM --> NET
    CFSTANDALONE --> NET

    GATE["Deployment constraint: default dev and production<br/>both publish host port 3001, so concurrent activation collides.<br/>Compose profiles do not enforce mutual exclusion."]
    DEVSVC -.-> GATE
    PRODSVC -.-> GATE
```

## ES-09.9 nginx route tables — dev vs production upstreams
```mermaid
flowchart LR
    subgraph DEVNGINX["nginx.dev.conf — listen 3001 nginx.dev.conf:55"]
        DUPRUST["upstream rust_backend<br/>127.0.0.1:4000 :43-46"]
        DUPVITE["upstream vite_frontend<br/>127.0.0.1:5173 :48-51"]
        DAPI["/api/ -> rust_backend :66-67"]
        DWSS["/wss, /ws/speech, /ws/mcp-relay -> rust_backend :85,104,123"]
        DSOLID["/solid/, /pods/ -> rust_backend/api/solid/ :164,190"]
        DHMR["/vite-hmr, /@vite, /node_modules -> vite_frontend :251,263"]
        DROOT["/ -> vite_frontend (dev server, no static build) :287"]
    end
    subgraph PRODNGINX["nginx.production.conf — listen 3001 :85"]
        PUPRUST["upstream rust_backend<br/>127.0.0.1:4001 max_fails=0 :69-72"]
        PAPI["/api/ -> rust_backend :114-115"]
        PWS["wss / ws/speech / ws/mcp-relay / ws/hybrid-status -> rust_backend :234-235"]
        PSOLID["/solid/, /pods/ -> rust_backend/api/solid/ :169,214"]
        PSTATIC["/, *.html, *.js/css/png -> static /app/client/dist :269,282,295"]
        PHEALTH["/health, /healthz, /readyz -> rust_backend or static :304,313,319"]
    end
    LEGACY["nginx.conf (root, listen 4000 :82)<br/>generic template, NOT referenced by any<br/>Dockerfile/compose COPY — kept as reference only"]

    DAPI --> DUPRUST
    DWSS --> DUPRUST
    DSOLID --> DUPRUST
    DHMR --> DUPVITE
    DROOT --> DUPVITE
    PAPI --> PUPRUST
    PWS --> PUPRUST
    PSOLID --> PUPRUST
    PHEALTH --> PUPRUST

    DIVNOTE["DIVERGENCE: dev proxies ALL non-API routes to the<br/>Vite dev server (live HMR); prod serves a static<br/>client/dist build directly from nginx root, only<br/>API/WS routes reach the backend upstream"]
    DROOT -.-> DIVNOTE
    PSTATIC -.-> DIVNOTE
```

## ES-09.10 agentbox flake rebuild gate, and the submodule pointer-bump flow
```mermaid
flowchart TB
    subgraph FLAKE["agentbox/flake.nix — image composition"]
        NIXPKG["Nix package set<br/>e.g. toolchains.ruflo gate agentbox/flake.nix:505"]
        SUPTEXT["supervisorText string<br/>agentbox/flake.nix:2115-2147<br/>program blocks e.g. management-api, bootstrap-seal"]
        SUPWRITE["writeText supervisord.conf<br/>agentbox/flake.nix:3158-3163"]
    end
    subgraph TOML["agentbox/agentbox.toml — RUNNING config, not a template"]
        GATEKEY["gate key e.g. interaction_plane.enabled"]
    end
    subgraph MANIFEST["agentbox/management-api/lib/system-manifest.js"]
        CATALOGUE["CATALOGUE entry<br/>system-manifest.js:42<br/>gate, service, apply_class"]
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
        GITMODULES[".gitmodules<br/>submodule agentbox<br/>url github.com/DreamLab-AI/agentbox.git"]
        SUBSTATUS["git submodule status<br/>+89301ec7...5535 agentbox<br/>+ prefix: checkout differs from index"]
        SUBUPDATE["cd agentbox and git checkout NEW_SHA<br/>then git add agentbox (records gitlink)"]
        SUBCOMMIT["git commit records new gitlink SHA<br/>in the VisionClaw superproject tree"]
    end
    GITMODULES --> SUBSTATUS
    SUBSTATUS --> SUBUPDATE
    SUBUPDATE --> SUBCOMMIT
    IMAGE -.->|"agentbox image built from the checked-out<br/>submodule commit, independent pin"| SUBSTATUS
```

## ES-09.11 VisionClaw ci.yml — Rust CPU/client blocking gates, GPU excluded
```mermaid
sequenceDiagram
    autonumber
    participant GH as GitHubPush/PR<br/>project/.github/workflows/ci.yml:41-46
    participant FMT as rust-fmt job<br/>project/.github/workflows/ci.yml:61 blocking
    participant CPU as rust-cpu job<br/>project/.github/workflows/ci.yml:75 blocking
    participant CLI as client job<br/>project/.github/workflows/ci.yml:121 blocking
    participant GATE as dev-auth-release-gate job<br/>project/.github/workflows/ci.yml:142 blocking
    participant LINT as client-quality job<br/>project/.github/workflows/ci.yml:218 advisory
    participant PW as playwright job<br/>project/.github/workflows/ci.yml:248 manual only

    GH->>FMT: cargo fmt --all --check :72-73
    GH->>CPU: cargo build CPU_CRATES :103-104
    CPU->>CPU: cargo clippy CPU_CRATES --all-targets :105-108
    CPU->>CPU: cargo test CPU_CRATES :109-110
    CPU->>CPU: cargo clippy/test -p visionclaw-integration-tests :116-119
    GH->>CLI: npm ci :135-136, then npm run test (vitest) :140
    GH->>GATE: hermetic text assertion over the<br/>committed Dockerfiles/entrypoints :151-155
    GATE->>GATE: no --release cargo line may name dev-auth :162-167
    GH->>LINT: npm run lint (ESLint) :236-237, npx tsc --noEmit :238-239
    Note over LINT: continue-on-error true :224, never a required check
    opt workflow_dispatch only :254
        GH->>PW: npx playwright install :269, npm run test:e2e :271
    end
    Note over GATE: INVARIANT: ADR-2037 via ADR-2086 - a production/release<br/>image must never carry the dev-auth cargo feature (it compiles in<br/>the Bearer dev-session-token bypass and stubs<br/>enforce_release_env_hygiene to a no-op).<br/>Hermetic: no cargo, no docker, no network - project/.github/workflows/ci.yml:143-150
    Note over CPU: DIVERGENCE: visionclaw-gpu and root server crate link<br/>CUDA at runtime, removed from hosted CI 2026-07-24 (project/.github/workflows/ci.yml:241-243)
    Note over CPU: GPU crates validated only on the developer CUDA<br/>host via scripts/launch.sh, not by any GitHub runner
```

## ES-09.12 VisionClaw docs-ci.yml — ADR ledger gate and documentation quality score
```mermaid
sequenceDiagram
    autonumber
    participant GH as push/PR touching docs/**<br/>docs-ci.yml:4-20
    participant ADR as validate-adr-ledger job<br/>docs-ci.yml:23
    participant DOC as validate-documentation job<br/>docs-ci.yml:34

    GH->>ADR: checkout fetch-depth 0 :30 (staleness diffs old commits)
    ADR->>ADR: node scripts/adr-index-gen.js docs/adr --check :32
    Note over ADR: checks frontmatter, supersession reciprocity,<br/>verified_commit staleness

    GH->>DOC: checkout
    DOC->>DOC: validate internal links :42-107
    DOC->>DOC: validate mermaid diagrams :109-162
    DOC->>DOC: check stale references :164-210
    Note over DOC: stale refs are warnings only, never a score penalty :207-210
    DOC->>DOC: validate directory structure :212-241
    Note over DOC: required Diataxis dirs: tutorials, how-to,<br/>explanation, reference, plus docs/README.md :221-231
    DOC->>DOC: score = (links_rate*60 + mermaid_rate*40)/100<br/>minus 10 per structure error, clamp 0-100 :254-260
    alt score below 50
        DOC->>GH: exit 1, quality threshold failed :326-328
    else score at or above 50
        DOC->>GH: pass :330
    end
```

## ES-09.13 VisionClaw ontology-publish.yml — vault pipeline to pod, pull model
```mermaid
sequenceDiagram
    autonumber
    participant GH as push/PR/dispatch<br/>ontology-publish.yml:3-27
    participant VAL as validate-source job<br/>ontology-publish.yml:44
    participant BLD as build-ontology job<br/>ontology-publish.yml:135
    participant REL as publish-release job<br/>ontology-publish.yml:220
    participant SRV as visionclaw-server boot<br/>src/services/ontology_pull.rs
    participant DEP as deploy-jss job (push path)<br/>ontology-publish.yml:290
    participant WS as notify-websocket job<br/>ontology-publish.yml:400
    participant PRP as pr-preview job<br/>ontology-publish.yml:479
    participant MIS as deploy-target-missing job<br/>ontology-publish.yml:533

    GH->>VAL: preflight gh repo view on the private ontology source :65-73<br/>ONTOLOGY_SOURCE_TOKEN, falling back to GITHUB_TOKEN :67-68
    alt source unreadable
        VAL--xGH: ::error + OPS ACTION naming the fine-grained PAT, exit 1 :74-87
    end
    GH->>VAL: checkout ontology source with the same token :89-95
    VAL->>VAL: detect changed markdown files :96-117
    VAL->>BLD: has_changes true, needs validate-source :138-139
    BLD->>BLD: checkout jjohare/visionGraph again with the token :150-155
    BLD->>BLD: python -m pipeline.build knowledge/pages<br/>— the vault's own converter :168-171
    BLD->>BLD: pack-pod-resources.py — public-only TTL, JSON-LD<br/>compacted against the vault context, LDP index manifest,<br/>substance floor 4000 classes / 100k triples :176-181
    BLD->>BLD: re-parse packed resources :183-199, upload<br/>ontology-ttl :201-206 and ontology-jsonld :208-213
    BLD->>REL: main only, needs validate-source + build-ontology :223-224
    REL->>REL: assemble five assets + SHA256SUMS :239-252,<br/>create or move release tag ontology-latest :253-272
    REL->>REL: upload --clobber :273-278, verify the public<br/>download path diffs SHA256SUMS :279-289
    Note over SRV: ADR-2106 pull model — at boot and every<br/>ONTOLOGY_PULL_INTERVAL_SECS: GET index.jsonld, compare<br/>visionflow:buildSha with the pod's, and if moved GET<br/>SHA256SUMS plus the files, verify every digest, then write<br/>via Storage in order: containers, .acl only if absent,<br/>content, manifest last. Fail-open. see ES-08.11
    SRV->>REL: GET releases/download/ontology-latest/{index.jsonld, SHA256SUMS, ...}
    alt vars.SOLID_POD_URL set (a pod a runner can reach)
        BLD->>DEP: needs build-ontology, ref main :293-294
        DEP->>DEP: backup current pod index for rollback :309-321
        DEP->>DEP: PUT visionflow.ttl, context/ontology/index jsonld :324-356, verify :359-385
        alt deployment fails
            DEP->>DEP: rollback to backed-up index :386-399
        end
        DEP->>WS: deployment_status success :403-404
        WS->>WS: POST to SOLID_POD_URL/.notifications :443-447, PATCH index.jsonld :448-453
    else unset (default: the in-process pod is not reachable from a hosted runner)
        REL->>MIS: ::notice: push deploy skipped, pull model in effect :536-540
    end
    GH->>PRP: pull_request only: comment with stats and SHAs :482-483
    Note over VAL,WS: RESOLVED ADR-2098 (2026-09-05): SOLID_POD_URL now<br/>defaults to the loopback /solid scope :31-38 — what the embedded<br/>solid-pod-rs serves in-process (ADR-032 M3). The POST to<br/>/.notifications is annotated a best-effort no-op there:<br/>that path is a GET WebSocket upgrade
    Note over BLD: RESOLVED 2026-09-06: the inline md_to_ttl.py + pyld<br/>converters read Logseq key:: lines, never the json-ld fence, and<br/>produced 0 owl:Class from 380 pages on run 34045488066 —<br/>replaced by the vault pipeline, 8,434 classes / 265k triples
    Note over GH,BLD: CORRECTED — ownership splits ACROSS the two repos, not along one.<br/>EXTERNAL: the vault jjohare/visionGraph owns pipeline.build, run under<br/>working-directory logseq-source (ontology-publish.yml:169-171).<br/>scripts/ontology/pack-pod-resources.py is THIS repo's own script, run<br/>from the repo root with no working-directory (ontology-publish.yml:181).<br/>An earlier revision of this note credited the vault with both — VG-04.1<br/>had it right. see VG-04.1, VG-04.2 and KG-05 for the publisher side
    Note over SRV,MIS: RESOLVED ADR-2106 (2026-09-06): deploy-jss had never<br/>run and could not from a hosted runner — delivery inverted to a<br/>boot pull from the ontology-latest release. see ES-08.11
```

## ES-09.14 VisionClaw xr-godot-ci.yml — gdext + GUT headless, Quest 3 advisory
```mermaid
sequenceDiagram
    autonumber
    participant GH as push/PR xr-client/**<br/>xr-godot-ci.yml:24-36
    participant RT as xr-rust-tests job<br/>xr-godot-ci.yml:54 blocking
    participant GUT as gut-headless job<br/>xr-godot-ci.yml:77 blocking
    participant Q3 as quest3-android job<br/>xr-godot-ci.yml:117 advisory

    GH->>RT: cargo test -p visionclaw-xr-gdext --all-features :73
    RT->>RT: cargo test -p visionclaw-xr-presence :75
    GH->>GUT: cargo build -p visionclaw-xr-gdext (debug cdylib) :89
    GUT->>GUT: install Godot 4.3-stable headless :90-96
    GUT->>GUT: vendor GUT 9.3.1 pinned tag :97-103
    GUT->>GUT: godot --headless -s gut_cmdln.gd :107
    GH->>Q3: continue-on-error true :120
    Q3->>Q3: cargo ndk build aarch64-linux-android :144
    Q3->>Q3: export Quest 3 arm64 APK :162-163
    Q3->>Q3: APK size gate, fail if greater than 80MB :164-169
    Note over Q3: advisory until first green hosted run,<br/>then promote to blocking (per file header comment)
```

## ES-09.15 agentbox invariants.yml — security invariant gates including loopback-publish
```mermaid
sequenceDiagram
    autonumber
    participant GH as push/PR touching compose,ADRs,scripts<br/>invariants.yml:6-24
    participant J as invariants job<br/>invariants.yml:30

    GH->>J: checkout fetch-depth 0 :36
    J->>J: check-seccomp.sh :47
    J->>J: check-nnp.sh (no-new-privileges) :50
    J->>J: check-ports-loopback.sh :53
    Note over J: ADR-2013: sweeps EVERY docker-compose*.yml via a real<br/>YAML parser (check-ports-loopback.mjs), replacing an<br/>awk line-walker that missed nested-mapping publishes
    Note over J: SANCTIONED allowlist: 9096 sovereign ingress,<br/>voice 8443/8444, browsercontainer 5903/8931/9222,<br/>gui-tools 5905/9876/9877, xr-runtime 5904
    Note over J: DIVERGENCE: implementation_status partial — the<br/>dated closeout says the scanner does not yet cover<br/>every equivalent publish syntax form
    J->>J: check-db-password.sh :56
    J->>J: check-secret-not-in-env.sh :59
    J->>J: check-single-metrics.js :62
    J->>J: check-no-npx-latest.sh (ratchet) :65
    J->>J: lint-skills.sh :68
    J->>J: deepsec-gate.test.mjs :71
    J->>J: check-manifest-catalogue.js (ADR-039 gate-path parity) :74
    J->>J: check-no-logseq-paths.sh :77
    Note over J: ADR-2028: vault.root is the single corpus path<br/>authority, greps for hard-coded workspace/logseq<br/>outside docs/archive and docs/adr exemptions
    J->>J: vault-frontmatter tests :80
    J->>J: adr-index-gen.js docs/adr --check :83
    J->>J: check-crate-licensing.sh :86
```

## ES-09.16 agentbox contract-tests.yml — adapter contract suites
```mermaid
sequenceDiagram
    autonumber
    participant GH as PR touching adapters/**<br/>contract-tests.yml:5-22
    participant C as contract job<br/>contract-tests.yml:34

    GH->>C: setup Node 22 (matches runtime image) :41-46
    C->>C: npm ci in management-api/ contract-tests.yml:50<br/>and repo-root npm ci --ignore-scripts contract-tests.yml:57
    C->>C: npx jest ../tests/contract/ with contract filename filter<br/>contract-tests.yml:61
    C->>C: upload contract-test-results artifact contract-tests.yml:65-70
    Note over C: every durable-state integration rides one of five<br/>adapter slots (beads, pods, memory, events, orchestrator)<br/>and must pass tests/contract/ for all implementation classes
```

## ES-09.17 agentbox manifest-validate.yml — config validator and TUI round-trip
```mermaid
sequenceDiagram
    autonumber
    participant GH as PR touching agentbox.toml,schema/**<br/>manifest-validate.yml:17-33
    participant V as validate job<br/>manifest-validate.yml:46

    GH->>V: setup Node 20, Python 3.11, Rust stable :51-65
    V->>V: cargo build --release services/agentbox-manifest :71
    V->>V: node agentbox-config-validate.js agentbox.toml :77
    loop each tests/tui/fixtures/valid-*.toml
        V->>V: agentbox-manifest tui-read fixture -> state.json :84
        V->>V: agentbox-manifest tui-write state.json -> out.toml :85
        V->>V: agentbox-config-validate.js out.toml :86
    end
    loop each tests/tui/fixtures/invalid-*.toml
        V->>V: assert failure with the expected E-code :95-112
    end
    V->>V: assert JSON Schema well-formed :112-114
    V->>V: assert all W-codes route to warnings :115-128
    V->>V: assert E021 fires when exception block missing :129
    Note over V: same validator the TUI runs on every section<br/>transition and the flake evaluator runs at build time
```

## ES-09.18 agentbox ci.yml aggregate gate, and the structurally-identical workflow family
```mermaid
sequenceDiagram
    autonumber
    participant GH as PR / push main<br/>agentbox/.github/workflows/ci.yml:15-19
    participant AGG as ci-passed job<br/>agentbox/.github/workflows/ci.yml:30

    GH->>AGG: wait-on-check-action, poll every 20s :38-42
    AGG->>AGG: require ShellCheck error, gitleaks,<br/>agentbox config validate + TUI round-trip :48
    alt all required checks succeed or skipped
        AGG->>GH: CI passed :51-54
    else any required check fails
        AGG->>GH: aggregate gate fails, branch protection blocks merge
    end
    Note over AGG: structurally-identical family (each is its own workflow<br/>file, one job, checkout+setup+run+upload pattern) —<br/>flake-check.yml (Nix eval x86_64/aarch64, statix lint)<br/>secret-scan.yml (gitleaks), shellcheck.yml (severity matrix)<br/>image-scan.yml (Trivy HIGH/CRITICAL gate + SBOM CycloneDX/SPDX)<br/>deepsec.yml (deepsec-gate against PR diff, Anthropic route)<br/>tui-tests.yml (cargo clippy+test services/agentbox-manifest)<br/>build-multi-arch.yml (Nix image build, GHCR push, manifest list)<br/>nix-flake-update.yml (scheduled flake update + PR)<br/>release.yml (CHANGELOG-derived GitHub Release body)
    Note over AGG: agentbox/.github/workflows/ontology-publish.yml is a stale copy of the<br/>pre-2026-09-06 VisionClaw ontology-publish.yml (inline md_to_ttl.py converter, jjohare/logseq<br/>default, no token preflight) — fails on every push, no agentbox consumer — not re-diagrammed
```

## ES-09.19 End-to-end artefact flow — source to running container
```mermaid
flowchart LR
    SRC["Source on host bind<br/>/mnt/mldata/githubs/AR-AI-Knowledge-Graph"]
    CIGATE["CI gates (ES-09.11 to ES-09.18)<br/>rust-fmt, rust-cpu, client, docs-ci,<br/>invariants, contract-tests, manifest-validate"]
    DOCKERBUILD["docker compose build<br/>Dockerfile.unified or Dockerfile.production<br/>host tmux tab 6 only (ES-09.1)"]
    IMAGE["Built image<br/>cachyos-v3 base + compiled binary or dev toolchain"]
    COMPOSEUP["docker compose --profile dev|production up<br/>docker-compose.unified.yml"]
    CONTAINER["visionclaw_container or visionclaw_prod_container<br/>supervisord manages nginx + rust-backend (+vite-dev in dev)"]
    NGINXROUTE["nginx.dev.conf or nginx.production.conf<br/>route table (ES-09.9)"]
    HEALTH["/api/health, /readyz<br/>docker-compose.unified.yml:168,232"]

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
    participant CB as "cargo build (dev container)"<br/>rust-backend-wrapper.sh:57
    participant URI as visionclaw-server src/uri<br/>src/uri/mod.rs:662
    participant FK as federation-kinds artefact<br/>agentbox/schema/federation-kinds.json:2-4
    participant CMP as dev compose mounts<br/>docker-compose.unified.yml:136
    participant BR as "BC20 bridge (agentbox, JS)"

    Note over URI,FK: ADR-2061 — federation-kinds.json is the SINGLE versioned<br/>authority for which urn:agentbox kinds cross the boundary.<br/>cross_from_agentbox derives its closed map from it — the JS bridge<br/>reads the same bytes at load. Neither side transcribes the list.
    URI->>FK: "const FEDERATION_KINDS_JSON = include_str!(\"../../agentbox/schema/federation-kinds.json\")"<br/>src/uri/mod.rs:662
    Note over URI: include_str! is COMPILE time, so the file must exist under /app<br/>or the lib cannot build at all — not a runtime read.
    CMP->>CB: "bind ${HOST_PROJECT_ROOT}/agentbox/schema -> /app/agentbox/schema:ro"<br/>docker-compose.unified.yml:136
    CB->>URI: compiles, artefact bytes baked into the binary
    FK-->>BR: EXTERNAL — the same file is read at load by the agentbox<br/>management-api bridge. see AB-17 and ES-03.1
    Note over CMP: INVARIANT — mount schema/ ONLY, never the whole submodule.<br/>agentbox carries 262 files ADR-2008 counts as build inputs, so a<br/>full mount would turn every agentbox bump into a spurious ~12 min<br/>visionclaw rebuild — schema/ is JSON alone and matches no input glob<br/>(docker-compose.unified.yml:127-135)
    Note over CB: RESOLVED 2026-09 — before this mount the dev image (which carries<br/>no source of its own) failed every build with<br/>"couldn't read src/uri/../../agentbox/schema/federation-kinds.json"<br/>docker-compose.unified.yml:131
```

## ES-09.21 scripts/launch.sh — env-file resolution and the two divergent code paths
```mermaid
flowchart TB
    START["launch.sh main()<br/>scripts/launch.sh:1170<br/>PROJECT_ROOT = dirname SCRIPT_DIR :15"]

    subgraph SANCTIONED["Sanctioned path — up / rebuild dev|prod"]
        LOADENV["load_environment<br/>looks for .env.$ENVIRONMENT :158"]
        PRODREQ["production: .env.prod REQUIRED<br/>refuses forbidden dev settings :167<br/>demands concrete values :176<br/>hard error if absent :189"]
        DEVFALL["dev: falls back to plain .env :193-198<br/>hard error if neither exists :200"]
        HPR["HOST_PROJECT_ROOT export :279,:287,:295<br/>— the DinD path translation the schema<br/>mount in ES-09.20 depends on"]
    end

    subgraph LEGACY["rebuild_agent_container — legacy MAD path"]
        CD["cd $PROJECT_ROOT/multi-agent-docker :812"]
        MK[".env absent -> cp env.example .env :817-819<br/>then blocks on an interactive read :822"]
    end

    START --> LOADENV
    LOADENV --> PRODREQ
    LOADENV --> DEVFALL
    LOADENV --> HPR
    START -->|"launch.sh rebuild-agent :1236"| CD --> MK

    FIX["RESOLVED 2026-09-06 — the auto-create branch tested for a dotted<br/>.env.example that has never existed in the tree, so it always fell<br/>through to the error exit. It now copies env.example, the template<br/>that actually ships (scripts/launch.sh:817-819, env.example)."]
    DIV["DIVERGENCE — multi-agent-docker/ is not present in this checkout,<br/>so rebuild_agent_container cds into a missing directory before it<br/>reaches the fixed branch. The repaired .env creation is correct but<br/>currently unreachable; the MAD stack is the deprecated predecessor<br/>whose only surviving trace is the mad-workspace volume. see ES-01.4"]
    INV["INVARIANT ADR-2027 — profile selection is env-file driven, and the<br/>production branch fails closed: a missing or dev-contaminated<br/>.env.prod aborts the launch rather than defaulting. see ES-10"]

    MK --> FIX
    CD --> DIV
    PRODREQ --> INV
```
