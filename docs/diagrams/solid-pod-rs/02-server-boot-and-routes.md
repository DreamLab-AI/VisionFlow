---
id: SP-02
title: Server boot, configuration layering and the full route table
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2004, ADR-2007]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mempool.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/pay.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/prov.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/cli/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/cli/install.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/config/schema.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/config/loader.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/config/sources.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/fs.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/memory.rs
verified_commit: 1d9da5270
---

## SP-02.1 Process entry point — CLI to listening socket

```mermaid
sequenceDiagram
    autonumber
    participant OS as Process start
    participant M as main<br/>solid-pod-rs-server/src/main.rs:219
    participant CLI as Cli::parse<br/>solid-pod-rs-server/src/main.rs:220
    participant OP as dispatch operator cmd<br/>solid-pod-rs-server/src/main.rs:235
    participant LD as ConfigLoader<br/>solid-pod-rs-server/src/main.rs:238
    participant BIND as bind_available<br/>solid-pod-rs-server/src/main.rs:154
    participant ST as build_storage<br/>solid-pod-rs-server/src/main.rs:138
    participant APP as build_app<br/>solid-pod-rs-server/src/lib.rs:4553

    OS->>M: exec
    M->>CLI: parse argv + env
    M->>M: tracing_subscriber::fmt with RUST_LOG filter<br/>solid-pod-rs-server/src/main.rs:227
    alt an operator subcommand was given
        M->>OP: dispatch and return — no HTTP lifecycle
        OP-->>OS: exit
    end
    M->>LD: with_defaults, optional with_file, with_env<br/>solid-pod-rs-server/src/main.rs:243
    LD-->>M: ServerConfig
    M->>M: host/port CLI overrides then cfg.validate<br/>solid-pod-rs-server/src/main.rs:254
    M->>BIND: bind host:port<br/>solid-pod-rs-server/src/main.rs:258
    BIND-->>M: TcpListener (possibly a shifted port)
    M->>ST: construct Arc dyn Storage<br/>solid-pod-rs-server/src/main.rs:269
    ST-->>M: FsBackend or MemoryBackend
    M->>APP: HttpServer::new closure per worker<br/>solid-pod-rs-server/src/main.rs:345
    Note over M: INVARIANT: the operator subcommand short-circuits BEFORE any config,<br/>storage or socket work — a one-shot admin command never binds a port.
```

## SP-02.2 AppState assembly order

```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>solid-pod-rs-server/src/main.rs:219
    participant S as AppState::new<br/>solid-pod-rs-server/src/lib.rs:371
    participant MP as log_mempool_selection_once<br/>solid-pod-rs-server/src/main.rs:284
    participant Q as FsQuotaStore::new<br/>solid-pod-rs-server/src/main.rs:290

    M->>S: AppState::new(storage)<br/>solid-pod-rs-server/src/main.rs:276
    S-->>M: defaults — mcp off, quota None, admin_key None<br/>solid-pod-rs-server/src/lib.rs:386
    M->>M: state.data_root from StorageBackendConfig::Fs root
    M->>MP: record endpoint + inferred network once (ADR-2007)
    opt feature quota AND default_quota_bytes > 0
        M->>Q: FsQuotaStore over data_root
    end
    M->>M: allowed_origins<br/>solid-pod-rs-server/src/main.rs:297
    M->>M: admin_key<br/>solid-pod-rs-server/src/main.rs:298
    M->>M: mcp_enabled = cli.mcp AND NOT cli.no_mcp<br/>solid-pod-rs-server/src/main.rs:301
    M->>M: deposit_txo_standin_enabled<br/>solid-pod-rs-server/src/main.rs:306
    M->>M: nodeinfo meta incl. open_registrations<br/>solid-pod-rs-server/src/main.rs:314
    M->>M: mashlib mode — module URL beats CDN version<br/>solid-pod-rs-server/src/main.rs:322
    Note over M: --no-mcp always wins over a baked-in JSS_MCP env value.
```

## SP-02.3 The security-relevant CLI/env register

```mermaid
classDiagram
    class Cli {
        +config  JSS_CONFIG  solid-pod-rs-server/src/main.rs:39
        +mashlib  JSS_MASHLIB  solid-pod-rs-server/src/main.rs:57
        +live_reload  JSS_LIVE_RELOAD  solid-pod-rs-server/src/main.rs:71
        +ssl_key  JSS_SSL_KEY  solid-pod-rs-server/src/main.rs:77
        +allowed_origins  SOLID_ALLOWED_ORIGINS  solid-pod-rs-server/src/main.rs:90
        +admin_key  SOLID_ADMIN_KEY  solid-pod-rs-server/src/main.rs:96
        +open_registration  JSS_OPEN_REGISTRATION  solid-pod-rs-server/src/main.rs:102
        +mcp  JSS_MCP  solid-pod-rs-server/src/main.rs:110
        +no_mcp (overrides mcp)  solid-pod-rs-server/src/main.rs:116
        +deposit_txo_standin  DEPOSIT_TXO_STANDIN_ENABLED  solid-pod-rs-server/src/main.rs:124
        +op  operator subcommand  solid-pod-rs-server/src/main.rs:131
    }
    note for Cli "INVARIANT: every one of these defaults to OFF/None. Registration is closed,\nMCP is off, the admin endpoint 403s with no key, and the unverified TXO\ndeposit stand-in is off — a bare `solid-pod-rs-server` opens no back door."
```

## SP-02.4 Layered configuration resolution

```mermaid
flowchart LR
    D["with_defaults<br/>solid-pod-rs/src/config/loader.rs:79"]
    F["with_file (JSON/YAML/TOML auto-detect)<br/>solid-pod-rs/src/config/loader.rs:94"]
    E["with_env — JSS_* variables<br/>solid-pod-rs/src/config/loader.rs:101"]
    C["with_cli_overlay<br/>solid-pod-rs/src/config/loader.rs:127"]
    L["load -> ServerConfig<br/>solid-pod-rs/src/config/loader.rs:156"]
    W["warnings collected, not fatal<br/>solid-pod-rs/src/config/loader.rs:208"]

    D --> F --> E --> C --> L --> W

    SRC["ConfigSource enum<br/>solid-pod-rs/src/config/sources.rs:63"]
    LF["load_file<br/>solid-pod-rs/src/config/sources.rs:107"]
    NS["normalise_file_shape<br/>solid-pod-rs/src/config/sources.rs:164"]
    LE["load_env<br/>solid-pod-rs/src/config/sources.rs:215"]
    PS["parse_size — 50MB / 1.5GB / bare bytes<br/>solid-pod-rs/src/config/sources.rs:467"]

    SRC --> LF --> NS
    SRC --> LE
    LE --> PS

    N["Later layers win: defaults, then file, then env, then CLI."]
    C -.-> N
```

## SP-02.5 ServerConfig shape and the one hard validation

```mermaid
classDiagram
    class ServerConfig {
        +server ServerSection  solid-pod-rs/src/config/schema.rs:27
        +storage StorageBackendConfig  solid-pod-rs/src/config/schema.rs:31
        +auth AuthConfig  solid-pod-rs/src/config/schema.rs:35
        +notifications NotificationsConfig  solid-pod-rs/src/config/schema.rs:39
        +security SecurityConfig  solid-pod-rs/src/config/schema.rs:43
        +extras ExtrasConfig  solid-pod-rs/src/config/schema.rs:51
        +validate()  solid-pod-rs/src/config/schema.rs:323
    }
    class ServerSection {
        +host  solid-pod-rs/src/config/schema.rs:113
        +port  solid-pod-rs/src/config/schema.rs:117
        +base_url Option  solid-pod-rs/src/config/schema.rs:121
    }
    class AuthConfig {
        +oidc_enabled (default false)  solid-pod-rs/src/config/schema.rs:190
    }
    class SecurityConfig {
        +dotfile_allowlist  solid-pod-rs/src/config/schema.rs:281
        +default_quota_bytes (default 0 = off)  solid-pod-rs/src/config/schema.rs:289
    }
    ServerConfig *-- ServerSection
    ServerConfig *-- AuthConfig
    ServerConfig *-- SecurityConfig
    note for ServerConfig "validate() rejects oidc_enabled = true with no oidc_issuer\n  solid-pod-rs/src/config/schema.rs:327 — a pod cannot claim OIDC with no issuer."
```

## SP-02.6 Storage backend selection — only two survive

```mermaid
stateDiagram-v2
    [*] --> Parse
    Parse --> Fs: StorageBackendConfig.Fs<br/>solid-pod-rs-server/src/main.rs:140
    Parse --> Memory: StorageBackendConfig.Memory<br/>solid-pod-rs-server/src/main.rs:147
    Parse --> Rejected: any other backend name

    Fs --> Ready: FsBackend.new(root)<br/>solid-pod-rs/src/storage/fs.rs:47
    Memory --> Ready: MemoryBackend.new()<br/>solid-pod-rs/src/storage/memory.rs:45
    Rejected --> [*]: configuration validation fails

    Ready --> [*]

    note right of Rejected
      DOC-DRIFT closed: the S3 backend was REMOVED in 0.5.0-alpha.8.
      Only Fs and Memory exist in the enum today, so an `s3` config
      value never reaches build_storage.
    end note
    note right of Fs
      data_root is captured only for the Fs arm
      (solid-pod-rs-server/src/main.rs:265), so the git and quota
      features are inert on a memory pod. See SP-06.
    end note
```

## SP-02.7 Port binding — busy-port shift

```mermaid
stateDiagram-v2
    [*] --> Requested
    Requested --> Ephemeral: port == 0 (1 attempt)<br/>solid-pod-rs-server/src/main.rs:155
    Requested --> Try0: port != 0 (11 attempts)
    Try0 --> Bound: bind succeeds
    Try0 --> TryN: AddrInUse and attempts remain<br/>solid-pod-rs-server/src/main.rs:171
    TryN --> Bound: warn port busy — shifted listener<br/>solid-pod-rs-server/src/main.rs:164
    TryN --> Failed: attempts exhausted or other io error
    Ephemeral --> Bound
    Bound --> [*]
    Failed --> [*]
    note right of Bound
      The listener is set non-blocking before hand-off to actix
      (solid-pod-rs-server/src/main.rs:160) — the actual port is
      re-read from local_addr, so base_url reflects the shift.
    end note
```

## SP-02.8 Middleware stack — registration order versus execution order

```mermaid
flowchart TD
    REQ["inbound request"]
    ELM["ErrorLoggingMiddleware (registered first, runs LAST)<br/>solid-pod-rs-server/src/lib.rs:4575"]
    CORS["CorsHeaders<br/>solid-pod-rs-server/src/lib.rs:4576"]
    NORM["NormalizePath TrailingSlash::MergeOnly<br/>solid-pod-rs-server/src/lib.rs:4580"]
    PTG["PathTraversalGuard<br/>solid-pod-rs-server/src/lib.rs:4581"]
    DFG["DotfileGuard<br/>solid-pod-rs-server/src/lib.rs:4582"]
    H["route handler"]

    REQ --> DFG --> PTG --> NORM --> CORS --> ELM --> H

    N1["INVARIANT: MergeOnly collapses // to / but never strips the trailing slash —<br/>the trailing slash is the LDP container/resource discriminator.<br/>solid-pod-rs-server/src/lib.rs:4578"]
    NORM -.-> N1
    N2["ErrorLoggingMiddleware is wrapped first so it observes every response,<br/>including ones that short-circuited inside an inner guard.<br/>solid-pod-rs-server/src/lib.rs:4572"]
    ELM -.-> N2
    N3["PathTraversalGuard rejects on path_is_traversal<br/>solid-pod-rs-server/src/lib.rs:3078"]
    PTG -.-> N3
```

## SP-02.9 Route table — discovery, payments, admin and account routes

```mermaid
flowchart LR
    subgraph WK["Well-known (always on)"]
        A1["GET /.well-known/solid<br/>solid-pod-rs-server/src/lib.rs:4590"]
        A2["GET /.well-known/webfinger<br/>solid-pod-rs-server/src/lib.rs:4593"]
        A3["GET /.well-known/nodeinfo<br/>solid-pod-rs-server/src/lib.rs:4596"]
        A4["GET /.well-known/nodeinfo/2.1<br/>solid-pod-rs-server/src/lib.rs:4600"]
        A5["GET /.well-known/apps<br/>solid-pod-rs-server/src/lib.rs:4634"]
    end
    subgraph GATED["Feature-gated discovery"]
        B1["GET /.well-known/did/nostr/{pubkey}.json — did-nostr<br/>solid-pod-rs-server/src/lib.rs:4608"]
        B2["GET /.well-known/nostr.json — nip05-endpoint<br/>solid-pod-rs-server/src/lib.rs:4620"]
        B3["GET /api/exports/all — export-jsonld<br/>solid-pod-rs-server/src/lib.rs:4630"]
    end
    subgraph PAYADMIN["Payments, proxy, admin"]
        C1["GET /pay/.info<br/>solid-pod-rs-server/src/lib.rs:4637"]
        C2["handlers::pay::register — the whole /pay/* surface<br/>solid-pod-rs-server/src/lib.rs:4643"]
        C3["GET /proxy — WAC-gated CORS proxy<br/>solid-pod-rs-server/src/lib.rs:4646"]
        C4["POST /_admin/provision/{pubkey}<br/>solid-pod-rs-server/src/lib.rs:4662"]
    end
    subgraph ACCT["Account and pod management"]
        D1["POST /.pods<br/>solid-pod-rs-server/src/lib.rs:4667"]
        D2["POST /api/accounts/new<br/>solid-pod-rs-server/src/lib.rs:4668"]
        D3["GET /pods/check/{name}<br/>solid-pod-rs-server/src/lib.rs:4669"]
        D4["POST /login/password<br/>solid-pod-rs-server/src/lib.rs:4670"]
        D5["POST /account/password/reset<br/>solid-pod-rs-server/src/lib.rs:4672"]
        D6["POST /account/password/change<br/>solid-pod-rs-server/src/lib.rs:4676"]
    end

    N["INVARIANT: every one of these registers BEFORE the LDP catch-all so a\nreserved prefix is never treated as a pod resource."]
    ACCT -.-> N
```

## SP-02.10 Route table — git, forge and MCP surfaces

```mermaid
flowchart TD
    MCP["POST /mcp + OPTIONS — only when state.mcp_enabled<br/>solid-pod-rs-server/src/lib.rs:4652"]
    FORGE["/forge and /forge/{tail} — cfg(feature forge)<br/>solid-pod-rs-server/src/lib.rs:4687"]
    BLOCK1["ANY /{tail}/.git -> 403<br/>solid-pod-rs-server/src/lib.rs:4698"]
    BLOCK2["ANY /{tail}/.git/{rest} -> 403<br/>solid-pod-rs-server/src/lib.rs:4705"]
    PANEL_OPT["OPTIONS /pods/{pk}/_git/{tail} — registered unconditionally<br/>solid-pod-rs-server/src/lib.rs:4715"]

    subgraph GITON["cfg(feature git)"]
        SMART["GET info/refs, POST git-upload-pack, POST git-receive-pack<br/>solid-pod-rs-server/src/lib.rs:4724"]
        PANEL["/pods/{pubkey}/_git/{status,log,diff,stage,unstage,commit,branches,branch,discard}<br/>solid-pod-rs-server/src/lib.rs:4731"]
        PROVR["handlers::prov::register — _prov resolve + anchor<br/>solid-pod-rs-server/src/lib.rs:4768"]
    end
    subgraph GITOFF["cfg(not(feature git))"]
        R501["the three smart-HTTP paths return 501<br/>solid-pod-rs-server/src/lib.rs:4780"]
    end

    MCP --> FORGE --> BLOCK1 --> BLOCK2 --> PANEL_OPT --> GITON
    PANEL_OPT --> GITOFF

    N1["INVARIANT: direct .git/ access is blocked unconditionally — the block is\nregistered outside every feature gate, so it survives a git-less build."]
    BLOCK1 -.-> N1
    N2["The forge registers BEFORE the pod-git catch-all so /forge/<o>/<n>.git/info/refs\nreaches the forge's own CGI, not the pod-git handler.<br/>solid-pod-rs-server/src/lib.rs:4681"]
    FORGE -.-> N2
    N3["OPTIONS preflight for the _git panel is registered even without the feature\nso a browser gets a valid CORS answer either way."]
    PANEL_OPT -.-> N3
```

## SP-02.11 Route table — the LDP catch-all, registered last

```mermaid
flowchart LR
    P1["POST /{tail}/ -> handle_post<br/>solid-pod-rs-server/src/lib.rs:4787"]
    P2["PUT /{tail}/ -> handle_put<br/>solid-pod-rs-server/src/lib.rs:4788"]
    G["GET /{tail} -> handle_get<br/>solid-pod-rs-server/src/lib.rs:4789"]
    H["HEAD /{tail} -> handle_get<br/>solid-pod-rs-server/src/lib.rs:4790"]
    PU["PUT /{tail} -> handle_put<br/>solid-pod-rs-server/src/lib.rs:4791"]
    PA["PATCH /{tail} -> handle_patch<br/>solid-pod-rs-server/src/lib.rs:4792"]
    DE["DELETE /{tail} -> handle_delete<br/>solid-pod-rs-server/src/lib.rs:4793"]
    CO["COPY /{tail} -> handle_copy<br/>solid-pod-rs-server/src/lib.rs:4796"]
    OP["OPTIONS /{tail} -> handle_options<br/>solid-pod-rs-server/src/lib.rs:4800"]

    P1 --> P2 --> G --> H --> PU --> PA --> DE --> CO --> OP

    N1["The trailing-slash POST/PUT variants register FIRST so a container write wins\nover the resource catch-all.<br/>solid-pod-rs-server/src/lib.rs:4785"]
    P1 -.-> N1
    N2["INVARIANT: HEAD is routed to handle_get, so HEAD inherits the same WAC read\ngate as GET — a private resource cannot be probed by HEAD. See SP-04.2."]
    H -.-> N2
    N3["COPY is registered by raw method bytes — it is not a standard actix verb.<br/>solid-pod-rs-server/src/lib.rs:4796"]
    CO -.-> N3
```

## SP-02.12 Graceful shutdown

```mermaid
sequenceDiagram
    autonumber
    participant SIG as SIGINT / SIGTERM
    participant SH as shutdown task<br/>solid-pod-rs-server/src/main.rs:370
    participant SRV as actix HttpServer<br/>solid-pod-rs-server/src/main.rs:367
    participant M as main

    Note over SRV: shutdown_timeout(30) then run()
    SIG->>SH: ctrl_c<br/>solid-pod-rs-server/src/main.rs:372
    SIG->>SH: terminate_signal (unix SIGTERM)<br/>solid-pod-rs-server/src/main.rs:390
    SH->>SRV: handle.stop(graceful = true)<br/>solid-pod-rs-server/src/main.rs:379
    SRV-->>M: server future resolves
    M->>M: await shutdown task, log "stopped cleanly"
    Note over SH: On non-unix the terminate branch is std::future::pending —<br/>only ctrl_c can trigger shutdown.<br/>solid-pod-rs-server/src/main.rs:400
```

## SP-02.13 Operator subcommands — the no-HTTP path

```mermaid
flowchart TD
    OP["OperatorCommand<br/>solid-pod-rs-server/src/cli/mod.rs:28"]
    Q["quota reconcile<br/>solid-pod-rs-server/src/cli/mod.rs:51"]
    A["account delete<br/>solid-pod-rs-server/src/cli/mod.rs:87"]
    I["invite create<br/>solid-pod-rs-server/src/cli/mod.rs:109"]
    D["dispatch<br/>solid-pod-rs-server/src/cli/mod.rs:327"]

    OP --> Q
    OP --> A
    OP --> I
    Q --> RQ["run_quota_reconcile<br/>solid-pod-rs-server/src/cli/mod.rs:158"]
    A --> RA["run_account_delete (Prompt-gated)<br/>solid-pod-rs-server/src/cli/mod.rs:253"]
    I --> RI["run_invite_create<br/>solid-pod-rs-server/src/cli/mod.rs:288"]
    RQ --> D
    RA --> D
    RI --> D

    INS["install subcommand — app fetch over git<br/>solid-pod-rs-server/src/cli/install.rs:231"]
    SPEC["parse_app_spec — bare name, org/repo, or full URL<br/>solid-pod-rs-server/src/cli/install.rs:81"]
    AUTH["build_auth_header — NIP-98 mint or bearer<br/>solid-pod-rs-server/src/cli/install.rs:131"]
    INS --> SPEC
    INS --> AUTH

    N["run_account_delete goes through a Prompt trait\n(solid-pod-rs-server/src/cli/mod.rs:224) so the destructive path is\nconfirmable and testable without stdin."]
    RA -.-> N
```
