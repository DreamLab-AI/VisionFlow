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
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/scheme.rs
verified_commit: 27e18995fc20790fa956acb41f7d8420544a13ec
---

## SP-02.1 Process entry point — CLI to listening socket

```mermaid
sequenceDiagram
    autonumber
    participant OS as Process start
    participant M as main<br/>solid-pod-rs-server/src/main.rs:261
    participant CLI as Cli::parse<br/>solid-pod-rs-server/src/main.rs:262
    participant OP as dispatch operator cmd<br/>solid-pod-rs-server/src/main.rs:277
    participant LD as ConfigLoader<br/>solid-pod-rs-server/src/main.rs:280
    participant BIND as bind_available<br/>solid-pod-rs-server/src/main.rs:196
    participant ST as build_storage<br/>solid-pod-rs-server/src/main.rs:180
    participant APP as build_app<br/>solid-pod-rs-server/src/lib.rs:5536

    OS->>M: exec
    M->>CLI: parse argv + env
    M->>M: tracing_subscriber::fmt with RUST_LOG filter<br/>solid-pod-rs-server/src/main.rs:269
    alt an operator subcommand was given
        M->>OP: dispatch and return — no HTTP lifecycle
        OP-->>OS: exit
    end
    M->>LD: with_defaults, optional with_file, with_env<br/>solid-pod-rs-server/src/main.rs:285
    LD-->>M: ServerConfig
    M->>M: host/port CLI overrides then cfg.validate<br/>solid-pod-rs-server/src/main.rs:296
    M->>BIND: bind host:port<br/>solid-pod-rs-server/src/main.rs:300
    BIND-->>M: TcpListener (possibly a shifted port)
    M->>ST: construct Arc dyn Storage<br/>solid-pod-rs-server/src/main.rs:311
    ST-->>M: FsBackend or MemoryBackend
    M->>M: start_forge — feature forge only, ADR-2010 ships it in forge-full builds<br/>solid-pod-rs-server/src/main.rs:460
    M->>APP: HttpServer::new closure per worker<br/>solid-pod-rs-server/src/main.rs:463
    Note over M: INVARIANT: the operator subcommand short-circuits BEFORE any config,<br/>storage or socket work — a one-shot admin command never binds a port.
    Note over APP,M: Graceful shutdown: shutdown_timeout(30) is set on the server before run()<br/>(main.rs:485). Both SIGINT (ctrl_c) and SIGTERM race in a tokio::select and<br/>either one calls handle.stop(graceful=true) (main.rs:489-497). On non-unix,<br/>terminate_signal is std::future::pending — only ctrl_c can trigger shutdown<br/>(main.rs:518).
```

**Tension:** `--ssl-key` and `--ssl-cert` are parsed in every build, but only the `tls` arm reads them (`solid-pod-rs-server/src/main.rs:467`). A build without `tls` takes the plain `listen` arm (`solid-pod-rs-server/src/main.rs:481`) and serves HTTP with no warning that the key and certificate were ignored.

## SP-02.2 AppState assembly order

```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>solid-pod-rs-server/src/main.rs:261
    participant S as AppState::new<br/>solid-pod-rs-server/src/lib.rs:507
    participant MP as log_mempool_selection_once<br/>solid-pod-rs-server/src/main.rs:326
    participant Q as FsQuotaStore::new<br/>solid-pod-rs-server/src/main.rs:332

    M->>S: AppState::new(storage)<br/>solid-pod-rs-server/src/main.rs:318
    S-->>M: defaults — mcp off, quota None, admin_key None<br/>solid-pod-rs-server/src/lib.rs:522
    M->>M: state.data_root from StorageBackendConfig::Fs root
    M->>MP: record endpoint + inferred network once (ADR-2007)
    opt feature quota AND default_quota_bytes > 0
        M->>Q: FsQuotaStore over data_root
    end
    M->>M: allowed_origins<br/>solid-pod-rs-server/src/main.rs:339
    M->>M: admin_key<br/>solid-pod-rs-server/src/main.rs:340
    M->>M: mcp_enabled = cli.mcp AND NOT cli.no_mcp<br/>solid-pod-rs-server/src/main.rs:343
    M->>M: nodeinfo meta incl. open_registrations<br/>solid-pod-rs-server/src/main.rs:347
    M->>M: mashlib mode — module URL beats CDN version<br/>solid-pod-rs-server/src/main.rs:355
    M->>M: conneg_enabled, trust_proxy from CLI or extras<br/>solid-pod-rs-server/src/main.rs:371
    M->>M: public_base_url pinned, must be http(s) else exit 2<br/>solid-pod-rs-server/src/main.rs:373
    M->>M: allowed_hosts trimmed list<br/>solid-pod-rs-server/src/main.rs:385
    M->>M: PodCrossIdentity resolver, remote did:nostr origins only if configured<br/>solid-pod-rs-server/src/main.rs:403
    opt activitypub requested
        M->>M: ApConfig from knobs, keypair load or create, else bail without the feature<br/>solid-pod-rs-server/src/main.rs:429
    end
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
        +conneg  JSS_CONNEG  solid-pod-rs-server/src/main.rs:120
        +trust_proxy  JSS_TRUST_PROXY  solid-pod-rs-server/src/main.rs:127
        +public_base_url  JSS_PUBLIC_BASE_URL  solid-pod-rs-server/src/main.rs:134
        +allowed_hosts  JSS_ALLOWED_HOSTS  solid-pod-rs-server/src/main.rs:140
        +did_nostr_origins  JSS_DID_NOSTR_ORIGINS  solid-pod-rs-server/src/main.rs:145
        +activitypub  JSS_ACTIVITYPUB  solid-pod-rs-server/src/main.rs:150
        +op  operator subcommand  solid-pod-rs-server/src/main.rs:173
    }
    note for Cli "INVARIANT: every one of these defaults to OFF/None. Registration is closed,\nMCP is off, and the admin endpoint 403s with no key — a bare\n`solid-pod-rs-server` opens no back door. trust_proxy is off too, so X-Forwarded-* cannot forge the NIP-98 URL scheme (main.rs:127). The unverified TXO deposit\nstand-in flag (DEPOSIT_TXO_STANDIN_ENABLED) was REMOVED in 0.5.0-alpha.11:\nthe branch it gated no longer exists."
```

**Open:** the off-by-default claim is about the CLI flags. Registration can also open from configuration, since `open_registrations` is true when `extras.invite_only` is false (`solid-pod-rs-server/src/main.rs:350`). The `trust_proxy` claim rests on the scheme helpers honouring proxy headers only when trusted (`solid-pod-rs-server/src/scheme.rs:29`); no request was run through them.

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
        +validate()  solid-pod-rs/src/config/schema.rs:356
    }
    class ServerSection {
        +host  solid-pod-rs/src/config/schema.rs:146
        +port  solid-pod-rs/src/config/schema.rs:150
        +base_url Option  solid-pod-rs/src/config/schema.rs:154
    }
    class AuthConfig {
        +oidc_enabled (default false)  solid-pod-rs/src/config/schema.rs:223
    }
    class SecurityConfig {
        +dotfile_allowlist  solid-pod-rs/src/config/schema.rs:314
        +default_quota_bytes (default 0 = off)  solid-pod-rs/src/config/schema.rs:322
    }
    ServerConfig *-- ServerSection
    ServerConfig *-- AuthConfig
    ServerConfig *-- SecurityConfig
    note for ServerConfig "validate() rejects oidc_enabled = true with no oidc_issuer\n  solid-pod-rs/src/config/schema.rs:360 — a pod cannot claim OIDC with no issuer."
    note for ServerConfig "Resolution order (solid-pod-rs/src/config/loader.rs:79,94,101,127): defaults\n  -> file -> env -> CLI overlay, later layers win. The binary never calls the\n  CLI overlay: it sets host and port by hand after load (main.rs:290).\n  The only loader warning is memory storage with a root set (loader.rs:181);\n  a bad shape is fatal at deserialisation (loader.rs:196). parse_size accepts\n  50MB/1.5GB or bare bytes (solid-pod-rs/src/config/sources.rs:488)."
```

**Drift:** an unparseable boolean in an env var is dropped without a warning; `JSS_OIDC_ENABLED` is inserted only when `parse_bool` succeeds (`solid-pod-rs/src/config/sources.rs:277`).

## SP-02.6 Storage backend selection — only two survive

```mermaid
stateDiagram-v2
    [*] --> Parse
    Parse --> Fs: StorageBackendConfig.Fs<br/>solid-pod-rs-server/src/main.rs:182
    Parse --> Memory: StorageBackendConfig.Memory<br/>solid-pod-rs-server/src/main.rs:189
    Parse --> Rejected: any other backend name

    Fs --> Ready: FsBackend.new(root)<br/>solid-pod-rs/src/storage/fs.rs:47
    Memory --> Ready: MemoryBackend.new()<br/>solid-pod-rs/src/storage/memory.rs:46
    Rejected --> [*]: the value is kept (sources.rs:267), then deserialising<br/>the tagged enum fails the load (loader.rs:196)

    Ready --> [*]

    note right of Rejected
      DOC-DRIFT closed: the S3 backend was REMOVED in 0.5.0-alpha.8.
      Only Fs and Memory exist in the enum today, so an `s3` config
      value never reaches build_storage.
    end note
    note right of Fs
      data_root is captured only for the Fs arm
      (solid-pod-rs-server/src/main.rs:307), so the git and quota
      features are inert on a memory pod. See SP-06.
    end note
```

## SP-02.7 Port binding — busy-port shift

```mermaid
stateDiagram-v2
    [*] --> Requested
    Requested --> Ephemeral: port == 0 (1 attempt)<br/>solid-pod-rs-server/src/main.rs:197
    Requested --> Try0: port != 0 (11 attempts)
    Try0 --> Bound: bind succeeds
    Try0 --> TryN: AddrInUse and attempts remain<br/>solid-pod-rs-server/src/main.rs:213
    TryN --> Bound: warn port busy — shifted listener<br/>solid-pod-rs-server/src/main.rs:206
    TryN --> Failed: attempts exhausted or other io error
    Ephemeral --> Bound
    Bound --> [*]
    Failed --> [*]
    note right of Bound
      The listener is set non-blocking before hand-off to actix
      (solid-pod-rs-server/src/main.rs:202). The actual port is
      re-read from local_addr (main.rs:301), so the default base_url
      reflects the shift; a configured base_url is kept as written
      (main.rs:312).
    end note
```

## SP-02.8 Middleware stack — registration order versus execution order

```mermaid
flowchart TD
    REQ["inbound request"]
    ELM["ErrorLoggingMiddleware (registered first, runs LAST)<br/>solid-pod-rs-server/src/lib.rs:5559"]
    CORS["CorsHeaders<br/>solid-pod-rs-server/src/lib.rs:5560"]
    NORM["NormalizePath TrailingSlash::MergeOnly<br/>solid-pod-rs-server/src/lib.rs:5564"]
    PTG["PathTraversalGuard<br/>solid-pod-rs-server/src/lib.rs:5565"]
    DFG["DotfileGuard plus HostPolicy — 421 for a Host outside allowed_hosts, else 403 on a denied dotfile<br/>solid-pod-rs-server/src/lib.rs:5566"]
    H["route handler"]

    REQ --> DFG --> PTG --> NORM --> CORS --> ELM --> H

    N1["INVARIANT: MergeOnly collapses // to / but never strips the trailing slash —<br/>the trailing slash is the LDP container/resource discriminator.<br/>solid-pod-rs-server/src/lib.rs:5562"]
    NORM -.-> N1
    N2["actix runs wraps in reverse registration order, so ErrorLoggingMiddleware is<br/>the INNERMOST layer. It logs 5xx from handlers only and never sees a 403 or<br/>421 from the outer guards, contrary to the source comment<br/>solid-pod-rs-server/src/lib.rs:5556"]
    ELM -.-> N2
    N3["PathTraversalGuard rejects on path_is_traversal<br/>solid-pod-rs-server/src/lib.rs:3827"]
    PTG -.-> N3
    N4["INVARIANT: with allowed_hosts non-empty a request for any other Host gets 421 Misdirected<br/>Request before any handler runs. Empty allows any host.<br/>solid-pod-rs-server/src/lib.rs:4176, solid-pod-rs-server/src/scheme.rs:256"]
    DFG -.-> N4
```

**Drift:** the product comment says `ErrorLoggingMiddleware` is the outermost layer and observes guard short-circuits (`solid-pod-rs-server/src/lib.rs:5555`). Under actix-web's documented reverse-order rule it is the innermost.

**Tension:** both guards run before `NormalizePath`, so they inspect the raw path before `//` is merged. A 403 or 421 from a guard also never passes through `CorsHeaders`, whose headers are added on the inner service's response (`solid-pod-rs-server/src/lib.rs:3901`), so a browser sees those rejections as CORS failures. This is source reading against the framework rule; no request was run.

## SP-02.9 Route table — discovery, payments, admin and account routes

```mermaid
flowchart LR
    subgraph WK["Well-known (always on)"]
        A1["GET /.well-known/solid<br/>solid-pod-rs-server/src/lib.rs:5577"]
        A2["GET /.well-known/webfinger<br/>solid-pod-rs-server/src/lib.rs:5580"]
        A3["GET /.well-known/nodeinfo<br/>solid-pod-rs-server/src/lib.rs:5583"]
        A4["GET /.well-known/nodeinfo/2.1<br/>solid-pod-rs-server/src/lib.rs:5587"]
        A5["GET /.well-known/apps<br/>solid-pod-rs-server/src/lib.rs:5621"]
    end
    subgraph GATED["Feature-gated discovery"]
        B1["GET /.well-known/did/nostr/{pubkey}.json — did-nostr<br/>solid-pod-rs-server/src/lib.rs:5595"]
        B2["GET /.well-known/nostr.json — nip05-endpoint<br/>solid-pod-rs-server/src/lib.rs:5607"]
        B3["GET /api/exports/all — export-jsonld<br/>solid-pod-rs-server/src/lib.rs:5617"]
    end
    subgraph PAYADMIN["Payments, proxy, admin"]
        C1["GET /pay/.info<br/>solid-pod-rs-server/src/lib.rs:5624"]
        C2["handlers::pay::register — the whole /pay/* surface<br/>solid-pod-rs-server/src/lib.rs:5630"]
        C3["GET /proxy — WAC-gated CORS proxy<br/>solid-pod-rs-server/src/lib.rs:5633"]
        C4["POST /_admin/provision/{pubkey}<br/>solid-pod-rs-server/src/lib.rs:5649"]
    end
    subgraph ACCT["Account and pod management"]
        D1["POST /.pods<br/>solid-pod-rs-server/src/lib.rs:5654"]
        D2["POST /api/accounts/new<br/>solid-pod-rs-server/src/lib.rs:5655"]
        D3["GET /pods/check/{name}<br/>solid-pod-rs-server/src/lib.rs:5656"]
        D4["POST /login/password<br/>solid-pod-rs-server/src/lib.rs:5657"]
        D5["POST /account/password/reset<br/>solid-pod-rs-server/src/lib.rs:5659"]
        D6["POST /account/password/change<br/>solid-pod-rs-server/src/lib.rs:5663"]
    end

    N["INVARIANT: every one of these registers BEFORE the LDP catch-all so a\nreserved prefix is never treated as a pod resource."]
    ACCT -.-> N
```

## SP-02.10 Route table — git, forge and MCP surfaces

```mermaid
flowchart TD
    MCP["POST /mcp + OPTIONS — only when state.mcp_enabled<br/>solid-pod-rs-server/src/lib.rs:5639"]
    FORGE["/forge and /forge/{tail} — cfg(feature forge)<br/>solid-pod-rs-server/src/lib.rs:5674"]
    BLOCK1["ANY /{tail}/.git -> 403<br/>solid-pod-rs-server/src/lib.rs:5685"]
    BLOCK2["ANY /{tail}/.git/{rest} -> 403<br/>solid-pod-rs-server/src/lib.rs:5692"]
    PANEL_OPT["OPTIONS /pods/{pk}/_git/{tail} — registered unconditionally<br/>solid-pod-rs-server/src/lib.rs:5703"]

    subgraph GITON["cfg(feature git)"]
        SMART["GET info/refs, POST git-upload-pack, POST git-receive-pack (routed only so handle_git refuses it — push is not served)<br/>solid-pod-rs-server/src/lib.rs:5716"]
        PANEL["/pods/{pubkey}/_git/{status,log,diff,stage,unstage,commit,branches,branch,discard}<br/>solid-pod-rs-server/src/lib.rs:5722"]
        PROVR["handlers::prov::register — _prov resolve + anchor<br/>solid-pod-rs-server/src/lib.rs:5760"]
    end
    subgraph GITOFF["cfg(not(feature git))"]
        R501["the three smart-HTTP paths return 501<br/>solid-pod-rs-server/src/lib.rs:5772"]
    end

    MCP --> FORGE --> BLOCK1 --> BLOCK2 --> PANEL_OPT --> GITON
    PANEL_OPT --> GITOFF

    N1["INVARIANT: direct .git/ access is blocked unconditionally — the block is\nregistered outside every feature gate, so it survives a git-less build."]
    BLOCK1 -.-> N1
    N2["The forge registers BEFORE the pod-git catch-all so /forge/<o>/<n>.git/info/refs\nreaches the forge's own CGI, not the pod-git handler.<br/>solid-pod-rs-server/src/lib.rs:5668"]
    FORGE -.-> N2
    N3["OPTIONS preflight for the _git panel is registered even without the feature\nso a browser gets a valid CORS answer either way."]
    PANEL_OPT -.-> N3
```

## SP-02.11 Route table — the LDP catch-all, registered last

```mermaid
flowchart LR
    P1["POST /{tail}/ -> handle_post<br/>solid-pod-rs-server/src/lib.rs:5779"]
    P2["PUT /{tail}/ -> handle_put<br/>solid-pod-rs-server/src/lib.rs:5780"]
    G["GET /{tail} -> handle_get<br/>solid-pod-rs-server/src/lib.rs:5781"]
    H["HEAD /{tail} -> handle_get<br/>solid-pod-rs-server/src/lib.rs:5782"]
    PU["PUT /{tail} -> handle_put<br/>solid-pod-rs-server/src/lib.rs:5783"]
    PA["PATCH /{tail} -> handle_patch<br/>solid-pod-rs-server/src/lib.rs:5784"]
    DE["DELETE /{tail} -> handle_delete<br/>solid-pod-rs-server/src/lib.rs:5785"]
    CO["COPY /{tail} -> handle_copy<br/>solid-pod-rs-server/src/lib.rs:5788"]
    OP["OPTIONS /{tail} -> handle_options<br/>solid-pod-rs-server/src/lib.rs:5792"]

    P1 --> P2 --> G --> H --> PU --> PA --> DE --> CO --> OP

    N1["The trailing-slash POST/PUT variants register FIRST so a container write wins\nover the resource catch-all.<br/>solid-pod-rs-server/src/lib.rs:5777"]
    P1 -.-> N1
    N2["INVARIANT: HEAD is routed to handle_get, so HEAD inherits the same WAC read\ngate as GET — a private resource cannot be probed by HEAD. See SP-04.2."]
    H -.-> N2
    N3["COPY is registered by raw method bytes — it is not a standard actix verb.<br/>solid-pod-rs-server/src/lib.rs:5788"]
    CO -.-> N3
```

## SP-02.13 Operator subcommands — the no-HTTP path

```mermaid
flowchart TD
    OP["OperatorCommand<br/>solid-pod-rs-server/src/cli/mod.rs:31"]
    Q["quota reconcile<br/>solid-pod-rs-server/src/cli/mod.rs:68"]
    A["account delete<br/>solid-pod-rs-server/src/cli/mod.rs:104"]
    I["invite create, list, revoke — persisted in the file store under root/.server/invites.json<br/>solid-pod-rs-server/src/cli/mod.rs:126"]
    D["dispatch<br/>solid-pod-rs-server/src/cli/mod.rs:360"]

    OP --> Q
    OP --> A
    OP --> I
    Q --> RQ["run_quota_reconcile<br/>solid-pod-rs-server/src/cli/mod.rs:191"]
    A --> RA["run_account_delete (Prompt-gated)<br/>solid-pod-rs-server/src/cli/mod.rs:286"]
    I --> RI["run_invite_command over FileInviteStore, create goes through run_invite_create<br/>solid-pod-rs-server/src/cli/mod.rs:396"]
    RQ --> D
    RA --> D
    RI --> D

    IV["install — an OperatorCommand variant<br/>solid-pod-rs-server/src/cli/mod.rs:59"]
    INS["run_install — app fetch over git, dispatched at cli/mod.rs:391<br/>solid-pod-rs-server/src/cli/install.rs:231"]
    OP --> IV
    IV --> INS
    INS --> D
    SPEC["parse_app_spec — bare name, org/repo, or full URL<br/>solid-pod-rs-server/src/cli/install.rs:81"]
    AUTH["build_auth_header — NIP-98 mint or bearer<br/>solid-pod-rs-server/src/cli/install.rs:131"]
    INS --> SPEC
    INS --> AUTH

    N["run_account_delete goes through a Prompt trait\n(solid-pod-rs-server/src/cli/mod.rs:257) so the destructive path is\nconfirmable and testable without stdin; --yes skips it\n(solid-pod-rs-server/src/cli/mod.rs:295)."]
    RA -.-> N
```

**Debt:** the shipped `account delete` builds a fresh `InMemoryUserStore` inside `dispatch` (`solid-pod-rs-server/src/cli/mod.rs:380`), so it can never find a real user and always reports "no such user". Its help text promises removal of pods and the WebID profile as well (`solid-pod-rs-server/src/cli/mod.rs:105`), but nothing on that path touches either.

**Debt:** `quota reconcile` is compiled only with the `quota` feature (`solid-pod-rs-server/src/cli/mod.rs:190`). The empty default build ships a fallback that returns a rebuild error (`solid-pod-rs-server/src/cli/mod.rs:250`).
