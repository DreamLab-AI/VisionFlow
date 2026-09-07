---
id: SP-06
title: Storage backends, quota, multitenancy, provisioning and git-versioned pods
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2004, ADR-2005]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/fs.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/memory.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/quota/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/multitenant.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/provision.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/dotfile.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/ssrf.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/cors.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/rate_limit.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/export.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/init.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/service.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/api.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/guard.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/config.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/identity.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/error.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/auth.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/metrics.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/webid.rs
verified_commit: 1d9da5270
---

## SP-06.1 The Storage seam and its two shipped backends

```mermaid
classDiagram
    class Storage {
        <<trait>>
        +get / put / delete / list / head / exists  solid-pod-rs/src/storage/mod.rs:73
        +create_container (default impl)  solid-pod-rs/src/storage/mod.rs:108
        +watch  solid-pod-rs/src/storage/mod.rs:127
    }
    class FsBackend {
        +new(root)  solid-pod-rs/src/storage/fs.rs:47
        -Dir capability handle  solid-pod-rs/src/storage/fs.rs:53
    }
    class MemoryBackend {
        +new()  solid-pod-rs/src/storage/memory.rs:45
    }
    class ResourceMeta {
        +etag / size / content_type  solid-pod-rs/src/storage/mod.rs:27
    }
    class StorageEvent {
        solid-pod-rs/src/storage/mod.rs:58
    }
    Storage <|.. FsBackend
    Storage <|.. MemoryBackend
    Storage ..> ResourceMeta
    Storage ..> StorageEvent
    note for Storage "create_container has a default implementation that writes a .meta marker\n(solid-pod-rs/src/storage/mod.rs:118); a filesystem backend overrides it to make\na real directory (solid-pod-rs/src/storage/fs.rs:350).\nDOC-DRIFT closed: the S3 backend was removed in 0.5.0-alpha.8 — the ecosystem\ndoc's 'S3 is configuration scaffolding only' line now overstates what exists."
```

## SP-06.2 Path normalisation — the traversal gate

```mermaid
flowchart TD
    IN["FsBackend::normalize(path)<br/>solid-pod-rs/src/storage/fs.rs:69"]
    LEAD["empty becomes /, otherwise force a leading slash<br/>solid-pod-rs/src/storage/fs.rs:70"]
    NUL{"contains a NUL byte?<br/>solid-pod-rs/src/storage/fs.rs:77"}
    COMP{"any ParentDir, RootDir or Prefix component?<br/>solid-pod-rs/src/storage/fs.rs:81"}
    ERR["PodError::InvalidPath<br/>solid-pod-rs/src/storage/fs.rs:87"]
    OK["normalised path"]
    REL["relative -> resolve under the capability root<br/>solid-pod-rs/src/storage/fs.rs:97"]

    IN --> LEAD --> NUL
    NUL -- yes --> ERR
    NUL -- no --> COMP
    COMP -- yes --> ERR
    COMP -- no --> OK --> REL

    N["The check is on parsed PATH COMPONENTS, not on a substring — a percent-encoded<br/>or oddly-spelled '..' still parses to Component::ParentDir."]
    COMP -.-> N
    N2["Belt and braces: PathTraversalGuard rejects at the middleware layer too<br/>(solid-pod-rs-server/src/lib.rs:3078), and DotfileGuard filters dotfiles<br/>(solid-pod-rs-server/src/lib.rs:3348). See SP-02.8."]
    ERR -.-> N2
```

## SP-06.3 Capability-confined filesystem access

```mermaid
flowchart LR
    OPEN["Dir::open_ambient_dir(root, ambient_authority())<br/>solid-pod-rs/src/storage/fs.rs:53"]
    CAP["every read/write/delete goes through that Dir handle<br/>solid-pod-rs/src/storage/fs.rs:14"]
    SYM["a symlink pointing outside the root cannot be followed<br/>solid-pod-rs/src/storage/fs.rs:510"]

    OPEN --> CAP --> SYM

    N["cap-std makes escape a TYPE-level property rather than a runtime string check:<br/>the Dir capability cannot name a path outside itself, so the symlink root-escape<br/>class of bug is closed by construction rather than by validation."]
    CAP -.-> N
    N2["A regression test pins it for read, write AND delete<br/>solid-pod-rs/src/storage/fs.rs:511"]
    SYM -.-> N2
```

## SP-06.4 Atomic write and the body/metadata pair

```mermaid
sequenceDiagram
    autonumber
    participant W as FsBackend::put<br/>solid-pod-rs/src/storage/fs.rs:232
    participant A as atomic_write<br/>solid-pod-rs/src/storage/fs.rs:111
    participant D as Dir capability
    participant R as concurrent reader

    W->>A: write the .meta.json sidecar first
    A->>D: create a dotted temp name<br/>solid-pod-rs/src/storage/fs.rs:123
    A->>D: write the bytes, then rename onto the target<br/>solid-pod-rs/src/storage/fs.rs:138
    Note over A: On any failure the temp file is removed<br/>solid-pod-rs/src/storage/fs.rs:142
    W->>A: then atomic_write the body
    R->>D: get() during the window
    D-->>R: the OLD pair, never a half-written one<br/>solid-pod-rs/src/storage/fs.rs:255
    W-->>W: ResourceMeta with a SHA-256 ETag<br/>solid-pod-rs/src/storage/fs.rs:107

    Note over D: list() hides both the META_SUFFIX sidecars and any .solid-pod-tmp- file<br/>solid-pod-rs/src/storage/fs.rs:306
    Note over W: DIVERGENCE (README status section): the 2026-08-19 audit records that<br/>filesystem writes still violate the advertised atomic-storage contract. The<br/>rename is atomic per file, but the body and its sidecar are two renames — a<br/>crash between them leaves a mismatched pair.
```

## SP-06.5 Change events — the notification source

```mermaid
sequenceDiagram
    autonumber
    participant FS as FsBackend::watch<br/>solid-pod-rs/src/storage/fs.rs:365
    participant N as notify recommended_watcher<br/>solid-pod-rs/src/storage/fs.rs:374
    participant CH as mpsc channel
    participant SUB as notification pump — see SP-08

    FS->>N: register a recursive watcher
    N->>CH: raw notify events<br/>solid-pod-rs/src/storage/fs.rs:373
    CH->>CH: drop events whose path ends with META_SUFFIX<br/>solid-pod-rs/src/storage/fs.rs:394
    CH->>CH: map notify::EventKind to StorageEvent<br/>solid-pod-rs/src/storage/fs.rs:404
    CH->>SUB: StorageEvent
    Note over FS: MemoryBackend::watch is the in-process equivalent<br/>solid-pod-rs/src/storage/memory.rs:198
    Note over SUB: Filtering the sidecar matters: a single logical write produces two file<br/>events, and un-filtered they would double every notification.
```

## SP-06.6 Per-pod quota — atomic reservation

```mermaid
sequenceDiagram
    autonumber
    participant H as LDP write handler
    participant RQ as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:2369
    participant Q as FsQuotaStore<br/>solid-pod-rs/src/quota/mod.rs:105
    participant S as Storage
    participant FQ as finish_quota_reservation<br/>solid-pod-rs-server/src/lib.rs:2397

    H->>RQ: pod_name_from_path plus the body size<br/>solid-pod-rs-server/src/lib.rs:2362
    RQ->>Q: reserve(pod, delta_bytes)<br/>solid-pod-rs/src/quota/mod.rs:272
    alt over quota
        Q-->>RQ: QuotaExceeded<br/>solid-pod-rs/src/quota/mod.rs:55
        RQ-->>H: the write never starts
    end
    Q-->>RQ: QuotaReservation<br/>solid-pod-rs-server/src/lib.rs:2357
    H->>S: perform the write
    H->>FQ: settle the reservation with the write outcome
    alt write failed
        FQ->>Q: record a compensating negative delta<br/>solid-pod-rs/src/quota/mod.rs:327
    end
    Note over Q: The .quota.json sidecar is written to a .quota.json.tmp-<pid>-<nanos><br/>then renamed, so the counter is never observed half-updated<br/>solid-pod-rs/src/quota/mod.rs:158
    Note over Q: reconcile (solid-pod-rs/src/quota/mod.rs:85) recomputes from a directory walk<br/>and sweeps stale .quota.json.tmp-* files (solid-pod-rs/src/quota/mod.rs:212) —<br/>the operator CLI drives it — see SP-02.13.
```

## SP-06.7 Multitenancy — path versus subdomain resolution

```mermaid
flowchart TD
    RES["PodResolver trait<br/>solid-pod-rs/src/multitenant.rs:38"]
    PR["PathResolver — host ignored<br/>solid-pod-rs/src/multitenant.rs:49"]
    SR["SubdomainResolver<br/>solid-pod-rs/src/multitenant.rs:68"]
    SP["strip_port<br/>solid-pod-rs/src/multitenant.rs:174"]
    FL["is_file_like_label — pass a label like index.html straight through<br/>solid-pod-rs/src/multitenant.rs:149"]
    SD["scrub_dotdot — ITERATIVE, loops until stable<br/>solid-pod-rs/src/multitenant.rs:184"]
    OUT["ResolvedPath<br/>solid-pod-rs/src/multitenant.rs:28"]

    RES --> PR --> OUT
    RES --> SR --> SP
    SR --> FL
    SR --> SD --> OUT

    N["INVARIANT: scrub_dotdot must stay ITERATIVE. A single pass leaves the '....//'<br/>bypass — the outer removal re-forms a '..' from the fragments the inner pass left<br/>behind. Pinned by scrub_dotdot_iterative_defeats_bypass<br/>solid-pod-rs/src/multitenant.rs:223"]
    SD -.-> N
    N2["A second, explicit guard rejects any surviving '..' before the label is used as<br/>a pod name<br/>solid-pod-rs/src/multitenant.rs:118"]
    SD -.-> N2
```

## SP-06.8 Pod provisioning

```mermaid
sequenceDiagram
    autonumber
    participant C as POST /.pods or /_admin/provision/{pubkey}
    participant G as provisioning_gate<br/>solid-pod-rs-server/src/lib.rs:2446
    participant L as PodCreateLimiter::check<br/>solid-pod-rs-server/src/lib.rs:480
    participant PN as provision_named_pod<br/>solid-pod-rs-server/src/lib.rs:2486
    participant P as provision_pod<br/>solid-pod-rs/src/provision.rs:207
    participant H as GitInitHook::try_init_repo<br/>solid-pod-rs/src/provision.rs:340
    participant S as Storage

    C->>G: registration closed unless open_registration or the admin PSK
    G-->>C: 403 when neither applies
    C->>L: one POST /.pods per IP per day<br/>solid-pod-rs-server/src/lib.rs:481
    L-->>C: 429-shaped refusal with the remaining seconds
    C->>PN: valid_pod_name check<br/>solid-pod-rs-server/src/lib.rs:2343
    PN->>P: ProvisionPlan<br/>solid-pod-rs/src/provision.rs:39
    P->>S: WebID profile, type indexes and their ACLs<br/>solid-pod-rs/src/provision.rs:119
    P->>S: build_public_type_index_acl<br/>solid-pod-rs/src/provision.rs:151
    opt feature git-auto-init
        P->>H: provision_pod_ext runs the hook AFTER every file is written<br/>solid-pod-rs/src/provision.rs:355
        H-->>P: errors are logged and swallowed<br/>solid-pod-rs/src/provision.rs:338
    end
    P-->>C: ProvisionOutcome<br/>solid-pod-rs/src/provision.rs:98
    Note over H: INVARIANT: a git-init failure must not roll back or prevent pod creation —<br/>a pod without a repo is a working pod, a half-provisioned pod is not.
    Note over G: check_admin_override matches the key EXACTLY<br/>solid-pod-rs/src/provision.rs:454
```

## SP-06.9 The SSRF policy

```mermaid
flowchart TD
    P["SsrfPolicy::from_env<br/>solid-pod-rs/src/security/ssrf.rs:137"]
    ENV["SSRF_ALLOWLIST / SSRF_DENYLIST / ALLOW_PRIVATE /<br/>ALLOW_LOOPBACK / ALLOW_LINK_LOCAL<br/>solid-pod-rs/src/security/ssrf.rs:21"]
    RC["resolve_and_check(url) -> the pinned IP<br/>solid-pod-rs/src/security/ssrf.rs:206"]
    C4["classify_v4<br/>solid-pod-rs/src/security/ssrf.rs:281"]
    C6["classify_v6 — catches IPv4-in-IPv6 bypasses<br/>solid-pod-rs/src/security/ssrf.rs:330"]
    META["is_known_metadata_hostname<br/>solid-pod-rs/src/security/ssrf.rs:557"]
    M["SecurityMetrics::record_ssrf_block<br/>solid-pod-rs/src/metrics.rs:63"]
    USE1["OIDC config and JWKS fetching — see SP-05.10"]
    USE2["did:nostr WebID resolution — see SP-05.13"]
    USE3["GET /proxy — validate_proxy_target then build_pinned_proxy_client<br/>solid-pod-rs-server/src/lib.rs:2798"]
    USE4["ActivityPub actor-key resolution and delivery — see SP-08"]

    P --> ENV
    P --> RC --> C4
    RC --> C6
    RC --> META
    RC --> M
    RC --> USE1
    RC --> USE2
    RC --> USE3
    RC --> USE4

    N["INVARIANT: the check RESOLVES the host and the caller then PINS the client to<br/>that IP (solid-pod-rs-server/src/lib.rs:2854), so a DNS rebind between the check<br/>and the request cannot redirect the fetch."]
    RC -.-> N
    N2["The proxy also strips hop-by-hop and identity-bearing response headers<br/>solid-pod-rs-server/src/lib.rs:2774"]
    USE3 -.-> N2
```

## SP-06.10 Dotfile allowlist, CORS and rate limiting

```mermaid
classDiagram
    class DotfileAllowlist {
        +DEFAULT_ALLOWED = .acl, .meta, .account  solid-pod-rs/src/security/dotfile.rs:24
        +from_env via DOTFILE_ALLOWLIST  solid-pod-rs/src/security/dotfile.rs:47
        +is_allowed(path)  solid-pod-rs/src/security/dotfile.rs:106
        +is_path_allowed free fn  solid-pod-rs/src/security/dotfile.rs:233
    }
    class CorsPolicy {
        +from_env  solid-pod-rs/src/security/cors.rs:95
        +preflight_headers  solid-pod-rs/src/security/cors.rs:163
        +response_headers  solid-pod-rs/src/security/cors.rs:209
        +DEFAULT_MAX_AGE_SECS 3600  solid-pod-rs/src/security/cors.rs:41
    }
    class RateLimiter {
        <<trait>>
        solid-pod-rs/src/security/rate_limit.rs:112
        +RateLimitSubject  solid-pod-rs/src/security/rate_limit.rs:50
        +RateLimitDecision  solid-pod-rs/src/security/rate_limit.rs:90
    }
    DotfileAllowlist ..> CorsPolicy
    CorsPolicy ..> RateLimiter
    note for DotfileAllowlist "INVARIANT: .account is present in the default allowlist — a pod that could not\nread its own account sidecar would be unusable. Everything else beginning with a\ndot is refused, so the sidecar surface is a closed set."
    note for CorsPolicy "The server's own CorsHeaders middleware (solid-pod-rs-server/src/lib.rs:3105)\nechoes the request Origin when allowed_origins is EMPTY — a local-dev default,\nnot a production one. See SP-02.3."
```

## SP-06.11 Git-versioned pods — repository lifecycle

```mermaid
stateDiagram-v2
    [*] --> Provisioned: provision_pod writes the pod tree
    Provisioned --> Initialised: GitAutoInit.init_repo_at<br/>solid-pod-rs-git/src/init.rs:101
    Provisioned --> Plain: no git feature, or the hook failed
    Initialised --> Configured: apply_write_config<br/>solid-pod-rs-git/src/config.rs:71
    Configured --> Identified: write_agent_identity<br/>solid-pod-rs-git/src/identity.rs:86
    Identified --> Marked: every LDP write commits — see SP-07
    Marked --> Marked: git-mark per PUT / POST / PATCH
    Plain --> [*]: writes are silently skipped by git_mark_write

    note right of Initialised
      GitAutoInit.with_branch names the initial branch
      solid-pod-rs-git/src/init.rs:77
      find_git_dir locates an existing repo instead of re-initialising
      solid-pod-rs-git/src/config.rs:37
    end note
    note right of Identified
      AGENT_DID_FILE agent.did.json (solid-pod-rs-git/src/identity.rs:40) and the
      nostr.privkey git-config key (solid-pod-rs-git/src/identity.rs:44) bind the
      repo to a did:nostr author, so a commit's author is the same principal WAC
      authorised. See SP-05.12.
    end note
    note right of Plain
      The runtime guard is a .git directory under data_root/{pod} — see SP-07.2.
      A memory-backed or cloud-backed pod is skipped even when the git feature is
      compiled in.
    end note
```

## SP-06.12 Git smart-HTTP — the WAC-gated CGI bridge

```mermaid
sequenceDiagram
    autonumber
    participant C as git client
    participant H as handle_git<br/>solid-pod-rs-server/src/lib.rs:4263
    participant A as BasicNostrExtractor::authorise<br/>solid-pod-rs-git/src/auth.rs:115
    participant W as enforce_write_ctx / enforce_read_ctx
    participant S as GitHttpService::handle<br/>solid-pod-rs-git/src/service.rs:263
    participant CGI as git-http-backend<br/>solid-pod-rs-git/src/service.rs:39

    C->>H: GET info/refs, or POST git-receive-pack
    H->>H: pod name is the first path segment — repo_root = data_root/{pod}<br/>solid-pod-rs-server/src/lib.rs:4287
    alt data_root unset
        H-->>C: 501 git requires fs-backend storage<br/>solid-pod-rs-server/src/lib.rs:4282
    end
    H->>A: resolve did:nostr from the Basic nostr: or Nostr credential
    A-->>H: a pubkey, or anonymous on any failure<br/>solid-pod-rs-server/src/lib.rs:4324
    H->>H: is_write decides the mode<br/>solid-pod-rs-git/src/service.rs:127
    H->>W: enforce against the POD-ROOT container ACL<br/>solid-pod-rs-server/src/lib.rs:4328
    W-->>H: 401 on a private pod so the client retries with credentials
    H->>S: GitRequest
    S->>CGI: spawn_cgi with bounded stdin/stdout reads<br/>solid-pod-rs-git/src/service.rs:354
    CGI-->>S: raw CGI output
    S->>S: parse_cgi_output<br/>solid-pod-rs-git/src/service.rs:498
    S-->>C: GitResponse plus GIT_CORS_HEADERS<br/>solid-pod-rs-git/src/service.rs:188

    Note over H: Before this gate git requests went straight to the CGI: a private pod's<br/>history was anonymously clonable and pushes were anonymous.
    Note over S: path_safe (solid-pod-rs-git/src/guard.rs:68) and extract_repo_slug<br/>(solid-pod-rs-git/src/guard.rs:18) keep the CGI's repo argument inside repo_root.
```

## SP-06.13 The `_git` control-panel REST surface

```mermaid
flowchart LR
    OWN["require_pod_owner_with_body — caller pubkey MUST equal the pod pubkey<br/>solid-pod-rs-server/src/lib.rs:3683"]
    ST["GET status -> git_status<br/>solid-pod-rs-git/src/api.rs:186"]
    LG["GET log -> git_log<br/>solid-pod-rs-git/src/api.rs:300"]
    DF["GET diff -> git_diff<br/>solid-pod-rs-git/src/api.rs:445"]
    AD["POST stage -> git_add<br/>solid-pod-rs-git/src/api.rs:465"]
    US["POST unstage -> git_unstage<br/>solid-pod-rs-git/src/api.rs:485"]
    CM["POST commit -> git_commit<br/>solid-pod-rs-git/src/api.rs:502"]
    BR["GET branches -> git_branches<br/>solid-pod-rs-git/src/api.rs:533"]
    CB["POST branch -> git_create_branch<br/>solid-pod-rs-git/src/api.rs:578"]
    DS["POST discard -> git_discard<br/>solid-pod-rs-git/src/api.rs:591"]

    OWN --> ST
    OWN --> LG
    OWN --> DF
    OWN --> AD
    OWN --> US
    OWN --> CM
    OWN --> BR
    OWN --> CB
    OWN --> DS

    N["INVARIANT: this surface is OWNER-ONLY and does not go through WAC. The gate is<br/>an identity equality check, not an ACL evaluation — history rewriting and<br/>discard are not delegable to a Control holder."]
    OWN -.-> N
    N2["validate_path (solid-pod-rs-git/src/api.rs:161) constrains every caller-supplied<br/>path before it reaches the git argv, and parse_status_output<br/>(solid-pod-rs-git/src/api.rs:192) parses porcelain rather than free text."]
    AD -.-> N2
    N3["resolve_commit (solid-pod-rs-git/src/api.rs:374) backs the _prov commit lookup<br/>— see SP-07.6."]
    LG -.-> N3
```

## SP-06.14 Pod export — the exit right

```mermaid
sequenceDiagram
    autonumber
    participant C as GET /api/exports/all
    participant H as handle_export_all<br/>solid-pod-rs-server/src/lib.rs:2269
    participant E as export_pod_jsonld<br/>solid-pod-rs/src/export.rs:179
    participant W as walk_resources<br/>solid-pod-rs/src/export.rs:122
    participant S as Storage

    C->>H: request the whole pod
    H->>H: owner-WAC gate inside the handler
    H->>E: ExportOptions<br/>solid-pod-rs/src/export.rs:91
    E->>W: recursive walk
    W->>S: list and get each resource
    W-->>E: PodExportEntry per resource<br/>solid-pod-rs/src/export.rs:53
    E-->>H: PodExportBundle<br/>solid-pod-rs/src/export.rs:74
    H-->>C: application/ld+json<br/>solid-pod-rs/src/export.rs:41

    Note over E: PRIVATE_CONTAINER_PREFIX (solid-pod-rs/src/export.rs:45) marks the subtree an<br/>export must treat specially — EXPORT_JSONLD_CONTEXT (solid-pod-rs/src/export.rs:37)<br/>names the bundle's vocabulary.
    Note over H: DIVERGENCE: the export route is behind the default-off export-jsonld feature<br/>(SP-01.8), so the README's 'leave at any time and take everything with you'<br/>claim needs an explicit build flag to hold as an HTTP route. The git clone URL<br/>(solid-pod-rs/src/webid.rs:46) is the always-available exit for a git-backed pod.
```
