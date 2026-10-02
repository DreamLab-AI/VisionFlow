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
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/ssrf.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/cors.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/init.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/service.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/api.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/guard.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/config.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/identity.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/auth.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/metrics.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/webid.rs
verified_commit: 4aeb66c1f083e7c3cda7b9a8762aaaca1f40c711
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
    N2["Belt and braces: PathTraversalGuard rejects at the middleware layer too<br/>(solid-pod-rs-server/src/lib.rs:3062), and DotfileGuard filters dotfiles<br/>(solid-pod-rs-server/src/lib.rs:3329). See SP-02.8."]
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
    Note over D: FsBackend::watch (solid-pod-rs/src/storage/fs.rs:365) registers a notify<br/>recommended_watcher (fs.rs:374) and drops any raw event whose path ends<br/>with META_SUFFIX before mapping it to a StorageEvent (fs.rs:394) — a single<br/>logical write produces two file events (body plus sidecar), and un-filtered<br/>they would double every notification the SP-08 pump receives. MemoryBackend::watch<br/>is the in-process equivalent (solid-pod-rs/src/storage/memory.rs:198).
```

## SP-06.6 Per-pod quota — atomic reservation

```mermaid
sequenceDiagram
    autonumber
    participant H as LDP write handler
    participant RQ as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:2353
    participant Q as FsQuotaStore<br/>solid-pod-rs/src/quota/mod.rs:105
    participant S as Storage
    participant FQ as finish_quota_reservation<br/>solid-pod-rs-server/src/lib.rs:2381

    H->>RQ: pod_name_from_path plus the body size<br/>solid-pod-rs-server/src/lib.rs:2346
    RQ->>Q: reserve(pod, delta_bytes)<br/>solid-pod-rs/src/quota/mod.rs:272
    alt over quota
        Q-->>RQ: QuotaExceeded<br/>solid-pod-rs/src/quota/mod.rs:55
        RQ-->>H: the write never starts
    end
    Q-->>RQ: QuotaReservation<br/>solid-pod-rs-server/src/lib.rs:2341
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
    participant G as provisioning_gate<br/>solid-pod-rs-server/src/lib.rs:2430
    participant L as PodCreateLimiter::check<br/>solid-pod-rs-server/src/lib.rs:464
    participant PN as provision_named_pod<br/>solid-pod-rs-server/src/lib.rs:2470
    participant P as provision_pod<br/>solid-pod-rs/src/provision.rs:207
    participant H as GitInitHook::try_init_repo<br/>solid-pod-rs/src/provision.rs:340
    participant S as Storage

    C->>G: registration closed unless open_registration or the admin PSK
    G-->>C: 403 when neither applies
    C->>L: one POST /.pods per IP per day<br/>solid-pod-rs-server/src/lib.rs:465
    L-->>C: 429-shaped refusal with the remaining seconds
    C->>PN: valid_pod_name check<br/>solid-pod-rs-server/src/lib.rs:2327
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
    USE3["GET /proxy — validate_proxy_target then build_pinned_proxy_client<br/>solid-pod-rs-server/src/lib.rs:2782"]
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

    N["INVARIANT: the check RESOLVES the host and the caller then PINS the client to<br/>that IP (solid-pod-rs-server/src/lib.rs:2838), so a DNS rebind between the check<br/>and the request cannot redirect the fetch."]
    RC -.-> N
    N2["The proxy also strips hop-by-hop and identity-bearing response headers<br/>solid-pod-rs-server/src/lib.rs:2758"]
    USE3 -.-> N2
```

## SP-06.10 CORS policy — origin echo and the credentials trap

```mermaid
flowchart TD
    ENV["CorsPolicy::from_env<br/>solid-pod-rs/src/security/cors.rs:95"]
    PF["preflight_headers(origin, method, headers)<br/>solid-pod-rs/src/security/cors.rs:163"]
    RF["response_headers(origin)<br/>solid-pod-rs/src/security/cors.rs:209"]
    EO["echo_origin<br/>solid-pod-rs/src/security/cors.rs:236"]
    WC{"AllowedOrigins::Wildcard?<br/>solid-pod-rs/src/security/cors.rs:238"}
    CRED{"allow_credentials?"}
    ECHO["echo the concrete request Origin<br/>solid-pod-rs/src/security/cors.rs:240"]
    STAR["emit literal *"]
    EX{"AllowedOrigins::Exact(set)<br/>contains origin?<br/>solid-pod-rs/src/security/cors.rs:249"}
    NONE["None -> caller drops all CORS headers"]
    VARY["Vary: Origin plus Access-Control-Allow-Credentials<br/>solid-pod-rs/src/security/cors.rs:174"]

    ENV --> PF
    ENV --> RF
    PF --> EO
    RF --> EO
    EO --> WC
    WC -- yes --> CRED
    CRED -- yes --> ECHO --> VARY
    CRED -- no --> STAR
    WC -- no --> EX
    EX -- yes --> ECHO
    EX -- no --> NONE

    N["INVARIANT: Access-Control-Allow-Origin: * is invalid with credentials per the<br/>Fetch spec, so wildcard+credentials degrades to echoing the concrete Origin<br/>and always sends Vary: Origin, never *."]
    CRED -.-> N
    N2["The server's CorsHeaders middleware (solid-pod-rs-server/src/lib.rs:3086) is the<br/>JSS-compatible caller: an empty allowed_origins list means every Origin is<br/>echoed back — a local-dev default, not a production one. See SP-02.3."]
    ENV -.-> N2
```

## SP-06.11 Git-versioned pods — repository lifecycle

```mermaid
stateDiagram-v2
    [*] --> Provisioned: provision_pod writes the pod tree
    Provisioned --> Initialised: GitAutoInit.init_repo_at<br/>solid-pod-rs-git/src/init.rs:101
    Provisioned --> Plain: no git feature, or the hook failed
    Initialised --> Configured: apply_write_config<br/>solid-pod-rs-git/src/config.rs:71
    Configured --> Identified: write_agent_identity<br/>solid-pod-rs-git/src/identity.rs:103
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
      AGENT_DID_FILE agent.did.json (solid-pod-rs-git/src/identity.rs:50) and the
      nostr.privkey git-config key (solid-pod-rs-git/src/identity.rs:54) bind the
      repo to a did:nostr author, so a commit's author is the same principal WAC
      authorised. See SP-05.12.
      Since alpha.10, given the secret key, agent.did.json is rendered from the
      full key, so an odd-y key publishes fe70103 (solid-pod-rs-git/src/identity.rs:121);
      without it the identifier-only fe70102 form is kept (solid-pod-rs-git/src/identity.rs:124).
      A secret whose x is not pubkey_hex is refused and nothing is written
      (solid-pod-rs-git/src/identity.rs:116-120).
    end note
    note right of Plain
      The runtime guard is a .git directory under data_root/{pod} — see SP-07.2.
      A memory-backed or cloud-backed pod is skipped even when the git feature is
      compiled in.
    end note
    note right of Marked
      pod_git_clone_url (solid-pod-rs/src/webid.rs:46) is the always-available exit
      for a git-backed pod: unlike the JSON-LD export route, which sits behind the
      default-off export-jsonld feature (see SP-10), the git clone URL needs no flag.
    end note
```

## SP-06.12 Git smart-HTTP — the WAC-gated CGI bridge

```mermaid
sequenceDiagram
    autonumber
    participant C as git client
    participant H as handle_git<br/>solid-pod-rs-server/src/lib.rs:4244
    participant A as BasicNostrExtractor::authorise<br/>solid-pod-rs-git/src/auth.rs:115
    participant W as enforce_write_ctx / enforce_read_ctx
    participant S as GitHttpService::handle<br/>solid-pod-rs-git/src/service.rs:263
    participant CGI as git-http-backend<br/>solid-pod-rs-git/src/service.rs:39

    C->>H: GET info/refs, or POST git-receive-pack
    H->>H: pod name is the first path segment — repo_root = data_root/{pod}<br/>solid-pod-rs-server/src/lib.rs:4268
    alt data_root unset
        H-->>C: 501 git requires fs-backend storage<br/>solid-pod-rs-server/src/lib.rs:4263
    end
    H->>A: resolve did:nostr from the Basic nostr: or Nostr credential
    A-->>H: a pubkey, or anonymous on any failure<br/>solid-pod-rs-server/src/lib.rs:4305
    H->>H: is_write decides the mode<br/>solid-pod-rs-git/src/service.rs:127
    H->>W: enforce against the POD-ROOT container ACL<br/>solid-pod-rs-server/src/lib.rs:4309
    W-->>H: 401 on a private pod so the client retries with credentials
    H->>S: GitRequest
    S->>CGI: spawn_cgi with bounded stdin/stdout reads<br/>solid-pod-rs-git/src/service.rs:354
    CGI-->>S: raw CGI output
    S->>S: parse_cgi_output<br/>solid-pod-rs-git/src/service.rs:498
    S-->>C: GitResponse plus GIT_CORS_HEADERS<br/>solid-pod-rs-git/src/service.rs:188

    Note over H: Before this gate git requests went straight to the CGI: a private pod's<br/>history was anonymously clonable and pushes were anonymous.
    Note over S: path_safe (solid-pod-rs-git/src/guard.rs:68) and extract_repo_slug<br/>(solid-pod-rs-git/src/guard.rs:18) keep the CGI's repo argument inside repo_root.
    Note over A: A separate owner-only `_git` control-panel REST surface (status/log/diff/<br/>stage/unstage/commit/branches/create-branch/discard) is gated by<br/>require_pod_owner_with_body — caller pubkey MUST equal the pod pubkey<br/>(solid-pod-rs-server/src/lib.rs:3664). INVARIANT: that gate is an identity<br/>equality check, not a WAC evaluation — history rewriting and discard are not<br/>delegable to a Control holder, unlike this smart-HTTP surface. validate_path<br/>(solid-pod-rs-git/src/api.rs:161) constrains every caller-supplied path before<br/>it reaches the git argv, parse_status_output (api.rs:192) parses porcelain<br/>rather than free text, and resolve_commit (api.rs:374) backs the _prov commit<br/>lookup — see SP-07.6.
```
