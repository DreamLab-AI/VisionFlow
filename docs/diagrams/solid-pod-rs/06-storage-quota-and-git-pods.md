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
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/dotfile.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/init.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/service.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/api.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/guard.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/push_guard.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/hardened.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/cli/invite_store.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/config.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/identity.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/auth.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/metrics.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/webid.rs
verified_commit: 76ab7aab3e882f32508b2461b3f51e4c09bdcfa1
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
    N2["Belt and braces: PathTraversalGuard rejects at the middleware layer too<br/>(solid-pod-rs-server/src/lib.rs:3430, path_is_traversal solid-pod-rs-server/src/lib.rs:3481), and<br/>DotfileGuard filters dotfiles (solid-pod-rs-server/src/lib.rs:3757). See SP-02.8."]
    ERR -.-> N2
    N3["The server also holds ONE canonical spelling: canonical_pod_path percent-decodes once, rejects . and .. segments,<br/>backslash and NUL, and both the permission check and the storage call use its output.<br/>solid-pod-rs-server/src/lib.rs:850"]
    N4["The dotfile filter folds case, so doc.ACL and a long-s x.meta.json variant are refused as guarded names<br/>on case-insensitive filesystems.<br/>solid-pod-rs/src/security/dotfile.rs:306"]
    OK -.-> N3
    COMP -.-> N4
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
    participant RQ as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:2652
    participant Q as FsQuotaStore<br/>solid-pod-rs/src/quota/mod.rs:105
    participant S as Storage
    participant FQ as finish_quota_reservation<br/>solid-pod-rs-server/src/lib.rs:2680

    H->>RQ: pod_name_from_path plus the body size<br/>solid-pod-rs-server/src/lib.rs:2645
    RQ->>Q: reserve(pod, delta_bytes)<br/>solid-pod-rs/src/quota/mod.rs:272
    alt over quota
        Q-->>RQ: QuotaExceeded<br/>solid-pod-rs/src/quota/mod.rs:55
        RQ-->>H: the write never starts
    end
    Q-->>RQ: QuotaReservation<br/>solid-pod-rs-server/src/lib.rs:2640
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
    SP["strip_port<br/>solid-pod-rs/src/multitenant.rs:189"]
    FL["is_file_like_label — pass a label like index.html straight through<br/>solid-pod-rs/src/multitenant.rs:164"]
    SD["scrub_dotdot — ITERATIVE, loops until stable<br/>solid-pod-rs/src/multitenant.rs:199"]
    OUT["ResolvedPath<br/>solid-pod-rs/src/multitenant.rs:28"]

    RES --> PR --> OUT
    RES --> SR --> SP
    SR --> FL
    SR --> SD --> OUT

    N["INVARIANT: scrub_dotdot must stay ITERATIVE. A single pass leaves the '....//'<br/>bypass — the outer removal re-forms a '..' from the fragments the inner pass left<br/>behind. Pinned by scrub_dotdot_iterative_defeats_bypass<br/>solid-pod-rs/src/multitenant.rs:245"]
    SD -.-> N
    N2["A second, explicit guard rejects any surviving '..' before the label is used as<br/>a pod name<br/>solid-pod-rs/src/multitenant.rs:133"]
    SD -.-> N2
```

## SP-06.8 Pod provisioning

```mermaid
sequenceDiagram
    autonumber
    participant C as POST /.pods or /_admin/provision/{pubkey}
    participant G as provisioning_gate<br/>solid-pod-rs-server/src/lib.rs:2729
    participant IV as FileInviteStore::consume<br/>solid-pod-rs-server/src/cli/invite_store.rs:315
    participant L as PodCreateLimiter::check<br/>solid-pod-rs-server/src/lib.rs:555
    participant PN as provision_named_pod<br/>solid-pod-rs-server/src/lib.rs:2806
    participant P as provision_pod<br/>solid-pod-rs/src/provision.rs:207
    participant H as GitInitHook::try_init_repo<br/>solid-pod-rs/src/provision.rs:340
    participant S as Storage

    C->>G: pod name, optional x-pod-admin-key and x-invite-token
    G->>G: valid_pod_name check, else 400<br/>solid-pod-rs-server/src/lib.rs:2734
    G-->>C: 403 when registration is closed and there is neither the admin PSK nor an invite<br/>solid-pod-rs-server/src/lib.rs:2757
    C->>L: one POST /.pods per IP per day<br/>solid-pod-rs-server/src/lib.rs:555
    L-->>C: 429-shaped refusal with the remaining seconds<br/>solid-pod-rs-server/src/lib.rs:2763
    opt registration closed and an invite supplied
        G->>IV: consume the token, recorded before the pod exists<br/>solid-pod-rs-server/src/lib.rs:2786
        IV-->>G: Granted, or 403 expired, used or invalid
    end
    C->>PN: provision the named pod
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
    USE3["GET /proxy — validate_proxy_target then build_pinned_proxy_client<br/>solid-pod-rs-server/src/lib.rs:3201"]
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

    N["INVARIANT: the check RESOLVES the host and the caller then PINS the client to<br/>that IP (solid-pod-rs-server/src/lib.rs:3257), so a DNS rebind between the check<br/>and the request cannot redirect the fetch."]
    RC -.-> N
    N2["The proxy also strips hop-by-hop and identity-bearing response headers<br/>solid-pod-rs-server/src/lib.rs:3379"]
    USE3 -.-> N2
```

## SP-06.10 CORS policy — origin echo and the credentials trap

```mermaid
flowchart TD
    ENV["CorsPolicy::from_env<br/>solid-pod-rs/src/security/cors.rs:98"]
    PF["preflight_headers(origin, method, headers)<br/>solid-pod-rs/src/security/cors.rs:166"]
    RF["response_headers(origin)<br/>solid-pod-rs/src/security/cors.rs:212"]
    EO["echo_origin<br/>solid-pod-rs/src/security/cors.rs:239"]
    WC{"AllowedOrigins::Wildcard?<br/>solid-pod-rs/src/security/cors.rs:241"}
    CRED{"allow_credentials?"}
    ECHO["echo the concrete request Origin<br/>solid-pod-rs/src/security/cors.rs:243"]
    STAR["emit literal *"]
    EX{"AllowedOrigins::Exact(set)<br/>contains origin?<br/>solid-pod-rs/src/security/cors.rs:252"}
    NONE["None -> caller drops all CORS headers"]
    VARY["Vary: Origin plus Access-Control-Allow-Credentials<br/>solid-pod-rs/src/security/cors.rs:177"]

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
    N2["The server's CorsHeaders middleware (solid-pod-rs-server/src/lib.rs:3505) is the<br/>JSS-compatible caller: an empty allowed_origins list means every Origin is<br/>echoed back — a local-dev default, not a production one. See SP-02.3."]
    ENV -.-> N2
```

## SP-06.11 Git-versioned pods — repository lifecycle

```mermaid
stateDiagram-v2
    [*] --> Provisioned: provision_pod writes the pod tree
    Provisioned --> Initialised: GitAutoInit.init_repo_at<br/>solid-pod-rs-git/src/init.rs:100
    Provisioned --> Plain: no git feature, or the hook failed
    Initialised --> Configured: apply_write_config<br/>solid-pod-rs-git/src/config.rs:69
    Configured --> Identified: write_agent_identity<br/>solid-pod-rs-git/src/identity.rs:103
    Identified --> Marked: every LDP write commits — see SP-07
    Marked --> Marked: git-mark per PUT / POST / PATCH
    Plain --> [*]: writes are silently skipped by git_mark_write

    note right of Initialised
      GitAutoInit.with_branch names the initial branch
      solid-pod-rs-git/src/init.rs:76
      find_git_dir locates an existing repo instead of re-initialising
      solid-pod-rs-git/src/config.rs:35
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

## SP-06.12 Git smart-HTTP — the WAC-gated CGI bridge, clone and fetch only

```mermaid
sequenceDiagram
    autonumber
    participant C as git client
    participant H as handle_git<br/>solid-pod-rs-server/src/lib.rs:4703
    participant A as BasicNostrExtractor::authorise<br/>solid-pod-rs-git/src/auth.rs:115
    participant W as enforce_read_ctx
    participant S as GitHttpService::handle<br/>solid-pod-rs-git/src/service.rs:283
    participant CGI as git-http-backend<br/>solid-pod-rs-git/src/service.rs:40

    C->>H: GET info/refs, or POST git-upload-pack or git-receive-pack
    H->>H: pod name is the first path segment — repo_root = data_root/{pod}<br/>solid-pod-rs-server/src/lib.rs:4727
    alt data_root unset
        H-->>C: 501 git requires fs-backend storage<br/>solid-pod-rs-server/src/lib.rs:4721
    end
    alt request is a write (git-receive-pack or its info/refs discovery)
        H-->>C: 403 git push is not served, before any credential is read<br/>solid-pod-rs-server/src/lib.rs:4754
    end
    H->>A: resolve did:nostr from the Basic nostr: or Nostr credential
    A-->>H: a pubkey, or anonymous on any failure<br/>solid-pod-rs-server/src/lib.rs:4775
    H->>H: is_write separates push from clone and fetch<br/>solid-pod-rs-git/src/service.rs:128
    H->>W: enforce Read against the POD-ROOT container ACL<br/>solid-pod-rs-server/src/lib.rs:4784
    W-->>H: 401 on a private pod so the client retries with credentials
    H->>S: GitRequest
    S->>CGI: spawn_cgi with bounded stdin/stdout reads<br/>solid-pod-rs-git/src/service.rs:389
    CGI-->>S: raw CGI output
    S->>S: parse_cgi_output<br/>solid-pod-rs-git/src/service.rs:539
    S-->>C: GitResponse plus GIT_CORS_HEADERS<br/>solid-pod-rs-git/src/service.rs:189

    Note over H: Before this gate git requests went straight to the CGI: a private pod's<br/>history was anonymously clonable and pushes were anonymous. Pushes are now<br/>refused outright (estate register ER-28, owner decision 2026-10-18,<br/>solid-pod-rs-server/src/lib.rs:5342).
    Note over S: path_safe (solid-pod-rs-git/src/guard.rs:68) and extract_repo_slug<br/>(solid-pod-rs-git/src/guard.rs:18) keep the CGI's repo argument inside repo_root.
    Note over A: A separate owner-only `_git` control-panel REST surface (status/log/diff/<br/>stage/unstage/commit/branches/create-branch/discard) is gated by<br/>require_pod_owner_with_body — caller pubkey MUST equal the pod pubkey<br/>(solid-pod-rs-server/src/lib.rs:4125). INVARIANT: that gate is an identity<br/>equality check, not a WAC evaluation — history rewriting and discard are not<br/>delegable to a Control holder, unlike this smart-HTTP surface. validate_path<br/>(solid-pod-rs-git/src/api.rs:160) constrains every caller-supplied path before<br/>it reaches the git argv, parse_status_output (api.rs:191) parses porcelain<br/>rather than free text, and resolve_commit (api.rs:373) backs the _prov commit<br/>lookup — see SP-07.6.
```

**Debt:** the pre-receive path guard (`solid-pod-rs-git/src/push_guard.rs:1`) and `GitHttpService::with_push_guard` (`solid-pod-rs-git/src/service.rs:254`) ship in the git crate, but `handle_git` refuses every push before the service runs (`solid-pod-rs-server/src/lib.rs:4754`), so the server installs no guard and the code is unreachable from it.

**Invariant:** every production `git` spawn goes through the hardened constructors: repo hooks, fsmonitor and ambient config are neutralised (`solid-pod-rs-git/src/hardened.rs:44`), and the CGI also runs with `core.symlinks=false` (`solid-pod-rs-git/src/service.rs:421`).
