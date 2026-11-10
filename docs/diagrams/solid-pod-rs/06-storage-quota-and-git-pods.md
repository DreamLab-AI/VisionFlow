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
  - ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs
verified_commit: 27e18995fc20790fa956acb41f7d8420544a13ec
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
        +new()  solid-pod-rs/src/storage/memory.rs:46
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
    note for Storage "create_container has a default implementation that writes a .meta marker\n(solid-pod-rs/src/storage/mod.rs:118); a filesystem backend overrides it to make\na real directory (solid-pod-rs/src/storage/fs.rs:350).\nThe S3 scaffold was removed in 0.5.0-alpha.8; unsupported backend names\nfail configuration validation (README storage paragraph)."
```

**Drift:** the ecosystem explanation still says "S3 is configuration/dependency scaffolding only; no implementation ships yet" (`crates/solid-pod-rs/docs/explanation/ecosystem-integration.md:105`). At `9b6143c` there is no S3 scaffold at all, and the root README says unsupported backend names, the removed S3 one included, fail validation. Wave-1 audit, 2026-11-02.

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

    N["The check is on parsed PATH COMPONENTS, not on a substring. The storage layer does<br/>not percent-decode: a literal %2e%2e is an ordinary name here. Decoding happens once,<br/>in the server's canonical_pod_path, before this gate runs."]
    COMP -.-> N
    N2["Belt and braces: PathTraversalGuard rejects at the middleware layer too<br/>(solid-pod-rs-server/src/lib.rs:3776, path_is_traversal solid-pod-rs-server/src/lib.rs:3827), and<br/>DotfileGuard filters dotfiles (solid-pod-rs-server/src/lib.rs:4103). See SP-02.8."]
    ERR -.-> N2
    N3["The server also holds ONE canonical spelling: canonical_pod_path percent-decodes once, rejects . and .. segments,<br/>backslash and NUL, and both the permission check and the storage call use its output.<br/>solid-pod-rs-server/src/lib.rs:949"]
    N4["The dotfile filter folds case, so doc.ACL and a long-s x.meta.json variant are refused as guarded names<br/>on case-insensitive filesystems.<br/>solid-pod-rs/src/security/dotfile.rs:306"]
    OK -.-> N3
    COMP -.-> N4
```

## SP-06.3 Capability-confined filesystem access

```mermaid
flowchart LR
    OPEN["Dir::open_ambient_dir(root, ambient_authority())<br/>solid-pod-rs/src/storage/fs.rs:53"]
    CAP["every read, write, list and delete goes through that Dir handle<br/>solid-pod-rs/src/storage/fs.rs:215"]
    SYM["a symlink pointing outside the root cannot be followed<br/>solid-pod-rs/src/storage/fs.rs:510"]

    OPEN --> CAP --> SYM

    N["cap-std makes escape a TYPE-level property rather than a runtime string check:<br/>the Dir capability cannot name a path outside itself, so the symlink root-escape<br/>class of bug is closed by construction rather than by validation."]
    CAP -.-> N
    N2["A regression test pins it for read, write AND delete<br/>solid-pod-rs/src/storage/fs.rs:510"]
    SYM -.-> N2
```

**Tension:** `watch` is the one operation outside the capability. It resolves an ambient `PathBuf` under `root` (`solid-pod-rs/src/storage/fs.rs:368`) and hands it to `notify`, because the watcher API takes host paths. It only observes events and never reads or writes content, but symlinks are not confined on that path.

## SP-06.4 Atomic write and the body/metadata pair

```mermaid
sequenceDiagram
    autonumber
    participant W as FsBackend::put<br/>solid-pod-rs/src/storage/fs.rs:232
    participant A as atomic_write<br/>solid-pod-rs/src/storage/fs.rs:111
    participant D as Dir capability
    participant R as concurrent reader

    W->>W: take the backend-wide write lock<br/>solid-pod-rs/src/storage/fs.rs:238
    W->>A: write the .meta.json sidecar first, tagged with the NEW body ETag
    A->>D: create a dotted temp name<br/>solid-pod-rs/src/storage/fs.rs:123
    A->>D: write the bytes, then rename onto the target<br/>solid-pod-rs/src/storage/fs.rs:138
    Note over A: On any failure the temp file is removed<br/>solid-pod-rs/src/storage/fs.rs:142
    W->>A: then atomic_write the body
    R->>W: get() in the same process waits on the read lock<br/>solid-pod-rs/src/storage/fs.rs:210
    W-->>R: after the body rename, the NEW pair
    Note over R: After a crash between the two renames, read_meta sees a sidecar ETag that does<br/>not match the body and ignores it, falling back to the guessed content type<br/>solid-pod-rs/src/storage/fs.rs:188
    W-->>W: ResourceMeta with a SHA-256 ETag<br/>solid-pod-rs/src/storage/fs.rs:107

    Note over D: list() hides both the META_SUFFIX sidecars and any .solid-pod-tmp- file<br/>solid-pod-rs/src/storage/fs.rs:306
    Note over W: Still two renames, not one. A crash between them leaves a NEW sidecar beside the<br/>OLD body. The ETag tag makes that pair detectable, so its content type and links<br/>are dropped rather than misapplied. The body itself is never torn.
    Note over D: FsBackend::watch (solid-pod-rs/src/storage/fs.rs:365) registers a notify<br/>recommended_watcher (fs.rs:374) and drops any raw event whose path ends<br/>with META_SUFFIX before mapping it to a StorageEvent (fs.rs:394) — a single<br/>logical write produces two file events (body plus sidecar), and un-filtered<br/>they would double every notification the SP-08 pump receives. MemoryBackend::watch<br/>is the in-process equivalent (solid-pod-rs/src/storage/memory.rs:220).
```

**Drift:** this panel previously cited the README status section for an open atomic-storage violation. At `9b6143c` the README says the dated audit's findings are remediated (0.5.0-alpha.13) and describes "atomic replacement writes". The code now pairs a backend-wide `RwLock` (`solid-pod-rs/src/storage/fs.rs:210`, `:238`) with an ETag-tagged sidecar (`solid-pod-rs/src/storage/fs.rs:254-257`) and a reader-side check (`solid-pod-rs/src/storage/fs.rs:185-193`).

**Invariant:** the lock is process-local. A second process writing the same root is not serialised. It relies on the ETag check alone, and between the two renames it can read a stale sidecar's content type.

## SP-06.6 Per-pod quota — atomic reservation

```mermaid
sequenceDiagram
    autonumber
    participant H as LDP write handler
    participant RQ as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:2895
    participant Q as FsQuotaStore<br/>solid-pod-rs/src/quota/mod.rs:105
    participant S as Storage
    participant FQ as finish_quota_reservation<br/>solid-pod-rs-server/src/lib.rs:2923

    H->>RQ: pod_name_from_path, delta = new size minus the current size from head()<br/>solid-pod-rs-server/src/lib.rs:2888
    RQ->>Q: reserve(pod, delta) only when the delta is positive<br/>solid-pod-rs/src/quota/mod.rs:283
    alt over quota
        Q-->>RQ: QuotaExceeded<br/>solid-pod-rs/src/quota/mod.rs:55
        RQ-->>H: the write never starts
    end
    Q-->>RQ: QuotaReservation<br/>solid-pod-rs-server/src/lib.rs:2883
    H->>S: perform the write
    H->>FQ: settle the reservation with the write outcome
    alt write failed after a positive reservation
        FQ->>Q: record a compensating negative delta<br/>solid-pod-rs/src/quota/mod.rs:338
    else write succeeded and the resource shrank
        FQ->>Q: record the negative delta
    end
    Note over Q: The .quota.json sidecar is written to a .quota.json.tmp-<pid>-<nanos><br/>then renamed, so the counter is never observed half-updated<br/>solid-pod-rs/src/quota/mod.rs:169
    Note over Q: reconcile (solid-pod-rs/src/quota/mod.rs:85) recomputes from a directory walk<br/>and sweeps stale .quota.json.tmp-* files (solid-pod-rs/src/quota/mod.rs:223) —<br/>the operator CLI drives it — see SP-02.13.
```

**Open:** the old size is read with `head()` before `reserve` takes the per-pod lock (`solid-pod-rs-server/src/lib.rs:2903-2909`). Two concurrent overwrites of one resource can both compute their delta from the same old size, so the counter can drift until `reconcile` recomputes it. Source tracing only; no concurrent-write probe was run.

## SP-06.7 Multitenancy — normalisation and path fallback

```mermaid
flowchart TD
    RES["PodResolver<br/>solid-pod-rs/src/multitenant.rs:38"] --> PR["PathResolver ignores host<br/>solid-pod-rs/src/multitenant.rs:49"]
    RES --> SR["SubdomainResolver strips port and checks suffix<br/>solid-pod-rs/src/multitenant.rs:77"]
    SR --> FILE{"File-like or non-matching host?"}
    FILE -->|yes| ROOT["No pod selected, original path returned<br/>solid-pod-rs/src/multitenant.rs:148-151"]
    FILE -->|no| SCRUB["Lowercase label, iteratively remove dotdot<br/>solid-pod-rs/src/multitenant.rs:124"]
    SCRUB --> CHECK{"Nonempty, no dot or slash?"}
    CHECK -->|yes| POD["Select normalised pod label<br/>solid-pod-rs/src/multitenant.rs:129-138"]
    CHECK -->|no| ROOT
    POD --> NOTE["al..ice normalises to alice<br/>not an HTTP rejection"]
```

**Boundary:** the helper prevents a residual traversal-like pod label, but normalises malformed labels and falls back to path semantics when it cannot select a pod. [P3](../../estate-review/2026-11-01-solid-policy-parsing.md) verifies these helper outputs; authority and host validation remain separate server contracts.

## SP-06.8 Pod provisioning

```mermaid
sequenceDiagram
    autonumber
    participant C as POST /.pods or POST /api/accounts/new
    participant G as provisioning_gate<br/>solid-pod-rs-server/src/lib.rs:2977
    participant L as PodCreateLimiter::check<br/>solid-pod-rs-server/src/lib.rs:661
    participant IV as FileInviteStore::consume<br/>solid-pod-rs-server/src/cli/invite_store.rs:315
    participant PN as provision_named_pod<br/>solid-pod-rs-server/src/lib.rs:3113
    participant S as Storage
    participant GI as GitAutoInit::init_repo_at<br/>solid-pod-rs-git/src/init.rs:100

    C->>G: pod name, optional x-pod-admin-key and x-invite-token
    G->>G: valid_pod_name check, else 400<br/>solid-pod-rs-server/src/lib.rs:2982
    G->>G: admin key compared in constant time<br/>solid-pod-rs-server/src/lib.rs:2997
    G-->>C: 403 when registration is closed and there is neither the admin key nor an invite<br/>solid-pod-rs-server/src/lib.rs:3005
    G->>L: one creation per IP per 24 h window
    L-->>C: 429 with Retry-After<br/>solid-pod-rs-server/src/lib.rs:3011
    opt registration closed and an invite supplied
        G->>IV: consume the token, recorded before the pod exists<br/>solid-pod-rs-server/src/lib.rs:3034
        IV-->>G: Granted, or 403 expired, used or invalid
    end
    C->>PN: provision the named pod
    PN->>S: owner ACL tree first, root, profile, public and inbox, before any content<br/>solid-pod-rs-server/src/lib.rs:3128
    PN->>S: .meta markers for the root, profile, inbox, public, private and settings containers<br/>solid-pod-rs-server/src/lib.rs:3136
    PN->>S: generated WebID profile card<br/>solid-pod-rs-server/src/lib.rs:3152
    opt feature git and fs-backend data_root
        PN->>GI: git init after the files are written<br/>solid-pod-rs-server/src/lib.rs:3167
        GI-->>PN: failure is logged as a warning, the pod stays created
    end
    PN-->>C: 201 with webId and podUri
    Note over PN: The core provision_pod (solid-pod-rs/src/provision.rs:207), with its type indexes,<br/>their ACLs and the GitInitHook, is NOT called by the server. It remains a library entry point.
    Note over C: POST /_admin/provision/{pubkey} is a separate handler: admin key only, no gate,<br/>no limiter, and it writes a did:nostr owner ACL straight to the filesystem<br/>(solid-pod-rs-server/src/lib.rs:4783).
```

**What it shows:** both public provisioning routes go through the same gate, in this order: name check, closed-registration check, per-IP limiter, invite redemption. Then `provision_named_pod` writes an owner ACL tree, container markers and a profile card through the server's storage.

**Drift:** this panel previously drew the server delegating to the core `provision_pod` and `provision_pod_ext`, and cited `check_admin_override` as the key comparison. At `27e1899` the server still does neither. It writes its own tree, ACLs first (`solid-pod-rs-server/src/lib.rs:3128-3134`), then `.meta` markers (`solid-pod-rs-server/src/lib.rs:3136-3151`), runs `GitAutoInit` directly under the `git` feature (`solid-pod-rs-server/src/lib.rs:3164-3169`), and compares the admin key with `subtle::ConstantTimeEq` (`solid-pod-rs-server/src/lib.rs:2997`). The library's `check_admin_override` (`solid-pod-rs/src/provision.rs:454`) is an exact, non-constant-time match with no in-tree server caller.

**Open:** RESOLVED at solid-pod-rs 27e1899. `provision_named_pod` now writes the pod's ACL tree before any content, so no resource of a new pod is governed by the operator's root ACL: the owner holds Read, Write and Control, the public may read the root, `profile/` and `public/` and append to `inbox/`, and `private/` and `settings/` inherit owner-only (`solid-pod-rs-server/src/lib.rs:3061-3108`, `solid-pod-rs-server/src/lib.rs:3124-3134`). The admin route likewise writes an owner-only ACL naming `did:nostr:<pubkey>` (`solid-pod-rs-server/src/lib.rs:4860-4866`). It writes with `tokio::fs`, outside the `FsBackend` capability handle (SP-06.3), under `data_root/pods/<pubkey>` (`solid-pod-rs-server/src/lib.rs:4845`).

## SP-06.9 The SSRF policy

```mermaid
flowchart TD
    P["SsrfPolicy::from_env<br/>solid-pod-rs/src/security/ssrf.rs:137"]
    ENV["SSRF_ALLOWLIST / SSRF_DENYLIST / ALLOW_PRIVATE /<br/>ALLOW_LOOPBACK / ALLOW_LINK_LOCAL<br/>solid-pod-rs/src/security/ssrf.rs:21"]
    RC["SsrfPolicy::resolve_and_check(url) — first resolved address,<br/>env toggles, metrics<br/>solid-pod-rs/src/security/ssrf.rs:206"]
    C4["classify_v4<br/>solid-pod-rs/src/security/ssrf.rs:281"]
    C6["classify_v6 — catches IPv4-in-IPv6 bypasses<br/>solid-pod-rs/src/security/ssrf.rs:330"]
    FREE["free fn resolve_and_check(host) — fixed policy, every resolved<br/>address checked, no env toggles, no metrics<br/>solid-pod-rs/src/security/ssrf.rs:503"]
    META["is_known_metadata_hostname short-circuit<br/>solid-pod-rs/src/security/ssrf.rs:557"]
    M["SecurityMetrics::record_ssrf_block<br/>solid-pod-rs/src/metrics.rs:63"]
    USE1["OIDC config and JWKS fetching — see SP-05.10"]
    USE2["did:nostr resolution: core DidNostrResolver via the policy (SP-05.22),<br/>solid-pod-rs-nostr resolver via the free fn (SP-05.13)"]
    USE3["GET /proxy — validate_proxy_target then build_pinned_proxy_client<br/>solid-pod-rs-server/src/lib.rs:3547"]
    USE4["ActivityPub — classify only, public addresses only — see SP-08<br/>solid-pod-rs/src/security/ssrf.rs:281"]

    P --> ENV
    P --> RC --> C4
    RC --> C6
    RC --> M
    FREE --> C4
    FREE --> C6
    FREE --> META
    RC --> USE1
    RC --> USE2
    FREE --> USE2
    FREE --> USE3
    C4 --> USE4

    N["INVARIANT: the check RESOLVES the host and the caller then PINS the client to<br/>that IP (solid-pod-rs-server/src/lib.rs:3603), so a DNS rebind between the check<br/>and the request cannot redirect the fetch."]
    RC -.-> N
    N2["The proxy also strips hop-by-hop and identity-bearing response headers<br/>solid-pod-rs-server/src/lib.rs:3725"]
    USE3 -.-> N2
```

**Tension:** there are two SSRF tiers. `SsrfPolicy::resolve_and_check` is env-configurable (`SSRF_ALLOW_PRIVATE` and the rest) and checks only the first resolved address (`solid-pod-rs/src/security/ssrf.rs:223-226`). It serves OIDC JWKS fetching and, through `SsrfPolicy::from_env`, the server's remote did:nostr cross-identity (`crates/solid-pod-rs-server/src/main.rs:417`). The free `resolve_and_check(host)` ignores every env toggle, checks every resolved address (`solid-pod-rs/src/security/ssrf.rs:529-535`) and short-circuits metadata hostnames. It serves `/proxy` (`solid-pod-rs-server/src/lib.rs:3594`) and the `solid-pod-rs-nostr` default check. An operator who sets `SSRF_ALLOWLIST` therefore changes OIDC behaviour but not `/proxy`, and only the policy tier records `record_ssrf_block` metrics. This panel previously drew one entry point serving all four consumers.

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
    STAR["echo the request Origin when sent, literal * only without one<br/>solid-pod-rs/src/security/cors.rs:249"]
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
    N2["The server does NOT use CorsPolicy. Its own CorsHeaders middleware (solid-pod-rs-server/src/lib.rs:3851)<br/>emits * without credentials when the allowlist is empty, and reflects the Origin with<br/>credentials only for a listed origin (solid-pod-rs-server/src/lib.rs:3940). See SP-02.3."]
    ENV -.-> N2
```

**Tension:** the library and the server answer the credentials trap in opposite ways. With wildcard origins and credentials on, `CorsPolicy` reflects any request Origin together with `Access-Control-Allow-Credentials: true` (`solid-pod-rs/src/security/cors.rs:242-247`). The server's `add_cors_headers` comment calls exactly that unsafe and never does it (`solid-pod-rs-server/src/lib.rs:3929-3941`). `CorsPolicy` has no in-tree server consumer, so this only matters to a library user who sets `CORS_ALLOW_CREDENTIALS` without an allowlist.

**Drift:** the doc comments on `CorsHeaders` (`solid-pod-rs-server/src/lib.rs:3850`) and its config field (`solid-pod-rs-server/src/lib.rs:348`) still say an empty allowlist echoes the request Origin. `add_cors_headers` emits `*` instead. This panel had repeated the comment.

## SP-06.11 Git-versioned pods — repository lifecycle

```mermaid
stateDiagram-v2
    [*] --> Provisioned: provision_pod writes the pod tree
    Provisioned --> Initialised: GitAutoInit.init_repo_at<br/>solid-pod-rs-git/src/init.rs:100
    Provisioned --> Plain: no git feature, or the hook failed
    Initialised --> Marked: every LDP write commits — see SP-07
    Initialised --> Configured: apply_write_config, only on a GitHttpService write<br/>solid-pod-rs-git/src/config.rs:69
    Initialised --> Identified: write_agent_identity, library only<br/>solid-pod-rs-git/src/identity.rs:103
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

**Drift:** this panel previously drew a single chain from Initialised through Configured and Identified to Marked. In the native server, `git_mark_write` needs only the `.git` directory (`solid-pod-rs-server/src/lib.rs:4287-4290`), so writes are marked straight after init. `apply_write_config` runs only inside `GitHttpService::handle` for a write request (`solid-pod-rs-git/src/service.rs:358-359`), and `handle_git` refuses every push before the service runs (SP-06.12). `write_agent_identity` has no in-tree caller. `init_repo_at` itself sets only `receive.denyCurrentBranch updateInstead` (`solid-pod-rs-git/src/init.rs:130-140`).

**Open:** pods from `POST /_admin/provision/{pubkey}` live under `data_root/pods/<pubkey>` (SP-06.8). `git_mark_write` takes the first path segment as the pod and looks for `data_root/<segment>/.git` (`solid-pod-rs-server/src/lib.rs:4278-4288`). If those pods are addressed as `/pods/<pubkey>/...`, the repository the admin route creates would never be found and their writes would be skipped as not git-backed. Source tracing suggests this; no request was run to confirm it.

## SP-06.12 Git smart-HTTP — the WAC-gated CGI bridge, clone and fetch only

```mermaid
sequenceDiagram
    autonumber
    participant C as git client
    participant H as handle_git<br/>solid-pod-rs-server/src/lib.rs:5049
    participant A as BasicNostrExtractor::authorise<br/>solid-pod-rs-git/src/auth.rs:115
    participant W as enforce_read_ctx
    participant S as GitHttpService::handle<br/>solid-pod-rs-git/src/service.rs:284
    participant CGI as git-http-backend<br/>solid-pod-rs-git/src/service.rs:41

    C->>H: GET info/refs, or POST git-upload-pack or git-receive-pack
    H->>H: pod name is the first path segment — repo_root = data_root/{pod}<br/>solid-pod-rs-server/src/lib.rs:5073
    alt data_root unset
        H-->>C: 501 git requires fs-backend storage<br/>solid-pod-rs-server/src/lib.rs:5067
    end
    alt pod directory absent
        H-->>C: 404 pod not found, before push refusal or any credential is read<br/>solid-pod-rs-server/src/lib.rs:5074
    end
    alt push or push discovery
        H-->>C: 403 git push is not served, before any credential is read<br/>solid-pod-rs-server/src/lib.rs:5100
    end
    H->>A: resolve did:nostr from the Basic nostr: or Nostr credential
    A-->>H: a pubkey, or anonymous on any failure<br/>solid-pod-rs-server/src/lib.rs:5121
    H->>H: is_write separates push from clone and fetch<br/>solid-pod-rs-git/src/service.rs:129
    H->>W: enforce Read against the POD-ROOT container ACL<br/>solid-pod-rs-server/src/lib.rs:5130
    W-->>H: 401 on a private pod so the client retries with credentials
    alt caller is not the pod owner
        H-->>C: 403 whole-history export requires the pod owner<br/>solid-pod-rs-server/src/lib.rs:5153
    end
    H->>S: GitHttpService rooted at data_root receives the original pod-prefixed GitRequest<br/>solid-pod-rs-server/src/lib.rs:5164
    S->>S: extract slug and join to root again<br/>solid-pod-rs-git/src/service.rs:295<br/>solid-pod-rs-git/src/service.rs:299
    alt selected repository absent
        S-->>C: 404, before CGI transport<br/>solid-pod-rs-git/src/service.rs:338
    else selected repository exists
        S->>CGI: spawn_cgi with bounded stdin/stdout reads<br/>solid-pod-rs-git/src/service.rs:390
        CGI-->>S: raw CGI output
        S->>S: parse_cgi_output<br/>solid-pod-rs-git/src/service.rs:550
        S-->>C: GitResponse plus GIT_CORS_HEADERS<br/>solid-pod-rs-git/src/service.rs:190
    end

    Note over H: Pushes are refused outright (estate register ER-28,<br/>solid-pod-rs-server/src/lib.rs:5710). Read admission names the root resource,<br/>and the owner-only check stands in for per-child or historical versions, which the transport does not evaluate (solid-pod-rs-server/src/lib.rs:5149-5153).
    Note over S: path_safe (solid-pod-rs-git/src/guard.rs:68) and extract_repo_slug<br/>(solid-pod-rs-git/src/guard.rs:18) keep the CGI's repo argument inside repo_root.
    Note over A: The separate owner-only _git control panel requires caller pubkey<br/>to equal pod pubkey. require_pod_owner_with_body is an identity check,<br/>not delegated WAC Control (solid-pod-rs-server/src/lib.rs:4471).<br/>That authority contract differs from the smart-HTTP root Read gate.
```

**Debt:** the pre-receive path guard (`solid-pod-rs-git/src/push_guard.rs:1`) and `GitHttpService::with_push_guard` (`solid-pod-rs-git/src/service.rs:255`) ship in the git crate, but `handle_git` refuses every push before the service runs (`solid-pod-rs-server/src/lib.rs:5100`), so the server installs no guard and the code is unreachable from it.

**Boundary:** the CGI constructs a restricted environment and sets `core.symlinks=false`; this transport setup does not implement per-child or historical WAC. The [Git export assessment](../../estate-review/2026-11-01-solid-git-export-and-baseline.md) separates the root/path mismatch from export authority. Both defects must be considered before making normal clones reachable. The topic is restamped at `9b6143c` by the 2026-11-02 wave-1 audit; SP-14 records the mirror publication repair.

**Drift:** since `1a66c2d` (2026-11-01) the CGI environment no longer sets `uploadpack.allowTipSHA1InWant=true`, as JSS does. It sets `allowTipSHA1InWant`, `allowReachableSHA1InWant` and `allowAnySHA1InWant` to `false` (`solid-pod-rs-git/src/service.rs:425-432`), so a client cannot fetch a hidden ref's objects by their hash. The service's own comment says this is not a read-isolation boundary inside a shared object store (`solid-pod-rs-git/src/service.rs:420-423`). The Boundary paragraph above still holds.

