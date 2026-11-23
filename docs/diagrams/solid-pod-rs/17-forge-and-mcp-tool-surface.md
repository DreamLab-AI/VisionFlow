---
id: SP-17
title: The forge and the MCP tool surface — dispatch, push token, ownership, browsing, mirror and the catalogue
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md]
adrs: [ADR-2009, ADR-2010]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-nostr/src/ws.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/router.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/auth.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/token.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/ownership.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/hosted.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/bodies.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/spine/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/spine/issues.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/repo/browse.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/html/views.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/request.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/config.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/api.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/api_issues.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/labels.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/pulls.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/trail.rs
  - ../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md
  - ../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md
  - ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/security-model.md
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs
  - ../project/agentbox/lib/solid-pod-rs.nix
  - ../solid-pod-rs/crates/solid-pod-rs-server/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-forge/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-forge/README.md
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/web_edit.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/mirror.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/publish.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/tools.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mcp/skills.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/src/operator_relays.rs
verified_commit: {solid-pod-rs: a28a74832dd0a6e47a531e34e4adc46a3946dfd3, agentbox: 59b1aca1cc3541b767872ecbc161c89e65d6ebca}
aliases:
  SP-08.10: SP-17.1
  SP-08.11: SP-17.2
  SP-08.12: SP-17.3
  SP-08.13: SP-17.4
  SP-08.18: SP-17.5
  SP-08.22: SP-17.6
  SP-08.14: SP-17.7
  SP-08.15: SP-17.8
---
Split from SP-08 on 2026-11-23: the forge and the pod-as-MCP-tool surface (formerly SP-08.10 to SP-08.15, SP-08.18 and SP-08.22). ActivityPub, the relay and estate consumers stay in SP-08. Old panel IDs resolve through the `aliases:` map above; the stamp is SP-08's, and the panels were not re-read.

## SP-17.1 The forge — request dispatch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SRV as handle_forge<br/>solid-pod-rs-server/src/lib.rs:6321
    participant RA as ForgeService::resolve_agent<br/>solid-pod-rs-forge/src/lib.rs:392
    participant A as auth::resolve_agent<br/>solid-pod-rs-forge/src/auth.rs:29
    participant F as ForgeService::handle<br/>solid-pod-rs-forge/src/lib.rs:442
    participant R as parse_route<br/>solid-pod-rs-forge/src/router.rs:398
    participant H as the matching h_* handler<br/>solid-pod-rs-forge/src/lib.rs:541

    C->>SRV: any method under /forge
    SRV->>SRV: build ForgeRequest<br/>solid-pod-rs-forge/src/request.rs:23
    SRV->>RA: resolve the caller first<br/>solid-pod-rs-server/src/lib.rs:6372
    RA->>A: with the host pinned to base_url when configured
    A->>A: 1. forge push token, Bearer f1.*<br/>solid-pod-rs-forge/src/auth.rs:42
    A->>A: 2. NIP-98, body-bound when a body exists, replay-checked<br/>solid-pod-rs-forge/src/auth.rs:58<br/>solid-pod-rs-forge/src/auth.rs:59
    A-->>SRV: ForgeAgent<br/>solid-pod-rs-forge/src/ownership.rs:16
    SRV->>F: request and agent
    F->>F: strip_prefix, percent_guard, parse_route<br/>solid-pod-rs-forge/src/router.rs:377<br/>solid-pod-rs-forge/src/lib.rs:472
    F->>R: parse
    R-->>F: Route<br/>solid-pod-rs-forge/src/router.rs:12
    F->>F: private_gate when private_repos is on<br/>solid-pod-rs-forge/src/lib.rs:477
    F->>H: dispatch
    H-->>F: ForgeResponse<br/>solid-pod-rs-forge/src/request.rs:85
    F-->>C: finish_page adds CSP and nosniff to HTML<br/>solid-pod-rs-forge/src/lib.rs:497

    Note over A: A missing or invalid credential resolves to ForgeAgent::Anonymous rather than<br/>an error (solid-pod-rs-forge/src/auth.rs:36, solid-pod-rs-forge/src/auth.rs:46,<br/>solid-pod-rs-forge/src/auth.rs:68). Handlers then apply their own owner checks.
    Note over H: Every repository reader resolves through resolve_repo<br/>(solid-pod-rs-forge/src/lib.rs:644) or gated_git_dir (solid-pod-rs-forge/src/lib.rs:775),<br/>which 503 an unmigrated mirror (solid-pod-rs-forge/src/lib.rs:674) — see SP-14.4.
    Note over R: GitSmart is rebuilt, not forwarded verbatim: only owner/name[.git]/suffix is<br/>accepted (solid-pod-rs-forge/src/lib.rs:821) and the CGI always gets<br/>owner/name.git plus the suffix (solid-pod-rs-forge/src/lib.rs:826). See SP-06.12.
```

**Drift:** `auth.rs` numbers NIP-98 as "Scheme 3" and says scheme 2, a pod session, "is injected by the server" (`solid-pod-rs-forge/src/auth.rs:50`). The reference server never injects one: it always calls `resolve_agent`, and its own comment says a pod session "would be" injected (`solid-pod-rs-server/src/lib.rs:6371`). In this server a forge caller is a token holder, a NIP-98 signer or anonymous.

## SP-17.2 The forge push token

```mermaid
flowchart LR
    KEY["load_or_create_token_key — 32 bytes in .forge-token-key,<br/>created 0600 in one call, never overwritten<br/>solid-pod-rs-forge/src/lib.rs:188<br/>solid-pod-rs-forge/src/lib.rs:209<br/>solid-pod-rs-forge/src/lib.rs:213"]
    MINT["token::mint(key, agent, iat, ttl)<br/>solid-pod-rs-forge/src/token.rs:66"]
    VER["token::verify(key, token, now), constant-time<br/>solid-pod-rs-forge/src/token.rs:87"]
    MAC["mac_tag — HMAC-SHA256 over the payload<br/>solid-pod-rs-forge/src/token.rs:126"]
    V1["TOKEN_VERSION 1, the f1. prefix<br/>solid-pod-rs-forge/src/token.rs:25"]
    ERR["TokenError<br/>solid-pod-rs-forge/src/token.rs:45"]
    API["POST /api/token — Route::ApiToken<br/>solid-pod-rs-forge/src/lib.rs:600"]

    KEY --> MINT --> MAC
    V1 --> MINT
    API --> MINT
    MAC --> VER --> ERR

    N["DIVERGENCE: this is a bespoke HMAC token format rather than an established one.<br/>It exists for the podless did:nostr identity — a git client that cannot mint a<br/>NIP-98 event per request holds a short-lived bearer instead. The version prefix<br/>at least makes a future format migration detectable."]
    MINT -.-> N
    N2["The key is generated on first use and persisted under the plugin dir, so it<br/>survives a restart but is not shared across deployments."]
    KEY -.-> N2
```

**Tension:** a token is "short-lived" only per token. `h_api_token` mints for any non-anonymous agent (`solid-pod-rs-forge/src/lib.rs:1488`), and a valid `f1.` bearer resolves to that agent, so a holder can renew indefinitely. There is no revocation short of replacing the key file. Requiring NIP-98 (not a bearer) to mint would bound a stolen token's life.

**Debt:** the key is two v4 UUIDs (`solid-pod-rs-forge/src/lib.rs:203`), 244 bits from the OS RNG rather than a direct 256-bit fill. RESOLVED at a28a748: it is created with `create_new` and mode `0600` in the same call (`solid-pod-rs-forge/src/lib.rs:209`, `solid-pod-rs-forge/src/lib.rs:213`), so it never exists with the umask's mode. The MAC uses the RustCrypto `hmac` crate.

## SP-17.3 The forge ownership model

```mermaid
classDiagram
    class ForgeAgent {
        <<enum>>
        solid-pod-rs-forge/src/ownership.rs:16
        +owner()  solid-pod-rs-forge/src/ownership.rs:37
        +author_id()  solid-pod-rs-forge/src/ownership.rs:49
        +is_nostr()  solid-pod-rs-forge/src/ownership.rs:60
        +can_write_namespace(owner_segment)  solid-pod-rs-forge/src/ownership.rs:68
    }
    class OwnerKind {
        <<enum>>
        solid-pod-rs-forge/src/ownership.rs:78
        +classify_owner(segment)  solid-pod-rs-forge/src/ownership.rs:88
    }
    class Names {
        +valid_name_segment  solid-pod-rs-forge/src/ownership.rs:101
        +strip_git_suffix  solid-pod-rs-forge/src/router.rs:595
    }
    class ForgeConfig {
        +normalized_prefix  solid-pod-rs-forge/src/config.rs:180
        +chain_allowed  solid-pod-rs-forge/src/config.rs:192
    }
    ForgeAgent ..> OwnerKind
    OwnerKind ..> Names
    ForgeConfig ..> ForgeAgent
    note for ForgeAgent "Ownership is namespace equality: an agent owns exactly the first path\nsegment equal to its owner(). There is no forge ACL. The rule is applied\nroute by route, not through one predicate."
```

**What it shows.** Owner identity is one segment: a pod username or a 64-hex nostr key. Write rights follow from equality with that segment.

**Drift:** `can_write_namespace` is not the single predicate for mutating routes. It guards smart-HTTP push (`solid-pod-rs-forge/src/lib.rs:837`) and trail ownership (`solid-pod-rs-forge/src/trail.rs:921`). Other routes compare `agent.owner()` inline (`solid-pod-rs-forge/src/api.rs:515`, `solid-pod-rs-forge/src/labels.rs:296`) and also let the author of an issue or pull request act on it in someone else's namespace (`solid-pod-rs-forge/src/api_issues.rs:165`, `solid-pod-rs-forge/src/pulls.rs:765`). Two further rules sit beside ownership: `open_edit` bypasses it for edits (SP-17.5), and any write into a mirror's served store is refused with 409 (`solid-pod-rs-forge/src/lib.rs:756`).

## SP-17.4 Forge repository browsing and the issue spine

```mermaid
flowchart TD
    BR["repo::browse — git plumbing<br/>solid-pod-rs-forge/src/repo/browse.rs:49"]
    VR["valid_rev<br/>solid-pod-rs-forge/src/repo/browse.rs:111"]
    VP["valid_repo_path<br/>solid-pod-rs-forge/src/repo/browse.rs:125"]
    LT["list_tree -> parse_ls_tree<br/>solid-pod-rs-forge/src/repo/browse.rs:168"]
    RB["read_blob<br/>solid-pod-rs-forge/src/repo/browse.rs:229"]
    CL["commit_log / commit_patch<br/>solid-pod-rs-forge/src/repo/browse.rs:265<br/>solid-pod-rs-forge/src/repo/browse.rs:326"]
    SP["SpineStore trait, FsSpineStore<br/>solid-pod-rs-forge/src/spine/mod.rs:24"]
    AW["spine atomic_write<br/>solid-pod-rs-forge/src/spine/mod.rs:115"]
    IX["IssueIndex allocate / by_state / set_state<br/>solid-pod-rs-forge/src/spine/issues.rs:106"]
    LD["load_issue_index / save_issue_index<br/>solid-pod-rs-forge/src/spine/issues.rs:142"]
    TP["ThreadPointer<br/>solid-pod-rs-forge/src/spine/issues.rs:44"]
    RT["render_thread — resolve bodies from pods or the hosted store<br/>solid-pod-rs-forge/src/bodies.rs:189"]
    HS["HostedStore for podless authors<br/>solid-pod-rs-forge/src/hosted.rs:23"]
    VIEW["html::views — the rendered pages<br/>solid-pod-rs-forge/src/html/views.rs:10"]

    VR --> BR
    VP --> BR
    BR --> LT
    BR --> RB
    BR --> CL
    SP --> AW
    SP --> LD --> IX --> TP --> RT --> HS
    LT --> VIEW
    RB --> VIEW
    CL --> VIEW
    RT --> VIEW

    N["INVARIANT: valid_rev and valid_repo_path constrain every caller-supplied ref<br/>and path before it becomes a git argv element — the forge shells out, so an<br/>unvalidated rev is an argument-injection vector. list_tree, read_blob and<br/>commit_log call them first."]
    VR -.-> N
    N2["An issue THREAD is a pointer into a pod (or the hosted store), not a copy — the<br/>forge stores the index and the pod stores the content, so the author keeps<br/>custody of their own words. BodyOutcome (solid-pod-rs-forge/src/bodies.rs:105)<br/>is how an unresolvable body renders without breaking the page."]
    TP -.-> N2
    N3["esc (solid-pod-rs-forge/src/request.rs:181), sanitize_filename (:200) and<br/>looks_textual (:217) are the escaping and content-type helpers the views use;<br/>the views module calls its first tier Phase 0 (solid-pod-rs-forge/src/html/views.rs:1)."]
    VIEW -.-> N3
    N4["A full object id that does not resolve is 404 in commit_log, not an empty list<br/>(solid-pod-rs-forge/src/repo/browse.rs:298): a mirror commit that was never<br/>published answers like any other missing object (SP-14.4)."]
    CL -.-> N4
```

## SP-17.5 The forge has outgrown the request-dispatch picture

```mermaid
flowchart TD
    R["Route enum — the first reading plus the routes below<br/>solid-pod-rs-forge/src/router.rs:12"]
    subgraph ALWAYS["Compiled without a feature"]
        FK["ApiFork<br/>solid-pod-rs-forge/src/router.rs:193"]
        CP["ApiCompare<br/>solid-pod-rs-forge/src/router.rs:200"]
        PL["ApiPulls<br/>solid-pod-rs-forge/src/router.rs:209"]
        XL["Xlogin — vendored login widget<br/>solid-pod-rs-forge/src/router.rs:237"]
        ED["ApiEdit — commit one file over HTTP<br/>solid-pod-rs-forge/src/router.rs:279"]
        PV["ApiPreview<br/>solid-pod-rs-forge/src/router.rs:287"]
        RS["RepoSearch<br/>solid-pod-rs-forge/src/router.rs:255"]
    end
    subgraph ANN["Feature announce"]
        NO["ApiNostr — kind 30617 and 30618 for a repo<br/>solid-pod-rs-forge/src/router.rs:169"]
        AN["ApiAnnounce — owner-triggered relay publish<br/>solid-pod-rs-forge/src/router.rs:177"]
        PUB["publish_event, ws and wss relays only<br/>solid-pod-rs-forge/src/publish.rs:240"]
    end
    subgraph MIR["Feature mirror, inert unless configured"]
        AM["ApiMirror<br/>solid-pod-rs-forge/src/router.rs:150"]
        PM["plan_mirror — fetch only when the followed key's state changes<br/>solid-pod-rs-forge/src/mirror.rs:640"]
    end
    subgraph ANC["Feature anchoring"]
        BT["BlocktrailsJson<br/>solid-pod-rs-forge/src/router.rs:101"]
        TR["trail module<br/>solid-pod-rs-forge/src/lib.rs:92"]
    end

    R --> ALWAYS
    R --> ANN
    R --> MIR
    R --> ANC
    NO --> PUB
    AN --> PUB
    AM --> PM

    N["INVARIANT: with open_edit false (the default) an edit needs an owner credential;<br/>with it true no authorisation check runs at all and any caller may edit any repo,<br/>so the service logs a warning when it is on<br/>(default false at solid-pod-rs-forge/src/config.rs:131, warning at solid-pod-rs-forge/src/lib.rs:322, gate at solid-pod-rs-forge/src/web_edit.rs:226)."]
    ED -.-> N
    N2["The module gates are in solid-pod-rs-forge/src/lib.rs:66 to :93 and the feature<br/>declarations in solid-pod-rs-forge/Cargo.toml:57, :61 and :65. Route variants<br/>always parse; the feature decides what the handler does. SP-17.1 to SP-17.4<br/>describe the dispatch, ownership, token and browse paths; they do not enumerate<br/>every route."]
    R -.-> N2
```

**What it shows.** The forge's route table includes fork, compare, pull-request, search, web-edit and preview routes that compile unconditionally, plus NIP-34 announcement (feature `announce`), relay mirroring (feature `mirror`) and repo-held blocktrails (feature `anchoring`). Since SP-GIT-02 the `migrate` module (`solid-pod-rs-forge/src/lib.rs:69`) and the reader gate are compiled without the `mirror` feature, so a plugin directory that was mirrored once stays gated in any build (SP-14.4).

**Why it is this way.** Each optional tier is a feature gate so a plain forge adds neither relay sockets nor the Bitcoin stack. The README states announcement publication is opt-in: with no `announce_relays` nothing is sent (`../solid-pod-rs/crates/solid-pod-rs-forge/README.md:24-33`).

**Drift (ADR-2009 vs code):** ADR-2009 says the mirror is default off and "nothing in `solid-pod-rs-server` starts it" (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md:34`); the server calls `start_forge` at boot when built with `forge` (`../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs:690-692`, now fatal on failure) and reads `FORGE_MIRROR_RELAYS` (`../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs:6167`). ADR-2010 supersedes the default-off clause and defines "on" as compiled in and started at boot, acting on nothing until relays and follow entries are configured (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:43`).

**Drift (ADR text vs the SP-GIT-02 repair):** both ADRs gained a review qualification dated 2026-11-01 (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md:75`, `../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:76`) that still treats hidden refs as the mechanism and complete object withholding as open (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:83`). At this revision the boundary is store isolation plus the reader gate (SP-14.3, SP-14.4), and neither ADR says so.

**Tension (ADR-2010 vs agentbox):** RESOLVED at agentbox 59b1aca as to the pin. ADR-2010 reads accepted / complete / staged (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2010-forge-and-mirror-are-on-in-shipped-builds.md:5-7`), and agentbox's pin enables `forge-full` (`../project/agentbox/lib/solid-pod-rs.nix:18`). At agentbox 59b1aca that pin is `eb22d21` (`../project/agentbox/lib/solid-pod-rs.nix:72`), the v0.5.0-alpha.24 release commit and the parent of the dependency-only a28a748 this topic is stamped at, so a build from it carries SP-16.6 to SP-17.6 as well as the hidden-tip repair and the SP-GIT-02 store isolation. The mirror stays inert while no follow entries are configured. ADR-2009 moved the same way, accepted / complete / staged (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md:5-7`).

**Open:** the forge README (`../solid-pod-rs/crates/solid-pod-rs-forge/README.md:33`) records the web-edit tier as "not included" for announce triggering, while `ApiEdit` is routed (`solid-pod-rs-forge/src/router.rs:279`); nothing states whether an edit announces.

## SP-17.6 The forge mirror dials only the relay address its SSRF check approved

```mermaid
flowchart TD
    EP["relay_endpoints: one parse yields the URL to dial and the<br/>http twin that is judged, ws and wss only<br/>solid-pod-rs-forge/src/mirror.rs:3335<br/>solid-pod-rs-forge/src/mirror.rs:3273"]
    JUDGE["relay_host_allowed under relay_ssrf<br/>solid-pod-rs-forge/src/mirror.rs:2411<br/>solid-pod-rs-forge/src/mirror.rs:3277"]
    WIRE["the server sets relay_ssrf to OperatorRelays<br/>built from FORGE_MIRROR_RELAYS<br/>solid-pod-rs-server/src/lib.rs:6213<br/>solid-pod-rs-server/src/lib.rs:6217"]
    OP["OperatorRelays: an exact listed host and port passes unresolved,<br/>any other host gets DefaultSsrfCheck<br/>solid-pod-rs-forge/src/operator_relays.rs:56<br/>solid-pod-rs-forge/src/operator_relays.rs:67"]
    DIAL["connect_relay: TCP to the approved IP and port,<br/>TLS and Host keep the name<br/>solid-pod-rs-forge/src/mirror.rs:3240<br/>solid-pod-rs-forge/src/mirror.rs:3258"]
    TRUST["no pinned address: an operator-listed relay, dialled as given<br/>solid-pod-rs-forge/src/mirror.rs:3246"]
    CLONE["clone URLs from announcements keep the default policy,<br/>no proxy, no redirects (SP-14.3)<br/>solid-pod-rs-forge/src/mirror.rs:2341<br/>solid-pod-rs-forge/src/mirror.rs:198"]
    EP --> JUDGE
    WIRE --> OP --> JUDGE
    JUDGE -->|approved address| DIAL
    JUDGE -->|operator relay| TRUST
    OP -.->|never reaches| CLONE
    N["INVARIANT: the host the policy judges is the host the socket is opened to,<br/>and a hostname is not resolved a second time between check and connect."]
    DIAL -.-> N
```

The loopback allowance covers the operator's relay sockets alone (SP-MIRROR-WS-REWRITE, AW1-01); the NIP-34 publisher takes the same `OperatorRelays` for its own relay list (`solid-pod-rs-server/src/lib.rs:6238`).
## SP-17.7 The pod as an MCP tool surface

```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant H as handle_mcp<br/>solid-pod-rs-server/src/mcp/mod.rs:159
    participant D as dispatch<br/>solid-pod-rs-server/src/mcp/mod.rs:97
    participant T as call_tool<br/>solid-pod-rs-server/src/mcp/tools.rs:1208
    participant AU as authorise — the REST enforcement path<br/>solid-pod-rs-server/src/mcp/tools.rs:151
    participant W as wac_check — metadata tools<br/>solid-pod-rs-server/src/mcp/tools.rs:105
    participant S as Storage

    A->>H: POST /mcp with a JSON-RPC message
    H->>H: NIP-98 identity from the request<br/>solid-pod-rs-server/src/mcp/mod.rs:179
    H->>H: origin_of fills the McpCtx<br/>solid-pod-rs-server/src/mcp/mod.rs:154
    H->>H: a subscribe call upgrades to SSE here<br/>solid-pod-rs-server/src/mcp/mod.rs:198
    H->>D: other messages
    D->>D: is_allowed_method gate<br/>solid-pod-rs-server/src/mcp/mod.rs:83<br/>solid-pod-rs-server/src/mcp/mod.rs:101
    D->>T: tools/call
    alt resource read or write, skill read
        T->>AU: enforce_read_ctx or enforce_write_ctx
    else list, head, ACL
        T->>W: sidecar paths elevated to Control<br/>solid-pod-rs-server/src/mcp/tools.rs:111
    end
    AU->>S: resolve and evaluate the ACL
    T-->>D: tool_json / tool_text / tool_error<br/>solid-pod-rs-server/src/mcp/mod.rs:70
    D-->>H: rpc_result or rpc_error<br/>solid-pod-rs-server/src/mcp/mod.rs:56
    H-->>A: JSON-RPC response

    Note over H: PROTOCOL_VERSION 2025-03-26<br/>solid-pod-rs-server/src/mcp/mod.rs:25
    Note over T: The endpoint is off by default (SP-02.3) and that default is the documented<br/>mitigation: the security model still says keep MCP disabled<br/>(../solid-pod-rs/crates/solid-pod-rs/docs/explanation/security-model.md:108).
```

**Drift:** the security model's warning (`../solid-pod-rs/crates/solid-pod-rs/docs/explanation/security-model.md:108`, AUD-009) describes an anonymous read path and sidecar operations without Control. At the pinned revision the source has the fixes that audit asked for: resource and skill reads go through `authorise` (`solid-pod-rs-server/src/mcp/tools.rs:151`), and the metadata gate elevates sidecars to Control (`solid-pod-rs-server/src/mcp/tools.rs:111`). An earlier reading attributed the warning to the README, which no longer mentions MCP.

**Open:** whether AUD-009 is closed needs the audit's own regression tests (anonymous read, Write without Control) run against this revision. This trace shows the gate is wired, not that every tool reaches it.

## SP-17.8 The MCP tool catalogue

```mermaid
flowchart LR
    subgraph RES["Resource tools"]
        LR2["list_resources<br/>solid-pod-rs-server/src/mcp/tools.rs:192"]
        RR["read_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:374"]
        WR["write_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:407"]
        CR["create_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:449"]
        DR["delete_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:533"]
        HR["head_resource<br/>solid-pod-rs-server/src/mcp/tools.rs:564"]
    end
    subgraph ACL["ACL tools"]
        RA["read_acl, Read gate on each key the active layout probes<br/>solid-pod-rs-server/src/mcp/tools.rs:787<br/>solid-pod-rs-server/src/mcp/tools.rs:802"]
        WA["write_acl, Write gate on the layout's write key,<br/>then its own lockout guard<br/>solid-pod-rs-server/src/mcp/tools.rs:912<br/>solid-pod-rs-server/src/mcp/tools.rs:738"]
        BA["build_acl_jsonld<br/>solid-pod-rs-server/src/mcp/tools.rs:853"]
    end
    subgraph SK["Skill and doc tools"]
        LS["list_skills<br/>solid-pod-rs-server/src/mcp/tools.rs:592"]
        GS["get_skill<br/>solid-pod-rs-server/src/mcp/tools.rs:612"]
        PS["get_pod_skill<br/>solid-pod-rs-server/src/mcp/tools.rs:640"]
        LD["list_docs / read_docs from the embedded DOCS_DIR<br/>solid-pod-rs-server/src/mcp/tools.rs:676"]
        PI["pod_info<br/>solid-pod-rs-server/src/mcp/tools.rs:708"]
    end
    subgraph JV["Jev tool, feature jev only"]
        PO["propose_operation — proposes, never executes<br/>solid-pod-rs-server/src/mcp/tools.rs:296"]
        JR["jev_rerank of list_resources<br/>solid-pod-rs-server/src/mcp/tools.rs:269"]
    end
    subgraph FED["Federation tool"]
        CRP["call_remote_pod, SSRF-checked target<br/>solid-pod-rs-server/src/mcp/tools.rs:1057<br/>solid-pod-rs-server/src/mcp/tools.rs:1085"]
        FG["federation_gate_path<br/>solid-pod-rs-server/src/mcp/tools.rs:1017"]
        FA["is_forwardable_auth_header allowlist<br/>solid-pod-rs-server/src/mcp/tools.rs:47"]
    end
    SUB["handle_subscribe — SSE change stream<br/>solid-pod-rs-server/src/mcp/tools.rs:1348"]

    RES --> SUB
    ACL --> SUB
    FED --> FG
    FED --> FA

    N["INVARIANT: call_remote_pod forwards only an ALLOWLISTED set of auth headers<br/>(case-insensitively) so an agent cannot smuggle an arbitrary credential through<br/>the pod — pinned by two tests<br/>(solid-pod-rs-server/src/mcp/tools.rs:1450 and :1459)."]
    FA -.-> N
    N2["path_matches_scope (solid-pod-rs-server/src/mcp/tools.rs:1324) bounds an SSE<br/>subscription to the scope the agent asked for, and each event is WAC-checked<br/>before it is framed by sse_event (:1335)."]
    SUB -.-> N2
    N3["Skills are discovered from CONVENTIONAL pod paths<br/>(solid-pod-rs-server/src/mcp/skills.rs:18, with :19 and :20 for apps and bots), so a pod publishes agent<br/>instructions as ordinary WAC-governed resources rather than as server config.<br/>discover_skills (:109) and read_skill (:152) are the two entry points."]
    SK -.-> N3
    N4["propose_operation is listed and dispatched only when the jev feature is built<br/>(solid-pod-rs-server/Cargo.toml:188) AND the process env enables it<br/>(solid-pod-rs-server/src/mcp/tools.rs:1226). It checks WAC read on the container,<br/>drops unreadable children before any Jev call, and proposes an LDP operation<br/>without executing it. jev_rerank only ever sees children the caller may read."]
    JV -.-> N4
```
