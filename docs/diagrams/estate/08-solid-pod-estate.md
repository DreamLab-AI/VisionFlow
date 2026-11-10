---
id: ES-08
title: Solid-pod estate — four deployments, write identity, access control
area: estate
governing:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/docs/DATA-authority-erasure.md
  - ../project/agentbox/docs/INGRESS-identity.md
adrs: [agentbox:ADR-2139, solid-pod-rs:ADR-2008, agentbox:ADR-2145, visionflow:ADR-2019, nostr-rust-forum:ADR-2012, agentbox:ADR-2096, agentbox:ADR-2099, agentbox:ADR-2102, visionclaw:ADR-2111, visionclaw:ADR-2015, visionclaw:ADR-2016, visionclaw:ADR-2017, visionclaw:ADR-2064, visionclaw:ADR-2068, visionclaw:ADR-2106, agentbox:ADR-2064]
sources:
  - ../project/Cargo.toml
  - ../project/src/handlers/solid_proxy_handler.rs
  - ../project/src/services/ontology_generation.rs
  - ../project/src/services/data_reconciliation.rs
  - ../project/src/services/ontology_pull.rs
  - ../project/src/main.rs
  - ../project/src/handlers/mod.rs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/docker-compose.solid-pods.yml
  - ../project/agentbox/docs/user/solid-pod.md
  - ../project/agentbox/lib/solid-pod-rs.nix
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/jev.rs
  - ../project/agentbox/services/agentbox-manifest/src/system_one.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/adr/ADR-2145-the-solid-pod-forge-and-mirror-are-compiled-in-and-inert-until-named.md
  - docs/adr/ADR-2019-forge-and-mirror-on-in-shipped-pods.md
  - ../project/agentbox/docs/user/multi-user-pods.md
  - ../project/agentbox/management-api/lib/pod-signer.js
  - ../project/agentbox/management-api/lib/elevation-publisher.js
  - ../project/agentbox/management-api/lib/agent-identity.js
  - ../project/agentbox/management-api/adapters/index.js
  - ../project/agentbox/management-api/adapters/pods/_solid-http-base.js
  - ../project/agentbox/management-api/adapters/pods/local-solid-rs.js
  - ../project/agentbox/management-api/routes/admin-users.js
  - ../project/client/src/services/SolidPodService.ts
  - ../project/client/src/services/solidPod/ldpClient.ts
  - ../project/client/src/services/api/authInterceptor.ts
  - ../project/client/src/services/solidPod/agentMemory.ts
  - ../project/client/src/services/solidPod/wacManager.ts
  - ../project/client/src/services/solidPod/typeIndex.ts
  - ../project/scripts/backup-sqlite.sh
  - ../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md
  - ../nostr-rust-forum/docs/adr/ADR-2012-d1-ledger-becomes-a-chain-view.md
  - ../project/agentbox/docs/adr/ADR-2099-the-chain-is-the-ledger-of-record.md
  - ../project/agentbox/docs/proposals/sovereign-settlement.md
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93, solid-pod-rs: 27e18995fc20790fa956acb41f7d8420544a13ec, nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df, visionflow: a4a6c20c34c0e298bc4a6725e71e58e00685f64c}
---
## ES-08.1 Four coexisting Solid-pod deployments — topology contrast

```mermaid
flowchart TB
    subgraph VC["VisionClaw process — single binary"]
        FLAG{{"cargo feature #quot;solid-pod-embed#quot;<br/>in default at Cargo.toml:273, declared at Cargo.toml:306 (default ON)"}}
        EMB["solid_proxy_handler.rs<br/>handle_solid_proxy:311"]
        STUB["no routes registered at all<br/>configure_routes twin, solid_proxy_handler.rs:1820-1824"]
        FLAG -->|"feature ON"| EMB
        FLAG -->|"feature OFF"| STUB
    end

    subgraph AB["agentbox container — supervised service"]
        SUP["supervisord [program:solid-pod]<br/>agentbox/flake.nix:2731"]
        SRV["solid-pod-rs-server  port 8484<br/>agentbox/agentbox.toml:539-540"]
        HTTPS["[program:https-bridge]<br/>agentbox/flake.nix:2748"]
        SUP -->|"exec solidPodRsLauncher"| SRV
        HTTPS -->|"TLS terminate to  port 8484"| SRV
    end

    subgraph CF["Cloudflare Tunnel overlay"]
        CFD["cloudflared-pod<br/>agentbox/docker-compose.solid-pods.yml:26-33"]
    end
    CFD -->|"pods-native.dreamlab-ai.com to agentbox port 8484"| SRV

    subgraph LEGACY["JavaScriptSolidServer — DELETED by ADR-2068"]
        JSS["was bin/jss.js  port 3000 at the repo root —<br/>a vendored third-party copy, now removed.<br/>see ES-08.10"]
    end

    MGMT["management-api adapters/pods/local-solid-rs.js<br/>DEFAULT_BASE is loopback port 8484, local-solid-rs.js:25"] -->|"HTTP LDP"| SRV
    CLIENTB["client SolidPodService.ts<br/>SOLID_POD_BASE_URL defaults to /solid, ldpClient.ts:19"] -->|"proxied #quot;/solid/*#quot;"| EMB

    NOTE1["DIVERGENCE: docker-compose.solid-pods.yml is only the Cloudflare<br/>Tunnel sidecar for THIS SAME port 8484 server (deployment 2) - docker-compose.solid-pods.yml:1-7.<br/>It is not a separate multi-user pods deployment. Multi-user pods<br/>(agentbox/docs/user/multi-user-pods.md) is a design scaffold, defaults<br/>off, targets this same server via admin-users.js - see ES-08.5"]
    NOTE2["DIVERGENCE: sources-of-truth conflict (legacy) - Pod claims<br/>write-master for some agent state while Oxigraph, GitHub public::true<br/>and RuVector claim primacy elsewhere per docs/DATA-authority-erasure.md<br/>Known divergences and open items; legacy prose unreconciled, not authority"]
    NOTE3["_solid-http-base.js:11-12 — this base class was renamed from<br/>local-jss.js after the Python/JSS stub was retired 2026-04-25;<br/>wire contract is protocol-level Solid, not JSS-specific"]
```

## ES-08.2 Agent pod write — NIP-98 signing via pod-signer (with unsigned fallback)

```mermaid
sequenceDiagram
    autonumber
    participant AG as Agent<br/>internal caller
    participant ADP as slotConfig<br/>adapters/index.js:38, pods case :62-100
    participant PS as pod-signer<br/>buildPodNip98:70
    participant BR as nostr-bridge or identity port<br/>pod-signer.js:76-78, :92-102
    participant BASE as SolidHttpPodsAdapter<br/>_solid-http-base.js:32
    participant SRV as solid-pod-rs-server<br/>port 8484

    Note over ADP,PS: INVARIANT: nip98 is null unless integrations.solid_pod_rs.sign_requests=true<br/>(pod-signer.js:73) — off keeps prior unsigned behaviour byte-identical (pod-signer.js:33-35).<br/>The SHIPPED manifest turns it on: sign_requests = true, agentbox.toml:590

    ADP->>PS: buildPodNip98(manifest, opts) pod-signer.js:70, called at adapters/index.js:75
    alt sign_requests off or no stack resolved
        PS-->>ADP: return null (pod-signer.js:73)
        Note over ADP,BASE: RESOLVED ADR-2064 (2026-09-05): sign_requests now sets requireSigned on the adapter<br/>even when no signer could be built, so a falsy nip98 fails closed instead of going out unsigned
    else sign_requests on
        PS-->>ADP: nip98 fn method,url,body returning string or null
        ADP->>BASE: withSigner(cfg) attaches opts.nip98 AND opts.requireSigned adapters/index.js:85-88
    end

    AG->>BASE: write(uri, body) _solid-http-base.js:119
    BASE->>BASE: this._fetch = (this._nip98 || this._requireSigned) ? _signedFetch : _rawFetch (line 62-64)
    alt signer configured (this._nip98 set)
        BASE->>PS: nip98(method,url,body) _signedFetch:78,95
        Note over PS,BR: ADR-2122: under role_isolation or sign_source = identity-port the header comes<br/>ONLY from the identity port, no key is read here (pod-signer.js:76-78, :132-136)
        PS->>PS: getSigner() lazy-load, cached (pod-signer.js:106-116), the default source is the<br/>sovereign identity, a named stack only via AGENTBOX_STACK or sign_stack (pod-signer.js:79, :95-101)
        alt key load fails
            PS-->>BASE: null (loadFailed=true, cached — never retried)
            Note over BASE,SRV: RESOLVED ADR-2064 (2026-09-05): _signedFetch throws typed SigningUnavailable<br/>when requireSigned and no header can be built — no request is emitted
            BASE-->>AG: throw SigningUnavailable (_solid-http-base.js:103-106)
        else key loads
            PS->>BR: buildNip98Header(signer,method,url,body) pod-signer.js:127
            BR-->>PS: base64 kind-27235 event
            PS-->>BASE: Authorization header, Nostr token
            BASE->>SRV: PUT/POST with Authorization Nostr token
        end
    else no signer and requireSigned (sign_requests on, signer unbuildable)
        BASE-->>AG: throw SigningUnavailable — no bytes emitted (_solid-http-base.js:84-91)
    else no signer and flag off
        BASE->>SRV: PUT/POST (unsigned, this._rawFetch — the pre-signing baseline)
    end
    SRV-->>BASE: 2xx / 401 / 403 (WAC — see ES-08.4)

    Note over PS: Lifecycle mirrors lib/elevation-publisher.js bridge+signer built ONCE,<br/>loaded lazily on first use, cached — a load failure is cached and never retried
    Note over AG,SRV: RESOLVED ADR-2064: the degraded-boot hole is closed with sign_requests on -<br/>a did:nostr:local placeholder boot (agent-identity.js:42-43, catch at :181) can no longer write<br/>unsigned to a default-deny pod - the slot throws instead. It remains reachable only<br/>with the flag off, where unsigned is the declared baseline
```

## ES-08.3 User pod write — SolidPodService through ldpClient

```mermaid
sequenceDiagram
    autonumber
    participant UI as SolidPodService<br/>SolidPodService.ts:115
    participant LDP as ldpClient<br/>fetchWithAuth ldpClient.ts:92
    participant NA as computeAuthHeaders<br/>authInterceptor.ts:51
    participant PROXY as VisionClaw "/solid" proxy<br/>ES-08.1
    participant SRV as embedded solid-pod-rs<br/>handle_solid_proxy solid_proxy_handler.rs:311

    UI->>UI: setPreference(key,value) SolidPodService.ts:212
    UI->>UI: setPreference calls getPodStructure SolidPodService.ts:213, which calls initPod() :201-202,<br/>POST /solid/pods/init :171-173
    UI->>LDP: putResource(preferences + key .jsonld, doc) SolidPodService.ts:216, ldpClient.ts:153
    LDP->>LDP: resolvePath(path) rewrites JSS hostnames to the proxy prefix ldpClient.ts:54
    LDP->>LDP: fetchWithAuth(url, PUT) ldpClient.ts:92

    alt nostrAuth.isAuthenticated() ldpClient.ts:98
        LDP->>NA: computeAuthHeaders(absoluteUrl, method, body) ldpClient.ts:105
        alt nostrAuth.isDevMode()
            NA-->>LDP: Authorization Bearer dev-session-token plus X-Nostr-Pubkey authInterceptor.ts:62-67
        else NIP-98 mode
            NA-->>LDP: Authorization Nostr token, nostrAuth.signRequest authInterceptor.ts:76-77
        end
        opt signing throws
            LDP->>LDP: logger.warn NIP-98 signing failed ldpClient.ts:107-108, request proceeds unsigned
        end
    else user present but cannot sign
        LDP->>LDP: logger.warn stale auth session ldpClient.ts:110-115, request goes unsigned
    end

    LDP->>PROXY: fetch url with method PUT, headers, credentials include ldpClient.ts:117
    PROXY->>SRV: handle_solid_proxy dispatch — see ES-08.4 for WAC evaluation
    SRV-->>LDP: 2xx / non-ok
    LDP-->>UI: true / false (logger.error on !response.ok, ldpClient.ts:167-169)

    Note over UI,SRV: Client never issues PATCH — ldpClient.ts exposes only<br/>GET/PUT/POST/DELETE/HEAD (fetchJsonLd,fetchTurtle,putResource,<br/>postResource,deleteResource,resourceExists). Every user write is a<br/>whole-resource PUT or a new-resource POST, never a partial PATCH
    Note over UI: PodStructure fields (profile, ontology_contributions,<br/>ontology_proposals, ontology_annotations, preferences, inbox)<br/>SolidPodService.ts:75-82 — see ES-08.7 classDiagram
```

## ES-08.4 WAC-denied read — embedded handle_solid_proxy

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant H as handle_solid_proxy<br/>solid_proxy_handler.rs:311-317
    participant AUTH as authenticate_request<br/>solid_proxy_handler.rs:246-254
    participant ACL as load_acl_for_path<br/>solid_proxy_handler.rs:891
    participant WAC as evaluate_access<br/>solid_pod_rs::wac, imported solid_proxy_handler.rs:63
    participant FS as FsBackend storage

    C->>H: GET /solid/{tail:.*} solid_proxy_handler.rs:1795
    H->>AUTH: authenticate_request(req) line 325
    AUTH->>AUTH: extract_user_identity, parse NIP-98 Authorization header
    alt NIP-98 present and valid
        AUTH-->>H: Ok Some did:nostr pubkey line 248
    else no identity, allow_anonymous=true
        AUTH-->>H: Ok None line 250
    else no identity, allow_anonymous=false
        AUTH-->>H: Err 401 Authentication required line 251-254
        H-->>C: 401, return resp line 327
    end

    H->>H: method_to_mode(method) line 335
    H->>ACL: load_acl_for_path(storage, FsBackend::authorization_path(storage_path)) lines 339-340
    ACL->>FS: get resource.acl then walk parents to root .acl lines 897-929
    FS-->>ACL: AclDocument or None
    ACL-->>H: acl_doc

    H->>WAC: evaluate_access(acl_doc, agent, path, access_mode, None) lines 343-349
    alt allowed
        WAC-->>H: true
        H->>H: dispatch GET/PUT/POST/DELETE/PATCH lines 368-379
        H-->>C: 200, handle_get, ES-08.6 for PATCH
    else denied, agent is None
        WAC-->>H: false
        H-->>C: 401 Authentication required lines 353-357
    else denied, agent present
        WAC-->>H: false
        H-->>C: 403 Access denied, detail WAC denies access_mode access to path lines 358-364
    end

    Note over H: RESOLVED ADR-2067 — the cfg(not(solid-pod-embed)) 503 stub twins are GONE<br/>(solid_proxy_handler.rs:1810-1819). With the feature off,<br/>configure_routes registers NO /solid routes at all (:1820-1824), so an<br/>unfeatured build 404s at the router rather than 503ing in a handler. see ES-08.1
    Note over ACL,FS: WAC resource-specific ACL, containers use dir slash dot acl,<br/>non-containers use resource dot acl WAC spec section 4.1, falls back to<br/>parent-container acl then root acl, solid_proxy_handler.rs:891-931.<br/>FsBackend is the PublishedStorage alias solid_proxy_handler.rs:48, see ES-08.11
```

## ES-08.5 Pod provisioning for a new user

```mermaid
sequenceDiagram
    autonumber
    participant CL as Client<br/>SolidPodService.ts:171
    participant R as init_pod or init_pod_nip98<br/>solid_proxy_handler.rs:1285, :1331
    participant EP as ensure_pod_exists<br/>solid_proxy_handler.rs:1148
    participant PE as pod_exists<br/>solid_proxy_handler.rs:962
    participant CS as create_pod_with_structure<br/>solid_proxy_handler.rs:969
    participant PP as provision_pod<br/>solid_pod_rs::provision, imported solid_proxy_handler.rs:61
    participant FS as FsBackend storage
    participant AU as admin-users.js<br/>POST /admin/users/provision admin-users.js:138

    CL->>R: POST /solid/pods/init with the session auth headers SolidPodService.ts:173
    Note over CL,R: the VisionClaw client calls init (Nostr session, solid_proxy_handler.rs:1290),<br/>not init-nip98, and init-nip98 (route solid_proxy_handler.rs:1793) has no caller in client/src
    R->>R: resolve the user: get_user_from_request :1290, or extract_user_identity :1332<br/>which validates the NIP-98 token at :225
    R->>EP: ensure_pod_exists(state, npub, pubkey, pod_base_url) solid_proxy_handler.rs:1306, :1366
    EP->>PE: pod_exists(storage, npub) solid_proxy_handler.rs:1154
    PE->>FS: exists(/npub/) solid_proxy_handler.rs:963-964
    alt pod already exists
        FS-->>PE: true
        PE-->>EP: true
        EP->>FS: backfill a missing root .acl solid_proxy_handler.rs:1157-1172
        EP-->>R: created=false, pod_structure(pod_base, npub) :1174
    else pod absent
        FS-->>PE: false
        PE-->>EP: false
        EP->>CS: create_pod_with_structure(storage,npub,pubkey,base_url) solid_proxy_handler.rs:1179
        CS->>CS: build ProvisionPlan, containers profile,ontology,<br/>ontology/contributions,proposals,annotations,preferences,inbox solid_proxy_handler.rs:978-990
        CS->>PP: provision_pod(storage, plan) solid_proxy_handler.rs:999
        PP->>FS: write profile card, root acl, each container
        alt provision_pod partial failure
            PP-->>CS: Err(e)
            CS->>CS: warn, continue with manual structure creation solid_proxy_handler.rs:1006-1011
        else success
            PP-->>CS: Ok(outcome), webid, containers_created
        end
        CS-->>EP: PodStructure solid_proxy_handler.rs:1088
        EP-->>R: created=true
    end
    R-->>CL: 200 in both branches, JSON pod_url, webid, created, structure solid_proxy_handler.rs:1369-1375

    Note over EP,CS: OPEN (source tracing): both structure builders pass pod_base, which already ends<br/>in the npub (solid_proxy_handler.rs:975, :1155), to pod_structure, which appends<br/>the npub again (solid_proxy_handler.rs:287-288, called :1088 and :1174).<br/>The pods/check path passes the bare base (:1264). Init therefore reports<br/>base/npub/npub/... container paths. Not exercised at runtime in this audit
    Note over AU: Separate agentbox-side path (management-api), same solid-pod-rs<br/>server, different call: AU calls POST /_admin/provision/pubkey on<br/>solid-pod-rs-server admin-users.js:43, PSK-gated by SOLID_ADMIN_KEY admin-users.js:30, :37-38
    Note over AU: DRIFT RESOLVED: agentbox/docs/user/multi-user-pods.md now says provision<br/>is implemented and only suspend and archive are 501 (multi-user-pods.md:85-93).<br/>Code agrees: provision answers 201 admin-users.js:192, suspend admin-users.js:241-243<br/>and archive admin-users.js:257-259 return 501
    Note over AU: sovereign_mesh.multi_user defaults off agentbox.toml:254-255,<br/>git auto_init true agentbox.toml:246 wires a git repo per pod via<br/>ensurePodGit admin-users.js:112 unless GIT_POD_ENABLED is false admin-users.js:33
```

**Open (doubled npub in the init response):** both structure builders pass a `pod_base` that already ends in the npub (`../project/src/handlers/solid_proxy_handler.rs:975`, `:1155`) to `pod_structure`, which appends the npub again (`../project/src/handlers/solid_proxy_handler.rs:287-288`, called at `:1088` and `:1174`), while `pods/check` passes the bare base (`:1264`). The `PodStructure` the client writes preferences under would therefore name `base/npub/npub/...`. Traced in source only and not exercised.

## ES-08.6 PATCH — N3 Patch and SPARQL-Update on an N-Triples graph, JSON Patch only for absent resources

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant H as handle_patch<br/>solid_proxy_handler.rs:744
    participant D as patch_dialect_from_mime<br/>solid_pod_rs::ldp, imported solid_proxy_handler.rs:54-58
    participant G as Graph::parse_ntriples<br/>solid_pod_rs::ldp
    participant FS as FsBackend storage

    C->>H: PATCH /solid/tail, Content-Type header, body
    H->>D: patch_dialect_from_mime(content_type) solid_proxy_handler.rs:756
    alt unsupported content type
        D-->>H: None
        H-->>C: 415 Unsupported patch format solid_proxy_handler.rs:758-766
    else recognised dialect
        D-->>H: N3 or SparqlUpdate or JsonPatch solid_proxy_handler.rs:757
    end
    H->>H: non-UTF-8 body is 400 Invalid patch body solid_proxy_handler.rs:769-777

    H->>FS: storage.get(path) fetch current resource solid_proxy_handler.rs:780
    alt resource exists
        FS-->>H: current_body, meta
        H->>G: Graph::parse_ntriples(current_str) solid_proxy_handler.rs:786
        Note over H,G: a body that is not N-Triples (Turtle, JSON-LD) fails to parse and is<br/>REPLACED by an empty graph solid_proxy_handler.rs:788
        alt dialect is N3
            H->>H: apply_n3_patch(graph, patch_str) solid_proxy_handler.rs:792
        else dialect is SparqlUpdate
            H->>H: apply_sparql_patch(graph, patch_str) solid_proxy_handler.rs:793
        else dialect is JsonPatch
            H-->>C: 415 JSON Patch not supported for RDF resources solid_proxy_handler.rs:794-800
        end
        alt patch applies
            H->>FS: put(path, new_body ntriples, application/n-triples) solid_proxy_handler.rs:806-812
            FS-->>H: meta with etag
            H-->>C: 200, ETag header solid_proxy_handler.rs:814-817
        else patch fails
            H-->>C: 422 Patch failed solid_proxy_handler.rs:828-831
        end
    else resource absent
        FS-->>H: PodError::NotFound
        H->>H: apply_patch_to_absent(dialect, patch_str) solid_proxy_handler.rs:836
        alt create succeeds
            H->>FS: put(path, new_graph ntriples) solid_proxy_handler.rs:844-846
            H-->>C: 201 Created, ETag header solid_proxy_handler.rs:848-851
        else create fails
            H-->>C: 422 Patch-to-absent failed solid_proxy_handler.rs:862-865
        end
    end

    Note over H: LANDED at the embedded VisionClaw path: N3 Patch and SPARQL-Update apply<br/>against a parsed RDF graph (imports solid_proxy_handler.rs:54-58, apply :792-793).<br/>Non-destructive only for a resource already stored as N-Triples
    Note over C,FS: JSON Patch dialect is accepted by patch_dialect_from_mime<br/>but rejected for existing resources solid_proxy_handler.rs:794-800, only usable<br/>via apply_patch_to_absent on a not-yet-existing resource
    Note over H: agentbox/docs/user/solid-pod.md:56 PATCH dialects table<br/>N3 Patch, SPARQL-Update, JSON Patch matches this handler's dialect set
```

**Tension (PATCH on a non-N-Triples resource):** `Graph::parse_ntriples` returns an error on the first line that is not an N-Triples statement (`../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs:826-837`, read at the `v0.5.0-alpha.21` tag VisionClaw pins), and the handler turns that error into an empty graph (`../project/src/handlers/solid_proxy_handler.rs:786-789`). A PATCH against a Turtle or JSON-LD resource, which is what `putResource` writes by default (`../project/client/src/services/solidPod/ldpClient.ts:153-157`), therefore stores only the patch's own triples as `application/n-triples` (`../project/src/handlers/solid_proxy_handler.rs:805-812`) and drops the previous content. The VisionClaw client never sends PATCH (ES-08.3), so no in-repo caller hits this; any other Solid client could. This was traced in source only and not reproduced. Closing it means refusing the PATCH (409 or 415) when the stored body does not parse, or parsing by the stored content type.

## ES-08.7 Client pod resource model

```mermaid
classDiagram
    class PodStructure {
        +string profile
        +string ontology_contributions
        +string ontology_proposals
        +string ontology_annotations
        +string preferences
        +string inbox
    }
    class PodInfo {
        +bool exists
        +string podUrl
        +string webId
        +string suggestedUrl
        +PodStructure structure
    }
    class PodInitResult {
        +bool success
        +string podUrl
        +string webId
        +bool created
        +PodStructure structure
        +string npub
        +string error
    }
    class JsonLdDocument {
        +string context
        +string type
        +string id
    }
    class TypeIndexDocument {
        +string type
        +List~TypeRegistration~ typeRegistration
    }
    class TypeRegistration {
        +string type
        +string forClass
        +string instance
        +string instanceContainer
    }
    class DiscoveredView {
        +string name
        +string url
    }
    class DiscoveredAgent {
        +string id
        +List~string~ capabilities
    }
    class AclEntry {
        +string agentWebId
        +List~string~ modes Read Write Append only
    }
    class AclMode {
        <<enumeration>>
        Read
        Write
        Append
        Control
    }

    PodInfo --> PodStructure : structure
    PodInitResult --> PodStructure : structure
    JsonLdDocument <.. TypeIndexDocument : extends
    TypeIndexDocument --> TypeRegistration : typeRegistration list
    TypeRegistration ..> DiscoveredView : forClass schema colon ViewAction
    TypeRegistration ..> DiscoveredAgent : forClass vf colon Agent
    AclEntry ..> AclMode : modes are a subset, never Control

    note for PodStructure "SolidPodService.ts:75-82, one container per<br/>pod facet, agent memory containers are agents<br/>slash agentId slash memory not modelled here<br/>agentMemoryContainerPath agentMemory.ts:41"
    note for TypeIndexDocument "typeIndex.ts:44, public Type Index at<br/>settingsBase publicTypeIndex.jsonld :74, ensurePublicTypeIndex:69.<br/>DEBT: settingsBase is cut at the first /settings/ typeIndex.ts:70-73,<br/>but the caller passes the preferences container SolidPodService.ts:447,<br/>which has no /settings/ segment solid_proxy_handler.rs:294"
    note for AclEntry "wacManager.ts:16, modes typed Read, Write or Append wacManager.ts:20,<br/>AclMode :14 adds Control, buildAclTurtle :30 grants owner Read Write Control :46<br/>and the agent its requested modes only :54,<br/>written by writeContainerAcl wacManager.ts:69-76"
```

**Debt (public Type Index path):** `ensurePublicTypeIndex` cuts its path at the first `/settings/` (`../project/client/src/services/solidPod/typeIndex.ts:70-73`). Its caller passes the preferences container (`../project/client/src/services/SolidPodService.ts:447`), which never contains that segment (`../project/src/handlers/solid_proxy_handler.rs:294`), so `indexOf` returns -1 and the Type Index lands at a nine-character prefix of the path. Traced in source only.

## ES-08.8 Supervised solid-pod service lifecycle

```mermaid
stateDiagram-v2
    [*] --> STOPPED : block emitted only when gated flake.nix:2729, solid-pod block flake.nix:2731-2740, https-bridge block flake.nix:2746-2757
    STOPPED --> STARTING : autostart true, flake.nix:2736
    STARTING --> RUNNING : solidPodRsLauncher exec succeeds
    STARTING --> BACKOFF : exec fails within startsecs, supervisord default, none set flake.nix:2731-2740
    BACKOFF --> STARTING : autorestart true, retry
    BACKOFF --> FATAL : retries exceed startretries
    RUNNING --> EXITED : process exits zero or signal
    RUNNING --> STOPPING : supervisorctl stop solid-pod
    EXITED --> STARTING : autorestart true, respawn
    STOPPING --> STOPPED
    FATAL --> [*]

    note right of STARTING : priority 30, user devuser<br/>environment SOLID_POD_PUBLIC_URL, SOLID_ADMIN_KEY<br/>AGENTBOX_REQUIRED_FOR_READINESS true<br/>only when sovereign_mesh is enabled and pods is local-solid-rs
    note right of RUNNING : https-bridge TLS-terminates to this port when<br/>sovereignCfg.https_bridge true
    note right of FATAL : DIVERGENCE, no point-in-time backup for the pod<br/>store, scripts backup-sqlite.sh covers SQLite only<br/>docs/DATA-authority-erasure.md Known divergences,<br/>no cross-store consistent restore, no declared RPO or RTO
```

## ES-08.9 deleteAgentMemory tombstone gap

```mermaid
sequenceDiagram
    autonumber
    participant UI as SolidPodService<br/>deleteAgentMemory SolidPodService.ts:366
    participant AM as agentMemory.ts<br/>deleteAgentMemory agentMemory.ts:191
    participant LDP as ldpClient<br/>deleteResource ldpClient.ts:201
    participant SRV as embedded solid-pod-rs<br/>ES-08.1
    participant RV as RuVector Postgres<br/>mcp claude-flow memory_search HNSW

    UI->>AM: deleteAgentMemory(podPath, agentId, key) SolidPodService.ts:366-369
    AM->>AM: agentMemoryContainerPath(podPath, agentId) agentMemory.ts:41, called at :196
    AM->>LDP: deleteResource(containerPath plus safeKey plus .jsonld) agentMemory.ts:197-198
    LDP->>SRV: HTTP DELETE ldpClient.ts:202-203
    SRV-->>LDP: 2xx, or 404 treated as success ldpClient.ts:205-210
    LDP-->>AM: true
    AM-->>UI: true

    Note over AM,RV: DIVERGENCE, deleteAgentMemory tombstone gap - this call graph<br/>never reaches RuVector. agentMemory.ts:191-199 has no reverse-tombstone<br/>dispatch to mcp claude-flow memory_delete or equivalent
    Note over RV: IF a corresponding embedding row exists and no independent erasure runs,<br/>this deletion path leaves that row untouched. No joined key-to-row lookup<br/>or live post-delete search was performed in this audit.<br/>The historical governing document calls this an erasure hole in<br/>docs/DATA-authority-erasure.md Known divergences and open items
    Note over UI,RV: see ES-07.9 for the RuVector embedding-pipeline side of<br/>this gap in the memory-and-embedding-estate topic
    Note over UI,RV: Invariant 6 of docs/DATA-authority-erasure.md requires<br/>a dropped secondary write to be logged, not swallowed - this call<br/>path does not even attempt the secondary write, let alone log its absence
```

## ES-08.10 JavaScriptSolidServer — legacy JS implementation, REMOVED from the repo

```mermaid
flowchart LR
    subgraph JSS["JavaScriptSolidServer/ — DELETED by ADR-2068"]
        PKG["was a 63 MB VENDORED COPY of the third-party<br/>github.com/JavaScriptSolidServer/JavaScriptSolidServer<br/>name javascript-solid-server, port 3000 default"]
        FEAT["had implemented LDP CRUD, N3 Patch, SPARQL Update,<br/>WAC dot acl, Solid-OIDC, NIP-98 Nostr auth,<br/>Git HTTP backend — all superseded by solid-pod-rs"]
    end

    subgraph RUST["solid-pod-rs (deployments 1 and 2)"]
        EMB2["VisionClaw embedded handler<br/>src/handlers/solid_proxy_handler.rs"]
        SRV2["agentbox supervised service port 8484"]
    end

    JSS -.->|"retired from the agentbox runtime 2026-04-25,<br/>then DELETED from the repo by ADR-2068"| RUST

    NOTE1["_solid-http-base.js:11-12, this client base class was<br/>previously named local-jss.js after the legacy JSS stub,<br/>retired 2026-04-25, renamed since the wire contract is<br/>protocol-level Solid not JSS-specific"]
    NOTE2["agentbox/docs/user/solid-pod.md:73-75, the 2026-04-25<br/>cleanup removed the Python local-jss stub entirely,<br/>either local-solid-rs, external, or off - no second<br/>pod implementation ships in agentbox"]
    NOTE3["RESOLVED ADR-2068 (vc-knowledge, 2026-09-05) — the directory is<br/>GONE from the working tree, not archived. It was a vendored copy of a<br/>third-party project, wired into no supervised process in either repo,<br/>and superseded by solid-pod-rs on both deployment paths. This diagram<br/>is retained as the record of what was removed and why."]
```

## ES-08.11 ADR-2106 — the embedded pod PULLS the published ontology at boot
```mermaid
sequenceDiagram
    autonumber
    participant M as "server boot (solid-pod-embed)"<br/>src/main.rs:927
    participant SP as spawn_boot_pull<br/>src/services/ontology_pull.rs:403
    participant CFG as OntologyPullConfig::from_env<br/>src/services/ontology_pull.rs:81
    participant P1 as pull_once<br/>src/services/ontology_pull.rs:303
    participant REL as "GitHub release ontology-latest"<br/>src/services/ontology_pull.rs:43
    participant ST as "pod Storage (FsBackend)"

    M->>SP: spawn_boot_pull(Arc::clone(solid_state.storage))
    SP->>CFG: read ONTOLOGY_PULL_ENABLED / _URL / _TIMEOUT_SECS / _INTERVAL_SECS<br/>ontology_pull.rs:81-104
    alt disabled
        SP-->>M: log "ontology pull disabled" and return<br/>ontology_pull.rs:408-409
    end
    SP->>P1: loop { pull_once — sleep(interval) } — break when interval None<br/>ontology_pull.rs:420-424
    P1->>REL: GET index.jsonld<br/>ontology_pull.rs:317
    P1->>ST: read the pod's stored manifest, compare buildSha and sourceSha, plus active-generation check<br/>ontology_pull.rs:320-327
    alt buildSha and sourceSha unchanged AND active atomic generation exists
        P1-->>SP: UpToDate, no mutation
    end
    P1->>REL: GET SHA256SUMS and four content resources<br/>ontology_pull.rs:334-340
    P1->>P1: Verify content hashes and manifest hash<br/>ontology_pull.rs:336-351
    Note over P1,ST: Preserve existing ACL, an existence-probe error aborts.<br/>Verified resources are staged as an immutable generation,<br/>then fsynced before atomic pointer activation.
    P1->>ST: publish_ontology five resources<br/>ontology_generation.rs:216
    ST->>ST: Sync content/sidecars, generation directory and pending pointer<br/>publish_with_failure ontology_generation.rs:127, syncs :178-199
    ST->>ST: Rename pointer, then sync root ontology_generation.rs:203-204
    P1-->>SP: Updated only after publication returns success
    Note over P1,ST: Pre-activation failures retain the old generation after restart.<br/>Post-rename root-sync failure means durability unknown,<br/>but visible resources still belong to a complete generation.
    Note over M,REL: Pull remains asynchronous and logs failures.<br/>Canonical manifest exposes visionflow:generation, browser pins<br/>JSON-LD/Turtle reads to immutable @generation paths.<br/>Pinned paths reuse canonical ACLs. Unpinned requests may straddle activation.
    Note over REL: EXTERNAL — the release is produced by the VisionClaw<br/>ontology-publish workflow from the jjohare/visionGraph vault.<br/>Delivery was INVERTED from a CI push to this boot pull because<br/>deploy-jss could never reach an in-process pod from a hosted<br/>runner. see ES-09.13, and VG-04.1 / VG-04.3 for the vault side.
```

The 2026-09-07 audit distinguishes ES-08.9's missing cross-store erasure dispatch from HNSW index degradation. This source path demonstrates neither bulk vector deletion nor degraded recall. ES-08.11 now documents the implemented generation boundary and pinned browser reads. Old unpinned consumers can still straddle activation; post-rename fsync failure does not imply the old pointer survived. The durable reconciliation journal has no production destructive adapters: subject/authority mappings and live erasure acceptance remain outstanding. See the [federation audit](../../estate-review/2026-09-07-federation-audit.md).

## ES-08.12 PROPOSED — the pod ledger becomes a chain view, and the hand-rolled crypto goes
```mermaid
flowchart TB
    CRYPTO["WHAT D1 REMOVED, as the record states the pre-port crate — it owned the estate's whole<br/>Bitcoin transaction construction and HAND-ROLLED it: BIP-341 scripts,<br/>the TapSighash and key-path Schnorr on raw k256, with big-endian mod-n<br/>scalar arithmetic written out by hand<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:36-40;<br/>the rust-bitcoin port that retires this posture is recorded as landed on main,<br/>unreleased (ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:133-172)"]

    PORT["PROPOSED D1 — bitcoin_tx.rs and mrc20.rs port to rust-bitcoin<br/>plus secp256k1; the hand-rolled scalar helpers are DELETED<br/>rather than kept beside the library ones. Highest priority in<br/>the programme because it is the only item that removes<br/>hand-rolled crypto.<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:51-56"]
    CRYPTO --> PORT

    GATE["PROPOSED D2 — the acceptance gate is three cross-implementation<br/>golden tests that must pass BYTE-IDENTICAL before and after the<br/>port. The fixtures are not regenerated to fit the new code.<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:57-63"]
    PORT --> GATE

    VIEW["PROPOSED D4 — WebLedger becomes a derived, height-stamped view.<br/>get_balance reads through the sidestr-node HTTP surface and folds<br/>the UTXOs for that did:nostr, with a bounded cache; a staleness<br/>bound is an ERROR, never a slightly old number.<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:69-73"]
    API["PROPOSED D5 — credit and debit LEAVE the public API. The one<br/>deliberately breaking change in the estate, because leaving the<br/>pair in place leaves a path by which a balance can exist that<br/>the chain does not know about.<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:74-80"]
    VIEW --> API

    TXO["PROPOSED D6 — the TXO stand-in deposit is DELETED, not left<br/>default-off: it credits sats for any parseable TXO URI guarded<br/>only by a replay key. A chain-settled estate must not keep a<br/>reachable free-money oracle.<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:81-85"]
    API --> TXO

    FORUM["PROPOSED, the forum half — the pod-worker D1 ledger is demoted<br/>to a derived view in lockstep, and solid-pod-rs moves with the<br/>host. ADR-2012-d1-ledger-becomes-a-chain-view.md:48-51"]
    VIEW --> FORUM

    PRE["INVARIANT the record sets on itself — non-atomic payment state<br/>is fixed FIRST. The crate's own README reproduces a critical<br/>finding saying not to carry value through the payment routes<br/>until it is fixed, and that is a precondition rather than a<br/>follow-up.<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:86-89"]
    TXO --> PRE

    NOTLIVE["STATUS — decision_status proposed, implementation_status PARTIAL<br/>on both records: owner decision Q17 (2026-10-02) ordered the crypto items<br/>done under D2's pinned golden fixtures; D1 and D6 landed, D4, D5, D7, D8 not built<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:5-7, ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:133-172,<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:5-7. The crate record says staged<br/>and unreleased, the forum record inactive. see ES-03.11"]
    DRIFT["DRIFT — the port is no longer unreleased: its commits 0befa5b (D1) and e62d028 (D6)<br/>are ancestors of the solid-pod-rs tags v0.5.0-alpha.11, alpha.17, alpha.18 and alpha.21 and of 27e18995, so<br/>VisionClaw's alpha.21 and agentbox's alpha.21 pin both ship it. The record's<br/>activation_status staged and its unreleased wording are stale<br/>ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:7, ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:135"]
    NOTLIVE --> DRIFT
    PORT --> NOTLIVE

    THREE["DIVERGENCE — this is one of THREE unsynced did:nostr-keyed sats<br/>ledgers, and the pod's is the one the chain-view work starts<br/>from. agentbox/docs/proposals/sovereign-settlement.md:61. see ES-03.12"]
    CRYPTO --> THREE
```

**Drift (ADR-2008 release status):** the record says the port landed on `main` unreleased and keeps `activation_status: staged` (`../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:7`, `:135`). But its D1 and D6 commits (`0befa5b`, `e62d028`) are ancestors of the `v0.5.0-alpha.11`, `v0.5.0-alpha.17`, `v0.5.0-alpha.18` and `v0.5.0-alpha.21` tags and of agentbox's Nix revision `27e18995`, so both VisionClaw's alpha.21 dependency (`../project/Cargo.toml:240`) and the agentbox image ship it. The record lives in solid-pod-rs and was not edited.

## ES-08.13 Forge and mirror in the native pod — compiled in, inert until named (ADR-2145)

```mermaid
flowchart TB
    PIN["native pod built from a pinned source revision,<br/>forge-full unless the forge block sets enable false<br/>agentbox/flake.nix:1563, agentbox/lib/solid-pod-rs.nix:61,64"]
    SRV["solid-pod-rs-server under supervisord, port 8484<br/>--public-base-url from SOLID_POD_PUBLIC_URL, --trust-proxy by default<br/>agentbox/flake.nix:1643-1644"]
    FORGE["git forge under /forge, NIP-34 publisher,<br/>forge-held anchoring, started at boot; the kind-30618 mirror is compiled in<br/>but starts only when a follow list is set, so it is NOT running in the shipped pod<br/>agentbox/agentbox.toml:603-608"]
    OFF["defaults: allow_mainnet false, open_edit false, private_repos false,<br/>mirror_follow empty, announce_relays empty<br/>agentbox/agentbox.toml:609-622"]
    JEV["System One consumer solid_pod_advisories = true agentbox/agentbox.toml:931: adds the jev Cargo feature agentbox/flake.nix:1553-1554<br/>and exports SOLID_POD_JEV=1 with SOLID_POD_JEV_ALLOW_CLOUD=1 unless [system_one].local_only names it<br/>agentbox/services/agentbox-manifest/src/system_one.rs:554-558, agentbox/services/agentbox-manifest/src/system_one.rs:594-618;<br/>the server reranks the forge's already-authorised repo and issue listings through the proxy, falling back to the forge's own bytes<br/>solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs:5494-5513, solid-pod-rs/crates/solid-pod-rs-server/src/jev.rs:153-181"]
    ENV["extra relays and follow entries belong in the root .env,<br/>not the tracked manifest<br/>agentbox/agentbox.toml:623-625"]
    PIN --> SRV --> FORGE
    FORGE --> OFF
    SRV -.->|"jev feature, local-only unless ALLOW_CLOUD=1"| JEV
    ENV -.->|"names what the mirror follows"| FORGE
    NOANCHOR["INVARIANT: the forge never broadcasts or spends<br/>agentbox/agentbox.toml:608"]
    FORGE --> NOANCHOR
```

**What it shows:** the native pod of ES-08.1 deployment 2 now carries a git forge that starts with the server and does nothing external until the owner names relays and follow entries. The kind-30618 mirror is compiled in but does not start while the follow list is empty, which is the shipped state: `mirror_config` returns `None` unless at least one relay and one follow entry survive (`../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs:5321-5342`), and the mirror is only built from that `Some` (`:5382-5389`). These lines are read at the solid-pod-rs audit pin; the image builds the same revision 27e18995 (`../project/agentbox/lib/solid-pod-rs.nix:64`), so the line numbers are the image's too.

**Why it is this way:** VisionFlow ADR-2019 decided that shipped pods run the forge and mirror with open edit, mainnet and any follow list opt-in, and agentbox ADR-2145 compiled it in behind the manifest block rather than a separate service.

**Tension (agentbox ADR-2145 vs BASELINE-container):** the ADR records `implementation_status: complete` and `activation_status: live` (`../project/agentbox/docs/adr/ADR-2145-the-solid-pod-forge-and-mirror-are-compiled-in-and-inert-until-named.md:6-7`), while the baseline still says the native pod is "pinned at `v0.5.0-alpha.17`" and "Staged until the image is rebuilt" (`../project/agentbox/docs/BASELINE-container.md:259`, `:264-265`). VisionFlow's own record for the same decision, ADR-2019, agrees with ADR-2145 on activation: `decision_status: accepted`, `implementation_status: partial`, `activation_status: live` (`docs/adr/ADR-2019-forge-and-mirror-on-in-shipped-pods.md:5-7`). The baseline wording is the one left behind.

**Open (ADR-2019 stays partial):** the 2026-10-30 close-out found `FORGE_MIRROR_FOLLOW` empty in the running pod, so no `mirror on kind-30618` line (`../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs:5387`) is logged and the shipped follow list (the operator's own key) is not wired. Closing it is an agentbox change to the pod launcher, and the record's `review_trigger` names it as the event that completes ADR-2019 (`docs/adr/ADR-2019-forge-and-mirror-on-in-shipped-pods.md:13`). The running-pod observation is the close-out's; this audit did not inspect the pod.

**Drift:** agentbox's pod guide says the image pins `solid-pod-rs` 0.5.0-alpha.13 (`../project/agentbox/docs/user/solid-pod.md:37`) and the baseline says alpha.17 (`../project/agentbox/docs/BASELINE-container.md:259`), while the Nix derivation pins the 0.5.0-alpha.21 workspace at revision 27e18995 (`../project/agentbox/lib/solid-pod-rs.nix:61`, `:64`), and VisionClaw's embedded dependency is also `0.5.0-alpha.21` (`../project/Cargo.toml:240`). VisionClaw and the agentbox image now share one release, while two agentbox documents still name two others. The cloud opt-in is configured; the server's own header still describes pod metadata as local-only by default and documents the opt-in that lifts it (`../solid-pod-rs/crates/solid-pod-rs-server/src/jev.rs:24-33`).
