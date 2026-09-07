---
id: SP-03
title: LDP request lifecycle — verbs, containers, content negotiation and PATCH
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2002, ADR-2004, ADR-2005]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/mashlib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/mod.rs
verified_commit: 1d9da5270
---

## SP-03.1 GET — the full read path

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant G as handle_get<br/>solid-pod-rs-server/src/lib.rs:1228
    participant GL as handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:2634
    participant AU as extract_pubkey<br/>solid-pod-rs-server/src/lib.rs:522
    participant W as enforce_read_ctx<br/>solid-pod-rs-server/src/lib.rs:994
    participant S as Storage
    participant CP as set_cache_policy<br/>solid-pod-rs-server/src/lib.rs:1186

    C->>G: GET /pod/notes/hello.ttl
    alt path contains a glob star
        G->>GL: delegate to the glob handler<br/>solid-pod-rs-server/src/lib.rs:1235
    end
    G->>AU: NIP-98 / dev-bearer verification (no body)
    AU-->>G: an optional pubkey
    G->>G: agent_uri — did:nostr prefix or a WebID URL<br/>solid-pod-rs-server/src/lib.rs:579
    G->>W: enforce acl:Read BEFORE any bytes<br/>solid-pod-rs-server/src/lib.rs:1245
    W-->>G: ResponseAudience Public or Private, else 401/403/503
    alt ldp::is_container(path)
        G->>S: container branch — see SP-03.2
    else resource
        G->>S: storage.get(path)
        S-->>G: body plus ResourceMeta
        G->>G: mashlib? then HTML wrapper (SP-03.2)
        G->>G: rdf_content_negotiate? then transcode (SP-03.10)
        G->>G: ETag from meta.etag<br/>solid-pod-rs-server/src/lib.rs:1396
        G->>CP: Cache-Control by audience (ADR-2002)
    end
    G-->>C: 200 with Link, WAC-Allow, Updates-via, ETag
    Note over G: A storage NotFound becomes a bare 404<br/>solid-pod-rs-server/src/lib.rs:1405
```

## SP-03.2 Container GET — three representations

```mermaid
flowchart TD
    START["handle_get, ldp::is_container true<br/>solid-pod-rs-server/src/lib.rs:1249"]
    HTML{"accept_includes_html?<br/>solid-pod-rs-server/src/lib.rs:1261"}
    IDX["try index.html child — serve it verbatim<br/>solid-pod-rs-server/src/lib.rs:1263"]
    REP["storage.container_representation<br/>solid-pod-rs-server/src/lib.rs:1277"]
    MASH{"mashlib::should_serve?<br/>solid-pod-rs-server/src/lib.rs:1286"}
    MHTML["mashlib HTML wrapper plus X-Frame-Options DENY<br/>solid-pod-rs-server/src/lib.rs:1296"]
    JSONLD["application/ld+json listing<br/>solid-pod-rs-server/src/lib.rs:1307"]
    DONE["plus Link, WAC-Allow, Updates-via headers"]

    START --> HTML
    HTML -- "yes, index.html exists" --> IDX
    HTML -- no --> REP
    IDX --> DONE
    REP --> MASH
    MASH -- yes --> MHTML --> DONE
    MASH -- no --> JSONLD --> DONE

    N["container_representation is a blanket-impl trait method over any Storage:<br/>list(path) then render_container<br/>solid-pod-rs/src/ldp.rs:2244"]
    REP -.-> N
    N2["should_serve keys on Accept plus Sec-Fetch-Dest plus the stored content type<br/>solid-pod-rs/src/mashlib.rs:122"]
    MASH -.-> N2
```

## SP-03.3 PUT — resource and container creation

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_put<br/>solid-pod-rs-server/src/lib.rs:1419
    participant AU as extract_pubkey_with_body<br/>solid-pod-rs-server/src/lib.rs:529
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:819
    participant Q as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:2369
    participant S as Storage
    participant GM as git_mark_write<br/>solid-pod-rs-server/src/lib.rs:3462

    C->>P: PUT /pod/notes/hello.ttl
    alt container path with a trailing slash
        P->>P: has_basic_container_link?<br/>solid-pod-rs-server/src/lib.rs:1410
        alt Link rel type ldp BasicContainer present
            P->>W: acl:Write on the container
            P->>S: create_container
            P-->>C: 201 with ETag and Link
        else
            P-->>C: 405 cannot PUT to a container<br/>solid-pod-rs-server/src/lib.rs:1450
        end
    end
    P->>AU: NIP-98 bound to method, URL and raw body hash
    P->>W: acl:Write — sidecar paths elevate to Control, see SP-04.5
    opt path is an acl or meta sidecar
        P->>P: proposed_acl_keeps_caller_control<br/>solid-pod-rs-server/src/lib.rs:1475
        P-->>C: 409 Conflict if the proposed ACL drops caller Control
    end
    P->>Q: atomic quota reservation
    P->>S: storage.put(path, body, content_type)
    P->>Q: finish_quota_reservation with the write outcome<br/>solid-pod-rs-server/src/lib.rs:1488
    P->>GM: additive provenance mark AFTER the write<br/>solid-pod-rs-server/src/lib.rs:1493
    GM-->>P: ProvenanceReceipt
    P-->>C: 201 with ETag, X-Provenance and Link
```

## SP-03.4 POST — Append, Slug resolution and unique-name minting

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_post<br/>solid-pod-rs-server/src/lib.rs:1532
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:819
    participant SL as ldp::resolve_slug<br/>solid-pod-rs/src/ldp.rs:155
    participant MU as mint_unique_target<br/>solid-pod-rs-server/src/lib.rs:1509
    participant S as Storage

    C->>P: POST to a container with a Slug header
    P->>W: acl:Append on the CONTAINER<br/>solid-pod-rs-server/src/lib.rs:1545
    P->>SL: join the Slug, capped at MAX_SLUG_BYTES<br/>solid-pod-rs/src/ldp.rs:144
    SL-->>P: target path
    alt target resolves to an acl or meta sidecar
        P->>W: RE-ENFORCE Write on the target, which elevates to Control<br/>solid-pod-rs-server/src/lib.rs:1573
        P->>P: proposed_acl_keeps_caller_control, else 409
        Note over P: P0-4 — minting a sidecar from Append-only rights would be<br/>privilege escalation on a sibling resource.
    else ordinary resource
        P->>MU: probe existence, append a numeric suffix, bounded<br/>solid-pod-rs-server/src/lib.rs:1516
        MU-->>P: a free target, hash-suffixed at the ceiling
    end
    P->>S: put(target, body, content_type)
    P-->>C: 201 with Location, ETag and X-Provenance
    Note over P: INVARIANT: an LDP POST creates, never overwrites — resolve_slug alone<br/>would clobber a repeated Slug and lose data silently.
```

## SP-03.5 PATCH — non-destructive, three dialects

```mermaid
flowchart TD
    IN["handle_patch<br/>solid-pod-rs-server/src/lib.rs:1619"]
    CONT{"is_container?"}
    C405["405 cannot PATCH a container<br/>solid-pod-rs-server/src/lib.rs:1626"]
    AUTH["enforce_write_ctx with AccessMode Write, never Append<br/>solid-pod-rs-server/src/lib.rs:1638"]
    DIAL{"ldp::patch_dialect_from_mime<br/>solid-pod-rs/src/ldp.rs:2162"}
    U415["415 unsupported patch dialect<br/>solid-pod-rs-server/src/lib.rs:1652"]
    UTF{"body is UTF-8?"}
    B400["400 patch body is not valid UTF-8<br/>solid-pod-rs-server/src/lib.rs:1658"]
    EX{"resource exists?"}
    SEED["seed_graph_from_patch_target<br/>solid-pod-rs-server/src/lib.rs:1896"]
    N3["apply_n3_patch<br/>solid-pod-rs/src/ldp.rs:1125"]
    SPQ["apply_sparql_patch<br/>solid-pod-rs/src/ldp.rs:1252"]
    JP["apply_json_patch, RFC 6902<br/>solid-pod-rs/src/ldp.rs:1966"]
    ABS["apply_patch_to_absent then create<br/>solid-pod-rs/src/ldp.rs:2212"]
    SER["graph_to_turtle delegates to to_ntriples<br/>solid-pod-rs-server/src/lib.rs:1795"]
    GUARD["F7 lockout guard on the post-patch ACL<br/>solid-pod-rs-server/src/lib.rs:1718"]
    OUT["204 No Content, or 201 on create, plus X-Provenance"]

    IN --> CONT
    CONT -- yes --> C405
    CONT -- no --> AUTH --> DIAL
    DIAL -- none --> U415
    DIAL -- some --> UTF
    UTF -- no --> B400
    UTF -- yes --> EX
    EX -- yes --> SEED
    SEED --> N3
    SEED --> SPQ
    EX -- "yes, json-patch" --> JP
    EX -- no --> ABS
    N3 --> SER
    SPQ --> SER
    ABS --> SER
    SER --> GUARD --> OUT
    JP --> OUT

    NN["INVARIANT: PATCH requires Write, not Append — an N3 patch carrying<br/>solid:deletes can destroy data, so Append-only principals are excluded.<br/>solid-pod-rs-server/src/lib.rs:1630"]
    AUTH -.-> NN
```

## SP-03.6 The non-destructive-write invariant

```mermaid
sequenceDiagram
    autonumber
    participant P as handle_patch<br/>solid-pod-rs-server/src/lib.rs:1619
    participant S as Storage
    participant SG as seed_graph_from_patch_target<br/>solid-pod-rs-server/src/lib.rs:1896
    participant AP as ldp::apply_n3_patch<br/>solid-pod-rs/src/ldp.rs:1125

    P->>S: get(path)
    S-->>P: the current stored body
    P->>SG: parse the stored body as N-Triples
    alt parses
        SG-->>P: seed Graph carrying the existing triples
        P->>AP: apply the patch ON TOP of the seed<br/>solid-pod-rs-server/src/lib.rs:1677
    else unparseable
        SG-->>P: error — fail closed, refuse the write<br/>solid-pod-rs-server/src/lib.rs:1673
    end
    P->>S: put the serialised N-Triples back
    Note over SG: Before this fix the working graph started EMPTY, so every incremental<br/>PATCH silently discarded the resource's existing triples. Refusing an<br/>unparseable body is deliberate — fail closed rather than destroy.
```

## SP-03.7 DELETE and OPTIONS

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant D as handle_delete<br/>solid-pod-rs-server/src/lib.rs:1990
    participant O as handle_options<br/>solid-pod-rs-server/src/lib.rs:2017
    participant OF as ldp::options_for<br/>solid-pod-rs/src/ldp.rs:1776
    participant S as Storage

    C->>D: DELETE a resource
    D->>D: extract_pubkey with no body, then enforce_write_ctx Write<br/>solid-pod-rs-server/src/lib.rs:1997
    D->>S: reserve a zero-size quota slot, then storage.delete
    alt deleted
        D-->>C: 204 No Content
    else NotFound
        D-->>C: 404<br/>solid-pod-rs-server/src/lib.rs:2012
    end

    C->>O: OPTIONS on a container
    O->>OF: allow, accept-post, accept-patch, accept-ranges
    OF-->>O: OptionsResponse<br/>solid-pod-rs/src/ldp.rs:1765
    O-->>C: 204 with Allow, Accept-Post, Accept-Patch, Accept-Ranges, Updates-via
    Note over O: OPTIONS runs NO WAC check — it advertises capability, not content.<br/>Accept-Patch is the fixed dialect list at solid-pod-rs/src/ldp.rs:1774.
```

## SP-03.8 COPY — the non-standard verb

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as handle_copy<br/>solid-pod-rs-server/src/lib.rs:2571
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:819
    participant S as Storage

    C->>H: COPY to a destination with a Source header
    H->>W: acl:Write on the DESTINATION<br/>solid-pod-rs-server/src/lib.rs:2578
    alt no Source header
        H-->>C: 400 Source header required<br/>solid-pod-rs-server/src/lib.rs:2594
    end
    H->>S: get(source)
    alt source missing
        H-->>C: 404 source resource not found<br/>solid-pod-rs-server/src/lib.rs:2600
    end
    H->>S: put(dest, body, source content type)<br/>solid-pod-rs-server/src/lib.rs:2606
    opt source has an acl sidecar
        H->>S: copy the sidecar alongside<br/>solid-pod-rs-server/src/lib.rs:2613
    end
    H-->>C: 201 with Location
    Note over H: DIVERGENCE: COPY authorises Write on the destination only — it never checks<br/>acl:Read on the SOURCE, so a principal with write rights on the destination<br/>can copy a resource it could not have read directly.
```

## SP-03.9 Glob GET

```mermaid
flowchart TD
    IN["GET whose path contains a star, routed to handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:2634"]
    PAT{"path ends with slash-star?<br/>solid-pod-rs-server/src/lib.rs:2640"}
    NF["404 unsupported glob pattern"]
    GATE["enforce_read_ctx on the FOLDER<br/>solid-pod-rs-server/src/lib.rs:2655"]
    LIST["storage.list(folder)<br/>solid-pod-rs-server/src/lib.rs:2657"]
    FILTER["keep only turtle, n-triples and n3 children<br/>solid-pod-rs-server/src/lib.rs:2666"]
    MERGE["concatenate bodies into one text/turtle response<br/>solid-pod-rs-server/src/lib.rs:2682"]
    EMPTY["404 no matching RDF resources<br/>solid-pod-rs-server/src/lib.rs:2679"]

    IN --> PAT
    PAT -- no --> NF
    PAT -- yes --> GATE --> LIST --> FILTER --> MERGE
    FILTER --> EMPTY

    N["INVARIANT (P0-1): the glob merge is gated on acl:Read of the folder, so a<br/>glob under a private container cannot bypass the plain container read check."]
    GATE -.-> N
    N2["DIVERGENCE: the merge gates on the FOLDER only. A child carrying its own,<br/>stricter acl sidecar is still included in the merged body."]
    FILTER -.-> N2
```

## SP-03.10 RDF content negotiation

```mermaid
flowchart LR
    ACC["Accept header"]
    BEST["best_explicit_rdf_format — highest-q EXPLICIT format<br/>solid-pod-rs-server/src/lib.rs:1805"]
    NEG["rdf_content_negotiate<br/>solid-pod-rs-server/src/lib.rs:1853"]
    FMT["ldp::RdfFormat::from_mime<br/>solid-pod-rs/src/ldp.rs:286"]
    NT["Graph::parse_ntriples<br/>solid-pod-rs/src/ldp.rs:793"]
    OUT1["to_ntriples<br/>solid-pod-rs/src/ldp.rs:694"]
    OUT2["to_jsonld<br/>solid-pod-rs/src/ldp.rs:714"]
    VERB["fall through to the stored bytes verbatim<br/>solid-pod-rs-server/src/lib.rs:1385"]

    ACC --> BEST --> NEG --> FMT
    NEG --> NT
    NT --> OUT1
    NT --> OUT2
    NEG -. "non-RDF, unparseable, or a wildcard Accept" .-> VERB

    N["A transcoded response carries Vary: Accept<br/>solid-pod-rs-server/src/lib.rs:1373"]
    OUT1 -.-> N
    N2["RDF resources persist as N-Triples, see graph_to_turtle in SP-03.5, so the<br/>same graph is servable as Turtle, N-Triples or JSON-LD on demand."]
    NT -.-> N2
```

## SP-03.11 Response headers set on every LDP response

```mermaid
classDiagram
    class ResponseHeaders {
        +set_link_headers  solid-pod-rs-server/src/lib.rs:1126
        +set_wac_allow  solid-pod-rs-server/src/lib.rs:1134
        +set_updates_via  solid-pod-rs-server/src/lib.rs:1141
        +set_provenance_headers  solid-pod-rs-server/src/lib.rs:1162
        +set_cache_policy  solid-pod-rs-server/src/lib.rs:1186
        +append_vary  solid-pod-rs-server/src/lib.rs:1205
    }
    class LinkHeaders {
        +ldp::link_headers(path)  solid-pod-rs/src/ldp.rs:119
    }
    class WacAllow {
        +wac::wac_allow_header  solid-pod-rs/src/wac/mod.rs:265
    }
    ResponseHeaders ..> LinkHeaders
    ResponseHeaders ..> WacAllow
    note for ResponseHeaders "set_updates_via rewrites https to wss and http to ws, then appends the\nnotifications path — solid-pod-rs-server/src/lib.rs:1145."
```

## SP-03.12 ADR-2002 cache policy — audience beats media type

```mermaid
stateDiagram-v2
    [*] --> Classify
    Classify --> Private: an anonymous caller would NOT have been granted
    Classify --> Public: an anonymous caller WOULD have been granted<br/>and no sidecar elevation happened

    Private --> VaryAuth: append Vary Authorization<br/>solid-pod-rs-server/src/lib.rs:1193
    VaryAuth --> NoStore: Cache-Control private no-store<br/>solid-pod-rs/src/ldp.rs:1915
    Public --> MediaType: cache_control_for(content_type)<br/>solid-pod-rs/src/ldp.rs:1901
    MediaType --> RdfPolicy: RDF gets private no-cache must-revalidate<br/>solid-pod-rs/src/ldp.rs:1861
    MediaType --> NoHeader: a public binary gets no Cache-Control

    NoStore --> [*]
    RdfPolicy --> [*]
    NoHeader --> [*]

    note right of Classify
      cache_control_for_response(content_type, audience)
      solid-pod-rs/src/ldp.rs:1947. Audience is decided by re-evaluating
      the already-resolved ACL with no principal — pure, no extra I/O.
      A sidecar read is never classified Public.
    end note
    note right of NoStore
      INVARIANT: a private response is never advertised as publicly
      cacheable. set_cache_policy never overwrites a Cache-Control a
      handler already set (solid-pod-rs-server/src/lib.rs:1198), so the
      mashlib wrapper's own no-store survives.
    end note
```

## SP-03.13 Error translation — PodError to HTTP status

```mermaid
flowchart LR
    TA["to_actix<br/>solid-pod-rs-server/src/lib.rs:499"]
    NF["NotFound to 404<br/>solid-pod-rs-server/src/lib.rs:501"]
    BR["BadRequest to 400<br/>solid-pod-rs-server/src/lib.rs:502"]
    UN["Unsupported to 415<br/>solid-pod-rs-server/src/lib.rs:503"]
    FB["Forbidden to 403<br/>solid-pod-rs-server/src/lib.rs:504"]
    UA["Unauthenticated to 401<br/>solid-pod-rs-server/src/lib.rs:505"]
    PF["PreconditionFailed to 412<br/>solid-pod-rs-server/src/lib.rs:506"]
    ISE["anything else to 500<br/>solid-pod-rs-server/src/lib.rs:507"]
    PP["patch_parse_err collapses Unsupported and BadRequest to 400<br/>solid-pod-rs-server/src/lib.rs:1783"]

    TA --> NF
    TA --> BR
    TA --> UN
    TA --> FB
    TA --> UA
    TA --> PF
    TA --> ISE
    TA -.-> PP

    N["patch_parse_err separates garbage in a supported dialect (400) from an<br/>unsupported dialect (415), which the dispatcher decides before parsing."]
    PP -.-> N
```
