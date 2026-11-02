---
id: SP-03
title: LDP request lifecycle — verbs, containers, content negotiation and PATCH
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2002, ADR-2004, ADR-2005]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/mashlib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/resolver.rs
verified_commit: 9b6143ce55b242c322c92d8c6bcc0c1a5371b451
---

## SP-03.1 GET — the full read path

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant G as handle_get<br/>solid-pod-rs-server/src/lib.rs:1422
    participant GL as handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:3035
    participant AU as extract_pubkey<br/>solid-pod-rs-server/src/lib.rs:596
    participant W as enforce_read_ctx<br/>solid-pod-rs-server/src/lib.rs:1164
    participant S as Storage
    participant CP as set_cache_policy<br/>solid-pod-rs-server/src/lib.rs:1380

    C->>G: GET /pod/notes/hello.ttl
    opt feature activitypub and path is /profile/card.jsonld
        G-->>C: return public actor document when Accept negotiates ActivityJSON<br/>solid-pod-rs-server/src/lib.rs:1433
    end
    alt path contains a glob star
        G->>GL: return glob handler result<br/>solid-pod-rs-server/src/lib.rs:1451
    end
    G->>AU: NIP-98 / dev-bearer verification (no body)
    AU-->>G: an optional pubkey
    G->>G: agent_uri — did:nostr prefix or a WebID URL<br/>solid-pod-rs-server/src/lib.rs:657
    G->>W: enforce acl:Read BEFORE any bytes<br/>solid-pod-rs-server/src/lib.rs:1461
    W-->>G: ResponseAudience Public or Private, else 401/403/503
    alt ldp::is_container(path)
        G->>S: container branch — see SP-03.2
    else resource
        G->>S: storage.get(path)
        S-->>G: body plus ResourceMeta
        G->>G: mashlib? then HTML wrapper (SP-03.2)
        G->>G: rdf_content_negotiate? then transcode (SP-03.10)
        G->>G: ETag from meta.etag<br/>solid-pod-rs-server/src/lib.rs:1620
        G->>CP: Cache-Control by audience (ADR-2002)
    end
    G-->>C: successful branch headers, ETag on verbatim or transcoded bodies
    Note over G: DEBT WAC-Allow is computed with no ACL document at lib.rs:1463,<br/>so successful reads advertise empty user and public modes.
    Note over G: A storage NotFound becomes a bare 404<br/>solid-pod-rs-server/src/lib.rs:1629
    Note over G: OPTIONS (handle_options, solid-pod-rs-server/src/lib.rs:2304, via<br/>ldp::options_for_conneg, solid-pod-rs/src/ldp.rs:1806) runs NO WAC check — it<br/>advertises capability, not content. Its Accept-Patch dialect list is fixed<br/>at solid-pod-rs/src/ldp.rs:1798, and the container Accept-Post adds text/n3 only when<br/>conneg is on (accept_post_for, solid-pod-rs/src/ldp.rs:102).
```

## SP-03.2 Container GET — three representations

```mermaid
flowchart TD
    START["handle_get, ldp::is_container true<br/>solid-pod-rs-server/src/lib.rs:1465"]
    HTML{"accept_includes_html?<br/>solid-pod-rs-server/src/lib.rs:1477"}
    IDX["try index.html child — require its own Read grant,<br/>serve HTML with optional live-reload injection, else fall through<br/>solid-pod-rs-server/src/lib.rs:1478"]
    REP["storage.container_representation<br/>solid-pod-rs-server/src/lib.rs:1501"]
    MASH{"mashlib::should_serve?<br/>solid-pod-rs-server/src/lib.rs:1510"}
    MHTML["mashlib HTML wrapper, no-store, X-Frame-Options DENY<br/>solid-pod-rs-server/src/lib.rs:1520"]
    JSONLD["application/ld+json listing<br/>solid-pod-rs-server/src/lib.rs:1531"]
    DONE["Cache policy uses child audience for index.html,<br/>container audience for the listing or wrapper.<br/>Plus Link, WAC-Allow, Updates-via.<br/>solid-pod-rs-server/src/lib.rs:1492"]

    START --> HTML
    HTML -- yes --> IDX
    HTML -- no --> REP
    IDX -- "allowed child, read succeeds" --> DONE
    IDX -- "denied or read fails" --> REP
    REP --> MASH
    MASH -- yes --> MHTML --> DONE
    MASH -- no --> JSONLD --> DONE

    N["container_representation is a blanket-impl trait method over any Storage:<br/>list(path) then render_container<br/>solid-pod-rs/src/ldp.rs:2278"]
    REP -.-> N
    N2["should_serve keys on Accept plus Sec-Fetch-Dest plus the stored content type<br/>solid-pod-rs/src/mashlib.rs:122"]
    MASH -.-> N2
```

## SP-03.3 PUT — resource and container creation

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_put<br/>solid-pod-rs-server/src/lib.rs:1677
    participant AU as extract_pubkey_with_body<br/>solid-pod-rs-server/src/lib.rs:603
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:953
    participant Q as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:2653
    participant S as Storage
    participant GM as git_mark_write<br/>solid-pod-rs-server/src/lib.rs:3917

    C->>P: PUT /pod/notes/hello.ttl
    alt container path with a trailing slash
        P->>P: has_basic_container_link?<br/>solid-pod-rs-server/src/lib.rs:1634
        alt Link rel type ldp BasicContainer present
            P->>W: acl:Write on the container
            P->>S: create_container
            P-->>C: 201 with ETag and Link
        else
            P-->>C: 405 cannot PUT to a container<br/>solid-pod-rs-server/src/lib.rs:1708
        end
    end
    P->>AU: NIP-98 bound to method, URL and raw body hash
    P->>W: acl:Write — sidecar paths elevate to Control, see SP-04.5
    opt path is an acl or meta sidecar
        P->>P: proposed_acl_keeps_caller_control<br/>solid-pod-rs-server/src/lib.rs:1732
        P-->>C: 409 unless the candidate ACL parses and grants the caller<br/>effective Control on the protected resource (SP-04.6)
        Note over P: A meta sidecar still needs Control to write, but the guard<br/>passes it without parsing it as an ACL<br/>solid-pod-rs-server/src/lib.rs:883
    end
    P->>P: validate_rdf_body — Turtle syntax must parse (400), text/n3 needs conneg (415)<br/>solid-pod-rs-server/src/lib.rs:1742
    Note over P: DEBT valid Turtle with incomplete condition shape can pass<br/>solid-pod-rs-server/src/lib.rs:1668
    P->>Q: atomic quota reservation
    P->>S: storage.put(path, body, content_type)
    P->>Q: finish_quota_reservation with the write outcome<br/>solid-pod-rs-server/src/lib.rs:1749
    P->>GM: additive provenance mark AFTER the write<br/>solid-pod-rs-server/src/lib.rs:1754
    GM-->>P: ProvenanceReceipt
    P-->>C: 201 with ETag, X-Provenance and Link
```

## SP-03.4 POST — Append, Slug resolution and unique-name minting

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_post<br/>solid-pod-rs-server/src/lib.rs:1793
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:953
    participant SL as ldp::resolve_slug<br/>solid-pod-rs/src/ldp.rs:179
    participant MU as mint_unique_target<br/>solid-pod-rs-server/src/lib.rs:1770
    participant S as Storage

    C->>P: POST to a container with a Slug header
    P->>W: acl:Append on the CONTAINER<br/>solid-pod-rs-server/src/lib.rs:1806
    P->>SL: validate then join the Slug: at most MAX_SLUG_BYTES (solid-pod-rs/src/ldp.rs:167), no slash, dot-dot, NUL or bare dot, else 400<br/>solid-pod-rs/src/ldp.rs:194
    SL-->>P: target path
    Note over SL: characters limited to A-Z a-z 0-9 dot underscore hyphen<br/>solid-pod-rs/src/ldp.rs:197<br/>no Slug gives a UUID name, solid-pod-rs/src/ldp.rs:207
    alt target resolves to an acl or meta sidecar
        P->>W: RE-ENFORCE Write on the target, which elevates to Control<br/>solid-pod-rs-server/src/lib.rs:1835
        P->>P: proposed_acl_keeps_caller_control, else 409 — acl only,<br/>a meta body passes the guard<br/>solid-pod-rs-server/src/lib.rs:1843
        Note over P: P0-4 — minting a sidecar from Append-only rights would be<br/>privilege escalation on a sibling resource. This branch does not<br/>mint a unique name, so a Control holder's POST replaces an existing sidecar.
    else ordinary resource
        P->>MU: probe existence, append a numeric suffix, bounded<br/>solid-pod-rs-server/src/lib.rs:1777
        MU-->>P: a free target, hash-suffixed at the ceiling<br/>solid-pod-rs-server/src/lib.rs:1790
    end
    P->>P: validate_rdf_body on the target, as for PUT<br/>solid-pod-rs-server/src/lib.rs:1860
    P->>S: put(target, body, content_type)
    P-->>C: 201 with Location, ETag, X-Provenance and Link<br/>solid-pod-rs-server/src/lib.rs:1872
    Note over P: INVARIANT: an LDP POST to an ordinary name creates, never overwrites —<br/>resolve_slug alone would clobber a repeated Slug and lose data silently.
```

**Tension:** `mint_unique_target` probes with `exists` and the later `put` is a plain write (`solid-pod-rs-server/src/lib.rs:1771`, `solid-pod-rs-server/src/lib.rs:1865`), so the no-overwrite rule is a check-then-act. Two concurrent POSTs with the same Slug can both choose the same free name. The hash fallback is a deterministic hash of the target, so it is not guaranteed fresh either (`solid-pod-rs-server/src/lib.rs:1788`). Only a concurrent run could confirm the race; the source does not prevent it.

## SP-03.5 PATCH — non-destructive, three dialects

```mermaid
flowchart TD
    IN["handle_patch<br/>solid-pod-rs-server/src/lib.rs:1884"]
    CONT{"is_container?"}
    C405["405 cannot PATCH a container<br/>solid-pod-rs-server/src/lib.rs:1891"]
    AUTH["enforce_write_ctx with AccessMode Write, never Append<br/>solid-pod-rs-server/src/lib.rs:1900"]
    DIAL{"ldp::patch_dialect_from_mime<br/>solid-pod-rs/src/ldp.rs:2196"}
    U415["415 unsupported patch dialect<br/>solid-pod-rs-server/src/lib.rs:1917"]
    UTF{"body is UTF-8?"}
    B400["400 patch body is not valid UTF-8<br/>solid-pod-rs-server/src/lib.rs:1923"]
    EX{"resource exists?"}
    SEED["seed_graph_from_patch_target<br/>solid-pod-rs-server/src/lib.rs:2183"]
    N3["apply_n3_patch<br/>solid-pod-rs/src/ldp.rs:1149"]
    SPQ["apply_sparql_patch<br/>solid-pod-rs/src/ldp.rs:1276"]
    JP["apply_json_patch, RFC 6902, on the stored JSON<br/>(a non-JSON body is replaced by an empty object)<br/>solid-pod-rs/src/ldp.rs:2000"]
    JGUARD["lockout guard on the patched JSON before storing<br/>solid-pod-rs-server/src/lib.rs:1961"]
    ABS["apply_patch_to_absent then create — N3 or SPARQL only,<br/>JSON Patch on an absent resource is 400<br/>solid-pod-rs/src/ldp.rs:2246"]
    SER["graph_to_turtle delegates to to_ntriples<br/>solid-pod-rs-server/src/lib.rs:2082"]
    GUARD["F7 lockout guard on the serialised post-patch ACL<br/>solid-pod-rs-server/src/lib.rs:1997"]
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
    JP --> JGUARD --> OUT

    NN["INVARIANT: PATCH requires Write, not Append — an N3 patch carrying<br/>solid:deletes can destroy data, so Append-only principals are excluded.<br/>solid-pod-rs-server/src/lib.rs:1895"]
    AUTH -.-> NN
```

**Debt:** an N3 or SPARQL PATCH cannot change or create an `.acl` at this revision. A stored Turtle ACL with `@prefix` lines does not parse as N-Triples, so the seed refuses it with 409 (`solid-pod-rs-server/src/lib.rs:2192`). Creating one through PATCH serialises N-Triples with full IRIs and passes them to the guard as `application/n-triples` (`solid-pod-rs-server/src/lib.rs:2035-2044`). The shared policy reader recognises Turtle only by a Turtle media type, `@prefix` or `acl:Authorization` (`solid-pod-rs/src/wac/resolver.rs:309-316`), so the guard classifies that body as Malformed and refuses it with 409. PUT, POST to a sidecar Slug, COPY onto a sidecar and JSON Patch still write ACLs; RDF PATCH cannot. This is source tracing; no route test exercises RDF PATCH on an ACL.

## SP-03.6 The non-destructive-write invariant

```mermaid
sequenceDiagram
    autonumber
    participant P as handle_patch<br/>solid-pod-rs-server/src/lib.rs:1884
    participant S as Storage
    participant SG as seed_graph_from_patch_target<br/>solid-pod-rs-server/src/lib.rs:2183
    participant AP as ldp::apply_n3_patch<br/>solid-pod-rs/src/ldp.rs:1149

    P->>S: get(path)
    S-->>P: the current stored body
    P->>SG: parse the stored body as N-Triples
    alt parses
        SG-->>P: seed Graph carrying the existing triples
        P->>AP: apply the patch ON TOP of the seed<br/>solid-pod-rs-server/src/lib.rs:1942
    else unparseable
        SG-->>P: error — fail closed, refuse the write<br/>solid-pod-rs-server/src/lib.rs:1938
    end
    P->>S: put the serialised N-Triples back, stored as text/turtle<br/>solid-pod-rs-server/src/lib.rs:2016
    Note over SG: Before this fix the working graph started EMPTY, so every incremental<br/>PATCH silently discarded the resource's existing triples. Refusing an<br/>unparseable body (409) is deliberate — fail closed rather than destroy.<br/>solid-pod-rs-server/src/lib.rs:2192
```

**Debt:** the invariant covers the N3 and SPARQL dialects only. The JSON Patch branch treats a stored body that is not JSON as `{}` and writes the patched object back under the stored content type (`solid-pod-rs-server/src/lib.rs:1949-1952`, `solid-pod-rs-server/src/lib.rs:1978`). So a JSON Patch that applies cleanly to an empty object, sent to a Turtle resource, replaces its triples where an N3 patch would be refused. This is source tracing and was not reproduced over HTTP.

## SP-03.8 COPY — the non-standard verb

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as handle_copy<br/>solid-pod-rs-server/src/lib.rs:2916
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:953
    participant S as Storage

    C->>H: COPY to a destination with a Source header
    H->>W: acl:Write on the DESTINATION<br/>solid-pod-rs-server/src/lib.rs:2923
    alt no Source header
        H-->>C: 400 Source header required<br/>solid-pod-rs-server/src/lib.rs:2939
    end
    H->>H: normalise_copy_source, dotfile allowlist, else 400 or 403<br/>solid-pod-rs-server/src/lib.rs:2942
    H->>W: acl:Read on the SOURCE, as for a plain GET<br/>solid-pod-rs-server/src/lib.rs:2951
    H->>S: get(source)
    alt source missing
        H-->>C: 404 source resource not found<br/>solid-pod-rs-server/src/lib.rs:2956
    end
    opt destination is an acl or meta sidecar
        H->>H: proposed_acl_keeps_caller_control BEFORE the write, else 409 —<br/>for an acl it also checks Turtle syntax, as COPY has no body validation<br/>solid-pod-rs-server/src/lib.rs:2962
    end
    H->>S: put(dest, body, source content type)<br/>solid-pod-rs-server/src/lib.rs:2980
    opt source has an acl sidecar
        H->>S: copy it only if the caller can read the source acl, write the destination acl<br/>and keeps Control, otherwise skip it silently<br/>solid-pod-rs-server/src/lib.rs:2990
    end
    H-->>C: 201 with Location
    Note over H: INVARIANT: COPY needs Write on the destination AND the same Read as a plain<br/>GET on the source, so write rights alone cannot copy an unreadable resource.<br/>solid-pod-rs-server/src/lib.rs:2951
```

## SP-03.9 Glob GET

```mermaid
flowchart TD
    IN["GET whose path contains a star, routed to handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:3035"]
    PAT{"path ends with slash-star?<br/>solid-pod-rs-server/src/lib.rs:3041"}
    NF["404 unsupported glob pattern"]
    GATE["enforce_read_ctx on the FOLDER<br/>solid-pod-rs-server/src/lib.rs:3057"]
    LIST["storage.list(folder)<br/>solid-pod-rs-server/src/lib.rs:3059"]
    SKIP["skip sub-containers, acl, meta and prov.ttl children<br/>solid-pod-rs-server/src/lib.rs:3063"]
    CHILD["enforce_read_inner per child — a refused or pay-per-read child is left out<br/>solid-pod-rs-server/src/lib.rs:3074"]
    FILTER["keep only turtle, n-triples and n3 children<br/>solid-pod-rs-server/src/lib.rs:3080"]
    MERGE["concatenate bodies into one text/turtle response<br/>solid-pod-rs-server/src/lib.rs:3085"]
    EMPTY["404 no matching RDF resources<br/>solid-pod-rs-server/src/lib.rs:3093"]

    IN --> PAT
    PAT -- no --> NF
    PAT -- yes --> GATE --> LIST --> SKIP --> CHILD --> FILTER --> MERGE
    FILTER --> EMPTY

    N["INVARIANT (P0-1): the glob merge is gated on acl:Read of the folder, so a<br/>glob under a private container cannot bypass the plain container read check."]
    GATE -.-> N
    N2["INVARIANT: each merged child is also gated on its own acl:Read, so a child with a<br/>stricter acl sidecar is left out of the body. Any Private child makes the audience Private.<br/>solid-pod-rs-server/src/lib.rs:3074"]
    CHILD -.-> N2
```

## SP-03.10 RDF content negotiation

```mermaid
flowchart LR
    ACC["Accept header"]
    BEST["best_explicit_rdf_format — highest-q EXPLICIT format<br/>solid-pod-rs-server/src/lib.rs:2092"]
    NEG["rdf_content_negotiate<br/>solid-pod-rs-server/src/lib.rs:2140"]
    FMT["ldp::RdfFormat::from_mime<br/>solid-pod-rs/src/ldp.rs:310"]
    NT["Graph::parse_ntriples<br/>solid-pod-rs/src/ldp.rs:817"]
    OUT1["to_ntriples<br/>solid-pod-rs/src/ldp.rs:718"]
    OUT2["to_jsonld<br/>solid-pod-rs/src/ldp.rs:738"]
    VERB["fall through to the stored bytes verbatim<br/>solid-pod-rs-server/src/lib.rs:1611"]

    ACC --> BEST --> NEG --> FMT
    NEG --> NT
    NT --> OUT1
    NT --> OUT2
    NEG -. "non-RDF, unparseable, or a wildcard Accept" .-> VERB

    N["A transcoded response carries Vary: Accept<br/>solid-pod-rs-server/src/lib.rs:1597"]
    OUT1 -.-> N
    N2["Only PATCH-written resources persist as N-Triples (graph_to_turtle, SP-03.5).<br/>PUT and POST store the client's bytes; a Turtle body with prefixes does not<br/>parse as N-Triples, so it is served verbatim whatever the Accept.<br/>solid-pod-rs-server/src/lib.rs:2154"]
    NT -.-> N2
    N3N["Also verbatim: empty Accept, same format as stored, or an RDF/XML target<br/>solid-pod-rs-server/src/lib.rs:2145-2151<br/>solid-pod-rs-server/src/lib.rs:2171"]
    NEG -.-> N3N
```

## SP-03.11 Response header helpers for LDP responses

```mermaid
classDiagram
    class ResponseHeaders {
        +set_link_headers  solid-pod-rs-server/src/lib.rs:1320
        +set_wac_allow  solid-pod-rs-server/src/lib.rs:1328
        +set_updates_via  solid-pod-rs-server/src/lib.rs:1335
        +set_provenance_headers  solid-pod-rs-server/src/lib.rs:1356
        +set_cache_policy  solid-pod-rs-server/src/lib.rs:1380
        +append_vary  solid-pod-rs-server/src/lib.rs:1399
    }
    class LinkHeaders {
        +ldp::link_headers(path)  solid-pod-rs/src/ldp.rs:142
    }
    class WacAllow {
        +wac::wac_allow_header  solid-pod-rs/src/wac/mod.rs:265
    }
    ResponseHeaders ..> LinkHeaders
    ResponseHeaders ..> WacAllow
    note for ResponseHeaders "set_updates_via rewrites https to wss and http to ws, then appends the\nnotifications path — solid-pod-rs-server/src/lib.rs:1339."
    note for LinkHeaders "rel=acl now names the resolver's own sidecar key, so a container\nadvertises /folder.acl — solid-pod-rs/src/ldp.rs:152."
```

**Drift:** this panel's heading used to say "every LDP response", but each handler picks its own subset. Successful GET branches set Link, WAC-Allow, Updates-Via and the cache policy (`solid-pod-rs-server/src/lib.rs:1623-1626`). PUT and POST set Link and X-Provenance (`solid-pod-rs-server/src/lib.rs:1759-1760`). PATCH sets only X-Provenance (`solid-pod-rs-server/src/lib.rs:2022`). COPY sets only Location (`solid-pod-rs-server/src/lib.rs:3024-3027`). Glob GET sets only the cache policy (`solid-pod-rs-server/src/lib.rs:3097`). A GET that misses storage returns a bare 404 (`solid-pod-rs-server/src/lib.rs:1629`). The panel lists the helpers, not a guarantee that every response carries them.

## SP-03.12 ADR-2002 cache policy — audience beats media type

```mermaid
flowchart TD
    A["Audience from enforce_read_inner<br/>solid-pod-rs-server/src/lib.rs:1276"] --> PRIVATE{"Private?"}
    PRIVATE -- yes --> V["append Vary Authorization<br/>solid-pod-rs-server/src/lib.rs:1385"]
    PRIVATE -- no --> EXIST
    V --> EXIST{"Cache-Control already set?<br/>solid-pod-rs-server/src/lib.rs:1387"}
    EXIST -- yes --> KEEP["Keep existing policy, e.g. mashlib no-store<br/>solid-pod-rs-server/src/lib.rs:1571"]
    EXIST -- no --> POLICY["cache_control_for_response<br/>solid-pod-rs/src/ldp.rs:1981"]
    POLICY -- Private --> NS["private, no-store<br/>solid-pod-rs/src/ldp.rs:1949"]
    POLICY -- Public --> MEDIA["cache_control_for by media type<br/>solid-pod-rs/src/ldp.rs:1935"]
    MEDIA -- RDF --> RDF["private, no-cache, must-revalidate<br/>solid-pod-rs/src/ldp.rs:1895"]
    MEDIA -- other --> NONE["No explicit Cache-Control"]
```

**Boundary:** this policy applies to the audience produced by the parsed ACL. Lost restrictions can misclassify a response as Public. Headers do not prove revocation through an intermediary cache. [HTTP acceptance and remaining delivery gaps](../../estate-review/2026-11-01-solid-http-delivery.md).
