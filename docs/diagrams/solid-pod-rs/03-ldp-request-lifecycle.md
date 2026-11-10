---
id: SP-03
title: LDP request lifecycle — verbs, containers, content negotiation and PATCH
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2002, ADR-2004, ADR-2005]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/n3_patch.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/mashlib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/resolver.rs
verified_commit: 27e18995fc20790fa956acb41f7d8420544a13ec
---

## SP-03.1 GET — the full read path

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant G as handle_get<br/>solid-pod-rs-server/src/lib.rs:1504
    participant GL as handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:3364
    participant AU as extract_pubkey<br/>solid-pod-rs-server/src/lib.rs:704
    participant W as enforce_read_ctx<br/>solid-pod-rs-server/src/lib.rs:1263
    participant S as Storage
    participant CP as set_cache_policy<br/>solid-pod-rs-server/src/lib.rs:1462

    C->>G: GET /pod/notes/hello.ttl
    opt feature activitypub and path is /profile/card.jsonld
        G-->>C: return public actor document when Accept negotiates ActivityJSON<br/>solid-pod-rs-server/src/lib.rs:1515
    end
    alt path contains a glob star
        G->>GL: return glob handler result<br/>solid-pod-rs-server/src/lib.rs:1533
    end
    G->>AU: NIP-98 / dev-bearer verification (no body)
    AU-->>G: an optional pubkey
    G->>G: agent_uri — did:nostr prefix or a WebID URL<br/>solid-pod-rs-server/src/lib.rs:765
    G->>W: enforce acl:Read BEFORE any bytes<br/>solid-pod-rs-server/src/lib.rs:1543
    W-->>G: ResponseAudience Public or Private, else 401/403/503
    alt ldp::is_container(path)
        G->>S: container branch — see SP-03.2
    else resource
        G->>S: storage.get(path)
        S-->>G: body plus ResourceMeta
        G->>G: mashlib? then HTML wrapper (SP-03.2)
        G->>G: rdf_content_negotiate? then transcode (SP-03.10)
        G->>G: ETag from meta.etag<br/>solid-pod-rs-server/src/lib.rs:1702
        G->>CP: Cache-Control by audience (ADR-2002)
    end
    G-->>C: successful branch headers, ETag on verbatim or transcoded bodies
    Note over G: DEBT WAC-Allow is computed with no ACL document at lib.rs:1545,<br/>so successful reads advertise empty user and public modes.
    Note over G: A storage NotFound becomes a bare 404<br/>solid-pod-rs-server/src/lib.rs:1711
    Note over G: OPTIONS (handle_options, solid-pod-rs-server/src/lib.rs:2546, via<br/>ldp::options_for_conneg, solid-pod-rs/src/ldp.rs:1752) runs NO WAC check — it<br/>advertises capability, not content. Its Accept-Patch dialect list is fixed<br/>at solid-pod-rs/src/ldp.rs:1744, and the container Accept-Post adds text/n3 only when<br/>conneg is on (accept_post_for, solid-pod-rs/src/ldp.rs:103).
```

## SP-03.2 Container GET — three representations

```mermaid
flowchart TD
    START["handle_get, ldp::is_container true<br/>solid-pod-rs-server/src/lib.rs:1547"]
    HTML{"accept_includes_html?<br/>solid-pod-rs-server/src/lib.rs:1559"}
    IDX["try index.html child — require its own Read grant,<br/>serve HTML with optional live-reload injection, else fall through<br/>solid-pod-rs-server/src/lib.rs:1560"]
    REP["storage.container_representation<br/>solid-pod-rs-server/src/lib.rs:1583"]
    MASH{"mashlib::should_serve?<br/>solid-pod-rs-server/src/lib.rs:1592"}
    MHTML["mashlib HTML wrapper, no-store, X-Frame-Options DENY<br/>solid-pod-rs-server/src/lib.rs:1602"]
    JSONLD["application/ld+json listing<br/>solid-pod-rs-server/src/lib.rs:1613"]
    DONE["Cache policy uses child audience for index.html,<br/>container audience for the listing or wrapper.<br/>Plus Link, WAC-Allow, Updates-via.<br/>solid-pod-rs-server/src/lib.rs:1574"]

    START --> HTML
    HTML -- yes --> IDX
    HTML -- no --> REP
    IDX -- "allowed child, read succeeds" --> DONE
    IDX -- "denied or read fails" --> REP
    REP --> MASH
    MASH -- yes --> MHTML --> DONE
    MASH -- no --> JSONLD --> DONE

    N["container_representation is a blanket-impl trait method over any Storage:<br/>list(path) then render_container<br/>solid-pod-rs/src/ldp.rs:2224"]
    REP -.-> N
    N2["should_serve keys on Accept plus Sec-Fetch-Dest plus the stored content type<br/>solid-pod-rs/src/mashlib.rs:122"]
    MASH -.-> N2
```

## SP-03.3 PUT — resource and container creation

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_put<br/>solid-pod-rs-server/src/lib.rs:1759
    participant AU as extract_pubkey_with_body<br/>solid-pod-rs-server/src/lib.rs:711
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:1052
    participant Q as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:2895
    participant S as Storage
    participant GM as git_mark_write<br/>solid-pod-rs-server/src/lib.rs:4250

    C->>P: PUT /pod/notes/hello.ttl
    alt container path with a trailing slash
        P->>P: has_basic_container_link?<br/>solid-pod-rs-server/src/lib.rs:1716
        alt Link rel type ldp BasicContainer present
            P->>W: acl:Write on the container
            P->>S: create_container
            P-->>C: 201 with ETag and Link
        else
            P-->>C: 405 cannot PUT to a container<br/>solid-pod-rs-server/src/lib.rs:1791
        end
    end
    P->>AU: NIP-98 bound to method, URL and raw body hash
    P->>W: acl:Write — sidecar paths elevate to Control, see SP-04.5
    opt path is an acl or meta sidecar
        P->>P: proposed_acl_keeps_caller_control<br/>solid-pod-rs-server/src/lib.rs:1815
        P-->>C: 409 unless the candidate ACL parses and grants the caller<br/>effective Control on the protected resource (SP-04.6)
        Note over P: A meta sidecar still needs Control to write, but the guard<br/>passes it without parsing it as an ACL<br/>solid-pod-rs-server/src/lib.rs:982
    end
    P->>P: validate_rdf_body — Turtle syntax must parse (400), text/n3 needs conneg (415)<br/>solid-pod-rs-server/src/lib.rs:1825
    Note over P: DEBT valid Turtle with incomplete condition shape can pass<br/>solid-pod-rs-server/src/lib.rs:1750
    P->>Q: atomic quota reservation
    P->>S: storage.put(path, body, content_type)
    P->>Q: finish_quota_reservation with the write outcome<br/>solid-pod-rs-server/src/lib.rs:1834
    P->>GM: additive provenance mark AFTER the write<br/>solid-pod-rs-server/src/lib.rs:1839
    GM-->>P: ProvenanceReceipt
    P-->>C: 201 with ETag, X-Provenance and Link
```

## SP-03.4 POST — Append, Slug resolution and unique-name minting

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_post<br/>solid-pod-rs-server/src/lib.rs:1883
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:1052
    participant SL as ldp::resolve_slug<br/>solid-pod-rs/src/ldp.rs:180
    participant MU as mint_unique_target<br/>solid-pod-rs-server/src/lib.rs:1855
    participant S as Storage

    C->>P: POST to a container with a Slug header
    P->>W: acl:Append on the CONTAINER<br/>solid-pod-rs-server/src/lib.rs:1896
    P->>SL: validate then join the Slug: at most MAX_SLUG_BYTES (solid-pod-rs/src/ldp.rs:168), no slash, dot-dot, NUL or bare dot, else 400<br/>solid-pod-rs/src/ldp.rs:195
    SL-->>P: target path
    Note over SL: characters limited to A-Z a-z 0-9 dot underscore hyphen<br/>solid-pod-rs/src/ldp.rs:198<br/>no Slug gives a UUID name, solid-pod-rs/src/ldp.rs:208
    alt target resolves to an acl or meta sidecar
        P->>W: RE-ENFORCE Write on the target, which elevates to Control<br/>solid-pod-rs-server/src/lib.rs:1925
        P->>P: proposed_acl_keeps_caller_control, else 409 — acl only,<br/>a meta body passes the guard<br/>solid-pod-rs-server/src/lib.rs:1933
        Note over P: P0-4 — minting a sidecar from Append-only rights would be<br/>privilege escalation on a sibling resource. This branch does not<br/>mint a unique name, so a Control holder's POST replaces an existing sidecar.
    else ordinary resource
        P->>MU: probe existence, append a numeric suffix, bounded<br/>solid-pod-rs-server/src/lib.rs:1871
        MU-->>P: a free target, hash-suffixed at the ceiling<br/>solid-pod-rs-server/src/lib.rs:1880
    Note over P: the minted name is locked and re-probed, so a concurrent POST cannot take it<br/>solid-pod-rs-server/src/lib.rs:261
    end
    P->>P: validate_rdf_body on the target, as for PUT<br/>solid-pod-rs-server/src/lib.rs:1956
    P->>S: put(target, body, content_type)
    P-->>C: 201 with Location, ETag, X-Provenance and Link<br/>solid-pod-rs-server/src/lib.rs:1968
    Note over P: INVARIANT: an LDP POST to an ordinary name creates, never overwrites —<br/>resolve_slug alone would clobber a repeated Slug and lose data silently.
```

**Tension:** RESOLVED at solid-pod-rs 27e1899. `mint_unique_target` still probes with `exists` (`solid-pod-rs-server/src/lib.rs:1856`), but POST now takes the name through `mint_and_lock_target`, which holds the resource's write lock across the re-check and the write, retries up to 64 times and refuses with 409 when no name is free (`solid-pod-rs-server/src/lib.rs:261-283`, `solid-pod-rs-server/src/lib.rs:1949`). The hash fallback is still a deterministic hash of the target (`solid-pod-rs-server/src/lib.rs:1878`), and the lock loop stops rather than re-probe it (`solid-pod-rs-server/src/lib.rs:268-271`).

## SP-03.5 PATCH — non-destructive, three dialects

```mermaid
flowchart TD
    IN["handle_patch<br/>solid-pod-rs-server/src/lib.rs:2002"]
    CONT{"is_container?"}
    C405["405 cannot PATCH a container<br/>solid-pod-rs-server/src/lib.rs:2009"]
    RD["a patch that inspects the content, an N3 with deletes or where, or any JSON Patch, also needs Read<br/>solid-pod-rs-server/src/lib.rs:2040"]
    AUTH["enforce_write_ctx with AccessMode Write, never Append<br/>solid-pod-rs-server/src/lib.rs:2055"]
    DIAL{"ldp::patch_dialect_from_mime<br/>solid-pod-rs/src/ldp.rs:2142"}
    U415["415 unsupported patch dialect<br/>solid-pod-rs-server/src/lib.rs:2065"]
    UTF{"body is UTF-8?"}
    B400["400 patch body is not valid UTF-8<br/>solid-pod-rs-server/src/lib.rs:2069"]
    EX{"resource exists?"}
    SEED["seed_graph_from_patch_target, Turtle first then N-Triples<br/>solid-pod-rs-server/src/lib.rs:2374"]
    N3["N3Patch parse then apply, re-exported by ldp<br/>solid-pod-rs/src/n3_patch.rs:205"]
    SPQ["apply_sparql_patch<br/>solid-pod-rs/src/ldp.rs:1212"]
    JP["apply_json_patch, RFC 6902, on the stored JSON<br/>(a non-JSON body is replaced by an empty object)<br/>solid-pod-rs/src/ldp.rs:1946"]
    JGUARD["lockout guard on the patched JSON before storing<br/>solid-pod-rs-server/src/lib.rs:2123"]
    ABS["apply_patch_to_absent then create — N3 or SPARQL only,<br/>JSON Patch on an absent resource is 400<br/>solid-pod-rs/src/ldp.rs:2192"]
    SER["graph_to_turtle delegates to to_ntriples<br/>solid-pod-rs-server/src/lib.rs:2272"]
    GUARD["F7 lockout guard on the serialised post-patch ACL<br/>solid-pod-rs-server/src/lib.rs:2165"]
    OUT["204 No Content, or 201 on create, plus X-Provenance"]

    IN --> CONT
    CONT -- yes --> C405
    CONT -- no --> RD --> AUTH --> DIAL
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

    NN["INVARIANT: PATCH requires Write, not Append — an N3 patch carrying<br/>solid:deletes can destroy data, so Append-only principals are excluded.<br/>solid-pod-rs-server/src/lib.rs:2050"]
    AUTH -.-> NN
```

**Debt:** RESOLVED at solid-pod-rs 27e1899. An N3 or SPARQL PATCH can now change or create an `.acl`. The seed reads a stored Turtle ACL with `@prefix` lines (`solid-pod-rs-server/src/lib.rs:2397`), the patched graph's ACL targets are put back in path form (`solid-pod-rs-server/src/lib.rs:2151-2153`), and the lockout guard sees the result as `text/turtle` (`solid-pod-rs-server/src/lib.rs:2000`, `solid-pod-rs-server/src/lib.rs:2165-2173`). A patched ACL that carries `acl:condition` is refused with 409, because the N-Triples write-back cannot preserve it (`solid-pod-rs-server/src/lib.rs:1981-1984`, `solid-pod-rs-server/src/lib.rs:2156-2158`, `solid-pod-rs-server/src/lib.rs:2216-2217`). This is source tracing; no route test was run for RDF PATCH on an ACL.

## SP-03.6 The non-destructive-write invariant

```mermaid
sequenceDiagram
    autonumber
    participant P as handle_patch<br/>solid-pod-rs-server/src/lib.rs:2002
    participant S as Storage
    participant SG as seed_graph_from_patch_target<br/>solid-pod-rs-server/src/lib.rs:2374
    participant AP as N3Patch::apply<br/>solid-pod-rs/src/n3_patch.rs:205

    P->>S: get(path)
    S-->>P: the current stored body
    P->>SG: parse the stored body as Turtle, else N-Triples
    alt parses
        SG-->>P: seed Graph carrying the existing triples
        P->>AP: apply the patch ON TOP of the seed<br/>solid-pod-rs-server/src/lib.rs:2099
    else unparseable
        SG-->>P: error — fail closed, refuse the write<br/>solid-pod-rs-server/src/lib.rs:2400
    end
    P->>S: put the serialised N-Triples back, stored as text/turtle<br/>solid-pod-rs-server/src/lib.rs:2182
    Note over SG: Before this fix the working graph started EMPTY, so every incremental<br/>PATCH silently discarded the resource's existing triples. Refusing an<br/>unparseable body (409) is deliberate — fail closed rather than destroy.<br/>solid-pod-rs-server/src/lib.rs:2400
```

**Debt:** the invariant covers the N3 and SPARQL dialects only. The JSON Patch branch treats a stored body that is not JSON as `{}` and writes the patched object back under the stored content type (`solid-pod-rs-server/src/lib.rs:2111-2114`, `solid-pod-rs-server/src/lib.rs:2140`). So a JSON Patch that applies cleanly to an empty object, sent to a Turtle resource, replaces its triples where an N3 patch would be refused. This is source tracing and was not reproduced over HTTP.

## SP-03.8 COPY — the non-standard verb

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as handle_copy<br/>solid-pod-rs-server/src/lib.rs:3239
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:1052
    participant S as Storage

    C->>H: COPY to a destination with a Source header
    H->>W: acl:Write on the DESTINATION<br/>solid-pod-rs-server/src/lib.rs:3246
    alt no Source header
        H-->>C: 400 Source header required<br/>solid-pod-rs-server/src/lib.rs:3262
    end
    H->>H: normalise_copy_source, dotfile allowlist, else 400 or 403<br/>solid-pod-rs-server/src/lib.rs:3265
    H->>W: acl:Read on the SOURCE, as for a plain GET<br/>solid-pod-rs-server/src/lib.rs:3274
    H->>S: get(source)
    alt source missing
        H-->>C: 404 source resource not found<br/>solid-pod-rs-server/src/lib.rs:3279
    end
    opt destination is an acl or meta sidecar
        H->>H: proposed_acl_keeps_caller_control BEFORE the write, else 409 —<br/>for an acl it also checks Turtle syntax, as COPY has no body validation<br/>solid-pod-rs-server/src/lib.rs:3285
    end
    H->>S: put(dest, body, source content type)<br/>solid-pod-rs-server/src/lib.rs:3307
    opt source has an acl sidecar
        H->>S: copy it only if the caller can read the source acl, write the destination acl<br/>and keeps Control, otherwise skip it silently<br/>solid-pod-rs-server/src/lib.rs:3318
    end
    H-->>C: 201 with Location
    Note over H: INVARIANT: COPY needs Write on the destination AND the same Read as a plain<br/>GET on the source, so write rights alone cannot copy an unreadable resource.<br/>solid-pod-rs-server/src/lib.rs:3274
```

## SP-03.9 Glob GET

```mermaid
flowchart TD
    IN["GET whose path contains a star, routed to handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:3364"]
    PAT{"path ends with slash-star?<br/>solid-pod-rs-server/src/lib.rs:3370"}
    NF["404 unsupported glob pattern"]
    GATE["enforce_read_ctx on the FOLDER<br/>solid-pod-rs-server/src/lib.rs:3386"]
    LIST["storage.list(folder)<br/>solid-pod-rs-server/src/lib.rs:3388"]
    SKIP["skip sub-containers, acl, meta and prov.ttl children<br/>solid-pod-rs-server/src/lib.rs:3392"]
    CHILD["enforce_read_inner per child — a refused or pay-per-read child is left out<br/>solid-pod-rs-server/src/lib.rs:3403"]
    FILTER["keep only turtle, n-triples and n3 children<br/>solid-pod-rs-server/src/lib.rs:3409"]
    MERGE["concatenate bodies into one text/turtle response<br/>solid-pod-rs-server/src/lib.rs:3414"]
    EMPTY["404 no matching RDF resources<br/>solid-pod-rs-server/src/lib.rs:3422"]

    IN --> PAT
    PAT -- no --> NF
    PAT -- yes --> GATE --> LIST --> SKIP --> CHILD --> FILTER --> MERGE
    FILTER --> EMPTY

    N["INVARIANT (P0-1): the glob merge is gated on acl:Read of the folder, so a<br/>glob under a private container cannot bypass the plain container read check."]
    GATE -.-> N
    N2["INVARIANT: each merged child is also gated on its own acl:Read, so a child with a<br/>stricter acl sidecar is left out of the body. Any Private child makes the audience Private.<br/>solid-pod-rs-server/src/lib.rs:3403"]
    CHILD -.-> N2
```

## SP-03.10 RDF content negotiation

```mermaid
flowchart LR
    ACC["Accept header"]
    BEST["best_explicit_rdf_format — highest-q EXPLICIT format<br/>solid-pod-rs-server/src/lib.rs:2282"]
    NEG["rdf_content_negotiate<br/>solid-pod-rs-server/src/lib.rs:2330"]
    FMT["ldp::RdfFormat::from_mime<br/>solid-pod-rs/src/ldp.rs:311"]
    NT["Graph::parse_ntriples<br/>solid-pod-rs/src/ldp.rs:826"]
    OUT1["to_ntriples<br/>solid-pod-rs/src/ldp.rs:727"]
    OUT2["to_jsonld<br/>solid-pod-rs/src/ldp.rs:747"]
    VERB["fall through to the stored bytes verbatim<br/>solid-pod-rs-server/src/lib.rs:1693"]

    ACC --> BEST --> NEG --> FMT
    NEG --> NT
    NT --> OUT1
    NT --> OUT2
    NEG -. "non-RDF, unparseable, or a wildcard Accept" .-> VERB

    N["A transcoded response carries Vary: Accept<br/>solid-pod-rs-server/src/lib.rs:1679"]
    OUT1 -.-> N
    N2["Only PATCH-written resources persist as N-Triples (graph_to_turtle, SP-03.5).<br/>PUT and POST store the client's bytes; a Turtle body with prefixes does not<br/>parse as N-Triples, so it is served verbatim whatever the Accept.<br/>solid-pod-rs-server/src/lib.rs:2344"]
    NT -.-> N2
    N3N["Also verbatim: empty Accept, same format as stored, or an RDF/XML target<br/>solid-pod-rs-server/src/lib.rs:2335-2341<br/>solid-pod-rs-server/src/lib.rs:2361"]
    NEG -.-> N3N
```

## SP-03.11 Response header helpers for LDP responses

```mermaid
classDiagram
    class ResponseHeaders {
        +set_link_headers  solid-pod-rs-server/src/lib.rs:1402
        +set_wac_allow  solid-pod-rs-server/src/lib.rs:1410
        +set_updates_via  solid-pod-rs-server/src/lib.rs:1417
        +set_provenance_headers  solid-pod-rs-server/src/lib.rs:1438
        +set_cache_policy  solid-pod-rs-server/src/lib.rs:1462
        +append_vary  solid-pod-rs-server/src/lib.rs:1481
    }
    class LinkHeaders {
        +ldp::link_headers(path)  solid-pod-rs/src/ldp.rs:143
    }
    class WacAllow {
        +wac::wac_allow_header  solid-pod-rs/src/wac/mod.rs:265
    }
    ResponseHeaders ..> LinkHeaders
    ResponseHeaders ..> WacAllow
    note for ResponseHeaders "set_updates_via rewrites https to wss and http to ws, then appends the\nnotifications path — solid-pod-rs-server/src/lib.rs:1421."
    note for LinkHeaders "rel=acl now names the resolver's own sidecar key, so a container\nadvertises /folder.acl — solid-pod-rs/src/ldp.rs:153."
```

**Drift:** this panel's heading used to say "every LDP response", but each handler picks its own subset. Successful GET branches set Link, WAC-Allow, Updates-Via and the cache policy (`solid-pod-rs-server/src/lib.rs:1705-1708`). PUT and POST set Link and X-Provenance (`solid-pod-rs-server/src/lib.rs:1844-1845`). PATCH sets only X-Provenance (`solid-pod-rs-server/src/lib.rs:2194`). COPY sets only Location (`solid-pod-rs-server/src/lib.rs:3353-3356`). Glob GET sets only the cache policy (`solid-pod-rs-server/src/lib.rs:3426`). A GET that misses storage returns a bare 404 (`solid-pod-rs-server/src/lib.rs:1711`). The panel lists the helpers, not a guarantee that every response carries them.

## SP-03.12 ADR-2002 cache policy — audience beats media type

```mermaid
flowchart TD
    A["Audience from enforce_read_inner<br/>solid-pod-rs-server/src/lib.rs:1375"] --> PRIVATE{"Private?"}
    PRIVATE -- yes --> V["append Vary Authorization<br/>solid-pod-rs-server/src/lib.rs:1467"]
    PRIVATE -- no --> EXIST
    V --> EXIST{"Cache-Control already set?<br/>solid-pod-rs-server/src/lib.rs:1469"}
    EXIST -- yes --> KEEP["Keep existing policy, e.g. mashlib no-store<br/>solid-pod-rs-server/src/lib.rs:1653"]
    EXIST -- no --> POLICY["cache_control_for_response<br/>solid-pod-rs/src/ldp.rs:1927"]
    POLICY -- Private --> NS["private, no-store<br/>solid-pod-rs/src/ldp.rs:1895"]
    POLICY -- Public --> MEDIA["cache_control_for by media type<br/>solid-pod-rs/src/ldp.rs:1881"]
    MEDIA -- RDF --> RDF["private, no-cache, must-revalidate<br/>solid-pod-rs/src/ldp.rs:1841"]
    MEDIA -- other --> NONE["No explicit Cache-Control"]
```

**Boundary:** this policy applies to the audience produced by the parsed ACL. Lost restrictions can misclassify a response as Public. Headers do not prove revocation through an intermediary cache. [HTTP acceptance and remaining delivery gaps](../../estate-review/2026-11-01-solid-http-delivery.md).
