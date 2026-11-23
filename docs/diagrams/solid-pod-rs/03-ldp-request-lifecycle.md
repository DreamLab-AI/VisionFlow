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
  - ../solid-pod-rs/crates/solid-pod-rs/src/rdf_input.rs
  - ../solid-pod-rs/CHANGELOG.md
verified_commit: a28a74832dd0a6e47a531e34e4adc46a3946dfd3
---


## SP-03.1 GET — the full read path

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant G as handle_get<br/>solid-pod-rs-server/src/lib.rs:1887
    participant GL as handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:4101
    participant AU as extract_pubkey<br/>solid-pod-rs-server/src/lib.rs:912
    participant W as enforce_read_ctx<br/>solid-pod-rs-server/src/lib.rs:1619
    participant S as Storage
    participant CP as set_cache_policy<br/>solid-pod-rs-server/src/lib.rs:1841

    C->>G: GET /pod/notes/hello.ttl
    Note over G: handle_get_inner does the work, and every answer under the<br/>remoteStorage container, refusals included, gets its protective headers<br/>solid-pod-rs-server/src/lib.rs:1893
    opt feature activitypub and path is /profile/card.jsonld
        G-->>C: return public actor document when Accept negotiates ActivityJSON<br/>solid-pod-rs-server/src/lib.rs:1914
    end
    alt path contains a glob star
        G->>GL: return glob handler result<br/>solid-pod-rs-server/src/lib.rs:1932
    end
    G->>AU: NIP-98 / dev-bearer verification (no body)
    AU-->>G: an optional pubkey
    G->>G: agent_uri — did:nostr prefix or a WebID URL<br/>solid-pod-rs-server/src/lib.rs:973
    G->>W: enforce acl:Read BEFORE any bytes<br/>solid-pod-rs-server/src/lib.rs:1942
    W-->>G: ResponseAudience Public or Private, else 401/403/503
    alt ldp::is_container(path)
        G->>S: container branch — see SP-03.2
    else resource
        G->>S: storage.get(path)
        S-->>G: body plus ResourceMeta
        G->>G: mashlib? then HTML wrapper (SP-03.2)
        G->>G: rdf_content_negotiate? then transcode (SP-03.10)
        G->>G: ETag from meta.etag<br/>solid-pod-rs-server/src/lib.rs:2101
        G->>CP: Cache-Control by audience (ADR-2002)
    end
    G-->>C: successful branch headers, ETag on verbatim or transcoded bodies
    Note over G: DEBT WAC-Allow is computed with no ACL document at lib.rs:1944,<br/>so successful reads advertise empty user and public modes.
    Note over G: A storage NotFound becomes a bare 404<br/>solid-pod-rs-server/src/lib.rs:2110
    Note over G: OPTIONS (handle_options, solid-pod-rs-server/src/lib.rs:3194, via<br/>ldp::options_for_conneg, solid-pod-rs/src/ldp.rs:1962) runs NO WAC check — it<br/>advertises capability, not content. Its Accept-Patch dialect list is fixed<br/>at solid-pod-rs/src/ldp.rs:1954, and the container Accept-Post adds text/n3 only when<br/>conneg is on (accept_post_for, solid-pod-rs/src/ldp.rs:103).
```

## SP-03.2 Container GET — three representations

```mermaid
flowchart TD
    START["handle_get, ldp::is_container true<br/>solid-pod-rs-server/src/lib.rs:1946"]
    HTML{"accept_includes_html?<br/>solid-pod-rs-server/src/lib.rs:1958"}
    IDX["try index.html child — require its own Read grant,<br/>serve HTML with optional live-reload injection, else fall through<br/>solid-pod-rs-server/src/lib.rs:1959"]
    REP["list_visible then ldp::render_container<br/>solid-pod-rs-server/src/lib.rs:1980<br/>solid-pod-rs-server/src/lib.rs:1981"]
    MASH{"mashlib::should_serve?<br/>solid-pod-rs-server/src/lib.rs:1988"}
    MHTML["mashlib HTML wrapper, no-store, X-Frame-Options DENY<br/>solid-pod-rs-server/src/lib.rs:1998"]
    JSONLD["application/ld+json listing<br/>solid-pod-rs-server/src/lib.rs:2009"]
    DONE["Cache policy uses child audience for index.html,<br/>container audience for the listing or wrapper.<br/>Plus Link, Accept-Ranges, WAC-Allow, Updates-via.<br/>solid-pod-rs-server/src/lib.rs:1972"]

    START --> HTML
    HTML -- yes --> IDX
    HTML -- no --> REP
    IDX -- "allowed child, read succeeds" --> DONE
    IDX -- "denied or read fails" --> REP
    REP --> MASH
    MASH -- yes --> MHTML --> DONE
    MASH -- no --> JSONLD --> DONE

    N["list_visible drops names listing_visible refuses: the dotfile policy<br/>(with the .well-known exemption) and server state such as trail secrets,<br/>provenance and ledger documents, so the root never names .git or .forge<br/>solid-pod-rs-server/src/lib.rs:1156<br/>solid-pod-rs-server/src/lib.rs:1176"]
    REP -.-> N
    N2["should_serve keys on Accept plus Sec-Fetch-Dest plus the stored content type<br/>solid-pod-rs/src/mashlib.rs:122"]
    MASH -.-> N2
    N3["The library's LdpContainerOps::container_representation still lists<br/>unfiltered; the server no longer uses it for GET<br/>solid-pod-rs/src/ldp.rs:2434"]
    REP -.-> N3
```

## SP-03.15 What a container listing carries — and the helpers the server does not call

```mermaid
flowchart TD
    GET["container GET, listing branch<br/>solid-pod-rs-server/src/lib.rs:1981"]
    RC["ldp::render_container — always PreferHeader::default<br/>solid-pod-rs/src/ldp.rs:1186<br/>solid-pod-rs/src/ldp.rs:1187"]
    FULL["Full JSON-LD: the container typed Container, BasicContainer, Resource<br/>solid-pod-rs/src/ldp.rs:1178"]
    MEM["ldp:contains, one entry per visible child<br/>solid-pod-rs/src/ldp.rs:1179"]
    TYP["a child ending in a slash is typed as a container too,<br/>any other child as Resource only<br/>solid-pod-rs/src/ldp.rs:1159<br/>solid-pod-rs/src/ldp.rs:1166"]
    OUT["served as application/ld+json<br/>solid-pod-rs-server/src/lib.rs:2009"]
    HTML["before it, two HTML branches: an Accept text/html request gets the container's<br/>index.html child when its own ACL allows Read, else falls through to the listing<br/>solid-pod-rs-server/src/lib.rs:1958-1978"]
    MASH["after render, the mashlib HTML wrapper with the JSON-LD embedded, when enabled<br/>and the request is a browser navigation<br/>solid-pod-rs-server/src/lib.rs:1988-2007"]
    LH["the type rels also go out as Link headers (SP-03.11)<br/>solid-pod-rs/src/ldp.rs:150"]

    HTML -.-> GET
    GET --> RC --> FULL --> OUT
    FULL -.-> MASH
    FULL --> MEM --> TYP
    OUT -.-> LH

    subgraph LIB["Library helpers with no caller in solid-pod-rs-server"]
        PREF["PreferHeader::parse — PreferMinimalContainer and PreferContainedIRIs<br/>solid-pod-rs/src/ldp.rs:247<br/>solid-pod-rs/src/ldp.rs:225"]
        TTL["render_container_turtle<br/>solid-pod-rs/src/ldp.rs:1191"]
        SMT["server_managed_triples — dcterms:modified, stat:size, stat:mtime,<br/>and ldp:contains for a container<br/>solid-pod-rs/src/ldp.rs:1044"]
        ILL["find_illegal_server_managed over SERVER_MANAGED_PREDICATES<br/>solid-pod-rs/src/ldp.rs:1098<br/>solid-pod-rs/src/ldp.rs:1089"]
    end
    RC -.-> LIB

    N["DRIFT: a Prefer header has no effect on the listing, the listing is never Turtle,<br/>and it carries no modified, size or mtime triples. The in-repo diagrams 02 and 04<br/>(removed by solid-pod-rs PR #18) drew those triples as rendered on GET."]
    SMT -.-> N
    N2["DEBT: the doc comment says client-supplied server-managed triples MUST be<br/>ignored at PUT time (solid-pod-rs/src/ldp.rs:1097); no server write path calls<br/>find_illegal_server_managed, so nothing in the server strips them from a PUT body."]
    ILL -.-> N2
```

- **What it shows:** the reference server's RDF listing is one fixed shape, the Full JSON-LD view with typed `ldp:contains` members; the only other answers to a container GET are HTML, the container's own `index.html` for an `Accept: text/html` request (`solid-pod-rs-server/src/lib.rs:1958-1978`) or the mashlib wrapper around that same JSON-LD (`solid-pod-rs-server/src/lib.rs:1988-2007`). The Prefer, Turtle and server-managed-triple helpers are library surface for embedders.
- **Why it is this way:** `render_container` is the backwards-compatible Phase 1 alias with the default preference, and the server's GET branch calls that alias rather than `render_container_jsonld` with a parsed `Prefer`.

## SP-03.3 PUT — resource and container creation

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_put<br/>solid-pod-rs-server/src/lib.rs:2158
    participant AU as extract_pubkey_with_body<br/>solid-pod-rs-server/src/lib.rs:919
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:1351
    participant Q as reserve_quota_for_size<br/>solid-pod-rs-server/src/lib.rs:3629
    participant S as Storage
    participant GM as git_mark_write<br/>solid-pod-rs-server/src/lib.rs:5066

    C->>P: PUT /pod/notes/hello.ttl
    alt container path with a trailing slash
        P->>P: has_basic_container_link?<br/>solid-pod-rs-server/src/lib.rs:2115
        alt Link rel type ldp BasicContainer present
            P->>W: acl:Write on the container
            P->>S: create_container under the resource lock<br/>solid-pod-rs-server/src/lib.rs:2177
            P-->>C: 201 with ETag and Link
        else
            P-->>C: 405 cannot PUT to a container<br/>solid-pod-rs-server/src/lib.rs:2190
        end
    end
    P->>AU: NIP-98 bound to method, URL and raw body hash
    P->>W: acl:Write — sidecar paths elevate to Control, see SP-04.5
    opt path is an acl or meta sidecar
        P->>P: check_proposed_acl<br/>solid-pod-rs-server/src/lib.rs:2216
        P-->>C: 400 if the body is not a parseable ACL, 409 unless it grants the caller<br/>effective Control on the protected resource (SP-13.1)<br/>solid-pod-rs-server/src/lib.rs:2218
        Note over P: A meta sidecar still needs Control to write, but the guard<br/>passes it without parsing it as an ACL<br/>solid-pod-rs-server/src/lib.rs:1274
    end
    P->>P: validate_rdf_body — Turtle syntax must parse (400), text/n3 needs conneg (415)<br/>solid-pod-rs-server/src/lib.rs:2222
    Note over P: DEBT valid Turtle with incomplete condition shape can pass<br/>solid-pod-rs-server/src/lib.rs:2149
    P->>P: take the resource lock, then If-Match / If-None-Match against the stored ETag<br/>solid-pod-rs-server/src/lib.rs:2224<br/>solid-pod-rs-server/src/lib.rs:2225
    P->>Q: atomic quota reservation
    P->>S: storage.put(path, body, content_type)
    P->>Q: finish_quota_reservation with the write outcome<br/>solid-pod-rs-server/src/lib.rs:2231
    P->>GM: additive provenance mark AFTER the write<br/>solid-pod-rs-server/src/lib.rs:2236
    GM-->>P: ProvenanceReceipt
    P-->>C: 201 with ETag, X-Provenance, Link and Accept-Ranges
    Note over P: INVARIANT: the precondition is judged under the same lock as the write,<br/>so a conditional PUT cannot race another writer's ETag change.
```

## SP-03.4 POST — Append, Slug resolution and unique-name minting

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as handle_post<br/>solid-pod-rs-server/src/lib.rs:2280
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:1351
    participant SL as ldp::resolve_slug<br/>solid-pod-rs/src/ldp.rs:187
    participant MU as mint_unique_target<br/>solid-pod-rs-server/src/lib.rs:2252
    participant S as Storage

    C->>P: POST to a container with a Slug header
    P->>W: acl:Append on the CONTAINER<br/>solid-pod-rs-server/src/lib.rs:2293
    P->>SL: validate then join the Slug: at most MAX_SLUG_BYTES (solid-pod-rs/src/ldp.rs:175), no slash, dot-dot, NUL or bare dot, else 400<br/>solid-pod-rs/src/ldp.rs:202
    SL-->>P: target path
    Note over SL: characters limited to A-Z a-z 0-9 dot underscore hyphen<br/>solid-pod-rs/src/ldp.rs:205<br/>no Slug gives a UUID name, solid-pod-rs/src/ldp.rs:215
    P->>P: 400 when the target is a name the dotfile policy refuses, a deliberate<br/>difference from JSS, which creates a file nothing can read or remove<br/>solid-pod-rs-server/src/lib.rs:2310
    alt target resolves to an acl or meta sidecar
        P->>W: RE-ENFORCE Write on the target, which elevates to Control<br/>solid-pod-rs-server/src/lib.rs:2329
        P->>P: check_proposed_acl, else 400 or 409 — acl only,<br/>a meta body passes the guard<br/>solid-pod-rs-server/src/lib.rs:2338
        Note over P: P0-4 — minting a sidecar from Append-only rights would be<br/>privilege escalation on a sibling resource. This branch does not<br/>mint a unique name, so a Control holder's POST replaces an existing sidecar.
    else ordinary resource
        P->>MU: probe existence, append a numeric suffix, bounded<br/>solid-pod-rs-server/src/lib.rs:2268
        MU-->>P: a free target, hash-suffixed at the ceiling<br/>solid-pod-rs-server/src/lib.rs:2277
    Note over P: the minted name is locked and re-probed, so a concurrent POST cannot take it<br/>solid-pod-rs-server/src/lib.rs:410
    end
    P->>P: validate_rdf_body on the target, as for PUT<br/>solid-pod-rs-server/src/lib.rs:2357
    P->>S: put(target, body, content_type)
    P-->>C: 201 with Location, ETag, X-Provenance and Link<br/>solid-pod-rs-server/src/lib.rs:2369
    Note over P: INVARIANT: an LDP POST to an ordinary name creates, never overwrites —<br/>resolve_slug alone would clobber a repeated Slug and lose data silently.
```

**Tension:** RESOLVED at solid-pod-rs 27e1899 and unchanged at a28a748. `mint_unique_target` still probes with `exists` (`solid-pod-rs-server/src/lib.rs:2253`), but POST takes the name through `mint_and_lock_target`, which holds the resource's write lock across the re-check and the write, retries up to `MINT_LOCK_ATTEMPTS` and refuses with 409 when no name is free (`solid-pod-rs-server/src/lib.rs:410-440`, `solid-pod-rs-server/src/lib.rs:2350`); a lock another process holds past the wait is 503 (`solid-pod-rs-server/src/lib.rs:409`). The hash fallback is still a deterministic hash of the target (`solid-pod-rs-server/src/lib.rs:2275`), and the lock loop stops rather than re-probe it (`solid-pod-rs-server/src/lib.rs:425-428`).

## SP-03.5 PATCH — the access mode is decided before parsing errors are answered

```mermaid
flowchart TD
    IN["handle_patch<br/>solid-pod-rs-server/src/lib.rs:2403"]
    CONT{"is_container?"}
    C405["405 cannot PATCH a container<br/>solid-pod-rs-server/src/lib.rs:2410"]
    PARSE["parse without answering: dialect, UTF-8, N3 against the resource URL<br/>solid-pod-rs-server/src/lib.rs:2425<br/>solid-pod-rs-server/src/lib.rs:2431"]
    RD["inspects the content (N3 with deletes or where, any JSON Patch,<br/>SPARQL with WHERE or a non-empty DELETE DATA): Read first, uncharged<br/>solid-pod-rs-server/src/lib.rs:2443<br/>solid-pod-rs-server/src/lib.rs:2458"]
    INS["insert_only: N3 with no deletes and no where,<br/>or SPARQL made only of INSERT DATA<br/>solid-pod-rs-server/src/lib.rs:2468<br/>solid-pod-rs/src/ldp.rs:1303"]
    AUTH["enforce_write_ctx with Append when insert-only, else Write<br/>sidecars and server state still elevate to Control<br/>solid-pod-rs-server/src/lib.rs:2475"]
    AO["append_only = insert-only and Write is NOT held (uncharged probe)<br/>solid-pod-rs-server/src/lib.rs:2490"]
    U415["415 unsupported patch dialect<br/>solid-pod-rs-server/src/lib.rs:2503"]
    B400["400 patch body is not valid UTF-8<br/>solid-pod-rs-server/src/lib.rs:2507"]
    NOBASE["append-only and no public origin: N3 re-read with no base,<br/>so a relative IRI is refused<br/>solid-pod-rs-server/src/lib.rs:2514"]
    LOCK["resource lock, then If-Match / If-None-Match<br/>solid-pod-rs-server/src/lib.rs:2528<br/>solid-pod-rs-server/src/lib.rs:2529"]
    BR["dispatch: append-only (SP-03.14), Write on an existing resource (SP-03.13),<br/>absent resource created<br/>solid-pod-rs-server/src/lib.rs:2534<br/>solid-pod-rs-server/src/lib.rs:2562<br/>solid-pod-rs-server/src/lib.rs:2673"]

    IN --> CONT
    CONT -- yes --> C405
    CONT -- no --> PARSE --> RD --> INS --> AUTH --> AO
    AO --> U415
    AO --> B400
    AO --> NOBASE --> LOCK --> BR

    NN["INVARIANT: only a patch that provably inserts and deletes nothing is<br/>authorised as Append (Solid Protocol and JSS); any delete, any where,<br/>JSON Patch, or an unparseable body needs full Write.<br/>solid-pod-rs-server/src/lib.rs:2460"]
    AUTH -.-> NN
    NO["Parse failures are answered after authorisation,<br/>so an unauthorised caller sees 401/403 first<br/>solid-pod-rs-server/src/lib.rs:2414"]
    PARSE -.-> NO
```

## SP-03.13 PATCH with Write on an existing or absent resource

```mermaid
flowchart TD
    EX{"resource exists?"}
    SEED["seed_graph_from_patch_target: Turtle against the resource URL,<br/>N-Triples fallback, else 409<br/>solid-pod-rs-server/src/lib.rs:3018"]
    N3["N3Patch apply on the seed<br/>solid-pod-rs/src/n3_patch.rs:237"]
    SPQ["apply_sparql_patch on the seed, SP-03.7<br/>solid-pod-rs/src/ldp.rs:1354"]
    JP["apply_json_patch, RFC 6902, on the stored JSON<br/>(a non-JSON body is replaced by an empty object)<br/>solid-pod-rs/src/ldp.rs:2156"]
    JGUARD["lockout guard on the patched JSON before storing<br/>solid-pod-rs-server/src/lib.rs:2598"]
    PATHS["ACL targets on this origin back to path form,<br/>an acl:condition result is 409<br/>solid-pod-rs-server/src/lib.rs:2627<br/>solid-pod-rs-server/src/lib.rs:2632"]
    ABS["apply_patch_to_absent then create — N3 or SPARQL only,<br/>JSON Patch on an absent resource is 400<br/>solid-pod-rs/src/ldp.rs:2402"]
    SER["graph_to_turtle delegates to to_ntriples<br/>solid-pod-rs-server/src/lib.rs:2749"]
    GUARD["F7 lockout guard on the serialised post-patch ACL,<br/>400 unparseable or 409 lockout<br/>solid-pod-rs-server/src/lib.rs:2641<br/>create path solid-pod-rs-server/src/lib.rs:2698"]
    OUT["204 No Content, or 201 on create, plus X-Provenance only"]

    EX -- "yes, N3" --> SEED --> N3
    EX -- "yes, SPARQL" --> SEED
    SEED --> SPQ
    EX -- "yes, json-patch" --> JP --> JGUARD --> OUT
    EX -- no --> ABS
    N3 --> PATHS
    SPQ --> PATHS
    ABS --> PATHS
    PATHS --> SER --> GUARD --> OUT
```

**Debt:** RESOLVED at solid-pod-rs 27e1899. An N3 or SPARQL PATCH can change or create an `.acl`. The seed reads a stored Turtle ACL with `@prefix` lines (`solid-pod-rs-server/src/lib.rs:3041`), the patched graph's ACL targets are put back in path form (`solid-pod-rs-server/src/lib.rs:2626-2628`), and the lockout guard sees the result as `text/turtle` (`solid-pod-rs-server/src/lib.rs:2401`, `solid-pod-rs-server/src/lib.rs:2640-2656`). A patched ACL that carries `acl:condition` is refused with 409, because the N-Triples write-back cannot preserve it (`solid-pod-rs-server/src/lib.rs:2382-2385`, `solid-pod-rs-server/src/lib.rs:2631-2633`, `solid-pod-rs-server/src/lib.rs:2692-2693`). This is source tracing; no route test was run for RDF PATCH on an ACL.

## SP-03.14 Append-only PATCH — exactly the inserted triples, or nothing

```mermaid
flowchart TD
    IN["existing resource, append_only<br/>solid-pod-rs-server/src/lib.rs:2534"]
    INS["apply the patch to an EMPTY graph: the inserted triples only<br/>solid-pod-rs-server/src/lib.rs:2536<br/>solid-pod-rs-server/src/lib.rs:2538"]
    ATS["append_to_stored<br/>solid-pod-rs-server/src/lib.rs:2920"]
    TYPE{"stored type Turtle or N-Triples,<br/>UTF-8, parses under a fixed base?<br/>solid-pod-rs-server/src/lib.rs:2926<br/>solid-pod-rs-server/src/lib.rs:2936"}
    EMPTY["nothing inserted: 204, nothing written<br/>solid-pod-rs-server/src/lib.rs:2548"]
    APP["fresh blank-node labels, triples appended as N-Triples,<br/>every stored byte kept, content type unchanged<br/>solid-pod-rs-server/src/lib.rs:2942<br/>solid-pod-rs-server/src/lib.rs:2967"]
    CHK{"combined bytes re-parsed equal stored plus inserted?<br/>ground triples as a set, blank-node triples as a multiset<br/>solid-pod-rs-server/src/lib.rs:2969"}
    REF["409 existing resource cannot be appended to<br/>one generic message, nothing written<br/>solid-pod-rs-server/src/lib.rs:2858"]
    WR["quota, put, git mark, 204<br/>solid-pod-rs-server/src/lib.rs:2550<br/>solid-pod-rs-server/src/lib.rs:2557"]

    IN --> INS --> ATS --> TYPE
    TYPE -- no --> REF
    TYPE -- yes --> EMPTY
    TYPE -- yes --> APP --> CHK
    CHK -- no --> REF
    CHK -- yes --> WR

    N["INVARIANT: an agent with Append and no Write can only add; the refusal is<br/>generic so it learns nothing about the stored body. Both parses are bounded<br/>by MAX_RDF_INPUT_BYTES and MAX_RDF_TRIPLES, so past 4 MiB a resource<br/>can no longer be appended to.<br/>solid-pod-rs/src/rdf_input.rs:47<br/>solid-pod-rs/src/rdf_input.rs:51"]
    CHK -.-> N
```

## SP-03.7 SPARQL Update — bounded, WHERE evaluated, absent deletes refused

```mermaid
flowchart TD
    IN["apply_sparql_patch<br/>solid-pod-rs/src/ldp.rs:1354"]
    SIZE["over SPARQL_UPDATE_MAX_BYTES (1 MiB): 400 before parsing<br/>solid-pod-rs/src/ldp.rs:1355"]
    OPS["more than 16 operations: 400 before anything runs<br/>solid-pod-rs/src/ldp.rs:1376"]
    PAT["a WHERE basic graph pattern over MAX_WHERE_PATTERNS (32): 400<br/>solid-pod-rs/src/ldp.rs:1387<br/>solid-pod-rs/src/n3_patch.rs:27"]
    BUD["one StepBudget for the whole request, shared by every operation<br/>solid-pod-rs/src/ldp.rs:1396<br/>solid-pod-rs/src/n3_patch.rs:45"]
    ID["INSERT DATA: add triples not already present<br/>solid-pod-rs/src/ldp.rs:1592"]
    DD["DELETE DATA: every named triple must be present, else 409<br/>and nothing is written<br/>solid-pod-rs/src/ldp.rs:1601<br/>solid-pod-rs/src/ldp.rs:1609"]
    DI["DELETE/INSERT ... WHERE, DELETE WHERE, INSERT ... WHERE:<br/>WHERE must be a plain BGP, USING refused<br/>solid-pod-rs/src/ldp.rs:1566<br/>solid-pod-rs/src/ldp.rs:1624"]
    SW["apply_sparql_where: solutions against the graph as it stood,<br/>then all deletes, then all inserts<br/>solid-pod-rs/src/n3_patch.rs:560"]
    ABSENT["a delete instance naming an absent triple: 409, nothing written<br/>solid-pod-rs/src/n3_patch.rs:596"]
    BROAD["join over MAX_PARTIAL_BINDINGS or MAX_MATCH_STEPS: 409 too broad,<br/>never truncated<br/>solid-pod-rs/src/n3_patch.rs:33<br/>solid-pod-rs/src/n3_patch.rs:38"]
    OTHER["anything else (named graphs, LOAD, CLEAR, FILTER, OPTIONAL, ...)<br/>refused, never ignored<br/>solid-pod-rs/src/ldp.rs:1648"]

    IN --> SIZE --> OPS --> PAT --> BUD
    BUD --> ID
    BUD --> DD
    BUD --> DI --> SW
    SW --> ABSENT
    SW --> BROAD
    BUD --> OTHER
```

## SP-03.6 The non-destructive-write invariant

```mermaid
sequenceDiagram
    autonumber
    participant P as handle_patch<br/>solid-pod-rs-server/src/lib.rs:2403
    participant S as Storage
    participant SG as seed_graph_from_patch_target<br/>solid-pod-rs-server/src/lib.rs:3018
    participant AP as N3Patch::apply<br/>solid-pod-rs/src/n3_patch.rs:237

    P->>S: get(path) under the resource lock
    S-->>P: the current stored body
    P->>SG: parse the stored body as Turtle against the resource URL, else N-Triples<br/>solid-pod-rs-server/src/lib.rs:3041
    alt parses
        SG-->>P: seed Graph carrying the existing triples
        P->>AP: apply the patch ON TOP of the seed<br/>solid-pod-rs-server/src/lib.rs:2574
    else unparseable
        SG-->>P: error — fail closed, refuse the write<br/>solid-pod-rs-server/src/lib.rs:3044
    end
    P->>S: put the serialised N-Triples back, stored as text/turtle<br/>solid-pod-rs-server/src/lib.rs:2658
    Note over SG: Before this fix the working graph started EMPTY, so every incremental<br/>PATCH silently discarded the resource's existing triples. Refusing an<br/>unparseable body (409) is deliberate — fail closed rather than destroy.<br/>solid-pod-rs-server/src/lib.rs:3044
    Note over SG: A stored body over the Turtle reader's bound falls back to N-Triples,<br/>where a relative IRI stays unresolved and fails to match (409)<br/>solid-pod-rs-server/src/lib.rs:3037
```

**Debt:** the invariant covers the N3 and SPARQL dialects only. The JSON Patch branch treats a stored body that is not JSON as `{}` and writes the patched object back under the stored content type (`solid-pod-rs-server/src/lib.rs:2586-2589`, `solid-pod-rs-server/src/lib.rs:2615`). So a JSON Patch that applies cleanly to an empty object, sent to a Turtle resource, replaces its triples where an N3 patch would be refused. This is source tracing and was not reproduced over HTTP.

## SP-03.8 COPY — the non-standard verb

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as handle_copy<br/>solid-pod-rs-server/src/lib.rs:3974
    participant W as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:1351
    participant S as Storage

    C->>H: COPY to a destination with a Source header
    H->>W: acl:Write on the DESTINATION<br/>solid-pod-rs-server/src/lib.rs:3981
    alt no Source header
        H-->>C: 400 Source header required<br/>solid-pod-rs-server/src/lib.rs:3997
    end
    H->>H: normalise_copy_source, dotfile allowlist, else 400 or 403<br/>solid-pod-rs-server/src/lib.rs:4000<br/>solid-pod-rs-server/src/lib.rs:4004
    H->>W: acl:Read on the SOURCE, as for a plain GET<br/>solid-pod-rs-server/src/lib.rs:4009
    H->>S: get(source)
    alt source missing
        H-->>C: 404 source resource not found<br/>solid-pod-rs-server/src/lib.rs:4014
    end
    opt destination is an acl or meta sidecar
        H->>H: check_proposed_acl BEFORE the write, 400 unparseable, 409 lockout —<br/>for an acl it also checks RDF syntax, as COPY has no body validation<br/>solid-pod-rs-server/src/lib.rs:4021
    end
    H->>S: put(dest, body, source content type) under the destination lock<br/>solid-pod-rs-server/src/lib.rs:4041<br/>solid-pod-rs-server/src/lib.rs:4043
    opt source has an acl sidecar
        H->>S: copy it only if the caller can read the source acl, write the destination acl<br/>and keeps Control, otherwise skip it silently<br/>solid-pod-rs-server/src/lib.rs:4054
    end
    H-->>C: 201 with Location
    Note over H: INVARIANT: COPY needs Write on the destination AND the same Read as a plain<br/>GET on the source, so write rights alone cannot copy an unreadable resource.<br/>solid-pod-rs-server/src/lib.rs:4009
    Note over H: The accompanying ACL is always the sidecar form x.acl<br/>whatever --acl-layout says (SP-13.3)<br/>solid-pod-rs-server/src/lib.rs:4052
```

## SP-03.9 Glob GET

```mermaid
flowchart TD
    IN["GET whose path contains a star, routed to handle_glob_get<br/>solid-pod-rs-server/src/lib.rs:4101"]
    PAT{"path ends with slash-star?<br/>solid-pod-rs-server/src/lib.rs:4107"}
    NF["404 unsupported glob pattern"]
    GATE["enforce_read_ctx on the FOLDER<br/>solid-pod-rs-server/src/lib.rs:4123"]
    LIST["storage.list(folder)<br/>solid-pod-rs-server/src/lib.rs:4125"]
    SKIP["skip sub-containers, acl, meta and prov.ttl children<br/>solid-pod-rs-server/src/lib.rs:4129"]
    CHILD["enforce_read_inner per child — a refused or pay-per-read child is left out<br/>solid-pod-rs-server/src/lib.rs:4140"]
    FILTER["keep only turtle, n-triples and n3 children<br/>solid-pod-rs-server/src/lib.rs:4146"]
    MERGE["concatenate bodies into one text/turtle response<br/>solid-pod-rs-server/src/lib.rs:4151"]
    EMPTY["404 no matching RDF resources<br/>solid-pod-rs-server/src/lib.rs:4159"]

    IN --> PAT
    PAT -- no --> NF
    PAT -- yes --> GATE --> LIST --> SKIP --> CHILD --> FILTER --> MERGE
    FILTER --> EMPTY

    N["INVARIANT (P0-1): the glob merge is gated on acl:Read of the folder, so a<br/>glob under a private container cannot bypass the plain container read check."]
    GATE -.-> N
    N2["INVARIANT: each merged child is also gated on its own acl:Read, so a child with a<br/>stricter acl sidecar is left out of the body. Any Private child makes the audience Private.<br/>solid-pod-rs-server/src/lib.rs:4140"]
    CHILD -.-> N2
```

## SP-03.10 RDF content negotiation

```mermaid
flowchart LR
    ACC["Accept header"]
    BEST["best_explicit_rdf_format — highest-q EXPLICIT format<br/>solid-pod-rs-server/src/lib.rs:2759"]
    NEG["rdf_content_negotiate, base = the URL being read<br/>solid-pod-rs-server/src/lib.rs:2809<br/>solid-pod-rs-server/src/lib.rs:2069"]
    FMT["ldp::RdfFormat::from_mime<br/>solid-pod-rs/src/ldp.rs:318"]
    JL["stored JSON-LD: parse_jsonld, the whole @graph,<br/>remote contexts never fetched<br/>solid-pod-rs-server/src/lib.rs:2825<br/>solid-pod-rs/src/rdf_input.rs:237"]
    TT["otherwise parse_turtle against the read URL,<br/>then lenient Graph::parse_ntriples<br/>solid-pod-rs-server/src/lib.rs:2831<br/>solid-pod-rs/src/ldp.rs:906"]
    OUT1["to_ntriples, also served as Turtle<br/>solid-pod-rs/src/ldp.rs:734"]
    OUT2["to_jsonld<br/>solid-pod-rs/src/ldp.rs:827"]
    VERB["fall through to the stored bytes verbatim<br/>solid-pod-rs-server/src/lib.rs:2092"]

    ACC --> BEST --> NEG --> FMT
    NEG --> JL
    NEG --> TT
    JL --> OUT1
    TT --> OUT1
    TT --> OUT2
    NEG -. "non-RDF, unparseable, or a wildcard Accept" .-> VERB

    N["A transcoded response carries Vary: Accept<br/>solid-pod-rs-server/src/lib.rs:2078"]
    OUT1 -.-> N
    N2["Relative IRIs in a stored Turtle body resolve against the read URL,<br/>so a transcode never emits a relative IRI into N-Triples;<br/>a prefixed Turtle body now transcodes instead of falling to verbatim<br/>solid-pod-rs-server/src/lib.rs:2826"]
    TT -.-> N2
    N3N["Also verbatim: empty Accept, same format as stored, or an RDF/XML target<br/>solid-pod-rs-server/src/lib.rs:2815-2821<br/>solid-pod-rs-server/src/lib.rs:2851"]
    NEG -.-> N3N
```

## SP-03.11 Response header helpers for LDP responses

```mermaid
classDiagram
    class ResponseHeaders {
        +set_link_headers  solid-pod-rs-server/src/lib.rs:1766
        +set_wac_allow  solid-pod-rs-server/src/lib.rs:1783
        +set_updates_via  solid-pod-rs-server/src/lib.rs:1792
        +set_provenance_headers  solid-pod-rs-server/src/lib.rs:1817
        +set_cache_policy  solid-pod-rs-server/src/lib.rs:1841
        +append_vary  solid-pod-rs-server/src/lib.rs:1860
    }
    class LinkHeaders {
        +ldp::link_headers(path)  solid-pod-rs/src/ldp.rs:143
        +ldp::link_headers_with_layout(path, layout)  solid-pod-rs/src/ldp.rs:150
    }
    class WacAllow {
        +wac::wac_allow_header  solid-pod-rs/src/wac/mod.rs:292
    }
    ResponseHeaders ..> LinkHeaders
    ResponseHeaders ..> WacAllow
    note for ResponseHeaders "set_link_headers also sets Accept-Ranges: none on a container,\nbytes on a resource — solid-pod-rs-server/src/lib.rs:1772.\nset_updates_via is a no-op unless the notifications route is mounted\n(solid-pod-rs-server/src/lib.rs:1793), then rewrites https to wss and http to ws\nand appends the notifications path — solid-pod-rs-server/src/lib.rs:1796."
    note for LinkHeaders "rel=acl names acl_advertised_key under the configured layout:\nthe sidecar by default — solid-pod-rs/src/ldp.rs:160 (SP-13.2)."
```

**Drift:** each handler picks its own subset of these helpers. Successful GET branches set Link with Accept-Ranges, WAC-Allow, Updates-Via and the cache policy (`solid-pod-rs-server/src/lib.rs:2104-2107`). PUT and POST set X-Provenance and Link with Accept-Ranges (`solid-pod-rs-server/src/lib.rs:2241-2242`, `solid-pod-rs-server/src/lib.rs:2376-2377`). PATCH sets only X-Provenance (`solid-pod-rs-server/src/lib.rs:2670`). COPY sets only Location (`solid-pod-rs-server/src/lib.rs:4090-4093`). Glob GET sets only the cache policy (`solid-pod-rs-server/src/lib.rs:4163`). A GET that misses storage returns a bare 404 (`solid-pod-rs-server/src/lib.rs:2110`). So the CHANGELOG's "Accept-Ranges is sent on GET, HEAD and write responses" (`CHANGELOG.md:364`) does not hold for PATCH, COPY or a 404.

## SP-03.12 ADR-2002 cache policy — audience beats media type

```mermaid
flowchart TD
    A["Audience from enforce_read_inner<br/>solid-pod-rs-server/src/lib.rs:1737"] --> PRIVATE{"Private?"}
    PRIVATE -- yes --> V["append Vary Authorization<br/>solid-pod-rs-server/src/lib.rs:1846"]
    PRIVATE -- no --> EXIST
    V --> EXIST{"Cache-Control already set?<br/>solid-pod-rs-server/src/lib.rs:1848"}
    EXIST -- yes --> KEEP["Keep existing policy, e.g. mashlib no-store<br/>solid-pod-rs-server/src/lib.rs:2049"]
    EXIST -- no --> POLICY["cache_control_for_response<br/>solid-pod-rs/src/ldp.rs:2137"]
    POLICY -- Private --> NS["private, no-store<br/>solid-pod-rs/src/ldp.rs:2105"]
    POLICY -- Public --> MEDIA["cache_control_for by media type<br/>solid-pod-rs/src/ldp.rs:2091"]
    MEDIA -- RDF --> RDF["private, no-cache, must-revalidate<br/>solid-pod-rs/src/ldp.rs:2051"]
    MEDIA -- other --> NONE["No explicit Cache-Control"]
```

**Boundary:** this policy applies to the audience produced by the parsed ACL. Lost restrictions can misclassify a response as Public. Headers do not prove revocation through an intermediary cache. [HTTP acceptance and remaining delivery gaps](../../estate-review/2026-11-01-solid-http-delivery.md).
