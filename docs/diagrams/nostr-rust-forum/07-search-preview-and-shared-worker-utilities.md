---
id: NF-07
title: search-worker, preview-worker and the shared rate-limit / ASCII utilities
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: []
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/store.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/embed.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/rerank.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/Cargo.toml
  - ../nostr-rust-forum/docs/jev-option.md
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/ssrf.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/parse.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/oembed.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-ascii/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/admin_shared.rs
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/ascii_img.rs
verified_commit: 8a5cf817e81b20e2b761899eab753768c0cb4988
---

## NF-07.1 search-worker — routes and gates

```mermaid
flowchart TB
    F["fetch<br/>nostr-bbs-search-worker/src/lib.rs:913"]
    OPT["Options preflight nostr-bbs-search-worker/src/lib.rs:917"]
    RL["rate limit 100 req / 60 s per IP, bucket in SEARCH_CONFIG KV<br/>nostr-bbs-search-worker/src/lib.rs:925"]
    RT["route nostr-bbs-search-worker/src/lib.rs:968"]
    H["GET /health, /status, / nostr-bbs-search-worker/src/lib.rs:972"]
    S["POST /search nostr-bbs-search-worker/src/lib.rs:977"]
    E["POST /embed nostr-bbs-search-worker/src/lib.rs:982"]
    I["POST /ingest - NIP-98 admin only nostr-bbs-search-worker/src/lib.rs:987"]
    NF404["404 nostr-bbs-search-worker/src/lib.rs:991"]
    CRON["scheduled nostr-bbs-search-worker/src/lib.rs:1002"]

    F --> OPT --> RL --> RT
    RT --> H & S & E & I
    RT --> NF404
    CRON --> WARM["load_store only - keeps the R2 connection warm<br/>nostr-bbs-search-worker/src/lib.rs:1004"]

    N1["DOC-DRIFT: the cron trigger nostr-bbs-search-worker/wrangler.toml:36 runs every five minutes but<br/>REINDEXES NOTHING - the handler's whole body is a load_store touch. There is no reindex or rebuild<br/>path anywhere in lib.rs or store.rs reachable from the schedule."]
    N2["/ingest is the only mutating route and it is admin-gated<br/>nostr-bbs-search-worker/src/lib.rs:716 via require_nip98_admin nostr-bbs-search-worker/src/auth.rs:63"]
```

## NF-07.2 The third admin authority — static-only, and the only real key in a template

```mermaid
flowchart LR
    VAR["ADMIN_PUBKEYS_VAR<br/>nostr-bbs-core/src/admin_shared.rs:91"]
    READ["search-worker reads it as its ONLY admin source<br/>nostr-bbs-search-worker/src/auth.rs:54"]
    CHK["is_admin nostr-bbs-search-worker/src/auth.rs:52<br/>enforced at ingest nostr-bbs-search-worker/src/auth.rs:86"]
    TPL["nostr-bbs-search-worker/wrangler.toml:33 ships a REAL 64-hex pubkey"]
    WHY["No D1 binding for members/whitelist in this worker<br/>nostr-bbs-search-worker/src/auth.rs:8"]

    VAR --> READ --> CHK
    TPL --> READ
    WHY --> READ

    N1["INVARIANT: the parse rule is shared through nostr_bbs_core so the comma / whitespace / npub-or-hex<br/>semantics cannot drift between the three workers - see NF-08.7"]
    N2["This worker is the ONLY wrangler template that declares ADMIN_PUBKEYS, yet the auth and relay<br/>workers both read it - the deployment gap is written up in NF-08.7 N4"]
    N3["It is also the only template shipping a non-placeholder value where every other template ships<br/>example.com or a zeroed key nostr-bbs-search-worker/wrangler.toml:33"]
```

## NF-07.3 Semantic search — embeddings and the RVF store

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant S as handle_search<br/>nostr-bbs-search-worker/src/lib.rs:420
    participant EM as embed_texts<br/>nostr-bbs-search-worker/src/embed.rs:69
    participant AI as Workers AI binding
    participant R2 as VECTORS bucket

    C->>S: POST /search { q, k }
    S->>S: trim the query, reject a whitespace-only one nostr-bbs-search-worker/src/lib.rs:182
    S->>S: k clamped to 1..=100 nostr-bbs-search-worker/src/lib.rs:480
    S->>EM: embed the query
    EM->>AI: @cf/baai/bge-small-en-v1.5 nostr-bbs-search-worker/src/embed.rs:86 embed.rs:18
    alt binding absent or inference fails
        EM->>EM: deterministic 3-pass hash embedding nostr-bbs-search-worker/src/embed.rs:80 embed.rs:115 embed.rs:121
    end
    EM->>EM: L2-normalise either way nostr-bbs-search-worker/src/embed.rs:104 embed.rs:129
    S->>R2: load the RVF store nostr-bbs-search-worker/src/lib.rs:301
    S->>S: cosine k-NN over overfetch_k candidates nostr-bbs-search-worker/src/store.rs:63 store.rs:72 nostr-bbs-search-worker/src/lib.rs:506
    S->>S: sort descending by score nostr-bbs-search-worker/src/store.rs:79
    S->>S: visible_top_k filters THEN truncates to k nostr-bbs-search-worker/src/lib.rs:507 nostr-bbs-search-worker/src/lib.rs:160
    S-->>C: hits

    Note over S: INVARIANT fail-loudly on model drift: an index built with one embedding model and queried with another returns 409, not silently wrong neighbours nostr-bbs-search-worker/src/lib.rs:495
    Note over EM: DIM is 384 nostr-bbs-search-worker/src/embed.rs:15 and the store's ENTRY_SIZE is derived from it - u64 label plus 384 f32 = 1544 bytes nostr-bbs-search-worker/src/store.rs:17
    Note over S: The hash fallback is deliberately kept at parity with a legacy TypeScript implementation nostr-bbs-search-worker/src/embed.rs:113
    Note over S: INVARIANT filter then truncate, never the other way round - the store truncates to its own k before this module filters, so asking for exactly k and filtering after could only shrink the set nostr-bbs-search-worker/src/lib.rs:150 nostr-bbs-search-worker/src/lib.rs:160
    Note over S: Over-fetch is 4x the requested k, capped at 400 and never below k nostr-bbs-search-worker/src/lib.rs:139 nostr-bbs-search-worker/src/lib.rs:144
    Note over S: A whitespace-only query used to survive the empty check and match the ENTIRE index at score 0 under the default minScore nostr-bbs-search-worker/src/lib.rs:182
```

## NF-07.4 RVF format and where the pieces live

```mermaid
flowchart TB
    RVF["RVF object in R2<br/>bucket VECTORS nostr-bbs-search-worker/wrangler.toml:15<br/>key RVF_STORE_KEY nostr-bbs-search-worker/wrangler.toml:32 read at nostr-bbs-search-worker/src/lib.rs:297"]
    SEGV["Segment 0 Vec - packed label + vector<br/>nostr-bbs-search-worker/src/store.rs:7"]
    SEGM["Segment 1 Meta - JSON format/dim/count/metric<br/>nostr-bbs-search-worker/src/store.rs:8 written store.rs:97"]
    SER["to_rvf_bytes nostr-bbs-search-worker/src/store.rs:88"]
    DES["from_rvf_bytes nostr-bbs-search-worker/src/store.rs:167"]
    KV["id-to-label mapping, model and publicLabels live in SEARCH_CONFIG KV<br/>nostr-bbs-search-worker/src/lib.rs:345"]

    RVF --> SEGV & SEGM
    SER --> RVF
    RVF --> DES
    KV -.-> DES

    N1["The deserialiser parses ONLY the Vec segment nostr-bbs-search-worker/src/store.rs:185 - the mapping<br/>the Meta segment documents is actually read back from KV, so the two stores must be written together"]
    N2["INVARIANT: visibility is fail-closed and deliberately left that way - only a label in publicLabels<br/>is anonymously visible nostr-bbs-search-worker/src/lib.rs:127 nostr-bbs-search-worker/src/lib.rs:128,<br/>and the fix for an empty result set is the ingest caller declaring public true, never a relaxation here"]
    N3["README.md:81 now says only that search uses the configured worker, embedding and storage stack; the<br/>384-dim L2-normalised cosine k-NN over an RVF store is stated nowhere in the README and is established<br/>by the code, see NF-07.3. DRIFT: the older README row claiming it is gone nostr-rust-forum/README.md:81"]
    N4["An all-private ingest batch is now REFUSED with a 400 before anything is written, because such a<br/>batch indexes content no search can ever return nostr-bbs-search-worker/src/lib.rs:205<br/>nostr-bbs-search-worker/src/lib.rs:783. A caller retracting vectors opts in with allowPrivate<br/>nostr-bbs-search-worker/src/lib.rs:254"]
```

## NF-07.5 preview-worker — routes, caching and origin policy

```mermaid
flowchart TB
    F["fetch nostr-bbs-preview-worker/src/lib.rs:498"]
    AL["install PREVIEW_ALLOWED_HOSTS into the SSRF allowlist<br/>nostr-bbs-preview-worker/src/lib.rs:511 set at ssrf.rs:153"]
    RL["rate limit 30 req / 60 s per IP against RATE_LIMIT KV<br/>nostr-bbs-preview-worker/src/lib.rs:519"]
    P["GET /preview nostr-bbs-preview-worker/src/lib.rs:533"]
    A["GET /ascii nostr-bbs-preview-worker/src/lib.rs:534"]
    H["GET /health nostr-bbs-preview-worker/src/lib.rs:535"]
    ST["GET /stats - static, points at CF Analytics nostr-bbs-preview-worker/src/lib.rs:536"]
    CACHE["Cloudflare Cache API keyed on a synthetic internal URL<br/>nostr-bbs-preview-worker/src/lib.rs:149 nostr-bbs-preview-worker/src/lib.rs:157"]
    TTL["TTLs: OpenGraph 10 days nostr-bbs-preview-worker/src/lib.rs:32,<br/>Twitter 1 day nostr-bbs-preview-worker/src/lib.rs:33,<br/>ASCII 7 days because the render is deterministic<br/>nostr-bbs-preview-worker/src/lib.rs:34"]

    F --> AL --> RL --> P & A & H & ST
    P --> CACHE --> TTL
    A --> CACHE

    N1["CORS origin comes from ALLOWED_ORIGIN nostr-bbs-preview-worker/src/lib.rs:95, applied at nostr-bbs-preview-worker/src/lib.rs:102"]
    N2["The preview handler runs the SSRF check BEFORE consulting the cache<br/>nostr-bbs-preview-worker/src/lib.rs:274 then nostr-bbs-preview-worker/src/lib.rs:284"]
    N3["Twitter/X URLs branch to oEmbed rather than OpenGraph parsing<br/>nostr-bbs-preview-worker/src/lib.rs:284"]
```

## NF-07.6 The SSRF guard — every blocked class

```mermaid
flowchart TB
    URL["candidate URL"]
    P0["unparseable to BLOCK ssrf.rs:294"]
    P1["scheme allowlist http/https only ssrf.rs:300"]
    P2["embedded credentials user:pass@ blocked ssrf.rs:305"]
    P3["ports other than 80 and 443 blocked ssrf.rs:311"]
    P4["egress allowlist when configured ssrf.rs:323"]
    P5["mDNS .local blocked ssrf.rs:331"]
    P6["pure-integer and pure-hex IP obfuscation blocked ssrf.rs:338 ssrf.rs:342"]
    P7["localhost and *.localhost blocked ssrf.rs:347"]
    P8["cloud metadata 169.254.169.254 and metadata.google.internal ssrf.rs:352 ssrf.rs:353"]
    P9["IPv4 private ranges 10/8 :448, 127/8 :451, 172.16/12 :454,<br/>192.168/16 at ssrf.rs:457, 169.254/16 at ssrf.rs:460,<br/>zero-net at ssrf.rs:463, reserved high range at ssrf.rs:466"]
    P10["IPv6 loopback and unspecified ssrf.rs:377; ULA fc00::/7,<br/>link-local fe80::/10, site-local fec0::/10 ssrf.rs:384"]
    P11["embedded-IPv4 forms: 6to4 2002::/16 ssrf.rs:400,<br/>NAT64 64:ff9b::/96 ssrf.rs:404, v4-mapped ssrf.rs:409"]

    URL --> P0 --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9 --> P10 --> P11

    N1["INVARIANT: the policy is re-run on EVERY redirect hop, not just the first - redirects are followed<br/>manually ssrf.rs:174 so each hop can be re-checked ssrf.rs:192, capped at MAX_REDIRECTS 3 ssrf.rs:33"]
    N2["Size guards: a Content-Length pre-check ssrf.rs:246 plus a streamed guard that aborts mid-body<br/>ssrf.rs:257; 2 MiB for HTML/JSON ssrf.rs:38, 10 MiB for images ssrf.rs:44"]
    N3["DIVERGENCE: there is NO wall-clock timeout - only redirect-count and body-size limits bound a fetch"]
    N4["DIVERGENCE acknowledged in-code: the Workers runtime exposes no resolve-then-pin primitive, so<br/>without PREVIEW_ALLOWED_HOSTS the guard is denylist-only and DNS-rebinding-vulnerable<br/>nostr-bbs-preview-worker/src/ssrf.rs:13. The allowlist ssrf.rs:100 ssrf.rs:114 is the real mitigation<br/>and the template does not set it - nostr-bbs-preview-worker/wrangler.toml:13 declares only ALLOWED_ORIGIN."]
```

## NF-07.7 Unfurl extraction

```mermaid
flowchart LR
    FETCH["fetch_open_graph_data<br/>nostr-bbs-preview-worker/src/parse.rs:203"]
    SSRFF["routed through ssrf_fetch_with_redirects parse.rs:213"]
    CAP["read_text_capped parse.rs:224"]
    OG["og:title parse.rs:31 and the reversed attribute order parse.rs:35<br/>og:description parse.rs:40, og:image parse.rs:50"]
    FALL["title falls back to the HTML title element parse.rs:178"]
    REL["relative image URLs resolved against the target parse.rs:186"]
    FAV["favicon via Google's s2 service, not the target site parse.rs:172"]
    OE["Twitter/X oEmbed publish.twitter.com oembed.rs:11<br/>host detector oembed.rs:38 oembed.rs:49<br/>omit_script, dnt, dark theme oembed.rs:65"]

    FETCH --> SSRFF --> CAP --> OG --> FALL --> REL
    OG --> FAV
    OE --> SSRFF

    N1["The oEmbed path deliberately avoids Response::json because that would bypass the body cap<br/>nostr-bbs-preview-worker/src/oembed.rs:90"]
    N2["The favicon choice is an outbound third-party dependency in an otherwise LAN-shaped design -<br/>every preview embeds a google.com favicon URL parse.rs:172"]
```

## NF-07.8 Shared utilities every worker links

```mermaid
classDiagram
    class RateLimit {
        client_ip
        check_rate_limit KV bucket
        ensure_replay_schema
        verify_nip98 with replay
    }
    class Replay {
        atomic INSERT OR IGNORE
        single-use NIP-98 token
    }
    class Ascii {
        image decode to phosphor HTML
        pure PNG JPEG GIF WebP BMP decoders
    }
    RateLimit --> Replay
    Ascii --> PreviewWorker
    Ascii --> BbsClient
    RateLimit --> AuthWorker
    RateLimit --> RelayWorker
    RateLimit --> PodWorker
    RateLimit --> SearchWorker
    RateLimit --> PreviewWorker

    note for RateLimit "Each worker calls check_rate_limit with its OWN<br/>KV binding and its own budget - auth 20/60s on<br/>SESSIONS nostr-bbs-auth-worker/src/lib.rs:175, search<br/>100/60s on SEARCH_CONFIG nostr-bbs-search-worker/src/lib.rs:925,<br/>preview 30/60s on RATE_LIMIT nostr-bbs-preview-worker/src/lib.rs:519"
    note for Replay "INVARIANT one shared replay database - the search<br/>worker binds REPLAY_DB nostr-bbs-search-worker/src/auth.rs:31<br/>and delegates to the shared verifier<br/>nostr-bbs-search-worker/src/auth.rs:40 exactly as the auth<br/>worker does. See NF-02.5 and NF-08.4"
    note for Ascii "The image feature pulls PURE-Rust decoders only,<br/>default-features off, so the wasm32 build stays<br/>lean nostr-rust-forum/Cargo.toml:197-200. The BBS client never converts<br/>client-side - it fetches a pre-rendered fragment<br/>from the preview worker nostr-bbs-bbs-client/src/ascii_img.rs:203"
```

## NF-07.9 The opt-in Jev rerank — off by default, can only permute what search already shows

```mermaid
flowchart TB
    FLAG["JEV_SEARCH_RERANK must be 1 or true; unset means off<br/>nostr-bbs-search-worker/src/lib.rs:585, flag name nostr-bbs-search-worker/src/rerank.rs:20"]
    VIS["visible_top_k filters then truncates to k - the rerank input<br/>nostr-bbs-search-worker/src/lib.rs:507"]
    GATE["match on query and jev_enabled - off yields the pre-change list and body<br/>nostr-bbs-search-worker/src/lib.rs:511"]
    KEEP["rerank_or_keep - any Fallback returns the input untouched<br/>nostr-bbs-search-worker/src/lib.rs:592"]
    SNIP["public snippets loaded from SEARCH_CONFIG KV key store_key:snippets<br/>nostr-bbs-search-worker/src/lib.rs:639"]
    RR["rerank scores the head only - HEAD_CAP 10<br/>nostr-bbs-search-worker/src/rerank.rs:79, cap nostr-bbs-search-worker/src/rerank.rs:22"]
    LOW["below the confidence floor, default 0.5, falls back<br/>nostr-bbs-search-worker/src/rerank.rs:90, default nostr-bbs-search-worker/src/rerank.rs:27"]
    ORD["apply_order permutes the head, the tail keeps cosine order<br/>nostr-bbs-search-worker/src/rerank.rs:143"]
    ING["ingest persists a snippet only when the flag is on AND the entry is public<br/>nostr-bbs-search-worker/src/lib.rs:795, nostr-bbs-search-worker/src/lib.rs:821"]

    VIS --> GATE
    FLAG --> GATE
    GATE -->|"flag on"| KEEP
    SNIP --> KEEP
    KEEP --> RR
    RR --> LOW
    RR --> ORD
    ING -.-> SNIP

    N1["INVARIANT: the reranker runs after the visibility filter, so it cannot surface a hidden candidate<br/>nostr-bbs-search-worker/src/lib.rs:507, nostr-bbs-search-worker/src/lib.rs:511"]
    N2["OPEN: the item is a proposed research candidate with no measured uplift and unfitted thresholds<br/>docs/jev-option.md:11"]
```

**What it shows.** The one optional step added to `POST /search`: when `JEV_SEARCH_RERANK` is on, the first ten already-visible hits are re-ordered by a remote typed-decision (Jev) score, and the response gains a `jev` object; any failure returns the pre-change list byte-for-byte.

**Why it is this way.** The option is declared in `docs/jev-option.md` rather than an ADR: every Jev item is "off by default" with nothing leaving the Worker, and falls back on any failure (`../nostr-rust-forum/docs/jev-option.md:4`). No search `wrangler.toml` variable sets the flag, so the shipped state is off (`../nostr-rust-forum/crates/nostr-bbs-search-worker/src/lib.rs:585`). With the flag on, the query text and up to ten public snippets leave the Worker (`../nostr-rust-forum/docs/jev-option.md:35`).

**Open:** nothing records any measured uplift from the rerank, and its confidence floor is an unfitted default (`../nostr-rust-forum/docs/jev-option.md:11`, `../nostr-rust-forum/crates/nostr-bbs-search-worker/src/rerank.rs:27`).
