---
id: LM-02
title: Facade boot, bundle activation, the route table and the error map
area: loom
governing:
  - ../loom/docs/design/RUST-ARCHITECTURE.md
  - ../loom/docs/design/ADR-135-ontology-loom-node.md
adrs: [loom:ADR-135, loom:ADR-137, loom:ADR-138, loom:ADR-140, loom:ADR-141]
sources:
  - ../loom/crates/loom-facade/src/main.rs
  - ../loom/crates/loom-facade/src/lib.rs
  - ../loom/crates/loom-facade/src/config.rs
  - ../loom/crates/loom-facade/src/state.rs
  - ../loom/crates/loom-facade/src/error.rs
  - ../loom/crates/loom-facade/src/routes/mod.rs
  - ../loom/crates/loom-facade/src/routes/attest.rs
  - ../loom/crates/loom-facade/src/routes/health.rs
  - ../loom/crates/loom-facade/src/bundle.rs
  - ../loom/crates/loom-facade/Cargo.toml
  - ../loom/crates/loom-attest-proofgate/src/lib.rs
  - ../loom/crates/loom-backend-openai/src/lib.rs
  - ../loom/crates/loom-scaffold/src/policy.rs
  - ../loom/crates/loom-embed-xinference/src/lib.rs
  - ../loom/crates/loom-graph-oxigraph/src/lib.rs
  - ../loom/crates/loom-vector-ruvector/src/hnsw.rs
  - ../loom/deploy/compose.profile-a.yml
  - ../loom/README.md
verified_commit: {loom: 355946dd2deacfdef21171d53eafd025887e447e}
---

## LM-02.1 Process start: activation first and fatal, every accelerator fail-open

```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>loom-facade/src/main.rs:34
    participant L as try_app_state<br/>loom-facade/src/lib.rs:157
    participant B as LoadedBundle<br/>loom-facade/src/bundle.rs:125
    participant R as axum serve<br/>loom-facade/src/main.rs:56
    M->>M: --build-info prints the receipt and exits, loom-facade/src/main.rs:35,36
    M->>L: try_app_state_from_env, loom-facade/src/main.rs:48
    L->>B: activate_or_fall_back FIRST, loom-facade/src/lib.rs:164
    B-->>L: verified bundle, degraded bundle, retained fallback, or BundleError
    Note over L: a BundleError is returned, not logged, so main exits before binding, loom-facade/src/lib.rs:54,58
    L->>L: lexical index load, else the empty-index floor, loom-facade/src/lib.rs:177,192
    L->>L: Oxigraph load, fail-open, loom-facade/src/lib.rs:199,205
    L->>L: HNSW open against the artefact contract, loom-facade/src/lib.rs:213,214
    L->>L: embedder verify spawned, never blocks the bind, loom-facade/src/lib.rs:236,237
    L->>L: backend from env, empty URL means a retrieval-only node, loom-facade/src/lib.rs:246,248
    L->>B: verify_loaded and disk_matches_loaded when attested, loom-facade/src/lib.rs:256,257,259
    Note over L,B: INVARIANT: an attested start whose disk marker moved during load refuses with ActivatedDrift, loom-facade/src/lib.rs:260
    L->>L: one retriever Arc serves both the lexical and manifest ports, loom-facade/src/lib.rs:270,273,274
    L-->>M: AppState, loom-facade/src/lib.rs:272
    M->>R: bind port from LOOM_FACADE_PORT on all interfaces, loom-facade/src/main.rs:49,53,54
```

**What it shows.** Only bundle activation can stop the process; a missing graph, vector index, embedder or backend degrades. The default port is 8080 (`loom-facade/src/config.rs:166`); Profile A overrides it (see LM-06.3).

## LM-02.2 Activation outcomes and the start-up fallback branch

```mermaid
flowchart TB
    START["activate_or_fall_back<br/>loom-facade/src/lib.rs:75"]
    GARB{"commit marker present<br/>but not valid JSON?<br/>loom-facade/src/lib.rs:80"}
    ACT["activate_or_degraded<br/>loom-facade/src/lib.rs:87"]
    OK["serve the data directory bundle<br/>loom-facade/src/lib.rs:88"]
    SEL{"fallback::select finds a retained<br/>generation that validates?<br/>loom-facade/src/lib.rs:99"}
    FB["activate_fallback on the rebased config<br/>loudly logged<br/>loom-facade/src/lib.rs:106,113,124"]
    DEG["garbled marker and no fallback:<br/>old degraded start<br/>loom-facade/src/lib.rs:100,101"]
    REF["return the original failure, refuse to serve<br/>loom-facade/src/lib.rs:104,130"]
    START --> GARB
    GARB -->|no| ACT
    ACT -->|ok| OK
    ACT -->|BundleError| SEL
    GARB -->|yes| SEL
    SEL -->|yes| FB
    SEL -->|no, garbled| DEG
    SEL -->|no| REF
    NODEG["degraded = no commit marker at all: identity hashed<br/>over what is visible, atomicity_verified false<br/>loom-facade/src/bundle.rs:38,41,42"]
    ACT --- NODEG
    PROSE["DEBT: on a fallback the prose index is re-attached by hand,<br/>because the adapter reads ONTOLOGY_PROSE_INDEX from env<br/>loom-facade/src/lib.rs:134,135,142,145"]
    FB --- PROSE
    SEE["see LM-05.4 for how select walks promotions.jsonl"]
    SEL --- SEE
```

**What it shows.** A fallback is never silent: it is logged at warn and reported as `status: fallback` on `/health` (`loom-facade/src/routes/health.rs:171,172`). A garbled marker with no usable fallback keeps the pre-closeout degraded start rather than refusing.

## LM-02.3 The route table and the tower layers

```mermaid
flowchart TB
    subgraph LAYERS["tower layers, outermost last, loom-facade/src/routes/mod.rs:77,81,82,83"]
        TO["timeout LOOM_TIMEOUT, 408 on elapse, default 600 s<br/>loom-facade/src/config.rs:119,167"]
        BL["body cap LOOM_MAX_BODY_BYTES, default 16 MiB<br/>loom-facade/src/config.rs:120,168"]
        CORS["CorsLayer permissive<br/>loom-facade/src/routes/mod.rs:82"]
        TRACE["TraceLayer for HTTP, the outermost layer<br/>loom-facade/src/routes/mod.rs:83"]
    end
    subgraph READ["model-free routes"]
        H["GET /health<br/>loom-facade/src/routes/mod.rs:54"]
        G["GET /loom/generation and /generation<br/>loom-facade/src/routes/mod.rs:55,56"]
        SC["POST /loom/scaffold and /scaffold<br/>loom-facade/src/routes/mod.rs:57,58"]
        SP["POST /loom/sparql and /sparql<br/>loom-facade/src/routes/mod.rs:59,60"]
        SE["POST /loom/search and /search<br/>loom-facade/src/routes/mod.rs:61,62"]
        SS["POST /loom/search/semantic, 404 unless<br/>LOOM_SEMANTIC_DEBUG_SURFACE<br/>loom-facade/src/routes/mod.rs:63,294,296"]
        MCP["POST /mcp, GET /mcp answers 405<br/>loom-facade/src/routes/mod.rs:65,124,126"]
    end
    subgraph MODEL["model routes"]
        CH["POST /v1/chat/completions<br/>loom-facade/src/routes/mod.rs:66"]
        MO["GET /v1/models<br/>loom-facade/src/routes/mod.rs:67"]
    end
    subgraph LEDGER["attest feature, default-on"]
        AT["POST /loom/attest<br/>loom-facade/src/routes/mod.rs:73"]
        AV["GET /loom/attest/verify<br/>loom-facade/src/routes/mod.rs:74"]
    end
    LAYERS --> READ --> MODEL --> LEDGER
    FEAT["default features include attest<br/>loom-facade/Cargo.toml:57,58"]
    LEDGER --- FEAT
    TENS["TENSION: POST /loom/attest takes signer as a free string and<br/>no layer authenticates the caller, behind permissive CORS<br/>loom-facade/src/routes/attest.rs:85,86,113,124<br/>loom-facade/src/routes/mod.rs:82"]
    AT --- TENS
    DRIFT["DOC-DRIFT: the router claims parity with app/loom_facade.py,<br/>which no longer exists, app/ holds only mirror.sh<br/>loom-facade/src/routes/mod.rs:1,2"]
    READ --- DRIFT
    SEE["see AB-24.12 and ES-06.6 for the consumer endpoint map"]
    MODEL --- SEE
```

**What it shows.** Every path keeps a bare alias for parity with the retired Python façade. The README's endpoint table (`../loom/README.md:67`) lists the same surface without the aliases or the semantic debug route.

## LM-02.4 What each route reports about the generation it served

```mermaid
flowchart TB
    GEN["GET /loom/generation<br/>loom-facade/src/routes/mod.rs:148"]
    ID["identity: what was LOADED, immutable for the process<br/>loom-facade/src/routes/mod.rs:149,167"]
    DRIFT["drift: verify_atomicity re-hashes loaded artefacts per request<br/>loom-facade/src/routes/mod.rs:151,152,153"]
    DISK["disk: what the data directory says now, and matches_loaded<br/>loom-facade/src/routes/mod.rs:175,176,177"]
    FBK["fallback: null, or the descriptor of the retained bundle served<br/>loom-facade/src/routes/mod.rs:182"]
    GEN --> ID
    GEN --> DRIFT
    GEN --> DISK
    GEN --> FBK
    HDR["x-loom-generation, x-loom-content-digest, x-loom-atomicity-verified<br/>loom-facade/src/routes/mod.rs:271,274,275,277"]
    USE["added on /mcp, /loom/sparql, /loom/search<br/>loom-facade/src/routes/mod.rs:111,116,250,265"]
    HDR --> USE
    BODY["scaffold and chat carry the generation inside the grounding body instead<br/>loom-facade/src/routes/mod.rs:228,233"]
    HDR --- BODY
    INV["INVARIANT: reload is a process boundary, no reload endpoint exists,<br/>a newer disk marker only shows as matches_loaded false<br/>loom-facade/src/bundle.rs:25,26,27,28"]
    DISK --- INV
```

**What it shows.** Loaded identity, disk identity and integrity are three separate fields, so a promoted-but-not-restarted node is visible rather than mistaken for a serving change. LM-05.3 shows the reload tick that reads exactly this report.

## LM-02.5 Error to HTTP status: the lexical floor fails hard, accelerators degrade

```mermaid
flowchart TB
    E["api_error_parts<br/>loom-facade/src/error.rs:45"]
    C400["400 bad_query: malformed client request<br/>loom-facade/src/error.rs:50"]
    C503["503 no_backend: retrieval-only node asked to delegate<br/>loom-facade/src/error.rs:53"]
    C502["502 backend_unreachable or backend_http with upstream_status<br/>loom-facade/src/error.rs:57,58,60,61,63,64"]
    C502E["502 embed_error on an explicit embed surface<br/>loom-facade/src/error.rs:71,72"]
    C200["200 degraded: graph unavailable, semantic unready, artefact rejected<br/>loom-facade/src/error.rs:79,80"]
    C500["500 internal: index unavailable, generation drift, bundle, attest, io, json<br/>loom-facade/src/error.rs:86,87,88,89,91"]
    E --> C400
    E --> C503
    E --> C502
    E --> C502E
    E --> C200
    E --> C500
    DIV["DIVERGENCE: Python propagated the upstream status, the Rust table folds<br/>every backend failure into 502 and attaches the upstream body<br/>loom-facade/src/error.rs:7,8,9,11"]
    C502 --- DIV
    LIVE["OPEN: a 200 from /health can coexist with missing corpus data,<br/>liveness does not establish grounding<br/>../loom/README.md:40"]
    C200 --- LIVE
```

**What it shows.** The table encodes ADR-135's rule that a missing accelerator is never a client error. The chat path wraps backend failures with the grounding contract before this table is reached (see LM-04.1).

## LM-02.6 The governance ledger behind POST /loom/attest

```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant H as attest handler<br/>loom-facade/src/routes/attest.rs:113
    participant L as ChainedLedger attest<br/>loom-attest-proofgate/src/lib.rs:306
    participant D as ledger.jsonl and its .head file
    C->>H: case_id, digest, outcome, signer, at, loom-facade/src/routes/attest.rs:79,82,84,86,88
    H->>H: refuse an empty case_id or digest, loom-facade/src/routes/attest.rs:114,115
    H->>H: passed is true only for promote, loom-facade/src/routes/attest.rs:70,71,122
    H->>L: GateVerdict with the case as subject, loom-facade/src/routes/attest.rs:120,128,131
    L->>L: hold the write lock, read the tail, next seq and prev hash, loom-attest-proofgate/src/lib.rs:309,314,315
    L->>L: sha256 of prev hash then canonical JSON, loom-attest-proofgate/src/lib.rs:152,156,157
    L->>D: append one line, then tmp-and-rename the head checkpoint, loom-attest-proofgate/src/lib.rs:357,362,242,244
    L-->>H: entry id
    H-->>C: 201 with entry_id and chain_head, loom-facade/src/routes/attest.rs:149,150
    Note over L,D: verify_chain re-hashes every entry and requires the tail to equal the head, catching truncation, loom-attest-proofgate/src/lib.rs:367,372,376,389,399,403
    Note over L,D: OPEN: the hash is unkeyed and the head sits beside the ledger, so a writer of both files can rewrite them consistently, loom-attest-proofgate/src/lib.rs:152,155
    Note over L,D: DEBT: appends use write_all with no fsync, loom-attest-proofgate/src/lib.rs:357
```

**What it shows.** The ledger records who admitted what after the decision was made elsewhere; it is not an approval step (`loom-facade/src/routes/attest.rs:8,10,11`). Both profiles move it off the read-only data mount onto a named volume (`../loom/deploy/compose.profile-a.yml:27,98`).

## LM-02.7 The configuration surface: one Config read once from env, plus adapter-local reads

```mermaid
flowchart TB
    FE["Config from_env, every knob keeps its Python env name<br/>loom-facade/src/config.rs:9,115,116"]
    subgraph CFG["Config groups, loom-facade/src/config.rs:15"]
        T["transport: LOOM_FACADE_PORT, LOOM_TIMEOUT, LOOM_MAX_BODY_BYTES, LOOM_DEPLOY_PROFILE<br/>loom-facade/src/config.rs:17,18,19,20,118,119,120,121"]
        BK["backend: DISTILL_BACKEND_URL, LOOM_MIN_MAX_TOKENS<br/>loom-facade/src/config.rs:23,24,122,123"]
        SC["scaffold: ONTOLOGY_INDEX, ONTOLOGY_PROSE_INDEX, ONTOLOGY_BUDGET,<br/>LOOM_MAX_SEEDS, LOOM_SCAFFOLD_HOPS, LOOM_SCAFFOLD_PROSE<br/>loom-facade/src/config.rs:27,28,29,30,31,32,124,129"]
        SE["semantic: LOOM_SEMANTIC_FALLBACK, LOOM_HNSW_ARTIFACT, LOOM_SEMANTIC_K,<br/>LOOM_SEMANTIC_MIN_INJECT, LOOM_SEMANTIC_SCORE_SCALE, LOOM_SEMANTIC_DEBUG_SURFACE<br/>loom-facade/src/config.rs:35,36,37,38,39,88,130,135"]
        FX["F1 to F3: LOOM_VERBATIM_MODE, LOOM_VERBATIM_THRESHOLD, LOOM_EXPOSURE_APPEND,<br/>LOOM_BACKEND_NO_THINK, LOOM_THINK_TOKEN_FLOOR<br/>loom-facade/src/config.rs:45,54,60,67,81,139,143"]
        LG["ledger and fallback: LOOM_LEDGER_PATH, LOOM_RETAINED_BUNDLES_DIR<br/>loom-facade/src/config.rs:95,104,144,145"]
        JV["Jev: LOOM_JEV and the LOOM_JEV_* family<br/>loom-facade/src/config.rs:109,148"]
    end
    FE --> CFG
    subgraph OUT["read outside Config"]
        O1["gate: LOOM_CONFIDENCE_INJECTION and three thresholds<br/>loom-scaffold/src/policy.rs:61,68,69,70"]
        O2["embedder: XINFERENCE_URL<br/>loom-embed-xinference/src/lib.rs:74"]
        O3["SPARQL: LOOM_SPARQL_LIMIT, LOOM_SPARQL_MAX_ROWS<br/>loom-graph-oxigraph/src/lib.rs:275,276"]
        O4["artefact contract from env<br/>loom-vector-ruvector/src/hnsw.rs:117"]
    end
    CFG --> OUT
    INV["INVARIANT: max_seeds defaults to 4 and prose to false, the Python facade values,<br/>not the domain's 6 and true, or byte-parity breaks<br/>loom-facade/src/config.rs:4,5,6,7,175,177"]
    SC --- INV
    BOOL["truthiness: unset keeps the default, 0, false, no or empty is false<br/>loom-facade/src/config.rs:211,213,214,215"]
    FE --- BOOL
    DEBT["DEBT: the backend adapter re-reads DISTILL_BACKEND_URL, LOOM_TIMEOUT and<br/>LOOM_MIN_MAX_TOKENS itself, LOOM_TIMEOUT as a float where Config parses an integer<br/>loom-backend-openai/src/lib.rs:73,74,75,77<br/>loom-facade/src/config.rs:119"]
    BK --- DEBT
```

**What it shows.** Most behaviour is set by environment at start and fixed for the process; nothing is reloaded. The gate thresholds and the adapter endpoints live outside `Config`, so `/health` and `--build-info` are the places to read what a node actually resolved (`loom-facade/src/main.rs:20`).
