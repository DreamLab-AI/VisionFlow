---
id: LM-03
title: Retrieval — lexical match, the confidence gate, semantic fusion, SPARQL clamp and the MCP plane
area: loom
governing:
  - ../loom/docs/design/RUST-ARCHITECTURE.md
  - ../loom/docs/design/ADR-140-loom-v2-agent-drivable-grounding.md
adrs: [loom:ADR-136, loom:ADR-137, loom:ADR-138, loom:ADR-140]
sources:
  - ../loom/crates/loom-facade/src/fusion.rs
  - ../loom/crates/loom-facade/src/state.rs
  - ../loom/crates/loom-facade/src/config.rs
  - ../loom/crates/loom-facade/src/lib.rs
  - ../loom/crates/loom-facade/src/mcp_host.rs
  - ../loom/crates/loom-facade/src/routes/mod.rs
  - ../loom/crates/loom-scaffold/src/lib.rs
  - ../loom/crates/loom-scaffold/src/match_.rs
  - ../loom/crates/loom-scaffold/src/policy.rs
  - ../loom/crates/loom-scaffold/src/tuning.rs
  - ../loom/crates/loom-scaffold/src/serialise.rs
  - ../loom/crates/loom-scaffold/src/manifest.rs
  - ../loom/crates/loom-graph-oxigraph/src/lib.rs
  - ../loom/crates/loom-mcp/src/lib.rs
  - ../loom/crates/loom-facade/src/bundle.rs
  - ../loom/docs/design/ADR-140-loom-v2-agent-drivable-grounding.md
verified_commit: {loom: 355946dd2deacfdef21171d53eafd025887e447e}
---

## LM-03.1 The fusion pipeline: lexical first, semantic only on a miss, one gate for both

```mermaid
flowchart TB
    Q["build_scaffold(query, opts)<br/>loom-facade/src/fusion.rs:34"]
    LEX["LEXICAL PRIMARY: retriever.seeds, errors propagate<br/>loom-facade/src/fusion.rs:42"]
    GATE{"top score at or above<br/>policy.min_inject_score?<br/>loom-facade/src/fusion.rs:47"}
    HIT["assemble with FusionPath LexicalHit, no embed call<br/>loom-facade/src/fusion.rs:48,50"]
    EN{"LOOM_SEMANTIC_FALLBACK on<br/>and the HNSW index ready?<br/>loom-facade/src/fusion.rs:56"}
    GEN{"semantic generation equals<br/>lexical generation?<br/>loom-facade/src/fusion.rs:58,59,60"}
    CAND["embed, nearest k, union-dedupe by IRI, lexical first<br/>loom-facade/src/fusion.rs:111,118,125,131"]
    THR{"LOOM_SEMANTIC_MIN_INJECT set and the<br/>scaled best score clears it?<br/>loom-facade/src/fusion.rs:75,76"}
    SEM["assemble with FusionPath SemanticFallback, same gate<br/>loom-facade/src/fusion.rs:77,82"]
    NONE["empty scaffold with this node's threshold<br/>loom-facade/src/fusion.rs:95,98"]
    Q --> LEX --> GATE
    GATE -->|yes| HIT
    GATE -->|no| EN
    EN -->|no| NONE
    EN -->|yes| GEN
    GEN -->|no, warn never-mixed-build| NONE
    GEN -->|yes| CAND
    CAND -->|embed or index error| NONE
    CAND --> THR
    THR -->|no| NONE
    THR -->|yes| SEM
    INV["INVARIANT I-P1: HNSW hits are candidate seeds handed back into assemble,<br/>no vector, cosine or summary is ever served<br/>loom-facade/src/fusion.rs:11,12,13"]
    SEM --- INV
    DEF["fallback default off, no default inject threshold<br/>loom-facade/src/config.rs:35,38,178,181"]
    EN --- DEF
    DRIFT["DOC-DRIFT: semantic_fallback_enabled is documented as switch AND ready,<br/>the body returns the switch alone, the caller adds is_ready<br/>loom-facade/src/state.rs:186,187,190"]
    EN --- DRIFT
```

**What it shows.** An accelerator failure degrades to no-match instead of an error; only the lexical floor can fail the request (`loom-facade/src/fusion.rs:17,19,20`). With the default thresholds the semantic branch is reachable only when no lexical seed survives.

## LM-03.2 The lexical matcher and the gate arithmetic

```mermaid
flowchart TB
    W["words of the prompt, stopwords and 1-char terms dropped<br/>loom-scaffold/src/match_.rs:23,30"]
    EX["exact title or slug n-gram, up to 4 words, +8.0 per word<br/>loom-scaffold/src/match_.rs:37,51<br/>loom-scaffold/src/tuning.rs:11,15"]
    OV["title-word overlap via the inverted index, 2.0 split over the title<br/>loom-scaffold/src/match_.rs:59,61<br/>loom-scaffold/src/tuning.rs:12"]
    SB["slug substring for terms of 5 chars or more, +0.75<br/>loom-scaffold/src/match_.rs:71,75,76<br/>loom-scaffold/src/tuning.rs:13,14"]
    SEED["seeds at or above MIN_SEED_SCORE 2.0, sorted score, quality, slug<br/>loom-scaffold/src/match_.rs:84,86<br/>loom-scaffold/src/tuning.rs:10"]
    W --> EX --> OV --> SB --> SEED
    DEC["decide(top, requested, gate)<br/>loom-scaffold/src/policy.rs:102"]
    SEED --> DEC
    OFF["injection off: Full, the requested budget<br/>loom-scaffold/src/policy.rs:107,108"]
    SKIP["no score, or top below min_inject_score: Skipped, no block<br/>loom-scaffold/src/policy.rs:110,111,113,114"]
    SCALE["frac = min 1, max of min_inject_fraction and top over strong_match_score<br/>budget = max 1, floor of requested times frac, Full when frac is 1 else Scaled<br/>loom-scaffold/src/policy.rs:116,118,122,123,128"]
    DEC --> OFF
    DEC --> SKIP
    DEC --> SCALE
    ENV["four env knobs, defaults 0, 8.0, 2.0, 0.4<br/>loom-scaffold/src/policy.rs:61,68,69,70<br/>loom-scaffold/src/tuning.rs:22,23,24"]
    DEC --- ENV
    DEBT["DEBT: the gate is read from env twice, once inside the retriever and<br/>once for AppState, and the two must agree<br/>loom-scaffold/src/lib.rs:223,246<br/>loom-facade/src/lib.rs:266"]
    ENV --- DEBT
    INV["INVARIANT: scoring weights are compile-time constants for Python byte-parity<br/>loom-scaffold/src/tuning.rs:3,4,5"]
    EX --- INV
```

**What it shows.** Confidence is a lexical additive score, not a probability: one exact single-word title hit scores 8.0, which is also the strong-match default. The gate only ever shrinks or withholds the requested budget.

## LM-03.3 Assembling the served block: sections, prose for the top two, clamp from the end

```mermaid
flowchart TB
    AS["LexicalIndex assemble, the query argument is unused<br/>loom-scaffold/src/lib.rs:398,400,404"]
    AB["assemble_block<br/>loom-scaffold/src/lib.rs:98"]
    G["decide on the top seed<br/>loom-scaffold/src/lib.rs:110,111"]
    NO["Skipped: empty block, grounding still records the score<br/>loom-scaffold/src/lib.rs:114,116,121"]
    SEC["one section per seed, prose only for the first PROSE_SEEDS<br/>loom-scaffold/src/lib.rs:133,137,142<br/>loom-scaffold/src/tuning.rs:27"]
    CL["clamp: drop whole sections from the END until the wrapped text fits<br/>loom-scaffold/src/serialise.rs:172,174,176,177,180"]
    OUT["Scaffold with block, seeds, top score, budget, fusion path, generation<br/>loom-scaffold/src/lib.rs:426,428,434"]
    AS --> AB --> G
    G -->|no budget| NO
    G -->|budget| SEC --> CL --> OUT
    WRAP["block is wrapped in ONTOLOGY CONTEXT header and footer<br/>loom-scaffold/src/tuning.rs:32,33"]
    CL --- WRAP
    DEBT["DEBT: compiled-in default index paths point at the legacy app/data directory<br/>loom-scaffold/src/tuning.rs:69,71"]
    AB --- DEBT
    SEE["see LM-04.1 for how the chat path merges this block into messages"]
    OUT --- SEE
```

**What it shows.** Seeds are kept whole or dropped; the clamp never truncates mid-section, so the lowest-ranked seeds go first. The facade overrides the default paths from `ONTOLOGY_INDEX` (`loom-facade/src/config.rs:124,172`).

## LM-03.4 SPARQL and label search: the read-only clamp over the published graph

```mermaid
flowchart TB
    LOAD["load: allowlist ontology.ttl and ontology-inferred.ttl, never a glob<br/>loom-graph-oxigraph/src/lib.rs:38,272,289"]
    V["validate: refuse update verbs and SERVICE<br/>loom-graph-oxigraph/src/lib.rs:50,176"]
    C["clamp: inject LIMIT on an un-LIMITed SELECT, default 10000<br/>loom-graph-oxigraph/src/lib.rs:41,198,207"]
    ROWS["row cap LOOM_SPARQL_MAX_ROWS, default 10000<br/>loom-graph-oxigraph/src/lib.rs:43,271"]
    LOAD --> V --> C --> ROWS
    FAIL["absent store: available false, GraphUnavailable, a labelled 200 upstream<br/>loom-graph-oxigraph/src/lib.rs:13,14,15"]
    LOAD --- FAIL
    INV["INVARIANT BC24 I11: the Loom serves the published ontology only<br/>loom-graph-oxigraph/src/lib.rs:9,10,11"]
    LOAD --- INV
    SEC["INVARIANT: the LIMIT clamp is a security control, not a parity feature<br/>loom-graph-oxigraph/src/lib.rs:69,70"]
    C --- SEC
```

**What it shows.** The graph is in-memory, read-only and loaded from exactly two files; both `/loom/sparql` and the MCP `loom.sparql` tool reach the same `GraphStore::query` (`loom-facade/src/routes/mod.rs:249`, `loom-facade/src/mcp_host.rs:195`).

## LM-03.5 The MCP plane: one JSON-RPC adapter, the facade as tool host

```mermaid
sequenceDiagram
    autonumber
    participant A as agent client
    participant R as mcp route<br/>loom-facade/src/routes/mod.rs:95
    participant D as dispatch<br/>loom-mcp/src/lib.rs:102
    participant H as ToolHost for AppState<br/>loom-facade/src/mcp_host.rs:93
    A->>R: POST /mcp, one message or a batch, loom-facade/src/routes/mod.rs:100,103
    R->>D: dispatch, loom-facade/src/routes/mod.rs:115
    D->>D: notification without id gets no answer, loom-mcp/src/lib.rs:107,108
    D->>D: initialize, ping, tools/list answered locally, loom-mcp/src/lib.rs:116,117,118
    D->>H: tools/call, loom-mcp/src/lib.rs:119
    H->>H: loom.browse returns addresses via the lexical port, loom-facade/src/mcp_host.rs:121,127
    H->>H: loom.resolve joins slug words into a query and runs build_scaffold, loom-facade/src/mcp_host.rs:154,159
    H->>H: loom.sparql uses the clamped graph query, loom-facade/src/mcp_host.rs:195
    H->>H: neighbours and paths are bounded walks, MAX_VISITED 4000, loom-facade/src/mcp_host.rs:32,202,225
    H-->>D: payload plus the grounding envelope
    D-->>R: JSON-RPC result or error code
    R-->>A: response with x-loom-generation headers, loom-facade/src/routes/mod.rs:116
    Note over H: TENSION: resolve is documented as turning addresses into seeds, but it re-matches the slug text, so the block holds whatever the matcher finds, found is computed apart, loom-facade/src/mcp_host.rs:150,163
    Note over D: DOC-DRIFT: lib.rs still names a stdio shim, ADR-140 records it removed the same day, loom-mcp/src/lib.rs:16, ADR-140-loom-v2-agent-drivable-grounding.md:611
```

**What it shows.** The adapter crate holds framing and schemas only (`loom-mcp/src/lib.rs:3,4,5`); all retrieval goes through the facade's ports, so the agentic plane cannot acquire its own policy. See AB-24.12 for the consumer side.

## LM-03.6 The session manifest: statements about the corpus, never content

```mermaid
flowchart TB
    MAN["loom.manifest over the loaded ScaffoldIndex, pure, O(classes)<br/>loom-scaffold/src/manifest.rs:1,2,9"]
    SAL["salience list: Iri, title, domain, quality addresses only<br/>loom-scaffold/src/manifest.rs:5,6,7"]
    SRC["ManifestSource is the same object as the retriever<br/>loom-scaffold/src/lib.rs:526,527"]
    MAN --> SAL
    SRC --> MAN
    DRIFT["DOC-DRIFT: the manifest cost note speaks of a hot-reloaded generation,<br/>while reload is a process restart<br/>loom-scaffold/src/manifest.rs:11,12<br/>loom-facade/src/bundle.rs:25,26,27"]
    MAN --- DRIFT
```

**What it shows.** An agent learns what the corpus covers before fetching; the content itself still comes only through `loom.resolve` and the gate.
