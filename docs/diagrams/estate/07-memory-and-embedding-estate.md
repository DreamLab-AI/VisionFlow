---
id: ES-07
title: RuVector memory and embedding estate
area: estate
governing:
  - ../project/agentbox/docs/LEARNING-memory.md
  - ../project/docs/DATA-authority-erasure.md
adrs: [agentbox:ADR-2014, agentbox:ADR-2015, agentbox:ADR-2016, agentbox:ADR-2082]
sources:
  - ../project/agentbox/mcp/servers/ruvector-mcp.cjs
  - ../project/agentbox/scripts/ruvector-recall-harness.mjs
  - ../project/agentbox/scripts/ruvector-sona-feeder.mjs
  - ../project/agentbox/scripts/ruvector-aggregate-sweep.mjs
  - ../project/agentbox/scripts/ruvector-pattern-distill.mjs
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/README.md
  - ../project/agentbox/tests/contract/ruvector-gates.contract.spec.js
  - ../project/agentbox/docs/adr/ADR-2014-memory-mcp-only-fail-closed.md
  - ../project/src/handlers/memory_flash_handler.rs
  - ../project/src/actors/agent_monitor_actor.rs
verified_commit: {visionclaw: 7d3ea2edb067432a57e6fe1fd951fd8254380bb8, agentbox: 5ab197a9d49e9721b85b791bf9efe30842c9e047}
---
## ES-07.1 Every RuVector client and the one shared embedder
```mermaid
flowchart TB
    subgraph clients["Clients — all writes MUST go through the MCP surface"]
        MCP["agentbox/mcp/servers/ruvector-mcp.cjs<br/>memory_store / memory_retrieve / memory_list / memory_search<br/>lines 253,267,279,290"]
        HOOKS["agentbox hooks + skills<br/>route via the same MCP server"]
        SWEEP["ruvector-aggregate-sweep.mjs"]
        DISTILL["ruvector-pattern-distill.mjs"]
        SONA["ruvector-sona-feeder.mjs"]
        HARNESS["ruvector-recall-harness.mjs"]
        VCMF["VisionClaw memory_flash_handler<br/>src/handlers/memory_flash_handler.rs:41<br/>OBSERVER ONLY — broadcasts access events"]
        AMA["agent_monitor_actor.rs:496-500<br/>narrates RuVector Memory Specialist activity"]
    end
    subgraph store["ruvector-postgres"]
        PG["host=ruvector-postgres port=5432<br/>dbname=ruvector user=ruvector<br/>$RUVECTOR_PG_CONNINFO — ruvector-mcp.cjs:48-57"]
        HNSW["HNSW index over 384-dim vectors"]
    end
    subgraph embed["Shared embedder"]
        XI["xinference /v1/embeddings<br/>$XINFERENCE_ENDPOINT default http://xinference:9997<br/>ruvector-mcp.cjs:92"]
        MODEL["bge-small-en-v1.5<br/>EMBEDDING_DIM = 384 — ruvector-mcp.cjs:93-94"]
    end

    MCP --> PG
    HOOKS --> MCP
    SWEEP --> PG
    DISTILL --> PG
    SONA --> PG
    HARNESS --> PG
    PG --> HNSW
    MCP -- "client-side embed before write" --> XI
    XI --> MODEL

    INV1["INVARIANT — agent access is the governed memory MCP ONLY.<br/>This container exposes agentbox-memory; server aliases differ.<br/>The claude-flow CLI and raw SQL INSERT bypass the embedding<br/>pipeline, so rows written that way are INVISIBLE to HNSW search."]
    INV2["INVARIANT — ruvector-mcp.cjs FAILS CLOSED with no sql.js<br/>fallback: cannot reach ruvector-postgres is FATAL<br/>(ruvector-mcp.cjs:158)"]
    DIV1["DIVERGENCE — VisionClaw has NO RuVector write client. Despite<br/>agent-facing narration, the only Rust touchpoint is the<br/>memory-flash WS broadcast (src/handlers/memory_flash_handler.rs:41,<br/>observational only, never embeds). There is no RuVectorAdapter<br/>type in src/ or crates/ (verified by grep)."]
    SCOPE["INVARIANT — pubkey scoping: NIP-98 callers are scoped to their<br/>own pubkey namespace (user:&lt;pubkey&gt;:proj:&lt;repo-slug&gt;:&lt;ns&gt;); a<br/>session cannot read another session's per-project namespace<br/>— agentbox.toml:358-370"]
    NS5["code-harness-lessons namespace — expel_lesson_extraction distils<br/>0-N rules per completed trajectory as ex:DistilledLesson<br/>(memory_type=semantic, durable) — agentbox.toml:612-614"]
    DIV3["DIVERGENCE — the harness's file-based auto-memory<br/>(~/.claude/projects/.../memory/, MEMORY.md) is INVISIBLE to this<br/>path and to every other agent in the mesh."]

    MCP --> INV1
    PG --> INV2
    VCMF --> DIV1
    MCP --> SCOPE
    HOOKS --> NS5
    MCP --> DIV3
```

## ES-07.2 memory_store — embed-then-write, fail-closed when the embedder is down
```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant M as ruvector-mcp.cjs<br/>memory_store:253
    participant X as xinference port 9997<br/>bge-small-en-v1.5
    participant P as ruvector-postgres port 5432
    participant H as HNSW index

    A->>M: memory_store{namespace, key, value, ttl}
    Note over M: entryId = `${WRITE_SOURCE_TYPE}:${namespace}:${key}`<br/>ruvector-mcp.cjs:130 — namespace defaults to "default"
    M->>X: POST /v1/embeddings (client-side embed)
    alt xinference reachable
        X-->>M: 384-dim vector
        M->>P: upsert row + vector
        P->>H: index insert
        H-->>P: ok
        P-->>M: stored
        M-->>A: ok
    else xinference unavailable
        X--xM: connect error
        Note over M,P: ADR-2014 FAIL-CLOSED — the store is REJECTED.<br/>A row without an embedding would be permanently<br/>invisible to semantic search (ruvector-mcp.cjs:166)
        alt RUVECTOR_EMBED_REPAIR=true
            M->>P: accept as a repairable PENDING write
            P-->>M: pending, awaiting later repair
            M-->>A: accepted pending
        else default
            M-->>A: REJECT — store refused until xinference returns
        end
    end
    Note over A,H: EMBED CAP — bge-small embeds only the first ~512 tokens<br/>(~2,500 chars) of a value. The tail is invisible to search.<br/>Keep values under ~2,000 chars and front-load the facts.<br/>Retrieve-by-key still returns the WHOLE value.
```

## ES-07.3 memory_search — HNSW semantic path with an ILIKE degradation
```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant M as ruvector-mcp.cjs<br/>memory_search:290
    participant X as xinference port 9997
    participant P as ruvector-postgres
    participant H as HNSW index

    A->>M: memory_search{query, namespace, limit}
    M->>X: embed(query)
    alt embedder healthy
        X-->>M: 384-dim query vector
        M->>P: namespace-scoped pgvector search
        P->>H: HNSW top-k
        H-->>P: candidates
        P-->>M: ranked rows
        M-->>A: semantic results (~100ms typical)
    else embedder down
        X--xM: error
        Note over M,P: DEGRADED — search falls back to ILIKE<br/>(substring match, no semantics) — ruvector-mcp.cjs:166
        M->>P: ILIKE scan
        P-->>M: literal matches only
        M-->>A: degraded results
    end
    opt reconnect probe
        M->>X: getEmbedding("reconnect probe")
        Note over M,X: recall-harness mirrors this probe at<br/>ruvector-recall-harness.mjs:173-174 — one probe<br/>flips xinferenceOk back to true
    end
    Note over A,H: namespace "*" performs a global cross-namespace search.<br/>AVOID memory_hybrid_search on large namespaces — it<br/>materialises the whole namespace (~72s on ruvnet-kb).
```

## ES-07.7 Recall gate — the band that must hold before and after any retrieval change
```mermaid
flowchart TB
    RUN["./agentbox.sh ruvector recall<br/>ruvector-recall-harness.mjs"]
    SELF["self-recall@10 — 200 rows<br/>the row's OWN stored embedding is the query<br/>SELF_NS_MIN_ROWS = 50 eligible rows per namespace<br/>agentbox/scripts/ruvector-recall-harness.mjs:15-17,80"]
    TRUE["true-recall@10 — 120 rows vs a forced exact<br/>brute-force scan as ground truth<br/>TRUE_TOTAL = 120, TRUE_NS_MIN_ROWS = 20<br/>agentbox/scripts/ruvector-recall-harness.mjs:18-20,82-83"]
    GATE["PASS iff median(self) >= 175/200<br/>AND median(true) >= 102/120 AND exactOk<br/>agentbox/scripts/ruvector-recall-harness.mjs:32, evaluated at agentbox/scripts/ruvector-recall-harness.mjs:228-236"]
    NSB["Per-namespace self-recall breakdown is surfaced<br/>but NOT gated — agentbox/scripts/ruvector-recall-harness.mjs:34"]
    D3["DIVERGENCE D3 (LEARNING-memory.md) — the harness gates<br/>true at >= 102/120 (agentbox/scripts/ruvector-recall-harness.mjs:32) while agentbox/CLAUDE.md<br/>and the reference doc quote >= 107/120.<br/>CODE IS AUTHORITATIVE for the gate; the prose band is a<br/>tighter operational target. self >= 175/200 agrees in both."]

    RUN --> SELF
    RUN --> TRUE
    SELF --> GATE
    TRUE --> GATE
    GATE --> NSB
    GATE --> D3
```

## ES-07.9 Learning loop — the gates that are off, and why
```mermaid
flowchart TB
    subgraph feeder["ruvector-sona-feeder.mjs"]
        F1["streams judged trajectories into<br/>ruvector_sona_learn under fixed 384-dim<br/>scope agentbox_memory"]
    end
    subgraph gates["agentbox.toml gates"]
        G1["sona_learn_enabled = OFF<br/>sona_apply_enabled = OFF<br/>agentbox.toml:470-471"]
        G2["attention_rerank = OFF<br/>agentbox.toml:469"]
        G3["pattern_distillation = true<br/>ENABLED 2026-07-21, 13 patterns live<br/>provenance judge:trajectory<br/>agentbox.toml:468"]
        G4["allow_namespace_repair = false<br/>agentbox.toml:480"]
        G5["allow_pattern_graduation = false RESERVED<br/>agentbox.toml:486"]
    end
    D1["DIVERGENCE D1 — SONA is INERT. The prebuilt<br/>@ruvector/sona@0.1.5 NAPI binary hardcodes<br/>embedding_dim = 256, so 384-dim learns return<br/>status:learned but accumulate NOTHING (verified live).<br/>Both gates stay off until a 384-dim-capable binary."]
    D1B["attention_rerank is OFF BY MEASUREMENT, not caution —<br/>on an L2-normalised corpus the attention blend is a<br/>mathematical identity (max diff 4e-7)."]
    D2["DIVERGENCE D2 — aggregate-count drift. agentbox.toml:455<br/>cites 78 aggregates >=20 samples (2026-08-31); the<br/>reference doc records 12 from the 2026-07-21 sweep.<br/>The toml is the running config and the newer number."]
    D4["DOC-DRIFT D4 — agentbox/README.md:330 still lists BOTH<br/>feed_retrieval and feed_routing as open gates awaiting the<br/>Wilson floor. Half of that is now stale: the running manifest<br/>has feed_retrieval = true since 2026-08-31 (agentbox.toml:455)<br/>and only feed_routing is still false (agentbox.toml:456)."]
    D5["DIVERGENCE D5 — pod-sync deletion has NO reverse<br/>tombstone. deleteAgentMemory() in the Pod does not revoke<br/>the RuVector-held agent memory: the embedding row persists<br/>and stays semantically searchable. Largest erasure hole.<br/>No point-in-time RuVector backup exists, so there is no<br/>cross-store consistent restore, RPO or RTO."]

    F1 --> G1
    G1 --> D1
    G2 --> D1B
    G3 --> D2
    G3 --> D4
    G4 --> D5
    G5 --> D5
```

## Audit qualification — 2026-09-07

The 2026-09-05 Agentbox recall closeout records self 189/200 and true 115/120 after a **serial** rebuild, versus self 151/200 after parallel rebuilding. These are historical receipts, not a fresh benchmark. The local RuVector source contains `hnsw_bulkdelete` and deleted-node flags; neither proves a deployed bulk-delete recall regression. Keep cross-store reverse tombstones (ADR-2060) distinct from index tombstones. No database mutation or reindex was performed. See [audit](../../estate-review/2026-09-07-agentbox-audit.md).

## ES-07.10 The governed server also fronts the orchestration tools, and memory is denied on that side
```mermaid
sequenceDiagram
    autonumber
    participant AG as an agent session
    participant RV as ruvector-mcp.cjs<br/>agentbox/mcp/servers/ruvector-mcp.cjs:30
    participant PROXY as createOrchestrationProxy<br/>agentbox/mcp/servers/ruvector-mcp.cjs:30
    participant MAN as agentbox.toml orchestration gate<br/>agentbox/agentbox.toml:430

    AG->>RV: memory_store / memory_search / memory_list / memory_retrieve
    RV-->>AG: served locally against ruvector-postgres,<br/>ruvector-mcp.cjs:566,614,617,620

    AG->>RV: swarm / agent / task / coordination tools
    RV->>MAN: is orchestration_proxy on
    MAN-->>RV: true, but the forwarded categories are ONLY<br/>swarm, agent — task and coordination were dropped<br/>2026-09-25 (unused, ~185 tok/tool), agentbox.toml:431
    RV->>PROXY: forward a swarm/agent call to ONE filtered<br/>ruflo child per session — task/coordination stay stubs
    alt the child is reachable
        PROXY-->>AG: the real ruflo result
    else the child is unavailable
        PROXY-->>AG: an honest stub, ok false and error unimplemented,<br/>ruvector-mcp.cjs:648,659
    end

    Note over RV,PROXY: INVARIANT — memory_* is NEVER forwarded to the proxy child.<br/>This server backs the memory tools itself and says so in the<br/>stub descriptions it ships, ruvector-mcp.cjs:306-311.<br/>The ADR-2014 access invariant is therefore unchanged.
    Note over RV: INVARIANT — the server replaces claude-flow mcp start so that<br/>every memory call goes through the embedding pipeline rather<br/>than round the side of it, ruvector-mcp.cjs:6. see AB-09
    Note over MAN: DEBT — the gate is a manifest boolean with no runtime probe,<br/>so an operator reading agentbox.toml:430 learns the intent and<br/>not whether a ruflo child is actually answering. The honest<br/>stub is the only signal, and it looks the same as a tool that<br/>was never implemented.
```
