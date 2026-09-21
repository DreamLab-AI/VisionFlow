---
id: AB-24
title: Ontology Loom facade and the model-swap seam
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2023, ADR-2053, ADR-2055, ADR-2075, ADR-2084]
sources:
  - ../project/agentbox/mcp/servers/lib/ontology-retrieval.js
  - ../project/agentbox/docs/adr/ADR-2084-one-published-loom-client-for-every-facade-caller.md
  - ../project/agentbox/services/dream-engine/src/llm.rs
  - ../project/agentbox/services/podcast-ingest/src/ingest/loom.rs
  - ../project/agentbox/services/podcast-ingest/src/promote/loom.rs
  - ../project/agentbox/services/explainer-tools/src/bin/loom_draft.rs
  - ../project/agentbox/services/explainer-tools/src/draft.rs
  - ../project/agentbox/services/agentbox-mcp/src/web_summary/llm.rs
  - ../loom/docs/design/ADR-139-per-request-scaffold-opt-out.md
  - ../project/agentbox/scripts/aoe-seed-sessions.mjs
  - ../project/agentbox/mcp/servers/lib/ontology-budget.js
  - ../project/agentbox/agentbox.toml
  - ../project/docker-compose.unified.yml
  - ../project/loom/README.md
  - ../project/agentbox/scripts/opf-router.py
  - ../project/agentbox/mcp/servers/lib/ontology-telemetry.js
  - ../project/agentbox/flake.nix
verified_commit: {agentbox: 1639f86abded1441ce148d6c47924dfaf34f96af, visionclaw: f223bbd40ab52f7848d38ff98211ece75456b7e2, loom: 39b5fc02aeca0abd9f12c32437a69b6e385d8375}
---

## AB-24.1 Two deployments of one facade contract — topology

```mermaid
flowchart TB
    subgraph consumers["Consumers hold a DOOR, never a raw model port (ADR-2023)"]
        RET["ontology-retrieval brain<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:734"]
        COND["ontology condense<br/>agentbox/agentbox.toml:805"]
        DREAM["dream-engine loom_url<br/>agentbox/agentbox.toml:2004"]
        SEED["AoE session seed slug=loom<br/>agentbox/agentbox.toml:1645"]
        SEEDRAW["AoE session seed slug=loom-raw #40;LEGACY ALIAS#41;<br/>agentbox/agentbox.toml:1652"]
        EMAIL["email gateway REASONER_BASE_URL<br/>see AB-27"]
    end
    subgraph depA["Deployment A — LAN facade on machinelearn .132"]
        F84["Loom facade<br/>machinelearn .132, port 8084, path /v1"]
    end
    subgraph depB["Deployment B — sidecar on visionclaw_network (compose profile loom)"]
        SIDE["loom-facade (Rust)<br/>docker-compose.unified.yml:298"]
        TMPFS["tmpfs /run/loom mode=0750 uid=65532<br/>docker-compose.unified.yml:343"]
        DATA["loom-data :ro generation<br/>docker-compose.unified.yml:338"]
    end
    subgraph model["The model — an operational detail BEHIND the door"]
        M85["loom-model port 8085 qwen3.8-27B<br/>DISTILL_BACKEND_URL"]
    end
    RET -->|"LOOM_FACADE_URL"| F84
    COND -->|"POST /v1/chat/completions"| F84
    DREAM -->|"llm_provider=loom only"| F84
    SEED -->|"model loom-lan/qwen3.8-27B"| F84
    SEEDRAW -->|"openCodeConfig points it at LOOM_BASE_URL with loom_options scaffold false<br/>aoe-seed-sessions.mjs:254, aoe-seed-sessions.mjs:227"| F84
    EMAIL -->|"http://loom:8080/v1"| SIDE
    F84 -->|"ml DNATs over the 25G rail 10.10.10.0/30"| M85
    SIDE -->|"DISTILL_BACKEND_URL blank = retrieval-only, /v1 returns 503"| M85
    DATA --> SIDE
    SIDE -->|"entrypoint copies .rvdb off :ro — opening redb mutates it"| TMPFS
    subgraph notes["Invariants and drift"]
        direction TB
        N1["WITHDRAWN in code: the loom-raw SEED no longer holds a raw model door. openCodeConfig<br/>gives loom-lan, loom-agent and loom-raw the SAME LOOM_BASE_URL #40;aoe-seed-sessions.mjs:213,<br/>aoe-seed-sessions.mjs:248, aoe-seed-sessions.mjs:254#41; and declines the scaffold per request<br/>instead #40;aoe-seed-sessions.mjs:227#41;, the ADR-139 answer to what the raw port was for.<br/>LOOM_RAW_BASE_URL survives only as a compose default #40;flake.nix:3173#41; that nothing seeds"]
        N2["RESOLVED ADR-2055: opf-router is the PRIVACY-FILTER redaction sidecar on OPF_PORT<br/>9092 (agentbox.toml [privacy_filter].port, scripts/opf-router.py:41, flake.nix<br/>[program:opf-router]). BASELINE-container previously described it as an<br/>OpenAI-compatible facade on port 8084 — corrected. No agentbox program serves<br/>port 8084 — that is the Loom facade on machinelearn"]
        N3["The loom-facade implementation lives OUTSIDE this repo at /home/devuser/workspace/loom.<br/>This repo holds the deployment contract only (loom/README.md:8-15)"]
        N1 ~~~ N2 ~~~ N3
    end
```

## AB-24.2 selectBackend — which store answers, and why

```mermaid
sequenceDiagram
    autonumber
    participant CALL as createDefaultRetrieval<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:734
    participant SEL as selectBackend<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:491
    participant ENV as process.env
    participant LF as makeLoomFetch<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:616
    participant VF as makeVcFetch<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:516
    participant TEL as createTelemetrySink<br/>agentbox/mcp/servers/lib/ontology-telemetry.js

    CALL->>SEL: selectBackend(opts, env)
    SEL->>ENV: read LOOM_FACADE_URL / VISIONCLAW_API_URL / LOOM_GENERATION
    Note over SEL,ENV: generation falls back LOOM_GENERATION then ONTOLOGY_GENERATION then null
    alt loomUrl set AND no injected vcFetch
        SEL-->>CALL: name=loom url=loomUrl configured=true reason=LOOM_URL_SET
        CALL->>LF: makeLoomFetch(opts)
        CALL->>TEL: canary() — startup liveness probe, loud on failure, fail-open
    else vcUrl empty
        SEL-->>CALL: name=none url=null configured=false reason=NOT_CONFIGURED
    else loomUrl set BUT vcFetch injected
        SEL-->>CALL: name=visionclaw reason=VC_FETCH_INJECTED
        CALL->>VF: makeVcFetch(opts) — tests / deliberate pinning
    else loomUrl unset
        SEL-->>CALL: name=visionclaw reason=LOOM_URL_UNSET
        CALL->>VF: makeVcFetch(opts) — DEFAULT_API http://visionclaw-server:4000
    end
    Note over SEL: INVARIANT: an unset LOOM_FACADE_URL is the ordinary VisionClaw path and NOT a fault — a<br/>CONFIGURED Loom that is unreachable is an operational fault
    Note over CALL: One brain = this module plus the shared backing stores, NOT one process
```

## AB-24.3 Loom-backed retrieval — /loom/search seed then /loom/sparql expand

```mermaid
sequenceDiagram
    autonumber
    participant CALL as ontology_ask
    participant ASK as ask<br/>ontology-retrieval.js:280
    participant VERIFY as loomGenerationVerifier<br/>ontology-retrieval.js:665
    participant FETCH as makeLoomFetch<br/>ontology-retrieval.js:616
    participant LOOM as selected Loom service
    CALL->>ASK: query, constraints, optional generation pin
    ASK->>VERIFY: verify before cache lookup
    VERIFY->>LOOM: GET /loom/generation
    alt missing, drifted or mismatched identity/model/corpus
        VERIFY-->>ASK: refusal
        ASK-->>CALL: empty labelled generation degradation, no fallback
    else verified loaded bundle
        VERIFY-->>ASK: digest plus generation and embedding contract
        ASK->>ASK: cache lookup with verified discriminator
        alt cache miss
            ASK->>FETCH: POST /loom/search
            FETCH->>LOOM: query
            LOOM-->>FETCH: existing body plus loaded identity headers
            FETCH-->>ASK: normalised hits, response identity
            ASK->>ASK: require matching response bundle and maturity/domain gates
            opt expansion requested
                ASK->>FETCH: child then outgoing POST /loom/sparql
                LOOM-->>FETCH: rows plus loaded identity headers
                FETCH-->>ASK: require matching identity on each response
            end
            ASK->>ASK: mismatch discards result otherwise serialise and budget clamp
        end
        ASK-->>CALL: scoped result with verified cache generation
    end
    Note over ASK,LOOM: Source staged. The old live API and mixed graph/semantic generation are refused.<br/>Default-graph provenance limitation remains separate from identity verification.
```

## AB-24.4 Cache key completeness and cache-hit constraint revalidation

```mermaid
sequenceDiagram
    participant ASK as ask<br/>ontology-retrieval.js:280
    participant VERIFY as loomGenerationVerifier<br/>ontology-retrieval.js:665
    participant KEY as cacheKey<br/>ontology-retrieval.js:75
    participant CACHE as TTL cache
    participant SAT as cacheEntrySatisfies<br/>ontology-retrieval.js:158
    ASK->>VERIFY: verify selected Loom before reading cache
    VERIFY-->>ASK: loaded graph/semantic identity plus embedding contract
    Note over ASK,VERIFY: No valid identity means no cached answer.<br/>Configured generation is compared as a pin, never used to relabel bytes.
    ASK->>KEY: all effective request constraints
    Note over KEY: generation discriminator hashes loaded generation, content digest,<br/>embedding model and dimensions, key also includes domain, budget and scope
    KEY-->>ASK: cache key
    ASK->>CACHE: get key
    alt hit
        CACHE-->>ASK: stored result and constraints
        ASK->>SAT: compare current constraints and budget
        alt matching
            ASK-->>ASK: replay result with preserved degradation state
        else mismatch
            ASK-->>ASK: re-retrieve, never truncate a previously clamped graph
        end
    else miss
        ASK-->>ASK: retrieve and verify response identity
    end
```

## AB-24.5 Backend, stage and outcome vocabularies

```mermaid
classDiagram
    class BACKENDS {
        <<frozen enum>>
        +LOOM "loom"
        +VISIONCLAW "visionclaw"
        +INJECTED "injected"
        +NONE "none"
    }
    class DEGRADED_STAGES {
        <<frozen enum>>
        +SEED "seed"
        +EXPANSION "expansion"
        +SPARQL "sparql"
        +BACKEND_UNAVAILABLE "backend-unavailable"
    }
    class DEGRADED_OUTCOMES {
        <<frozen enum>>
        +BACKEND_CONFIGURED_UNAVAILABLE "backend_configured_but_unavailable"
        +BACKEND_NOT_CONFIGURED "backend_not_configured"
        +SEED_REJECTED "seed_rejected"
        +EXPANSION_UNAVAILABLE "expansion_unavailable"
    }
    class BackendSelection {
        +String name
        +String url
        +Boolean configured
        +String generation
        +String reason
    }
    class AskResult {
        +String turtle
        +String breadcrumb
        +List~String~ seed_iris
        +Number tokens_used
        +Boolean truncated
        +String provenance
        +Boolean cache_hit
        +Boolean degraded
        +List~String~ degraded_stages
        +Boolean full_denied
        +String domain
        +String backend
        +Boolean backend_configured
        +String generation
        +Number latency_ms
    }
    class MATURITY_RANK {
        <<frozen map>>
        +draft 0
        +developing 1
        +emerging 2
        +growing 3
        +established 4
        +mature 5
    }
    BackendSelection --> BACKENDS : name drawn from
    AskResult --> DEGRADED_STAGES : degraded_stages drawn from
    AskResult --> BACKENDS : backend drawn from
    AskResult ..> DEGRADED_OUTCOMES : error drawn from
    note for DEGRADED_OUTCOMES "BACKEND_CONFIGURED_UNAVAILABLE is deliberately distinct from BACKEND_NOT_CONFIGURED —<br/>collapsing the two hides a dead facade behind a normal fallback<br/>(ontology-retrieval.js:108 DEGRADED_OUTCOMES)"
    note for MATURITY_RANK "classifyCause splits availability/timeout from auth_or_validation so a 401 is never<br/>reported as unavailability (ontology-retrieval.js:768-702)"
```

## AB-24.6 POST /v1/chat/completions — scaffold injection then delegate

```mermaid
sequenceDiagram
    autonumber
    participant C as Consumer<br/>holds the door, never the model
    participant FAC as loom-facade<br/>LOOM_FACADE_PORT 8080
    participant IDX as staged generation :ro<br/>docker-compose.unified.yml:338
    participant XI as Xinference bge-small-en-v1.5 384-dim<br/>XINFERENCE_URL
    participant M as model behind DISTILL_BACKEND_URL

    rect rgb(235,242,250)
        Note over C,FAC: retrieval tier — NO model needed (loom/README.md:21,26-32)
        C->>FAC: GET /health
        FAC-->>C: liveness, corpus generation stamp, backend/graph/index readiness, injection_policy
        C->>FAC: GET /loom/generation
        FAC-->>C: the corpus generation identity being served
        C->>FAC: POST /loom/scaffold
        FAC->>IDX: scaffold-index.json + prose-index.json lookup
        FAC-->>C: budget-clamped ontology grounding, ONTOLOGY_BUDGET default 1500
    end
    rect rgb(250,240,235)
        Note over C,M: delegation tier — REQUIRES a model
        C->>FAC: POST /v1/chat/completions
        alt DISTILL_BACKEND_URL blank (docker-compose.unified.yml:310)
            FAC-->>C: 503 — retrieval-only deployment
        else backend configured
            FAC->>IDX: scaffold-inject the LAST user message
            opt LOOM_SEMANTIC_FALLBACK=1 (default 0, gated off until the recall bench clears)
                FAC->>XI: embed query, 384-dim
                XI-->>FAC: query vector
            end
            alt LOOM_CONFIDENCE_INJECTION=1 (default 0 — master switch OFF)
                FAC->>FAC: score matches against LOOM_STRONG_MATCH_SCORE 8.0
                alt score < LOOM_MIN_INJECT_SCORE 2.0
                    FAC->>FAC: skip injection entirely
                else
                    FAC->>FAC: keep matches at or above LOOM_MIN_INJECT_FRACTION 0.4 of budget
                end
            end
            FAC->>M: delegate chat-completions
            Note over FAC,M: PROTOCOL: reasoning backends truncate to EMPTY below LOOM_MIN_MAX_TOKENS 1536 — the<br/>400-to-empty trap (docker-compose.unified.yml:316-317)
            M-->>FAC: completion
            FAC-->>C: completion
        end
        C->>FAC: GET /v1/models
        FAC-->>M: model identity passthrough
    end
    Note over C: INVARIANT: grounding stays on the LAN — the facade delegates only to a LAN/local model
```

## AB-24.7 Model swap — zero consumer change

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator
    participant CFG as deployment config<br/>DISTILL_BACKEND_URL / agentbox.toml
    participant FAC as Loom facade port 8084 or loom:8080
    participant OLD as outgoing model
    participant NEW as incoming model port 8085
    participant CONS as every consumer<br/>see AB-24.1

    Note over CONS: consumers hold ONLY the door URL — none names a model port for scaffolded work
    OP->>NEW: stage the new model behind the facade
    OP->>CFG: repoint DISTILL_BACKEND_URL (compose) or loom_url stays constant (toml)
    OP->>FAC: restart / reload
    FAC->>NEW: delegate subsequent /v1/chat/completions
    FAC--xOLD: no longer delegated to
    CONS->>FAC: unchanged calls
    FAC-->>CONS: unchanged contract
    Note over OP,CONS: INVARIANT ADR-2023: swapping the deployed model must NOT touch any consumer — the model<br/>is an operational detail behind port 8084
    Note over CFG: history — Gemma then Muse then Qwen3.8-27B — agentbox.toml:2007 loom_model =<br/>qwen3.8-27B, agentbox.toml:2011 loom_max_tokens = 32768
    Note over FAC: RESOLVED — GOVERNANCE-capabilities now cites agentbox.toml by [section].key rather than<br/>raw line (ADR-2052 changelog 0.1.1) and correctly states ".loom_max_tokens = 32768, raised<br/>from 16384" — the manifest has loom_url at agentbox.toml:2004 and loom_max_tokens at<br/>agentbox.toml:2011 — the cap was raised after glm-5.3 burned ~16k reasoning tokens and hit the old 16384<br/>cap with empty content twice (agentbox.toml comment at :2008-2010)
    Note over FAC: RESOLVED — GOVERNANCE-capabilities now cites session seeds as `slug = "loom"` /<br/>`slug = "loom-raw"` under [[interaction_plane.session_seeds]] (no raw line number) — the<br/>manifest has slug=loom at agentbox.toml:1645 and slug=loom-raw at agentbox.toml:1652
    Note over NEW: DIVERGENCE: HP's old 192.168.2.48 is DEAD — a stale model-backend route black-holes<br/>every synthesis while /health still answers
```

## AB-24.8 Deployment B bring-up and the staging traps

```mermaid
stateDiagram-v2
    [*] --> ImageBuilt
    ImageBuilt --> Staged : operator stages a full generation
    note right of ImageBuilt
        Image is NOT built from this repo — no Dockerfile here.
        docker build -f loom/deploy/Dockerfile -t loom:rust /home/devuser/workspace
        The build context is the workspace PARENT because loom path-depends
        on the sibling ruvector crate and COPY cannot escape its context.
        loom/README.md:36-45
    end note
    Staged --> Starting : docker compose --profile loom up -d loom
    note right of Staged
        A full generation is scaffold-index.json + prose-index.json
        + the TTLs + ontology-corpus.rvdb with its .generation.json sidecar.
        There is NO mirror-on-start step in the Rust image — ONTOLOGY_SITE and
        LOOM_MIRROR_ON_START are gone and the generation is served immutably.
        loom/README.md:56-71
    end note
    Starting --> RvdbCopied : entrypoint copies .rvdb to tmpfs /run/loom
    note right of Starting
        Opening the .rvdb mutates the redb file even for READS because the
        HNSW index is repacked on open, so it cannot be served from the
        read-only mount. tmpfs uid/gid MUST stay 65532 to match the image's
        non-root user or the copy fails EACCES.
        loom/README.md:78-81, docker-compose.unified.yml:339-343
    end note
    RvdbCopied --> Healthy : GET /health returns 200
    RvdbCopied --> EmptyFloor : source empty or mis-pointed
    note right of EmptyFloor
        THE EMPTY-FLOOR TRAP. The facade still starts and /health still
        returns 200, but the log reads "lexical index NOT loaded ... empty
        floor". That is a STAGING bug, not a dead container — check the
        mount before you check the process. loom/README.md:73-76
    end note
    EmptyFloor --> Staged : repoint LOOM_DATA_SOURCE
    Healthy --> RetrievalOnly : DISTILL_BACKEND_URL blank
    Healthy --> FullService : DISTILL_BACKEND_URL set
    RetrievalOnly --> FullService : model attached behind the seam
    FullService --> Healthy : healthcheck every 30s, 3 retries, 25s start_period
    RetrievalOnly --> [*]
    FullService --> [*]
```

## AB-24.9 Consumer register — which door each one holds

```mermaid
flowchart LR
    subgraph doors["Doors"]
        D84["LAN facade port 8084/v1"]
        D80["sidecar loom:8080/v1"]
        D85["raw model port 8085, compose default only<br/>agentbox/flake.nix:3173, seeded by nothing"]
    end
    RET["ontology-retrieval brain<br/>LOOM_FACADE_URL<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:491"] --> D84
    COND["ontology condense endpoint<br/>agentbox/agentbox.toml:807<br/>model qwen3.8-27B style openai max_concurrency 2 #40;agentbox.toml:810#41;"] --> D84
    DREAM["dream_machine loom_url<br/>agentbox/agentbox.toml:2004"] --> D84
    SEEDL["session seed slug=loom<br/>agentbox/agentbox.toml:1645<br/>model loom-lan/qwen3.8-27B agentbox.toml:1647, scaffolded for knowledge work"] --> D84
    SEEDR["session seed slug=loom-raw<br/>agentbox/agentbox.toml:1652<br/>model loom-agent/current agentbox.toml:1654, model-agnostic passthrough"] --> D84
    EMAIL["email gateway<br/>REASONER_BASE_URL http://loom:8080/v1<br/>loom/README.md:19-21"] --> D80
    CUST["security.deepsec custom ai_base_url<br/>agentbox/agentbox.toml:1963 #40;deepsec#39;s own AI-reviewer<br/>backend, NOT the #91;consultants#93; tier#41;"] --> D80
    D84 --> M["qwen3.8-27B"]
    D80 --> M
    D85 --> M
    subgraph notes["Invariants and drift"]
        direction TB
        N1["RESOLVED ADR-2053: the dream engine's default provider is Z.AI by deliberate<br/>choice — GOVERNANCE-capabilities now states this and names the egress posture.<br/>loom_url/loom_model select the LAN-only path when llm_provider = loom. See AB-23"]
        N2["PROPOSED ADR-2074: the ADR-051 deferred-distillation tools become a discrete<br/>manifest-gated MCP server with a job URN kind and distill plus recombine beads<br/>ADR-2023 remaining is the ORIGIN of this gap, not its resolution (see AB-26)"]
        N3["STAGED ADR-2075: GET generation before cache lookup<br/>bind loaded digest, semantic generation and bge 384 cosine<br/>response identity headers must match; old live server is rejected"]
        N4["PROPOSED ADR-2076: benchmark /loom/search plus /loom/sparql on its own terms with a<br/>frozen recall band in the shape of the RuVector recall gate - scaffold and chat numbers<br/>are never cited as evidence for this path"]
        N5["app/ontology-mcp is a standalone stdio MCP server left in place with no build or run<br/>path from this repo, pending a decision on where it should live (loom/README.md:108-115)"]
        N1 ~~~ N2 ~~~ N3 ~~~ N4 ~~~ N5
    end
```

## AB-24.10 ADR-2084 - five callers, one published client

```mermaid
flowchart TB
    subgraph before["Before: five hand-rolled chat-completions clients"]
        B1["each knew a DIFFERENT subset of the same three traps<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:21-28"]
        B2["only dream-engine knew a facade can answer 200 with ontology prose<br/>and never call the model"]
        B3["only loom-draft knew about truncation retries and the ADR-139 assertion"]
        B4["only agentbox-mcp floored max_tokens, so the others could ask a<br/>reasoning model for 400 and read empty content as success"]
        B1 --> B2 --> B3 --> B4
    end
    subgraph after["After: the loom-client crate, published from the loom repository"]
        C1["dream-engine call_loom<br/>agentbox/services/dream-engine/src/llm.rs:196"]
        C2["podcast-ingest extraction<br/>agentbox/services/podcast-ingest/src/ingest/loom.rs:34"]
        C3["podcast-ingest promotion<br/>agentbox/services/podcast-ingest/src/promote/loom.rs:69"]
        C4["explainer-loom-draft<br/>agentbox/services/explainer-tools/src/bin/loom_draft.rs:22"]
        C5["web-summary MCP server<br/>agentbox/services/agentbox-mcp/src/web_summary/llm.rs:65"]
    end
    B4 --> C1
    C1 --> C2 --> C3 --> C4 --> C5
    subgraph knows["What the one place now knows"]
        direction TB
        K1["INVARIANT ADR-2084: hand-rolled chat-completions construction against a<br/>facade is PROHIBITED - a sixth caller takes the crate<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:37"]
        K2["verbatim and scaffold are INDEPENDENT switches. Three callers sent only<br/>the first and agentbox-mcp sent neither, so a page summary could be<br/>answered from the ontology without the page being read<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:30-33"]
        K3["a reasoning model truncated below 1536 tokens returns EMPTY content,<br/>not a short answer - web_summary/llm.rs:20 floors every ask and<br/>web_summary/llm.rs:63 clamps the caller's number up to it"]
        K1 ~~~ K2 ~~~ K3
    end
    C5 -.-> knows
```

**Invariant:** the web-summary path clamps every request to at least 1536 tokens and is pinned by `max_tokens_is_always_clamped_to_at_least_1536` (`../project/agentbox/services/agentbox-mcp/src/web_summary/llm.rs:151`); a scaffold-only serve is reported as a failure, not an answer (`../project/agentbox/services/agentbox-mcp/src/web_summary/llm.rs:198`).

## AB-24.11 ADR-139 - the per-request scaffold opt-out

```mermaid
sequenceDiagram
    autonumber
    participant ONT as ontology subject<br/>agentbox/services/dream-engine/src/llm.rs:211
    participant CODE as non-ontology subject<br/>agentbox/services/explainer-tools/src/bin/loom_draft.rs:196
    participant CL as loom-client
    participant FAC as the facade, the stable model door
    participant M as the model behind the door

    Note over ONT,CODE: the SUBJECT decides, and the decision belongs to the REQUEST -<br/>no header, no environment switch (ADR-139-per-request-scaffold-opt-out.md:32)
    ONT->>CL: LoomOptions::declining_verbatim()
    CL->>FAC: POST chat completions, scaffold ON, verbatim declined
    FAC->>M: scaffold-injected prompt
    M-->>FAC: completion
    FAC-->>CL: served_mode names the regime, corpus_backed true
    CODE->>CL: LoomOptions::passthrough()
    CL->>FAC: POST chat completions with loom_options scaffold false
    Note over CL,FAC: no retrieval, no injection, no verbatim serve, no thinking control,<br/>the Loom-private field stripped, the body otherwise forwarded unchanged<br/>(ADR-139-per-request-scaffold-opt-out.md:23-25)
    FAC->>M: the body as the caller wrote it
    M-->>FAC: completion
    FAC-->>CODE: served_mode passthrough, grounding status passthrough,<br/>corpus_backed false, injected_tokens 0<br/>(ADR-139-per-request-scaffold-opt-out.md:26-30)
    Note over FAC: absence of the key, or any value other than the boolean false, leaves the<br/>scaffold ON (ADR-139-per-request-scaffold-opt-out.md:31)
    Note over M: INVARIANT ADR-139: the direct model port stays an implementation detail<br/>behind the door and is NOT a documented consumer path<br/>(ADR-139-per-request-scaffold-opt-out.md:33-34)
```

**Debt:** grounding applied to a subject the ontology does not cover is a WRONG answer, not a weak one: on 2026-09-09 a packet about a test script scored the blockchain class Node at 42 and was served from the corpus in 40 ms with zero completion tokens (`../loom/docs/design/ADR-139-per-request-scaffold-opt-out.md:13-16`), which is why `../project/agentbox/services/explainer-tools/src/draft.rs:4` declines the scaffold on every request.

## Audit qualification - 2026-09-07

The execution pass implements consumer verification in `ontology-retrieval.js::loomGenerationVerifier`: GET generation before every cache lookup, compare the configured pin and loaded digest/model/corpus, and validate identity headers on search/SPARQL responses. The local Loom route implementation preserves response bodies and adds those headers. This source is staged: the live façade was probed and still reports lexical generation 2026-08-22 versus semantic 2026-08-17, without loaded identity/embedding fields. A coordinated bundle/server rollout is required before activating the stricter client. See [execution evidence](../../estate-review/closeout/2026-09-07-execution-agentbox.md). The graph loader still uses the shared default graph, so ADR-2073 remains open. Generation reporting is not an automatic corpus reload.
