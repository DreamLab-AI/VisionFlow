---
id: AB-24
title: Ontology Loom facade and the model-swap seam
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2023, ADR-2053, ADR-2055, ADR-2075]
sources:
  - ../project/agentbox/mcp/servers/lib/ontology-retrieval.js
  - ../project/agentbox/mcp/servers/lib/ontology-budget.js
  - ../project/agentbox/agentbox.toml
  - ../project/docker-compose.unified.yml
  - ../project/loom/README.md
  - ../project/agentbox/scripts/opf-router.py
  - ../project/agentbox/mcp/servers/lib/ontology-telemetry.js
  - ../project/agentbox/flake.nix
verified_commit: a0ee1fe5740baa38e14c4ff3fe512dd557bcbb6e
---

## AB-24.1 Two deployments of one facade contract — topology

```mermaid
flowchart TB
    subgraph consumers["Consumers hold a DOOR, never a raw model port (ADR-2023)"]
        RET["ontology-retrieval brain<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:734"]
        COND["ontology condense<br/>agentbox/agentbox.toml:675"]
        DREAM["dream-engine loom_url<br/>agentbox/agentbox.toml:1734"]
        SEED["AoE session seed slug=loom<br/>agentbox/agentbox.toml:1374"]
        SEEDRAW["AoE session seed slug=loom-raw<br/>agentbox/agentbox.toml:1381"]
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
    SEEDRAW -->|"raw port 8085 — explicit coding/benchmark path"| M85
    EMAIL -->|"http://loom:8080/v1"| SIDE
    F84 -->|"ml DNATs over the 25G rail 10.10.10.0/30"| M85
    SIDE -->|"DISTILL_BACKEND_URL blank = retrieval-only, /v1 returns 503"| M85
    DATA --> SIDE
    SIDE -->|"entrypoint copies .rvdb off :ro — opening redb mutates it"| TMPFS
    subgraph notes["Invariants and drift"]
        direction TB
        N1["RESOLVED ADR-2070 #40;2026-09-05#41;: not a breach. ADR-045 one-front-door is an INGRESS rule<br/>#40;port 9096, NIP-98, control surfaces reaching INTO the box#41; and says nothing about EGRESS to a LAN<br/>model host. The raw port 8085 door is deliberate and named #40;flake.nix LOOM_RAW_BASE_URL, the loom-raw<br/>session seed#41; — agent-choice and benchmark-only for raw coding, never a fallback and never<br/>auto-routed when the facade errors. Knowledge-work consumers hold port 8084. A third door needs an ADR"]
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
    Note over CFG: history — Gemma then Muse then Qwen3.8-27B — agentbox.toml:1735 loom_model =<br/>qwen3.8-27B, :1739 loom_max_tokens = 32768
    Note over FAC: RESOLVED — GOVERNANCE-capabilities now cites agentbox.toml by [section].key rather than<br/>raw line (ADR-2052 changelog 0.1.1) and correctly states ".loom_max_tokens = 32768, raised<br/>from 16384" — the working tree has loom_url at agentbox.toml:1734 and loom_max_tokens at<br/>:1739 — the cap was raised after glm-5.3 burned ~16k reasoning tokens and hit the old 16384<br/>cap with empty content twice (agentbox.toml comment at :1736-1738)
    Note over FAC: RESOLVED — GOVERNANCE-capabilities now cites session seeds as `slug = "loom"` /<br/>`slug = "loom-raw"` under [[interaction_plane.session_seeds]] (no raw line number) — the<br/>working tree has slug=loom at agentbox.toml:1374 and slug=loom-raw at :1381
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
        D85["raw model port 8085 — named egress"]
    end
    RET["ontology-retrieval brain<br/>LOOM_FACADE_URL<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:491"] --> D84
    COND["ontology condense endpoint<br/>agentbox/agentbox.toml:675<br/>model qwen3.8-27B style openai max_concurrency 2"] --> D84
    DREAM["dream_machine loom_url<br/>agentbox/agentbox.toml:1734"] --> D84
    SEEDL["session seed slug=loom<br/>agentbox/agentbox.toml:1374<br/>model loom-lan/qwen3.8-27B — scaffolded, knowledge work"] --> D84
    SEEDR["session seed slug=loom-raw<br/>agentbox/agentbox.toml:1381<br/>model loom-raw/qwen3.8-27B — no scaffold, coding"] --> D85
    EMAIL["email gateway<br/>REASONER_BASE_URL http://loom:8080/v1<br/>loom/README.md:19-21"] --> D80
    CUST["security.deepsec custom ai_base_url<br/>agentbox/agentbox.toml:1693 #40;deepsec#39;s own AI-reviewer<br/>backend, NOT the #91;consultants#93; tier#41;"] --> D80
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

## Audit qualification — 2026-09-07

The execution pass implements consumer verification in `ontology-retrieval.js::loomGenerationVerifier`: GET generation before every cache lookup, compare the configured pin and loaded digest/model/corpus, and validate identity headers on search/SPARQL responses. The local Loom route implementation preserves response bodies and adds those headers. This source is staged: the live façade was probed and still reports lexical generation 2026-08-22 versus semantic 2026-08-17, without loaded identity/embedding fields. A coordinated bundle/server rollout is required before activating the stricter client. See [execution evidence](../../estate-review/closeout/2026-09-07-execution-agentbox.md). The graph loader still uses the shared default graph, so ADR-2073 remains open. Generation reporting is not an automatic corpus reload.
