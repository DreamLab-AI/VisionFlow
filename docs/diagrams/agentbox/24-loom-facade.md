---
id: AB-24
title: Ontology Loom facade and the model-swap seam
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2023, ADR-2053, ADR-2055, ADR-2075, ADR-2084, ADR-2122, ADR-2140, ADR-2153]
sources:
  - ../project/agentbox/mcp/servers/lib/ontology-retrieval.js
  - ../project/agentbox/mcp/servers/lib/ontology-jev.js
  - ../project/agentbox/docs/adr/ADR-2084-one-published-loom-client-for-every-facade-caller.md
  - ../project/agentbox/services/dream-engine/src/llm.rs
  - ../project/agentbox/services/podcast-ingest/src/ingest/loom.rs
  - ../project/agentbox/services/podcast-ingest/src/promote/loom.rs
  - ../project/agentbox/services/explainer-tools/src/bin/loom_draft.rs
  - ../project/agentbox/services/explainer-tools/src/draft.rs
  - ../project/agentbox/services/agentbox-mcp/src/web_summary/llm.rs
  - ../loom/docs/design/ADR-139-per-request-scaffold-opt-out.md
  - ../project/agentbox/scripts/session-provision.mjs
  - ../project/agentbox/services/agentbox-sessions/src/manifest.rs
  - ../project/agentbox/mcp/servers/lib/ontology-budget.js
  - ../project/agentbox/agentbox.toml
  - ../project/docker-compose.unified.yml
  - ../project/loom/README.md
  - ../project/agentbox/scripts/opf-router.py
  - ../project/agentbox/mcp/servers/lib/ontology-telemetry.js
  - ../project/agentbox/flake.nix
  - ../loom/docs/design/ADR-140-loom-v2-agent-drivable-grounding.md
  - ../loom/docs/design/ADR-141-loom-consumes-the-vault-build.md
  - ../loom/crates/loom-mcp/src/lib.rs
  - ../loom/crates/loom-mcp/src/schema.rs
  - ../loom/crates/loom-facade/src/mcp_host.rs
  - ../project/agentbox/config/egress-policy.json
  - ../project/agentbox/config/custody/env-classes.json
  - ../project/agentbox/scripts/ci/render-egress-register.js
  - ../loom/crates/loom-facade/src/routes/mod.rs
  - ../loom/crates/loom-facade/src/routes/attest.rs
  - ../loom/crates/loom-facade/src/routes/health.rs
  - ../loom/crates/loom-domain/src/model.rs
  - ../project/agentbox/services/dream-engine/src/engine.rs
  - ../project/agentbox/services/dream-engine/src/config.rs
  - ../project/agentbox/services/dream-engine/src/estate.rs
  - ../project/agentbox/scripts/hp-loom-reconcile.sh
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../loom/crates/loom-facade/src/jev.rs
  - ../project/agentbox/scripts/dream-machine-nightly.mjs
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
verified_commit: {agentbox: 0b0f7918961e629280faf86f88227535abf514b5, visionclaw: e74bcddc772d47a45b1971156a2bab540eaa027c, loom: 2fad3e085c3807a1effbf724d4f31fc6ea05a0f9}
---

## AB-24.1 Two deployments of one facade contract — topology

```mermaid
flowchart TB
    subgraph consumers["Consumers hold a DOOR, never a raw model port (ADR-2023)"]
        RET["ontology-retrieval brain<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:776"]
        COND["ontology condense<br/>agentbox/agentbox.toml:1117"]
        DREAM["dream-engine loom_url<br/>agentbox/agentbox.toml:2569"]
        SEEDRAW["agentbox-sessions seed slug=loom-raw #40;the only Loom seed left#41;<br/>agentbox/agentbox.toml:2176"]
        EMAIL["email gateway REASONER_BASE_URL<br/>see AB-27"]
    end
    subgraph depA["Deployment A — LAN facade on machinelearn .132"]
        F84["Loom facade<br/>machinelearn .132, port 8084, path /v1"]
    end
    subgraph depB["Deployment B — sidecar on visionclaw_network (compose profile loom)"]
        SIDE["loom-facade (Rust)<br/>docker-compose.unified.yml:371"]
        TMPFS["tmpfs /run/loom mode=0750 uid=65532<br/>docker-compose.unified.yml:427"]
        DATA["loom-data :ro generation<br/>docker-compose.unified.yml:422"]
    end
    subgraph model["The model — an operational detail BEHIND the door"]
        M85["loom-model port 8085 qwen3.8-27B<br/>DISTILL_BACKEND_URL"]
    end
    RET -->|"LOOM_FACADE_URL"| F84
    COND -->|"POST /v1/chat/completions"| F84
    DREAM -->|"llm_provider=loom only"| F84
    SEEDRAW -->|"session-provision.mjs openCodeConfig points it at LOOM_BASE_URL with loom_options scaffold false<br/>session-provision.mjs:104, session-provision.mjs:118, session-provision.mjs:142-147"| F84
    EMAIL -->|"http://loom:8080/v1"| SIDE
    F84 -->|"ml DNATs over the 25G rail 10.10.10.0/30"| M85
    SIDE -->|"DISTILL_BACKEND_URL blank = retrieval-only, /v1 returns 503"| M85
    DATA --> SIDE
    SIDE -->|"entrypoint copies .rvdb off :ro — opening redb mutates it"| TMPFS
    subgraph notes["Invariants and drift"]
        direction TB
        N1["WITHDRAWN in code: the loom-raw SEED no longer holds a raw model door. openCodeConfig<br/>gives loom-lan, loom-agent and loom-raw the SAME LOOM_BASE_URL #40;session-provision.mjs:104,<br/>session-provision.mjs:131-147#41; and declines the scaffold per request<br/>instead #40;session-provision.mjs:118#41;, the ADR-139 answer to what the raw port was for.<br/>LOOM_RAW_BASE_URL survives only as a compose default #40;flake.nix:3961#41; that nothing seeds"]
        N2["RESOLVED ADR-2055: opf-router is the PRIVACY-FILTER redaction sidecar on OPF_PORT<br/>9092 (agentbox.toml [privacy_filter].port, scripts/opf-router.py:41, flake.nix<br/>[program:opf-router]). BASELINE-container previously described it as an<br/>OpenAI-compatible facade on port 8084 — corrected. No agentbox program serves<br/>port 8084 — that is the Loom facade on machinelearn"]
        N3["The loom-facade implementation lives OUTSIDE this repo at /home/devuser/workspace/loom.<br/>This repo holds the deployment contract only (loom/README.md:8-15)"]
        N1 ~~~ N2 ~~~ N3
    end
```

**Drift:** the scaffolded `loom` session seed (`loom-lan/qwen3.8-27B`) was dropped from the manifest, leaving `loom-raw` as the only Loom seed, and the new session plane asserts exactly that roster (`../project/agentbox/agentbox.toml:2175-2179`, `../project/agentbox/services/agentbox-sessions/src/manifest.rs:167-168`); `openCodeConfig` in the provisioner still defines the `loom-lan` provider, so the unscaffolded and scaffolded doors both stay configured (`../project/agentbox/scripts/session-provision.mjs:131-135`). The manifest comment on the neighbouring `deepseek` seed still says "AoE" (`../project/agentbox/agentbox.toml:2169`), and GOVERNANCE-capabilities still describes both a scaffolded `slug = "loom"` seed and a raw `loom-raw/qwen3.8-27B` path via `:8085` (`../project/agentbox/docs/GOVERNANCE-capabilities.md:312-314`).

**Resolved:** the replacement of Agent of Empires is no longer proposed: ADR-2153 is `accepted`, implementation `partial`, activation `live` (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`), so the `loom-raw` seed is hosted by the owned `agentbox-sessions` plane and its OpenCode provider config comes from `scripts/session-provision.mjs`, not from `aoe-seed-sessions.mjs`, which is deleted.

## AB-24.2 selectBackend — which store answers, and why

```mermaid
sequenceDiagram
    autonumber
    participant CALL as createDefaultRetrieval<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:776
    participant SEL as selectBackend<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:533
    participant ENV as process.env
    participant LF as makeLoomFetch<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:658
    participant VF as makeVcFetch<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:558
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
    participant ASK as ask<br/>ontology-retrieval.js:298
    participant VERIFY as loomGenerationVerifier<br/>ontology-retrieval.js:707
    participant FETCH as makeLoomFetch<br/>ontology-retrieval.js:658
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
            opt Jev option configured and the request did not decline the scaffold (ontology-retrieval.js:317)
                ASK->>ASK: jev.evaluate over the admitted candidates, annotation only unless a confidence threshold is set (ontology-retrieval.js:435-445)
                Note over ASK: a failed evaluation leaves the flag-off result and is not cached under the Jev key (ontology-retrieval.js:437)
            end
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
    Note over ASK: BACKEND_CONFIGURED_UNAVAILABLE is deliberately distinct from BACKEND_NOT_CONFIGURED —<br/>collapsing the two hides a dead facade behind a normal fallback (ontology-retrieval.js:106-117).<br/>classifyCause splits availability/timeout from auth_or_validation so a 401 is never reported as<br/>unavailability (ontology-retrieval.js:812-817)
```

## AB-24.4 Cache key completeness and cache-hit constraint revalidation

```mermaid
sequenceDiagram
    participant ASK as ask<br/>ontology-retrieval.js:298
    participant VERIFY as loomGenerationVerifier<br/>ontology-retrieval.js:707
    participant KEY as cacheKey<br/>ontology-retrieval.js:76
    participant CACHE as TTL cache
    participant SAT as cacheEntrySatisfies<br/>ontology-retrieval.js:162
    ASK->>VERIFY: verify selected Loom before reading cache
    VERIFY-->>ASK: loaded graph/semantic identity plus embedding contract
    Note over ASK,VERIFY: No valid identity means no cached answer.<br/>Configured generation is compared as a pin, never used to relabel bytes.
    ASK->>KEY: all effective request constraints
    Note over KEY: generation discriminator hashes loaded generation, content digest,<br/>embedding model and dimensions, key also includes domain, budget and scope,<br/>and the Jev question-set version only when a Jev evaluator is configured (ontology-retrieval.js:82)
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

## AB-24.6 POST /v1/chat/completions — scaffold injection then delegate

```mermaid
sequenceDiagram
    autonumber
    participant C as Consumer<br/>holds the door, never the model
    participant FAC as loom-facade<br/>LOOM_FACADE_PORT 8080
    participant IDX as staged generation :ro<br/>docker-compose.unified.yml:422
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
        alt DISTILL_BACKEND_URL blank (docker-compose.unified.yml:394)
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
            opt LOOM_JEV on (default off in the facade, but the sidecar compose now sets 1 against the local systemone:8097 route with cloud refused, docker-compose.unified.yml:379-386), after retrieval, the generation guard and the scaffold=false opt-out
                FAC->>FAC: gate may withhold the scaffold and rank may reorder it, each inert until its confidence threshold is set (loom-facade/src/routes/mod.rs:383-384, loom-facade/src/jev.rs:732, loom-facade/src/jev.rs:795)
            end
            FAC->>M: delegate chat-completions
            Note over FAC,M: PROTOCOL: reasoning backends truncate to EMPTY below LOOM_MIN_MAX_TOKENS 1536 — the<br/>400-to-empty trap (docker-compose.unified.yml:400-401)
            M-->>FAC: completion
            opt LOOM_JEV on
                FAC->>FAC: support check adds a jev block to the response, the answer text is never touched (loom-facade/src/routes/mod.rs:470-476, loom-facade/src/jev.rs:891)
            end
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
    Note over CFG: history — Gemma then Muse then Qwen3.8-27B. Since 2026-10-01 the manifest pins NO model:<br/>loom_model is empty, meaning the single model the Loom advertises at /models (agentbox.toml:2570-2572),<br/>loom_max_tokens = 32768 (agentbox.toml:2576). The nightly script discovers it (dream-machine-nightly.mjs:229)
    Note over CFG: INVARIANT: the Rust dream engine does not discover — the supervisor exports LOOM_MODEL empty<br/>(flake.nix:2900) and llm_config passes it through as the request model (estate.rs:106), so a loom night<br/>sends an empty model name and depends on the Loom answering an empty name with its loaded model. Observed by a live<br/>request to the Loom door on 2026-10-02: it answered with a loom.generation block. The serde default<br/>qwen3.8-27B (config.rs:171) applies only when the key is absent, and the night provider defaults<br/>to zai (agentbox.toml:2582, config.rs:184), so loom is an opt-in
    Note over FAC: RESOLVED — GOVERNANCE-capabilities now cites agentbox.toml by [section].key rather than<br/>raw line (ADR-2052 changelog 0.1.1) and correctly states ".loom_max_tokens = 32768, raised<br/>from 16384" — the manifest has loom_url at agentbox.toml:2569 and loom_max_tokens at<br/>agentbox.toml:2576 — the cap was raised after glm-5.3 burned ~16k reasoning tokens and hit the old 16384<br/>cap with empty content twice (agentbox.toml comment at :2573-2575)
    Note over FAC: the manifest keeps one Loom session seed under [[interaction_plane.session_seeds]]:<br/>slug=loom-raw at agentbox.toml:2176 (the scaffolded slug=loom seed was dropped)
    Note over NEW: DIVERGENCE: HP's old 192.168.2.48 is DEAD — a stale model-backend route black-holes<br/>every synthesis while /health still answers
```

**Drift:** `../project/agentbox/docs/GOVERNANCE-capabilities.md:316` still names **Qwen3.8-27B** as the current model by `[dream_machine].loom_model`, which has been empty since `878f23311` (`../project/agentbox/agentbox.toml:2572`): the manifest now leaves the model to whatever the Loom advertises.

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
        loom/README.md:78-81, docker-compose.unified.yml:423-427
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
        D85["raw model port 8085, compose default only<br/>agentbox/flake.nix:3961, seeded by nothing"]
    end
    RET["ontology-retrieval brain<br/>LOOM_FACADE_URL<br/>agentbox/mcp/servers/lib/ontology-retrieval.js:533"] --> D84
    COND["ontology condense endpoint<br/>agentbox/agentbox.toml:1117<br/>model qwen3.8-27B style openai max_concurrency 2 #40;agentbox.toml:1118-1120#41;"] --> D84
    DREAM["dream_machine loom_url<br/>agentbox/agentbox.toml:2569"] --> D84
    SEEDR["session seed slug=loom-raw<br/>agentbox/agentbox.toml:2176<br/>model loom-agent/current agentbox.toml:2178, model-agnostic passthrough"] --> D84
    EMAIL["email gateway<br/>REASONER_BASE_URL http://loom:8080/v1<br/>loom/README.md:19-21"] --> D80
    CUST["security.deepsec custom ai_base_url<br/>agentbox/agentbox.toml:2528 #40;deepsec#39;s own AI-reviewer<br/>backend, NOT the #91;consultants#93; tier#41;"] --> D80
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
        C1["dream-engine call_loom<br/>agentbox/services/dream-engine/src/llm.rs:205"]
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
    participant ONT as ontology subject<br/>agentbox/services/dream-engine/src/llm.rs:220
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

## AB-24.12 ADR-140 — a third door, same gate: the agentic MCP plane

```mermaid
sequenceDiagram
    autonumber
    participant AG as MCP agent client
    participant MCP as POST /mcp<br/>loom-facade/src/routes/mod.rs:65,95
    participant DISP as loom_mcp::dispatch<br/>loom-mcp/src/lib.rs:102
    participant CT as call_tool<br/>loom-mcp/src/lib.rs:146
    participant TH as ToolHost for AppState<br/>loom-facade/src/mcp_host.rs:93
    participant CHAT as POST /v1/chat/completions<br/>loom-facade/src/routes/mod.rs:66

    Note over AG,MCP: ADR-140 D1 - two protocol surfaces, ONE binary, ONE AppState, ONE gate<br/>(loom-facade/src/routes/mod.rs:64-65)
    AG->>MCP: JSON-RPC tools/call (single or batch)
    MCP->>DISP: dispatch(&state, message)
    DISP->>CT: name plus arguments
    alt loom.manifest
        CT->>TH: manifest(salience)<br/>mcp_host.rs:94
    else loom.browse
        CT->>TH: browse(query, kind, n)<br/>mcp_host.rs:118 - addresses only, never content
    else loom.resolve
        CT->>TH: resolve(iris)<br/>mcp_host.rs:146 - the ONLY tool returning corpus content
    else loom.sparql
        CT->>TH: sparql(query)<br/>mcp_host.rs:191
    else loom.neighbours / loom.paths
        CT->>TH: neighbours/paths over the reasoned graph<br/>mcp_host.rs:199,222
    end
    TH-->>CT: ToolOutcome{payload, grounding}
    CT-->>DISP: isError distinguishes no-match from could-not-look<br/>(loom-mcp/src/lib.rs:212-214)
    DISP-->>MCP: JSON-RPC response
    MCP-->>AG: structuredContent plus grounding envelope
    Note over CHAT: the injection plane is UNCHANGED - single-turn and non-agentic<br/>consumers still get one static scaffold merge per request
    Note over TH: INVARIANT loom-mcp/src/lib.rs:1-9: this crate holds NO retrieval logic -<br/>every call delegates to the SAME ports, fusion pipeline and gate chat_completions uses,<br/>so the agentic door cannot acquire a second retrieval policy
```

**What it shows:** `/mcp` (ADR-140 D1) puts a JSON-RPC tool surface beside `/v1/chat/completions` in the same `loom-facade` binary, dispatching six tools (`loom.manifest`, `loom.browse`, `loom.resolve`, `loom.sparql`, `loom.neighbours`, `loom.paths`) through the one `ToolHost` implementation the chat plane's scaffold assembly also uses. **Why it is this way:** ADR-139's incident — a static scaffold fragment served the blockchain class `Node` to a consumer discussing a test script — is, per [Evo] (arXiv:2609.15779, cited ADR-140 §1), the generic failure mode of pre-reasoning static injection; per-turn tool-driven retrieval is the ecosystem's answer, and Loom ships it as a second surface rather than replacing the first because non-agentic consumers (the email gateway, single-turn lookups) have nothing to drive a tool loop with.

## AB-24.13 ADR-141 — the served generation becomes `visionGraph@sha`, and a governance ledger route lands

```mermaid
flowchart TB
    subgraph build["Corpus origin"]
        VAULT["vault build over visionGraph<br/>emits C3 marker: id, commit, content_digest,<br/>class_count, page_count, vocabulary_version, stale_after"]
    end
    subgraph loom["loom-facade::mirror"]
        MIR["reads C3 marker as GenerationSource::VaultBuild<br/>loom-domain/src/model.rs:509-513<br/>still reads legacy mirror marker too - ONE /loom/generation shape either way"]
    end
    subgraph health["GET /health"]
        STALE["generation.stale_after echoed, generation.stale judged<br/>against this node's clock via Generation::is_stale<br/>loom-domain/src/model.rs:460-462, model.rs:480-487, routes/health.rs:139-148"]
    end
    subgraph ledger["POST /loom/attest - contract C5"]
        ATT["attest#40;State, Json#60;AttestRequest#62;#41;<br/>loom-facade/src/routes/attest.rs:113<br/>outcome: Promote #124; Demote #124; Reject #124; Expired<br/>attest.rs:44-55"]
        VER["GET /loom/attest/verify - re-hashes the chain<br/>attest.rs:159"]
    end
    VAULT --> MIR
    MIR --> STALE
    MIR -->|"chain-hashed, one append per human decision"| ATT
    ATT --> VER
    subgraph notes["Invariants and drift"]
        direction TB
        N1["INVARIANT ADR-141 D2: a generation past stale_after is STILL SERVED, honestly<br/>labelled - staleness is carried, never enforced (loom-domain/src/model.rs:459-462)"]
        N2["INVARIANT attest.rs:20-25: passed=true means this decision ADMITTED content -<br/>true only for Promote, false for Demote/Reject/Expired; a demotion is a change,<br/>not an admission"]
        N3["RESOLVED ADR-141 D5: loom-mcp-stdio deleted - inside the estate agents use the<br/>vault CLI and loom-client; #47;mcp stays HTTP-only, for hosts outside the estate"]
        N4["Status: Proposed, 2026-09-22. Not yet reflected in this topic's governing doc<br/>(GOVERNANCE-capabilities.md) or in ADR-2084 - see AB-24.10 for the crate this<br/>generation identity change does not yet touch"]
        N1 ~~~ N2 ~~~ N3 ~~~ N4
    end
```

**What it shows:** the corpus stops being GitHub-published; `loom-facade::mirror` now recognises a `vault build` marker (`GenerationSource::VaultBuild`) naming the generation `visionGraph@<sha>`, and `/health` reports a `stale_after` promise plus this node's own staleness judgement without ever refusing to serve. A `POST /loom/attest` route gives the corpus's human promotion/demotion decisions a chain-hashed, restart-surviving ledger. **Why it is this way:** ADR-141 fixes the gap ADR-140 §1.3 measured — a Loom bundle stale by a month against the vault it should mirror — and, per ADR-136 D5, moves `AttestationLedger` from a build/CI-time port to a serving-path writer so `passed` on an entry answers "is this in the corpus because someone said so", the one question the audit trail exists for.

**Debt (corpus door behind the vault, 2026-10-02):** the next generation `visionGraph@015ca2c1f` is built, promoted and verified, but the door still serves `visionGraph@ae913f93…`, and the `vault` 0.1.0 baked into the agentbox image predates the portable-records file, so it cannot produce a promotable bundle (`../loom/docs/design/ADR-141-loom-consumes-the-vault-build.md:141-150`). The pin is fixed in source, since `vaultSrc` now names VisionClaw `f3bbb430ea` (`../project/agentbox/flake.nix:31-32`), but that reaches a running `vault` only at the next image rebuild. ADR-141 also records that the corpus build is now `ontomark build` and `vault` survives only as the governance adapter, with the served identity unchanged (`../loom/docs/design/ADR-141-loom-consumes-the-vault-build.md:9`); AB-24.15 draws who publishes the generation now.

## AB-24.14 Custody X-1 step 1 at the Loom door — what the register and the env classes say

```mermaid
flowchart TB
    subgraph reg["Prompt egress register, rendered into SECURITY-profiles - config/egress-policy.json:48-55"]
        R1["loom-door: prompts to LOOM_BASE_URL, the façade on the HP host, LAN class;<br/>gated by condense enabled and llm_provider; accepted by ADR-2070<br/>config/egress-policy.json:224-236"]
        R2["loom-raw: prompt AND code to the raw model, LAN class;<br/>marker no-gate, the session seed is chosen per session<br/>config/egress-policy.json:237-252"]
        R3["zai-night: the dream engine's DEFAULT provider is Z.AI, a vendor;<br/>marker no-accepted-record, nothing accepts the corpus leaving<br/>config/egress-policy.json:346-361"]
    end
    subgraph cls["Env classes - config/custody/env-classes.json"]
        C1["LOOM_BASE_URL, LOOM_FACADE_URL, LOOM_HP_HOST, LOOM_MODEL, LOOM_RAW_BASE_URL, LOOM_URL:<br/>NON_SECRET - env-classes.json:537-542"]
        C2["ZAI_ANTHROPIC_API_KEY: DEVUSER_CLASS, a Q8 exception, stays in<br/>devuser's environment this step - env-classes.json:227-230"]
        C3["no Loom or provider variable is in the ROLE set, so<br/>[security].role_isolation scrubs nothing on this path<br/>env-classes.json:33, env-classes.json:90"]
    end
    R1 --> C1
    R2 --> C1
    R3 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> SAME["the door, its consumers and AB-24.9's register are the same with the flag<br/>on or off: the Loom path holds no key, only a URL"]
    reg -.-> DR["DRIFT: the register declares 0919dc39a and cites manifest lines 2179<br/>and 2163 and flake line 3438; at this revision those lines are<br/>a session seed's worktree key, a session seed's tool key and an empty line - llm_provider is now agentbox.toml:2582 and the LOOM_BASE_URL default<br/>is now flake.nix:3957 - while its renderer checks bounds only<br/>scripts/ci/render-egress-register.js:110-119"]
```

**Invariant:** the Loom path carries no secret: every Loom variable is classified `NON_SECRET`, and the only credential beside it, the Z.AI key the dream engine falls back to, is `DEVUSER_CLASS` by the Q8 exception (`../project/agentbox/config/custody/env-classes.json:537-542`, `../project/agentbox/config/custody/env-classes.json:227-230`). So custody X-1 step 1 changes nothing at this door in either mode.

**Drift (egress register vs the integration head):** the register is stamped `0919dc39a` (`../project/agentbox/config/egress-policy.json:51`), and its loom-door and zai-night rows cite manifest lines 2179 and 2163 and flake line 3438 (`../project/agentbox/config/egress-policy.json:231-232`, `../project/agentbox/config/egress-policy.json:355`). Those were the right lines at that stamp. At this revision the manifest has grown by more than three hundred lines since: line 2179 is a session seed's `worktree = true`, line 2163 is a session seed's `tool = "claude"` and `[dream_machine].llm_provider` is `agentbox.toml:2582` (`../project/agentbox/agentbox.toml:2582`); the façade default moved to `flake.nix:3957` (`../project/agentbox/flake.nix:3957`). The renderer's `--check` only tests that a cited line falls inside the file, so it cannot see the shift (`../project/agentbox/scripts/ci/render-egress-register.js:110-119`).

## AB-24.15 ADR-2140 — the vault stays here, the HP Loom serves a built generation reconciled at every boot

```mermaid
flowchart TB
    subgraph here["This host — the vault is canonical"]
        VAULT["visionGraph repository, [vault].repo<br/>HEAD is the only thing ever published<br/>agentbox/agentbox.toml:1181-1183"]
        BOOT["boot Phase 5d, background, timeout 1800, fail-open<br/>agentbox/config/entrypoint-unified.sh:1209-1219"]
        REC["hp-loom-reconcile.sh --mode off, check or full<br/>agentbox/scripts/hp-loom-reconcile.sh:91"]
        WT["clean detached worktree at HEAD, so uncommitted edits are never served<br/>agentbox/scripts/hp-loom-reconcile.sh:142"]
        VAL["ontomark validate --vault all, then ontomark build<br/>refused on errors, nothing built or sent<br/>agentbox/scripts/hp-loom-reconcile.sh:146, agentbox/scripts/hp-loom-reconcile.sh:152"]
        PRO["promote_vault_build then verify as visionGraph at HEAD<br/>agentbox/scripts/hp-loom-reconcile.sh:155"]
    end
    subgraph hp["The HP — a Loom host, no agentbox, no vault copy"]
        DATA["data/ swapped under the .promotion-in-flight marker,<br/>previous generation kept in data.rollback<br/>agentbox/scripts/hp-loom-reconcile.sh:185"]
        RELOAD["its own verifying reload_published_generation<br/>agentbox/scripts/hp-loom-reconcile.sh:174"]
        DOOR["Loom door port 8084 serves visionGraph@sha"]
    end
    VAULT --> BOOT --> REC
    REC -->|"served id equals HEAD"| CUR["state current, nothing recreated<br/>agentbox/scripts/hp-loom-reconcile.sh:119"]
    REC -->|"mode check and ids differ"| STALE["state stale, report only<br/>agentbox/scripts/hp-loom-reconcile.sh:120"]
    REC -->|"mode full and ids differ"| WT --> VAL --> PRO
    PRO -->|"stream to the HP over ssh on the 25G rail"| DATA --> RELOAD --> DOOR
    DOOR -->|"GET /loom/generation compared with HEAD"| REC
    subgraph notes["Invariants and drift"]
        direction TB
        N1["INVARIANT ADR-2140: the HP never reads the vault, it serves a built bundle whose identity is in the bundle,<br/>so the consumers of AB-24.1 see no change; the HP agentbox overlay and its compose file are retired<br/>docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md:1-15"]
        N2["a failure is recorded in the receipt and exits 0: unreachable, deferred, refused and failed<br/>never stop a container; --strict turns them into a non-zero exit<br/>agentbox/scripts/hp-loom-reconcile.sh:13-15"]
        N1 ~~~ N2
    end
```

**What it shows:** how the generation behind the Loom door reaches the HP. Every boot starts `hp-loom-reconcile.sh` in the background; it compares the committed vault HEAD with the id `/loom/generation` reports and, in `full` mode only, builds that HEAD from a clean worktree, promotes and verifies the bundle, streams it to the HP, swaps it under an in-flight marker and waits for the door to report the new id (`../project/agentbox/scripts/hp-loom-reconcile.sh:119-120`). **Why it is this way:** on 2026-10-13 the HP held only the Loom's built bundle plus a second agentbox that had drifted from this node; ADR-2140 makes the vault canonical on this host and the reconcile the one publisher, so a vault change cannot silently leave the door serving yesterday's corpus (`../project/agentbox/docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md:1-15`). The Python reloader on the HP is gone: the Loom's verifying reload is the Rust `reload_published_generation` binary, and the served bundle under `data/` is no longer tracked in the Loom repository.

**Drift:** ADR-2140 is `accepted`, `complete` and `live` (`../project/agentbox/docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md:5-7`), but the governing document never mentions the boot reconcile, the `[vault].loom_sync` key or the retired HP agentbox, and its model-swap section still reads as if the generation were staged by hand (`../project/agentbox/docs/GOVERNANCE-capabilities.md:309-311`).

## Audit qualification - 2026-09-07

The execution pass implements consumer verification in `ontology-retrieval.js::loomGenerationVerifier`: GET generation before every cache lookup, compare the configured pin and loaded digest/model/corpus, and validate identity headers on search/SPARQL responses. The local Loom route implementation preserves response bodies and adds those headers. This source is staged: the live façade was probed and still reports lexical generation 2026-08-22 versus semantic 2026-08-17, without loaded identity/embedding fields. A coordinated bundle/server rollout is required before activating the stricter client. See [execution evidence](../../estate-review/closeout/2026-09-07-execution-agentbox.md). The graph loader still uses the shared default graph, so ADR-2073 remains open. Generation reporting is not an automatic corpus reload.
