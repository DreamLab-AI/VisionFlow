---
id: AB-29
title: Metaharness router console — AoE dispatch plane phase 0
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2079, ADR-2080]
sources:
  - ../project/agentbox/config/model-router/artefacts.json
  - ../project/agentbox/config/model-router/console.mjs
  - ../project/agentbox/config/model-router/README.md
  - ../project/agentbox/config/harness-wrappers/router.sh
  - ../project/agentbox/scripts/model-router-fetch.sh
  - ../project/agentbox/scripts/aoe-seed-sessions.mjs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/services/agentbox-manifest/src/tui_read.rs
  - ../project/agentbox/services/agentbox-manifest/src/tui_sections.rs
verified_commit: 2c521c5bb
---

## AB-29.1 Composition — what is baked, and why the npm tarball lacks it

```mermaid
flowchart TB
    subgraph gap["Two measured upstream defects (ADR-2080 Context)"]
        T1["'@claude-flow/cli' npm tarball 'files' list excludes assets/model-router<br/>no seed corpus, KRR model, calibrator or OpenRouter alternates in the closure"]
        T2["ruflo task-embedder.js imports retired '@xenova/transformers'<br/>closure ships '@huggingface/transformers' 4.2.0 instead — import fails silently"]
    end
    subgraph manifest["Single pinned manifest"]
        ART["config/model-router/artefacts.json<br/>schema_version 1"]
        RUFLO["ruflo v3.38.20 @ e21aa352…<br/>12 files: seed-rows, seed-router.krr,<br/>3x calibrator, fastgrnn.safetensors, openrouter-alts…<br/>artefacts.json:5-11"]
        EMB["Xenova/all-MiniLM-L6-v2 q8<br/>HF revision 751bff37…, dim 384<br/>artefacts.json:12-19"]
        FILES["17 files total, each {dest,url,sha256,size}<br/>artefacts.json:20"]
    end
    subgraph readers["Two readers of the ONE manifest"]
        NIX["flake.nix modelRouterArtefacts<br/>flake.nix:111 — fetchurl per file, sha256-pinned"]
        FETCH["scripts/model-router-fetch.sh<br/>reads m.files, sha256sum-verifies, curl -fL, refuses hash mismatch"]
    end
    subgraph outputs["Two possible artefact locations"]
        BAKED["/opt/agentbox/model-router<br/>flake.nix:1734-1736 — rebuild class, byte-identical-when-off"]
        FALLBACK["$WORKSPACE/.agentbox/model-router<br/>pre-rebuild fallback"]
    end
    T1 --> ART
    T2 --> ART
    ART --> RUFLO --> FILES
    ART --> EMB --> FILES
    FILES --> NIX --> BAKED
    FILES --> FETCH --> FALLBACK
    subgraph notes["Invariants"]
        direction TB
        N1["INVARIANT: never edit hashes by hand — re-pin with a new upstream ref and recompute (artefacts.json:2 purpose string)"]
        N2["INVARIANT: fetch refuses any file whose recomputed sha256 differs from the manifest (model-router-fetch.sh hash-mismatch branch)"]
        N1 ~~~ N2
    end
```

## AB-29.2 Boot env export — [model_routing.neural] to AGENTBOX_MODEL_ROUTER_*

```mermaid
sequenceDiagram
    autonumber
    participant TOML as agentbox.toml<br/>[model_routing.neural]<br/>agentbox.toml:1038-1045
    participant EP as entrypoint-unified.sh<br/>config/entrypoint-unified.sh:1587
    participant DISK as /opt/agentbox/model-router<br/>or $WORKSPACE/.agentbox/model-router
    participant ENV as runtime-env file<br/>config/entrypoint-unified.sh:2317-2318

    EP->>TOML: _ab_toml_bool model_routing.neural enabled
    alt enabled = false
        EP->>ENV: _MRN_EXPORTS="" (nothing exported)
    else enabled = true
        EP->>TOML: read assets_dir, provider, quality_bar, cost_ceiling_usd_per_mtok, privacy_tier, trajectory
        EP->>DISK: test -f $_MRN_DIR/seed-router.krr.json
        alt baked dir missing the artefact
            EP->>DISK: fall back to $WORKSPACE/.agentbox/model-router<br/>config/entrypoint-unified.sh:1591-1593
        end
        EP->>ENV: export AGENTBOX_MODEL_ROUTER_ENABLED=1, _DIR, _PROVIDER,<br/>_QUALITY_BAR, _COST_CEILING_USD_PER_MTOK, _PRIVACY_TIER,<br/>_TRAJECTORY, _STATE_DIR<br/>config/entrypoint-unified.sh:1599-1606
        EP->>EP: echo readiness line naming artefact dir + provider + bar<br/>config/entrypoint-unified.sh:1608
    end
    Note over EP,ENV: DELIBERATELY NOT exported: CLAUDE_FLOW_ROUTER_* — the console sets those<br/>in its own process only (ADR-2080 D3), so this boot path can never re-route<br/>other ruflo work to an external provider
```

## AB-29.3 Seed → wrapper → console sequence — a task typed in the AoE `router` session

```mermaid
sequenceDiagram
    autonumber
    participant SEEDER as aoe-seed-sessions.mjs<br/>scripts/aoe-seed-sessions.mjs:110-114
    participant CFG as ~/.config/agent-of-empires/config.toml<br/>custom_agents.router
    participant AOE as AoE session `router` (tmux window 8)
    participant WRAP as router.sh<br/>config/harness-wrappers/router.sh:24
    participant CONS as console.mjs<br/>config/model-router/console.mjs

    SEEDER->>SEEDER: WRAPPER_SLUGS.router = {file: 'router.sh', detectAs: null}<br/>scripts/aoe-seed-sessions.mjs:113
    SEEDER->>CFG: customAgents.router = path.join(WRAPPER_DIR, 'router.sh')<br/>scripts/aoe-seed-sessions.mjs:272-273 (no detectAs alias — router is its own program)
    AOE->>WRAP: exec router.sh (session program for slug=router, agentbox.toml:1387-1391)
    WRAP->>WRAP: _die if console missing, node absent, or ruflo not resolvable<br/>router.sh:47,51,52
    WRAP->>WRAP: locate artefacts: $AGENTBOX_MODEL_ROUTER_DIR else<br/>/opt/agentbox/model-router else $WORKSPACE/.agentbox/model-router<br/>router.sh:58-64
    alt no seed-router.krr.json found anywhere
        WRAP-->>AOE: _die "run ./agentbox.sh model-router fetch"<br/>router.sh:65-68
    end
    alt provider=openrouter AND OPENROUTER_API_KEY empty AND egress on
        WRAP-->>AOE: _die "OPENROUTER_API_KEY is empty" — no silent fallback to another billing key<br/>router.sh:69-75
    end
    WRAP->>WRAP: export AGENTBOX_PROFILE, AGENTBOX_MODEL_ROUTER_PRIVACY_TIER="public" (pinned, not inherited)<br/>router.sh:79
    WRAP->>CONS: exec node console.mjs "$@"<br/>router.sh:84
    CONS->>CONS: TIER check — die unless "public"<br/>console.mjs:94
    CONS->>CONS: EGRESS_OFF (AGENTBOX_EGRESS=0) forces DRY=true<br/>console.mjs:99-102
    CONS->>CONS: resolveAssetsDir() verifies all 8 REQUIRED files exist<br/>console.mjs:128-141
    CONS->>CONS: loadEmbedder() — MiniLM q8 pipeline, offline (allowRemoteModels=false)<br/>console.mjs:174-183
    CONS-->>AOE: banner() — artefacts dir, corpus provenance, provider, quality bar<br/>console.mjs:195
    Note over WRAP,CONS: HARD-FAIL WRAPPER SEMANTICS: every precondition failure is a loud _die with a<br/>fix instruction, never a silent no-op or a fallback to a different billing key
```

## AB-29.4 Routing decision flow — embed, KRR predict, availability preflight, execute

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator (types a task)
    participant EMB as embed()<br/>console.mjs:174-183
    participant RTR as ruflo ModelRouter<br/>console.mjs:229 route()
    participant KRR as @metaharness/router (KRR/k-NN)<br/>inside the ruflo closure
    participant AVAIL as loadAvailability()<br/>console.mjs:255
    participant OR as OpenRouter API
    participant EXEC as execute()<br/>console.mjs:292
    participant LEDGER as console-ledger.jsonl<br/>console.mjs:304 ledger()

    OP->>EMB: task text
    EMB-->>RTR: MiniLM q8 embedding, mean-pooled, normalised (384-dim)
    RTR->>KRR: router.route(task, embedding)
    KRR-->>RTR: {model: tier, modelId, routedBy: "metaharness-krr",<br/>confidence, complexity, alternatives[]}
    RTR->>AVAIL: isAvailable(d.modelId) — 24h-cached GET /api/v1/models<br/>console.mjs:255-268
    alt picked slug retired/unlisted on OpenRouter
        AVAIL-->>RTR: not in live list
        RTR->>RTR: fallbackChain(d)[0] — same-tier ranked alt, then router<br/>alternatives by score, then tier ladder<br/>console.mjs:280-289
        RTR->>RTR: d.unavailablePick set to old id, d.routedBy suffixed with availability
    end
    RTR-->>OP: printDecision(d) — model, tier, provider, confidence, cost/MTok
    RTR->>LEDGER: append {type:"decision", ...d}
    alt --dry-run
        Note over EXEC: route only — no provider call, no execution receipt
    else execute
        RTR->>EXEC: execute(decision, prompt)
        EXEC->>OR: callAnthropicMessages({provider:"openrouter", model, prompt})
        alt HTTP 404/402/400 or "no longer available" (UNAVAILABLE_RE)<br/>console.mjs:290
            OR-->>EXEC: error
            EXEC->>LEDGER: recordOutcome(d, "escalated")
            loop up to 3 fallback candidates<br/>console.mjs:280 fallbackChain
                EXEC->>OR: retry with next candidate modelId
            end
        else ok
            OR-->>EXEC: completion + usage
        end
        EXEC-->>OP: printResult — output, tokens, cost, duration
        EXEC->>LEDGER: append {type:"result", tier, modelId, ok, usage, costUsd}
        RTR->>RTR: recordModelOutcome / recordModelOutcomeByModelId (bandit priors)<br/>console.mjs recordOutcome()
    end
    Note over KRR: INVARIANT: the query embedding MUST come from the same model the seed<br/>corpus was embedded with (Xenova/all-MiniLM-L6-v2) or every KRR prediction<br/>is meaningless (artefacts.json:12-19 "why")
```

## AB-29.5 Receipts, trajectory sink and the promotion-gate consumer

```mermaid
flowchart LR
    subgraph console["console.mjs"]
        DEC["decision row<br/>console.mjs:229 route()"]
        RES["result row<br/>console.mjs:292 execute()"]
        OUT["outcome row<br/>recordOutcome()"]
    end
    subgraph sinks["State dir: $WORKSPACE/.agentbox/model-router-state/"]
        LEDGER["console-ledger.jsonl<br/>console.mjs:304 ledger() — {v,ts,type,...} one line per event"]
        TRAJ["DRACO-shaped trajectory row<br/>written by ruflo itself via CLAUDE_FLOW_ROUTER_TRAJECTORY=1<br/>console.mjs:157 setDefault"]
        BANDIT["bandit priors (.swarm/model-router-state.json)<br/>updated by recordModelOutcome / recordModelOutcomeByModelId"]
        AVAILCACHE["openrouter-models.json<br/>24h live-tariff cache, console.mjs:255"]
    end
    subgraph gate["ADR-150 promotion-gate analyser (ruflo, unaltered)"]
        ANALYSE["router-parallel-analyze.mjs<br/>quality >2%, cost <1%, p95 latency <5%"]
    end
    subgraph spike["ADR-2079 §1 research spike (open work)"]
        SPIKE["one week of routed-vs-default measurements<br/>ADR-2079 Consequences: Phase 0 receipts ARE the raw material"]
    end
    DEC --> LEDGER
    RES --> LEDGER
    OUT --> LEDGER
    OUT --> BANDIT
    RTR2["router.route()"] -.-> TRAJ
    TRAJ --> ANALYSE
    LEDGER --> SPIKE
    ANALYSE --> SPIKE
    subgraph notes["Invariants"]
        direction TB
        N1["INVARIANT ADR-2080 D5: ruflo's promotion-gate analyser applies UNCHANGED —<br/>the console writes ruflo's own DRACO row shape, nothing is re-derived"]
        N2["DIVERGENCE (ADR-2079 open work): the fleet-wide dispatcher and privacy-tier<br/>scorecard in Decision §3 are NOT built — only this Phase 0 console exists"]
        N1 ~~~ N2
    end
```

## AB-29.6 Privacy/public-only invariant — the tier gate at every layer

```mermaid
stateDiagram-v2
    [*] --> WrapperCheck
    WrapperCheck: router.sh pins AGENTBOX_MODEL_ROUTER_PRIVACY_TIER="public"<br/>(router.sh:79) — a session cannot inherit any other value
    WrapperCheck --> ConsoleCheck
    ConsoleCheck: console.mjs reads TIER, lower-cased<br/>console.mjs:93-96
    ConsoleCheck --> Refused: TIER !== "public"
    ConsoleCheck --> EgressCheck: TIER === "public"
    Refused: die() — "this console dispatches to an external provider<br/>and only serves PUBLIC work. Personal/LAN-only → Loom sessions"
    Refused --> [*]
    EgressCheck: AGENTBOX_EGRESS=0 (ADR-2026 switch)?<br/>console.mjs:99-102
    EgressCheck --> DryRunForced: egress off
    EgressCheck --> NormalRun: egress on
    DryRunForced: DRY=true forced — route only, zero provider calls
    NormalRun: route + execute via OpenRouter
    DryRunForced --> [*]
    NormalRun --> [*]
    note right of Refused
        Manifest schema also constrains this at the config layer:
        privacy_tier accepts only "public" (ADR-2080 Decision §4,
        agentbox.toml:1043 comment "the ONLY accepted value")
    end note
    note right of NormalRun
        ADR-2079 §4: privacy tier is the FIRST routing axis, before cost.
        The raw Loom door (loom-raw seed) is never a routing fallback
        for this console — see AB-24.1/.9 for the Loom facade's own doors.
    end note
```

## AB-29.7 Failure modes — every hard-fail is loud, with a fix instruction

```mermaid
flowchart TD
    START(["router.sh execs"]) --> C1{"console.mjs readable?<br/>router.sh:46"}
    C1 -->|no| D1["_die: console missing — reinstall or rebuild<br/>router.sh:46-48"]
    C1 -->|yes| C2{"node on PATH?"}
    C2 -->|no| D2["_die: node is not on PATH<br/>router.sh:51"]
    C2 -->|yes| C3{"ruflo on PATH OR<br/>RUFLO_NODE_MODULES set?<br/>router.sh:52"}
    C3 -->|no| D3["_die: ruflo not found —<br/>console imports the router from the baked closure<br/>router.sh:52-54"]
    C3 -->|yes| C4{"seed-router.krr.json found in<br/>baked dir or fallback dir?<br/>router.sh:58-64"}
    C4 -->|no| D4["_die: artefacts not found —<br/>run ./agentbox.sh model-router fetch, or rebuild<br/>with [model_routing.neural].enabled=true<br/>router.sh:65-68"]
    C4 -->|yes| C5{"provider=openrouter AND<br/>OPENROUTER_API_KEY empty AND egress on?<br/>router.sh:69-75"}
    C5 -->|yes| D5["_die: OPENROUTER_API_KEY empty —<br/>no silent fallback to another billing key (N-01 posture)<br/>router.sh:70-74"]
    C5 -->|no| EXECC["exec node console.mjs"]
    EXECC --> C6{"privacy tier === public?<br/>console.mjs:94"}
    C6 -->|no| D6["die: privacy tier not public —<br/>use the Loom sessions instead<br/>console.mjs:94-96"]
    C6 -->|yes| C7{"all 8 REQUIRED artefact<br/>files resolve?<br/>console.mjs:128-141"}
    C7 -->|no| D7["die: router artefacts missing, lists every dir tried<br/>console.mjs:138-141"]
    C7 -->|yes| C8{"@huggingface/transformers<br/>entry point found?<br/>console.mjs:174-176"}
    C8 -->|no| D8["die: @huggingface/transformers not found under &lt;closure&gt;<br/>console.mjs:176"]
    C8 -->|yes| C9{"neuralRouterStatus().available?"}
    C9 -->|no| D9["die: neural router unavailable: &lt;reason&gt;<br/>console.mjs:193"]
    C9 -->|yes| RUN(["banner() + REPL or --once"])
    RUN --> C10{"execute() gets a 404/402/400 or<br/>'no longer available'? UNAVAILABLE_RE<br/>console.mjs:290"}
    C10 -->|yes| FB["recordOutcome escalated, then walk<br/>fallbackChain up to 3 candidates<br/>console.mjs:337-345"]
    C10 -->|no ok| DONE(["printResult + ledger + bandit outcome"])
    FB --> DONE
    subgraph legend["Never happens"]
        NEVER["a precondition failure that silently no-ops or falls back<br/>to a different billing key — every _die/die names the cause and the fix"]
    end
```

## AB-29.8 Surfaces — CLI, manifest catalogue gate and TUI field

```mermaid
flowchart TB
    subgraph cli["./agentbox.sh model-router <sub>"]
        CMD["cmd_model_router()<br/>agentbox.sh:1930"]
        FETCH2["fetch → exec model-router-fetch.sh<br/>agentbox.sh:1935"]
        CHECK2["check → exec model-router-fetch.sh --check<br/>agentbox.sh:1936"]
        STATUS2["status → exec node console.mjs --status<br/>agentbox.sh:1937"]
        ROUTE2["route 'task' → exec node console.mjs --once<br/>agentbox.sh:1938-1939"]
        CONSOLE2["console → exec harness-wrappers/router.sh<br/>agentbox.sh:1940"]
        DISPATCH["top-level case dispatch<br/>agentbox.sh:2123"]
    end
    subgraph catalogue["Manifest gate catalogue"]
        GATE["id: model-routing-neural<br/>gate: model_routing.neural.enabled<br/>apply_class: rebuild<br/>management-api/lib/system-manifest.js:109-111"]
    end
    subgraph tui["agentbox-manifest TUI"]
        READ["F('model_routing.neural.enabled', D::B(false), false)<br/>services/agentbox-manifest/src/tui_read.rs:127"]
        RENDER["[model_routing.neural] section renderer<br/>services/agentbox-manifest/src/tui_sections.rs:190-193"]
    end
    DISPATCH --> CMD
    CMD --> FETCH2
    CMD --> CHECK2
    CMD --> STATUS2
    CMD --> ROUTE2
    CMD --> CONSOLE2
    GATE -.->|"same dotted key model_routing.neural.enabled"| READ
    READ --> RENDER
    subgraph notes["Invariants"]
        direction TB
        N1["INVARIANT: apply_class=rebuild — toggling the gate alone does not bake the artefacts;<br/>a rebuild is required (fallback dir + fetch script bridge the gap pre-rebuild)"]
        N1
    end
```
