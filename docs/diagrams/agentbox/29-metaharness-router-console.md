---
id: AB-29
title: Metaharness router console — phase 0 on the agentbox-sessions plane
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2079, ADR-2080, ADR-2153]
sources:
  - ../project/agentbox/config/model-router/artefacts.json
  - ../project/agentbox/config/model-router/console.mjs
  - ../project/agentbox/config/model-router/retort-benchmarks.json
  - ../project/agentbox/config/model-router/README.md
  - ../project/agentbox/config/harness-wrappers/router.sh
  - ../project/agentbox/scripts/model-router-fetch.sh
  - ../project/agentbox/services/agentbox-sessions/src/contract.rs
  - ../project/agentbox/services/agentbox-sessions/src/plane.rs
  - ../project/agentbox/services/agentbox-sessions/src/tmux.rs
  - ../project/agentbox/config/session-plane/launch.json
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/services/agentbox-manifest/src/tui_read.rs
  - ../project/agentbox/services/agentbox-manifest/src/tui_sections.rs
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../project/agentbox/docs/adr/ADR-2079-aoe-is-the-dispatch-plane-of-a-fleet-model-router.md
verified_commit: 7208f7e0d749b44efcfe24bd503c567c9ff8f9bc
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
        NIX["flake.nix modelRouterArtefacts<br/>flake.nix:185 — fetchurl per file, sha256-pinned"]
        FETCH["scripts/model-router-fetch.sh<br/>reads m.files, sha256sum-verifies, curl -fL, refuses hash mismatch"]
    end
    subgraph outputs["Two possible artefact locations"]
        BAKED["/opt/agentbox/model-router<br/>flake.nix:2245-2246 — rebuild class, byte-identical-when-off"]
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
    participant TOML as agentbox.toml<br/>[model_routing.neural]<br/>agentbox.toml:1736-1746
    participant EP as entrypoint-unified.sh<br/>config/entrypoint-unified.sh:2308
    participant DISK as /opt/agentbox/model-router<br/>or $WORKSPACE/.agentbox/model-router
    participant ENV as runtime-env file<br/>config/entrypoint-unified.sh:3811

    EP->>TOML: _ab_toml_bool model_routing.neural enabled
    alt enabled = false
        EP->>ENV: _MRN_EXPORTS="" (nothing exported)
    else enabled = true
        EP->>TOML: read assets_dir, provider, quality_bar, cost_ceiling_usd_per_mtok, privacy_tier, trajectory
        EP->>DISK: test -f $_MRN_DIR/seed-router.krr.json
        alt baked dir missing the artefact
            EP->>DISK: fall back to $WORKSPACE/.agentbox/model-router<br/>config/entrypoint-unified.sh:2311-2313
        end
        EP->>ENV: export AGENTBOX_MODEL_ROUTER_ENABLED=1, _DIR, _PROVIDER,<br/>_QUALITY_BAR, _COST_CEILING_USD_PER_MTOK, _PRIVACY_TIER,<br/>_TRAJECTORY, _STATE_DIR<br/>config/entrypoint-unified.sh:2320-2327
        EP->>EP: echo readiness line naming artefact dir + provider + bar<br/>config/entrypoint-unified.sh:2329
    end
    Note over EP,ENV: DELIBERATELY NOT exported: CLAUDE_FLOW_ROUTER_* — the console sets those<br/>in its own process only (ADR-2080 D3), so this boot path can never re-route<br/>other ruflo work to an external provider
```

## AB-29.3 Seed → wrapper → console sequence — a task typed in the agentbox-sessions `router` session

```mermaid
sequenceDiagram
    autonumber
    participant PLANE as session plane reconciler<br/>services/agentbox-sessions/src/plane.rs:200
    participant CON as launch contract<br/>services/agentbox-sessions/src/contract.rs:68
    participant LJ as launch.json wrapper template<br/>config/session-plane/launch.json:13
    participant TMUX as tmux session `abs_router`<br/>services/agentbox-sessions/src/tmux.rs:27
    participant WRAP as router.sh<br/>config/harness-wrappers/router.sh:24
    participant CONS as console.mjs<br/>config/model-router/console.mjs

    PLANE->>CON: plan_for(seed) resolves the [[interaction_plane.session_seeds]] entry slug=router, tool=custom:router<br/>(agentbox.toml:2185-2189, plane.rs:200-201)
    CON->>CON: harness-wrappers/router.sh exists AND tool == custom:router, so kind = wrapper<br/>contract.rs:69-71, and a wrapper never receives a model (contract.rs:85)
    CON->>LJ: wrapper template argv = [{wrapper}], so the launch is router.sh itself<br/>launch.json:13-15
    PLANE->>TMUX: Plan.name = session_name(slug) = abs_router, owned by the plane<br/>plane.rs:220, tmux.rs:27-28
    TMUX->>WRAP: exec router.sh (env-scrubbed by plan_for, argv from the template)
    WRAP->>WRAP: _die if console missing, node absent, or ruflo not resolvable<br/>router.sh:47,51,52
    WRAP->>WRAP: locate artefacts: $AGENTBOX_MODEL_ROUTER_DIR else<br/>/opt/agentbox/model-router else $WORKSPACE/.agentbox/model-router<br/>router.sh:58-64
    alt no seed-router.krr.json found anywhere
        WRAP-->>TMUX: _die "run ./agentbox.sh model-router fetch"<br/>router.sh:65-68
    end
    alt provider=openrouter AND OPENROUTER_API_KEY empty AND egress on
        WRAP-->>TMUX: _die "OPENROUTER_API_KEY is empty" — no silent fallback to another billing key<br/>router.sh:69-75
    end
    WRAP->>WRAP: export AGENTBOX_PROFILE, AGENTBOX_MODEL_ROUTER_PRIVACY_TIER="public" (pinned, not inherited)<br/>router.sh:79
    WRAP->>CONS: exec node console.mjs "$@"<br/>router.sh:84
    CONS->>CONS: TIER check — die unless "public"<br/>console.mjs:94
    CONS->>CONS: EGRESS_OFF (AGENTBOX_EGRESS=0) forces DRY=true<br/>console.mjs:99-102
    CONS->>CONS: resolveAssetsDir() verifies all 8 REQUIRED files exist<br/>console.mjs:128-141
    CONS->>CONS: loadEmbedder() — MiniLM q8 pipeline, offline (allowRemoteModels=false)<br/>console.mjs:174-183
    CONS-->>TMUX: banner — artefacts dir, corpus provenance, provider, quality bar<br/>console.mjs:199
    Note over WRAP,CONS: HARD-FAIL WRAPPER SEMANTICS: every precondition failure is a loud _die with a<br/>fix instruction, never a silent no-op or a fallback to a different billing key
```

**Drift:** the Agent of Empires seeder this topic first drew (`scripts/aoe-seed-sessions.mjs`, `custom_agents.router`) is deleted, yet the wrapper header still says AoE registers and execs it (`../project/agentbox/config/harness-wrappers/router.sh:5-6`), as does the flake comment (`../project/agentbox/flake.nix:2244`) and the `agentbox.toml` seed comment "resolved by the seeder's WRAPPER_SLUGS" (`../project/agentbox/agentbox.toml:2187`).

**Tension (ADR-2079 vs ADR-2153):** ADR-2079, the AoE dispatch-plane record, is `superseded` by ADR-2153 (`../project/agentbox/docs/adr/ADR-2079-aoe-is-the-dispatch-plane-of-a-fleet-model-router.md:5-9`), which is now `accepted`, `implementation_status: partial`, `activation_status: live` and makes the router an ordinary seed (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`, `../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:75`), but its acceptance round trip excludes `router`, so only the clean launch above is evidenced, not a routed task (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:65`).

## AB-29.4 Routing decision flow — embed, KRR predict, availability preflight, execute

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator (types a task)
    participant EMB as embed()<br/>console.mjs:174-183
    participant RTR as route<br/>console.mjs:239
    participant KRR as @metaharness/router (KRR/k-NN)<br/>inside the ruflo closure
    participant AVAIL as loadAvailability<br/>console.mjs:284
    participant OR as OpenRouter API
    participant EXEC as execute<br/>console.mjs:320
    participant LEDGER as ledger<br/>console.mjs:332

    OP->>EMB: task text
    EMB-->>RTR: MiniLM q8 embedding, mean-pooled, normalised (384-dim)
    RTR->>KRR: router.route(task, embedding)
    KRR-->>RTR: {model: tier, modelId, routedBy: "metaharness-krr",<br/>confidence, complexity, alternatives[]}
    RTR->>AVAIL: isAvailable(d.modelId) — 24h-cached GET /api/v1/models<br/>console.mjs:284-297
    alt picked slug retired/unlisted on OpenRouter
        AVAIL-->>RTR: not in live list
        RTR->>RTR: fallbackChain(d)[0] — same-tier ranked alt, then router<br/>alternatives by score, then tier ladder<br/>console.mjs:308-316
        RTR->>RTR: d.unavailablePick set to old id, d.routedBy suffixed with availability
    end
    RTR-->>OP: printDecision(d) — model, tier, provider, confidence, cost/MTok
    RTR->>LEDGER: append {type:"decision", ...d}
    alt --dry-run
        Note over EXEC: route only — no provider call, no execution receipt
    else execute
        RTR->>EXEC: execute(decision, prompt)
        EXEC->>OR: callAnthropicMessages({provider:"openrouter", model, prompt})
        alt HTTP 404/402/400 or "no longer available" (UNAVAILABLE_RE)<br/>console.mjs:318
            OR-->>EXEC: error
            EXEC->>LEDGER: recordOutcome(d, "escalated")
            loop up to 3 fallback candidates<br/>console.mjs:374
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
        DEC["decision row<br/>console.mjs:239 route"]
        RES["result row<br/>console.mjs:320 execute"]
        OUT["outcome row<br/>recordOutcome()"]
    end
    subgraph sinks["State dir: $WORKSPACE/.agentbox/model-router-state/"]
        LEDGER["console-ledger.jsonl<br/>console.mjs:332 ledger — {v,ts,type,...} one line per event"]
        TRAJ["DRACO-shaped trajectory row<br/>written by ruflo itself via CLAUDE_FLOW_ROUTER_TRAJECTORY=1<br/>console.mjs:159 setDefault"]
        BANDIT["bandit priors (.swarm/model-router-state.json)<br/>updated by recordModelOutcome / recordModelOutcomeByModelId"]
        AVAILCACHE["openrouter-models.json<br/>24h live-tariff cache, console.mjs:284"]
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
        agentbox.toml:1744 comment "the ONLY accepted value")
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
    C7 -->|yes| C8{"@huggingface/transformers<br/>entry point found?<br/>console.mjs:175-177"}
    C8 -->|no| D8["die: @huggingface/transformers not found under the closure<br/>console.mjs:177"]
    C8 -->|yes| C9{"neuralRouterStatus().available?"}
    C9 -->|no| D9["die: neural router unavailable with the reason<br/>console.mjs:197"]
    C9 -->|yes| RUN(["banner() + REPL or --once"])
    RUN --> C10{"execute() gets a 404/402/400 or<br/>'no longer available'? UNAVAILABLE_RE<br/>console.mjs:318"}
    C10 -->|yes| FB["recordOutcome escalated, then walk<br/>fallbackChain up to 3 candidates<br/>console.mjs:369-380"]
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
        CMD["cmd_model_router<br/>agentbox.sh:2390"]
        FETCH2["fetch runs model-router-fetch.sh<br/>agentbox.sh:2395"]
        CHECK2["check runs model-router-fetch.sh --check<br/>agentbox.sh:2396"]
        STATUS2["status runs node console.mjs --status<br/>agentbox.sh:2397"]
        ROUTE2["route 'task' runs node console.mjs --once<br/>agentbox.sh:2398-2399"]
        CONSOLE2["console runs harness-wrappers/router.sh<br/>agentbox.sh:2400"]
        DISPATCH["top-level case dispatch<br/>agentbox.sh:2895"]
    end
    subgraph catalogue["Manifest gate catalogue"]
        GATE["id: model-routing-neural<br/>gate: model_routing.neural.enabled<br/>apply_class: rebuild<br/>management-api/lib/system-manifest.js:118-120"]
    end
    subgraph tui["agentbox-manifest TUI"]
        READ["F('model_routing.neural.enabled', D::B(false), false)<br/>services/agentbox-manifest/src/tui_read.rs:128"]
        RENDER["[model_routing.neural] section renderer<br/>services/agentbox-manifest/src/tui_sections.rs:190"]
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

## AB-29.9 Retort heuristics - external measurements bend the tier, never the price

```mermaid
sequenceDiagram
    autonumber
    participant KRR as router.route<br/>agentbox/config/model-router/console.mjs:242
    participant R as route<br/>agentbox/config/model-router/console.mjs:239
    participant RET as retort-benchmarks.json<br/>agentbox/config/model-router/retort-benchmarks.json:2
    participant D as printDecision<br/>agentbox/config/model-router/console.mjs:341

    Note over RET: LOADED ONLY IF PRESENT - fs.existsSync then parse, otherwise null,<br/>so the console runs identically without the file (console.mjs:194-195)
    KRR-->>R: tier, modelId, routedBy, confidence, complexity
    alt retort loaded AND complexity below RETORT_CX_ROUTINE 0.3 AND tier is not the cheapest
        R->>R: force the cheapest tier, routedBy gains +retort-h1 (console.mjs:252-255)
        Note over R: h1 - on the easy task EVERY model reached requirement_coverage 1.00,<br/>so cost and speed are the only differentiators<br/>(retort-benchmarks.json:12)
    else retort loaded AND complexity above RETORT_CX_HARD 0.6 AND tier is the cheapest
        R->>R: step up one tier, routedBy gains +retort-h2 (console.mjs:256-259)
    else
        R->>R: the KRR pick stands unchanged
    end
    R->>D: the decision, with the retort hint attached in JSON mode (console.mjs:342)
    Note over RET: EXTERNAL: these are adrianco/retort's measurements, not ours -<br/>the console trusts the RANKING and the cost ratios, never the absolute<br/>numbers, because Retort scores on its own rubric<br/>(retort-benchmarks.json:3, retort-benchmarks.json:9)
    Note over RET: every cost is AGENT-LEVEL, the whole agentic loop per run, which is why<br/>per-token list price is a poor proxy for per-task cost<br/>(retort-benchmarks.json:8)
```

**Invariant:** the Retort file changes only which tier is chosen; the quality bar, the cost ceiling and the privacy tier stay manifest values projected through `AGENTBOX_MODEL_ROUTER_*` (`../project/agentbox/agentbox.toml:1739`, `../project/agentbox/agentbox.toml:1743`, `../project/agentbox/agentbox.toml:1744`).

**Open:** the manifest now carries Retort's measured guidance as a comment on `quality_bar` (`../project/agentbox/agentbox.toml:1740-1742`) while the shipped default stays 0.50 (`../project/agentbox/agentbox.toml:1739`); nothing records whether the 0.25 to 0.35 band that comment recommends for routine work has been run against this estate's own corpus.
