---
id: AB-15
title: Capability gating, spend caps and consultants
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/SECURITY-profiles.md
adrs: [ADR-2020, ADR-2031, ADR-2033, ADR-2080, ADR-2153]
sources:
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/scripts/agentbox-config-validate.js
  - ../project/agentbox/services/agentbox-ops/src/cost_cap/mod.rs
  - ../project/agentbox/services/agentbox-ops/src/bin/tree-search-cap.rs
  - ../project/agentbox/management-api/lib/pay402.js
  - ../project/agentbox/management-api/routes/payments.js
  - ../project/agentbox/management-api/lib/llm-marketplace.js
  - ../project/agentbox/management-api/routes/llm-marketplace.js
  - ../project/agentbox/management-api/lib/headroom.js
  - ../project/agentbox/management-api/lib/headroom-mcp-tools.js
  - ../project/agentbox/management-api/routes/system.js
  - ../project/agentbox/mcp/consultants/shared/consultant-base.js
  - ../project/agentbox/mcp/consultants/shared/consult-advice.js
  - ../project/agentbox/docs/SECURITY-profiles.md
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../project/agentbox/services/agentbox-sessions/src/contract.rs
  - ../project/agentbox/config/session-plane/launch.json
  - ../project/agentbox/services/agentbox-manifest/src/system_one.rs
  - ../project/agentbox/mcp/consultants/shared/model-diversity.js
  - ../project/agentbox/mcp/consultants/shared/spawn-cli.js
  - ../project/agentbox/mcp/consultants/shared/memory-logger.js
  - ../project/agentbox/mcp/consultants/antigravity/server.js
  - ../project/agentbox/services/agentbox-manifest/src/main.rs
  - ../project/agentbox/services/agentbox-manifest/src/tomlval.rs
  - ../project/agentbox/services/agentbox-manifest/src/tui_write.rs
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/skills/tree-search-coder/SKILL.md
  - ../project/agentbox/skills/SKILL-DIRECTORY.md
  - ../project/agentbox/scripts/visioncraft-deepsec-gate.sh
  - ../project/agentbox/services/agentbox-manifest/src/deepsec_gate.rs
  - ../project/agentbox/docs/adr/ADR-2020-capability-gating.md
  - ../project/agentbox/docs/adr/ADR-2031-consultant-model-manifest-projection.md
  - ../project/agentbox/docs/adr/ADR-2033-deepsec-security-gate.md
  - ../project/agentbox/lib/npm-cli.nix
  - ../project/agentbox/mcp/servers/lib/ontology-retrieval.js
  - ../project/agentbox/scripts/ci/check-manifest-catalogue.js
  - ../project/agentbox/services/agentbox-ops/src/cost_cap/ledger.rs
  - ../project/agentbox/management-api/lib/authority.js
  - ../project/agentbox/tests/sovereign/authority.test.js
  - ../project/agentbox/skills/mcp.json
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/flake.nix
  - ../project/agentbox/config/harness-wrappers/router.sh
  - ../project/agentbox/docs/adr/ADR-2080-metaharness-router-console-under-aoe.md
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
---

## AB-15.1 Gate lattice — manifest to package set to supervisor to runtime trace
```mermaid
flowchart TD
    subgraph M["agentbox.toml — declared gates"]
        M1["[skills.tree_search_coder]<br/>agentbox.toml:1137"]
        M2["[toolchains].deepsec<br/>agentbox.toml:2353"]
        M3["[security.deepsec]<br/>agentbox.toml:2628"]
        M4["[dream_machine]<br/>agentbox.toml:2674"]
        M5["[payments]<br/>agentbox.toml:1987"]
        M6["[consultants.*]<br/>agentbox.toml:1743-1774"]
        M7["[claude_code].permission_mode<br/>+ permission_deny (ADR-2116)"]
        M8["[vault].cli (ADR-2107/2108)"]
    end
    subgraph N["Nix package set — rebuild class"]
        N1["flake.nix deepsecPkg closure"]
        N2["lib/npm-cli.nix makeNpmService"]
    end
    subgraph S["supervisord blocks — boot class"]
        S1["dream-engine program"]
        S2["management-api program"]
    end
    subgraph C["CATALOGUE entries<br/>management-api/lib/system-manifest.js:39"]
        C1["tree-search-coder<br/>apply_class rebuild<br/>system-manifest.js:424"]
        C2["deepsec<br/>gates toolchains.deepsec + security.deepsec.enabled<br/>apply_class rebuild<br/>system-manifest.js:94"]
        C3["dream-machine<br/>apply_class boot<br/>system-manifest.js:175"]
        C4["payments<br/>apply_class boot<br/>system-manifest.js:226"]
        C5["consultants<br/>apply_class boot<br/>system-manifest.js:247"]
        C6["claude-code-permissions<br/>apply_class boot<br/>system-manifest.js:262"]
        C7["instruction-tiers<br/>gate null, apply_class boot<br/>system-manifest.js:268"]
        C8["claude-cred-sync<br/>apply_class rebuild<br/>system-manifest.js:274"]
        C9["vault-cli<br/>apply_class rebuild<br/>system-manifest.js:415"]
    end
    R["runtime trace<br/>GET /v1/system stateOf()<br/>system-manifest.js:474"]

    M1 -->|"rebuild"| C1
    M2 -->|"rebuild — bakes npm closure"| N1
    N1 --> C2
    M3 -->|"live — policy read at run time"| C2
    M4 -->|"boot — entrypoint reconciles, dream-engine self-reads /etc/agentbox.toml"| S1
    S1 --> C3
    M5 -->|"boot"| S2
    S2 --> C4
    M6 -->|"boot"| S2
    S2 --> C5
    M7 -->|"boot — agentbox-manifest permissions-project every restart"| C6
    M7 -.->|"gate null, always projected"| C7
    M8 -->|"rebuild — bakes /opt/agentbox/bin/vault"| C9
    S2 -.->|"toolchains.claude_code, rebuild"| C8
    C1 --> R
    C2 --> R
    C3 --> R
    C4 --> R
    C5 --> R
    C6 --> R
    C7 --> R
    C8 --> R
    C9 --> R
    N2 -.->|"rebuild — aci_shell/tree_search_coder npm closures"| C1

    note1["PROPOSED ADR-2077: a procedure ADR carrying the exact five-build<br/>nix rebuild sequence - closure identity and runtime trace are proved<br/>separately, a gate that cannot pass is a named exception, and the<br/>receipt lands in docs/reference/gap-close-evidence"]
    C1 -.-> note1
```

## AB-15.2 A gated tool call — off, on, and spend-capped branches
```mermaid
sequenceDiagram
    autonumber
    participant Agent as Claude session
    participant Skill as tree-search-coder skill<br/>skills/tree-search-coder/SKILL.md
    participant CLI as tree-search-cap bin<br/>services/agentbox-ops/src/bin/tree-search-cap.rs:1
    participant Limiter as Limiter::reserve<br/>services/agentbox-ops/src/cost_cap/mod.rs:361
    participant Ledger as Ledger (flock)<br/>services/agentbox-ops/src/cost_cap/ledger.rs

    rect rgb(235,235,235)
    Note over Agent,Skill: Branch A — [skills.tree_search_coder].enabled = false
    Agent->>Skill: invoke tree-search-coder
    Skill->>CLI: tree-search-cap reserve --run r1 --estimate 0.10
    CLI->>Limiter: reserve_at(run_id, estimate, now)
    Limiter-->>CLI: Err CapabilityDisabled skill=tree_search_coder
    CLI-->>Skill: exit 3, JSON ok=false refused=capability_disabled
    Note over Skill,Agent: INVARIANT: byte-identical-when-off — no candidate dispatched, no kernel session, no ledger row written
    end

    rect rgb(220,240,220)
    Note over Agent,Ledger: Branch B — gate on, under cap
    Agent->>Skill: invoke tree-search-coder N=3
    Skill->>CLI: reserve --run r2 --estimate 0.10
    CLI->>Limiter: reserve_at("r2", 0.10, now)
    Limiter->>Ledger: with_state(exclusive flock)
    Ledger-->>Limiter: run.committed=0.00, outstanding=0.00
    Limiter-->>CLI: Ok Reservation id=res-... remaining_usd=0.40
    CLI-->>Skill: exit 0, reservation JSON
    Skill->>Skill: dispatch sparc:coder branch in fresh kernel
    Skill->>CLI: settle --run r2 --reservation res-... --actual 0.09
    CLI->>Limiter: settle_at(stub, 0.09, Completed, now)
    Limiter-->>CLI: Settlement committed_usd=0.09 remaining_usd=0.41
    CLI-->>Skill: exit 0
    end

    rect rgb(250,225,225)
    Note over Agent,Ledger: Branch C — gate on, spend-capped refusal
    Agent->>Skill: invoke tree-search-coder branch 6
    Skill->>CLI: reserve --run r3 --estimate 0.13
    CLI->>Limiter: reserve_at("r3", 0.13, now)
    Limiter->>Ledger: with_state — committed 0.38 + outstanding 0.00 + 0.13 above cap 0.50
    Limiter-->>CLI: Err SpendCapExceeded cap_usd=0.50 committed_usd=0.38 requested_usd=0.13
    CLI-->>Skill: exit 3, JSON refused=spend_cap_exceeded
    Note over Skill: INVARIANT: tree-search-coder never auto-routed, always carries spend_cap_usd (ADR-2020)
    end
```

## AB-15.3 Boot projection — manifest to catalogue to runtime state
```mermaid
sequenceDiagram
    autonumber
    participant TOML as agentbox.toml<br/>agentbox/agentbox.toml:2628
    participant Entry as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:3392-3393
    participant MB as agentbox-manifest bin<br/>agentbox/services/agentbox-manifest/src/main.rs:557
    participant Sup as supervisord<br/>dream-engine program
    participant API as management-api<br/>routes/system.js:44
    participant SM as system-manifest.js<br/>CATALOGUE + buildSystemView<br/>management-api/lib/system-manifest.js:515

    Note over Entry: boot — Phase reconciles every restart
    Entry->>MB: agentbox-manifest toml-string --manifest /etc/agentbox.toml --path consultants.antigravity.model
    MB->>TOML: tomlval::parse_file_lenient(&manifest)<br/>tomlval.rs:58 get()
    alt value present and is string
        MB-->>Entry: prints the model string, exit 0
    else missing, non-string, unparseable manifest
        MB-->>Entry: prints empty string, exit 0
        Note over MB: fail-open — toml-string always exits 0 (main.rs:19)
    end
    Entry->>Entry: export AGENTBOX_ANTIGRAVITY_MODEL only if not already set
    Entry->>Sup: reconcile [dream_machine] enabled — dream-engine reads /etc/agentbox.toml itself at start
    Note over Sup: apply_class boot — dream-engine self-reads the manifest,<br/>so a restart (not a hot edit) is what applies a dream_machine change
    Note over Entry: ADR-2118, same boot phase — entrypoint-unified.sh:3032 also runs<br/>agentbox-manifest instructions-project (CATALOGUE id instruction-tiers, gate null),<br/>composing ~/.claude/CLAUDE.md and ~/workspace/AGENTS.md / CLAUDE.md from<br/>config/instructions/ every restart, repo-authoritative, drift-checked

    Note over API: live — every GET /v1/system request
    API->>SM: buildSystemView(manifest, adapters)
    loop for each CATALOGUE entry
        SM->>SM: resolveGate(manifest, entry.gate) -- tomlval-equivalent dotted lookup<br/>system-manifest.js:460
        SM->>SM: stateOf(manifest, entry) -- on, off or available<br/>system-manifest.js:474
    end
    SM-->>API: apply_classes, core, surfaces, modules, counts
    API-->>API: reply.send (generated_at, ...view, execution)
    Note over SM: RESOLVED ADR-2069 (2026-09-05): catalogue drift now FAILS the build.<br/>scripts/ci/check-manifest-catalogue.js rejects any agentbox.toml boolean key that is neither<br/>catalogued nor in an explicit BASELINE, and the BASELINE is a ratchet — an entry that has since<br/>been catalogued, or has left the toml, fails too, so the list can only shrink. 16 baselined keys<br/>are real capabilities still owed a CATALOGUE entry and warn on every run (check-manifest-catalogue.js:167,225)
```

## AB-15.4 Apply-class lifecycle of a gate
```mermaid
stateDiagram-v2
    [*] --> Declared
    Declared: Declared in agentbox.toml, e.g. security.deepsec.enabled
    Declared --> LiveApplied: apply_class live -- operator edits, next request reads it
    Declared --> BootPending: apply_class boot -- entrypoint reconciles every restart
    Declared --> RebuildPending: apply_class rebuild -- changes the Nix image composition

    LiveApplied --> LiveApplied: read at operation time, no restart needed
    note right of LiveApplied
        e.g. security.deepsec fail_on, max_duration -- read per agentbox-manifest deepsec-gate run
    end note

    BootPending --> BootApplied: agentbox.sh restart -- entrypoint-unified.sh re-runs
    BootApplied --> BootPending: manifest edited again while running -- drift until next restart
    note right of BootApplied
        e.g. dream_machine.enabled -- dream-engine self-reads manifest at process start
    end note

    RebuildPending --> RebuildApplied: agentbox.sh rebuild -- flake.nix recomposes package set plus supervisor block
    note right of RebuildPending
        e.g. toolchains.deepsec, skills.tree_search_coder, skills.aci_shell
        operator MUST rebuild -- a restart alone leaves the old package set
    end note

    LiveApplied --> [*]
    BootApplied --> [*]
    RebuildApplied --> [*]
```

## AB-15.5 tree-search-coder spend cap — manifest to validator to enforcement
```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator
    participant Val as agentbox-config-validate.js<br/>agentbox/scripts/agentbox-config-validate.js:1715
    participant TOML as agentbox.toml<br/>[skills.tree_search_coder]<br/>agentbox.toml:1137-1144
    participant Skill as tree-search-coder SKILL.md<br/>agentbox/skills/tree-search-coder/SKILL.md:67
    participant CLI as tree-search-cap<br/>src/bin/tree-search-cap.rs:1
    participant Lim as Limiter<br/>cost_cap/mod.rs:330,336

    Op->>Val: node agentbox-config-validate.js agentbox.toml
    Val->>TOML: read max_candidates, per_branch_timeout_s, spend_cap_usd
    alt max_candidates undeclared or above 5
        Val-->>Op: W051 token spend scales linearly with N
    end
    alt spend_cap_usd not a positive number
        Val-->>Op: W052 no default-unlimited mode for tree-search
    end
    alt tree_search_coder.enabled and NOT code_interpreter.enabled
        Val-->>Op: E052 branch scoring requires kernel execution
    end
    Note over Val: INVARIANT: tree-search-coder always carries a spend_cap_usd (ADR-2020)

    Op->>Skill: explicit invocation policy — the directory requires opt-in<br/>(SKILL-DIRECTORY.md:65), see AB-22.12 for the recommendation boundary
    Skill->>CLI: tree-search-cap config
    CLI->>Lim: CapConfig::load(manifest)
    Lim-->>CLI: enabled, spend_cap_usd=0.50, max_candidates=5, per_branch_timeout_s=60
    loop candidate 0..N-1 (max_candidates)
        Skill->>CLI: reserve --run RUN_ID --estimate E_i
        CLI->>Lim: reserve_at(RUN_ID, E_i, now)
        alt admitted at least max_candidates
            Lim-->>CLI: Err CandidateLimitExceeded
            CLI-->>Skill: exit 3
            Note over Skill: halt loop, return best-so-far candidate
        else committed+outstanding+E_i above cap
            Lim-->>CLI: Err SpendCapExceeded
            CLI-->>Skill: exit 3
            Note over Skill: halt loop, return best-so-far candidate (SKILL.md:70 enforced spend_cap_usd)
        else granted
            Lim-->>CLI: Ok Reservation
            CLI-->>Skill: exit 0
            Skill->>Skill: sparc:coder generates candidate in fresh KernelSession (ADR-018)
            opt branch exceeds per_branch_timeout_s mid-run
                Skill->>Lim: guard_branch(reservation, now) via CLI wrapper
                Lim-->>Skill: Err BranchTimeout
            end
            Skill->>CLI: settle --run RUN_ID --reservation res-... --actual A_i [--failed]
            CLI->>Lim: settle_at(stub, A_i, outcome, now)
            Lim-->>CLI: Settlement committed_usd, remaining_usd, overran_estimate
        end
    end
    Skill-->>Op: winning candidate, scored by assertion-pass count, tie-break shortest code
```

## AB-15.6 Full HTTP 402 challenge, payment, retry, settle
```mermaid
sequenceDiagram
    autonumber
    participant Agent as Agentbox consumer<br/>[payments.consumer]<br/>agentbox.toml:2006
    participant Merchant as External merchant endpoint
    participant Classify as pay402.classify()<br/>lib/pay402.js:165
    participant Pay as /v1/pay/* routes<br/>routes/payments.js:147-655
    participant Pod as solid-pod-rs<br/>127.0.0.1:8484

    Agent->>Merchant: original request
    Merchant-->>Agent: 402 Payment Required<br/>body.accepts[] or X-Pay-Currency: sats
    Agent->>Classify: classify({status:402, headers, body})
    alt agentbox-ledger — X-Pay-Currency sats + deposit_endpoint, or accepts[] entry
        Classify-->>Agent: scheme agentbox-ledger, payable = CONSUMER_ENABLED==="true"
        Note over Classify: both header and accepts[] present and cost_sats != accepts amount<br/>-> scheme unknown, reason amount-mismatch (pay402.js:220)
    else x402 — integer x402Version + accepts[] with scheme/network
        Classify-->>Agent: scheme x402, payable false (version!=1 -> reason unsupported-version)
    else l402 — status 401 or 402, WWW-Authenticate L402 or LSAT
        Classify-->>Agent: scheme l402, payable false always (macaroon+invoice required, ln-prefix checked)
    else unknown
        Classify-->>Agent: scheme unknown, payable false, reason null — terminal, fail-closed
    end

    rect rgb(220,240,220)
    Note over Agent,Pod: payable path — [payments.consumer].enabled required (agentbox.toml:2007)
    Agent->>Pay: POST /v1/pay/estimate {endpoint, units}
    Pay-->>Agent: 200 estimated_sats, hold_sats = ceil(estimated*HOLD_BUFFER_RATIO)
    Agent->>Pay: GET /v1/pay/balance (NIP-98 auth)
    Pay->>Pod: GET /pay/.balance
    Pod-->>Pay: balance_sats
    Pay-->>Agent: 200 balance_sats, dream_balance
    alt balance_sats at least required
        Agent->>Merchant: retry original request with payment proof
        Merchant-->>Agent: 200 OK — settled
    else insufficient balance
        Agent->>Pay: POST /v1/pay/deposit
        Pay-->>Agent: 501 deposit-not-served, deposit_url, credits nothing (routes/payments.js:303-308)
        Note over Agent,Pod: a deposit is a verified MRC20 body POSTed, NIP-98 signed for the pod's own URL,<br/>to the pod's /pay/.deposit — this route cannot forward one (routes/payments.js:278-285)
        Agent->>Pod: POST /pay/.deposit, MRC20 body, outside agentbox
    end
    end
    Note over Agent,Pay: DEBT — the authority classification table names payment_settlement<br/>zero-tolerance as an irreversible above-threshold sats spend (agentbox.toml:1497),<br/>but no production path passes that actionClass to the gate. The only call sites<br/>are tests/sovereign/authority.test.js:129 and authority.test.js:248, and routes/payments.js never<br/>reaches management-api/lib/authority.js:226. see AB-14.11
    Note over Agent,Pay: DEBT — spend is accounted in three stores that never reconcile:<br/>the flock-guarded cost_cap ledger file (services/agentbox-ops/src/cost_cap/ledger.rs:119,136),<br/>the pod-held sats balance behind GET /v1/pay/balance (management-api/routes/payments.js:205),<br/>and the marketplace token_budget carried on a grant (management-api/lib/llm-marketplace.js:85).<br/>No surface sums them, so no single number answers what this container has spent
    Note over Agent: DIVERGENCE: [payments.consumer].enabled = false by default (agentbox.toml:2007),<br/>[payments.broadcast].enabled = false and well_known = false (agentbox.toml:2016-2017) — see AB-11.15
```

## AB-15.7 routes/payments.js — GET info, GET balance, POST deposit
```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant R as paymentRoutes<br/>routes/payments.js:141
    participant Pod as solid-pod-rs<br/>127.0.0.1:8484

    Note over R: all routes behind global onRequest auth hook — bearer or NIP-98 (payments.js:31)

    C->>R: GET /v1/pay/info<br/>payments.js:147
    R->>Pod: GET /pay/.info
    alt pod-rs reachable
        Pod-->>R: 200 upstream chains
        R-->>C: 200 chains, base_cost_sats, dream_per_sat, tiers, operator_did
    else pod-rs unreachable
        Note over R: logged, falls through to local config — never a 502 here (payments.js:174-180)
        R-->>C: 200 chains=[btc] (local default), tiers from TIER_MULTIPLIERS
    end

    C->>R: GET /v1/pay/balance<br/>payments.js:203
    alt callerDid(req) is null — bearer auth, no identity
        R-->>C: 401 identity-required
    else NIP-98 identity present
        R->>Pod: GET /pay/.balance (Authorization forwarded)
        alt pod-rs error
            Pod-->>R: non-2xx or unreachable
            R-->>C: err.statusCode or 502 payment-service-error
        else ok
            Pod-->>R: 200 balance_sats
            R-->>C: 200 did, balance_sats, dream_balance=balance_sats*DREAM_PER_SAT, X-Balance header
        end
    end

    C->>R: POST /v1/pay/deposit<br/>payments.js:286
    R-->>C: 501 deposit-not-served, message, deposit_url = POD_PUBLIC_URL/pay/.deposit or null (payments.js:303-308)
    Note over R: calls nothing — the pod accepts only a verified MRC20 body NIP-98 signed for its own URL,<br/>which a header signed for /v1/pay/deposit cannot be (payments.js:278-285)
```

## AB-15.8 routes/payments.js — POST estimate, POST buy, POST withdraw
```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant R as paymentRoutes<br/>routes/payments.js:141
    participant Pod as solid-pod-rs<br/>127.0.0.1:8484

    C->>R: POST /v1/pay/estimate {endpoint, units}<br/>payments.js:315
    alt TIER_MULTIPLIERS[endpoint] undefined
        R-->>C: 400 unknown-tier
    else valid tier
        R->>R: perUnitSats = BASE_COST_SATS*multiplier<br/>hold_sats = ceil(estimated*HOLD_BUFFER_RATIO)
        R-->>C: 200 estimated_sats, hold_sats, dream_tokens, breakdown (local computation, no pod-rs call)
    end

    C->>R: POST /v1/pay/buy {amount}<br/>payments.js:402
    alt callerDid(req) is null
        R-->>C: 401 identity-required
    else identified
        R->>Pod: GET /pay/.balance
        alt pod-rs error
            Pod-->>R: non-2xx or unreachable
            R-->>C: err.statusCode or 502 payment-service-error
        else currentBalance below costSats (ceil(amount/DREAM_PER_SAT))
            R-->>C: 402 insufficient-balance, required_sats, balance_sats, X-Cost header
        else sufficient balance
            R->>Pod: POST /pay/.buy {dream_amount, cost_sats}
            alt pod-rs error
                Pod-->>R: non-2xx or unreachable
                R-->>C: err.statusCode or 502 payment-service-error
            else ok
                Pod-->>R: 200 new_sat_balance
                R-->>C: 200 purchased=true, dream_amount, cost_sats, new_sat_balance, new_dream_balance
            end
        end
    end

    C->>R: POST /v1/pay/withdraw {amount}<br/>payments.js:525
    alt callerDid(req) is null
        R-->>C: 401 identity-required
    else floor(amount/DREAM_PER_SAT) at most 0
        R-->>C: 400 amount-too-small
    else identified and non-zero
        R->>Pod: GET /pay/.balance
        alt pod-rs error
            Pod-->>R: non-2xx or unreachable
            R-->>C: err.statusCode or 502 payment-service-error
        else currentDreamBalance below amount
            R-->>C: 402 insufficient-dream-balance, required_dream, dream_balance
        else sufficient DREAM balance
            R->>Pod: POST /pay/.withdraw {dream_amount, sats_received}
            alt pod-rs error
                Pod-->>R: non-2xx or unreachable
                R-->>C: err.statusCode or 502 payment-service-error
            else ok
                Pod-->>R: 200 new_sat_balance
                R-->>C: 200 withdrawn=true, dream_burned, sats_received, new_sat_balance, new_dream_balance
            end
        end
    end
```

## AB-15.9 LLM marketplace — advertise, discover, request, grant, receipt, revoke
```mermaid
sequenceDiagram
    autonumber
    participant Prov as Provider
    participant Cons as Consumer
    participant R as llmMarketplaceRoutes<br/>routes/llm-marketplace.js:58
    participant OB as Orderbook<br/>lib/llm-marketplace.js:220
    participant Gate as authorityGate.guard<br/>see AB-14.x

    Prov->>R: POST /v1/llm/advertise {model, context_window, endpoint}<br/>routes/llm-marketplace.js:99
    alt validateAdvertisement fails
        R-->>Prov: 422 validation_failed
    else valid
        R->>OB: addAdvertisement(pubkey, body) -- key pubkey:model, replaceable (kind 38300)
        R-->>Prov: 200 event, urn, stored=true
    end

    Cons->>R: GET /v1/llm/discover ?min_context_window&max_cost_per_m_token&capabilities<br/>:180
    R->>OB: findMatches(filter) or getAdvertisements(provider)
    OB-->>R: matching advertisements
    R-->>Cons: 200 advertisements, count

    Cons->>R: POST /v1/llm/request {token_budget, min_capabilities, ...}<br/>:221 (kind 38301)
    R->>OB: findMatches(body)
    OB-->>R: candidate ads
    R-->>Cons: 200 event, matches, count

    alt provider grants
        Prov->>R: POST /v1/llm/grant {request_event_id, grantee_pubkey, model, token_allocation}<br/>:275
        R->>OB: addGrant(grantId, {providerPubkey, consumerPubkey, tokenAllocation, tokensUsed:0, expiresAt})
        R-->>Prov: 201 event (kind 38302), grant_id, urn
    else provider denies
        Prov->>R: POST /v1/llm/deny {request_event_id, grantee_pubkey, reason}<br/>:341
        R-->>Prov: 200 event (kind 38303)
    end

    Cons->>R: POST /v1/llm/receipt {grant_id, consumer_pubkey, model, tokens_used}<br/>:378
    R->>OB: recordUsage(grant_id, tokens_used)
    alt tokensUsed + tokens_used above tokenAllocation, or grant not found
        OB-->>R: false
        R-->>Cons: 402 budget_exceeded
    else within allocation
        OB-->>R: true, running total updated
        R-->>Cons: 200 event (kind 38304), accepted=true, urn
    end

    Prov->>R: POST /v1/llm/revoke {grant_id, reason}<br/>:438
    opt authorityEnabled -- mandate_revoke is a zero-tolerance action class
        R->>Gate: guard(actionClass: mandate_revoke) -- kind-31402 ActionRequest, blocks for signed kind-31403 (see AB-14.x)
        alt gate.decision != allow
            Gate-->>R: DENY -- fail-closed, no timeout/reject/unverified escape
            R-->>Prov: 403 authority_denied
        end
    end
    R->>OB: revokeGrant(grant_id)
    R-->>Prov: 200 event (kind 38305), revoked=true

    Note over R: GET /v1/llm/grants (routes/llm-marketplace.js:534, pruneExpired + getActiveGrants(pubkey)) and<br/>GET /v1/llm/stats (routes/llm-marketplace.js:556, orderbook.stats() + KINDS) are read-only summaries, not shown above
```

## AB-15.10 headroom compression-capacity model and its 3 MCP tools
```mermaid
sequenceDiagram
    autonumber
    participant MCP as MCP caller
    participant Tools as headroom-mcp-tools.js<br/>handleTool()<br/>lib/headroom-mcp-tools.js:111
    participant HR as headroom.js<br/>compress/retrieve/stats<br/>lib/headroom.js:1
    participant Addon as headroom_napi.node<br/>(Rust N-API addon)
    participant TOML as [compression] [compression.slots]<br/>agentbox.toml

    Note over HR: DIVERGENCE: this is a context-COMPRESSION capacity model<br/>(ttl_minutes, max_entries, target_ratio), not a USD spend/budget cap —<br/>no financial "budget" field exists in headroom.js (lib/headroom.js:70-84)

    MCP->>Tools: call headroom_compress {content, content_type}<br/>:65
    Tools->>HR: detectContentType(content) -- when content_type="auto"
    HR->>Addon: addon.detectContentType(content)
    Tools->>HR: smartCrush, compressLog or compressDiff(content) by detected type
    HR->>HR: _readManifestConfig() -- caches [compression] from loadManifest()<br/>:56 enabled, backend, ttl_minutes, max_entries, target_ratio
    HR->>TOML: read [compression.slots].{memory,pods,beads,orchestrator}
    Note over HR: INVARIANT: slots.events is hard-coded false always -- the audit trail is never compressed (headroom.js:15,79)
    HR->>Addon: addon.smartCrush(content, {targetRatio}) (or compressLog/compressDiff)
    alt addon absent, compression.enabled=false, or ratio at least 1.0
        HR-->>Tools: passthrough {compressed:false, content unchanged}
        Note over HR: fail-open contract -- headroom.js:6-9
    else beneficial compression
        Addon-->>HR: {compressed, ratio, ccrEntries}
        HR-->>Tools: {compressed:true, ratio, ccrEntries}
    end
    Tools-->>MCP: {compressed, ratio, ccr_entries}

    MCP->>Tools: call headroom_retrieve {hash}<br/>:34
    Tools->>HR: ccrRetrieve(hash)
    HR->>Addon: addon.ccrRetrieve(hash)
    alt entry found
        Addon-->>HR: Buffer
        HR-->>Tools: decompressed content
        Tools-->>MCP: {content, size_bytes}
    else expired or evicted
        Addon-->>HR: null
        HR-->>Tools: null
        Tools-->>MCP: {error: expired}
    end

    MCP->>Tools: call headroom_stats {}<br/>:53
    Tools->>HR: ccrStats()
    HR->>Addon: addon.ccrStats()
    Addon-->>HR: entries, bytes_stored, hit_count, miss_count
    HR-->>Tools: normalised stats
    Tools-->>MCP: {entries, bytes_stored, hit_count, miss_count, hit_rate}
```

## AB-15.11 Consultant model projection at boot — env wins, TUI preserves (ADR-2031)
```mermaid
sequenceDiagram
    autonumber
    participant PreBoot as Pre-boot environment
    participant Entry as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:3392-3393
    participant MB as agentbox-manifest toml-string<br/>services/agentbox-manifest/src/main.rs:557
    participant TOML as agentbox.toml<br/>[consultants.antigravity]<br/>agentbox.toml:1754-1758
    participant Reg as skills/mcp.json registry default
    participant Srv as antigravity server.js<br/>mcp/consultants/antigravity/server.js:31

    rect rgb(220,240,220)
    Note over PreBoot,Entry: branch A — AGENTBOX_ANTIGRAVITY_MODEL set before boot
    Entry->>Entry: [ -z "${AGENTBOX_ANTIGRAVITY_MODEL:-}" ] is false -- skip projection
    Note over Entry: INVARIANT: a non-empty pre-boot env override wins (ADR-2031)
    end

    rect rgb(230,230,250)
    Note over Entry,TOML: branch B — env unset, manifest declares a model
    Entry->>MB: agentbox-manifest toml-string --manifest /etc/agentbox.toml --path consultants.antigravity.model
    MB->>TOML: tomlval::get(&cfg, "consultants.antigravity.model")
    TOML-->>MB: Value::String("gemini-4-hydrogen")
    MB-->>Entry: prints "gemini-4-hydrogen", exit 0
    Entry->>Entry: export AGENTBOX_ANTIGRAVITY_MODEL="gemini-4-hydrogen"
    end

    rect rgb(250,235,215)
    Note over Entry,TOML: branch C — env unset, manifest missing/non-string/unparseable
    Entry->>MB: agentbox-manifest toml-string ...
    MB->>TOML: parse_file_lenient -- unreadable or key absent
    MB-->>Entry: prints "" (empty), exit 0 -- fail-open (main.rs:19)
    Entry->>Entry: export AGENTBOX_ANTIGRAVITY_MODEL="" -- registry default applies downstream
    end

    Entry->>Srv: spawn consultant-antigravity with AGENTBOX_ANTIGRAVITY_MODEL in env
    Srv->>Srv: const MODEL = process.env.AGENTBOX_ANTIGRAVITY_MODEL, falling back to 'gemini-4-hydrogen' (server.js:31)
    Note over Srv,Reg: registry and server defaults both name Hydrogen (skills/mcp.json:343, server.js:31), the registry auth default is api_key (skills/mcp.json:344).<br/>Setup-template and URL-context defaults still name Flash (ADR-2031-consultant-model-manifest-projection.md:94-99).
    Srv->>Srv: authMode reads AGENTBOX_ANTIGRAVITY_AUTH_MODE, anything but oauth is api_key (server.js:34-36)<br/>the entrypoint projects it from [providers.gemini].auth_mode when the Antigravity CLI is enabled<br/>(entrypoint-unified.sh:3381-3384), and the shipped manifest sets oauth (agentbox.toml:1607), ADR-2214
    alt auth mode oauth
        Srv->>Srv: syncOAuthToken copies the primary sign-in token into AGY_HOME, never links it (server.js:38-61)<br/>no key and no provider override are passed, so agy uses the subscription session (server.js:64-66)
    else api_key
        Srv->>Srv: cliAuth chooses GEMINI_API_KEY, then GOOGLE_API_KEY, then GOOGLE_GEMINI_API_KEY (server.js:68-82)<br/>a key selects the Gemini provider and custom Hydrogen model, without one the CLI uses cached OAuth in AGY_HOME
    end
    Srv->>Srv: rates(now) returns source-configured API-equivalent estimates (server.js:85-93)
    alt Hydrogen
        Srv-->>Srv: flat configured rates, no dated multiplier
    else legacy Flash
        Srv-->>Srv: configured rates double on 2027-01-01
    else unknown model
        Srv-->>Srv: null rates and cost_usd null (server.js:129-132)
    end
    Note over Srv: healthCheck runs --version and models with the same auth settings.<br/>It is discovery, not proof of key validity, quota or successful inference (server.js:137-168).

    rect rgb(245,220,220)
    Note over TOML: TUI save path — tui_write.rs:37 run()
    Note over TOML: if state has no "consultants.antigravity.model" key, carry forward<br/>the EXISTING manifest's value rather than overwrite it (tui_write.rs:37-48)
    Note over TOML: INVARIANT: a TUI save never resets an operator's model (ADR-2031)
    end
```

## AB-15.12 Consultant consult() call — consultant-base.js, model-diversity, memory-logger
```mermaid
sequenceDiagram
    autonumber
    participant Caller as Coordinator (Claude session)
    participant Base as BaseConsultant._handleConsult<br/>consultant-base.js:275
    participant Onto as ontoBrain().ask<br/>ontology-retrieval.js
    participant Spawn as spawnCli<br/>spawn-cli.js:36
    participant CLI as agy, codex or z (vendor CLI)
    participant Div as model-diversity.js<br/>verificationRecord/selectVerifier<br/>model-diversity.js:208,162
    participant Log as MemoryLogger.log<br/>memory-logger.js:67
    participant API as management-api<br/>/v1/agent-events/emit

    Caller->>Base: CallToolRequest consult {question, context_excerpt, producer_family, ontology_context}
    Base->>Base: validate question non-empty (consultant-base.js:276)
    opt ontologyAugmentEnabled(args) -- args.ontology_context===true or CONSULT_ONTOLOGY_AUGMENT=1
        Base->>Onto: ask({query, model_tier:'sonnet', mode:'expand', max_tokens:1500})
        alt ontology call throws
            Onto-->>Base: fail-open -- proceed ungrounded (consultant-base.js:298 catch)
        else turtle returned
            Onto-->>Base: {turtle, seed_iris, tokens_used, degraded}
            Base->>Base: prepend Turtle to context_excerpt, coordinator's excerpt preserved whole
        end
    end
    Base->>Spawn: this.callConsult({question, context_excerpt, format}) under<br/>_withTimeout(timeout_ms)
    Spawn->>CLI: spawn(cmd, args, env: scrubbed + PASSTHROUGH_ENV TLS/proxy vars)<br/>spawn-cli.js:47,51
    alt exit code non-zero or SIGKILL on timeout
        CLI-->>Spawn: {code!=0, killed, stderr}
        Spawn-->>Base: throws -- memlog.log({ok:false, error}), rethrow
    else success
        CLI-->>Spawn: {stdout, code:0}
        Spawn-->>Base: {response, model, tokens, cost_usd, citations}
    end
    Base->>Base: mint consultation_urn via uris.mint(kind:'activity', payload:{consultant,question})<br/>(ADR-013 URN grammar)
    opt args.producer_family set -- REC-8 anti-fox closure verification
        Base->>Div: verificationRecord({producerFamily, verifier:this.name, task:'closure-verification'})
        Div-->>Base: {producer_family, verifier_family, anti_fox_ok}
        alt anti_fox_ok===false
            Base->>Base: logger.error -- same-family self-verification WARNING (:360-364)
        end
    end
    Base->>Log: memlog.log({ok:true, question, model, tokens, cost_usd, latency_ms, verification})
    opt AGENTBOX_INTELLIGENCE_DIR set and response_len is number
        Log->>Log: _writeIntelligenceSignal -- ADR-043 QualitySignal JSON (memory-logger.js:87-113)
    end
    Base->>API: POST /v1/agent-events/emit (Bearer MANAGEMENT_API_KEY, else TASK_ACTION_TOKEN) -- fire-and-forget, best-effort<br/>(consultant-base.js:13-14, consultant-base.js:46-50)
    Base-->>Caller: envelope {ok, response, model, tokens, cost_usd, citations, consultation_urn, verification}
```

## AB-15.13 deepsec gate — host adapter, policy projection, hand-off to the VisionCraft executor
```mermaid
sequenceDiagram
    autonumber
    participant Op as visioncraft caller
    participant Wrap as visioncraft-deepsec-gate.sh<br/>scripts/visioncraft-deepsec-gate.sh:1-8
    participant Proj as agentbox-manifest deepsec-gate<br/>services/agentbox-manifest/src/deepsec_gate.rs:114
    participant TOML as [security.deepsec]<br/>agentbox.toml:2628-2640
    participant VC as visioncraft deepsec<br/>executor owned by the VisionCraft package

    Op->>Wrap: visioncraft-deepsec-gate.sh gate flags
    alt agentbox-manifest not on PATH
        Wrap-->>Op: exit 78 -- rebuild or install the pinned host tools (:4-6)
    else present
        Wrap->>Proj: exec agentbox-manifest deepsec-gate with the same arguments (:8)
    end
    Proj->>Proj: refuse a caller-supplied --policy-json, the host owns it (deepsec_gate.rs:115-120)
    Proj->>TOML: read AGENTBOX_CONFIG, else /etc/agentbox.toml (deepsec_gate.rs:121-125)
    alt explicit manifest missing or unreadable
        Proj-->>Op: refuse, paid review not run (deepsec_gate.rs:136-138)
    else default manifest absent
        Proj->>Proj: policy_source defaults (deepsec_gate.rs:133-135)
    end
    Proj->>Proj: typed projection, an unknown field or a wrong type is an error, never a silent default (deepsec_gate.rs:54-81)
    Proj->>Proj: DEEPSEC_GATE_KEY environment overrides win over the manifest, credentials by NAME only (deepsec_gate.rs:82-104)
    Proj->>VC: exec visioncraft deepsec --policy-json JSON plus the caller flags, replacing the process (deepsec_gate.rs:157-162)
    alt visioncraft not installed
        Proj-->>Op: error, no fallback review ran (deepsec_gate.rs:163)
    else
        VC-->>Op: route selection, findings validation, output limits, deadline, receipt and exit code
    end
    Note over Proj,VC: EGRESS and route rules (the operator's own CLI session, a named direct vendor, the LAN Loom) are<br/>executed inside the VisionCraft package, not in this repository - see the visioncraft area topics.<br/>The host reads typed TOML and passes credential names only (docs/adr/ADR-2033-deepsec-security-gate.md:75-91)
    Note over Op: HISTORICAL ACCEPTANCE ADR-2033: the 2026-10-30 close-out records activation live from a real --diff receipt (result BLOCK, deepsec 2.3.10, docs/adr/ADR-2033-deepsec-security-gate.md:543-548),<br/>but the 2026-11-03 amendment moved the gate to VisionCraft and its front matter says activation staged again,<br/>historical receipts describing the retired shell wrapper (docs/adr/ADR-2033-deepsec-security-gate.md:7 and ADR-2033-deepsec-security-gate.md:78-80)
```

## AB-15.14 ADR-2080 — the metaharness router console gate
```mermaid
flowchart TD
    T["[model_routing.neural]<br/>agentbox.toml:1845-1855<br/>enabled, provider, quality_bar,<br/>cost_ceiling_usd_per_mtok, privacy_tier=public,<br/>trajectory, assets_dir"]
    SEED["[[interaction_plane.session_seeds]]<br/>slug=router, tool=custom:router<br/>agentbox.toml:2294-2296"]
    CAT["CATALOGUE id: model-routing-neural<br/>gate model_routing.neural.enabled<br/>apply_class rebuild<br/>system-manifest.js:118"]
    NIX["flake.nix modelRoutingNeuralCfg<br/>+ modelRouterAssets fetchurl closure<br/>flake.nix:286-288"]
    BAKE["flake.nix bake block -- gate off ⇒<br/>nothing copied, byte-identical-when-off<br/>flake.nix:2443-2449"]
    ENTRY["entrypoint-unified.sh boot reconcile<br/>AGENTBOX_MODEL_ROUTER_* exported,<br/>scoped to the router session process only<br/>entrypoint-unified.sh:2299-2315"]
    WRAP["config/harness-wrappers/router.sh<br/>exec'd by the agentbox-sessions router seed --<br/>asserts privacy_tier=public, hard-fails loudly"]
    CONSOLE["config/model-router/console.mjs<br/>embeds offline, asks metaharness router<br/>inside the baked ruflo closure, executes<br/>through OpenRouter"]
    CLI["agentbox.sh model-router<br/>fetch, check, status, route or console<br/>agentbox.sh:2396"]
    CATCHECK["scripts/ci/check-manifest-catalogue.js<br/>BASELINE: model_routing.neural.trajectory<br/>sub-option of the catalogued gate"]

    T -->|"apply_class rebuild"| NIX
    NIX --> BAKE
    BAKE -->|"/opt/agentbox/model-router"| WRAP
    T -->|"read at boot"| ENTRY
    ENTRY -->|"env scoped to this process only, never global CLAUDE_FLOW_ROUTER_*"| WRAP
    SEED -->|"launch contract picks the wrapper template, contract.rs:72-75, launch.json:14-16"| WRAP
    WRAP --> CONSOLE
    CLI -->|"fetch: pre-rebuild fallback dir, hash-verified"| CONSOLE
    T --> CAT
    CAT --> CATCHECK

    NOTE1["INVARIANT ADR-2080 D3 -- nothing in the ruflo tree is patched;<br/>the router is a dedicated session-plane seed whose program is an agentbox console"]
    NOTE2["INVARIANT ADR-2079 §4 -- privacy_tier is pinned to 'public'; the console<br/>refuses any other value, so this path never carries non-public work"]
    WRAP -.-> NOTE1
    WRAP -.-> NOTE2
```



**Open:** ADR-2153 is now `accepted`, `implementation_status: partial`, `activation_status: live` and supersedes ADR-2079, so the router console is an ordinary session-plane seed (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`, `../project/agentbox/agentbox.toml:2294-2296`); its acceptance round trip excludes `router`, so the wrapper's hard-fail path is exercised by launch only, not by a routed task (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:211`).

**Drift (spend gating vs the wallet that now exists):** the capability lattice above gates LLM spend and HTTP 402 settlement, and a second spend surface has since been published — `sidestr-wallet` consults a `SpendPolicy` before every signature and ships only a permissive implementation, leaving the authority gate (ADR-2100) to the caller, which nothing in this topic yet supplies (see AB-33.7). The level-2 review that reshaped the federation design was itself run through the consultant tier drawn at AB-15.12 (see AB-35).

**Drift (manifest comment and ADR vs pin):** the manifest comment still names Deepsec 2.3.10 (`../project/agentbox/agentbox.toml:2353`), while the flake now pins 2.3.13 (`../project/agentbox/flake.nix:843-847`). ADR-2033's Decision still says the exact pin is 2.3.12 (`../project/agentbox/docs/adr/ADR-2033-deepsec-security-gate.md:175-176`), so neither the comment nor the record names the current pin; the host-to-VisionCraft hand-off it records is unchanged (`../project/agentbox/docs/adr/ADR-2033-deepsec-security-gate.md:183-188`) and the older execution receipt remains historical.

## AB-15.15 consult_advice — an advisory, flag-gated route suggestion on every consultant server
```mermaid
sequenceDiagram
    autonumber
    participant Caller as Coordinator
    participant Base as BaseConsultant<br/>consultant-base.js:212
    participant Adv as consult-advice.js<br/>assess consult-advice.js:140
    participant Jev as Jev endpoint<br/>keyless local backend, consult_advice is now listed in local_only, see the note below

    Caller->>Base: tools/list
    alt advisoryEnabled() false (AGENTBOX_JEV_CONSULT_ADVICE unset or 0, consult-advice.js:42,183)
        Base-->>Caller: consult, health, cost_estimate only (consultant-base.js:233)
    else flag on
        Base-->>Caller: the same three plus consult_advice (consultant-base.js:233)
    end
    Caller->>Base: consult_advice {question, remaining_budget_usd?, specialists?, deep_reasoning?}
    Base->>Adv: assess (consultant-base.js:393)
    Adv->>Adv: flag-off, malformed and no-adapter all return ok false without a request (consult-advice.js:147-149)
    Adv->>Jev: one typed question, 4000 ms deadline, no retry (consult-advice.js:49,156)
    Note over Adv,Jev: localOnly follows AGENTBOX_JEV_CONSULT_LOCAL_ONLY (consult-advice.js:151)<br/>the boot projection exports it as 1 only when consult_advice is listed in [system_one].local_only (system_one.rs:538-593),<br/>and the shipped manifest now lists it (agentbox.toml:924-943), so once the new image is live the route pins the keyless local backend
    alt Jev unavailable, malformed or below min confidence
        Adv-->>Base: ok false, reason (consult-advice.js:166)
        Base-->>Caller: advice null, fallback_reason
    else answered
        Adv-->>Base: route none or specialist or deep_reasoning, confidence, advisory_only true (consult-advice.js:177)
        Base-->>Caller: advice plus question_set_version
    end
    Note over Base: INVARIANT: the tool never performs, blocks or re-routes a consult<br/>and never changes a spend cap, consultant-base.js:393
```

**What it shows.** An opt-in fourth tool on every consultant MCP server. It is not listed unless its flag is set; when set it asks a Jev endpoint whether the coordinator should consult at all, and returns a typed suggestion that the coordinator is free to ignore.
**Why it is this way.** Whether to call a consultant was the one decision no gate or hook covered, and spend caps stay deterministic code, so the model only advises (`../project/agentbox/mcp/consultants/shared/consult-advice.js:20`).

**Open:** the flag is now projected by `[system_one.consumers].consult_advice` (`../project/agentbox/agentbox.toml:962`, `../project/agentbox/services/agentbox-manifest/src/system_one.rs:453-458`) and SECURITY-profiles records the route as STAGED for rebuild, local-only, with no measured uplift claimed (`../project/agentbox/docs/SECURITY-profiles.md:96`). The manifest now makes the local-only half true in its own terms: `consult_advice` is in `[system_one].local_only` and `backend` is the local `laya` route (`../project/agentbox/agentbox.toml:913-943`), so the projection pins it to the keyless local backend (`../project/agentbox/services/agentbox-manifest/src/system_one.rs:578-593`). **Drift (code comment vs manifest):** the projector's comment still says `local_only` is empty by default (`../project/agentbox/services/agentbox-manifest/src/system_one.rs:534-537`). Until the candidate image boots the flag stays unset; whether a measured uplift will be recorded is not stated.
