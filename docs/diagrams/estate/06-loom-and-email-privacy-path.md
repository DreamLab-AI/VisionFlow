---
id: ES-06
title: Ontology Loom and the email privacy path
area: estate
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [agentbox:ADR-2139, agentbox:ADR-2140, loom:ADR-140, agentbox:ADR-2153, agentbox:ADR-2023, agentbox:ADR-2079, agentbox:ADR-2080, agentbox:ADR-2084, loom:ADR-139, loom:ADR-141, loom:ADR-142, agentbox:ADR-051]
sources:
  - ../project/agentbox/skills/email-search/SKILL.md
  - ../project/agentbox/docs/adr/ADR-2023-loom-facade.md
  - ../project/loom/README.md
  - ../loom/crates/loom-facade/src/reload.rs
  - ../loom/crates/loom-facade/src/routes/mod.rs
  - ../loom/crates/loom-facade/src/jev.rs
  - ../loom/docs/jev-option.md
  - ../loom/docs/design/ADR-140-loom-v2-agent-drivable-grounding.md
  - ../project/agentbox/docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../loom/crates/loom-facade/src/bin/reload_published_generation.rs
  - ../loom/docs/design/ADR-142-consume-versioned-shared-ontology-contracts.md
  - ../loom/deploy/loom-generation-reload.service
  - ../project/docker-compose.unified.yml
  - ../project/docs/BASELINE-architecture.md
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/config/model-router/artefacts.json
  - ../project/agentbox/docs/adr/ADR-2080-metaharness-router-console-under-aoe.md
  - ../project/agentbox/docs/adr/ADR-2084-one-published-loom-client-for-every-facade-caller.md
  - ../project/agentbox/lib/explainer-tools.nix
verified_commit: {visionclaw: 8692b12ba987ecb87165566beb7c320a2446e4b4, agentbox: 6500582dfae23bf8a38a6942916d208154e43625, loom: 5ad860863b588c0cdb2290e31c69e39d82755aeb}
---
## ES-06.1 Topology — the gateway holds a façade, never a model port
```mermaid
flowchart LR
    subgraph box["agentbox container"]
        AG["agent / skill<br/>mcp email-gateway"]
    end
    subgraph vcnet["visionclaw_network (docker bridge)"]
        GW["email-mcp-gateway:8765<br/>streamable-HTTP MCP, bearer auth"]
        LOOMB["loom sidecar — Deployment B, NOT RUNNING BY DEFAULT<br/>hostname loom, LOOM_FACADE_PORT 8080<br/>docker-compose.unified.yml:382-385,<br/>gated behind the loom compose profile :453-454"]
        XI["xinference:9997/v1<br/>bge-small-en-v1.5 / 384"]
    end
    subgraph ml["machinelearn .132"]
        DNAT["hp-nat.service DNAT port 8084<br/>plus MSS clamp for the 9000 to 1500 step-down"]
    end
    subgraph hp["HP-Desktop — downstream, NO LAN IP"]
        LOOMA["Loom façade port 8084 — Deployment A<br/>COLOCATED with the model"]
        MODEL["loom-model container port 8085<br/>Qwen3.8-27B, cutover 2026-08-14"]
    end

    AG -- "MCP tools ask_email / fetch_* / refresh_inbox" --> GW
    GW -- "REASONER_BASE_URL" --> LOOMB
    GW -- "REASONER_BASE_URL alternative, the gateway host DNAT<br/>to the facade on port 8084<br/>agentbox/skills/email-search/SKILL.md:79" --> DNAT
    DNAT -- "point-to-point 25 G rail to the connected node<br/>agentbox/skills/email-search/SKILL.md:81" --> LOOMA
    LOOMA -- "DISTILL_BACKEND_URL" --> MODEL
    LOOMB -- "DISTILL_BACKEND_URL<br/>blank = retrieval-only" --> MODEL
    LOOMB -- "semantic fallback query embed" --> XI

    INV1["INVARIANT — consumers hold the FAÇADE, never the model port.<br/>The deployed model is a URL behind DISTILL_BACKEND_URL;<br/>swapping Muse to Gemma to Qwen3.8 to next never touches a<br/>consumer. This is the no-technical-debt-on-upgrade guarantee."]
    INV2["INVARIANT — the Loom IS the email privacy system. It<br/>delegates ONLY to a LAN/local model, never to a cloud<br/>endpoint, so mail content never leaves the LAN."]
    DIV0["DIVERGENCE — Deployment B is SPECIFIED, not running. The loom<br/>service is gated behind compose profile loom (docker-compose.unified.yml:453-454),<br/>so a default up never starts it, and its image is not built from this repo:<br/>the referenced loom/deploy/Dockerfile does not exist in this checkout<br/>(docker-compose.unified.yml:376-381, the checkout holds README.md and app/ only). Deployment A (port 8084 on HP)<br/>is the live path. see ES-01.6"]
    DIV1["DIVERGENCE — the connected node's old LAN address is DEAD. It<br/>sits downstream with no LAN IP and the gateway host routes and<br/>NATs it over the rail. Commit 2899b3b7e generalised the literal<br/>address out of the public repo, so the record now names it only<br/>as a retired address, agentbox/docs/adr/ADR-2023-loom-facade.md:55<br/>and agentbox/skills/email-search/SKILL.md:99. see ES-06.7"]
    DIV2["DIVERGENCE GOVERNANCE-capabilities — ADR-051 (Loom) is<br/>decision_status PROPOSED while the Loom is<br/>production-critical. Interim authority is the governing doc."]
    DIV3["RESOLVED ADR-2070 — ingress and egress have different contracts.<br/>Loom facade port 8084 and raw model port 8085 are named egress doors.<br/>Task-specific raw calls do not bypass the NIP-98 ingress proxy."]

    INV1 --- INV2
    LOOMA --> DIV1
    LOOMA --> DIV2
    MODEL --> DIV3
```

## ES-06.2 ask_email — tier 1, egress filter applied
```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant G as email-mcp-gateway:8765
    participant R as retrieval (bge-m3)
    participant L as Loom façade
    participant M as LAN model
    participant F as gpt-oss-safeguard<br/>egress filter

    A->>G: ask_email{query, date_from?, date_to?, sender?, folder?, top_k?}
    Note over A,G: Transport auth — Authorization Bearer<br/>AGENTBOX_EMAIL_GATEWAY_TOKEN gates ANY call
    alt bearer missing or wrong
        G-->>A: 401 — re-provision the token
    else bearer ok
        G->>R: retrieve candidate messages
        R-->>G: top_k matches
        G->>L: POST /v1/chat/completions (REASONER_BASE_URL)
        L->>L: scaffold-inject the last user message (see ES-06.5)
        L->>M: delegate to the LAN model
        M-->>L: completion
        L-->>G: grounded answer
        G->>F: sanitize
        F-->>G: gist plus abstracted evidence — roles, date buckets, opaque ref_id
        G-->>A: sanitized answer, NEVER raw mail
    end
    Note over G,F: INVARIANT — ask_email ALWAYS sanitizes regardless of<br/>any pubkey passed to it. Raw data comes only from the<br/>raw tools. Default posture is to prefer this tool.
```

## ES-06.3 Break-glass tier — raw tools behind a second, capability gate
```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant G as email-mcp-gateway:8765
    participant AL as PRIVILEGED_NOSTR_PUBKEYS allow-list
    participant S as message store

    A->>G: fetch_email_raw{query, nostr_pubkey} or fetch_email_by_ref{ref_id, nostr_pubkey}
    rect rgb(240,230,230)
    Note over A,G: TIER 1 — transport bearer gates any call
    alt bearer invalid
        G-->>A: 401
    else bearer valid
        Note over G,AL: TIER 2 — capability. The nostr_pubkey ARGUMENT is<br/>checked against the server allow-list. Every attempt is<br/>logged with an 8-char fingerprint.
        G->>AL: lookup pubkey
        alt pubkey on the allow-list
            AL-->>G: authorised
            G->>S: fetch raw
            S-->>G: real headers, sender, subject, date, full text
            G-->>A: RAW — egress filter BYPASSED
        else absent, empty, or npub instead of hex
            AL-->>G: not authorised
            G-->>A: {"authorized": false} and NO data
        end
    end
    end
    Note over A,AL: INVARIANT — a Nostr PUBLIC key is the publishable half,<br/>an identity/capability token, not a secret. Passing it as a<br/>tool argument is the design, not a leak. What never leaves<br/>the box is the bearer token and the Nostr PRIVATE key.
    Note over G: Read the operator pubkey from AGENTBOX_X_ONLY_PUBKEY_HEX<br/>at call time. Never hardcode the hex into skill source.
```

## ES-06.4 refresh_inbox — on-demand ingest across two Proton accounts
```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant G as email-mcp-gateway:8765
    participant PB as Proton Bridge (in-container)
    participant P1 as primary IMAP_USER
    participant P2 as secondary IMAP_USER_2
    participant IX as index

    A->>G: refresh_inbox
    Note over G: Pull new mail NOW rather than waiting for the ~4h crawl.<br/>Used for password resets, one-time codes, verification links.
    G->>PB: open bridge
    loop within a time budget, primary first
        G->>P1: crawl new messages
        P1-->>G: messages
        G->>P2: crawl new messages
        P2-->>G: messages
    end
    G->>IX: ingest and embed
    IX-->>G: indexed
    G-->>A: newest messages plus accounts_crawled[]
    Note over G,P2: A third slot IMAP_USER_3 exists but is EMPTY.<br/>The scheduled crawl logs "IMAP ingest complete across<br/>2/2 account(s)".
    Note over A,G: DIVERGENCE — a 180 s refresh_inbox timeout is the<br/>signature of a stale reasoning route, NOT a dead<br/>container. see ES-06.7
```

## ES-06.5 Loom scaffold-inject — grounding, budget clamp and the confidence gate
```mermaid
sequenceDiagram
    autonumber
    participant C as consumer
    participant F as loom-facade
    participant IX as scaffold + prose index<br/>ontology-corpus.rvdb
    participant XI as xinference /v1/embeddings
    participant B as DISTILL_BACKEND_URL

    C->>F: POST /v1/chat/completions
    F->>IX: lexical match on the last user message
    IX-->>F: scored matches
    opt LOOM_SEMANTIC_FALLBACK=1
        F->>XI: embed query (bge-small-en-v1.5 / 384)
        XI-->>F: vector
        F->>IX: semantic fallback
        Note over F,IX: DIVERGENCE — gated OFF by default (=0) until the<br/>recall bench clears (loom RUST-ARCHITECTURE 8.4)
    end
    alt LOOM_CONFIDENCE_INJECTION=1
        F->>F: apply the gate
        Note over F: LOOM_STRONG_MATCH_SCORE 8.0 — at/above gives full budget<br/>and is the confidence denominator<br/>LOOM_MIN_INJECT_SCORE 2.0 — below skips injection entirely<br/>LOOM_MIN_INJECT_FRACTION 0.4 — weakest kept match's budget share
    else LOOM_CONFIDENCE_INJECTION=0 (default Profile B)
        F->>F: master switch OFF, thresholds pinned at loom-scaffold tuning.rs defaults
    end
    opt LOOM_JEV=1 (code default OFF, ON in the loom compose service, no measured uplift)
        F->>F: jev_stages: gate may withhold an engaged scaffold, then rank may reorder candidates<br/>../loom/crates/loom-facade/src/routes/mod.rs:383-384, :495-512
        Note over F: Local-only System One endpoint unless LOOM_JEV_ALLOW_CLOUD=1, loom jev.rs:232-234. Compose now sets ALLOW_CLOUD=1 and reaches the agentbox proxy agentbox:8094 through the shared anchor, selected backend TypeSafe, with gate, rank and support on but no gate min-confidence, so the gate stays shadow<br/>../project/docker-compose.unified.yml:387-397, ../loom/crates/loom-facade/src/jev.rs:353. With the flag off or any<br/>fallback the request and response are byte-identical to the pre-change ones<br/>../loom/docs/jev-option.md:3-5
    end
    F->>F: clamp scaffold to ONTOLOGY_BUDGET (1500 tokens)
    alt DISTILL_BACKEND_URL set
        F->>B: delegate with the injected scaffold
        Note over F,B: LOOM_MIN_MAX_TOKENS = 1536 — reasoning backends<br/>truncate to EMPTY below this (the 400-to-empty trap)
        B-->>F: completion
        F-->>C: grounded completion, plus a jev support annotation when LOOM_JEV=1<br/>../loom/crates/loom-facade/src/routes/mod.rs:470-476
    else DISTILL_BACKEND_URL blank
        F-->>C: 503 — retrieval-only deployment
    end
    Note over C,B: BENCHMARK — static scaffold lifts grounded recall ~3.5x and<br/>is ~3-6x faster than cold parametric reasoning. Prose adds<br/>~nothing and was dropped as a default. Agentic tool-traversal<br/>is model-dependent (Gemma strong, Muse weak).
    Note over F: The four confidence vars are stated EXPLICITLY in compose so<br/>/health.injection_policy can be diffed against the file. The<br/>2026-09-02 dream cycle found a doc/deploy mismatch caused<br/>precisely by leaving them unstated (loom ADR-138).
```

## ES-06.6 Loom endpoint map — what needs a model and what does not
```mermaid
flowchart TB
    subgraph nomodel["Retrieval only — NO model required"]
        E1["GET /health<br/>liveness, corpus generation stamp,<br/>backend/graph/index readiness, injection_policy"]
        E2["GET /loom/generation<br/>the corpus generation identity being served"]
        E3["POST /loom/scaffold<br/>budget-clamped ontology grounding"]
        E4["POST /loom/sparql<br/>read-only clamped SPARQL over the reasoned closure"]
        E5["POST /loom/search<br/>label/substring search over the store"]
        E8["POST /mcp, the agentic plane (loom ADR-140 D1)<br/>loom.manifest, browse, resolve, sparql, neighbours, paths<br/>../loom/crates/loom-facade/src/routes/mod.rs:65"]
    end
    subgraph needsmodel["Delegates to DISTILL_BACKEND_URL"]
        E6["POST /v1/chat/completions<br/>scaffold-inject then delegate"]
        E7["GET /v1/models<br/>model identity passthrough"]
    end
    GEN["A full generation = scaffold-index.json + prose-index.json<br/>+ the TTLs + ontology-corpus.rvdb with its .generation.json"]
    T1["TRAP the empty floor — with an empty or mis-pointed<br/>LOOM_DATA_SOURCE the façade STILL starts and /health STILL<br/>returns 200, but the log reads lexical index NOT loaded ...<br/>empty floor. That is a staging bug, not a dead container.<br/>Check the mount before you check the process."]
    T2["The .rvdb is mounted read-only, but OPENING it mutates the<br/>redb file even for reads (the HNSW index is repacked on open),<br/>so the entrypoint copies it to a writable tmpfs at /run/loom.<br/>The tmpfs uid/gid must stay 65532 to match the image's<br/>non-root user or that copy fails EACCES."]
    D1["Reload is the Rust binary reload_published_generation (reload_published_generation.rs:185), which replaced the<br/>deleted scripts/reload-published-generation.py. reconcile (reload.rs:380) calls verified_bundle (reload.rs:211):<br/>the six REQUIRED artefacts (reload.rs:36), strict marker, per-file sha256, sidecar. It writes reload-pending<br/>before the restart (reload.rs:399) and served only after the live identity matches (reload.rs:413).<br/>One tick at a time: try_lock (reload_published_generation.rs:161). The unit runs the binary<br/>(loom-generation-reload.service:6). ADR-142 records the first served Ontomark bundle (ADR-142-consume-versioned-shared-ontology-contracts.md:56) and that the<br/>stale 23 Sep generation was the host unit still calling the deleted script (ADR-142-consume-versioned-shared-ontology-contracts.md:58)."]
    D2["DIVERGENCE — app/ontology-mcp/ (stdio MCP server, JavaScript)<br/>survived the Python deletion but now has NO build or run path<br/>from this repo. PRD-025/ADR-135 treat it as a thin client of<br/>the Loom index; placement is an open decision."]

    RECON["GENERATION SOURCE — agentbox starts hp-loom-reconcile.sh in the background at every boot<br/>([vault].loom_sync, full here): it builds the committed vault HEAD, promotes and verifies it, pushes<br/>the BUILT bundle and starts the Loom's verifying reload. The connected node never reads the vault<br/>agentbox/config/entrypoint-unified.sh:1211-1219, agentbox/agentbox.toml:1176-1184"]

    nomodel --> GEN
    RECON --> GEN
    GEN --> T1
    GEN --> T2
    GEN --> D1
    needsmodel --> D2
```

**Tension:** loom ADR-140 is `Status: Proposed` (`../loom/docs/design/ADR-140-loom-v2-agent-drivable-grounding.md:3`), yet its P0 is in service: the façade routes `/mcp` (`../loom/crates/loom-facade/src/routes/mod.rs:65`) and the note records that agentbox's `ontology-augment` grounds through it, with P1 onward not started (`../loom/docs/design/ADR-140-loom-v2-agent-drivable-grounding.md:632-634`).

**Open:** the Jev option inside the façade is a proposed research candidate with no measured uplift; the code default is off (`LOOM_JEV=1` to enable, `../loom/docs/jev-option.md:3-6`) but the unified compose arms it (`../project/docker-compose.unified.yml:391-397`) and, since 2026-10-30, lets it reach the cloud-backed agentbox proxy instead of the laya façade (`LOOM_JEV_ALLOW_CLOUD` default 1, `:395`), although the façade's own header still justifies local-only by private corpus content (`../loom/crates/loom-facade/src/jev.rs:30-34`); CONFIGURED, taking effect when the new agentbox image is activated and the loom service is recreated (`../project/docs/BASELINE-architecture.md:42-53`).

## ES-06.7 The black-hole hang — health green, every reasoning call stalls
```mermaid
sequenceDiagram
    autonumber
    participant A as agent
    participant G as email-mcp-gateway:8765
    participant H as GET /health
    participant DEAD as the retired model-host address<br/>agentbox/docs/adr/ADR-2023-loom-facade.md:55
    participant GOOD as the gateway-host DNAT to the facade, port 8084<br/>agentbox/skills/email-search/SKILL.md:79

    rect rgb(240,230,230)
    Note over A,DEAD: BROKEN STATE — the stale-route trap, agentbox/skills/email-search/SKILL.md:210-216
    A->>H: GET /health
    H-->>A: 200, container healthy on visionclaw_network,<br/>safeguard plus embedder ready
    A->>G: ask_email or refresh_inbox
    G->>DEAD: POST /v1/chat/completions (stale REASONER_BASE_URL)
    Note over G,DEAD: BLACK HOLE — packets are routed nowhere. Every<br/>synthesis stalls to timeout while /health still answers.
    DEAD--xG: no response
    G-->>A: 180 s timeout / whole-session tool loss
    end
    Note over A,DEAD: FINGERPRINT is exactly that split — health GREEN,<br/>all reasoning calls stall. The fix is the ROUTE, not<br/>the container. Do not restart the gateway.
    rect rgb(230,240,230)
    Note over A,GOOD: FIXED STATE
    A->>G: ask_email
    G->>GOOD: REASONER_BASE_URL is the Loom base URL, not a raw model port<br/>agentbox/skills/email-search/SKILL.md:79
    GOOD-->>G: completion from the current loom-model
    G-->>A: answer
    end
    Note over GOOD: VERIFY — ask the facade for its model list on port 8084 and<br/>check that health reports backend_reachable true, then match the<br/>CURRENT backend rather than a fixed string and recreate the<br/>container. agentbox/skills/email-search/SKILL.md:109
```

## ES-06.8 MCP client link loss and the mid-session JSON-RPC recovery
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code MCP client
    participant G as email-mcp-gateway:8765
    participant M as cold LAN model

    rect rgb(240,230,230)
    Note over CC,M: FAILURE — tools absent for the WHOLE session
    CC->>G: initialize handshake at session start
    G->>M: warm the backend
    M-->>G: slow (30s+ per call, SSE held open)
    Note over CC,G: The client disables an HTTP/SSE server for the ENTIRE<br/>session when the startup initialize exceeds MCP_TIMEOUT,<br/>and it does NOT retry HTTP servers after boot.
    G--xCC: handshake exceeds MCP_TIMEOUT
    CC-->>CC: server disabled for the session
    end
    Note over CC,G: DURABLE FIX (in the build, applied on next nix buildout) —<br/>MCP_TIMEOUT=60000 and MCP_TOOL_TIMEOUT=180000 set in<br/>flake.nix, tunable via [skills.email_search]<br/>.mcp_startup_timeout_ms / .mcp_tool_timeout_ms, plus a<br/>detached boot warm-up in config/entrypoint-unified.sh.
    rect rgb(230,240,230)
    Note over CC,G: MID-SESSION RECOVERY — verified 2026-07-06
    CC->>G: POST $AGENTBOX_EMAIL_GATEWAY_URL/mcp — initialize
    Note over CC,G: Authorization Bearer $AGENTBOX_EMAIL_GATEWAY_TOKEN<br/>Accept application/json, text/event-stream
    G-->>CC: response carrying the Mcp-Session-Id header
    CC->>G: notifications/initialized (with Mcp-Session-Id)
    CC->>G: tools/call
    G-->>CC: result
    end
    Note over CC,M: INVARIANT — the container itself never went down. Only<br/>the harness client link did, so driving the streamable-HTTP<br/>server directly over JSON-RPC restores the capability.
```

## ES-06.9 Backend-swap runbook — one port, stop-park-promote-start
```mermaid
stateDiagram-v2
    [*] --> Verify
    Verify --> StopCurrent
    StopCurrent --> ParkOld
    ParkOld --> PromoteNew
    PromoteNew --> StartNew
    StartNew --> Confirm
    Confirm --> [*]
    Confirm --> Rollback
    Rollback --> [*]

    note right of Verify
        Confirm the target's GGUF FIRST via
        docker inspect (MAIN_GGUF / ALIAS)
        so the right container is promoted.
        Reached over ssh john@10.10.10.1.
    end note
    note right of StopCurrent
        docker stop loom-model.
        Only ONE container can hold port 8085.
        Alternates park as loom-model-NAMEbak (Exited).
    end note
    note right of PromoteNew
        docker rename loom-model loom-model-OLDbak
        then rename loom-model-NEWbak to loom-model.
    end note
    note right of Confirm
        GET /v1/models on port 8084 shows the new alias and
        /health shows backend_reachable true, then
        smoke-test a real /v1/chat/completions.
        Reasoning models need max_tokens >= 2048
        or they return empty.
    end note
    note right of Rollback
        Reversible — the parked OLDbak restores
        the same way. Both carry
        restart-policy unless-stopped, so a
        manually-stopped alternate stays down
        and will not fight for the port.
    end note
    note right of StartNew
        GOTCHA HP's login shell is fish, so
        set -e in an SSH heredoc errors
        harmlessly. The docker lines still run.
    end note
```

## ES-06.10 Two model-selection planes, and the privacy line between them
```mermaid
flowchart TB
    subgraph PRIV["Plane 1 — the Loom façade: PRIVATE work, LAN-only"]
        FAC["Loom façade, the stable door<br/>ADR-2023-loom-facade.md:55<br/>consumers hold this URL, never a model port"]
        MDL["deployed model behind DISTILL_BACKEND_URL<br/>Qwen3.8-27B in the loom-model container<br/>email-search/SKILL.md:95"]
        MAIL["email-mcp-gateway port 8765<br/>REASONER_BASE_URL points at the façade"]
        FAC --> MDL
        MAIL --> FAC
    end

    subgraph PUB["Plane 2 — ADR-2080 metaharness router console: PUBLIC work only"]
        GATE["[model_routing.neural] gate<br/>agentbox.toml:1736-1737"]
        ART["pinned artefacts baked at rebuild<br/>config/model-router/artefacts.json<br/>flake.nix:179-185, installed flake.nix:2245-2247"]
        ENV["entrypoint exports AGENTBOX_MODEL_ROUTER_*<br/>entrypoint-unified.sh:2307-2327<br/>assets_dir resolved at :2310, fallback :2311-2314"]
        CON["router session console, a seed on agentbox-sessions (agentbox.toml:2186)<br/>embeds the task offline, asks the router for the<br/>cheapest OpenRouter model above the quality bar,<br/>executes it, writes a labelled receipt"]
        GATE --> ART --> ENV --> CON
    end

    INV1["INVARIANT ADR-2079 section 4 — privacy_tier is pinned to public<br/>(agentbox.toml:1744) and the console EXITS on any other value.<br/>Private corpora never reach an OpenRouter model; that is what<br/>keeps plane 2 disjoint from plane 1."]
    INV2["INVARIANT — the router env is SESSION-SCOPED. The entrypoint<br/>exports AGENTBOX_MODEL_ROUTER_* for the router seed only and<br/>never exports CLAUDE_FLOW_ROUTER_* globally (agentbox.toml:1734),<br/>so no other agent silently inherits a cost-optimal route."]
    INV3["INVARIANT ADR-2023 — plane 1 swaps its model behind the façade with<br/>zero consumer change. Plane 2 swaps its model PER TASK by price.<br/>Neither plane may name the other's endpoint: a raw model port in<br/>consumer config re-creates the coupling ADR-2023 removed<br/>(ADR-2023-loom-facade.md:55-56)."]
    DEG["DIVERGENCE — the gate can be on with no artefacts present. The<br/>entrypoint then prints the fetch instruction rather than failing<br/>(entrypoint-unified.sh:2331), so an operator sees a console<br/>that is enabled-but-inert until ./agentbox.sh model-router fetch<br/>or a rebuild lands the pinned set. see AB-15 and AB-28."]

    NOPIN["INVARIANT 2026-10-01, 878f23311 — no agentbox default names the<br/>Loom's model any more. dream_machine loom_model is empty<br/>(agentbox.toml:2575), the dream-engine supervisor passes an empty<br/>LOOM_MODEL (flake.nix:2900) and the generated compose defaults<br/>LOOM_MODEL empty (flake.nix:3962). Empty means use the single<br/>model the facade advertises at /models, so a swap behind the<br/>facade no longer breaks a seeder that asked for a stale alias."]
    MDL --> NOPIN
    PRIV --> INV3
    PUB --> INV3
    GATE --> INV1
    ENV --> INV2
    ENV --> DEG
```

**Resolved:** ADR-2153 is live, so the router console of plane 2 is an ordinary `[[interaction_plane.session_seeds]]` entry (`slug = "router"`, `../project/agentbox/agentbox.toml:2186`) on the owned session plane, not an Agent of Empires session; ADR-2080 (`../project/agentbox/docs/adr/ADR-2080-metaharness-router-console-under-aoe.md`) keeps its scope rules and its AoE-era title (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`, `:82`).

**Tension (ADR-2080 acceptance item 3, recorded 2026-10-30):** the public-only invariant above holds for the console, which exits on any other tier, but the router wrapper pins the tier to `public` before it starts the console, so setting `AGENTBOX_MODEL_ROUTER_PRIVACY_TIER=private` cannot make the wrapper exit non-zero as the acceptance text asks (`../project/agentbox/docs/adr/ADR-2080-metaharness-router-console-under-aoe.md:129-132`). The record's `review_trigger` says items 1, 2 and 4 passed on the running image on 2026-10-30 and leaves item 3 to an owner ruling, so ADR-2080 stays `accepted` / `partial` / `staged` (`../project/agentbox/docs/adr/ADR-2080-metaharness-router-console-under-aoe.md:5-7,13`).

## ES-06.11 One published loom-client — every facade caller, one place that knows the traps
```mermaid
flowchart TB
    BEFORE["BEFORE — five callers of the facade grew up independently and<br/>each hand-rolled chat completions, each having learned a<br/>different subset of the same lessons<br/>agentbox/docs/adr/ADR-2084-one-published-loom-client-for-every-facade-caller.md:21-28"]

    subgraph LESSONS["The three lessons, each known by exactly one caller"]
        L1["a facade can answer HTTP 200 with ontology prose and never<br/>call the model at all, :25-26"]
        L2["a truncated reasoning model returns EMPTY content, not a<br/>short answer, so max_tokens needs a floor, :27-28"]
        L3["grounding applied to a subject the ontology does not cover<br/>is a wrong answer, which is what the passthrough assertion<br/>catches, :26-27"]
    end
    BEFORE --> LESSONS

    DEC["DECISION — every facade caller uses the published loom-client<br/>crate; hand-rolled chat-completions construction against a<br/>facade is PROHIBITED<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:37-39"]
    LESSONS --> DEC

    OWNS["The crate owns four things no caller reimplements: the<br/>max_tokens floor, truncation retry on a doubled budget, the<br/>refusal of scaffold-only responses, and the passthrough<br/>assertion. ADR-2084-one-published-loom-client-for-every-facade-caller.md:41-43"]
    DEC --> OWNS

    OPT["TWO INDEPENDENT PER-REQUEST SWITCHES — verbatim and scaffold.<br/>LoomOptions exposes them as separate constructors,<br/>declining_verbatim for a subject the ontology covers and<br/>passthrough for one it does not, so the choice is explicit<br/>rather than copied from a neighbour.<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:43-45"]
    OWNS --> OPT

    SCAF["scaffold false makes the facade a PLAIN PROXY for that one<br/>request. It is the supported way to send a non-ontology subject,<br/>a codebase or a document, through the same door. Conflating the<br/>two switches is what left a page-summary request open to being<br/>answered from the ontology without the page being read at all.<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:31-33"]
    OPT --> SCAF

    NIX["The Nix packaging says the same thing: the facade protocol is<br/>NOT reimplemented in agentbox, it comes from the published crate<br/>that dream-engine and podcast-ingest also use.<br/>agentbox/lib/explainer-tools.nix:11-16"]
    DEC --> NIX

    BARE["INVARIANT — a facade that reports no loom telemetry is treated<br/>as passthrough by nature, so the crate works unchanged against a<br/>bare OpenAI-compatible server.<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:47-48"]
    OWNS --> BARE

    LIC["The loom repository was relicensed from AGPL-3.0-only to<br/>MIT OR Apache-2.0, which is what made publication possible.<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:52-54. see ES-12"]
    DEC --> LIC
```
