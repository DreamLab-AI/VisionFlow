---
id: AB-05
title: Manifest gate catalogue, vault path authority and the agentbox.sh CLI
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2003, ADR-2028, ADR-2029, ADR-2034, ADR-2036, ADR-2037, ADR-2038, ADR-2039, ADR-2080, ADR-2091, ADR-2092, ADR-2093, ADR-2094]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/management-api/routes/system.js
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/scripts/ruvector-sidecar-update.sh
  - ../project/agentbox/config/validate-artifacts.sh
  - ../project/agentbox/config/artifact-probes.json
  - ../project/agentbox/schema/agentbox.toml.schema.json
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/scripts/agentbox-config-validate.js
  - ../project/agentbox/scripts/recall-fixtures/recall-fixture.v1.json
  - ../project/agentbox/scripts/ruvector-recall-harness.mjs
  - ../project/agentbox/skills/mcp.json
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/scripts/post-deploy-cleanup.sh
  - ../project/agentbox/management-api/routes/llm-marketplace.js
verified_commit: 1639f86ab
---

## AB-05.1 GET /v1/system — catalogue plus live introspection
```mermaid
sequenceDiagram
    autonumber
    participant C as Operator or cockpit
    participant R as routes/system.js:33<br/>fastify.get /v1/system
    participant BV as buildSystemView<br/>system-manifest.js:327
    participant CAT as CATALOGUE const<br/>system-manifest.js:39-279
    participant ST as stateOf<br/>system-manifest.js:296
    participant RG as resolveGate<br/>system-manifest.js:282
    participant AD as resolved adapters

    C->>R: GET /v1/system
    R->>BV: buildSystemView(manifest, adapters) — routes/system.js:42
    BV->>BV: core = manifest entry :330 plus identity entry :334
    loop for slot of beads pods memory events orchestrator (system-manifest.js:338-339)
        BV->>AD: read adapter.impl and adapter.CONTRACT_VERSION
        AD-->>BV: impl or 'unresolved', contract_version or null (:343-344)
        BV->>BV: push core entry adapter-<slot> (:341-346)
    end
    BV->>BV: build resolved vault block (:349-372)
    loop for entry of CATALOGUE (system-manifest.js:376)
        BV->>ST: stateOf(manifest, entry)
        ST->>RG: resolveGate(manifest, entry.gate)
        RG-->>ST: gate value walked down the dotted path (:283-290)
        ST-->>BV: on | off | available
        BV->>BV: push to surfaces or modules by entry.layer (:388)
    end
    BV-->>R: apply_classes, core, vault, surfaces, modules, counts (:391-404)
    R-->>C: live system view
    Note over BV,CAT: INVARIANT — the catalogue is documentation-as-data but STATE is always introspected from the parsed agentbox.toml at request time, never hard-coded (system-manifest.js:6-13)
    Note over BV: counts block emits core, surfaces_on, surfaces, modules_on, modules (:397-403)
    Note over BV,AD: DOC-DRIFT — the adapter-slot summary at system-manifest.js:345 repeats "observability → privacy → JSON-LD". Only two layers are in the wrap chain — see AB-04.4
    Note over BV,AD: RESOLVED ADR-2036: system-manifest.js:345 now reads<br/>"every dispatch wrapped by observability → privacy redaction<br/>(ADR-2036). JSON-LD encoding is a per-surface gated stage<br/>invoked by the owning route, not a dispatch layer" — see AB-04.4
    Note over R: sibling route GET /v1/system/audit-chain verifies the hash-chained events JSONL (routes/system.js:47)
```

## AB-05.2 Catalogue shape and the real surface/module census
```mermaid
flowchart TD
    CAT["CATALOGUE array<br/>system-manifest.js:39-279<br/>72 entries total"] --> SURF["layer 'surface' — 13 entries"]
    CAT --> MOD["layer 'module' — 59 entries"]
    SURF --> S1["ungated: management-api, terminal, setup-wizard,<br/>uri-resolver, agent-events-stream, metrics<br/>system-manifest.js:41-71"]
    SURF --> S2["gated: code-server, jupyter, desktop, comfyui,<br/>linked-data-viewer, interaction-plane, tab0-bridge<br/>system-manifest.js:50-79"]
    MOD --> M1["toolchain + CLI: ruflo, agentic-qe, nagual-qe, deepsec,<br/>codebase-memory, metaharness, codex, opencode, rust-toolchain,<br/>model-routing, model-routing-neural ADR-2080 :109-111"]
    MOD --> M2["GPU/media: qgis-mcp, blender-mcp, imagemagick-mcp,<br/>ffmpeg, pytorch, cuda, gaussian-splatting"]
    MOD --> M3["sidecars flagged heavy: browser-sidecar, gui-tools-sidecar,<br/>voice-console, privacy-filter"]
    MOD --> M4["sovereign: sovereign-mesh, solid-pod, linked-data,<br/>payments, llm-marketplace, project-tracking, consultants"]
    MOD --> M5["memory: ruvector-external, memory-learning,<br/>memory-hygiene, ruvnet-brain, compression"]
    MOD --> M6["corpus: vault :249, vault-tui :252, ontology"]
    CAT --> FIELDS["per-entry fields: id, name, layer, gate or gates,<br/>service, apply_class, summary, heavy"]
    FIELDS --> CORE["core layer emitted separately by buildSystemView<br/>manifest, identity, five adapter-<slot> entries<br/>system-manifest.js:305-329"]
    CAT -.-> DRIFT["DOC-DRIFT — BASELINE-container.md:112 (ADR-2039 resolution) says<br/>60 entries = 13 surfaces + 47 modules. Verified census at 1639f86ab<br/>is 72 entries = 13 surfaces + 59 modules — the count has moved<br/>on twice since that resolution note was written"]
```

## AB-05.3 stateOf — how a gate value becomes a state word
```mermaid
flowchart TD
    E["catalogue entry"] --> A{"Array.isArray(entry.gates)?<br/>system-manifest.js:299"}
    A -->|yes| MG["resolve every gate path"]
    MG --> MG1{"any value === true?"}
    MG1 -->|yes| ON1["state 'on' — :301"]
    MG1 -->|no| MG2{"any value === false?"}
    MG2 -->|yes| OFF1["state 'off' — :302"]
    MG2 -->|no| AV1["state 'available' — :303"]
    A -->|no| B{"entry.gate falsy?<br/>system-manifest.js:305"}
    B -->|yes| ON2["state 'on' — ungated surface,<br/>present whenever the image is"]
    B -->|no| RG["resolveGate(manifest, entry.gate)<br/>system-manifest.js:282"]
    RG --> W["walk the dotted path key by key<br/>system-manifest.js:283-286"]
    W --> SEC{"cursor is an object?<br/>system-manifest.js:289"}
    SEC -->|yes| SECE["section gate resolves via its .enabled key — :289-290"]
    SEC -->|no| VAL["scalar value — :292"]
    SECE --> D
    VAL --> D{"value type"}
    D -->|"true"| ON3["state 'on' — :307"]
    D -->|"false"| OFF2["state 'off' — :308"]
    D -->|"string"| MODE{"value in off, none or entry.off_values?<br/>system-manifest.js:315-317"}
    D -->|"undefined"| AV2["state 'available' — catalogued<br/>but unconfigured — :319"]
    MODE -->|yes| OFF3["state 'off'"]
    MODE -->|no| ON4["state 'on'"]
    MODE -.-> VT["ADR-2029 — vault.tui is a mode string naming a thing<br/>not a state, so 'none' is the conventional disabled value.<br/>Vanilla default tui = 'none' reports vault-tui as off.<br/>ADR-2091 — skill-router declares its OWN off value, table,<br/>because the pre-2091 path is a real mode, system-manifest.js:224"]
```

## AB-05.4 [vault] — the single corpus path authority, catalogued as two entries
```mermaid
flowchart TD
    TOML["agentbox.toml [vault]<br/>root required, pages, format, tui,<br/>working, transcripts"] --> SCHEMA["schema/agentbox.toml.schema.json<br/>root is required"]
    TOML --> E1["catalogue entry 'vault'<br/>gate vault.format — apply_class BOOT<br/>system-manifest.js:249-251"]
    TOML --> E2["catalogue entry 'vault-tui'<br/>gate vault.tui — apply_class REBUILD<br/>system-manifest.js:252-254"]
    E1 --> WHY1["root/pages/format are read ONCE by the entrypoint<br/>at container start — a restart picks them up"]
    E2 --> WHY2["tui decides the Nix package set (ADR-2029)<br/>none to rune needs ./agentbox.sh rebuild, not a restart"]
    E1 --> SPLIT["ADR-039 honesty rule — one entry claiming 'boot' for both<br/>would tell an operator that flipping tui and restarting<br/>gets them the Rune TUI. It does not.<br/>system-manifest.js:242-248"]
    E2 --> SPLIT
    TOML --> VB["resolved vault block from buildSystemView<br/>system-manifest.js:357-372"]
    VB --> VB1["enabled = Boolean(root) — :358"]
    VB --> VB2["pages = root minus trailing slashes + '/' + pages default 'pages' — :360"]
    VB --> VB3["format default 'obsidian' when root set — :361"]
    VB --> VB4["tui default 'none' when root set — :362"]
    VB --> VB5["ADR-2028 amendment 2026-09-02 — working_root, working_pages,<br/>transcripts sibling-vault keys — :364-366"]
    VB --> VB6["env_root, env_pages, env_working_pages, env_transcripts<br/>read from the process the container actually booted with — :367-370"]
    VB6 --> DRIFT["drift = root set AND VAULT_ROOT set AND they differ<br/>system-manifest.js:371"]
    DRIFT --> DOC["so /v1/system and the doctor can show manifest-vs-running drift"]
    TOML -.-> DIV["DIVERGENCE — BASELINE 'Vault compatibility and Notes qualification 2026-09-04'.<br/>ADR-2028 is partial for universal disablement: the no-vault resolver clears<br/>VAULT_PAGES but retains a legacy ONTOLOGY_PAGES_DIR override consumers prefer.<br/>See AB-02 for the entrypoint resolution path"]
```

## AB-05.5 agentbox.sh top-level subcommand dispatch
```mermaid
flowchart LR
    ARG["arg parse loop<br/>agentbox.sh:1901-1922"] --> AL{"subcommand in the<br/>allowlist at agentbox.sh:1911?"}
    AL -->|no| ERR["Unknown command then usage then exit 1<br/>agentbox.sh:1917-1919"]
    AL -->|yes| CASE["execute case CMD<br/>agentbox.sh:2435-2471"]
    CASE --> G1["access: ssh :2436, vnc :2437, browser :2438,<br/>code :2439, api :2440, all :2441, ip :2443"]
    CASE --> G2["lifecycle: up :2449, down :2450, build :2451,<br/>rebuild :2452, update :2453, status :2442"]
    CASE --> G3["provisioning: provision :2444, setup :2445,<br/>start-browser :2446, migrate-workspace :2468, preflight :2469"]
    CASE --> G4["data: backup :2447, restore :2448,<br/>ruvector :2454, ruvnet-brain :2455"]
    CASE --> G5["observe: logs :2456, shell :2457, health :2458"]
    CASE --> G6["sidecars: browsercontainer :2459, gui-tools :2460,<br/>openmed :2461, voice :2462, model-router :2464 ADR-2080,<br/>xr-runtime :2465, android :2466"]
    G2 --> RB["cmd_rebuild agentbox.sh:1065<br/>= cmd_down then cmd_build --variant runtime<br/>then cmd_up --build then post-deploy-cleanup.sh"]
    G4 --> RV["cmd_ruvector agentbox.sh:1022<br/>exec bash scripts/ruvector-sidecar-update.sh — see AB-05.6"]
    G5 --> HL["cmd_health agentbox.sh:1131 — see AB-05.7"]
    G5 --> SH["cmd_shell agentbox.sh:1114"]
    SH -.-> RES1["RESOLVED ADR-2038: cmd_shell agentbox.sh:1128 now uses<br/>cd /home/devuser/workspace/profiles/${profile} && exec fish — the old<br/>finding was that it execed the retired literal /workspace/profiles/PROFILE<br/>path (agentbox/CLAUDE.md 'Runtime model gotchas'); every other profile<br/>path in the script already used SCRIPT_DIR/workspace/profiles (agentbox.sh:351, :515)"]
```

## AB-05.6 agentbox.sh ruvector — a dispatch table split across two files
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant AB as cmd_ruvector<br/>agentbox.sh:1022
    participant SC as ruvector-sidecar-update.sh<br/>scripts/ruvector-sidecar-update.sh:1172-1189 dispatch
    participant H as scripts/ruvector-recall-harness.mjs

    OP->>AB: ./agentbox.sh ruvector <subcmd> [args]
    AB->>SC: exec bash SCRIPT_DIR/scripts/ruvector-sidecar-update.sh "$@" (agentbox.sh:1029)
    Note over AB: image pin lives in agentbox.toml [integrations.ruvector_external] and is mirrored into docker-compose.yml (agentbox.sh:1024-1027)
    alt status | check | test (ruvector-sidecar-update.sh:1173-1175)
        SC-->>OP: sidecar state, pinned-vs-Docker-Hub comparison
    else update (:1176)
        SC->>SC: dump then pg_basebackup snapshot then candidate rehearsal then swap
    else rollback (:1177)
        SC->>SC: restore previous image plus datadir from the recorded snapshot
    else migrate-trajectories | repair-namespaces | backfill-embeddings | archive-legacy | aggregate-effectiveness | build-metadata-gin (:1178-1183)
        SC-->>OP: DRY-RUN by default — each needs --yes plus its manifest flag
    else recall (:1184)
        SC->>SC: cmd_recall (:1146) — require_prod_running then node present
        SC->>SC: resolve governed MCP env via mcp_env_pairs from .mcp.json (:1156)
        SC->>H: env ENVP node ruvector-recall-harness.mjs "$@" (:1167)
        H-->>OP: harness sets its own exit code — PASS 0, FAIL non-zero (ruvector-sidecar-update.sh:1161)
    else anything else
        SC-->>OP: die unknown subcommand (:1188)
    end
    Note over SC,H: recall is READ-ONLY, no gate of its own — fixture scripts/recall-fixtures/recall-fixture.v1.json is frozen and checked in (:1159)
    Note over H: classes self-recall@10, true-recall@10 vs forced exact scan, exact-token — median-of-3 no-regression band (ruvector-sidecar-update.sh:1160-1161)
    Note over H: artefact lands in backups/ruvector-sidecar/recall-runs/<utc>.json (ruvector-sidecar-update.sh:1162) — retrieval-geometry gate boundary, see AB-20
    Note over AB,SC: DOC-DRIFT — the usage text at agentbox.sh:48 lists the ruvector subcommands but OMITS recall, which ruvector-sidecar-update.sh:1184 implements and :1188 advertises
    Note over AB,SC: RESOLVED ADR-2038: recall is in the ruvector subcommand list at<br/>agentbox.sh:48 and has a usage example at agentbox.sh:96
```

## AB-05.7 agentbox.sh health — exit-code contract
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant SH as cmd_health<br/>agentbox.sh:1131
    participant H as GET /health<br/>server.js:565-576
    participant M as GET /v1/meta<br/>localhost:9090

    OP->>SH: ./agentbox.sh health [--json]
    SH->>H: curl -sf HEALTH_URL (agentbox.sh:619 and :1142)
    alt curl fails
        SH-->>OP: ERROR could not reach — exit 1 (agentbox.sh:1143-1145)
    else --json passed
        SH-->>OP: raw JSON then exit 0 (agentbox.sh:1149-1150)
    else jq absent
        SH-->>OP: warning plus raw response (agentbox.sh:1221-1222)
    else pretty path
        SH->>SH: degraded = jq '.adapters // {} | select(.value != healthy and != off)' (agentbox.sh:1157-1162)
        SH->>SH: degraded_count = jq '.degraded_count // 0' (agentbox.sh:1163)
        SH->>SH: print adapter/<slot> lines from .adapters (agentbox.sh:1167-1170)
        SH->>M: curl /v1/meta then read observability.metrics_endpoint (agentbox.sh:1184-1187)
        M-->>SH: metrics endpoint
        SH->>SH: print first 5 non-comment metric lines (agentbox.sh:1193-1198)
        alt degraded non-empty OR degraded_count > 0
            SH-->>OP: exit 1 (agentbox.sh:1217-1218)
        else
            SH-->>OP: exit 0
        end
    end
    Note over SH,H: RESOLVED ADR-2037 — the old finding was that /health emits<br/>status, uptime, image_hash, manifest_checksum, adapters, degraded_count, note<br/>(server.js:567-575) with no services key, so a stale jq '.services // {}' read<br/>left the exit-1 branch unreachable. cmd_health now derives failure from<br/>.adapters (a slot fails when neither "healthy" nor "off", agentbox.sh:1157-1162)<br/>plus .degraded_count (agentbox.sh:1163) — exit 1 at agentbox.sh:1217-1218<br/>is reachable, and the condition now also fails on speech_failed (agentbox.sh:1217).<br/>Same fix as AB-04.16
    Note over SH: BASELINE-container Adapter spine stage 4 claims agentbox.sh health exits non-zero if any slot gauge is 0 — it never reads the agentbox_adapter_health gauge at all. See AB-04.16
    Note over H: /health self-describes as human-inspection-only and points orchestrators at /ready (server.js:574)
```

## AB-05.8 Artifact validation gate — the last check before exec supervisord
```mermaid
sequenceDiagram
    autonumber
    participant EP as config/entrypoint-unified.sh
    participant VA as validate-artifacts.sh<br/>config/validate-artifacts.sh:11
    participant PF as config/artifact-probes.json<br/>16 probes
    participant SH as probe_command shell

    EP->>VA: invoked immediately before exec supervisord (validate-artifacts.sh:11)
    VA->>VA: set -euo pipefail (:13)
    VA->>PF: read AGENTBOX_PROBES_FILE default /opt/agentbox/config/artifact-probes.json (:15)
    alt probes file missing
        VA-->>EP: log ProbesFileMissing then exit 1 (:32-34)
    end
    alt jq absent
        VA-->>EP: log MissingDependency tool=jq then exit 1 (:37-40)
    end
    loop for each probe entry
        VA->>SH: run probe_command
        alt exit 0
            SH-->>VA: pass
        else required_for_readiness true
            SH-->>VA: fail — validate-artifacts.sh exits 1 (:5)
        else optional
            SH-->>VA: warn and continue (:6)
        end
    end
    VA-->>EP: pino-style JSON lines on stdout with agentbox.stage bootstrap (:8-9 and :26-30)
    Note over PF: probe fields capability_id, entrypoint_path, required_for_readiness, probe_command
    Note over PF: only 2 of 16 are required_for_readiness — management-api and mcp-nostr-bridge, both node --check syntax gates
    Note over PF: optional probes cover openai-codex-mcp, lazy-fetch-mcp, browser-sidecar HTTP health, ruflo-cli, claude-flow-cli, native-sqlite-backend, self-learning-hook-adapter, agentic-qe-cli, nagual-qe-cli, codebase-memory-mcp-cli, mermaid-cli, code-interpreter-mcp, code-interpreter-wheelhouse, aci-shell-mcp
    Note over VA,SH: the gate is a SYNTAX and PRESENCE check, not a behavioural one — node --check and --version dominate
```

## AB-05.9 Apply-class as an operator decision procedure
```mermaid
stateDiagram-v2
    [*] --> EditManifest
    EditManifest --> Validate: agentbox-config-validate.js plus schema/agentbox.toml.schema.json
    Validate --> Rejected: structural schema violation
    Validate --> Classify: valid manifest
    Rejected --> EditManifest

    Classify --> Live: apply_class live
    Classify --> Boot: apply_class boot
    Classify --> Rebuild: apply_class rebuild

    Live --> Effective: read at operation time, no restart
    Boot --> RestartNeeded: entrypoint reconciles every boot
    RestartNeeded --> Effective: docker restart or agentbox.sh up
    Rebuild --> RebuildNeeded: changes the Nix image composition
    RebuildNeeded --> Effective: agentbox.sh rebuild at agentbox.sh:1065

    Effective --> Verify: GET /v1/system shows state and apply_class
    Verify --> [*]

    note right of Live
        APPLY_CLASSES system-manifest.js:28
        example browser-sidecar
    end note
    note right of Boot
        APPLY_CLASSES system-manifest.js:29
        example vault gate vault.format
    end note
    note right of Rebuild
        APPLY_CLASSES system-manifest.js:30
        example vault-tui gate vault.tui ADR-2029
        also code-server, jupyter, desktop, comfyui
    end note
    note left of Classify
        INVARIANT adding a gate means gating BOTH the Nix
        package set AND the supervisor block, plus a
        catalogue entry with an honest apply class
    end note
```

## AB-05.10 Schema surfaces and the setup wizard boundary
```mermaid
flowchart TD
    S1["agentbox/schema/agentbox.toml.schema.json<br/>the only file in schema/"] --> V["scripts/agentbox-config-validate.js<br/>static stage — see AB-01.4"]
    S2["agentbox/schemas/mcp/<br/>MCP registry schemas"] --> P["skills/mcp.json projector — see AB-09"]
    V --> GATE["valid gate set feeds the flake evaluator — see AB-01.1"]
    S1 --> SW["setup-wizard catalogue entry<br/>system-manifest.js:47-49<br/>service 'setup', apply_class boot"]
    SW --> SWB["ephemeral localhost manifest editor with schema validation<br/>EXITS AFTER SAVING"]
    SWB -.-> DIV["DIVERGENCE BASELINE Known divergences — operations moved to the<br/>AoE cockpit. Legacy docs describing pseudo-user isolation<br/>gemini-user and friends are dead paths, not the runtime model"]
    S1 --> VS["[vault] keys root required, pages, format, tui<br/>plus ADR-2028 amendment working and transcripts"]
    VS --> AB4["consumed by buildSystemView vault block — see AB-05.4"]
    GATE --> MAN["/v1/system catalogue — see AB-05.1"]
```

## AB-05.11 agentbox.sh model-router — ADR-2080 CLI dispatch, gated by [model_routing.neural]
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant AB as cmd_model_router<br/>agentbox.sh:2075
    participant FETCH as scripts/model-router-fetch.sh
    participant CONSOLE as config/model-router/console.mjs
    participant WRAP as config/harness-wrappers/router.sh

    OP->>AB: ./agentbox.sh model-router <sub> [args]
    alt sub == fetch (agentbox.sh:2080)
        AB->>FETCH: exec scripts/model-router-fetch.sh "$@"
        Note right of FETCH: populates $WORKSPACE/.agentbox/model-router,<br/>hash-verified against config/model-router/artefacts.json — pre-rebuild fallback
    else sub == check (agentbox.sh:2081)
        AB->>FETCH: exec scripts/model-router-fetch.sh --check "$@"
    else sub == status (agentbox.sh:2082, default when sub omitted :2076)
        AB->>CONSOLE: exec node console.mjs --status "$@"
    else sub == route (agentbox.sh:2083-2084)
        AB->>AB: require at least 1 positional arg or usage+exit 2
        AB->>CONSOLE: exec node console.mjs --once "<task>" "$@"
    else sub == console (agentbox.sh:2085)
        AB->>WRAP: exec config/harness-wrappers/router.sh "$@"
        Note right of WRAP: same wrapper the AoE `router` seed's<br/>custom_agents program execs — see AB-02.20
    else unknown (agentbox.sh:2086-2095)
        AB-->>OP: usage heredoc, exit 2
    end

    Note over AB,CONSOLE: EXTERNAL — console.mjs's own request/response flow to the OpenRouter<br/>provider and the KRR/k-NN router logic is AB-29 (model-routing-neural topic), not here
    Note over AB: cmd_model_router itself is UNGATED — the CLI runs regardless of<br/>[model_routing.neural].enabled, fetch/status/console fail loud if the<br/>rebuild-baked or fetch-populated artefact dir is absent — see AB-01.11
```

## AB-05.12 setup/agentbox-setup — pre-boot wizard pointer (audit gap 3, full detail in AB-30)

```mermaid
flowchart LR
    CAT["catalogue entry 'setup-wizard'<br/>system-manifest.js:47-49<br/>service 'setup', apply_class boot"] --> BIN["agentbox-setup Rust binary<br/>setup/server/src/main.rs"]
    BIN -.-> DETAIL["full request flow (config round trip,<br/>management-API proxy, three-tier fallback<br/>to static browser mode) — see AB-30.2, AB-30.3"]
```

## AB-05.13 The gate families added since 2026-09-06: System One, compaction, resources
```mermaid
flowchart TB
    TOML["agentbox.toml"] --> F1["[features.jev_compaction] enabled = true<br/>agentbox.toml:622, :645"]
    TOML --> F2["[features.sovereign_system_one] enabled = false<br/>agentbox.toml:654, :679"]
    TOML --> F3["[skills.routing] router = jev<br/>agentbox.toml:876, :895"]
    TOML --> F4["[resources] mcp_hub, hooks.shim, session_hygiene"]
    TOML --> F5["[voice] enabled = false<br/>agentbox.toml:1681"]

    F1 --> C1["catalogue jev-compaction, apply_class boot<br/>system-manifest.js:212-213"]
    F2 --> C2["catalogue sovereign-system-one, apply_class boot<br/>system-manifest.js:215-216"]
    F3 --> C3["catalogue skill-router, off_values table, apply_class boot<br/>system-manifest.js:223-224"]
    F4 --> C4["catalogue mcp-hub, hook-shim, teammate-gc, all rebuild<br/>system-manifest.js:270-277"]
    F5 --> C5["catalogue voice-console, apply_class live<br/>system-manifest.js:166-167"]

    C1 --> B1["the entrypoint installs or uninstalls the plugin<br/>and sets CLAUDE_CODE_ENABLE_FUNCTION_HOOKS<br/>entrypoint-unified.sh:2208, :2216"]
    C2 --> B2["the entrypoint runs agentbox-manifest sso-project and<br/>projects baseUrl, model and backendLocal into the plugin<br/>entrypoint-unified.sh:2053-2056"]
    C3 --> B3["the entrypoint registers or de-registers the<br/>UserPromptSubmit hook, see AB-08<br/>entrypoint-unified.sh:2115, :2130"]

    C5 --> DIV["DIVERGENCE - the manifest declares voice off while the voice<br/>stack has its own compose lifecycle outside agentbox up,<br/>so a running console reports state off, agentbox.toml:1681"]
    C4 --> REB["INVARIANT - mcp-hub is rebuild-class because its supervisor<br/>program is composed into the image, so flipping the gate and<br/>restarting does not add it, system-manifest.js:271"]
```

**What it shows.** The five manifest gate families added since the last stamp of this topic, each traced from its `agentbox.toml` section to its catalogue entry and to the boot step that applies it.
**Why it is this way.** ADR-039's honesty rule: a gate is catalogued with the apply class of the place it is consumed, so `sovereign_system_one` is boot-class here even though its sidecar has a lifecycle of its own (`../project/agentbox/management-api/lib/system-manifest.js:216`).

**Tension (manifest vs running estate):** `[voice].enabled = false` (`../project/agentbox/agentbox.toml:1681`) while the voice console runs under its own compose lifecycle, so the live view reports `voice-console` off for a surface that is up (`../project/agentbox/management-api/lib/system-manifest.js:166`).

**Debt:** `payment_settlement` is declared `zero-tolerance` with its own task properties (`../project/agentbox/agentbox.toml:980`, `:1028`) but no route passes that action class to the authority gate; `mandate_revoke` is the only class any route names (`../project/agentbox/management-api/routes/llm-marketplace.js:457`).

**Debt:** the agent, command and skill registries governed by ADR-2092 are manifest files of their own (`registered-agents.txt`, `registered-commands.txt`, `registered-skills.txt`) with no `agentbox.toml` gate and no catalogue entry, so `GET /v1/system` cannot report what the reconcilers did (`../project/agentbox/config/entrypoint-unified.sh:2662`, `../project/agentbox/management-api/lib/system-manifest.js:39`).

**Drift (gate catalogue vs the sealed chain):** `[sidechain]` is a fully specified manifest gate — catalogue rows, apply classes and three supervised programs — that cannot be added to `agentbox.toml` at all, because the schema sets `additionalProperties: false` at the top level and has no `sidechain` entry; the chain it would govern has nevertheless been sealed and is running outside the manifest entirely (see AB-32.5, AB-34.5).
