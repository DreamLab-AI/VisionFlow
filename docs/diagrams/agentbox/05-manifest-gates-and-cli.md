---
id: AB-05
title: Manifest gate catalogue, vault path authority and the agentbox.sh CLI
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2003, ADR-2028, ADR-2029, ADR-2034, ADR-2036, ADR-2037, ADR-2038, ADR-2039, ADR-2080, ADR-2091, ADR-2092, ADR-2093, ADR-2094, ADR-2095, ADR-2107, ADR-2108, ADR-2110, ADR-2116, ADR-2118, ADR-2122, ADR-2140, ADR-2142, ADR-2144, ADR-2145, ADR-2146, ADR-2149, ADR-2151, ADR-2152, ADR-2139, ADR-2153, ADR-2154, ADR-2155]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/management-api/routes/system.js
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/scripts/ruvector-sidecar-update.sh
  - ../project/agentbox/config/validate-artifacts.sh
  - ../project/agentbox/config/artifact-probes.json
  - ../project/agentbox/docs/adr/ADR-2146-activity-sink-hook.md
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
  - ../project/agentbox/docs/adr/ADR-2152-render-mermaid-in-browsercontainer.md
  - ../project/agentbox/docs/adr/ADR-2154-pele-renders-mermaid-in-the-image.md
  - ../project/agentbox/docs/adr/ADR-2139-system-one-judge-as-a-vendor-slot.md
  - ../project/agentbox/docs/adr/ADR-2155-system-one-consumer-activation.md
  - ../project/agentbox/docs/reference/system-one-activation.md
  - ../project/agentbox/services/agentbox-manifest/src/system_one.rs
  - ../project/agentbox/lib/factrail.nix
  - ../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md
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
  - ../project/agentbox/config/instructions/
  - ../project/agentbox/services/agentbox-manifest/src/instructions.rs
  - ../project/agentbox/services/agentbox-manifest/src/cred_sync.rs
  - ../project/agentbox/services/agentbox-manifest/src/main.rs
  - ../project/agentbox/services/agentbox-manifest/src/role_accounts.rs
  - ../project/agentbox/config/role-accounts.json
  - ../project/agentbox/setup/agentbox.default.toml
  - ../project/agentbox/flake.nix
  - ../project/agentbox/management-api/lib/junkiejarvis-clarify.js
  - ../project/agentbox/services/agentbox-manifest/tests/golden/live-agentbox.toml
  - ../project/agentbox/scripts/experiments/exp-b8-label-log.cjs
verified_commit: 6500582dfae23bf8a38a6942916d208154e43625
---

## AB-05.1 GET /v1/system — catalogue plus live introspection
```mermaid
sequenceDiagram
    autonumber
    participant C as Operator or cockpit
    participant R as routes/system.js:44<br/>fastify.get /v1/system
    participant BV as buildSystemView<br/>system-manifest.js:494
    participant CAT as CATALOGUE const<br/>system-manifest.js:39-436
    participant ST as stateOf<br/>system-manifest.js:453
    participant RG as resolveGate<br/>system-manifest.js:439
    participant AD as resolved adapters

    C->>R: GET /v1/system
    R->>BV: buildSystemView(manifest, adapters, health) — routes/system.js:60
    BV->>BV: core = manifest entry :497 plus identity entry :501
    loop for slot of beads pods memory events orchestrator (system-manifest.js:505-506)
        BV->>AD: read adapter.impl and adapter.CONTRACT_VERSION
        AD-->>BV: impl or 'unresolved', contract_version or null (:510-511)
        BV->>BV: push core entry adapter-<slot> (:508-513)
    end
    BV->>BV: build resolved vault block (:516-539)
    loop for entry of CATALOGUE (system-manifest.js:543)
        BV->>ST: stateOf(manifest, entry)
        ST->>RG: resolveGate(manifest, entry.gate)
        RG-->>ST: gate value walked down the dotted path (:440-448)
        ST-->>BV: on | off | available
        BV->>BV: push to surfaces or modules by entry.layer (:556)
    end
    BV-->>R: apply_classes, core, vault, health, surfaces, modules, counts (:562-580)
    R-->>C: live system view
    Note over BV,CAT: INVARIANT — the catalogue is documentation-as-data but STATE is always introspected from the parsed agentbox.toml at request time, never hard-coded (system-manifest.js:6-13)
    Note over BV: counts block emits core, surfaces_on, surfaces, modules_on, modules (:573-579)
    Note over BV,AD: RESOLVED ADR-2036: system-manifest.js:512 reads<br/>"every dispatch wrapped by observability → privacy redaction<br/>(ADR-2036). JSON-LD encoding is a per-surface gated stage<br/>invoked by the owning route, not a dispatch layer" — see AB-04.4
    Note over ST: stateOf first consults entry.requires (:456) and a false<br/>parent gate (:458) before the multi-gate and scalar branches
    Note over R: sibling route GET /v1/system/audit-chain verifies the hash-chained events JSONL (routes/system.js:65)
```

## AB-05.2 Catalogue shape and the real surface/module census
```mermaid
flowchart TD
    CAT["CATALOGUE array<br/>system-manifest.js:39-436<br/>121 entries total"] --> SURF["layer 'surface' — 14 entries"]
    CAT --> MOD["layer 'module' — 107 entries"]
    SURF --> S1["ungated: management-api, terminal, setup-wizard,<br/>uri-resolver, agent-events-stream, metrics<br/>system-manifest.js:41-71"]
    SURF --> S2["gated: code-server, jupyter, desktop, comfyui,<br/>linked-data-viewer, interaction-plane, tab0-bridge<br/>system-manifest.js:50-77"]
    MOD --> M1["toolchain + CLI: ruflo, agentic-qe, nagual-qe, deepsec,<br/>role-isolation ADR-2122, codebase-memory, metaharness-binaries,<br/>metaharness-plugin, ruflo-console, codex, model-routing,<br/>model-routing-neural ADR-2080, antigravity, agentbox-sessions-binary,<br/>rust-toolchain, opencode — system-manifest.js:85-130"]
    MOD --> M2["GPU/media: qgis-mcp, blender-mcp, imagemagick-mcp, ffmpeg,<br/>pytorch, docs-toolchain, code-interpreter, web-researcher,<br/>compression, tailscale, dream-machine, cuda<br/>system-manifest.js:133-169, gaussian-splatting :415-417"]
    MOD --> M3["sidecars flagged heavy: browser-sidecar :172, gui-tools-sidecar :175,<br/>voice-console :181, privacy-filter :241<br/>system-manifest.js:172-182, :241-243"]
    MOD --> M4["sovereign/governance: sovereign-mesh :190, solid-pod :195,<br/>linked-data :214, payments :217, payments-sidestr :220,<br/>poker-citizen :223, poker-citizen-dreamlab-txbt4 :226,<br/>poker-coach :229, llm-marketplace :232, project-tracking :235,<br/>consultants :238, plugins :244, orchestration-proxy :247<br/>system-manifest.js:190-247"]
    MOD --> M5["memory: ruvector-external :204, memory-learning :207,<br/>memory-hygiene :210, ruvnet-brain :250<br/>system-manifest.js:204-212, :250-252"]
    MOD --> M6["NEW ADR-2116/ADR-2118: claude-code-permissions :253,<br/>claude-code-model-roles :256 (ADR-2134), instruction-tiers :259,<br/>claude-cred-sync :265<br/>system-manifest.js:253-267"]
    MOD --> M7["System One / Jev routing ADR-2091/2093/2094/2095/2110:<br/>jev-compaction :268, jev-checkpoints :271 (ADR-2135), system-one-proxy :274 (ADR-2139), system-one-network :277 (2026-10-30),<br/>sovereign-system-one :280, 19 system-one-consumer-* gates :288-344,<br/>skill-router :345, skill-router-cascade :348, routing-teacher-labels :351<br/>system-manifest.js:268-352 — see AB-05.13"]
    MOD --> M8["ontology :360, harness-bridge :367, colloquy :370,<br/>podcast-ingest :373, NEW diagram-review :376, NEW seo-watch :379<br/>system-manifest.js:360-381"]
    MOD --> M9["corpus: vault :397, vault-tui :400,<br/>vault-cli :403 NEW ADR-2107/ADR-2108<br/>system-manifest.js:397-405"]
    MOD --> M10["aci-shell :409, tree-search-coder :412, resources-envelope :421,<br/>mcp-hub :424, hook-shim :427, teammate-gc :430, NEW target-gc :433 (ADR-2138)<br/>system-manifest.js:409-435"]
    MOD --> M11["NEW since e4993a3b: sidechain system-manifest.js:382 (three gates,<br/>requires sidechain.enabled), sidechain-dreamlab-txbt4 :385 ADR-2103"]
    MOD --> M12["NEW since 3f05ae9e7: junkiejarvis :80 (surface), blender-local :139, isaac-sim-sidecar :178,<br/>voicelink :184, voicelink-hooks :187, solid-pod-forge :198, solid-pod-trust-proxy :201,<br/>collaboration-floor :262, activity-sink :357, vault-loom-sync :406 — see AB-05.17"]
    CAT --> FIELDS["per-entry fields: id, name, layer, gate or gates,<br/>service, apply_class, summary, heavy, requires"]
    FIELDS --> CORE["core layer emitted separately by buildSystemView<br/>manifest, identity, five adapter-<slot> entries<br/>system-manifest.js:495-513"]
```

## AB-05.3 stateOf — how a gate value becomes a state word
```mermaid
flowchart TD
    E["catalogue entry"] --> REQ{"entry.requires has a gate that is not true?<br/>system-manifest.js:456"}
    REQ -->|yes| OFFR["state 'off' — a false parent<br/>dominates every child"]
    REQ -->|no| PG{"entry.gate resolves false?<br/>system-manifest.js:458"}
    PG -->|yes| OFFR
    PG -->|no| A{"Array.isArray(entry.gates)?<br/>system-manifest.js:461"}
    A -->|yes| MG["resolve every gate path"]
    MG --> MG1{"any value === true?"}
    MG1 -->|yes| ON1["state 'on' — :463"]
    MG1 -->|no| MG2{"any value === false?"}
    MG2 -->|yes| OFF1["state 'off' — :464"]
    MG2 -->|no| AV1["state 'available' — :465"]
    A -->|no| B{"entry.gate falsy?<br/>system-manifest.js:467"}
    B -->|yes| ON2["state 'on' — ungated surface,<br/>present whenever the image is"]
    B -->|no| RG["resolveGate(manifest, entry.gate)<br/>system-manifest.js:439"]
    RG --> W["walk the dotted path key by key<br/>system-manifest.js:440-444"]
    W --> SEC{"cursor is an object?<br/>system-manifest.js:446"}
    SEC -->|yes| SECE["section gate resolves via its .enabled key — :446-448"]
    SEC -->|no| VAL["scalar value — :449"]
    SECE --> D
    VAL --> D{"value type"}
    D -->|"true"| ON3["state 'on' — :469"]
    D -->|"false"| OFF2["state 'off' — :470"]
    D -->|"string"| MODE{"value in off, none or entry.off_values?<br/>system-manifest.js:477-479"}
    D -->|"undefined"| AV2["state 'available' — catalogued<br/>but unconfigured — :481"]
    MODE -->|yes| OFF3["state 'off'"]
    MODE -->|no| ON4["state 'on'"]
    MODE -.-> VT["ADR-2029 — vault.tui is a mode string naming a thing<br/>not a state, so 'none' is the conventional disabled value.<br/>Vanilla default tui = 'none' reports vault-tui as off.<br/>ADR-2091 — skill-router declares its OWN off value, table,<br/>because the pre-2091 path is a real mode, system-manifest.js:345"]
```

## AB-05.4 [vault] — the single corpus path authority, catalogued as three entries
```mermaid
flowchart TD
    TOML["agentbox.toml [vault]<br/>root required, pages, format, tui, cli,<br/>working, transcripts"] --> SCHEMA["schema/agentbox.toml.schema.json<br/>root is required"]
    TOML --> E1["catalogue entry 'vault'<br/>gate vault.format — apply_class BOOT<br/>system-manifest.js:397-399"]
    TOML --> E2["catalogue entry 'vault-tui'<br/>gate vault.tui — apply_class REBUILD<br/>system-manifest.js:400-402"]
    TOML --> E3["NEW ADR-2107/ADR-2108: catalogue entry 'vault-cli'<br/>gate vault.cli — apply_class REBUILD<br/>system-manifest.js:403-405"]
    E1 --> WHY1["root/pages/format are read ONCE by the entrypoint<br/>at container start — a restart picks them up"]
    E2 --> WHY2["tui decides the Nix package set (ADR-2029)<br/>none to rune needs ./agentbox.sh rebuild, not a restart"]
    E3 --> WHY3["cli bakes the vault corpus CLI into the package set;<br/>the ontology-bridge MCP server it replaces is retired,<br/>so false leaves agents with no programmatic door"]
    E3 --> LIVE["boot Phase 5d(ii) liveness gate runs vault --version<br/>and prints a rebuild notice on failure — entrypoint-unified.sh:1159-1192"]
    E1 --> SPLIT["ADR-039 honesty rule — one entry claiming 'boot' for both<br/>would tell an operator that flipping tui and restarting<br/>gets them the Rune TUI. It does not.<br/>system-manifest.js:391-396"]
    E2 --> SPLIT
    TOML --> VB["resolved vault block from buildSystemView<br/>system-manifest.js:524-539"]
    VB --> VB1["enabled = Boolean(root) — :525"]
    VB --> VB2["pages = root minus trailing slashes + '/' + pages default 'pages' — :527"]
    VB --> VB3["format default 'obsidian' when root set — :528"]
    VB --> VB4["tui default 'none' when root set — :529"]
    VB --> VB5["ADR-2028 amendment 2026-09-02 — working_root, working_pages,<br/>transcripts sibling-vault keys — :531-533"]
    VB --> VB6["env_root, env_pages, env_working_pages, env_transcripts<br/>read from the process the container actually booted with — :534-537"]
    VB6 --> DRIFT["drift = root set AND VAULT_ROOT set AND they differ<br/>system-manifest.js:538"]
    DRIFT --> DOC["so /v1/system and the doctor can show manifest-vs-running drift"]
    TOML -.-> DIV["DIVERGENCE — BASELINE 'Vault compatibility and Notes qualification 2026-09-04'.<br/>ADR-2028 is partial for universal disablement: the no-vault resolver clears<br/>VAULT_PAGES but retains a legacy ONTOLOGY_PAGES_DIR override consumers prefer.<br/>See AB-02 for the entrypoint resolution path"]
```

## AB-05.5 agentbox.sh top-level subcommand dispatch
```mermaid
flowchart LR
    ARG["arg parse loop<br/>agentbox.sh:2038-2059"] --> AL{"subcommand in the<br/>allowlist at agentbox.sh:2048?"}
    AL -->|no| ERR["Unknown command then usage then exit 1<br/>agentbox.sh:2054-2057"]
    AL -->|yes| CASE["execute case CMD<br/>agentbox.sh:2861-2903"]
    CASE --> G1["access: ssh :2862, vnc :2863, browser :2864,<br/>code :2865, api :2866, all :2867, status :2868, ip :2869"]
    CASE --> G2["lifecycle: up :2875, down :2876, build :2877,<br/>NEW prepare :2878, NEW activate :2879 (ADR-2132),<br/>rebuild :2880, update :2881, NEW security :2882 (ADR-2137, see the Debt below)"]
    CASE --> G3["provisioning: provision :2870, setup :2871,<br/>start-browser :2872, migrate-workspace :2899,<br/>migrate-claude-home ADR-2118 :2900, preflight :2901"]
    CASE --> G4["data: backup :2873, restore :2874,<br/>ruvector :2883, ruvnet-brain :2884"]
    CASE --> G5["observe: logs :2885, shell :2886, health :2887"]
    CASE --> G6["sidecars: browsercontainer :2888, gui-tools :2889,<br/>NEW isaac-sim :2890 (ADR-2149), openmed :2891, voice :2892,<br/>NEW config :2893 (plan, validate), systemone :2894 ADR-2094,<br/>model-router :2895 ADR-2080, xr-runtime :2896, android :2897,<br/>NEW concat :2898 gated AGENTBOX_ENABLE_CONCAT=1 agentbox.sh:616-618"]
    G2 --> RB["cmd_rebuild agentbox.sh:1084 now delegates to<br/>scripts/runtime-delivery.sh rebuild — prepare a candidate<br/>image, then scoped activation, without recreating sidecars<br/>or volumes (ADR-2132); agentbox.sh prepare :2878 and<br/>activate :2879 call the same flow's halves"]
    G4 --> RV["cmd_ruvector agentbox.sh:1041<br/>exec bash scripts/ruvector-sidecar-update.sh :1048 — see AB-05.6"]
    G5 --> HL["cmd_health agentbox.sh:1130 — see AB-05.7"]
    G5 --> SH["cmd_shell agentbox.sh:1110"]
    SH -.-> RES1["RESOLVED ADR-2038: cmd_shell agentbox.sh:1127 now uses<br/>cd /home/devuser/workspace/profiles/${profile} && exec fish — the old<br/>finding was that it execed the retired literal /workspace/profiles/PROFILE<br/>path (agentbox/CLAUDE.md 'Runtime model gotchas'); every other profile<br/>path in the script already used SCRIPT_DIR/workspace/profiles"]
```

**Debt:** the `config` arm of the dispatch case (`../project/agentbox/agentbox.sh:2893`) is unreachable: the argument loop admits only the allowlisted subcommands and `config` is not among them (`../project/agentbox/agentbox.sh:2048`), so `./agentbox.sh config plan "<request>"` prints "Unknown command" although `cmd_config` exists (`../project/agentbox/agentbox.sh:2441-2447`); `scripts/config-plan.cjs` and `scripts/agentbox-config-validate.js` still run directly.

**Drift (resolved, `security` subcommand):** the `security` arm used to be unreachable because the allowlist omitted it; the allowlist at this stamp lists `security` (`../project/agentbox/agentbox.sh:2048`) and the arm runs `scripts/security-scan.sh` (`../project/agentbox/agentbox.sh:2882`), which `cmd_up --build` and `cmd_build` also call directly (`../project/agentbox/agentbox.sh:765`, `:940`).

## AB-05.6 agentbox.sh ruvector — a dispatch table split across two files
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant AB as cmd_ruvector<br/>agentbox.sh:1041
    participant SC as ruvector-sidecar-update.sh<br/>scripts/ruvector-sidecar-update.sh:1777-1797 dispatch
    participant H as scripts/ruvector-recall-harness.mjs

    OP->>AB: ./agentbox.sh ruvector <subcmd> [args]
    AB->>SC: exec bash SCRIPT_DIR/scripts/ruvector-sidecar-update.sh "$@" (agentbox.sh:1048)
    Note over AB: image pin lives in agentbox.toml [integrations.ruvector_external] and is mirrored into docker-compose.yml (agentbox.sh:1042-1047)
    alt status | check | test (ruvector-sidecar-update.sh:1778-1780)
        SC-->>OP: sidecar state, pinned-vs-Docker-Hub comparison
    else update (ruvector-sidecar-update.sh:1781)
        SC->>SC: dump then pg_basebackup snapshot then candidate rehearsal then swap
    else rollback (ruvector-sidecar-update.sh:1782)
        SC->>SC: restore previous image plus datadir from the recorded snapshot
    else migrate-trajectories | repair-namespaces | backfill-embeddings | archive-legacy | aggregate-effectiveness | build-metadata-gin (ruvector-sidecar-update.sh:1783-1788)
        SC-->>OP: DRY-RUN by default — each needs --yes plus its manifest flag
    else recall (ruvector-sidecar-update.sh:1789)
        SC->>SC: cmd_recall (:1245) — require_prod_running then node present
        SC->>SC: resolve governed MCP env via mcp_env_pairs from .mcp.json (:1251)
        SC->>H: run_recall_harness (:1262) runs env ENVP node "$RECALL_HARNESS" "$@" (:1207), default scripts/ruvector-recall-harness.mjs (:1198)
        H-->>OP: harness sets its own exit code — PASS 0, FAIL non-zero (ruvector-sidecar-update.sh:1256)
    else reindex | reader-role | hba-harden (ruvector-sidecar-update.sh:1790-1792, ADR-2133)
        SC-->>OP: serial HNSW rebuild with recall gate before and after, reader role, pg_hba hardening
    else anything else
        SC-->>OP: die unknown subcommand (:1796)
    end
    Note over SC,H: recall is READ-ONLY, no gate of its own — fixture scripts/recall-fixtures/recall-fixture.v1.json is frozen and checked in (:1254)
    Note over H: classes self-recall@10, true-recall@10 vs forced exact scan, exact-token — median-of-3 no-regression band (ruvector-sidecar-update.sh:1255-1256)
    Note over H: artefact lands in backups/ruvector-sidecar/recall-runs/<utc>.json (ruvector-sidecar-update.sh:1257) — retrieval-geometry gate boundary, see AB-20
    Note over AB,SC: DOC-DRIFT — RESOLVED: the usage text at agentbox.sh:51 lists recall,<br/>which ruvector-sidecar-update.sh:1789 implements and :1796 advertises
    Note over AB,SC: RESOLVED ADR-2038: recall is in the ruvector subcommand list at<br/>agentbox.sh:51 and has a usage example at agentbox.sh:104
```

## AB-05.7 agentbox.sh health — exit-code contract
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant SH as cmd_health<br/>agentbox.sh:1130
    participant H as GET /health<br/>server.js:602-613
    participant M as GET /v1/meta<br/>localhost:9090

    OP->>SH: ./agentbox.sh health [--json]
    SH->>H: curl -sf HEALTH_URL (agentbox.sh:633 and :1141)
    alt curl fails
        SH-->>OP: ERROR could not reach — exit 1 (agentbox.sh:1143-1145)
    else --json passed
        SH-->>OP: raw JSON then exit 0 (agentbox.sh:1147-1149)
    else jq absent
        SH-->>OP: warning plus raw response (agentbox.sh:1221-1222)
    else pretty path
        SH->>SH: degraded = jq '.adapters // {} | select(.value != healthy and != off)' (agentbox.sh:1158-1162)
        SH->>SH: degraded_count = jq '.degraded_count // 0' (agentbox.sh:1163)
        SH->>SH: print adapter/<slot> lines from .adapters (agentbox.sh:1167-1176)
        SH->>M: curl /v1/meta then read observability.metrics_endpoint (agentbox.sh:1185-1187)
        M-->>SH: metrics endpoint
        SH->>SH: print first 5 non-comment metric lines (agentbox.sh:1193-1201)
        alt degraded non-empty OR degraded_count > 0
            SH-->>OP: exit 1 (agentbox.sh:1217-1218)
        else
            SH-->>OP: exit 0
        end
    end
    Note over SH,H: RESOLVED ADR-2037 — the old finding was that /health emits<br/>status, uptime, image_hash, manifest_checksum, adapters, degraded_count, note<br/>(server.js:604-612) with no services key, so a stale jq '.services // {}' read<br/>left the exit-1 branch unreachable. cmd_health now derives failure from<br/>.adapters (a slot fails when neither "healthy" nor "off", agentbox.sh:1158-1162)<br/>plus .degraded_count (agentbox.sh:1163) — exit 1 at agentbox.sh:1217-1218<br/>is reachable, and the condition now also fails on speech_failed (agentbox.sh:1204-1217).<br/>Same fix as AB-04.16
    Note over SH: BASELINE-container Adapter spine stage 4 claims agentbox.sh health exits non-zero if any slot gauge is 0 — it never reads the agentbox_adapter_health gauge at all. See AB-04.16
    Note over H: /health self-describes as human-inspection-only and points orchestrators at /ready (server.js:611)
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
    Note over PF: optional probes cover openai-codex-mcp, lazy-fetch-mcp, browser-sidecar HTTP health, ruflo-cli, claude-flow-cli, native-sqlite-backend, self-learning-hook-adapter, agentic-qe-cli, nagual-qe-cli, codebase-memory-mcp-cli, pele-render, code-interpreter-mcp, code-interpreter-wheelhouse, aci-shell-mcp
    Note over VA,SH: the gate is a SYNTAX and PRESENCE check, not a behavioural one — node --check and --version dominate
    Note over PF: the render probe is now pele-render --check on a one-line flowchart (artifact-probes.json:75-78), since ADR-2154 replaced the mmdc sidecar client of ADR-2152<br/>it renders in-process, so a pass says the pinned Pele parses a diagram, not merely that a client exists
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
    RebuildNeeded --> Effective: agentbox.sh rebuild at agentbox.sh:1084<br/>(runtime-delivery.sh rebuild — prepare then activate, ADR-2132)

    Effective --> Verify: GET /v1/system shows state and apply_class
    Verify --> [*]

    note right of Live
        APPLY_CLASSES system-manifest.js:27
        example browser-sidecar
    end note
    note right of Boot
        APPLY_CLASSES system-manifest.js:28
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
    SWB -.-> DIV["DIVERGENCE BASELINE Known divergences — operations use the voice/operator<br/>console and the terminal session picker. Legacy docs describing pseudo-user isolation<br/>gemini-user and friends are dead paths, not the runtime model"]
    SWB --> BIN["agentbox-setup Rust binary<br/>setup/server/src/main.rs<br/>full request flow, config round trip, management-API proxy,<br/>three-tier fallback to static browser mode — see AB-30.2, AB-30.3"]
    S1 --> VS["[vault] keys root required, pages, format, tui<br/>plus ADR-2028 amendment working and transcripts"]
    VS --> AB4["consumed by buildSystemView vault block — see AB-05.4"]
    GATE --> MAN["/v1/system catalogue — see AB-05.1"]
```

## AB-05.11 agentbox.sh model-router — ADR-2080 CLI dispatch, gated by [model_routing.neural]
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant AB as cmd_model_router<br/>agentbox.sh:2390
    participant FETCH as scripts/model-router-fetch.sh
    participant CONSOLE as config/model-router/console.mjs
    participant WRAP as config/harness-wrappers/router.sh

    OP->>AB: ./agentbox.sh model-router <sub> [args]
    alt sub == fetch (agentbox.sh:2395)
        AB->>FETCH: exec scripts/model-router-fetch.sh "$@"
        Note right of FETCH: populates $WORKSPACE/.agentbox/model-router,<br/>hash-verified against config/model-router/artefacts.json — pre-rebuild fallback
    else sub == check (agentbox.sh:2396)
        AB->>FETCH: exec scripts/model-router-fetch.sh --check "$@"
    else sub == status (agentbox.sh:2397, default when sub omitted :2392)
        AB->>CONSOLE: exec node console.mjs --status "$@"
    else sub == route (agentbox.sh:2398-2399)
        AB->>AB: require at least 1 positional arg or usage+exit 2
        AB->>CONSOLE: exec node console.mjs --once "<task>" "$@"
    else sub == console (agentbox.sh:2400)
        AB->>WRAP: exec config/harness-wrappers/router.sh "$@"
        Note right of WRAP: same wrapper the agentbox-sessions `router` seed<br/>execs — see AB-29.3
    else unknown (agentbox.sh:2401-2410)
        AB-->>OP: usage heredoc, exit 2
    end

    Note over AB,CONSOLE: EXTERNAL — console.mjs's own request/response flow to the OpenRouter<br/>provider and the KRR/k-NN router logic is AB-29 (model-routing-neural topic), not here
    Note over AB: cmd_model_router itself is UNGATED — the CLI runs regardless of<br/>[model_routing.neural].enabled, fetch/status/console fail loud if the<br/>rebuild-baked or fetch-populated artefact dir is absent — see AB-01.11
```

## AB-05.13 The gate families added since 2026-09-06: System One, compaction, permissions, instructions, cred-sync, resources
```mermaid
flowchart TB
    TOML["agentbox.toml"]
    TOML --> F1["[features.jev_compaction] enabled = true, fallback = rules<br/>agentbox.toml:742, :780, :802"]
    TOML --> F2["[features.sovereign_system_one] enabled = false<br/>agentbox.toml:966, :991"]
    TOML --> F3["[skills.routing] router = jev, cascade, label_log<br/>label_log = false since 2026-10-04 — the bounded EXP-B8 run<br/>STOPPED (WITHDRAW, commit b1e42f148)<br/>agentbox.toml:1273, :1293, :1309, :1323"]
    TOML --> F4["[resources] mcp_hub, hooks.shim, session_hygiene<br/>agentbox.toml:1578, :1596, :1608, :1612"]
    TOML --> F5["[voice] enabled = true<br/>agentbox.toml:2209-2210"]
    TOML --> F6["NEW ADR-2116: [claude_code] permission_mode :704,<br/>permission_deny :713-741"]
    TOML --> F7["NEW ADR-2118: config/instructions/ (not a toml key —<br/>gate is null, the directory's own presence)"]
    TOML --> F8["NEW ADR-2118: [toolchains] claude_code = true :2237<br/>service claude-cred-sync"]
    TOML --> F9["NEW PRD-024 P1: [sidechain] enabled, mirror, faucet<br/>agentbox.toml:1919-1925; enabled = true again after the ENOSPC recovery (:1921)"]
    TOML --> F11["NEW ADR-2134: [claude_code] model, effort, advisor_model<br/>agentbox.toml:727-729"]
    TOML --> F12["NEW ADR-2135: [features.jev_checkpoints] enabled = true<br/>agentbox.toml:807"]
    TOML --> F13["NEW ADR-2139: [system_one] enabled = true, backend typesafe again (amendment 2026-10-29),<br/>bind 0.0.0.0:8094 with network = true (docker networks only, never published, 2026-10-30),<br/>local_only = [] (every consumer takes the proxy), local_backend laya kept as the opt-in private route,<br/>skill_list_at_least = 0.6 (router lists runner-up skills, needs skill_verify)<br/>agentbox.toml:863, :890-896"]
    TOML --> F14["NEW ADR-2155: [system_one.consumers], 19 keys all true<br/>14 exported as AGENTBOX_JEV_*, 5 exported only for other repos<br/>agentbox.toml:902-923"]
    F9 ~~~ F10["NEW 2026-10-02: [sovereign_mesh] junkiejarvis = true<br/>the only switch, the env override is gone<br/>agentbox.toml:50, junkiejarvisEnabled junkiejarvis-clarify.js:447-453"]
    TOML --> F10

    F1 --> C1["catalogue jev-compaction (factrail), apply_class REBUILD<br/>since ADR-2121, was boot - system-manifest.js:268-269"]
    F2 --> C2["catalogue sovereign-system-one, apply_class boot<br/>system-manifest.js:280-281"]
    F3 --> C3["catalogue skill-router :345-346, skill-router-cascade :348-349,<br/>routing-teacher-labels :351-352, off_values table, all apply_class boot"]
    F4 --> C4["catalogue mcp-hub, hook-shim, teammate-gc, all rebuild<br/>system-manifest.js:424-432"]
    F5 --> C5["catalogue voice-console, apply_class live<br/>system-manifest.js:181-182"]
    F6 --> C6["NEW catalogue claude-code-permissions, apply_class boot<br/>system-manifest.js:253-255"]
    F7 --> C7["NEW catalogue instruction-tiers, gate null, apply_class boot<br/>system-manifest.js:259-261"]
    F8 --> C8["NEW catalogue claude-cred-sync, apply_class rebuild<br/>system-manifest.js:265-267"]
    F9 --> C9["NEW catalogue sidechain, three gates, apply_class rebuild<br/>a false parent gate reports off - system-manifest.js:382-384, :458"]

    C1 --> B1["flake.nix bakes factrail pinned to rev d270a0a1 (lib/factrail.nix:36), which records the judge that answered from the proxy headers,<br/>then the entrypoint sets CLAUDE_CODE_ENABLE_FUNCTION_HOOKS,<br/>drops any retired jev-compaction install, migrates its store, installs factrail@agentbox<br/>entrypoint-unified.sh:3045, :3055-3059, :3069, :3142 - gate off uninstalls :3153-3164"]
    C2 --> B2["the entrypoint runs agentbox-manifest sso-project and<br/>projects baseUrl, model and backendLocal into the plugin<br/>entrypoint-unified.sh:2700-2703"]
    C3 --> B3["the entrypoint registers or de-registers the<br/>UserPromptSubmit hook, see AB-08<br/>entrypoint-unified.sh:2791, :2809"]
    C6 --> B6["agentbox-manifest permissions-project reconciles root<br/>settings.json and every profile's settings every boot — not seed-if-unset<br/>entrypoint-unified.sh:1831-1840"]
    C7 --> B7["agentbox-manifest instructions-project composes<br/>~/.claude/CLAUDE.md, ~/workspace/AGENTS.md, ~/workspace/CLAUDE.md<br/>from config/instructions/ every boot — entrypoint-unified.sh:2985-3007"]
    F11 --> C11["NEW catalogue claude-code-model-roles, gate claude_code.advisor_model,<br/>off_values off, apply_class boot - system-manifest.js:256-258"]
    F12 --> C12["NEW catalogue jev-checkpoints, apply_class rebuild<br/>system-manifest.js:271-273"]
    F13 --> C13["NEW catalogue system-one-proxy, service system-one-proxy, apply_class boot (system-manifest.js:274-276),<br/>and system-one-network, gate system_one.network, apply_class boot (system-manifest.js:277-279)"]
    C11 --> B11["permissions-project writes model, effortLevel and advisorModel into settings<br/>every boot, skipping gateway profiles<br/>services/agentbox-manifest/src/main.rs:226"]
    C12 --> B12["the entrypoint installs jev-checkpoints@agentbox with the manifest as userConfig<br/>entrypoint-unified.sh:3168, :3036"]
    C13 --> B13["system-one-project writes /run/agentbox/system-one.json and projects the proxy URL<br/>entrypoint-unified.sh:2670-2683, see AB-02.3"]
    F14 --> C14["NEW catalogue system-one-consumer-* x19, requires system_one.enabled, boot class<br/>except solid-pod-advisories, ontomark-suggestions and vault-rank, rebuild<br/>system-manifest.js:288-344"]
    C14 --> B14["system_one.rs exports one AGENTBOX_JEV_* flag per in-repo consumer (:409-516, the skill_list_at_least rate only with skill_verify on, :469-481), then AGENTBOX_JEV_X_LOCAL_ONLY=0 for all twelve unless the consumer is named in [system_one].local_only (empty by default; :522-577);<br/>the five external consumers get VISIONFLOW_JEV, DREAMLAB_JEV, SOLID_POD_JEV, VISIONCLAW_JEV_PATCH_RANK and ONTOMARK_JEV exports, each with _ALLOW_CLOUD=1 and no laya endpoint unless listed, which pins it to the laya route or switches it off (:537-623)<br/>services/agentbox-manifest/src/system_one.rs:409-623"]
    C9 --> B9["flake.nix bakes program:sidestr-producer, -mirror, -faucet<br/>see AB-01.1 and AB-02.6"]
    C8 --> B8["[program:claude-cred-sync] polls container ~/.claude/.credentials.json<br/>against host-claude bind every 2s, later expiresAt wins<br/>flake.nix:3266-3278, services/agentbox-manifest/src/cred_sync.rs"]

    C5 --> DIV["RESOLVED - the manifest now declares voice on, apply-class sidecar,<br/>so a running console no longer reports state off, agentbox.toml:2210"]
    F10 --> C10["catalogue junkiejarvis, layer surface, gate sovereign_mesh.junkiejarvis,<br/>apply_class boot - system-manifest.js:80-82;<br/>server.js:1449-1450 and the forum-suggestions tenant<br/>both call junkiejarvisEnabled(manifest)"]
    C4 --> REB["INVARIANT - mcp-hub is rebuild-class because its supervisor<br/>program is composed into the image, so flipping the gate and<br/>restarting does not add it, system-manifest.js:424-426"]
```

**What it shows.** The thirteen manifest gate families added since 2026-09-06 — System One, Jev compaction (now factrail, ADR-2121) and its cascade/teacher-label addenda, the ADR-2116 permission posture, ADR-2118 instruction tiers and credential sync, the PRD-024 `[sidechain]` gate of 2026-09-30, and, since 2026-10, the ADR-2134 model roles, the ADR-2135 Jev checkpoints and the ADR-2139 System One vendor slot (`[system_one]`, one proxy in front of every typed-decision consumer, loopback in the container and, since the 2026-10-30 amendment, `bind = "0.0.0.0:8094"` with `network = true` for the private docker networks, `../project/agentbox/agentbox.toml:893-895`; the manifest ships it on with backend `typesafe` again since 2026-10-29, `../project/agentbox/agentbox.toml:892`, and nineteen `[system_one.consumers]` switches, `../project/agentbox/agentbox.toml:902-923`) — each traced from its `agentbox.toml` section to its catalogue entry and to the boot step that applies it.
**Why it is this way.** ADR-039's honesty rule: a gate is catalogued with the apply class of the place it is consumed, so `sovereign_system_one` is boot-class here even though its sidecar has a lifecycle of its own (`../project/agentbox/management-api/lib/system-manifest.js:281`); `claude-cred-sync` is rebuild-class because the polling daemon is a supervised program baked into the image (`flake.nix:3267`), not because the credential merge logic itself needs a rebuild. The same rule moved `jev-compaction` from boot to rebuild when ADR-2121 replaced the vendored plugin with a factrail binary and plugin that `flake.nix` bakes only when the gate is on (`../project/agentbox/management-api/lib/system-manifest.js:269`).

**Tension (resolved by this revision, manifest vs running estate):** the manifest used to declare `[voice].enabled = false` while the voice console ran under its own compose lifecycle, so the live view reported `voice-console` off for a surface that was up. It now reads `enabled = true`, apply-class sidecar (`../project/agentbox/agentbox.toml:2210`), which the catalogue entry gates on (`../project/agentbox/management-api/lib/system-manifest.js:181-182`).

**Open:** the nineteen consumer switches and the unified route are configured, not live: ADR-2155 is `accepted`, `implementation_status: complete`, `activation_status: staged` (`../project/agentbox/docs/adr/ADR-2155-system-one-consumer-activation.md:5-7`), so the running container keeps the old routing until the new image is activated, and five consumers (visionflow, website, solid pod, ontomark, vault rank) are projected as exports only (`../project/agentbox/services/agentbox-manifest/src/system_one.rs:537-623`), their deployment sitting outside Agentbox (`../project/agentbox/docs/reference/system-one-activation.md:23-26`).

**Drift:** the activation note still says the running public vendor remains SubQ and that private consumers use the `laya` backend (`../project/agentbox/docs/reference/system-one-activation.md:3-6`), but ADR-2139's amendments record the switch back to `typesafe` with a 200 from model `jev-1.14.0` and, on 2026-10-30, every consumer through the proxy with `local_only` empty (`../project/agentbox/docs/adr/ADR-2139-system-one-judge-as-a-vendor-slot.md:256-270`, `:274-295`), factrail `d270a0a1` recording the answering backend from the proxy headers (`../project/agentbox/lib/factrail.nix:36`).

**Debt:** `payment_settlement` is declared `zero-tolerance` with its own task properties (`../project/agentbox/agentbox.toml:1405`, `:1429`) but no route passes that action class to the authority gate; `mandate_revoke` is the only class any route names (`../project/agentbox/management-api/routes/llm-marketplace.js:473`).

**Drift (resolved 2026-10-02, JunkieJarvis gate):** the manifest comment used to promise that `JUNKIEJARVIS_ENABLED` overrode `[sovereign_mesh].junkiejarvis`; `f63760e19` made the key the only switch and turned it on (`../project/agentbox/agentbox.toml:45-50`, still true at this revision), and management-api now reads it through `junkiejarvisEnabled(manifest)` alone (`../project/agentbox/management-api/server.js:1449-1450`, `../project/agentbox/management-api/lib/junkiejarvis-clarify.js:447-453`).

**Drift (resolved, JunkieJarvis catalogue entry):** `junkiejarvis` gates a forum-facing agent and the forum-suggestions tenant's clarifying DMs and used to have no `CATALOGUE` entry; it now has one, a `surface` entry gated on `sovereign_mesh.junkiejarvis` with apply-class `boot` (`../project/agentbox/management-api/lib/system-manifest.js:80-82`), so `GET /v1/system` can say whether it is on.

**Drift (golden fixture vs manifest):** the manifest parser's golden copy of the live manifest still carries the retired wording, with BOTH the gate and the env var required and the gate false (`../project/agentbox/services/agentbox-manifest/tests/golden/live-agentbox.toml:45-51`).

**Tension (ADR-030 gate vs EXP-B8):** EXP-B8 stopped on 2026-10-04 (WITHDRAW, `b1e42f148`; `label_log = false` since, `../project/agentbox/agentbox.toml:1323`), but the experiment tick itself signs its one forum summary with the JunkieJarvis key (`../project/agentbox/scripts/experiments/exp-b8-label-log.cjs:528-530`; since custody W2 the key is read as a role secret and, under `[security].role_isolation`, never from the repo `.env`) and publishes it (`../project/agentbox/scripts/experiments/exp-b8-label-log.cjs:550-563`) without consulting `junkiejarvisEnabled`, so a manifest with `junkiejarvis = false` would still see a JunkieJarvis post if the withdrawn tick fires again.

**Debt:** the agent, command and skill registries governed by ADR-2092 are manifest files of their own (`registered-agents.txt`, `registered-commands.txt`, `registered-skills.txt`) with no `agentbox.toml` gate and no catalogue entry, so `GET /v1/system` cannot report what the reconcilers did (`../project/agentbox/config/entrypoint-unified.sh:3783`, `../project/agentbox/management-api/lib/system-manifest.js:39`).

**Drift (resolved 2026-09-30, gate catalogue vs the sealed chain):** `[sidechain]` used to be a fully specified gate the schema refused; it now exists in the schema (`../project/agentbox/schema/agentbox.toml.schema.json:3971-3975`, itself `additionalProperties: false`), in `agentbox.toml` (`../project/agentbox/agentbox.toml:1919-1925`) and in the catalogue (`../project/agentbox/management-api/lib/system-manifest.js:382-384`), gating the interim producer, mirror and faucet (see AB-32.5, AB-34.5).

**Tension (proposed design vs the landed gate):** the BASELINE's PROPOSED supervised-set table still gates `sidestr-producer` on `[sidechain.signer].enabled` (`../project/agentbox/docs/BASELINE-container.md:402`), while the code gates it on the top-level `[sidechain].enabled`, which the live manifest now sets true after the ENOSPC recovery (`../project/agentbox/agentbox.toml:1921`); the `signer` and `bridge` sub-blocks (`../project/agentbox/docs/BASELINE-container.md:361-362`) have no schema entry, so adding either is a schema change too.

## AB-05.14 Instruction tier projection (ADR-2118, floor ADR-2144) — the mount contract four tiers rewrite every boot
```mermaid
sequenceDiagram
    autonumber
    participant EP as entrypoint-unified.sh:2985-3007
    participant AM as agentbox-manifest instructions-project<br/>main.rs:457-473
    participant IR as instructions::run<br/>instructions.rs:141-222
    participant COMP as compose/composed<br/>instructions.rs:71-74
    participant FS as target files

    EP->>EP: mount /etc/agentbox/instructions read-only<br/>(config/instructions/ tracked + gitignored local/)
    EP->>EP: _FLOOR_ON from [claude_code].floor, an absent key means on (entrypoint-unified.sh:2112-2113)
    EP->>AM: instructions-project --layers ... --managed-out /etc/claude-code/CLAUDE.md, or --no-managed when the floor is off (entrypoint-unified.sh:2994-2996)
    AM->>IR: run(layers, global_out, managed_out, workspace_out, workspace_claude_out, check)
    opt managed_out is Some (instructions.rs:151)
        IR->>COMP: composed(layers, floor) — floor.md + local/floor.md
        IR->>FS: project floor to the managed policy path (instructions.rs:152)
        alt managed write fails and not check
            IR->>IR: floor_fallback = composed floor, printed loudly (instructions.rs:169-171)
        end
    end
    loop for tier in global, workspace, workspace.claude (instructions.rs:180-190)
        IR->>COMP: composed(layers, tier) — read <tier>.md + local/<tier>.md
        COMP-->>IR: header (:35-40) plus tracked and local text joined, or None if both absent
        alt workspace.claude
            IR->>IR: embed composed workspace text at the @AGENTS.md line (:110-112)
        else global with a floor_fallback
            IR->>IR: prepend(next, floor) between header and body (:87-93, :113-115)
        end
        IR->>FS: read(target) == next? Unchanged (:116-117) : write tmp then rename (:125-128)
        FS-->>IR: Written | Unchanged | Drift (--check, :119-121) | NoLayers (:108)
    end
    IR-->>EP: fail-open — a write error never blocks boot, check mode exits non-zero on drift only
    EP->>FS: floor off and the managed file carries the GENERATED marker, rm it (entrypoint-unified.sh:3004-3007)
    EP->>FS: Codex: append managed floor, global and workspace files to ~/.codex/AGENTS.md when each is under 12000 bytes (entrypoint-unified.sh:3586-3589)
    Note over EP,FS: INVARIANT — the repo is authoritative: outputs are rewritten every boot, so a live edit to<br/>~/.claude/CLAUDE.md or ~/workspace/{AGENTS,CLAUDE}.md never survives the next restart (instructions.rs:29-30)
    Note over EP: global_out is None when ~/.claude is still a legacy whole-directory host bind<br/>(no /var/lib/agentbox/host-claude sign) — the host keeps its own CLAUDE.md, entrypoint-unified.sh:2989-2993
```

**What it shows.** How the instruction tiers that sit above every repository get from `config/instructions/` layers (tracked public, gitignored `local/` estate) to the files every agent reads, on every single boot. Since ADR-2144 there are four tiers: the floor goes to Claude Code's managed policy path `/etc/claude-code/CLAUDE.md` (loaded by every session and subagent, `claudeMdExcludes` cannot drop it), and only if that write fails is it prepended to the global tier. Codex has no managed path, so the entrypoint appends the managed file, then the global and workspace files, to its `AGENTS.md`.
**Why it is this way.** ADR-2118: the files used to be hand-kept on a host bind and a Docker volume with no history and no review; composing them from the repo makes the global/workspace tiers a reviewable, versioned product surface instead of drift-prone container state. ADR-2144 (accepted, live) adds the floor tier so custom subagents that start without the user and project notes still meet it, and ties it to the `decline-recorder` hook under the same `[claude_code].floor` gate (`../project/agentbox/config/entrypoint-unified.sh:2112`, `../project/agentbox/management-api/lib/system-manifest.js:262`).

**Invariant:** an unwritable managed path never drops the floor: the projector falls back into the global tier (`../project/agentbox/services/agentbox-manifest/src/instructions.rs:169-172`).

## AB-05.15 Claude OAuth credential sync (ADR-2118) — converging container and host `.credentials.json`
```mermaid
sequenceDiagram
    autonumber
    participant SV as supervisord<br/>[program:claude-cred-sync]<br/>flake.nix:3266-3279
    participant CLI as agentbox-manifest cred-sync<br/>main.rs:475-484
    participant RUN as cred_sync::run<br/>cred_sync.rs:136-152
    participant SYNC as sync_once<br/>cred_sync.rs:109-129
    participant MERGE as merge<br/>cred_sync.rs:29-62
    participant CV as container ~/.claude/.credentials.json
    participant HV as host-claude bind<br/>/var/lib/agentbox/host-claude/.credentials.json

    SV->>CLI: exec agentbox-manifest cred-sync --container CV --host HV --interval-secs 2
    CLI->>RUN: run(container, host, interval=2s, once=false)
    RUN->>RUN: host bind directory absent? log and exit 0 — nothing to sync (cred_sync.rs:137-143)
    loop every interval (cred_sync.rs:152-156)
        RUN->>SYNC: sync_once(container, host)
        SYNC->>CV: load — mtime, parsed JSON or None (:70-80)
        SYNC->>HV: load — mtime, parsed JSON or None (:70-80)
        alt either side exists but fails to parse (mid-write)
            SYNC-->>RUN: skip this tick, Ok(0) (:112-114)
        else both present
            SYNC->>MERGE: merge(container_val, host_val, container.mtime >= host.mtime)
            MERGE->>MERGE: a token record (has numeric expiresAt)? later expiresAt wins whole (:32-37)
            MERGE->>MERGE: else merge object keys recursively, so an MCP login on one side<br/>is never lost to a Claude refresh on the other (:39-51)
            MERGE-->>SYNC: merged document
        end
        SYNC->>SYNC: for each side whose value != merged: atomic temp-file + rename write (:122-127)
        SYNC-->>RUN: count of files rewritten
        RUN-->>SV: log reconciled (n) or nothing when 0 (cred_sync.rs:144-148, invoked at :154)
    end
    Note over MERGE: INVARIANT — a rotating OAuth refresh token invalidates the OTHER side's refresh<br/>token, so both copies must converge fast — polling every 2s is deliberately cheap against<br/>token lifetimes of hours (cred_sync.rs:131-133)
    Note over SV,HV: claude-code-permissions and claude-cred-sync are siblings of the same ADR-2118 boot<br/>sweep but independent gates — see AB-05.13 F6/F8
```

**What it shows.** The polling reconciliation loop that keeps the container-owned `agentbox-claude-home` volume's OAuth tokens converged with the host's bound `~/.claude`, so a refresh on either side never logs the other out.
**Why it is this way.** `~/.claude` became a container-owned volume (not a host bind) so the container's private key material is never exposed on the host filesystem; the host still needs current tokens to run its own Claude Code, hence a merge daemon rather than a symlink — Claude Code's write strategy (in-place vs temp+rename) is unknown, so a symlink is not trusted to survive a refresh (cred_sync.rs:1-17).

## AB-05.16 [security].role_isolation — one rebuild-class gate, one table, four derived artefacts (ADR-2122)
```mermaid
flowchart TB
    KEY["[security] role_isolation = false<br/>agentbox.toml:2506, setup default agentbox.default.toml:1771"]
    SCH["schema security.role_isolation boolean, default false<br/>agentbox.toml.schema.json:2206-2209"]
    CATE["catalogue role-isolation, gate security.role_isolation, apply_class rebuild (was boot, ADR-2122 ER-9)<br/>system-manifest.js:97-99"]
    TBL["config/role-accounts.json schema agentbox.role-accounts/1<br/>role-accounts.json:3-4 uid_range 960-979, :5-7 gid 965 reserved"]
    subgraph CLI["agentbox-manifest role-accounts (main.rs:286, dispatch main.rs:529-546)"]
        CHK["check: load and validate, exit 1 listing every rule<br/>role_accounts.rs:297 validate, dispatched main.rs:531"]
        PW["passwd and group lines<br/>role_accounts.rs:607, :624"]
        ISO["isolate: today's supervisord.conf to the roles config<br/>and the delivery plan, role_accounts.rs:865, :1093"]
    end
    subgraph BUILD["flake.nix image build, every image, both modes"]
        B1["/etc/supervisord.roles.conf and /etc/agentbox/role-secrets.tsv<br/>flake.nix:4017"]
        B2["/etc/passwd role lines flake.nix:4037<br/>/etc/group role lines flake.nix:4050"]
    end
    BOOT["entrypoint reads it once in Stage A and exports it<br/>entrypoint-unified.sh:354, see AB-02.24"]
    KEY --> SCH
    KEY --> CATE
    KEY --> BOOT
    TBL --> CLI
    PW --> B2
    ISO --> B1
    B1 --> BOOT
    KEY -.-> DRIFT["DRIFT: the key's own comment says the identity port (W3), the W2b<br/>at-rest custody migrate/revert and the W4 sidechain state move are<br/>built but not yet proven in an image, agentbox.toml:2499-2505,<br/>yet serve-identity is baked, flake.nix:2821, and the W2 env capture<br/>and scrub run before exec, entrypoint-unified.sh:497 and :1285"]
```

**What it shows.** The custody gate as the manifest sees it. `[security].role_isolation` is a plain boolean, false in the live manifest and in the setup default, declared in the schema and catalogued as apply-class `rebuild` since ADR-2122 ER-9 (it was `boot`). Everything it switches between is derived at image build from one table, `config/role-accounts.json`, by the `agentbox-manifest role-accounts` subcommands: the passwd and group lines, the isolated supervisor config and the delivery plan. The supervisor pick, the secrets delivery and the environment scrub are read at boot, but the devuser sudo route is removed at image build from the build-time flag (`roleIsolationBaked`), so a boot-time flip alone leaves the box reporting `degraded:devuser-sudo` and needs `./agentbox.sh rebuild` (`../project/agentbox/management-api/lib/system-manifest.js:97-99`, `../project/agentbox/flake.nix:3664`, `:4081`).
**Why it is this way.** The custody design originally made the flag boot-class so the owner's rehearsal could flip it and roll it back without a rebuild; ADR-2122's ER-9 closed that as rebuild-class because `config/bake-devuser-privilege.sh` takes the build-time flag and leaves devuser in `wheel` with passwordless sudo otherwise. Deriving the roles config from today's rendered config, rather than writing it by hand, keeps the two configs from drifting: only `user=` and `environment=` of role programs change (`../project/agentbox/services/agentbox-manifest/src/role_accounts.rs:11-15`).

**Tension (catalogue vs manifest comment):** the catalogue classes `role-isolation` as `rebuild` (`../project/agentbox/management-api/lib/system-manifest.js:97-99`) while the comment above the key still says "Boot-class: the image ships both supervisor configs; restart to apply" (`../project/agentbox/agentbox.toml:2499`).

**Invariant:** the isolated supervisor config is a pure function of today's rendered config and the role table, so no role program can exist in one config and not the other (`../project/agentbox/flake.nix:4017`, `../project/agentbox/services/agentbox-manifest/src/role_accounts.rs:865`).

**Drift (manifest comment vs this revision):** the comment above `role_isolation` says the identity port and the custody migration (W2-W4) are not in the image yet, so role programs fail closed under the flag (`../project/agentbox/agentbox.toml:2499-2505`). At this revision `[program:serve-identity]` is baked (`../project/agentbox/flake.nix:2821`) and the W2 capture and scrub run before `exec supervisord` (`../project/agentbox/config/entrypoint-unified.sh:497`, `../project/agentbox/config/entrypoint-unified.sh:1285`). The comment also adds a new condition the old text did not have: until the W3b consumer cutover lands, JunkieJarvis, the mirror hook and dream-engine cannot sign under the flag. The advice to leave the flag off until the rehearsal passes still holds. The stated reason is stale.

## AB-05.17 Gate families added in October 2026: floor, voicelink, activity sink, forge, Blender sidecar-first, Loom sync, Isaac Sim
```mermaid
flowchart TB
    TOML["agentbox.toml"]
    TOML --> G1["[claude_code] floor = true<br/>agentbox.toml:740, ADR-2144 accepted and live"]
    TOML --> G2["[voicelink] enabled = true, record_events = true<br/>agentbox.toml:2226, :2233, ADR-2151 proposed, none, inactive"]
    TOML --> G3["[integrations.activity_sink] enabled = false<br/>agentbox.toml:529, ADR-2146 accepted, complete, inactive"]
    TOML --> G4["[integrations.solid_pod_rs] trust_proxy = true and .forge enable = true<br/>agentbox.toml:553, :608, ADR-2145 accepted and live"]
    TOML --> G5["[skills.spatial_and_3d] blender_local = false<br/>agentbox.toml:635, sidecar-first Blender"]
    TOML --> G6["[vault] loom_sync = full<br/>agentbox.toml:1184, ADR-2140 accepted and live"]
    NOKEY["no manifest key: the Isaac Sim sidecar is opt-in through<br/>./agentbox.sh isaac-sim, ADR-2149 proposed, partial, staged<br/>agentbox.sh:2890"]

    G1 --> C1["catalogue collaboration-floor, apply_class boot<br/>system-manifest.js:262-264"]
    G2 --> C2["catalogue voicelink, apply_class rebuild :184-186<br/>and voicelink-hooks, apply_class boot :187-189"]
    G3 --> C3["catalogue activity-sink, apply_class boot<br/>system-manifest.js:357-359"]
    G4 --> C4["catalogue solid-pod-forge :198, solid-pod-trust-proxy :201,<br/>both apply_class rebuild"]
    G5 --> C5["catalogue blender-local, requires skills.spatial_and_3d.blender,<br/>apply_class rebuild, system-manifest.js:139-141"]
    G6 --> C6["catalogue vault-loom-sync, apply_class boot<br/>system-manifest.js:406-408"]
    NOKEY --> C7["catalogue isaac-sim-sidecar, gate null, apply_class live<br/>system-manifest.js:178-180"]

    C1 --> B1["entrypoint projects the floor tier to /etc/claude-code/CLAUDE.md and<br/>registers decline-recorder on Stop and SubagentStop, see AB-05.14<br/>entrypoint-unified.sh:2112-2139, :2985-2998"]
    C2 --> B2["flake.nix bakes voice-paned only when voicelinkOn, linked at<br/>/opt/agentbox/bin/voice-paned - flake.nix:1847-1849, :2128;<br/>entrypoint registers Stop and Notification hooks :2172-2183<br/>and the Codex notify line :873-883"]
    C3 --> B3["entrypoint registers config/hooks/activity-sink.cjs only when enabled<br/>and the url is http or https - entrypoint-unified.sh:2853-2871"]
    C4 --> B4["flake.nix adds forge-full and the --trust-proxy launcher flag<br/>flake.nix:1547, :1628"]
    C5 --> B5["flake.nix bakes the in-container Blender only when blender and blender_local<br/>flake.nix:1476, :4357"]
    C6 --> B6["every boot runs hp-loom-reconcile.sh in the background, fail-open<br/>entrypoint-unified.sh:1208-1218"]
```

**What it shows.** The gate families that arrived between the previous stamp and this one, from manifest key through catalogue entry and apply class to the code that consumes it. Ten entries joined the catalogue (one surface, nine modules), taking it to 102 (`../project/agentbox/management-api/lib/system-manifest.js:39`, `:379`). The apply class follows where the gate is consumed: the voicelink binary is baked at image composition, so `voicelink` is rebuild-class, while the hook registration that feeds its event spool is reconciled every boot.
**Why it is this way.** ADR-039's honesty rule again: each entry carries the class of its consumer. ADR-2144 (accepted, live) and ADR-2145 (accepted, live) are drawn as live. ADR-2146 is accepted but inactive and ADR-2149 is proposed, so the activity sink is drawn off (its manifest key is false) and the Isaac sidecar is shown only as an opt-in CLI path with no manifest gate.

**Tension (ADR-2151 vs manifest):** ADR-2151 is `proposed`, implementation `none`, activation `inactive` (`../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md:5`), while the live manifest sets `[voicelink].enabled = true` and `record_events = true` (`../project/agentbox/agentbox.toml:2226`, `:2235`), so the image bakes voice-paned and boot registers its hooks (`../project/agentbox/flake.nix:1847`, `../project/agentbox/config/entrypoint-unified.sh:2168`).

**Open (status moved 2026-10-30):** ADR-2146 is now `decision_status: accepted` with `implementation_status: complete` and `activation_status: inactive` (`../project/agentbox/docs/adr/ADR-2146-activity-sink-hook.md:5-7`); the decision question is closed, but is the activity sink meant to stay off until a consumer sets the endpoint? The manifest ships `enabled = false` with an empty `url` (`../project/agentbox/agentbox.toml:529`).

**Open:** ADR-2149 (Isaac Sim) is proposed and staged, yet the catalogue already carries an ungated `live` entry for it (`../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md:5`, `../project/agentbox/management-api/lib/system-manifest.js:178`).

**Open:** ADR-2152 (Mermaid rendering in browsercontainer) is `superseded` by ADR-2154, which is `accepted`, `implementation_status: partial` and `activation_status: staged` (`../project/agentbox/docs/adr/ADR-2152-render-mermaid-in-browsercontainer.md:5`, `../project/agentbox/docs/adr/ADR-2154-pele-renders-mermaid-in-the-image.md:5-7`); the artifact probe already runs the Pele check rather than `mmdc` (`../project/agentbox/config/artifact-probes.json:75-78`).
