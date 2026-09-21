---
id: AB-28
title: Agentbox service crates and the manifest binary
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2030, ADR-2031, ADR-2032, ADR-2084, ADR-2085, ADR-2086, ADR-2104]
sources:
  - ../project/agentbox/services/agentbox-manifest/src/main.rs
  - ../project/agentbox/services/agentbox-manifest/src/mcp.rs
  - ../project/agentbox/services/agentbox-manifest/src/routing.rs
  - ../project/agentbox/services/agentbox-manifest/src/proxy.rs
  - ../project/agentbox/services/agentbox-manifest/src/stacks.rs
  - ../project/agentbox/services/agentbox-manifest/src/plugins.rs
  - ../project/agentbox/services/agentbox-manifest/src/tui_read.rs
  - ../project/agentbox/services/agentbox-manifest/src/tui_sections.rs
  - ../project/agentbox/services/agentbox-manifest/src/tui_write.rs
  - ../project/agentbox/services/agentbox-manifest/src/tomlval.rs
  - ../project/agentbox/services/agentbox-ops/src/lib.rs
  - ../project/agentbox/services/agentbox-ops/src/process_identity.rs
  - ../project/agentbox/services/agentbox-ops/src/cost_cap/mod.rs
  - ../project/agentbox/services/agentbox-ops/src/voyager/gate.rs
  - ../project/agentbox/services/agentbox-mcp/src/main.rs
  - ../project/agentbox/services/agentbox-mcp/src/hub/mod.rs
  - ../project/agentbox/services/skill-tools/src/lib.rs
  - ../project/agentbox/services/ontology-tools/src/lib.rs
  - ../project/agentbox/services/podcast-ingest/src/lib.rs
  - ../project/agentbox/services/explainer-tools/src/bin/loom_draft.rs
  - ../project/agentbox/services/agentbox-mcp/src/web_summary/llm.rs
  - ../project/agentbox/crates/colloquy/colloquy-mcp/src/server.rs
  - ../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md
  - ../project/agentbox/docs/adr/ADR-2084-one-published-loom-client-for-every-facade-caller.md
  - ../project/agentbox/mcp/servers/harness-bridge.js
  - ../project/agentbox/mcp/servers/substrate-tools.js
  - ../project/agentbox/mcp/servers/governance-bridge.js
  - ../project/agentbox/mcp/servers/decision-tools.js
  - ../project/agentbox/mcp/servers/nostr-bridge.js
  - ../project/agentbox/mcp/servers/mcp-ws-relay.js
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/.agentic-qe/llm-config.json
  - ../project/agentbox/flake.nix
  - ../project/agentbox/mcp/mcp.json
  - ../project/agentbox/services/agentbox-manifest/tests/golden.rs
  - ../project/agentbox/services/agentbox-ops/tests/reaper_default_is_read_only.rs
  - ../project/agentbox/skills/mcp.json
  - scripts/diagram-index-gen.cjs
  - ../project/agentbox/lib/agentbox-manifest.nix
  - ../project/agentbox/lib/agentbox-mcp.nix
  - ../project/agentbox/lib/agentbox-ops.nix
  - ../project/agentbox/lib/dream-engine.nix
  - ../project/agentbox/lib/ontology-tools.nix
  - ../project/agentbox/lib/podcast-ingest.nix
  - ../project/agentbox/lib/skill-tools.nix
  - ../project/agentbox/mcp/servers/continual-harness.cjs
  - ../project/agentbox/mcp/servers/ontology-bridge.js
  - ../project/agentbox/mcp/servers/ontology-local.cjs
  - ../project/agentbox/mcp/servers/ontology-propose.js
  - ../project/agentbox/mcp/servers/ontology-workingset.cjs
  - ../project/agentbox/mcp/servers/ruvector-mcp.cjs
  - ../project/agentbox/services/agentbox-manifest/tests/consultant_model.rs
  - ../project/agentbox/services/agentbox-manifest/tests/golden_entrypoint.rs
  - ../project/agentbox/services/agentbox-mcp/src/gemini_url_context/api.rs
  - ../project/agentbox/services/agentbox-mcp/src/imagemagick/args.rs
  - ../project/agentbox/services/agentbox-mcp/src/imagemagick/exec.rs
  - ../project/agentbox/services/agentbox-mcp/src/web_summary/fetch.rs
  - ../project/agentbox/services/agentbox-mcp/src/web_summary/youtube.rs
  - ../project/agentbox/services/skill-tools/src/docs_alignment/ascii_diagrams.rs
  - ../project/agentbox/services/skill-tools/src/docs_alignment/links.rs
  - ../project/agentbox/services/skill-tools/src/docs_alignment/links_external.rs
  - ../project/agentbox/services/skill-tools/src/docs_alignment/mermaid.rs
  - ../project/agentbox/services/skill-tools/src/docs_alignment/models.rs
  - ../project/agentbox/services/skill-tools/src/docs_alignment/orchestrator.rs
  - ../project/agentbox/services/skill-tools/src/docs_alignment/report_sections.rs
verified_commit: {agentbox: 1639f86abded1441ce148d6c47924dfaf34f96af, visionflow: df22182f365f7bc7b4664e4374d150ff893e6b05}
---

## AB-28.1 agentbox-manifest — the boot-time projection surface

```mermaid
flowchart TB
    subgraph cli["agentbox-manifest — one clap binary, Boot-time TOML/JSON projection for agentbox<br/>(agentbox-manifest/src/main.rs:45-49)"]
        direction TB
        subgraph mcpg["MCP projection — src/mcp.rs"]
            C1["mcp-set-server --file --name<br/>agentbox-manifest/src/main.rs:78-85"]
            C2["mcp-reconcile-aqe --file --provider<br/>agentbox-manifest/src/main.rs:86-93"]
            C3["mcp-protect-namespace --file --server --namespace<br/>agentbox-manifest/src/main.rs:94-102"]
            C4["mcp-deregister-fork --file<br/>agentbox-manifest/src/main.rs:103-107"]
            C18["mcp-hub-project --file --state --out --hub-url --bind<br/>agentbox-manifest/src/main.rs:57-77, lifts stdio servers<br/>behind the loopback hub, --disable restores them"]
        end
        subgraph plug["Plugins — src/plugins.rs"]
            C5["plugin-register --file --key --install-path --message<br/>agentbox-manifest/src/main.rs:108-123"]
            C6["plugin-list --manifest<br/>agentbox-manifest/src/main.rs:124-128"]
        end
        subgraph proj["Config projection"]
            C7["nip98-config --manifest --out<br/>agentbox-manifest/src/main.rs:129-135 · src/proxy.rs"]
            C8["model-routing-project --manifest --workspace --dry-run<br/>agentbox-manifest/src/main.rs:136-144 · src/routing.rs"]
            C9["provision-stacks<br/>agentbox-manifest/src/main.rs:145-146 · src/stacks.rs"]
            C19["sso-project --manifest --format<br/>agentbox-manifest/src/main.rs:155-164, ADR-2094. Disabled<br/>prints NOTHING and exits 0, so the cloud path is byte-identical"]
        end
        subgraph tui["TUI round-trip"]
            C10["tui-read config state<br/>agentbox-manifest/src/main.rs:147-148 · src/tui_read.rs"]
            C11["tui-write state output existing<br/>agentbox-manifest/src/main.rs:183-188 · src/tui_write.rs"]
            C12["state-get file key<br/>agentbox-manifest/src/main.rs:181-182"]
            C13["state-set file key value<br/>agentbox-manifest/src/main.rs:183-188"]
            C14["state-set-bool file key value<br/>agentbox-manifest/src/main.rs:189-194"]
        end
        subgraph read["Manifest readers — src/tomlval.rs"]
            C15["toml-bool --manifest --path<br/>agentbox-manifest/src/main.rs:165-171 · prints 1 or 0, ALWAYS exits 0"]
            C16["toml-string --manifest --path<br/>agentbox-manifest/src/main.rs:172-178 · prints a string or empty, ALWAYS exits 0"]
            C17["embedding-dim<br/>agentbox-manifest/src/main.rs:179-180 · reads an OpenAI-shaped response on stdin"]
        end
    end
    EP["config/entrypoint-unified.sh"] --> cli
    subgraph notes["Invariants and drift"]
        direction TB
        N1["INVARIANT: this ONE binary owns every manifest read and config projection at boot. It<br/>replaced about 377 lines of inline python3 in the entrypoint plus four scripts, so<br/>PYTHON3 IS NO LONGER A BOOT DEPENDENCY — python3 stays in the image only for the<br/>supervised Python services opf-router and code-interpreter"]
        N2["SECURITY: mcp-set-server reads the spec JSON from STDIN specifically so bearer tokens<br/>and passwords never appear in the process list (agentbox-manifest/src/main.rs:78-79)"]
        N3["toml-bool and toml-string ALWAYS EXIT 0 — a missing key is an empty answer, not a boot<br/>failure, so the entrypoint can read an absent gate without set -e killing the boot"]
        N4["mcp-protect-namespace is APPEND-ONLY on the governed server's protected list<br/>(agentbox-manifest/src/main.rs:94) — see AB-20.2 for the protected-namespace write guard it feeds"]
        N1 ~~~ N2 ~~~ N3 ~~~ N4
    end
```

## AB-28.2 Boot projection sequence

```mermaid
sequenceDiagram
    autonumber
    participant EP as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh
    participant BIN as agentbox-manifest<br/>agentbox/services/agentbox-manifest/src/main.rs:217
    participant TOML as /etc/agentbox.toml
    participant MCPJ as .mcp.json
    participant PROXY as nip98-proxy config
    participant AQE as .agentic-qe/llm-config.json
    participant PROF as WORKSPACE/profiles

    Note over BIN: main() calls restore_default_sigpipe() FIRST (agentbox-manifest/src/main.rs:217-218)
    Note over BIN: Rust installs SIG_IGN for SIGPIPE at startup, which turns a closed downstream pipe into<br/>a PANIC-WITH-BACKTRACE on the next println!. The entrypoint pipes this binary into sed<br/>and consumes plugin-list through command substitution, and a backtrace in the boot log<br/>would be both alarming and useless (agentbox-manifest/src/main.rs:197-203)
    EP->>BIN: toml-bool --manifest /etc/agentbox.toml --path <dotted.gate>
    BIN->>TOML: read
    BIN-->>EP: "1" or "0", exit 0 always
    loop each enabled MCP server
        EP->>BIN: mcp-set-server --file .mcp.json --name <server> with the spec on STDIN
        BIN->>MCPJ: upsert
    end
    EP->>BIN: mcp-deregister-fork --file .mcp.json
    BIN->>MCPJ: de-register any ruvector-mcp OUTSIDE /opt/agentbox (ADR-036 D2, agentbox-manifest/src/main.rs:103)
    Note over BIN,MCPJ: this is what keeps a stray forked memory server from shadowing the governed one — see<br/>AB-20
    EP->>BIN: mcp-reconcile-aqe --file .mcp.json --provider <p>
    Note over BIN: an EMPTY or omitted provider REMOVES AQE_LLM_PROVIDER rather than blanking it<br/>(agentbox-manifest/src/main.rs:90)
    EP->>BIN: mcp-protect-namespace --file .mcp.json --server claude-flow --namespace <ns>
    EP->>BIN: nip98-config --manifest --out
    BIN->>PROXY: project [interaction_plane.proxy] (ADR-069, agentbox-manifest/src/main.rs:129) — see AB-10
    EP->>BIN: model-routing-project --manifest --workspace --dry-run?
    BIN->>AQE: project [model_routing] into EVERY .agentic-qe/llm-config.json (ADR-041, agentbox-manifest/src/main.rs:136)
    EP->>BIN: provision-stacks
    BIN->>PROF: provision the per-stack profile tree under WORKSPACE/profiles
    EP->>BIN: plugin-list --manifest
    BIN-->>EP: name<TAB>source for enabled, VALIDATED [[plugins.packages]] (agentbox-manifest/src/main.rs:124)
    loop each plugin to install
        EP->>BIN: plugin-register --file installed_plugins.json --key --install-path --message
        Note over BIN: --message is printed ONLY when the plugin was actually added (agentbox-manifest/src/main.rs:116). --now<br/>freezes the installedAt/lastUpdated stamp and is TEST-ONLY and hidden — without it the<br/>value is the wall clock, which no golden could pin (agentbox-manifest/src/main.rs:119-121)
    end
    Note over EP,BIN: every failure path prints to stderr and returns ExitCode::FAILURE (agentbox-manifest/src/main.rs:221-224)
```

## AB-28.3 Consultant model projection (ADR-2031)

```mermaid
sequenceDiagram
    autonumber
    participant EP as entrypoint-unified.sh
    participant BIN as agentbox-manifest toml-string<br/>agentbox/services/agentbox-manifest/src/tomlval.rs
    participant TOML as agentbox.toml [consultants.*]<br/>agentbox/agentbox.toml:1207
    participant ENV as consultant environment
    participant TUI as TUI save path

    EP->>BIN: toml-string --manifest --path consultants.<name>.model
    BIN->>TOML: read [consultants.<name>].model
    alt a NON-EMPTY pre-boot environment override exists
        ENV-->>ENV: the override WINS — the manifest does not clobber an operator's deliberate choice
    else no override
        BIN-->>ENV: project the manifest value into the consultant's environment at boot
    end
    TUI->>TOML: an operator saves the TUI
    Note over TUI,TOML: INVARIANT ADR-2031: a TUI save NEVER resets an operator's model
    Note over TOML: consultant sections: agentbox.toml:1212 codex, agentbox.toml:1218 antigravity,<br/>agentbox.toml:1224 zai, agentbox.toml:1231 perplexity, agentbox.toml:1236 deepseek
    Note over TOML: INVARIANT ADR-2031: cost figures are DATED API-EQUIVALENT ESTIMATES or null, NEVER a<br/>stale constant
    Note over BIN: INVARIANT: consultant models come FROM THE MANIFEST — agentbox-manifest toml-string is<br/>the single projection path
```

## AB-28.4 TUI manifest round-trip

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator TUI
    participant RD as tui-read<br/>agentbox/services/agentbox-manifest/src/tui_read.rs
    participant STATE as flat TUI state JSON
    participant GET as state-get / state-set / state-set-bool<br/>agentbox/services/agentbox-manifest/src/main.rs:181-194
    participant WR as tui-write<br/>agentbox/services/agentbox-manifest/src/tui_write.rs
    participant TOML as canonical agentbox.toml

    OP->>RD: tui-read <config> <state>
    RD->>TOML: parse agentbox.toml
    RD->>STATE: emit the FLAT TUI state document
    loop operator edits
        OP->>GET: state-get <file> <key>
        GET-->>OP: one key
        OP->>GET: state-set <file> <key> <value> or state-set-bool
        GET->>STATE: write one key
    end
    OP->>WR: tui-write <state> <output> [existing]
    WR->>STATE: read the flat state
    opt existing supplied
        WR->>TOML: preserve the existing document's shape and comments
    end
    WR->>TOML: emit CANONICAL agentbox.toml
    Note over RD,WR: the round-trip is pinned by golden tests —<br/>agentbox/services/agentbox-manifest/tests/golden.rs, golden_entrypoint.rs and<br/>consultant_model.rs
    Note over WR: a TUI save must not reset an operator's consultant model (ADR-2031) — see AB-28.3
```

## AB-28.5 agentbox-ops — the retired-Python tool suite

```mermaid
flowchart TB
    subgraph lib["agentbox-ops shared modules — agentbox/services/agentbox-ops/src/lib.rs:7-15"]
        M1["cost_cap — spend ceilings + ledger"]
        M2["distil — expel lesson extraction"]
        M3["hermes — scheduling"]
        M4["process_identity — argv daemon identification (ADR-2032)"]
        M5["procs — process enumeration"]
        M6["pyjson — Python-shaped JSON compatibility"]
        M7["solar — PV geometry and yield"]
        M8["token_audit — token accounting"]
        M9["voyager — skill-library gate"]
    end
    subgraph bins["Binaries — agentbox/services/agentbox-ops/src/bin/"]
        B1["comfyui-generate — see AB-27"]
        B2["expel-distil"]
        B3["hermes-scheduler"]
        B4["mcp-call"]
        B5["pvgis-fetch"]
        B6["report-preflight"]
        B7["ruflo-daemon-gc"]
        B8["solar-optimize"]
        B9["token-audit"]
        B10["tree-search-cap — the spend cap in AB-22"]
        B11["voyager-gate"]
        B12["yt-transcript-archive"]
    end
    M1 --> B10
    M1 --> B9
    M2 --> B2
    M3 --> B3
    M4 --> B7
    M5 --> B7
    M7 --> B8
    M7 --> B5
    M8 --> B9
    M9 --> B11
    subgraph notes["Invariants and drift"]
        direction TB
        N1["Each binary REPLACES a Python script retired by the 2026-09-02 estate legacy audit. The<br/>modules hold the behaviour worth unit-testing independently of the CLI shell around it<br/>(agentbox-ops/src/lib.rs:1-5)"]
        N2["ADR-2032 daemon identification argv boundaries — process_identity.rs plus<br/>process_identity_tests.rs. ruflo-daemon-gc must identify its targets by ARGV shape,<br/>never by a name substring that could match an unrelated process"]
        N3["agentbox/services/agentbox-ops/tests/reaper_default_is_read_only.rs pins the safety<br/>default: the reaper is READ-ONLY unless explicitly told otherwise"]
        N4["ADR-2030 permissive licensing for publishable service crates — these crates carry<br/>LICENSE-APACHE and LICENSE-MIT, see agentbox/services/LICENSING-NOTICE.md"]
        N1 ~~~ N2 ~~~ N3 ~~~ N4
    end
```

## AB-28.6 agentbox-mcp — one binary, three stdio MCP servers plus the streamable-HTTP hub

```mermaid
sequenceDiagram
    autonumber
    participant SUP as supervisord<br/>agentbox/flake.nix:2252
    participant BIN as agentbox-mcp<br/>agentbox/services/agentbox-mcp/src/main.rs:65
    participant LOG as tracing_subscriber
    participant T as rmcp stdio transport
    participant SRV as the selected server
    participant HOST as MCP host
    participant HUB as serve<br/>agentbox/services/agentbox-mcp/src/hub/mod.rs:292

    SUP->>BIN: agentbox-mcp <subcommand>
    Note over BIN: "Unified agentbox MCP server (imagemagick, web-summary, gemini-url-context, hub)"<br/>(agentbox-mcp/src/main.rs:28)
    BIN->>LOG: fmt().with_writer(std::io::stderr)
    Note over BIN,LOG: INVARIANT: logging MUST go to STDERR — stdout is the JSON-RPC stdio transport channel<br/>and ANY STRAY BYTE ON IT CORRUPTS THE PROTOCOL STREAM (agentbox-mcp/src/main.rs:66-69)
    Note over LOG: EnvFilter from the environment, defaulting to "info" (agentbox-mcp/src/main.rs:70-72)
    BIN->>BIN: let cli = Cli::parse() (agentbox-mcp/src/main.rs:75)
    alt Imagemagick (agentbox-mcp/src/main.rs:38)
        BIN->>T: rmcp::transport::stdio()
        BIN->>SRV: ImageMagickServer::new().serve(transport) (agentbox-mcp/src/main.rs:78-82)
        Note over SRV: image processing with format conversion, resizing, cropping and batch operations.<br/>Modules args.rs / exec.rs / types.rs — see AB-27
    else web-summary (agentbox-mcp/src/main.rs:41)
        BIN->>T: rmcp::transport::stdio()
        BIN->>SRV: WebSummary server
        Note over SRV: URL summarization with YouTube transcripts and topic generation. Modules fetch.rs /<br/>youtube.rs / llm.rs / types.rs
    else gemini-url-context (agentbox-mcp/src/main.rs:44)
        BIN->>T: rmcp::transport::stdio()
        BIN->>SRV: GeminiUrlContext server
        Note over SRV: URL expansion and analysis using Gemini's URL Context API. Modules api.rs / types.rs
    else hub {config, bind, wait_config_secs} (agentbox-mcp/src/main.rs:46-61)
        BIN->>HUB: hub::serve(&config, bind, wait_config_secs) (agentbox-mcp/src/main.rs:101-105)
        Note over HUB: ADR-2034 — shared streamable-HTTP front for stateless stdio MCP servers, loopback<br/>only, config written at boot by agentbox-manifest mcp-hub-project — see AB-09
        HUB-->>HOST: HTTP, not stdio — this branch never reaches SRV/T below
    end
    opt not the hub branch
        SRV->>HOST: JSON-RPC over stdio
        SRV->>SRV: service.waiting().await
    end
```

## AB-28.7 skill-tools — Rust ports backing three skills

```mermaid
flowchart LR
    subgraph mods["agentbox/services/skill-tools/src/lib.rs:12-14"]
        U["uiux — BM25 search + design-system generation<br/>backs the ui-ux-pro-max-skill skill"]
        W["wardley — map generation, heuristics, interactive D3 rendering, strategic analysis<br/>backs the wardley-maps skill"]
        D["docs_alignment — link / mermaid / ASCII validation and reporting<br/>backs the docs-alignment skill"]
    end
    subgraph ub["uiux binaries"]
        UB1["uiux_search"]
    end
    subgraph wb["wardley binaries"]
        WB1["wardley_generate"]
        WB2["wardley_mapper"]
        WB3["wardley_quick_map"]
        WB4["wardley_heuristics"]
        WB5["wardley_interactive"]
        WB6["wardley_strategic_analyzer"]
    end
    subgraph db["docs_alignment binaries"]
        DB1["docs_alignment_bin"]
        DB2["docs_check_mermaid"]
        DB3["docs_validate_links"]
        DB4["docs_detect_ascii"]
        DB5["docs_generate_report"]
    end
    U --> ub
    W --> wb
    D --> db
    subgraph dmods["docs_alignment modules"]
        DM1["links.rs · links_external.rs"]
        DM2["mermaid.rs"]
        DM3["ascii_diagrams.rs"]
        DM4["orchestrator.rs · cli.rs"]
        DM5["report.rs · report_sections.rs · models.rs"]
    end
    D --> dmods
    subgraph notes["Invariants and drift"]
        direction TB
        N1["Each module is SELF-CONTAINED and backs one or more [[bin]] targets declared in<br/>Cargo.toml (skill-tools/src/lib.rs:9-10)"]
        N2["docs_check_mermaid is the skill-side mermaid validator. This diagrams tree is validated<br/>instead by VisionFlow's scripts/diagram-index-gen.cjs (the tree moved to the estate canon on 2026-09-07) --render, which renders every block through mmdc<br/>— a different and stricter gate"]
        N3["skill invocation and the manifest gates that enable these skills are AB-22"]
        N1 ~~~ N2 ~~~ N3
    end
```

## AB-28.8 The MCP server fleet in mcp/servers

```mermaid
flowchart LR
    subgraph node["agentbox/mcp/servers — Node MCP servers"]
        S1["ruvector-mcp.cjs — governed memory, fails closed<br/>see AB-20"]
        S2["ontology-bridge.js · ontology-propose.js · ontology-local.cjs · ontology-workingset.cjs<br/>see AB-25"]
        S3["harness-bridge.js — harness_list/inspect/validate/audit<br/>see AB-22"]
        S4["RETIRED: precedent-bridge.js is DELETED, superseded by the colloquy crates,<br/>see AB-28.11"]
        S5["continual-harness.cjs — evidence-anchored signed refines<br/>see AB-22"]
        S6["substrate-tools.js — refine_* · ws_* · spawn_child/ready/complete<br/>see AB-26"]
        S7["governance-bridge.js"]
        S8["decision-tools.js"]
        S9["nostr-bridge.js — see AB-13"]
        S10["mcp-ws-relay.js"]
    end
    subgraph rust["Rust MCP binaries"]
        R1["agentbox-mcp imagemagick, web-summary, gemini-url-context<br/>see AB-28.6"]
        R2["colloquy-mcp: query, propose, confirm, flag, reflect, status<br/>agentbox/crates/colloquy/colloquy-mcp/src/server.rs:29<br/>gate skills.colloquy agentbox.toml:906, tier agentbox.toml:913"]
    end
    subgraph proj["Projection — see AB-09 and AB-22.10"]
        P1["agentbox/mcp/mcp.json — the fleet declaration"]
        P2["agentbox/skills/mcp.json — the skills-side projection"]
        P3["agentbox-manifest mcp-set-server / mcp-reconcile-aqe / mcp-deregister-fork<br/>agentbox/services/agentbox-manifest/src/mcp.rs"]
        P4["workspace .mcp.json — what the harness actually loads"]
    end
    P1 --> P3
    P2 --> P3
    P3 --> P4
    P4 --> node
    P4 --> rust
    subgraph notes["Invariants and drift"]
        direction TB
        N1["Every server reads its gates from agentbox.toml through the entrypoint's env projection<br/>— no server parses the manifest itself, so one gate edit reaches every surface"]
        N2["INVARIANT: a gated-off server is NOT REGISTERED AT ALL rather than<br/>registered-and-disabled — byte-identical-when-off"]
        N3["RESOLVED since AB-22.13 was written: the [skills.harness] block now reads the<br/>manifest gate, agentbox-manifest toml-bool --path skills.harness.enabled,<br/>config/entrypoint-unified.sh:1861. Its [skills.precedent] twin was removed from<br/>agentbox.toml with the precedent bridge, so no stanza is left unguarded"]
        N4["INVARIANT: the colloquy binary is symlinked as /opt/agentbox/bin/colloquy-mcp<br/>(agentbox/flake.nix:1711) and NEVER as a /nix/store path, because a store path is<br/>content-addressed and garbage-collected, which is how the registration came to fail<br/>ENOENT against a path no longer on disk (flake.nix:1707-1708)"]
        N1 ~~~ N2 ~~~ N3 ~~~ N4
    end
```

## AB-28.9 Service crate publishing posture

```mermaid
classDiagram
    class PublishableServiceCrate {
        <<ADR-2030>>
        +LICENSE-APACHE
        +LICENSE-MIT
        +README.md
        +Cargo.toml with repository metadata
        +inline rustdoc on every public item
    }
    class agentbox_manifest {
        boot-time TOML/JSON projection
        clap subcommands
        golden tests
    }
    class agentbox_ops {
        retired-Python tool suite
        12 binaries
        9 shared modules
    }
    class agentbox_mcp {
        3 stdio MCP servers
        rmcp transport
    }
    class skill_tools {
        uiux · wardley · docs_alignment
        12 binaries
    }
    class ontology_tools {
        parser · validator · writer · modifier
        enrichment · link_validator
        see AB-25
    }
    class podcast_ingest {
        corpus ingest
        see AB-27
    }
    class explainer_tools {
        explainer-loom-draft
        LICENSE-APACHE and LICENSE-MIT
        see AB-28.12
    }
    class dream_engine {
        nightly cycle
        see AB-23
    }
    class headroom_napi {
        NAPI compression addon
        see AB-26
    }
    PublishableServiceCrate <|-- agentbox_manifest
    PublishableServiceCrate <|-- agentbox_ops
    PublishableServiceCrate <|-- agentbox_mcp
    PublishableServiceCrate <|-- skill_tools
    PublishableServiceCrate <|-- ontology_tools
    PublishableServiceCrate <|-- podcast_ingest
    PublishableServiceCrate <|-- explainer_tools
    PublishableServiceCrate <|-- dream_engine
    PublishableServiceCrate <|-- headroom_napi
    note for PublishableServiceCrate "ADR-2030 permissive licensing for publishable service crates. Dual LICENSE-APACHE +<br/>LICENSE-MIT are present in agentbox-manifest, agentbox-ops, ontology-tools and<br/>podcast-ingest. The estate-wide notice is agentbox/services/LICENSING-NOTICE.md"
    note for agentbox_manifest "Nix packaging for each crate lives in agentbox/lib — agentbox-manifest.nix,<br/>agentbox-ops.nix, agentbox-mcp.nix, skill-tools.nix, ontology-tools.nix,<br/>podcast-ingest.nix, dream-engine.nix, headroom-compress.nix. A crate is in the package<br/>set only when its manifest gate is on, which is what makes byte-identical-when-off real<br/>at IMAGE level, not just runtime"
```

## AB-28.10 A new gate joins the TUI round-trip — [model_routing.neural] (ADR-2080)

```mermaid
sequenceDiagram
    autonumber
    participant TOML as agentbox.toml<br/>[model_routing.neural].enabled
    participant RD as tui-read FIELDS<br/>agentbox/services/agentbox-manifest/src/tui_read.rs:127
    participant STATE as flat TUI state JSON
    participant SEC as tui_sections::render<br/>agentbox/services/agentbox-manifest/src/tui_sections.rs:16
    participant WR as tui-write<br/>agentbox/services/agentbox-manifest/src/tui_write.rs

    Note over RD: FIELDS gains one entry — F#40;"model_routing.neural.enabled", D::B#40;false#41;, false#41;<br/>tui_read.rs:127 — default OFF, not a secret field
    TOML->>RD: tui-read parses [model_routing.neural].enabled into the flat state
    RD->>STATE: state key model_routing.neural.enabled = true#124;false
    Note over STATE: only enabled round-trips through the flat state — provider, quality_bar,<br/>cost_ceiling_usd_per_mtok, privacy_tier, trajectory and assets_dir are NOT TUI fields —<br/>see AB-29 for the manifest keys that stay operator-edited-toml-only
    STATE->>SEC: tui-write regenerates [model_routing.neural]
    SEC->>WR: emits the block verbatim — enabled from state#40;s#41;, every OTHER key a FIXED literal<br/>#40;provider "openrouter", quality_bar 0.50, cost_ceiling_usd_per_mtok 0,<br/>privacy_tier "public", trajectory true, assets_dir "/opt/agentbox/model-router" —<br/>tui_sections.rs:193-200#41;
    Note over SEC: DOC-DRIFT: a TUI save therefore RESETS provider/quality_bar/cost_ceiling/trajectory/<br/>assets_dir to these literals even if the operator hand-edited agentbox.toml to different<br/>values — unlike [consultants.*].model#40;ADR-2031, AB-28.3#41; there is no read-back-and-preserve<br/>path for the non-enabled model-router keys yet
    WR->>TOML: canonical agentbox.toml, [model_routing.neural] block rewritten
    Note over TOML: privacy_tier stays the literal "public" on every TUI-driven write — the TUI cannot<br/>accidentally widen the console's egress scope #40;see AB-29.5#41;
```

## AB-28.11 The precedent bridge is retired, colloquy is the successor (ADR-2085/2086)

```mermaid
flowchart TB
    subgraph gone["DELETED at 70d017a3b - nothing to migrate"]
        G1["mcp/servers/precedent-bridge.js - precedent_list, match, promote, retire"]
        G2["management-api/lib/precedent-service.js"]
        G3["the governance-precedents namespace was EMPTY and nothing called the tools,<br/>so the retirement needed no revertible migration"]
        G1 --> G3
        G2 --> G3
    end
    subgraph crates["crates/colloquy - clean-room in Rust, six crates"]
        P1["colloquy-core - the standard, pure and wasm-capable, PUBLISHED"]
        P2["colloquy-view - presentation models, depends on core alone"]
        P3["colloquy-nostr - kinds 38100 to 38105, owns the NIP-01 structs"]
        P4["colloquy-store - local, shared and relay behind ONE trait"]
        P5["colloquy-backends - INTERNAL: the governed ruvector-mcp child and the relay socket"]
        P6["colloquy-mcp - INTERNAL binary, the six verbs over stdio<br/>agentbox/crates/colloquy/colloquy-mcp/src/server.rs:29"]
    end
    G3 --> P6
    subgraph verbs["The six verbs, tier chosen by COLLOQUY_TIER"]
        V1["query colloquy-mcp/src/server.rs:29"]
        V2["propose colloquy-mcp/src/server.rs:43"]
        V3["confirm colloquy-mcp/src/server.rs:59"]
        V4["flag colloquy-mcp/src/server.rs:71"]
        V5["reflect colloquy-mcp/src/server.rs:83"]
        V6["status colloquy-mcp/src/server.rs:107"]
    end
    P6 --> verbs
    subgraph gates["Manifest gates - agentbox.toml [skills.colloquy]"]
        M1["enabled agentbox.toml:906 - gates the MCP REGISTRATION only,<br/>the binary is always baked, exactly as the bridge it supersedes was"]
        M2["tier agentbox.toml:913 - shared by default: the agents in one container<br/>are one operator's, and a private tier-1 store would lose every learning<br/>at the end of the session"]
        M3["namespace agentbox.toml:919 - its OWN namespace, never patterns"]
        M4["principal agentbox.toml:926 - empty derives it from the Nostr operator"]
        M5["reflect_candidates agentbox.toml:933 - a Stop hook registered at<br/>config/entrypoint-unified.sh:1425 that writes CANDIDATES, never units"]
        M1 ~~~ M2 ~~~ M3 ~~~ M4 ~~~ M5
    end
    P6 --> gates
```

**Debt (corpus vs repo):** this topic listed the deleted `mcp/servers/precedent-bridge.js` in its own `sources:` until this pass, so every citation into it was an unresolvable warning and the file-existence check was failing the whole tree; the successor is `../project/agentbox/crates/colloquy/colloquy-mcp/src/server.rs:29`.

**Invariant:** mixing colloquy units into the `patterns` namespace would move the frozen recall band, so the shared tier gets its own (`../project/agentbox/agentbox.toml:919`).

**Invariant:** an agent that authorises itself defeats principal collapse, so the principal must never equal the agent's own member id and the binary exits rather than start that way (`../project/agentbox/agentbox.toml:923-925`).

## AB-28.12 ADR-2104 - the hub fails hard instead of lying

```mermaid
stateDiagram-v2
    [*] --> Launched
    Launched --> Waiting : serve calls wait_for_config first
    note right of Launched
        [program:agentbox-mcp-hub] starts at supervisor priority 205
        (flake.nix:2548), long before the bootstrap program reaches the
        block that writes the hub config. The command carries
        --wait-config-secs 120 (flake.nix:2540).
    end note
    Waiting --> Serving : the projection appears
    Waiting --> FailedLoudly : 120s elapse with no file
    note right of Waiting
        A 15-second heartbeat logs how long it has waited
        (hub/mod.rs:283-286), polling every 500ms.
    end note
    note right of FailedLoudly
        The error NAMES the projection that did not run
        (agentbox-manifest mcp-hub-project) and the gate
        ([resources.mcp_hub]), then bails: a hub with no
        config serves nothing (hub/mod.rs:265-281).
        startsecs=130 with autorestart=unexpected parks it
        FATAL instead of restarting forever (flake.nix:2545-2546).
    end note
    FailedLoudly --> [*]
    Serving --> Refused : the bind is not loopback
    note right of Serving
        HubConfig::load then the loopback assertion
        (hub/mod.rs:298-300).
    end note
    Refused --> [*]
    Serving --> [*]
```

**Invariant:** the hub's wait is bounded and its failure is loud, because the unbounded version read `RUNNING` for three days over a port it had never bound while nine hub-routed servers refused connections (`../project/agentbox/services/agentbox-mcp/src/hub/mod.rs:261-264`, `../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md:23-32`).

**Open:** ADR-2104 is recorded `decision_status: proposed` with `implementation_status: partial` (`../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md:5-7`), so the wider rule that an MCP server is a disposable adapter over a crate (`:36-40`) is not yet a compliance surface for the servers in AB-28.8.

## AB-28.13 ADR-2084 - the facade client the service crates share

```mermaid
flowchart TB
    CRATE["loom-client, published from the loom repository under MIT OR Apache-2.0<br/>ADR-2084-one-published-loom-client-for-every-facade-caller.md:37-39"]
    CRATE --> A["dream-engine - see AB-23.15"]
    CRATE --> B["podcast-ingest - extraction and promotion"]
    CRATE --> C["explainer-tools explainer-loom-draft<br/>agentbox/services/explainer-tools/src/bin/loom_draft.rs:22"]
    CRATE --> D["agentbox-mcp web-summary<br/>agentbox/services/agentbox-mcp/src/web_summary/llm.rs:65"]
    C --> C1["declines the scaffold per request, ADR-139 - a codebase is NOT in the<br/>ontology, and the client FAILS the call if the facade grounds it anyway<br/>rather than answering about the wrong subject<br/>loom_draft.rs:12-15, loom_draft.rs:196"]
    D --> D1["floors max_tokens at 1536 - a reasoning model truncated below that<br/>returns EMPTY content, not a short answer<br/>web_summary/llm.rs:20, web_summary/llm.rs:63"]
    subgraph lic["ADR-2030 licensing, closed for this crate at e57156a8f"]
        direction TB
        L1["explainer-tools declares MIT OR Apache-2.0 in its manifest and now<br/>CARRIES both texts plus a README - the crate-licensing gate had been<br/>failing at HEAD with the declaration and no files"]
        L2["it joins agentbox-manifest, agentbox-ops, ontology-tools, podcast-ingest<br/>and dream-engine under agentbox/services/LICENSING-NOTICE.md - see AB-28.9"]
        L1 ~~~ L2
    end
    C -.-> lic
```

**Invariant:** `explainer-loom-draft` replaced `skills/explainer/scripts/loom-draft.mjs`, so the explainer path is a supervised Rust binary rather than a script, and the facade protocol it speaks comes from the published crate rather than being reimplemented (`../project/agentbox/services/explainer-tools/src/bin/loom_draft.rs:1-2`).

