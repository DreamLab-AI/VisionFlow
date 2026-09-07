---
id: AB-08
title: Claude Code hook pipeline and its handlers
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2015, ADR-2026, ADR-2007]
sources:
  - ../project/agentbox/config/hooks/claude-flow-hook-adapter.cjs
  - ../project/agentbox/config/hooks/trust-seed.cjs
  - ../project/agentbox/config/hooks/nostr-live-mirror.cjs
  - ../project/agentbox/config/hooks/project-tracking-publish.cjs
  - ../project/agentbox/config/hooks/ontology-monitor.cjs
  - ../project/agentbox/config/hooks/ruvnet-brain-ground.cjs
  - ../project/agentbox/config/hooks/trajectory-recorder.cjs
  - ../project/agentbox/config/hooks/dream-inbox-surface.cjs
  - ../project/agentbox/config/hooks/fleet-session-start.sh
  - ../project/agentbox/config/hooks/fleet-tab-name.sh
  - ../project/agentbox/config/hooks/lib/trajectory-util.cjs
  - ../project/agentbox/config/hooks/lib/egress-policy.cjs
  - ../project/agentbox/config/hooks/README.md
  - ../project/agentbox/services/agentbox-manifest/src/stacks.rs
  - ../project/agentbox/services/agentbox-manifest/src/stacks_env.rs
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/mcp/servers/lib/ontology-local.js
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
  - ../project/agentbox/config/tab0-bridge/deploy.sh
  - ../project/agentbox/config/tab0-bridge/turn-sink.cjs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/mcp/servers/lib/ontology-push.js
  - ../project/agentbox/scripts/dream-inbox.mjs
verified_commit: 2c521c5bb
---

Registration ground truth (2026-09-07): `~/.claude/settings.json` is boot-generated and never tracked, so AB-08.1–AB-08.7
cite the sites that WRITE it — `config/entrypoint-unified.sh` for the root session and
`services/agentbox-manifest/src/stacks.rs` `learning_hooks#40;#41;` for per-profile stacks — the split `config/hooks/README.md` owns.

## AB-08.1 Root-session registration — what entrypoint-unified.sh seeds into ~/.claude/settings.json
```mermaid
flowchart TB
    S["settings path resolved once, CLAUDE_CONFIG_DIR<br/>fallback /home/devuser/.claude/settings.json<br/>entrypoint-unified.sh:1172"]
    subgraph MIR["nostr-live-mirror.cjs — always registered"]
    direction TB
    M1["block entrypoint-unified.sh:1167, baked /opt hook path :1173"]
    M2["entrypoint-unified.sh:1182 events SessionStart, UserPromptSubmit, Stop, SessionEnd<br/>idempotent marker test on the command string :1184"]
    M3["entrypoint-unified.sh:1186 push node HOOK EVENT, timeout 8000<br/>runtime off switch AGENTBOX_LIVE_MIRROR=0, see AB-08.9"]
    M1 --> M2 --> M3
    end
    subgraph FLT["fleet-session-start.sh — SessionStart"]
    direction TB
    F1["fleet registration entrypoint-unified.sh:1202<br/>off switch AGENTBOX_NOSTR_GATEWAY=0 in fleet-session-start.sh"]
    F2["entrypoint-unified.sh:1205 marker test, :1206 push timeout 8000"]
    F1 --> F2
    end
    subgraph ONT["ontology-monitor.cjs — SessionEnd, gated"]
    direction TB
    O1["entrypoint-unified.sh:1220 gate defaults to 0<br/>read from manifest path ontology_monitor.enabled :1224"]
    O2["entrypoint-unified.sh:1245 ON: push timeout 200000<br/>AND seed env AGENTBOX_ONTOLOGY_MONITOR=1 :1250"]
    O3["entrypoint-unified.sh:1254 OFF: filter the hook back out<br/>delete the emptied SessionEnd array :1258<br/>INVARIANT ADR-2020 byte-identical-when-off :1219"]
    O4["gate source agentbox.toml:103-104 enabled = true"]
    O1 --> O2
    O1 --> O3
    O1 --> O4
    end
    subgraph TRU["trust-seed.cjs — SessionStart plus one direct run"]
    direction TB
    T1["block entrypoint-unified.sh:1276, hook path :1284<br/>gate AGENTBOX_TRUST_SEED, direct run first :1285"]
    T2["entrypoint-unified.sh:1293 marker test<br/>:1294 push timeout 8000 with continueOnError true"]
    T1 --> T2
    end
    subgraph TRJ["trajectory-recorder.cjs — Stop and SubagentStop, gated"]
    direction TB
    J1["block entrypoint-unified.sh:1377, hook path :1390<br/>gate: BOTH memory_learning flags :1391"]
    J2["entrypoint-unified.sh:1412 specs = Stop, SubagentStop ONLY<br/>reconcile strips prior wiring over 4 legacy events :1413 and :1415<br/>then push timeout 10000 behind an inline env prefix :1421"]
    J3["entrypoint-unified.sh:1439 gate off: strip over the same 4 events<br/>:1442-1443 keep-filter, :1445 log the de-registration"]
    J4["gate source agentbox.toml:413-414 enabled + record_trajectories"]
    J1 --> J2
    J1 --> J3
    J1 --> J4
    end
    subgraph OTH["turn-sink, dream-inbox, ruvnet-brain"]
    direction TB
    X1["entrypoint-unified.sh:1339 tab0-bridge turn-sink.cjs deployed path<br/>:1347 events UserPromptSubmit and Stop, :1350 timeout 8000"]
    X2["entrypoint-unified.sh:1455 dream-inbox-surface.cjs, live-checkout fallback :1456<br/>gate DREAM_INBOX_HOOK :1457, UserPromptSubmit timeout 5000 :1471"]
    X3["entrypoint-unified.sh:1854 ruvnet-brain-ground.cjs<br/>gate RUVNET_BRAIN_GROUNDING_HOOK :1855, timeout 5000 :1868"]
    X4["gate source agentbox.toml:634 grounding_hook = true"]
    X1 --> X2 --> X3 --> X4
    end
    subgraph SHM["hook shim reconcile — ADR-2034 §1"]
    direction TB
    H1["entrypoint-unified.sh:1299 block, gate AGENTBOX_HOOK_SHIM :1309"]
    H2["entrypoint-unified.sh:1310 agentbox-hook reconcile --root WORKSPACE --depth 2<br/>rewrites per-project ruflo CLI hooks to the resident shim"]
    H1 --> H2
    end
    S --> MIR --> FLT --> ONT --> TRU --> TRJ --> OTH --> SHM
```

## AB-08.2 Per-profile registration — stacks.rs learning_hooks, and what is NOT a hook
```mermaid
flowchart TB
    subgraph PP["workspace/profiles/#60;stack#62;/.claude/settings.json"]
    direction TB
    P0["build_profile writes hooks: learning_hooks#40;env, gates#41;<br/>stacks.rs:215, fn at :28"]
    P1["every entry is node ADAPTER action #124;#124; true<br/>stacks.rs:32, adapter default stacks_env.rs:42-45"]
    P2["PreToolUse stacks.rs:54<br/>Bash → pre-command t=5000 :55<br/>Write#124;Edit#124;MultiEdit → pre-edit t=5000 :56"]
    P3["PostToolUse :58<br/>Write#124;Edit#124;MultiEdit → post-edit t=10000 :59<br/>Bash → post-command t=5000 :60"]
    P4["UserPromptSubmit → route t=12000 :62<br/>SessionStart → session-restore t=15000 :63"]
    P5["SessionEnd :64 = session-end t=10000 :37<br/>+ nostr summary hook when mobile_bridge :38-43<br/>+ ontology-monitor when ontology_monitor :45-50"]
    P0 --> P1 --> P2 --> P3 --> P4 --> P5
    end
    subgraph NOT["NOT hooks — hooks/README.md:40-46"]
    direction TB
    N1["project-tracking-publish.cjs — a CLI the management API spawns<br/>from POST /v1/projects/:id/publish, reads a digest on stdin<br/>hooks/README.md:44, see AB-08.12"]
    N2["fleet-tab-name.sh — shelled by fleet-session-start.sh:17<br/>hooks/README.md:45"]
    N3["lib/egress-policy.cjs, lib/trajectory-util.cjs — required libraries<br/>hooks/README.md:46, see AB-08.9 and AB-08.14"]
    N1 --> N2 --> N3
    end
    subgraph INV["invariants and divergences"]
    direction TB
    I1["INVARIANT: claude-flow-hook-adapter.cjs is wired ONLY per-profile<br/>stacks.rs:215 — never into the root settings.json<br/>hooks/README.md:27-31"]
    I2["INVARIANT ADR-2068: the root ontology gap is closed —<br/>entrypoint-unified.sh:1245 registers and :1254-1258 retracts,<br/>so the off state leaves no trace, hooks/README.md:25"]
    I3["INVARIANT: a new hook is not wired by dropping a file in config/hooks/ —<br/>register it in the site that owns its session class<br/>hooks/README.md:50-53"]
    I1 --> I2 --> I3
    end
    PP --> NOT --> INV
```

## AB-08.3 PreToolUse — what agentbox actually registers
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core<br/>profile stack
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:106
    participant CLI as claude-flow hooks CLI<br/>AGENTBOX_FLOW_BIN, default claude-flow<br/>claude-flow-hook-adapter.cjs:32
    participant EP as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1377

    Note over CC,AD: PreToolUse is a PER-PROFILE event only — stacks.rs:54 registers it into<br/>workspace/profiles/#60;stack#62;/.claude/settings.json, never the root file
    CC->>AD: stdin JSON, matcher=Bash, argv#91;2#93;=pre-command, t=5000<br/>stacks.rs:55
    AD->>AD: parsePayload of stdin, take tool_input.command<br/>claude-flow-hook-adapter.cjs:108 and :113
    AD->>CLI: hooks pre-command --command CMD, timeout 5000<br/>claude-flow-hook-adapter.cjs:126-128
    AD-->>CC: exit 0 unconditionally :144-149
    CC->>AD: stdin JSON, matcher=Write#124;Edit#124;MultiEdit, argv#91;2#93;=pre-edit, t=5000<br/>stacks.rs:56
    AD->>AD: take tool_input.file_path :112
    AD->>CLI: hooks pre-edit --file FILE, timeout 5000 :120-122
    AD-->>CC: exit 0
    Note over EP: INVARIANT: the ROOT session registers NOTHING on PreToolUse.<br/>The reconcile loop STRIPS any prior trajectory-recorder wiring from<br/>PreToolUse/PostToolUse #40;legacy volumes#41; entrypoint-unified.sh:1413-1418
    Note over EP: specs is Stop and SubagentStop only :1412 — the per-tool grading design<br/>was replaced by transcript-driven grading :1409-1411, see AB-08.13
```

## AB-08.4 PostToolUse — post-edit and post-command, and why the root session opted out
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core<br/>profile stack
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:106
    participant CLI as claude-flow hooks CLI<br/>claude-flow-hook-adapter.cjs:67-73
    participant EP as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1409

    CC->>AD: matcher=Write#124;Edit#124;MultiEdit, argv#91;2#93;=post-edit, t=10000<br/>stacks.rs:59
    AD->>CLI: hooks post-edit --file FILE, timeout 10000<br/>claude-flow-hook-adapter.cjs:123-125
    Note over AD,CLI: stdout is NOT forwarded for edit/command actions — only route and<br/>session-restore pass a signal allowlist :74-81
    CC->>AD: matcher=Bash, argv#91;2#93;=post-command, t=5000<br/>stacks.rs:60
    AD->>CLI: hooks post-command --command CMD, timeout 5000 :129-131
    AD-->>CC: exit 0 — try/catch wraps main#40;#41;, process.exit#40;0#41; is unconditional :144-149
    Note over EP: DOC-DRIFT closed: PostToolUse never fires for a FAILED Bash command, so per-tool<br/>grading missed every failure — entrypoint-unified.sh:1409-1411.<br/>Root grading moved to the transcript at Stop/SubagentStop :1412
    Note over EP: INVARIANT: the reconcile loop is unconditional over the 4 legacy events<br/>:1415-1418 on the ON path and :1439-1443 on the OFF path, so a volume<br/>carrying the old wiring is repaired on the next boot either way
```

## AB-08.5 UserPromptSubmit — four root hooks plus the per-profile route
```mermaid
sequenceDiagram
    autonumber
    participant U as User turn
    participant NM as nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:375
    participant RBG as ruvnet-brain-ground.cjs<br/>agentbox/config/hooks/ruvnet-brain-ground.cjs:37
    participant TS as turn-sink.cjs<br/>agentbox/config/tab0-bridge/turn-sink.cjs:1
    participant DI as dream-inbox-surface.cjs<br/>agentbox/config/hooks/dream-inbox-surface.cjs:24
    participant AD as claude-flow-hook-adapter.cjs route<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:117
    participant M as Model context

    Note over U,M: root registrations: entrypoint-unified.sh:1182 mirror, :1347 turn-sink,<br/>:1471 dream-inbox, :1868 brain-ground — all four on this one event
    U->>NM: node HOOK UserPromptSubmit, timeout 8000<br/>entrypoint-unified.sh:1186
    NM->>NM: bodyForEvent gives #129; #91;shortId#93; prompt text<br/>nostr-live-mirror.cjs:279-282
    NM-->>U: return 0 — see AB-08.9 for the egress gate and wrap
    par independent root groups, unordered wrt each other
        U->>RBG: node HOOK #124;#124; true, timeout 5000<br/>entrypoint-unified.sh:1868
        RBG-->>M: JSON #123;result:continue, additionalContext#125; on a RuvNet/classical-sub match<br/>ruvnet-brain-ground.cjs:71-75, see AB-08.11
    and
        U->>TS: node HOOK UserPromptSubmit #124;#124; true, timeout 8000<br/>entrypoint-unified.sh:1350
        Note over TS: the sink is deployed to the workspace copy, not the baked one<br/>entrypoint-unified.sh:1339 — see AB-12 for the bridge itself
    and
        U->>DI: node HOOK #124;#124; true, timeout 5000<br/>entrypoint-unified.sh:1471
        DI-->>M: JSON #123;result:continue, additionalContext#125; when inbox items are due<br/>dream-inbox-surface.cjs:37 and :42-58, see AB-08.12
    end
    U->>AD: PER-PROFILE ONLY: route, timeout 12000<br/>stacks.rs:62
    AD-->>M: stdout filtered by ROUTE_SIGNAL, then an optional #91;ONTOLOGY#93; breadcrumb<br/>claude-flow-hook-adapter.cjs:46 and :117-118, see AB-08.8
```

## AB-08.6 SessionStart — mirror, fleet tab, trust seed, and the per-profile restore
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core
    participant NM as nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:375
    participant FS as fleet-session-start.sh<br/>agentbox/config/hooks/fleet-session-start.sh:13
    participant FTN as fleet-tab-name.sh<br/>agentbox/config/hooks/fleet-tab-name.sh:13
    participant TSD as trust-seed.cjs<br/>agentbox/config/hooks/trust-seed.cjs:70
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:132

    CC->>NM: node HOOK SessionStart, timeout 8000<br/>entrypoint-unified.sh:1186
    NM-->>CC: body is #9654; session shortId started#40;source#41;<br/>nostr-live-mirror.cjs:274-277
    CC->>FS: bash HOOK #124;#124; true, timeout 8000<br/>entrypoint-unified.sh:1206
    FS->>FTN: bash fleet-tab-name.sh, errors swallowed<br/>fleet-session-start.sh:17
    Note over FTN: no TMUX or no tmux binary → exit 0<br/>fleet-tab-name.sh:14-15
    FTN->>FTN: name = git remote basename → toplevel → cwd basename :19-27
    FTN->>FTN: pin the name: automatic-rename off, allow-rename off, rename-window :32-34
    FTN->>FTN: write $HOME/.claude/fleet/#36;win#125;.json registry entry :37-40
    FS->>FS: gateway not running and AGENTBOX_NOSTR_GATEWAY!=0<br/>fleet-session-start.sh:19-20 → nohup node gateway.cjs, disown :23-24
    FS->>FS: deploy.sh present and AGENTBOX_TAB0_BRIDGE!=0<br/>fleet-session-start.sh:34 → nohup bash deploy.sh, disown :35-36
    CC->>TSD: node HOOK #124;#124; true, timeout 8000, continueOnError<br/>entrypoint-unified.sh:1294
    Note over TSD: targets = WORKSPACE + findRepos depth 5 + extra argv<br/>trust-seed.cjs:72, findRepos at :57-68
    TSD->>TSD: set hasTrustDialogAccepted and hasCompletedProjectOnboarding<br/>in the ~/.claude.json projects map :81-84
    alt any entry newly trusted
        TSD->>TSD: back up to WORKSPACE/.agentbox/claude.json.pre-trust-seed, then write in place :93-98
    else already trusted or dry-run
        TSD-->>CC: log only, no write :86-89
    end
    Note over TSD: INVARIANT fail-open — any throw is caught at top level, stderr only :102
    CC->>AD: PER-PROFILE ONLY: session-restore, timeout 15000<br/>stacks.rs:63
    Note over AD: stdout filtered by RESTORE_SIGNAL before injection<br/>claude-flow-hook-adapter.cjs:47 and :132-134
```

## AB-08.7 SessionEnd, Stop and SubagentStop — consolidation, mirror, ontology, trajectory close
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core
    participant NM as nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:375
    participant OM as ontology-monitor.cjs<br/>agentbox/config/hooks/ontology-monitor.cjs:232
    participant TS as turn-sink.cjs Stop<br/>agentbox/config/tab0-bridge/turn-sink.cjs:1
    participant TR as trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:555
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:135

    rect rgb(240,240,255)
    Note over CC,OM: SessionEnd
    CC->>NM: node HOOK SessionEnd, timeout 8000<br/>entrypoint-unified.sh:1186
    NM-->>CC: body is #9632; session shortId ended#40;reason#41;<br/>nostr-live-mirror.cjs:289-291
    CC->>OM: node HOOK #124;#124; true, timeout 200000 when the gate is on<br/>entrypoint-unified.sh:1245
    Note over OM: master switch AGENTBOX_ONTOLOGY_MONITOR seeded by entrypoint-unified.sh:1250<br/>so the hook is never a registered no-op — see AB-08.11
    CC->>AD: PER-PROFILE ONLY: session-end, timeout 10000<br/>stacks.rs:64 and :37
    end
    rect rgb(255,245,235)
    Note over CC,TR: Stop and SubagentStop
    CC->>NM: node HOOK Stop, timeout 8000<br/>entrypoint-unified.sh:1186
    NM-->>CC: body is the last assistant text from transcript_path<br/>nostr-live-mirror.cjs:284-287, scan at :248-264
    CC->>TS: node HOOK Stop #124;#124; true, timeout 8000<br/>entrypoint-unified.sh:1350
    CC->>TR: inline env prefix + node HOOK EVENT, timeout 10000<br/>entrypoint-unified.sh:1421, env prefix built at :1404-1406
    alt both gates on
        TR->>TR: handleClose on Stop or SubagentStop<br/>trajectory-recorder.cjs:569-571, see AB-08.13
    else either gate off — the default
        TR-->>CC: return 0 immediately :559-561, byte-identical to no hook present
    end
    Note over CC,TR: INVARIANT: registration itself is gated, not just the body —<br/>gate off de-registers over all 4 legacy events entrypoint-unified.sh:1439-1443
    end
```

## AB-08.8 claude-flow-hook-adapter.cjs — stdin-to-CLI translation (per-profile stacks only)
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core<br/>#40;profile stack, e.g. claude-core#41;
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:106
    participant CLI as claude-flow hooks CLI<br/>AGENTBOX_FLOW_BIN env, default claude-flow (:32)
    participant OP as ontology-push.js<br/>mcp/servers/lib #40;optional#41;
    participant M as Model context

    Note over CC,AD: registered by stacks.rs learning_hooks#40;#41; #40;stacks.rs:27-63#41;<br/>into workspace/profiles/#60;name#62;/.claude/settings.json - NOT this session's root settings.json
    CC->>AD: stdin JSON, argv[2]=action (:106-109)
    AD->>AD: parsePayload#40;readStdin#40;#41;#41; - malformed/empty JSON -#62; #123;#125; (:49-65)
    AD->>AD: extract tool_input.file_path, tool_input.command, prompt (:109-113)
    alt action=route
        AD->>CLI: spawnSync claude-flow hooks route --task PROMPT t=12000 (:117,67-73)
        CLI-->>AD: stdout #40;latency/alternatives dump, WASM-fallback banner possible#41;
        AD->>AD: filter stdout by ROUTE_SIGNAL regex #40;INFO#124;INTELLIGENCE#124;Agent:#124;Matched Pattern#41; (:46,74-81)
        AD-->>M: filtered lines only - full dump would pollute every turn (:41-46)
        opt ONTOLOGY_INJECT set and prompt non-empty
            AD->>OP: require ontology-push.js, getOntologyBreadcrumb#40;prompt#41; (:87-104)
            OP-->>AD: breadcrumb line or throws
            AD-->>M: #91;ONTOLOGY#93; breadcrumb appended, fail-open on any require/call error
        end
    else action=pre-edit #40;file present#41;
        AD->>CLI: hooks pre-edit --file FILE t=5000 (:120-122)
    else action=post-edit #40;file present#41;
        AD->>CLI: hooks post-edit --file FILE t=10000 (:123-125)
    else action=pre-command #40;command present#41;
        AD->>CLI: hooks pre-command --command CMD t=5000 (:126-128)
    else action=post-command #40;command present#41;
        AD->>CLI: hooks post-command --command CMD t=5000 (:129-131)
    else action=session-restore
        AD->>CLI: hooks session-restore t=15000 (:132-134)
        CLI-->>AD: stdout filtered by RESTORE_SIGNAL (:47,74-81)
        AD-->>M: filtered lines
    else action=session-end
        AD->>CLI: hooks session-end t=10000 (:135-137)
    else unknown action
        AD->>AD: no-op, never signals error (:138-140)
    end
    Note over AD: TRANSFORMERS_CACHE/HF_HOME forced to writable tmpfs path (:33-39,71)<br/>else @xenova/transformers ENOENT#39;s against the read-only Nix store
    Note over AD: whole main#40;#41; wrapped in try/catch, process.exit#40;0#41; unconditional (:144-149)<br/>INVARIANT: adapter holds NO learning state, intelligence lives in the CLI backend<br/>#40;header :13-17 cites stale ADR-015, pre-2026-consolidation id#41;
```
## AB-08.9 nostr-live-mirror.cjs — egress policy, redaction, NIP-59 gift wrap
```mermaid
sequenceDiagram
    autonumber
    participant H as Hook event
    participant M as main<br/>config/hooks/nostr-live-mirror.cjs:375
    participant P as policy<br/>config/hooks/lib/egress-policy.cjs:139
    participant R as recipientAllowed<br/>config/hooks/lib/egress-policy.cjs:59
    participant B as bodyForEvent<br/>config/hooks/nostr-live-mirror.cjs:272
    participant W as NIP-59 publisher
    H->>M: event name and pending stdin
    M->>P: global and live-mirror switches, sender identity
    alt disabled or sender unavailable
        P-->>M: skipped with reason
    else global admission passes
        M->>R: actual explicit recipient or derived child recipient
        alt missing, empty, malformed enumeration or unlisted key
            R-->>M: denied before stdin or body access
            M-->>H: skipped with reason
        else recipient enumerated
            M->>M: read stdin
            M->>B: compose event-specific text
            B-->>M: body
            M->>P: redactForEgress config/hooks/lib/egress-policy.cjs:118
            alt redaction fails
                P-->>M: null, skipped
            else redaction succeeds
                M->>M: bound composition and preserve activity URN
                alt dry-run
                    M-->>H: redacted local preview only
                else live
                    M->>W: gift-wrap and publish
                    W-->>M: accepted or failed with reason
                end
            end
        end
    end
    Note over M,R: G4 source requires a non-empty valid recipient set, including dry-run.<br/>25 isolated tests pass. Deployment needs an explicit reviewed recipient set.<br/>No messages were sent by the closeout tests.
```

## AB-08.10 trust-seed.cjs — folder-trust and worktree discovery
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code SessionStart
    participant TSD as trust-seed.cjs<br/>agentbox/config/hooks/trust-seed.cjs:70
    participant FS as filesystem
    participant CFG as ~/.claude.json<br/>trust-seed.cjs:27

    CC->>TSD: node trust-seed.cjs #91;--depth N#93; #91;--dry-run#93; #91;extra-path...#93; (:20,30-39)
    TSD->>FS: findRepos#40;WORKSPACE, depth=5, #91;#93;#41; recursive walk (:57-68)
    loop each dir entry, skip node_modules#124;target#124;.git#124;dist#124;build etc (:28,62)
        FS->>TSD: isProjectDir#40;p#41; = isGitRoot#40;p#41; OR has Cargo.toml#124;package.json#124;pyproject.toml#124;flake.nix#124;justfile (:41-55)
        opt isProjectDir true
            TSD->>TSD: acc.push#40;p#41; (:64)
        end
        TSD->>FS: recurse findRepos#40;p, depth-1, acc#41; (:65)
    end
    TSD->>TSD: targets = Set#40;WORKSPACE, ...repos, ...extra argv resolved#41; (:72)
    TSD->>CFG: JSON.parse#40;readFileSync#40;CONFIG#41;#41; - ENOENT tolerated, other errors abort (:74-76)
    loop each target dir
        alt cfg.projects#91;dir#93; already hasTrustDialogAccepted AND hasCompletedProjectOnboarding
            TSD->>TSD: skip, no change (:82)
        else
            TSD->>TSD: cfg.projects#91;dir#93; = #123;...entry, hasTrustDialogAccepted:true, hasCompletedProjectOnboarding:true#125; (:83-84)
        end
    end
    alt --dry-run OR added===0
        TSD-->>CC: stdout summary line only, CFG untouched (:86-89)
    else
        TSD->>FS: copyFileSync CONFIG to WORKSPACE/.agentbox/claude.json.pre-trust-seed backup (:93-97)
        TSD->>CFG: writeFileSync CONFIG, JSON.stringify#40;cfg, null, 2#41; #43; newline (:98)
        TSD-->>CC: stdout summary line #40;N checked, M newly trusted#41; (:99)
    end
    Note over TSD: INVARIANT: never removes/overwrites OTHER per-project keys in cfg.projects (:14)<br/>fail-open: top-level try/catch, stderr-only on error, no process.exit#40;1#41; anywhere (:102)
    Note over TSD: NOT FOUND IN CODE: no permissions.defaultMode or auto-mode<br/>opt-in dialog logic exists in this 102-line file - only trust-dialog fields are written
```
## AB-08.11 ruvnet-brain-ground.cjs and ontology-monitor.cjs
```mermaid
sequenceDiagram
    autonumber
    participant U as UserPromptSubmit
    participant RBG as ruvnet-brain-ground.cjs<br/>agentbox/config/hooks/ruvnet-brain-ground.cjs:37
    participant M as Model context
    participant SE as SessionEnd
    participant OM as ontology-monitor.cjs<br/>agentbox/config/hooks/ontology-monitor.cjs:232
    participant ONT as local ontology route<br/>mcp/servers/lib/ontology-local.js
    participant ZAI as claude-zai CLI<br/>AGENTBOX_ZAI_BIN, ZAI_URL (:149-178)
    participant FB as forum broker gate<br/>NostrBridge kind 31402

    rect rgb(235,245,255)
    U->>RBG: stdin JSON, hook.userInput#124;hook.prompt (:37-44)
    RBG->>RBG: REPO_PATTERN test against 25 RuvNet repo names (:15-21,33-35,50)
    RBG->>RBG: CLASSICAL_SUBS test - pinecone#124;pgvector#124;chromadb#124;weaviate#124;langchain#124;llamaindex#124;hnswlib (:23-31,61-69)
    alt any pattern matched
        RBG-->>M: JSON #123;result:continue, additionalContext: #91;GROUNDING#93;/#91;REDIRECT#93; lines#125; (:71-75)
    else no match
        RBG-->>M: JSON #123;result:continue#125; only (:85-87)
    end
    Note over RBG: fail-open: any JSON.parse or regex error -#62; exit#40;#41; path, no process.exit#40;1#41; anywhere (:79-81)
    end
    rect rgb(255,240,240)
    SE->>OM: stdin JSON payload, gated by AGENTBOX_ONTOLOGY_MONITOR=1 (:16,70-75)
    alt master switch off, or no ZAI key, or publish mode missing MANAGEMENT_API_KEY/NOSTR_RELAYS
        OM-->>SE: log no-op, exit 0 (:233-234)
    else gated on, BUDGET_MS=180000 wall clock (:31,34)
        OM->>OM: gatherWork#40;payload#41; - git status --porcelain + transcript tail 12k chars (:86-112)
        OM->>ONT: createLocalOntology#40;#41;.classList#40;limit:100000#41; (:118-122)
        OM->>OM: matchConcepts#40;#41; word-boundary label match, MAX_CONCEPTS=8 (:32,115-144)
        alt no concepts matched
            OM-->>SE: log no-op, exit 0 (:240)
        else
            OM->>ZAI: spawnCli claude-zai -p PROMPT, STRICT JSON proposals#91;#93; (:147-188)
            ZAI-->>OM: #123;proposals: #91;#123;iri,label,kind,title,summary,rationale#125;#93;#125;, MAX_PROPOSALS=5
            OM->>OM: fingerprint#40;p#41; = sha256#40;iri#124;kind#124;normalised summary#41;, filter against seen ledger (:50-56,245-247)
            alt fresh.length===0
                OM-->>SE: log all already seen, exit 0 (:247)
            else MODE=publish #40;AGENTBOX_ONTOLOGY_MONITOR_MODE, default dryrun#41;
                OM->>FB: buildActionRequest#40;panelId, category:ontology, kind 31402#41; per proposal (:201-228)
                FB-->>OM: published, NIP-33 d-tag = panelIdFor#40;p#41; so a repeat REPLACES the prior panel (:63-67)
            else MODE=dryrun
                OM->>OM: stageLocally#40;#41; appends $AGENTBOX_STATE/ontology-proposals.jsonl (:191-200)
            end
            OM->>OM: saveSeen#40;seen#41; ledger capped to last 5000 fingerprints (:57-62,255-256)
        end
    end
    Note over OM: human approval via 31403 happens OUTSIDE this hook - it only proposes #40;PRD-014 governed elevation#41;
    end
```
## AB-08.12 project-tracking-publish.cjs and dream-inbox-surface.cjs
```mermaid
sequenceDiagram
    autonumber
    participant CALLER as management API<br/>POST /v1/projects/:id/publish #40;NOT a Claude Code hook trigger#41;
    participant PTP as project-tracking-publish.cjs<br/>agentbox/config/hooks/project-tracking-publish.cjs:202
    participant MAPI as management-api<br/>GET /v1/projects on port 9090 (:133-180)
    participant NPB as nostr-pod-bridge track<br/>spawnSync binary (:184-200)
    participant U as UserPromptSubmit
    participant DI as dream-inbox-surface.cjs<br/>agentbox/config/hooks/dream-inbox-surface.cjs:24
    participant INBOX as dream-inbox.json<br/>runtime inbox path<br/>defined by dream-inbox-surface.cjs:20 INBOX
    participant M as Model context

    rect rgb(245,235,255)
    Note over CALLER,NPB: RESOLVED ADR-2068 #40;2026-09-05#41;: not a divergence — this is a CLI, not a hook.<br/>The management API spawns it from the /v1/projects publish route #40;routes/projects.js lines 28 and 322#41;.<br/>config/hooks/README.md section 3 names every file in config/hooks/ that is a CLI or helper rather than a hook
    CALLER->>PTP: node project-tracking-publish.cjs, optional ProjectTrackingDigest on stdin (:202,210-225)
    alt AGENTBOX_PROJECT_TRACKING_PUBLISH===0, or bridge secrets absent (:60-66,204-206)
        PTP-->>CALLER: return 0, silent no-op
    else
        alt stdin carries a valid digest #40;project_id present#41;
            PTP->>PTP: use verbatim (:215-216)
        else stdin is a TrackedProject or empty
            PTP->>MAPI: GET /v1/projects, Authorization Bearer MANAGEMENT_API_KEY (:135-146)
            MAPI-->>PTP: project list, mapped via toDigest#40;#41; (:92-110,228-233)
        end
        loop each digest
            PTP->>NPB: spawnSync nostr-pod-bridge track, digest JSON on stdin, t=30000 (:183-200)
            NPB-->>PTP: kind-30841 addressable event, d-tag=project slug, dual-write pod inbox
        end
        PTP-->>CALLER: log published N/total, return 0 (:241)
    end
    Note over PTP: guard setTimeout DEADLINE_MS=30000#43;1500 force-exits process (:246-247)
    end
    rect rgb(235,255,240)
    U->>DI: stdin JSON #40;payload unused - reads INBOX file directly#41; (:24-28)
    DI->>INBOX: JSON.parse#40;readFileSync#40;INBOX#41;#41; (:30)
    DI->>DI: filter status===open AND now-last_surfaced #62; RESURFACE_HOURS#42;3600, slice MAX_PER_TURN=2 (:21-22,33-36)
    alt due.length===0
        DI-->>M: JSON #123;result:continue#125; only (:37,65-67)
    else
        DI->>INBOX: stamp last_surfaced=now on due items, writeFileSync (:39-40)
        DI-->>M: JSON #123;result:continue, additionalContext: #91;DREAM INBOX#93; items #43; dream-inbox.mjs answer instructions#125; (:42-58)
    end
    Note over DI: fail-open: any error #40;missing file, bad JSON#41; -#62; exit#40;#41; path (:59-61)
    end
```
## AB-08.13 trajectory-recorder.cjs — Stop/SubagentStop boundary
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core<br/>Stop or SubagentStop
    participant TR as trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:555
    participant UT as trajectory-util.cjs<br/>agentbox/config/hooks/lib/trajectory-util.cjs:1
    participant STASH as os.tmpdir stash<br/>agentbox-traj-#60;sha12#62;.json<br/>trajectory-recorder.cjs:207-209
    participant PG as ruvector-postgres<br/>trajectories / trajectory_steps
    participant MAPI as management-api<br/>/v1/agent-events/emit on port 9090

    CC->>TR: node trajectory-recorder.cjs #60;event#62;, stdin JSON, RUVECTOR_* env inline (:555-556)
    Note over TR: ADR-2015: transcript-driven because PostToolUse does NOT fire<br/>for non-zero-exit Bash calls on this build (:22-26)
    alt NOT #40;RUVECTOR_MEMORY_LEARNING_ENABLED AND RUVECTOR_RECORD_TRAJECTORIES#41;
        TR-->>CC: return 0, byte-identical to no hook present #40;DEFAULT-OFF, :559-561#41;
    else event in #91;Stop, SubagentStop#93; (trajectory-recorder.cjs:570-571)
        TR->>STASH: readStash#40;session#41; - processedLines watermark #43; ctcPending queue (:389 and :211-217)
        TR->>UT: scanTranscript#40;lines, fromLine#41; grades each Bash tool_use by tool_result.is_error (:296-368)
        loop each Bash step found
            TR->>UT: redact#40;command#41; - I10 fail-closed, null command skips the step entirely (:346)
            TR->>UT: gradeResult#40;is_error,stderr,interrupted#41; -#62; success/failure + quality score (:340)
        end
        alt pg module unavailable
            TR-->>CC: log, watermark NOT advanced, lines retried next Stop #40;ADR-2015 closeout, :413#41;
        else
            TR->>STASH: ctcEmits = stash.ctcPending FIFO-drained first, bound CTC_QUEUE_MAX=2000 (:426,163)
            TR->>PG: INSERT trajectories #40;id, task, agent, status, metadata#41; ON CONFLICT DO NOTHING (:442-448)
            loop each graded step
                TR->>PG: INSERT trajectory_steps #40;id=sha12#40;tool_use_id#41;, action, result jsonb, quality#41; ON CONFLICT DO NOTHING (:468-479)
                TR->>TR: push ctcEmitBodyFromStep#40;#41; onto ctcEmits while #60; CTC_QUEUE_MAX (:487-491)
            end
            TR->>PG: UPDATE trajectories SET ended_at, status=complete, metadata#124;#124;jsonb_build_object#40;..., ctc_emit_queued, ctc_emit_carried_in#41; (:503-517)
            TR->>STASH: watermark advances ONLY after successful persist (:522)
            TR->>MAPI: emitCtcStepsBestEffort#40;ctcEmits#41; POST /v1/agent-events/emit, CTC_EMIT_CAP=200/invocation (:170-180 and :533)
            MAPI-->>TR: #123;attempted, deferred#125; - surplus beyond the cap is NOT dropped here (:180)
            TR->>STASH: cur.ctcPending = deferred #40;bounded CTC_QUEUE_MAX#41;, cur.ctcPendingOverflow accumulates true drops (:543-545)
            Note over TR: overflow beyond CTC_QUEUE_MAX IS dropped and logged as INCOMPLETE (:539-541)<br/>this is the only stash write outside the persistence path (:534-536)
        end
    end
    Note over TR: guard setTimeout 8000ms force-exits process regardless of PG/HTTP state (:577)
    Note over TR: see AB-21 for the learning loop this feeds #40;ReasoningBank / SONA consumers of trajectories#41;
```
## AB-08.14 lib/ shared helpers — consumers
```mermaid
flowchart TD
    LIBDIR["agentbox/config/hooks/lib/<br/>two modules, one consumer each<br/>hooks/README.md:46"]
    UT["trajectory-util.cjs<br/>agentbox/config/hooks/lib/trajectory-util.cjs:1"]
    EG["egress-policy.cjs<br/>agentbox/config/hooks/lib/egress-policy.cjs:1<br/>JS half of the ADR-2026 content-egress policy :3-10"]
    LIBDIR --> UT
    LIBDIR --> EG
    F1["sha12#40;s#41; trajectory-util.cjs:18<br/>content-addressed step/trajectory ids"]
    F2["commandPattern#40;command#41; trajectory-util.cjs:38<br/>+ hasResidualSecret#40;text#41; trajectory-util.cjs:180"]
    F3["redact#40;command#41; trajectory-util.cjs:196<br/>I10 fail-closed redaction gate"]
    F4["deriveOutcome#40;toolResponse#41; trajectory-util.cjs:227<br/>+ gradeResult#40;isError,stderr,interrupted#41; trajectory-util.cjs:285"]
    F5["tokenCountOf#40;usage#41; trajectory-util.cjs:309<br/>+ usageIdentityOf#40;rec#41; trajectory-util.cjs:341"]
    F6["handoffIdFrom#40;env,fallbackId#41; trajectory-util.cjs:367<br/>CTC chain-correlation id"]
    F7["ctcEmitBodyFromStep#40;step,opts#41; trajectory-util.cjs:392<br/>builds /v1/agent-events/emit body"]
    UT --> F1 & F2 & F3 & F4 & F5 & F6 & F7
    G1["OUTCOME skipped#124;attempted#124;accepted#124;failed<br/>egress-policy.cjs:29-34"]
    G2["recipientAllowed#40;recipient#41; :59-68<br/>64-hex grammar #43; mandatory non-empty AGENTBOX_MIRROR_RECIPIENTS allowlist :46-53"]
    G3["redactForEgress#40;text#41; :115-124<br/>null return means fail-closed SKIP"]
    G4["egressDecision#40;pathId, opts#41; :136-162<br/>global, per-path and redaction-disabled switches"]
    EG --> G1 & G2 & G3 & G4
    TR["trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:47<br/>ONLY consumer of trajectory-util.cjs"]
    NM["nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:44<br/>ONLY consumer of egress-policy.cjs, see AB-08.9"]
    F1 & F2 & F3 & F4 & F5 & F6 & F7 --> TR
    G1 & G2 & G3 & G4 --> NM
    PAIR["INVARIANT ADR-2026: the Rust twin services/nostr-pod-bridge/src/egress_policy.rs<br/>must satisfy the SAME fixture tests/fixtures/egress-redaction.v1.json<br/>egress-policy.cjs:6-10 — a shared comment is not a shared contract"]
    EG -.-> PAIR
    OTHER["the remaining hooks import no lib/ module — each is self-contained,<br/>duplicating small helpers #40;readStdin, envFirst#41; independently:<br/>claude-flow-hook-adapter, trust-seed, ruvnet-brain-ground,<br/>ontology-monitor, dream-inbox-surface, project-tracking-publish,<br/>fleet-session-start and fleet-tab-name"]
    LIBDIR -.-> OTHER
```
