---
id: AB-08
title: Claude Code hook pipeline and its handlers
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2015, ADR-2026, ADR-2007, ADR-2068, ADR-2090, ADR-2091, ADR-2093, ADR-2094, ADR-2121, ADR-2144, ADR-2146, ADR-2151]
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
  - ../project/agentbox/config/hooks/lib/failure-jev.cjs
  - ../project/agentbox/config/hooks/lib/ontology-proposal-jev.cjs
  - ../project/agentbox/config/hooks/decline-recorder.cjs
  - ../project/agentbox/config/hooks/activity-sink.cjs
  - ../project/agentbox/docs/adr/ADR-2144-the-floor-is-a-managed-tier-and-a-decline-is-recorded-verbatim.md
  - ../project/agentbox/docs/adr/ADR-2146-activity-sink-hook.md
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
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
  - ../project/agentbox/config/hooks/skill-route.cjs
  - ../project/agentbox/config/hooks/session-state.cjs
  - ../project/agentbox/config/hooks/lib/skill-route.cjs
  - ../project/agentbox/config/hooks/lib/hook-output.cjs
  - ../project/agentbox/lib/factrail.nix
  - ../project/agentbox/scripts/factrail-store-migrate.mjs
  - ../project/agentbox/docs/adr/ADR-2121-factrail-implements-jev-compaction.md
verified_commit: 0b0f7918961e629280faf86f88227535abf514b5
---

Registration ground truth (2026-09-29): `~/.claude/settings.json` is boot-generated and never tracked, so AB-08.1–AB-08.7
cite the sites that WRITE it — `config/entrypoint-unified.sh` for the root session and
`services/agentbox-manifest/src/stacks.rs` `learning_hooks#40;#41;` for per-profile stacks — the split `config/hooks/README.md` owns. All hook `timeout` values are now SECONDS, not
milliseconds (`services/agentbox-manifest/src/stacks.rs:29-30`); a bare `UserPromptSubmit` → `route` per-profile hook no
longer exists (ADR-2091 consolidated per-turn routing into `skill-route.cjs`, see AB-08.5).

## AB-08.1 Root-session registration — what entrypoint-unified.sh seeds into ~/.claude/settings.json
```mermaid
flowchart TB
    S["settings path resolved once, CLAUDE_CONFIG_DIR<br/>fallback /home/devuser/.claude/settings.json<br/>entrypoint-unified.sh:1695"]
    subgraph MIR["nostr-live-mirror.cjs — always registered, defaults to Stop only"]
    direction TB
    M1["block entrypoint-unified.sh:1695, baked /opt hook path :1696"]
    M2["entrypoint-unified.sh:1705 events SessionStart, UserPromptSubmit, Stop, SessionEnd<br/>idempotent marker test on the command string :1707"]
    M3["entrypoint-unified.sh:1710 push node HOOK EVENT, timeout 8 SECONDS<br/>each event registered, but bodyForEvent skips any event outside<br/>AGENTBOX_LIVE_MIRROR_EVENTS (default Stop only) nostr-live-mirror.cjs:278-279"]
    M1 --> M2 --> M3
    end
    subgraph FLT["fleet-session-start.sh — SessionStart"]
    direction TB
    F1["block entrypoint-unified.sh:1722, baked hook path :1721<br/>off switch AGENTBOX_NOSTR_GATEWAY=0 gates the gateway launch inside the script, not registration"]
    F2["entrypoint-unified.sh:1729 marker test, :1730 push timeout 8 seconds"]
    F1 --> F2
    end
    subgraph ONT["ontology-monitor.cjs — SessionEnd, gated, detached"]
    direction TB
    O1["entrypoint-unified.sh:1747 gate read via agentbox-manifest toml-bool<br/>defaults to 0 on any read failure"]
    O2["entrypoint-unified.sh:1770 ON: push timeout 10 seconds (hook returns at once,<br/>a detached child does the 180s review) AND seed env AGENTBOX_ONTOLOGY_MONITOR=1 :1775"]
    O3["entrypoint-unified.sh:1779-1783 OFF: filter the hook back out<br/>delete the emptied SessionEnd array :1783<br/>INVARIANT ADR-2020 byte-identical-when-off"]
    O4["gate source agentbox.toml:115-116 enabled = true"]
    O1 --> O2
    O1 --> O3
    O1 --> O4
    end
    subgraph TRU["trust-seed.cjs — boot-time run ONLY, no longer a hook"]
    direction TB
    T1["entrypoint-unified.sh:1812 hook path, :1813 gate AGENTBOX_TRUST_SEED != 0<br/>DELIBERATELY not a SessionStart hook: walked ~1,170 paths/session (avg 3.2s),<br/>raced Claude Code's own ~/.claude.json writes"]
    T2["entrypoint-unified.sh:1814 node trust-seed.cjs, output to stderr<br/>callable by hand for a post-boot worktree: node trust-seed.cjs #60;path#62;"]
    T1 --> T2
    end
    subgraph TRJ["trajectory-recorder.cjs — Stop and SubagentStop, gated"]
    direction TB
    J1["block entrypoint-unified.sh:2025, hook path :2038<br/>gate: BOTH memory_learning flags checked before the block runs"]
    J2["entrypoint-unified.sh:2063 specs = Stop, SubagentStop ONLY<br/>reconcile strips prior wiring over 4 legacy events :2063-2066<br/>then push timeout 10 seconds behind an inline env prefix :2067-2071"]
    J3["entrypoint-unified.sh:2087 gate off: strip over the same 4 events<br/>:2090 keep-filter, :2093 log the de-registration"]
    J4["gate source agentbox.toml:476-477 enabled + record_trajectories"]
    J1 --> J2
    J1 --> J3
    J1 --> J4
    end
    subgraph OTH["turn-sink, dream-inbox, ruvnet-brain"]
    direction TB
    X1["entrypoint-unified.sh:1902 tab0-bridge turn-sink.cjs deployed path<br/>:1910 events UserPromptSubmit and Stop, :1913 timeout 5 seconds"]
    X2["entrypoint-unified.sh:2208 dream-inbox-surface.cjs, live-checkout fallback :2209<br/>gate DREAM_INBOX_HOOK :2210, UserPromptSubmit timeout 5 seconds :2224"]
    X3["entrypoint-unified.sh:2623 ruvnet-brain-ground.cjs<br/>gate RUVNET_BRAIN_GROUNDING_HOOK :2624, timeout 5 seconds :2637"]
    X4["gate source agentbox.toml:1075 grounding_hook = true"]
    X1 --> X2 --> X3 --> X4
    end
    subgraph SHM["hook shim reconcile — ADR-2034 §1"]
    direction TB
    H1["entrypoint-unified.sh:1827 block, gate AGENTBOX_HOOK_SHIM"]
    H2["entrypoint-unified.sh:1828 agentbox-hook reconcile --root WORKSPACE --depth 2<br/>rewrites per-project ruflo CLI hooks to the resident shim"]
    H1 --> H2
    end
    S --> MIR --> FLT --> ONT --> TRU --> TRJ --> OTH --> SHM
```

## AB-08.2 Per-profile registration — stacks.rs learning_hooks, and what is NOT a hook
```mermaid
flowchart TB
    subgraph PP["workspace/profiles/#60;stack#62;/.claude/settings.json"]
    direction TB
    P0["build_profile starts from learning_hooks#40;env, gates#41; stacks.rs:290, fn at :34,<br/>then appends floor, voicelink, session-plane (:324-334) and activity-sink entries :293-337, writes hooks at :350<br/>see AB-08.16 for those four gated additions"]
    P1["every learning_hooks entry is node ADAPTER action #124;#124; true<br/>stacks.rs:38, timeouts are SECONDS :29-30"]
    P2["PreToolUse stacks.rs:69<br/>Bash → pre-command t=5 :70<br/>Write#124;Edit#124;MultiEdit → pre-edit t=5 :71"]
    P3["PostToolUse :73<br/>Write#124;Edit#124;MultiEdit → post-edit t=10 :74<br/>Bash → post-command t=5 :75"]
    P4["SessionStart → session-restore t=15 :77<br/>NO UserPromptSubmit entry: the per-profile route hook<br/>is RETIRED, comment stacks.rs:31-33, test :527-528"]
    P5["SessionEnd :78 = session-end t=10 :43<br/>+ nostr summary hook when mobile_bridge, DETACHED via setsid :44-56<br/>+ ontology-monitor when ontology_monitor, also DETACHED t=10 :58-65"]
    P0 --> P1 --> P2 --> P3 --> P4 --> P5
    end
    subgraph NOT["NOT hooks — hooks/README.md"]
    direction TB
    N1["project-tracking-publish.cjs — a CLI the management API spawns<br/>from POST /v1/projects/:id/publish, reads a digest on stdin<br/>hooks/README.md:68, see AB-08.12"]
    N2["fleet-tab-name.sh — shelled by fleet-session-start.sh:17<br/>hooks/README.md:69"]
    N3["lib/egress-policy.cjs, lib/trajectory-util.cjs, lib/skill-route.cjs — required<br/>libraries, skill-route.cjs shared with the /route CLI<br/>hooks/README.md:71, see AB-08.14. DOC-DRIFT: lib/hook-output.cjs, lib/jev-option.cjs,<br/>lib/failure-jev.cjs and lib/ontology-proposal-jev.cjs are not listed here"]
    N1 --> N2 --> N3
    end
    subgraph INV["invariants and divergences"]
    direction TB
    I1["INVARIANT: claude-flow-hook-adapter.cjs is wired ONLY per-profile<br/>through learning_hooks stacks.rs:290 into the file written at :350 — never into the root settings.json"]
    I2["INVARIANT ADR-2068: the root ontology gap is closed —<br/>entrypoint-unified.sh:1770 registers and :1779-1783 retracts,<br/>so the off state leaves no trace"]
    I3["INVARIANT: a new hook is not wired by dropping a file in config/hooks/ —<br/>register it in the site that owns its session class<br/>hooks/README.md:99-102"]
    I1 --> I2 --> I3
    end
    PP --> NOT --> INV
```

## AB-08.5 UserPromptSubmit — five root hooks; the per-profile route hook no longer exists
```mermaid
sequenceDiagram
    autonumber
    participant U as User turn
    participant NM as nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:394
    participant RBG as ruvnet-brain-ground.cjs<br/>agentbox/config/hooks/ruvnet-brain-ground.cjs:115
    participant TS as turn-sink.cjs<br/>agentbox/config/tab0-bridge/turn-sink.cjs:1
    participant DI as dream-inbox-surface.cjs<br/>agentbox/config/hooks/dream-inbox-surface.cjs:34
    participant SR as main<br/>agentbox/config/hooks/skill-route.cjs:29
    participant M as Model context

    Note over U,M: root registrations: entrypoint-unified.sh:1705 mirror, :1910 turn-sink,<br/>:2210 dream-inbox, :2624 brain-ground, :2749 skill-route — five on this one event, plus the gated decline-recorder, session-state :1952 (ADR-2153) and activity-sink (AB-08.16).<br/>INVARIANT: stacks.rs learning_hooks registers no per-profile UserPromptSubmit route hook<br/>(stacks.rs:31-33, test :527-528) and only those gated additions add one — skill-route.cjs is the sole per-turn router (ADR-2091)
    U->>NM: node HOOK UserPromptSubmit, timeout 8 seconds<br/>entrypoint-unified.sh:1710
    NM->>NM: bodyForEvent#40;#41; checks mirroredEvents#40;#41; first, returns null if not listed<br/>nostr-live-mirror.cjs:290-291
    NM-->>U: DEFAULT: null body on UserPromptSubmit, egress skipped — only Stop is<br/>mirrored by default, DEFAULT_MIRROR_EVENTS :277, widen via AGENTBOX_LIVE_MIRROR_EVENTS :278-282, see AB-13.9
    par independent root groups, unordered wrt each other
        U->>RBG: node HOOK #124;#124; true, timeout 5 seconds<br/>entrypoint-unified.sh:2637
        RBG->>RBG: analyse#40;prompt#41;: DISTINCTIVE name fires alone,<br/>an ESTATE name needs an upstream-intent cue in the SAME sentence<br/>ruvnet-brain-ground.cjs:90-98
        RBG->>RBG: CLASSICAL_SUBS match needs selection intent in the same sentence<br/>ruvnet-brain-ground.cjs:100-109
        RBG-->>M: emitContext via lib/hook-output.cjs, or nothing<br/>ruvnet-brain-ground.cjs:121-124, see AB-08.11
    and
        U->>TS: node HOOK UserPromptSubmit #124;#124; true, timeout 5 seconds<br/>entrypoint-unified.sh:1913
        Note over TS: the sink is deployed to the workspace copy, not the baked one<br/>entrypoint-unified.sh:1902 — see AB-12 for the bridge itself
    and
        U->>SR: ADR-2091 gate inlined as AGENTBOX_SKILL_ROUTER=jev<br/>entrypoint-unified.sh:2749, env prefix :2773, timeout = max#40;8, ceil#40;2 times judge_ms / 1000#41;#41; :2792
        SR->>SR: lib.route#40;prompt, cfg#41;: one Choice over every routable skill<br/>hooks/skill-route.cjs:37 via lib/skill-route.cjs
        SR->>SR: candidates also take claude.ai synced skills as anthropic-skills:name and enabled<br/>plugins' skills as plugin:skill, the Skill tool's own names (lib/skill-route.cjs:293, lib/skill-route.cjs:312),<br/>read best-effort, off with AGENTBOX_SKILL_ROUTE_CLAUDE_DIR=0 (lib/skill-route.cjs:397)
        SR-->>M: emitContext#40;lib.formatContext#40;r, cfg#41;#41; via lib/hook-output.cjs, or nothing<br/>hooks/skill-route.cjs:42
        Note over SR: ADR-2139 behind the loopback proxy the hook sends no Authorization and the no-key skip is waived (lib/skill-route.cjs:405, :576, :621, :644-646),<br/>the flag rides the inlined command as AGENTBOX_SYSTEM_ONE_PROXY=1 (entrypoint-unified.sh:2760, :2776), cost uses the proxy's x-system-one-usd-per-mtok-in header (lib/skill-route.cjs:670-673)
        Note over SR: INVARIANT fail-open - any error, timeout, 429/529 or a none pick<br/>returns without injection, hooks/skill-route.cjs:20 and hooks/skill-route.cjs:45
    and
        U->>DI: node HOOK #124;#124; true, timeout 5 seconds<br/>entrypoint-unified.sh:2224
        DI->>DI: skip machine-generated turns #40;task-notification, agent-message,<br/>system-reminder#41;, then count OPEN items and rate-limit via a sidecar<br/>stamp file, dream-inbox-surface.cjs:40-51
        DI-->>M: hookSpecificOutput.additionalContext = one-line pointer at the forum<br/>governance panel #40;ADR-2115#41;, or #123;#125;, dream-inbox-surface.cjs:52-61, see AB-08.12
    end
```

## AB-08.6 SessionStart — mirror, fleet tab, and the per-profile restore; trust-seed is boot-only now
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core
    participant NM as nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:394
    participant FS as fleet-session-start.sh<br/>agentbox/config/hooks/fleet-session-start.sh:13
    participant FTN as fleet-tab-name.sh<br/>agentbox/config/hooks/fleet-tab-name.sh:13
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:132

    CC->>NM: node HOOK SessionStart, timeout 8 seconds<br/>entrypoint-unified.sh:1710
    NM-->>CC: DEFAULT: null body, skipped — SessionStart is not in the default<br/>mirroredEvents set, nostr-live-mirror.cjs:278 and :290-291, see AB-13.9
    CC->>FS: bash HOOK #124;#124; true, timeout 8 seconds<br/>entrypoint-unified.sh:1730
    FS->>FTN: bash fleet-tab-name.sh, errors swallowed<br/>fleet-session-start.sh:17
    Note over FTN: no TMUX or no tmux binary → exit 0<br/>fleet-tab-name.sh:14-15
    FTN->>FTN: name = git remote basename → toplevel → cwd basename :19-27
    FTN->>FTN: pin the name: automatic-rename off, allow-rename off, rename-window :32-34
    FTN->>FTN: write $HOME/.claude/fleet/#36;win#125;.json registry entry :37-40
    FS->>FS: gateway not running and AGENTBOX_NOSTR_GATEWAY!=0<br/>fleet-session-start.sh:19-20 → nohup node gateway.cjs, disown :23-24
    FS->>FS: deploy.sh present and AGENTBOX_TAB0_BRIDGE!=0<br/>fleet-session-start.sh:34 → nohup bash deploy.sh, disown :35-36
    Note over CC: trust-seed.cjs is NOT registered here any more. It ran ONCE at boot,<br/>entrypoint-unified.sh:1812-1814, and is invoked by hand for a worktree made after<br/>boot: node trust-seed.cjs #60;path#62; — see AB-08.10
    CC->>AD: PER-PROFILE ONLY: session-restore, timeout 15 seconds<br/>stacks.rs:77
    Note over AD: stdout filtered by RESTORE_SIGNAL before injection<br/>claude-flow-hook-adapter.cjs:47 and :132-134
```

## AB-08.7 SessionEnd, Stop and SubagentStop — consolidation, mirror, detached ontology, trajectory close
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core
    participant NM as nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:394
    participant OM as ontology-monitor.cjs<br/>agentbox/config/hooks/ontology-monitor.cjs:271
    participant TS as turn-sink.cjs Stop<br/>agentbox/config/tab0-bridge/turn-sink.cjs:1
    participant TR as trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:577
    participant AD as claude-flow-hook-adapter.cjs<br/>agentbox/config/hooks/claude-flow-hook-adapter.cjs:135

    rect rgb(240,240,255)
    Note over CC,OM: SessionEnd
    CC->>NM: node HOOK SessionEnd, timeout 8 seconds<br/>entrypoint-unified.sh:1710
    NM-->>CC: DEFAULT: null body, skipped — SessionEnd is not in the default<br/>mirroredEvents set, nostr-live-mirror.cjs:278 and :290-291, see AB-13.9
    CC->>OM: node HOOK #124;#124; true, timeout 10 seconds when the gate is on<br/>entrypoint-unified.sh:1770 — the review itself budgets 180s but runs DETACHED,<br/>ontology-monitor.cjs:38
    OM->>OM: not runInForeground: fork a detached child via spawn, stdio ignored<br/>except a logfile, unref, exit 0 at once — ontology-monitor.cjs:249-266
    Note over OM: master switch AGENTBOX_ONTOLOGY_MONITOR seeded by entrypoint-unified.sh:1775<br/>so the hook is never a registered no-op — see AB-08.11
    CC->>AD: PER-PROFILE ONLY: session-end, timeout 10000<br/>stacks.rs:78 and :43
    end
    rect rgb(255,245,235)
    Note over CC,TR: Stop and SubagentStop
    CC->>NM: node HOOK Stop, timeout 8 seconds<br/>entrypoint-unified.sh:1710
    NM-->>CC: body is the last assistant text from transcript_path — Stop IS the default<br/>mirroredEvents entry, nostr-live-mirror.cjs:303-306, scan at :251-264, see AB-13.9
    CC->>TS: node HOOK Stop #124;#124; true, timeout 5 seconds<br/>entrypoint-unified.sh:1913
    CC->>TR: inline env prefix + node HOOK EVENT, timeout 10 seconds<br/>entrypoint-unified.sh:2069, env prefix built at :2051-2054
    alt both gates on
        TR->>TR: handleClose on Stop or SubagentStop<br/>trajectory-recorder.cjs:591-593, see AB-08.13
    else either gate off — the default
        TR-->>CC: return 0 immediately :581-583, byte-identical to no hook present
    end
    Note over CC,TR: INVARIANT: registration itself is gated, not just the body —<br/>gate off de-registers over all 4 legacy events entrypoint-unified.sh:2087-2093
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
        Note over AD,CLI: pre-edit/post-edit/pre-command/post-command all call run#40;#41; with NO signal<br/>arg, so the #40;signal && ...#41; stdout gate never fires for them - only<br/>route and session-restore forward filtered CLI stdout (:74-81)
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
## AB-08.10 trust-seed.cjs — a BOOT-TIME script now, not a hook; folder-trust, worktree discovery and per-profile stores
```mermaid
sequenceDiagram
    autonumber
    participant EP as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1812
    participant TSD as trust-seed.cjs main#40;#41;<br/>agentbox/config/hooks/trust-seed.cjs:88
    participant FS as filesystem
    participant CFG as each Claude config store<br/>trust-seed.cjs:91

    Note over EP,TSD: runs ONCE at boot, gated AGENTBOX_TRUST_SEED!=0 (entrypoint-unified.sh:1813), never as a<br/>SessionStart hook - it walked ~1,170 paths/session (avg 3.2s) and raced Claude<br/>Code's own ~/.claude.json writes (entrypoint-unified.sh:1805-1810). Callable by hand post-boot for a<br/>new worktree: node trust-seed.cjs #60;path#62; (entrypoint-unified.sh:1810)
    EP->>TSD: node trust-seed.cjs #91;--depth N#93; #91;--dry-run#93; #91;extra-path...#93; (entrypoint-unified.sh:1814, parseArgs trust-seed.cjs:38-47)
    TSD->>FS: findRepos#40;WORKSPACE, depth=5, #91;#93;#41; recursive walk (trust-seed.cjs:65-76)
    loop each dir entry, skip node_modules#124;target#124;.git#124;dist#124;build#124;.tmp#124;.cache#124;.venv#124;venv (trust-seed.cjs:36,70)
        FS->>TSD: isProjectDir#40;p#41; = isGitRoot#40;p#41; OR has Cargo.toml#124;package.json#124;pyproject.toml#124;flake.nix#124;justfile (trust-seed.cjs:49-63)
        opt isProjectDir true
            TSD->>TSD: acc.push#40;p#41; (trust-seed.cjs:72)
        end
        TSD->>FS: recurse findRepos#40;p, depth-1, acc#41; (trust-seed.cjs:73)
    end
    TSD->>TSD: targets = Set#40;WORKSPACE, ...repos, ...extra argv resolved#41; (trust-seed.cjs:90)
    TSD->>FS: profileStores#40;#41; lists WORKSPACE/profiles/slug/.claude/.claude.json, keeping only stores that already exist (trust-seed.cjs:78-86)
    Note over TSD,CFG: a harness-wrapper profile that has never run has no store and is skipped (trust-seed.cjs:12-13).<br/>Before this change only ~/.claude.json was seeded, and a profile reads only its own CLAUDE_CONFIG_DIR store (trust-seed.cjs:9-12)
    loop each store in #91;~/.claude.json, ...profile stores#93; (trust-seed.cjs:91)
        TSD->>CFG: seedStore#40;store, targets, args#41; JSON.parse#40;readFileSync#40;store#41;#41; - ENOENT tolerated, other errors skip this store (trust-seed.cjs:100-102)
        loop each target dir
            alt cfg.projects#91;dir#93; already hasTrustDialogAccepted AND hasCompletedProjectOnboarding
                TSD->>TSD: skip, no change (trust-seed.cjs:108)
            else
                TSD->>TSD: cfg.projects#91;dir#93; = #123;...entry, hasTrustDialogAccepted:true, hasCompletedProjectOnboarding:true#125; (trust-seed.cjs:109-110)
            end
        end
        alt --dry-run OR added===0
            TSD-->>EP: stderr summary line only, store untouched (trust-seed.cjs:112-115)
        else
            TSD->>FS: copyFileSync store to WORKSPACE/.agentbox/NAME.pre-trust-seed backup, NAME claude.json or slug.claude.json (trust-seed.cjs:119-124)
            TSD->>CFG: writeFileSync store, JSON.stringify#40;cfg, null, 2#41; #43; newline (trust-seed.cjs:125)
            TSD-->>EP: stderr summary line #40;store, N checked, M newly trusted#41; (trust-seed.cjs:126)
        end
    end
    Note over TSD: progress goes to STDERR ONLY (trust-seed.cjs:25) - a hook's stdout is injected into<br/>model context, and this is deliberately not a hook any more
    Note over TSD: INVARIANT: never removes/overwrites OTHER per-project keys in cfg.projects (trust-seed.cjs:18)<br/>fail-open per store: seedStore errors are caught and reported, the loop continues (trust-seed.cjs:92-94), and a top-level try/catch backs it (trust-seed.cjs:129)
```
## AB-08.11 ruvnet-brain-ground.cjs (rewritten trigger logic) and ontology-monitor.cjs (now detached)
```mermaid
sequenceDiagram
    autonumber
    participant U as UserPromptSubmit
    participant RBG as ruvnet-brain-ground.cjs<br/>agentbox/config/hooks/ruvnet-brain-ground.cjs:115
    participant HO as lib/hook-output.cjs<br/>agentbox/config/hooks/lib/hook-output.cjs:32
    participant SE as SessionEnd
    participant OMP as ontology-monitor.cjs parent<br/>agentbox/config/hooks/ontology-monitor.cjs:271
    participant OMC as ontology-monitor.cjs DETACHED CHILD<br/>agentbox/config/hooks/ontology-monitor.cjs:258
    participant ONT as local ontology route<br/>mcp/servers/lib/ontology-local.js
    participant ZAI as claude-zai CLI<br/>AGENTBOX_ZAI_BIN, ZAI_URL (:177-183)
    participant FB as forum broker gate<br/>NostrBridge kind 31402

    rect rgb(235,245,255)
    U->>RBG: stdin JSON, hook.userInput#124;hook.prompt (:120-121)
    RBG->>RBG: analyse#40;prompt#41; splits into sentences (:85-87)
    RBG->>RBG: DISTINCTIVE name #40;ruvnet, qudag, ruv-fann...#41; fires alone,<br/>an ESTATE name #40;ruvector, claude-flow, ruflo...#41; needs an<br/>UPSTREAM_CUES match in the SAME sentence (:36-47,74-77,96-98)
    RBG->>RBG: CLASSICAL_SUBS #40;pinecone, pgvector, langchain...#41; needs<br/>SELECTION intent #40;use, migrate, compare...#41; in the same sentence (:59-70,100-109)
    alt any pattern matched
        RBG->>HO: emitContext#40;ctx#41; — GROUNDING and/or up to 2 REDIRECT lines,<br/>capped at MAX_CONTEXT_CHARS=1200 (:34,79-82,111-112,121-124)
        HO-->>U: hookSpecificOutput.additionalContext (:22-26,32-41)
    else no match
        RBG-->>U: no stdout — emitContext resolves false, main returns (:121-122)
    end
    Note over RBG: fail-open: any JSON.parse error in main -#62; return, no process.exit#40;1#41; anywhere (:120)
    end
    rect rgb(255,240,240)
    SE->>OMP: stdin JSON payload, gated by AGENTBOX_ONTOLOGY_MONITOR=1 (:78,272-273)
    alt gatedOff#40;#41;: master switch off, no ZAI key, or publish mode missing<br/>MANAGEMENT_API_KEY/NOSTR_RELAYS
        OMP-->>SE: log no-op, exit 0 (:77-82,272-273)
    else gated on
        OMP->>OMC: NOT runInForeground: spawn detached child, stdio ignored except a<br/>logfile, unref, exit 0 AT ONCE — SessionEnd never waits (:249-268, exit :274)
        Note over OMC: BUDGET_MS=180000 wall clock, read from AGENTBOX_ONTOLOGY_MONITOR_BUDGET_MS (:38,41)
        OMC->>OMC: gatherWork#40;payload#41; - git status --porcelain + transcript tail 12k chars (:93-119)
        OMC->>ONT: createLocalOntology#40;#41;.classList#40;limit:100000#41; (:125-129)
        OMC->>OMC: matchConcepts#40;#41; word-boundary label match, MAX_CONCEPTS=8 (:39,122-151)
        alt no concepts matched
            OMC-->>SE: log no-op, exit 0 (:282)
        else
            OMC->>ZAI: spawnCli claude-zai -p PROMPT, STRICT JSON proposals#91;#93; (:154-195)
            ZAI-->>OMC: #123;proposals: #91;#123;iri,label,kind,title,summary,rationale#125;#93;#125;, MAX_PROPOSALS=5
            OMC->>OMC: fingerprint#40;p#41; = sha256#40;iri#124;kind#124;normalised summary#41;, filter against seen ledger (:57-60,287-288)
            alt fresh.length===0
                OMC-->>SE: log all already seen, exit 0 (:289)
            else
                OMC->>OMC: Jev option, off unless AGENTBOX_JEV_ONTOLOGY_PROPOSALS=1:<br/>lib/ontology-proposal-jev.cjs verify#40;fresh#41; annotates each proposal,<br/>fail-open to the same array (:290-298, flag lib/ontology-proposal-jev.cjs:41)
                OMC->>OMC: verdicts appended to ontology-proposals-jev.jsonl,<br/>never part of the signed 31402 (:299, stageVerdicts :205-212)
                OMC->>OMC: held only when AGENTBOX_JEV_PROPOSAL_HOLD_BELOW is set in 0 to 0.5<br/>and supported is below it, held ones go to the local review queue<br/>(:300, lib/ontology-proposal-jev.cjs:175-186)
            end
            alt emit empty after the Jev hold
                OMC-->>SE: mark fresh seen, exit 0 (:302)
            else MODE=publish #40;AGENTBOX_ONTOLOGY_MONITOR_MODE, default dryrun#41;
                OMC->>FB: buildActionRequest#40;panelId, category:ontology, kind 31402#41; per proposal (:216-243)
                FB-->>OMC: published, NIP-33 d-tag = panelIdFor#40;p#41; so a repeat REPLACES the prior panel (:70-74)
            else MODE=dryrun
                OMC->>OMC: stageLocally#40;emit#41; appends $AGENTBOX_STATE/ontology-proposals.jsonl (:195-203, call :307)
            end
            OMC->>OMC: saveSeen#40;seen#41; ledger capped to last 5000 fingerprints (:64-69,310-311)
        end
    end
    Note over OMC: human approval via 31403 happens OUTSIDE this hook - it only proposes #40;PRD-014 governed elevation#41;
    end
```
## AB-08.12 project-tracking-publish.cjs and dream-inbox-surface.cjs — DREAM-2115: a reminder pointer, not a relay
```mermaid
sequenceDiagram
    autonumber
    participant CALLER as management API<br/>POST /v1/projects/:id/publish #40;NOT a Claude Code hook trigger#41;
    participant PTP as project-tracking-publish.cjs<br/>agentbox/config/hooks/project-tracking-publish.cjs:205
    participant MAPI as management-api<br/>GET /v1/projects on port 9090 (:136-180)
    participant NPB as nostr-pod-bridge track<br/>spawnSync binary (:186-200)
    participant U as UserPromptSubmit
    participant DI as dream-inbox-surface.cjs<br/>agentbox/config/hooks/dream-inbox-surface.cjs:34
    participant INBOX as dream-inbox.json #43; .surfaced stamp<br/>DREAM_INBOX_PATH override, dream-inbox-surface.cjs:28-29
    participant M as Model context

    rect rgb(245,235,255)
    Note over CALLER,NPB: RESOLVED ADR-2068 #40;2026-09-05#41;: not a divergence — this is a CLI, not a hook.<br/>The management API spawns it from the /v1/projects publish route #40;routes/projects.js lines 28 and 322#41;.<br/>config/hooks/README.md section 3 names every file in config/hooks/ that is a CLI or helper rather than a hook
    CALLER->>PTP: node project-tracking-publish.cjs, optional ProjectTrackingDigest on stdin (:205,213-228)
    alt AGENTBOX_PROJECT_TRACKING_PUBLISH===0, or bridge secrets absent (:60-66,204-206)
        PTP-->>CALLER: return 0, silent no-op
    else
        alt stdin carries a valid digest #40;project_id present#41;
            PTP->>PTP: use verbatim (:218-219)
        else stdin is a TrackedProject or empty
            PTP->>MAPI: GET /v1/projects, Authorization Bearer MANAGEMENT_API_KEY (:135-146)
            MAPI-->>PTP: project list, mapped via toDigest#40;#41; (:92-110,228-233)
        end
        loop each digest
            PTP->>NPB: spawnSync nostr-pod-bridge track, digest JSON on stdin, t=30000 (:186-200)
            NPB-->>PTP: kind-30841 addressable event, d-tag=project slug, dual-write pod inbox
        end
        PTP-->>CALLER: log published N/total, return 0 (:241)
    end
    Note over PTP: guard setTimeout DEADLINE_MS=30000#43;1500 force-exits process (:248-249)
    end
    rect rgb(235,255,240)
    Note over DI: DRIFT resolved #40;ADR-2115#41;: no longer relays item text. The nightly engine now<br/>publishes each item as a forum governance case, this hook adds ONE pointer line<br/>with a count, dream-inbox-surface.cjs:2-10
    U->>DI: stdin JSON, parse prompt and skip MACHINE_TURN turns #40;task-notification,<br/>agent-message, system-reminder#41; so the pointer lands on a real user turn (:40-42)
    DI->>INBOX: JSON.parse#40;readFileSync#40;INBOX#41;#41;, count status===open only #40;:43-46#41;
    alt open===0
        DI-->>M: #123;#125; #40;no hookSpecificOutput#41; (:46,68-70)
    else
        DI->>INBOX: read STAMP #40;INBOX.surfaced#41;, rate-limit to once per RESURFACE_HOURS=4 (:48-51)
        alt within RESURFACE_HOURS of the last stamp
            DI-->>M: #123;#125; (:51,68-70)
        else
            DI-->>M: hookSpecificOutput.additionalContext = #91;DREAM#93; N decision#40;s#41; await you in<br/>the forum governance panel, DREAM_GOVERNANCE_URL (:52-56)
            DI->>INBOX: writeFileSync#40;STAMP, now#41; ONLY after the write callback confirms flush (:57-61)
        end
    end
    Note over DI: fail-open: any error #40;missing file, bad JSON#41; -#62; exit#40;#41; path returning #123;#125; (:62-64,68-70)
    end
```
## AB-08.13 trajectory-recorder.cjs — Stop/SubagentStop boundary
```mermaid
sequenceDiagram
    autonumber
    participant CC as Claude Code core<br/>Stop or SubagentStop
    participant TR as trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:577
    participant UT as trajectory-util.cjs<br/>agentbox/config/hooks/lib/trajectory-util.cjs:1
    participant STASH as os.tmpdir stash<br/>agentbox-traj-#60;sha12#62;.json<br/>trajectory-recorder.cjs:223-225
    participant PG as ruvector-postgres<br/>trajectories / trajectory_steps
    participant MAPI as management-api<br/>/v1/agent-events/emit on port 9090

    CC->>TR: node trajectory-recorder.cjs #60;event#62;, stdin JSON, RUVECTOR_* env inline (:577-578)
    Note over TR: ADR-2015: transcript-driven because PostToolUse does NOT fire<br/>for non-zero-exit Bash calls on this build (:22-26)
    alt NOT #40;RUVECTOR_MEMORY_LEARNING_ENABLED AND RUVECTOR_RECORD_TRAJECTORIES#41;
        TR-->>CC: return 0, byte-identical to no hook present #40;DEFAULT-OFF, :581-583#41;
    else event in #91;Stop, SubagentStop#93; (trajectory-recorder.cjs:592-593)
        TR->>STASH: readStash#40;session#41; - processedLines watermark #43; ctcPending queue (:405 and :227-233)
        TR->>UT: scanTranscript#40;lines, fromLine#41; grades each Bash tool_use by tool_result.is_error (:312-384)
        loop each Bash step found
            TR->>UT: redact#40;command#41; - I10 fail-closed, null command skips the step entirely (:362)
            TR->>UT: gradeResult#40;is_error,stderr,interrupted#41; -#62; success/failure + quality score (:356)
        end
        TR->>UT: Jev option, off unless AGENTBOX_JEV_FAILURE_TAXONOMY=1: jevFailureTags pre-pass refines only failures the taxonomy left unmapped, before any database connection (:121-129, call :438)
        Note over TR: lib/failure-jev.cjs bounds the pre-pass to AGENTBOX_JEV_FAILURE_MAX_CALLS requests, default 3, under one deadline hard-capped at 4000 ms (lib/failure-jev.cjs:23-25)<br/>a miss keeps the base tag, and failure_jev is written to the step result only when the judge answered (trajectory-recorder.cjs:478-481)
        alt pg module unavailable
            TR-->>CC: log, watermark NOT advanced, lines retried next Stop #40;ADR-2015 closeout, :429#41;
        else
            TR->>STASH: ctcEmits = stash.ctcPending FIFO-drained first, bound CTC_QUEUE_MAX=2000 (:443,179)
            TR->>PG: INSERT trajectories #40;id, task, agent, status, metadata#41; ON CONFLICT DO NOTHING (:459-465)
            loop each graded step
                TR->>PG: INSERT trajectory_steps #40;id=sha12#40;tool_use_id#41;, action, result jsonb, quality#41; ON CONFLICT DO NOTHING (:490-501)
                TR->>TR: push ctcEmitBodyFromStep#40;#41; onto ctcEmits while #60; CTC_QUEUE_MAX (:509-513)
            end
            TR->>PG: UPDATE trajectories SET ended_at, status=complete, metadata#124;#124;jsonb_build_object#40;..., ctc_emit_queued, ctc_emit_carried_in#41; (:525-539)
            TR->>STASH: watermark advances ONLY after successful persist (:544)
            TR->>MAPI: emitCtcStepsBestEffort#40;ctcEmits#41; POST /v1/agent-events/emit, CTC_EMIT_CAP=200/invocation (:186-196 and :555)
            MAPI-->>TR: #123;attempted, deferred#125; - surplus beyond the cap is NOT dropped here (:196)
            TR->>STASH: cur.ctcPending = deferred #40;bounded CTC_QUEUE_MAX#41;, cur.ctcPendingOverflow accumulates true drops (:565-567)
            Note over TR: overflow beyond CTC_QUEUE_MAX IS dropped and logged as INCOMPLETE (:561-563)<br/>this is the only stash write outside the persistence path (:556-558)
        end
    end
    Note over TR: guard setTimeout 8000ms force-exits process regardless of PG/HTTP state (:599)
    Note over TR: see AB-21 for the learning loop this feeds #40;ReasoningBank / SONA consumers of trajectories#41;
```
## AB-08.14 lib/ shared helpers — consumers
```mermaid
flowchart TD
    LIBDIR["agentbox/config/hooks/lib/<br/>hooks/README.md:71 documents three; hook-output.cjs, jev-option.cjs, failure-jev.cjs and ontology-proposal-jev.cjs are undocumented there"]
    UT["trajectory-util.cjs<br/>agentbox/config/hooks/lib/trajectory-util.cjs:1"]
    EG["egress-policy.cjs<br/>agentbox/config/hooks/lib/egress-policy.cjs:1<br/>JS half of the ADR-2026 content-egress policy :3-10"]
    HO["hook-output.cjs<br/>agentbox/config/hooks/lib/hook-output.cjs:1<br/>the ONE stdout shape Claude Code honours :3-19"]
    LIBDIR --> UT
    LIBDIR --> EG
    LIBDIR --> HO
    F1["sha12#40;s#41; trajectory-util.cjs:18<br/>content-addressed step/trajectory ids"]
    F2["commandPattern#40;command#41; trajectory-util.cjs:38<br/>+ hasResidualSecret#40;text#41; trajectory-util.cjs:180"]
    F3["redact#40;command#41; trajectory-util.cjs:196<br/>I10 fail-closed redaction gate"]
    F4["deriveOutcome#40;toolResponse#41; trajectory-util.cjs:227<br/>+ gradeResult#40;isError,stderr,interrupted#41; trajectory-util.cjs:285"]
    F5["tokenCountOf#40;usage#41; trajectory-util.cjs:309<br/>+ usageIdentityOf#40;rec#41; trajectory-util.cjs:341"]
    F6["handoffIdFrom#40;env,fallbackId#41; trajectory-util.cjs:367<br/>CTC chain-correlation id"]
    F7["ctcEmitBodyFromStep#40;step,opts#41; trajectory-util.cjs:392<br/>builds /v1/agent-events/emit body"]
    UT --> F1 --> F2 --> F3 --> F4 --> F5 --> F6 --> F7
    G1["OUTCOME skipped#124;attempted#124;accepted#124;failed<br/>egress-policy.cjs:29-34"]
    G2["recipientAllowed#40;recipient#41; :59-68<br/>64-hex grammar #43; mandatory non-empty AGENTBOX_MIRROR_RECIPIENTS allowlist :46-53"]
    G3["redactForEgress#40;text#41; :115-124<br/>null return means fail-closed SKIP"]
    G4["egressDecision#40;pathId, opts#41; :136-162<br/>global, per-path and redaction-disabled switches"]
    EG --> G1 & G2 & G3 & G4
    H1["contextPayload#40;#41; :22-26 — the ONLY honoured shape:<br/>hookSpecificOutput.hookEventName/.additionalContext"]
    H2["emitContext#40;#41; :32-44 — resolves true only once the write flushes"]
    HO --> H1 & H2
    TR["trajectory-recorder.cjs<br/>agentbox/config/hooks/trajectory-recorder.cjs:47<br/>ONLY consumer of trajectory-util.cjs"]
    NM["nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs:46<br/>ONLY consumer of egress-policy.cjs"]
    RBG["ruvnet-brain-ground.cjs:123<br/>+ hooks/skill-route.cjs:27 — BOTH consumers of hook-output.cjs"]
    F7 --> TR
    G1 & G2 & G3 & G4 --> NM
    H1 & H2 --> RBG
    PAIR["INVARIANT ADR-2026: the Rust twin services/nostr-pod-bridge/src/egress_policy.rs<br/>must satisfy the SAME fixture tests/fixtures/egress-redaction.v1.json<br/>egress-policy.cjs:6-10 — a shared comment is not a shared contract"]
    EG -.-> PAIR
    OTHER["the remaining hooks import no lib/ module — each is self-contained:<br/>claude-flow-hook-adapter, trust-seed #40;now a boot-time script, see AB-08.10#41;,<br/>ontology-monitor, dream-inbox-surface #40;composes its own JSON, does not<br/>call hook-output.cjs#41;, project-tracking-publish, fleet-session-start<br/>and fleet-tab-name"]
    LIBDIR -.-> OTHER
```

Session-mirror egress: see AB-13.9.

## AB-08.15 A third registration mechanism: the factrail function-hook plugin (ADR-2093, implemented by ADR-2121)
```mermaid
sequenceDiagram
    autonumber
    participant NIX as factrailPkg<br/>lib/factrail.nix:36
    participant EP as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:3010
    participant MIG as migrate<br/>scripts/factrail-store-migrate.mjs:74
    participant CC as Claude Code engine
    participant SHIM as factrail shim plus binary
    participant J as System One judge

    NIX->>NIX: build DreamLab-AI/factrail at one pinned 40-char rev, vendored lockfile,<br/>whole-workspace tests in the build (lib/factrail.nix:36, :52, :60)
    NIX->>EP: flake.nix links /opt/agentbox/bin/factrail and bakes the shim as<br/>config/claude-plugins/factrail, both only when the gate is on (flake.nix:2115, flake.nix:2196)
    EP->>CC: set or clear CLAUDE_CODE_ENABLE_FUNCTION_HOOKS in settings.json env (:3045-3046)
    EP->>CC: uninstall any retired jev-compaction@agentbox WHATEVER the gate (:3052-3059)
    Note over EP,CC: two compaction plugins on one session.compact chain would both act,<br/>ADR-2121-factrail-implements-jev-compaction.md:96-97
    EP->>MIG: gate on and plugin plus binary baked (:3061), run the store migration (:3069)
    MIG->>MIG: copy every tainted taint-session record and an unset switch position from<br/>jev-compaction's store into factrail's, rename the old file .migrated (factrail-store-migrate.mjs:86-104)
    EP->>EP: project each manifest pair only if the baked plugin declares it, skip empty values<br/>_jc_project (:3087), binary = the stable /opt path, never a store path (:3039)
    EP->>CC: claude plugin install factrail@agentbox with that userConfig (:3142)
    Note over EP,CC: reinstall on a change of plugin code, userConfig OR the binary's own hash<br/>(:3131, :3134), gate off uninstalls and drops the marketplace (:3150-3165),<br/>ADR-2020 byte-identical-when-off
    CC->>SHIM: load the shim, which registers seven events and delegates every decision<br/>to the binary over a versioned stdin-stdout protocol<br/>(ADR-2121-factrail-implements-jev-compaction.md:55-60, README.md:83)
    Note over EP,CC: ADR-2135 a SECOND plugin joins the same marketplace and function-hook switch: jev-checkpoints@agentbox, installed under [features.jev_checkpoints].enabled (entrypoint-unified.sh:3168-3256),<br/>its cache compared to the baked tree by content digest (:3191) and its projected --config words stamped (:3184), so a changed manifest value reinstalls, eight events (README.md:84)<br/>the shared marketplace is dropped only when factrail, ruflo-console and jev-checkpoints are all off (entrypoint-unified.sh:3160-3164)
    CC->>SHIM: tool.call and skill.prompt mark a session STICKY-tainted at the source
    CC->>SHIM: session.compact with the transcript
    alt tainted and the judge is the cloud one
        SHIM-->>CC: compacted on LOCAL fact rails, nothing sent (fallback rules, agentbox.toml:784)
    else judged
        SHIM->>J: text, capped tool inputs and only ok-or-error plus a size per result
        J-->>SHIM: keep the call, keep its result verbatim
        SHIM-->>CC: what Jev lets go is REDUCED, not deleted - head, fact lines and tail kept,<br/>full output saved with the path in the note (agentbox.toml:704-708)
    end
```

**What it shows.** A plugin is neither a shell hook in `config/hooks/` nor an MCP server: the engine loads the shim in process, the shim hands every decision to a Rust binary, and nothing in the hooks directory registers either. Since ADR-2121 (2026-10-02) both halves come from one pinned factrail commit baked into the image, and the vendored `jev-compaction` plugin is deleted and actively uninstalled.
**Why it is this way.** ADR-2093 wanted compaction to drop and truncate tool calls rather than rewrite text, which the `settings.json` hook contract cannot express (`../project/agentbox/config/hooks/README.md:73-78`); ADR-2121 kept that policy and replaced the implementation, because fact rails kept 50 of 50 pre-registered facts where the deleting plugin kept 4 (`../project/agentbox/docs/adr/ADR-2121-factrail-implements-jev-compaction.md:42-45`).

**Drift (resolved 2026-10-02):** `hooks/README.md` listed four plugin events where the plugin registered seven; it now lists all seven for `factrail` (`../project/agentbox/config/hooks/README.md:83`).

**Invariant:** the sticky email taint survives the change of plugin — the boot migration only widens taint and never overwrites factrail's own switch (`../project/agentbox/scripts/factrail-store-migrate.mjs:86-94`), and it runs before the install (`../project/agentbox/config/entrypoint-unified.sh:3069`).

**Invariant:** the shim and the binary can never disagree on the hook protocol, because one `rev` supplies both and the plugin's `binary` option is the stable `/opt/agentbox/bin/factrail` path (`../project/agentbox/lib/factrail.nix:36`, `../project/agentbox/config/entrypoint-unified.sh:3039`).

**Open:** the shim's event handlers and the binary's fact-rail logic live in the factrail repository, which this corpus does not map, so no claim about them here is checked at a revision; the events and the fence are cited from the agentbox README and ADR-2121 only (`../project/agentbox/docs/adr/ADR-2121-factrail-implements-jev-compaction.md:55-60`).

**Open:** ADR-2121 is `activation_status: staged` — no image with factrail has yet been built and booted, and the residency soak that gates `live` has not run (`../project/agentbox/docs/adr/ADR-2121-factrail-implements-jev-compaction.md:152-154`).

**Debt:** three registration mechanisms (plus four gated additions, AB-08.16) now decide what runs on a session boundary (the entrypoint's root `settings.json`, `stacks.rs` per profile, and the plugin marketplace), and only the README relates them (`../project/agentbox/config/hooks/README.md:53-63`).

## AB-08.16 Four gated additions to both registration sites — decline-recorder, voice-paned record, session-state, activity-sink
```mermaid
flowchart TB
    subgraph ROOT["root session: entrypoint-unified.sh rewrites ~/.claude/settings.json hooks every boot"]
    direction TB
    R1["decline-recorder.cjs on Stop, SubagentStop, UserPromptSubmit, timeout 10<br/>gate [claude_code].floor, default on<br/>entrypoint-unified.sh:2115-2138, de-registered when off :2142-2155"]
    R2["voice-paned record --source claude on Stop and Notification, timeout 5<br/>gate [voicelink].enabled AND record_events<br/>entrypoint-unified.sh:2168-2196, canonical-entry reconcile :2185-2194"]
    R4["session-state.cjs on UserPromptSubmit, Stop, Notification, timeout 3 (ADR-2153)<br/>gate [interaction_plane].enabled, retracted when off, plus Codex notify :1921-1940<br/>entrypoint-unified.sh:1944-1956, push :1952"]
    R3["activity-sink.cjs on PostToolUse file tools, UserPromptSubmit, Stop, timeout 3<br/>gate [integrations.activity_sink].enabled with an http(s) url<br/>entrypoint-unified.sh:2853-2935, strip then push :2908-2928"]
    end
    subgraph PROF["per-profile stack: stacks.rs build_profile rewrites the whole profile settings.json"]
    direction TB
    P1["floor entries appended when gates.floor<br/>stacks.rs:293-308, WORKSPACE inlined by sh_quote :82-84"]
    P2["voicelink entries appended when gates.voicelink_record<br/>stacks.rs:311-323, constants :180-185"]
    P4["session-state entries appended when gates.session_plane, timeout 3, no WORKSPACE inlined<br/>stacks.rs:324-334"]
    P3["add_activity_sink_hooks when gates.activity_sink is Some<br/>stacks.rs:95-138, call :335-337, profile name becomes the actor label :99-104"]
    end
    subgraph GATE["gate derivation, stacks_env.rs"]
    direction TB
    G1["floor = claude_code.floor, default true<br/>stacks_env.rs:164"]
    G2["voicelink_record = voicelink.enabled AND voicelink.record_events default true<br/>stacks_env.rs:165-166"]
    G4["session_plane = interaction_plane.enabled, default false<br/>stacks_env.rs:163"]
    G3["activity_sink = ActivitySink::from_manifest, None unless enabled AND usable<br/>stacks_env.rs:188"]
    end
    subgraph MAN["agentbox.toml as shipped"]
    direction TB
    M1["claude_code floor = true, agentbox.toml:740"]
    M2["voicelink enabled = true agentbox.toml:2223, record_events = true :2230"]
    M4["interaction_plane enabled = true, agentbox.toml:2079"]
    M3["integrations.activity_sink enabled = false, url empty, agentbox.toml:529"]
    end
    MAN --> GATE --> PROF
    GATE --> ROOT
    NOTE1["INVARIANT: decline-recorder never blocks a turn, it exits 0 with nothing on stdout<br/>decline-recorder.cjs:42-44"]
    NOTE2["INVARIANT: activity-sink prints nothing on stdout, exits 0 always, under a hard deadline<br/>activity-sink.cjs:43-45, :327"]
    NOTE3["TENSION: ADR-2151 is decision_status proposed, implementation none, activation inactive<br/>ADR-2151-voicelink-edge-voice.md:5-7, yet voicelink enabled = true registers hooks on every boot<br/>agentbox.toml:2223"]
    NOTE4["OPEN: ADR-2146 is decision_status proposed with implementation complete, activation inactive<br/>ADR-2146-activity-sink-hook.md:5-7, and the shipped manifest leaves the gate off agentbox.toml:529"]
    ROOT --> NOTE1
    PROF --> NOTE2
    ROOT --> NOTE3
    ROOT --> NOTE4
```

**What it shows.** Since the last stamp four more handlers ride the same two registration sites as the learning hooks: the floor's `decline-recorder.cjs` (ADR-2144, live), `voice-paned record` (ADR-2151), `session-state.cjs` (ADR-2153, live, status marks for the session plane, silent and fail-open, `../project/agentbox/config/hooks/session-state.cjs:3`, `../project/agentbox/config/hooks/session-state.cjs:13-20`) and `activity-sink.cjs` (ADR-2146). Each is written twice, once as inline Node in the entrypoint for the root session and once in Rust for every wrapper profile, and each is stripped when its gate is off so a container without the gate is byte-identical.

**Why it is this way.** Isolated profiles replace `CLAUDE_CONFIG_DIR`, so root settings are not inherited and the floor's recording promise has to be repeated per profile (`../project/agentbox/services/agentbox-manifest/src/stacks.rs:291-292`); the profile file is rewritten whole on every projection, which is why a stale command self-heals there without a reconcile step (`../project/agentbox/services/agentbox-manifest/src/stacks.rs:309-310`). The token value of the activity sink is never written into `settings.json`, only the file path or the environment variable name (`../project/agentbox/config/entrypoint-unified.sh:2858-2859`).

**Tension (ADR-2151 vs manifest):** ADR-2151 is `decision_status: proposed`, `implementation_status: none` and `activation_status: inactive` (`../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md:5-7`) while `[voicelink] enabled = true` and `record_events = true` register the recorder hooks on every boot (`../project/agentbox/agentbox.toml:2223`, `../project/agentbox/agentbox.toml:2230`).

**Open:** ADR-2146 is `decision_status: proposed` with `implementation_status: complete` and `activation_status: inactive` (`../project/agentbox/docs/adr/ADR-2146-activity-sink-hook.md:5-7`); is the sink meant to ship enabled anywhere, given the manifest leaves it off with an empty `url` (`../project/agentbox/agentbox.toml:529`)?

**Debt:** each of the four registrations exists as two hand-kept implementations, entrypoint JavaScript and `stacks.rs`, and the entrypoint comment names the Rust file as the "per-profile twin" (`../project/agentbox/config/entrypoint-unified.sh:2159-2167`).

**Invariant:** the floor gate off leaves no decline-recorder entry in either site, because the entrypoint filters existing entries out and the profile projection appends only `if gates.floor` (`../project/agentbox/config/entrypoint-unified.sh:2142-2155`, `../project/agentbox/services/agentbox-manifest/src/stacks.rs:293`).
