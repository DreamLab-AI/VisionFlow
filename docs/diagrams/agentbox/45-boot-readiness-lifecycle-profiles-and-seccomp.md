---
id: AB-45
title: Boot readiness, program lifecycle, tmux layout, profile isolation and seccomp
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2007, ADR-2029, ADR-2144, ADR-2146, ADR-2151, ADR-2153]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/flake.nix
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/config/tmux-autostart.sh
  - ../project/agentbox/config/harness-wrappers/zai.sh
  - ../project/agentbox/config/harness-wrappers/openrouter.sh
  - ../project/agentbox/config/harness-wrappers/_provider-url.sh
  - ../project/agentbox/services/agentbox-manifest/src/stacks.rs
  - ../project/agentbox/services/agentbox-sessions/src/contract.rs
  - ../project/agentbox/scripts/session-provision.mjs
  - ../project/agentbox/config/session-plane/launch.json
  - ../project/agentbox/config/harness-wrappers/router.sh
  - ../project/agentbox/config/seccomp-agentbox.json
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/scripts/ci/check-seccomp.sh
  - ../project/agentbox/docs/adr/ADR-2007-profile-isolation.md
  - ../project/agentbox/agentbox.toml
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
aliases:
  AB-02.9: AB-45.1
  AB-02.10: AB-45.2
  AB-02.11: AB-45.3
  AB-02.12: AB-45.4
  AB-02.13: AB-45.5
  AB-02.14: AB-45.6
  AB-02.16: AB-45.7
---
Split from AB-02 on 2026-11-23: what the booted container reports and how its programs and profiles behave (formerly AB-02.9 to AB-02.14 and AB-02.16). Boot phases and the supervision tree stay in AB-02. Old panel IDs resolve through the `aliases:` map above; the stamp is AB-02's, and the panels were not re-read.

## AB-45.1 readiness — /ready vs /health
```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator/probe
    participant R as management-api routes<br/>server.js:473 /ready, server.js:601 /health
    participant BS as bootstrapState<br/>server.js:58-66 fs watch of BOOTSTRAP_SENTINEL
    participant ML as adapters/manifest-loader<br/>server.js:512
    participant AH as adapterHealth map<br/>server.js:519
    participant FS as fs.promises.access<br/>server.js:536-542

    O->>R: GET /ready
    R->>BS: read bootstrapState.completed (poll every 2s of /run/agentbox/bootstrap.done, server.js:77)
    alt bootstrap.done sentinel absent
        R->>R: missing.push('bootstrap.done sentinel') (server.js:506)
    end
    R->>ML: loadManifest() (server.js:512)
    Note over R: 2. Adapter health — every non-off slot must be healthy (server.js:509)
    loop for slot, impl in manifestAdapters (server.js:517)
        alt impl === 'off'
            R->>R: continue (server.js:518)
        else adapterHealth[slot] !== 'healthy' (server.js:519)
            R->>R: missing.push adapter:<slot> not healthy (server.js:520)
        end
    end
    Note over R: 3. Required filesystem paths (server.js:524)
    R->>FS: access WORKSPACE, /var/lib/ruvector (server.js:529)
    opt manifestAdapters.pods === local-solid-rs (server.js:531)
        R->>R: add integrations.solid_pod_rs.storage_root to requiredPaths (server.js:531-534)
    end
    opt sovereign_mesh.publish_agent_events === true (server.js:548)
        alt NOSTR_RELAYS env empty (server.js:551)
            R->>R: missing.push publish_agent_events but NOSTR_RELAYS empty (server.js:552)
        end
    end
    alt missing.length greater than 0 (server.js:578)
        R-->>O: 503 ready:false reason, missing[], degraded[], components (server.js:579-585)
    else
        R-->>O: 200 ready:true since bootstrapState.since, degraded[], components such as elevation_publisher (server.js:589-595, server.js:568-573)
    end

    rect rgb(240,240,240)
    Note over O,R: /health (server.js:601-639) is human-inspection-only —<br/>note field at server.js:637 says "Use /ready for orchestrator readiness probes"
    O->>R: GET /health
    R-->>O: 200 status ok/degraded, uptime, adapters map, adapter_failures and adapter_failure_files<br/>(a failure class and a file path per slot, never error text), degraded_count (server.js:628-638)
    end
```

## AB-45.2 supervisord program lifecycle
```mermaid
stateDiagram-v2
    [*] --> Stopped
    Stopped --> Starting: autostart=true (every program block)
    Starting --> Backoff: process exits before startsecs elapses
    Backoff --> Starting: retry, up to default startretries=3 (unset in flake.nix)
    Backoff --> Fatal: startretries exceeded
    Starting --> Running: survives startsecs window
    Running --> Exited: autorestart=false one-shot completes
    Running --> Starting: autorestart=true and unexpected exit
    Running --> Stopping: supervisorctl stop or shutdown
    Stopping --> Stopped: SIGTERM handled within stopwaitsecs
    Fatal --> [*]
    Exited --> [*]

    note right of Exited
        one-shot class autorestart=false
        program bootstrap flake.nix:2878, startsecs=0 at flake.nix:2883
        program bootstrap-seal flake.nix:2908, startsecs=0 at flake.nix:2913, timeout 120s at flake.nix:2915
        program tailscale-up flake.nix:3029, startsecs=0 at flake.nix:3035
        program tmux-autostart flake.nix:3209, startsecs=0 at flake.nix:3215
    end note
    note right of Running
        DOC-DRIFT most programs set no startretries in flake.nix
        and rely on the supervisord built-in default startretries=3, autorestart=true
        e.g. management-api flake.nix:2892, nostr-relay flake.nix:2976 and flake.nix:2988
        exceptions agentbox-mcp-hub startretries=2 at flake.nix:3571,
        system-one-proxy startretries=30 at flake.nix:3291
        and sidestr-producer startretries=5 at flake.nix:3381, a pin mismatch stops at FATAL
    end note
    note left of Starting
        startsecs varies by program
        agentbox-sessions flake.nix:3153, startsecs=5 at flake.nix:3160
        nip98-proxy flake.nix:3166, startsecs=3 at flake.nix:3173
        code-server flake.nix:3092, startsecs=5 at flake.nix:3110
        xwayland-session flake.nix:2785, startsecs=5 at flake.nix:2791
    end note
```

## AB-45.3 tmux-autostart window layout
```mermaid
flowchart TB
    START["program:tmux-autostart<br/>flake.nix:3209, priority=95<br/>runs config/tmux-autostart.sh"]
    ENV["fish conf.d/agentbox-runtime.fish<br/>entrypoint-unified.sh:4252-4271<br/>sources RUNTIME_ENV_FILE, fallback to durable copy<br/>every new tmux window's fish shell sources it on start"]
    W0["window 0 Claude<br/>tmux-autostart.sh:291<br/>tab0-bridge injection target<br/>CLAUDE_CONFIG_DIR=/home/devuser/.claude (tmux-autostart.sh:292), AGENTBOX_PROFILE=tab0 (tmux-autostart.sh:294)"]
    W1["window 1 Agent<br/>tmux-autostart.sh:317<br/>agent execution workspace"]
    W2["window 2 Services<br/>tmux-autostart.sh:324<br/>supervisorctl status (tmux-autostart.sh:325)"]
    W3["window 3 Build<br/>tmux-autostart.sh:330"]
    W4["window 4 Logs<br/>tmux-autostart.sh:338<br/>supervisorctl tail -f management-api (tmux-autostart.sh:339), split pane"]
    W5["window 5 System<br/>tmux-autostart.sh:345<br/>systemscape restart loop (tmux-autostart.sh:374) after waiting for a mounted hosts file (tmux-autostart.sh:358-369), split with btm/htop"]
    W6["window 6 VNC<br/>tmux-autostart.sh:387<br/>host shell, X display one, port 5901 status"]
    W7["window 7 Git<br/>tmux-autostart.sh:400<br/>git status in PROJECT dir"]
    W8["window 8 Sessions<br/>tmux-autostart.sh:408<br/>agentbox-sessions pick (tmux-autostart.sh:409-410) or a disabled notice (tmux-autostart.sh:412),<br/>attach lands here when the binary exists (tmux-autostart.sh:465-466)"]
    W9["window 9 Notes<br/>tmux-autostart.sh:102 _notes_window, called tmux-autostart.sh:421<br/>ADR-2029 Rune markdown TUI over vault"]

    START --> W0 --> W1 --> W2 --> W3 --> W4 --> W5 --> W6 --> W7 --> W8 --> W9

    G1{"AGENTBOX_VAULT_ENABLED<br/>tmux-autostart.sh:106,124"}
    G2{"VAULT_TUI == rune ?<br/>tmux-autostart.sh:104-105,136"}
    G3{"a rune that RUNS found?<br/>tmux-autostart.sh:163-168<br/>PATH then WORKSPACE/.cargo/bin,<br/>first to pass --version wins"}
    G4{"WORKSPACE/.rune-home<br/>writable? tmux-autostart.sh:190-201"}
    LAUNCH["env HOME=rune_home rune -w cwd<br/>tmux-autostart.sh:222"]

    ENV -.->|"AGENTBOX_VAULT_ENABLED, VAULT_TUI inherited"| W9
    W9 --> G1
    G1 -->|"0 vault disabled"| REFUSE1["refuse: no vault in agentbox.toml<br/>tmux-autostart.sh:124-132"]
    G1 -->|"enabled default 1"| G2
    G2 -->|"not rune e.g. none"| REFUSE2["refuse: VAULT_TUI execution off-switch<br/>tmux-autostart.sh:136-150<br/>even if a rune binary is present"]
    G2 -->|"rune"| G3
    G3 -->|"absent"| REFUSE3["refuse: rebuild or cargo install<br/>tmux-autostart.sh:175-187<br/>DEBT: the hint installs aka-rider/rune v1.4.0 (tmux-autostart.sh:183),<br/>not the DreamLab fork the image bakes"]
    G3 -->|"present"| G4
    G3 -.->|"candidates that fail --version"| STALE["reported as stale loaders, skipped<br/>tmux-autostart.sh:170-173"]
    G4 -->|"not writable"| REFUSE4["refuse: not launching degraded<br/>tmux-autostart.sh:201-217"]
    G4 -->|"writable"| LAUNCH
```

## AB-45.4 profile isolation — per-profile HOME and CLAUDE_CONFIG_DIR (ADR-2007)
```mermaid
sequenceDiagram
    autonumber
    participant E as entrypoint-unified.sh<br/>Phase 4 entrypoint-unified.sh:970-971
    participant AM as agentbox-manifest provision-stacks<br/>services/agentbox-manifest/src/stacks.rs:187 build_profile
    participant FS as WORKSPACE/profiles/STACK
    participant SEED as session-provision.mjs<br/>scripts/session-provision.mjs:203
    participant WRAP as harness wrapper<br/>config/harness-wrappers/zai.sh|openrouter.sh

    E->>AM: agentbox-manifest provision-stacks, runs as root Phase 4 (entrypoint-unified.sh:971)
    loop for each stack in STACKS_JSON (stacks.rs:374-375)
        AM->>FS: build_profile writes root=WORKSPACE/profiles/STACK (stacks.rs:188-189)
        AM->>FS: symlink profiles/STACK/projects and /workspace (stacks.rs:199-200)
        AM->>FS: write .env with AGENT_STACK=STACK (stacks.rs:191-201)
        opt Claude-hosted profile
            AM->>FS: hooks start from learning_hooks (stacks.rs:290), then add the decline-recorder entries on Stop, SubagentStop and UserPromptSubmit when the floor gate is on (stacks.rs:293-308, ADR-2144)
            AM->>FS: add voice-paned record on Stop and Notification when voicelink record_events is on (stacks.rs:311-326, ADR-2151)
            AM->>FS: add session-state.cjs on UserPromptSubmit, Stop and Notification when the session plane is on (stacks.rs:327-337, ADR-2153)
            AM->>FS: add the activity-sink hooks when [integrations.activity_sink] is usable (stacks.rs:338-340, ADR-2146)
            AM->>FS: write .claude/settings.json with the assembled hooks (stacks.rs:342-360)
        end
    end
    Note over E: chown -R 1000:1000 WORKSPACE/profiles after provision-stacks (entrypoint-unified.sh:980)
    SEED->>FS: provision profiles/openrouter/.claude/settings.local.json ANTHROPIC_BASE_URL/AUTH_TOKEN (session-provision.mjs:16-34, run by the daemon at AB-46.3)
    SEED->>FS: provision profiles/zai/.claude/settings.local.json (session-provision.mjs:36-53)
    Note right of SEED: ADR-043 D4.1 — a distinct AGENTBOX_PROFILE per session<br/>yields a distinct persisted did:nostr identity
    Note over WRAP: at session launch the wrapper pins HOME=PROFILE and<br/>CLAUDE_CONFIG_DIR=PROFILE/.claude (see AB-45.5)
    Note over E,WRAP: DIVERGENCE — profile isolation routes configuration under<br/>ONE OS user devuser — harnesses are isolated by directory, not by OS user<br/>(docs/adr/ADR-2007-profile-isolation.md:59), not an OS access boundary
```

## AB-45.5 harness wrapper invocation — Z.AI / OpenRouter redirect assertion
```mermaid
sequenceDiagram
    autonumber
    participant SP as agentbox-sessions tmux session<br/>services/agentbox-sessions/src/contract.rs:83 wrapper kind
    participant W as zai.sh / openrouter.sh<br/>config/harness-wrappers/zai.sh:1
    participant PV as provider_url_validate<br/>config/harness-wrappers/_provider-url.sh:71-72
    participant SET as settings.local.json<br/>WORKSPACE/profiles/SLUG/.claude
    participant C as claude binary

    SP->>W: argv is the wrapper path alone (config/session-plane/launch.json:16), exec zai.sh (SLUG=zai EXPECT_HOST=z.ai) or openrouter.sh (EXPECT_HOST=openrouter.ai, openrouter.sh:26-28)
    W->>SET: check PROFILE dir and SETTINGS file exist (zai.sh:92-100)
    alt profile or settings.local.json missing
        W-->>SP: _die fatal, exit 1 (zai.sh:41-52,92-100)
    end
    W->>SET: _json_env_field reads ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN (zai.sh:103-104)
    alt BASE_URL or AUTH_TOKEN empty
        W-->>SP: _die fatal — redirect not provisioned (zai.sh:106-112)
    end
    W->>PV: provider_url_validate BASE_URL EXPECT_HOST PROVIDER_URL_ALLOWED_PORTS=443 (zai.sh:120, _provider-url.sh:72)
    Note over PV: ADR-2007 closeout 2026-09-05 — full authority parse:<br/>scheme must be https, user-info rejected, host must equal<br/>EXPECT_HOST or a dot-suffixed subdomain, port in allow-list (443)
    alt validation fails (wrong host, http scheme, userinfo spoof, bad port)
        PV-->>W: PROVIDER_URL_DIAG diagnostic, return 1 (_provider-url.sh:83-84)
        W-->>SP: _die — hard-fail loud, would mis-bill direct-Anthropic key (zai.sh:121-128)
    else validated
        W->>W: unset the caller's CLAUDE_*, ANTHROPIC_* and MAX_THINKING_TOKENS, keep CLAUDE_FLOW_* (zai.sh:139-144)<br/>then export HOME=PROFILE, CLAUDE_CONFIG_DIR=PROFILE/.claude (zai.sh:148-149)
        W->>W: export ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY="" (zai.sh:150-152)
        W->>W: export AGENTBOX_PROFILE default SLUG (zai.sh:155)
        opt W == openrouter.sh (openrouter.sh:141-156)
            W->>SET: read "model" from settings.local.json, export as ANTHROPIC_MODEL<br/>and every tier alias, caller value wins (openrouter.sh:141-150)
            Note right of W: RESOLVED 2026-09-26 (ADR-2007/ADR-2111 re-verification) —<br/>settings.local.json is a project-level file CLAUDE_CONFIG_DIR does not read,<br/>so the model pin sat unread and OpenRouter billed Opus instead of the<br/>configured model — also pins CLAUDE_CODE_PROMPT_CACHE_TTL=1h (openrouter.sh:156)
        end
        W-->>SP: echo credential-free confirmation line (zai.sh:159)
        W->>C: exec claude "$@" (zai.sh:160)
    end
    Note over W: DOC-DRIFT (confirmed still open) — agentbox/docs/BASELINE-container.md:342<br/>still says ADR-2007 is partial and the wrapper host assertion is substring-based —<br/>code (zai.sh:114-128, _provider-url.sh) has since implemented full URL parsing,<br/>closed 2026-09-05 per the code's own ADR-2007 closeout comments
```

## AB-45.6 VAULT env propagation — resolve to PID1 to supervised programs
```mermaid
sequenceDiagram
    autonumber
    participant TOML as agentbox.toml vault section<br/>entrypoint-unified.sh:69-89
    participant VR as _ab_vault_resolve<br/>entrypoint-unified.sh:89
    participant PID1 as supervisord PID1 env<br/>entrypoint-unified.sh:1299 exec
    participant PROG as supervised programs<br/>flake.nix program blocks

    VR->>TOML: _ab_toml_val vault root (entrypoint-unified.sh:90), pages/format/tui (entrypoint-unified.sh:102-106), working/transcripts (entrypoint-unified.sh:111-112),<br/>loom_sync/loom_url/loom_host (entrypoint-unified.sh:114-116), repo (entrypoint-unified.sh:124)
    alt vault section absent, VAULT_ROOT empty (entrypoint-unified.sh:91)
        VR-->>VR: echo "[vault] disabled — no [vault] in agentbox.toml" (entrypoint-unified.sh:95), return (entrypoint-unified.sh:96)
        Note right of VR: fail-loud absent-vault branch — every consumer<br/>disables itself rather than indexing a stale tree
    else VAULT_ROOT resolved (entrypoint-unified.sh:98)
        VR->>VR: VAULT_PAGES = root, or root/pages when [vault].pages names a folder (entrypoint-unified.sh:102-104)
        VR->>VR: VAULT_REPO from [vault].repo, else derived from VAULT_ROOT<br/>(knowledge/working to parent, else root itself) (entrypoint-unified.sh:124-130)
        alt VAULT_REPO/ontology/vocabulary.yaml absent (entrypoint-unified.sh:131)
            VR-->>VR: warn, VAULT_REPO="" (governed vault writes then refuse) (entrypoint-unified.sh:132-133)
        end
        VR->>VR: export VAULT_ROOT/VAULT_REPO/VAULT_PAGES/VAULT_FORMAT/VAULT_TUI/<br/>VAULT_WORKING_ROOT/VAULT_WORKING_PAGES/VAULT_TRANSCRIPTS/HP_LOOM_SYNC/LOOM_URL/LOOM_HP_HOST (entrypoint-unified.sh:135-136)
        Note right of VR: RESOLVED 2026-10-31 — the explicit ONTOLOGY_PAGES_DIR override and its<br/>AGENTBOX_VAULT_LEGACY_PATHS opt-in are removed, VAULT_PAGES is the only path<br/>authority whether the vault is on or off (entrypoint-unified.sh:77-84)
    end
    VR->>PID1: exec supervisord inherits VAULT_ROOT/REPO/PAGES/FORMAT/TUI/WORKING/TRANSCRIPTS (entrypoint-unified.sh:1299)
    PID1->>PROG: every program child inherits PID1 env at spawn, no VAULT_* set per-program in flake.nix
    Note over PID1,PROG: exceptions since custody W0 and W3 - docker-read-proxy and serve-identity<br/>start through env -i and see only the variables named on their command line<br/>(flake.nix:3052, flake.nix:3077)
```

## AB-45.7 seccomp profile — supplemental denylist
```mermaid
flowchart TB
    COMPOSE["docker-compose.yml:127-129<br/>security_opt no-new-privileges:true<br/>seccomp=./config/seccomp-agentbox.json"]
    PROFILE["seccomp-agentbox.json<br/>config/seccomp-agentbox.json<br/>defaultAction SCMP_ACT_ALLOW"]
    SOCK["rule 1: socket syscall<br/>args index0 value38 SCMP_CMP_EQ<br/>action SCMP_ACT_ERRNO"]
    DENY["rule 2: 46 named syscalls<br/>action SCMP_ACT_ERRNO"]
    CI["scripts/ci/check-seccomp.sh<br/>asserts defaultAction ALLOW and<br/>the 46-syscall denylist is not dropped"]

    COMPOSE --> PROFILE
    PROFILE --> SOCK
    PROFILE --> DENY
    SOCK -.->|"CVE-2026-31431 AF_ALG(38) algif_aead splice() privesc"| SOCKNOTE["blocks AF_ALG socket() only,<br/>every other socket family still ALLOWed"]
    DENY -.->|"kernel-module and namespace-escape surface"| DENYLIST["mount, umount2, pivot_root, setns, unshare,<br/>ptrace, bpf, init_module, kexec_load, reboot,<br/>swapon/off, ustat, vm86, userfaultfd, keyctl ... 46 total"]
    CI -->|"CI gate on every PR"| PROFILE
```
