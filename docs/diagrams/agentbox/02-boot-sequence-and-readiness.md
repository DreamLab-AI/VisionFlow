---
id: AB-02
title: Boot sequence, supervision tree and readiness
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2003, ADR-2007, ADR-2028, ADR-2029, ADR-2034, ADR-2063, ADR-2080, ADR-2092, ADR-2104, ADR-2122, ADR-2140, ADR-2144, ADR-2146, ADR-2151, ADR-2153]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/config/lib/role-custody.sh
  - ../project/agentbox/config/docker-read-proxy.cjs
  - ../project/agentbox/config/role-accounts.json
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/config/tmux-autostart.sh
  - ../project/agentbox/config/hooks/trust-seed.cjs
  - ../project/agentbox/config/seal-bootstrap.sh
  - ../project/agentbox/config/harness-wrappers/zai.sh
  - ../project/agentbox/config/harness-wrappers/openrouter.sh
  - ../project/agentbox/config/harness-wrappers/_provider-url.sh
  - ../project/agentbox/services/agentbox-manifest/src/stacks.rs
  - ../project/agentbox/services/agentbox-sessions/src/main.rs
  - ../project/agentbox/services/agentbox-sessions/src/plane.rs
  - ../project/agentbox/services/agentbox-sessions/src/reconcile.rs
  - ../project/agentbox/services/agentbox-sessions/src/worktree.rs
  - ../project/agentbox/services/agentbox-sessions/src/trust.rs
  - ../project/agentbox/services/agentbox-sessions/src/tmux.rs
  - ../project/agentbox/services/agentbox-sessions/src/status.rs
  - ../project/agentbox/services/agentbox-sessions/src/contract.rs
  - ../project/agentbox/scripts/session-provision.mjs
  - ../project/agentbox/config/session-plane/launch.json
  - ../project/agentbox/config/harness-wrappers/router.sh
  - ../project/agentbox/tests/cli/opencode-loom.test.mjs
  - ../project/agentbox/services/agentbox-mcp/src/hub/mod.rs
  - ../project/agentbox/services/agentbox-mcp/src/main.rs
  - ../project/agentbox/config/seccomp-agentbox.json
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/scripts/ci/check-seccomp.sh
  - ../project/agentbox/docs/adr/ADR-2007-profile-isolation.md
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/mcp/servers/lib/ontology-index-build.js
  - ../project/agentbox/mcp/servers/ruvector-mcp.cjs
  - ../project/agentbox/scripts/reconcile-skills.sh
  - ../project/agentbox/scripts/project-skill-roots.mjs
  - ../project/agentbox/scripts/reconcile-agents.sh
  - ../project/agentbox/scripts/reconcile-commands.sh
  - ../project/agentbox/agents/registered-agents.txt
  - ../project/agentbox/services/agentbox-manifest/src/mcp.rs
  - ../project/agentbox/config/registered-commands.txt
  - ../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
verified_commit: 0b0f7918961e629280faf86f88227535abf514b5
---
## AB-02.1 boot phases 1-3 — vault resolution, directories, sovereign identity
```mermaid
sequenceDiagram
    autonumber
    participant D as Docker/PID1
    participant E as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1
    participant TV as _ab_toml_val<br/>entrypoint-unified.sh:44
    participant FS as rootfs/tmpfs
    participant NPB as nostr-pod-bridge<br/>entrypoint-unified.sh:756

    D->>E: exec entrypoint-unified.sh<br/>set -euo pipefail (:17)
    E->>E: PATH = _ab_root_path_sanitise(PATH) — /nix/store and fixed system dirs only, no workspace path (:212, :227)
    E->>E: source config/lib/role-custody.sh when readable, definitions only (:276-279)
    Note over E: ADR-2028 vault resolve runs before Phase 1<br/>every supervised program inherits PID1 env
    E->>TV: _ab_vault_resolve() (:96) calls _ab_toml_val vault root (:101)
    TV->>FS: awk anchored-section parse of AGENTBOX_CONFIG (/etc/agentbox.toml)
    alt VAULT_ROOT empty (entrypoint-unified.sh:102)
        E->>E: unset VAULT_* export AGENTBOX_VAULT_ENABLED=0 (:103-105)
        E-->>D: echo "[vault] disabled — no [vault] in agentbox.toml" (:106)
        alt AGENTBOX_VAULT_LEGACY_PATHS=1 (entrypoint-unified.sh:110)
            E->>E: retain ONTOLOGY_PAGES_DIR (:111-113)
        else no legacy opt-in (entrypoint-unified.sh:117)
            E->>E: clear ONTOLOGY_PAGES_DIR="" (:121-122)
        end
    else VAULT_ROOT set (entrypoint-unified.sh:126)
        E->>E: VAULT_PAGES = root plus the optional pages folder, where absent, . and ./ all mean the root itself (:130-132, ADR-OM-007 flatten), HP_LOOM_SYNC default check from vault.loom_sync (:142)<br/>export VAULT_ROOT/VAULT_PAGES/VAULT_FORMAT/VAULT_TUI/VAULT_WORKING_ROOT/VAULT_TRANSCRIPTS/HP_LOOM_SYNC/LOOM_URL/LOOM_HP_HOST AGENTBOX_VAULT_ENABLED=1 (:163-164)
        E->>E: export ONTOLOGY_PAGES_DIR default to VAULT_PAGES (:173)
    end
    E->>E: AGENTBOX_ROLE_ISOLATION = _ab_toml_bool security role_isolation, exported to PID1 (:354)
    alt role_isolation = 1 and ab_role_isolation_ready (entrypoint-unified.sh:361)
        E->>FS: ab_secrets_root_prepare /run/secrets root 0711, guard moves to /run/secrets/.root-guard (:362-363)
    else flag on but roles config or plan missing
        E-->>D: ROLE-ISOLATION-UNAVAILABLE, boot continues as role_isolation = 0 (:365-366)
    end
    E->>FS: _ab_root_state_dir_prepare AB_ROOT_STATE_DIR — arms Stage B's one-shot guard (:372)
    E-->>D: echo "[1/8] Preparing runtime directories..." (:375)
    E->>FS: mkdir -p WORKSPACE, RUVECTOR_DATA_DIR, SOLID_POD_ROOT, /run/secrets... (:376-387)
    opt role_isolation off (entrypoint-unified.sh:396)
        E->>FS: chmod 0700 /run/secrets, chown 1000:1000 (:397-398)
    end
    E->>E: _ab_role_env_capture — flag on, each ROLE var to /run/secrets/role/NAME 0400 and unset (:497)
    E->>FS: chown 1000:1000 volume roots, non-recursive (:575-589)
    E->>FS: _ab_docker_socket_access — flag off chmod o+rw the host socket, flag on leave it and record ok or degraded docker-socket (:598, :624)
    E-->>D: echo "[2/8] Bootstrapping sovereign mesh identity..." (:741)
    E->>NPB: nostr-pod-bridge bootstrap (:756)
    Note right of NPB: k256 + nostr-bbs-core keypair, pod ACL/DID docs,<br/>writes /run/agentbox/identity.env 0600 — self-gates on<br/>sovereign_mesh.enabled, silent on success
    E-->>D: echo "[3/8] Ensuring workspace defaults..." (:777)
    E->>FS: mkdir WORKSPACE/agents if absent (:779)
    E->>FS: ln -sf /home/devuser/.claude WORKSPACE/.claude (:792)
    opt DREAM_CMD_SRC exists and no dream.md (entrypoint-unified.sh:801-810)
        E->>FS: cp dream.md to /home/devuser/.claude/commands/ (:810)
    end
    opt skill-creator not yet registered (entrypoint-unified.sh:826)
        E->>E: agentbox-manifest plugin-register --key skill-creator@claude-plugins-official (:826-831)
    end
    opt codex plugin baked and not registered (entrypoint-unified.sh:851)
        E->>E: agentbox-manifest plugin-register --key codex@openai-codex (:851-856)
    end
    opt voicelink enabled and record_events, Codex config present (:894-900)
        E->>FS: reconcile a top-level notify line naming voice-paned into the Codex config, an operator-set notify is never replaced (:900-927, ADR-2151)
    end
```

## AB-02.2 boot phases 4-6 — provisioning, artifact validation, supervisord exec, Stage B closure probes
```mermaid
sequenceDiagram
    autonumber
    participant E as entrypoint-unified.sh<br/>config/entrypoint-unified.sh:957-958
    participant AM as agentbox-manifest<br/>entrypoint-unified.sh:958
    participant FS as rootfs/tmpfs
    participant SV as supervisord<br/>entrypoint-unified.sh:1286
    participant B as [program:bootstrap] Stage B<br/>entrypoint-unified.sh:1300

    E-->>E: echo "[4/8] Provisioning agent stacks..." (:957)
    E->>AM: agentbox-manifest provision-stacks (:958), which adds the session-state hooks to each profile when the session plane is on (stacks.rs:324)
    E->>FS: chown -R 1000:1000 WORKSPACE/profiles (:967)
    E-->>E: echo "[5/8] Validating runtime closure..." (:974)
    E->>FS: bash validate-artifacts.sh (:975)
    alt validate-artifacts.sh fails (entrypoint-unified.sh:975)
        E-->>E: fatal BootstrapFailed, exit 1 (:976-979)
    end
    E->>FS: mkdir /run/agentbox /run/agentbox/hooks (:991) — bootstrap-seal writes here later
    E->>FS: when interaction_plane.enabled, mkdir /run/agentbox-sessions, chown 1000:1000, chmod 0755 — the daemon binds its socket in it (:986-989, ADR-2153)
    Note over E: ADR-2153 retired the N-05 AoE token-directory check here (mode 700 owner 1000, the N-05-VIOLATION marker): there is no token file, authorisation is the socket peer uid, see AB-16.7
    Note over E,FS: ADR-2040 — code-server and jupyter-lab credential minting<br/>runs here (:1008-1082), between the session-directory prep and the identity.env source<br/>below. Not re-diagrammed here — see AB-06.7 and AB-07.9
    E->>FS: source /run/agentbox/identity.env into PID1 env (:1092-1093)
    Note over E: under role_isolation identity.env carries public lines only, no AGENTBOX_NSEC<br/>or AGENTBOX_BRIDGE_SK (entrypoint-unified.sh:1095-1098), see AB-36
    E->>FS: _ab_role_key_file_own — flag on, the bridge key file goes to ab-identity 0400 (:1099)
    alt AGENTBOX_BRIDGE_SK set and role_isolation off (entrypoint-unified.sh:1112)
        E->>FS: write /run/secrets/nostr.key 0400 devuser, unset AGENTBOX_BRIDGE_SK from env (:1111-1120)
    end
    Note over E: ADR-2028 D3 — ontology PUSH cache refresh (Phase 5c)
    alt ONTO_PAGES empty (entrypoint-unified.sh:1150)
        E-->>E: echo "[5c/8] ontology PUSH cache refresh skipped ([vault] disabled)" (:1151)
    else ONTO_PAGES dir exists and builder present (entrypoint-unified.sh:1152)
        E->>FS: run_as_devuser node ontology-index-build.js (:1154-1156)
    end
    Note over E: Phase 5d(ii) — vault CLI liveness gate (ADR-2107/2108):<br/>the corpus's only programmatic door now, so a missing/broken<br/>binary is fail-LOUD, never fail-fatal (:1160-1179)
    alt AGENTBOX_VAULT_ENABLED=1 (entrypoint-unified.sh:1182)
        E->>FS: /opt/agentbox/bin/vault --version liveness probe (:1183-1185)
        alt probe succeeds
            E-->>E: echo "[5d/8] vault OK — version (bin)" (:1186)
        else binary present, probe fails
            E-->>E: echo "[5d/8] vault FAILED its liveness probe — rebuild the image" (:1188-1191)
        else binary absent
            E-->>E: echo "[5d/8] vault MISSING — [vault].cli on, no binary baked" (:1192-1195)
        end
    else vault disabled
        E-->>E: echo "[5d/8] vault gate skipped ([vault] disabled)" (:1198)
    end
    Note over E: Phase 5d(iii) — HP Loom reconcile (ADR-2140), never waited on (:1201-1206)
    alt AGENTBOX_VAULT_ENABLED=1 and HP_LOOM_SYNC not off (entrypoint-unified.sh:1209)
        E-->>E: echo the previous receipt, state at ts, detail, from hp-loom-reconcile.json (:1214)
        alt /opt/agentbox/scripts/hp-loom-reconcile.sh executable (:1216)
            E->>FS: background run_as_devuser, timeout 1800, --mode check or full, log to hp-loom-reconcile.log (:1217-1218)
            E-->>E: echo "[5d/8] HP Loom reconcile started in the background" (:1219)
        else script missing
            E-->>E: echo "[5d/8] HP Loom reconcile SKIPPED - rebuild the image" (:1221)
        end
    else loom_sync off or vault disabled
        E-->>E: echo "[5d/8] HP Loom reconcile skipped" (:1225)
    end
    opt AGENTBOX_TAB0_BRIDGE_SUPERVISED=1 and BRIDGE_TOKEN unset (entrypoint-unified.sh:1239)
        E->>FS: generate/read BRIDGE_TOKEN, write 0600 secrets file (:1240-1255)
    E->>E: eval agentbox-manifest system-one-project --format shell into PID1 env before supervisord, so sibling services see the typed-decision slot (:1258-1262, ADR-2139)
    end
    opt role_isolation = 1 (entrypoint-unified.sh:1272)
        E->>FS: ab_role_secrets_deliver /etc/agentbox/role-secrets.tsv into /run/secrets/role (:1274)
        E->>FS: append ok or degraded secrets-delivery to /run/secrets/role-isolation.state (:1275-1281)
        E->>E: _AB_SUPERVISORD_CONF = ab_supervisord_conf_pick 1, the roles config (:1280)
    end
    E-->>E: echo "[5b/8] Starting supervisord..." (:1283)
    E->>E: _ab_role_env_scrub — flag on, any ROLE var still in PID1 env is a LEAK, unset (:1285)
    E->>SV: exec supervisord -c _AB_SUPERVISORD_CONF -n (:1286)
    Note over E,SV: Stage A process image is REPLACED by supervisord (exec) —<br/>PID1 is now supervisord, not the shell
    SV->>B: spawn [program:bootstrap] with AGENTBOX_BOOTSTRAP_STAGE=B (flake.nix, not this file)
    B->>B: re-export WORKSPACE/RUVECTOR_DATA_DIR/SOLID_POD_ROOT/AGENTBOX_CONFIG defaults (:1327-1332)
    alt AGENTBOX_VAULT_ENABLED unset (entrypoint-unified.sh:1306)
        B->>B: _ab_vault_resolve() again (standalone-invocation fallback) (:1306)
    end
    B->>B: _ab_stage_b_claim AB_ROOT_STATE_DIR — atomic mkdir stage-b.claimed (:1315)
    alt claim made
        B-->>SV: Stage B claimed, one-shot (:1317)
    else already claimed this container start
        B-->>SV: no-op exit 0, a supervisorctl replay runs nothing (:1318)
    else guard dir missing or not root 0700
        B-->>SV: REFUSED exit 1 (:1319)
    end
    B-->>SV: echo "[6/8] Validating pre-packaged service closures..." (:1333)
    loop _probe_closure for management-api, mcp, gated toolchains (entrypoint-unified.sh:1335-1349)
        B->>FS: test -d node_modules under each closure dir
        alt node_modules missing (entrypoint-unified.sh:1344)
            B-->>SV: fatal MissingArtifactDetected, exit 1 (:1344-1347)
        end
    end
    B-->>SV: echo "[6/8] Service closures OK." (:1377)
```

## AB-02.3 boot phases 7-8 — ruflo plugins, manifest projection, runtime-env publish
```mermaid
sequenceDiagram
    autonumber
    participant B as [program:bootstrap] Stage B<br/>entrypoint-unified.sh:1451
    participant AM as agentbox-manifest<br/>entrypoint-unified.sh:2270
    participant MCP as .mcp.json<br/>entrypoint-unified.sh:1556
    participant FS as rootfs/tmpfs
    participant RT as runtime-env.sh<br/>entrypoint-unified.sh:3820
    participant TS as trust-seed.cjs<br/>run once from entrypoint-unified.sh:1812

    B-->>B: echo "[7/8] Bootstrapping ruflo plugins..." (:1451)
    B->>FS: mkdir ~/.claude-flow/plugins, /var/cache/ruflo-plugins (:1453-1457)
    opt claude-flow-config.template.json present, config.json stale (entrypoint-unified.sh:1467)
        B->>FS: sed RUVECTOR_PG_PASSWORD into ~/.claude-flow/config.json (:1468-1473)
    end
    Note over B: PRD-018/ADR-036 D6 — read RuVector memory gate flags<br/>via _ab_toml_bool memory_learning.* (:1493-1505), fail-open all-off
    B->>MCP: ensure .mcp.json points at ruvector-mcp.cjs, idempotent (:1523-1585)
    B->>MCP: inject PRD-018 RuVector memory gate env (:1587-1664)
    opt browser-gpu sidecar reachable (entrypoint-unified.sh:2245-2246)
        B->>AM: agentbox-manifest mcp-set-server --name browser-gpu (:2248)
    end
    B->>AM: agentbox-manifest mcp-reconcile-aqe --provider "$_MR_PROVIDER_ARG" (:2270)
    opt _MR_ENABLED=1 — ADR-041 model routing (entrypoint-unified.sh:2280)
        B->>AM: run_as_devuser agentbox-manifest model-routing-project --manifest AGENTBOX_CONFIG --workspace WORKSPACE (:2280-2284)
        Note right of AM: projects [model_routing.routes] into every<br/>.agentic-qe/llm-config.json under the workspace,<br/>a runtime file the repository does not track
    end
    Note over B: ADR-069 — interaction_plane.proxy projected every boot (:2335-2346)
    opt AGENTBOX_CONFIG exists and agentbox-manifest present (entrypoint-unified.sh:2341)
        B->>AM: agentbox-manifest nip98-config --manifest AGENTBOX_CONFIG --out .agentbox/nip98-proxy-config.json (:2342)
        B->>FS: chown 1000:1000, chmod 600 nip98-proxy-config.json (:2344-2345)
    end
    Note over B: hook registration reconciled into root settings.json every boot, each fail-open (|| true)
    B->>FS: decline-recorder on Stop, SubagentStop and UserPromptSubmit when claude_code.floor is on, default on (:2098-2157, ADR-2144)
    B->>FS: voice-paned record on Stop and Notification when voicelink.enabled and record_events (:2159-2201, ADR-2151)
    B->>FS: session-state.cjs hook on UserPromptSubmit, Stop and Notification into root settings.json when interaction_plane.enabled, pruned when off (:1944-1956, ADR-2153)
    B->>FS: Codex config.toml notify line set to session-state.cjs codex, keeping --voice when voicelink records, restored when off (:1921-1940)
    B->>FS: activity-sink hooks when [integrations.activity_sink] has a usable url (:2853-2939, ADR-2146)
    B->>AM: agentbox-manifest instructions-project, with --managed-out /etc/claude-code/CLAUDE.md when the floor is on, --no-managed otherwise (:2994-2996, ADR-2144)
    Note over B: ADR-2139 vendor slot — _ab_toml_bool system_one enabled (entrypoint-unified.sh:2670) selects system-one-project (:2673-2683), else sso-project (:2700-2702)
    opt [system_one].enabled and the projection is non-empty (entrypoint-unified.sh:2672, :2675)
        B->>AM: agentbox-manifest system-one-project --format json, then shell and plugin-config (:2673, :2681-2683)
        B->>FS: atomic tmp-and-mv /run/agentbox/system-one.json, the proxy's only config (:2678-2679)
    end
    Note over B: unusable slot writes a disabled stub and falls back to sso-project, so the seal's promised file always exists (entrypoint-unified.sh:2685-2693)
    Note over B: RESOLVED — the ontology-bridge MCP registration that used to sit<br/>here (gated ENABLE_ONTOLOGY) is GONE (ADR-2107/2108): agents reach the<br/>corpus only through the vault CLI and the Loom over HTTP (:2352-2355)
    opt AGENTBOX_TRUST_SEED not 0 and node present (entrypoint-unified.sh:1812)
        B->>TS: node trust-seed.cjs marks workspace root and worktrees trusted, once (:1812-1814)
        Note right of TS: DOC-DRIFT resolved — deliberately NOT a SessionStart hook any<br/>more (it walked ~1,170 paths per session start, avg 3.2s, and raced<br/>Claude Code's own ~/.claude.json writes) — hook shim reconcile below prunes<br/>any stale registration
    end
    Note over B: after the hub projection (see AB-02.17) the claude-flow entry is mirrored into user scope (entrypoint-unified.sh:3389-3397)
    B->>AM: agentbox-manifest mcp-mirror-server --from .mcp.json --to ~/.claude/.claude.json --name claude-flow (:3395-3396, mcp.rs:143)
    B-->>B: echo "[8/8] Publishing environment hints..." (:3807)
    B->>RT: cat RUNTIME_ENV_FILE=/run/agentbox/runtime-env.sh heredoc (:3820-3911)
    Note right of RT: role_isolation on adds DOCKER_HOST=unix:///run/docker-ro.sock,<br/>the GET-only read proxy, to devuser shells (entrypoint-unified.sh:3815-3818)
    Note right of RT: exports WORKSPACE, RUVECTOR_PG_CONNINFO, VAULT_ROOT/PAGES/<br/>FORMAT/TUI/WORKING_ROOT/TRANSCRIPTS, ONTOLOGY_PAGES_DIR,<br/>AGENTBOX_INTERACTION_PLANE_*, CUDA_PATH etc (see AB-02.11)
    B->>FS: _ab_quarantine_stale_cargo_bins, ELF bins in WORKSPACE/.cargo/bin whose<br/>/nix/store loader was collected move to .stale-loader/, numbered, never deleted (:3912-3930)
    Note right of FS: ADR-2029 D4 follow-up (2026-10-01) - .cargo/bin outlives rebuilds, so a dead loader<br/>once shadowed baked rune and agentbox-hook. Custody W0 now APPENDS it, devuser shells only,<br/>never root (entrypoint-unified.sh:3896-3898). Loader path read from bytes, nothing executed (:3918-3920)
    B->>FS: ln -sf RUNTIME_ENV_FILE /etc/profile.d/agentbox-runtime.sh best-effort (:3976)
    B->>FS: cp RUNTIME_ENV_FILE to durable WORKSPACE/.agentbox-runtime-env.sh (:3979)
    B->>FS: write fish conf.d/agentbox-runtime.fish sourcing the env file (:3985-4004)
```

## AB-02.4 supervision tree — core and identity programs
```mermaid
flowchart TB
    BOOT["program:bootstrap<br/>flake.nix:2638<br/>no user= line -&gt; runs as root<br/>priority=5 autorestart=false one-shot"]
    MGMT["program:management-api<br/>flake.nix:2652<br/>user=devuser bind 0.0.0.0:ENV_MANAGEMENT_API_PORT default 9090<br/>priority=20 REQUIRED_FOR_READINESS=true"]
    CRED["program:claude-cred-sync (ADR-2118)<br/>flake.nix:3266<br/>gate toolchains.claude_code, converges ~/.claude/.credentials.json<br/>both ways with the host bind; exits 0 (stopped) if the bind is absent<br/>user=devuser priority=30 autorestart=unexpected"]
    SEAL["program:bootstrap-seal<br/>flake.nix:2668<br/>user=devuser priority=99 autorestart=false one-shot<br/>writes /run/agentbox/bootstrap.done, timeout 120s"]
    SOLID["program:solid-pod<br/>flake.nix:2680<br/>gate sovereign_mesh.enabled and local-solid-rs active<br/>user=devuser priority=30 REQUIRED_FOR_READINESS=true"]
    HTTPSB["program:https-bridge<br/>flake.nix:2697<br/>gate sovereign_mesh.enabled and https_bridge<br/>user=devuser priority=32<br/>self-signed TLS to plain-HTTP target, https-bridge/https-proxy.js<br/>full cert-provisioning and forwarding flow: see AB-30.4, AB-30.4"]
    GATERELAY{"podBridgeEnabled? ADR-2003<br/>flake.nix:1711 relayLocal and pod_bridge"}
    NRELAY1["program:nostr-relay native<br/>flake.nix:2734<br/>bind 127.0.0.1:7777 (AGENTBOX_RELAY_BIND)<br/>user=devuser, ab-identity under role_isolation priority=35 REQUIRED_FOR_READINESS=false"]
    NRELAY2["program:nostr-relay nostr-rs-relay<br/>flake.nix:2746<br/>config /etc/agentbox/nostr-relay.toml<br/>user=devuser, ab-identity under role_isolation priority=35 REQUIRED_FOR_READINESS=false"]
    NGW["program:nostr-gateway<br/>flake.nix:2411<br/>user=devuser, ab-gateway under role_isolation priority=234<br/>off switch AGENTBOX_NOSTR_GATEWAY=0"]
    DRP["program:docker-read-proxy (custody W0)<br/>flake.nix:2799<br/>no user= -&gt; starts root, env -i, binds /run/docker-ro.sock,<br/>then drops itself to uid 65534 plus the socket's group<br/>flag off: one line, exit 0, EXITED is expected priority=20"]
    SID["program:serve-identity (custody W3)<br/>flake.nix:2821<br/>env -i, nostr-pod-bridge serve-identity, the identity port<br/>user=devuser here, ab-identity in supervisord.roles.conf<br/>flag off: exits 0 without reading a key priority=30"]
    ISO["/etc/supervisord.roles.conf<br/>flake.nix:4017 agentbox-manifest role-accounts isolate<br/>a pure function of this config and config/role-accounts.json:<br/>only user= and environment= of role programs differ"]
    TSD["program:tailscaled<br/>flake.nix:2772<br/>gate networking.tailscale, no user= -&gt; root<br/>socket /var/run/tailscale priority=15"]
    TSU["program:tailscale-up<br/>flake.nix:2787<br/>gate networking.tailscale, no user= -&gt; root<br/>priority=16 autorestart=false one-shot"]

    BOOT -->|"priority 5, runs before 20/30"| MGMT
    BOOT -->|"priority 5, runs before 30"| CRED
    MGMT -->|"REQUIRED_FOR_READINESS=true"| SEAL
    SOLID -->|"REQUIRED_FOR_READINESS=true"| SEAL
    GATERELAY -->|true| NRELAY1
    GATERELAY -->|false| NRELAY2
    MGMT -.->|"verifyNip98 contract reused"| HTTPSB
    TSD --> TSU
    NGW -.->|"/spawn through the session socket, ab-sessions member under role_isolation"| SESSREF["program:agentbox-sessions<br/>see AB-02.7"]
    ISO -.->|"role_isolation = 1 picks it at exec"| SID
    ISO -.-> NRELAY1
    ISO -.-> NGW
    DRP -.-> N1["DEBT: the comment still says TODO(custody W1) move into the role registry<br/>lib/role-accounts.nix (flake.nix:2808), but the registry is<br/>config/role-accounts.json and the proxy has no account in it"]
```

## AB-02.6 supervision tree — gated toolchain programs
```mermaid
flowchart TB
    QGIS["program:qgis-mcp<br/>flake.nix:2354<br/>gate spatial.qgis, TCP proxy to gui-tools-service:9877<br/>user=devuser priority=230"]
    BLEND["program:blender-mcp<br/>flake.nix:2381<br/>gate spatial.blender, bridges 127.0.0.1:9876 to external GUI sidecar<br/>user=devuser priority=231"]
    JLAB["program:jupyter-lab<br/>flake.nix:2454<br/>gate data_science.jupyter, bind 0.0.0.0:8888 — RESOLVED ADR-2040,<br/>JUPYTER_TOKEN now minted at boot (see AB-07.9), no longer tokenless<br/>user=devuser priority=232"]
    IMGM["program:imagemagick-mcp<br/>flake.nix:2711<br/>gate media.imagemagick<br/>user=devuser priority=210"]
    COMFY["program:comfyui-builtin<br/>flake.nix:2874<br/>gate media.comfyui_builtin, bind 127.0.0.1:8188<br/>user=devuser priority=220"]
    OPF["program:opf-router<br/>flake.nix:2759<br/>gate privacyFilterEnabled, OPF_PORT default 9092<br/>user=devuser priority=240"]
    DREAM["program:dream-engine<br/>flake.nix:2896<br/>gate dreamEngineEnabled (:1761), LOOM_URL default 192.168.2.132:8084/v1<br/>user=devuser priority=230"]
    SIDEP["program:sidestr-producer<br/>flake.nix:3105<br/>gate sidechain.enabled (:321), shipped true (agentbox.toml:1918, re-enabled on recovered validated prefixes after the ENOSPC failure, :1917) so the producer block is emitted; runs the baked upstream, refuses on a pin mismatch<br/>user=devuser, ab-sidestr-dreamlab under role_isolation priority=260 startretries=5"]
    SIDEM["program:sidestr-mirror<br/>flake.nix:3123<br/>gate sidechain.mirror, only with enabled (:322)<br/>user=devuser priority=261"]
    SIDEF["program:sidestr-faucet<br/>flake.nix:3140<br/>gate sidechain.faucet, only with enabled (:323)<br/>user=devuser, ab-faucet-dreamlab under role_isolation priority=262"]
    CODES["program:code-server<br/>flake.nix:2850<br/>gate toolchains.code_server, bind 0.0.0.0:8080 — RESOLVED ADR-2040,<br/>auth password with a boot-minted credential (see AB-07.9), not auth none<br/>user=devuser priority=50"]

    SO1["program:system-one-proxy (ADR-2139)<br/>flake.nix:3035<br/>gate system_one.enabled (:314), loopback typed-decision proxy,<br/>reads only /run/agentbox/system-one.json, vendor keys from env<br/>user=devuser priority=60 autorestart=unexpected"]
    SEO["program:seo-watch-cron<br/>flake.nix:3062<br/>gate seo_watch.enabled (:308), supercronic over the website repo crontab<br/>user=devuser priority=250"]
    TGC["program:target-gc (ADR-2138)<br/>flake.nix:3354<br/>gate resources.target_gc.enabled (:243), idle Cargo target sweeper<br/>user=devuser priority=246"]
    OPF -.->|"privacy filter mode gate"| DREAM
    SIDEP -->|"block files"| SIDEM
    SIDEP -->|"local producer"| SIDEF
    DREAM -.->|"ZAI_ANTHROPIC_API_KEY, RUVECTOR_PG_CONNINFO inherited from PID1"| RVNOTE["note: secrets never written<br/>into generated supervisor text"]
```

## AB-02.7 supervision tree — desktop stack and interaction plane
```mermaid
flowchart TB
    GATESTACK{"desktop.stack ADR-2003<br/>flake.nix:199-200, Nix if/else-if/else"}
    HYPR["program:hyprland<br/>flake.nix:2531<br/>user=devuser priority=40"]
    XWAY["program:xwayland-session<br/>flake.nix:2545<br/>user=devuser priority=41"]
    WAYVNC["program:wayvnc<br/>flake.nix:2556<br/>bind 0.0.0.0:5901 user=devuser priority=42"]
    XORG["program:xorg-nvidia<br/>flake.nix:2567<br/>no user= -&gt; root priority=40"]
    I3A["program:i3wm xorg-nvidia branch<br/>flake.nix:2577<br/>user=devuser priority=41"]
    X11VNC["program:x11vnc<br/>flake.nix:2588<br/>bind 0.0.0.0:5901 user=devuser priority=42"]
    XVNC["program:xvnc<br/>flake.nix:2599<br/>bind 0.0.0.0:5901 no user= -&gt; root priority=40"]
    I3B["program:i3wm i3-x11 default branch<br/>flake.nix:2577<br/>user=devuser priority=41"]
    SESS["program:agentbox-sessions<br/>flake.nix:2910<br/>gate interaction_plane.enabled (:418), no TCP listener, unix socket /run/agentbox-sessions/sessions.sock<br/>user=devuser priority=47 startsecs=5"]
    NIP98["program:nip98-proxy<br/>flake.nix:2923<br/>bind 0.0.0.0:9096, named operator routes only, unknown routes 404 (proxy.mjs)<br/>user=devuser, ab-ingress under role_isolation priority=46"]
    TAB0["program:tab0-bridge<br/>flake.nix:2953<br/>gate sovereign_mesh.enabled<br/>user=devuser priority=236"]
    TMUX["program:tmux-autostart<br/>flake.nix:2966<br/>user=devuser priority=95 autorestart=false one-shot"]

    GATESTACK -->|"hyprland-wayland"| HYPR --> XWAY --> WAYVNC
    GATESTACK -->|"xorg-nvidia"| XORG --> I3A --> X11VNC
    GATESTACK -->|"i3-x11 default"| XVNC --> I3B
    TAB0 -->|"session socket"| SESS
    NIP98 -.->|"/mgmt/* forwarded"| MGMTREF["management-api<br/>see AB-02.4"]
    TAB0 -.->|"AGENTBOX_TAB0_BRIDGE_SUPERVISED=1"| TMUX
```

## AB-02.8 PID1 root, priority ordering and bootstrap-seal
```mermaid
sequenceDiagram
    autonumber
    participant SV as supervisord PID1<br/>flake.nix:2623 supervisord section nodaemon=true
    participant BOOT as program:bootstrap<br/>flake.nix:2638 no user= line means root
    participant MGMT as program:management-api<br/>flake.nix:2652 user=devuser :2655 priority=20 :2659
    participant SOLID as program:solid-pod<br/>flake.nix:2680 user=devuser :2683 priority=30 :2687
    participant SEAL as program:bootstrap-seal<br/>agentbox/config/seal-bootstrap.sh priority=99
    participant SC as supervisorctl status<br/>seal-bootstrap.sh:100

    Note over SV: supervisord itself inherits root from the exec'd<br/>entrypoint-unified.sh Stage A (entrypoint-unified.sh:1286)
    SV->>BOOT: spawn priority=5 :2644, environment AGENTBOX_BOOTSTRAP_STAGE=B (:2640)
    Note right of BOOT: no user= line — Stage B (phases 6-8) runs as ROOT,<br/>needed for chown/mkdir under devuser volumes
    SV->>MGMT: spawn priority=20 :2659, user=devuser (:2655)
    SV->>SOLID: spawn priority=30, user=devuser, gated sovereign_mesh.enabled (:2678-2680)
    Note over SV: every program below priority=99 launches in ascending<br/>priority order but does not block on prior RUNNING state
    Note over SV,BOOT: RESOLVED ADR-2063 (2026-09-05) - a program that needs a file Stage B writes later<br/>waits for it with a bounded timeout instead of crashing into FATAL (see AB-02.17)
    SV->>SEAL: spawn priority=99 last, user=devuser (:2670-2675)
    SEAL->>SEAL: _required_programs() awk-scans /etc/supervisord.conf<br/>for AGENTBOX_REQUIRED_FOR_READINESS=true blocks (seal-bootstrap.sh:44-68)
    loop poll every 2s up to BOOTSTRAP_SEAL_TIMEOUT=120s (seal-bootstrap.sh:96-115)
        SEAL->>SC: supervisorctl status <program> for each required program
        alt any required program not RUNNING
            SEAL-->>SEAL: log WaitingForProgram, continue loop (seal-bootstrap.sh:103-105)
        end
    end
    alt timeout elapsed before all RUNNING (seal-bootstrap.sh:117)
        SEAL-->>SV: log BootstrapSealTimeout, exit 1 — sentinel NEVER written (seal-bootstrap.sh:118-123)
    else all required programs RUNNING
        SEAL->>SEAL: write /run/agentbox/bootstrap.done atomically via tmp+mv (seal-bootstrap.sh:187-191)
        Note right of SEAL: INVARIANT — DDD-001 BootstrapCompletion:<br/>sentinel existence is the sole BootstrapCompleted signal
    end
```

## AB-02.9 readiness — /ready vs /health
```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator/probe
    participant R as management-api routes<br/>server.js:468 /ready, :581 /health
    participant BS as bootstrapState<br/>server.js:57-65 fs watch of BOOTSTRAP_SENTINEL
    participant ML as adapters/manifest-loader<br/>server.js:504
    participant AH as adapterHealth map<br/>server.js:511
    participant FS as fs.promises.access<br/>server.js:528-534

    O->>R: GET /ready
    R->>BS: read bootstrapState.completed (poll every 2s of /run/agentbox/bootstrap.done, server.js:76)
    alt bootstrap.done sentinel absent
        R->>R: missing.push('bootstrap.done sentinel') (:498)
    end
    R->>ML: loadManifest() (:504)
    Note over R: 2. Adapter health — every non-off slot must be healthy (:501)
    loop for slot, impl in manifestAdapters (server.js:509)
        alt impl === 'off'
            R->>R: continue (:510)
        else adapterHealth[slot] !== 'healthy' (server.js:511)
            R->>R: missing.push adapter:<slot> not healthy (:512)
        end
    end
    Note over R: 3. Required filesystem paths (:516)
    R->>FS: access WORKSPACE, /var/lib/ruvector (:521)
    opt manifestAdapters.pods === local-solid-rs (server.js:523)
        R->>R: add integrations.solid_pod_rs.storage_root to requiredPaths (:523-526)
    end
    opt sovereign_mesh.publish_agent_events === true (server.js:540)
        alt NOSTR_RELAYS env empty (server.js:543)
            R->>R: missing.push publish_agent_events but NOSTR_RELAYS empty (:544)
        end
    end
    alt missing.length greater than 0 (server.js:560)
        R-->>O: 503 ready:false reason, missing[] (:561-567)
    else
        R-->>O: 200 ready:true since bootstrapState.since, degraded[] (:570-575)
    end

    rect rgb(240,240,240)
    Note over O,R: /health (server.js:581-613) is human-inspection-only —<br/>note field at :611 says "Use /ready for orchestrator readiness probes"
    O->>R: GET /health
    R-->>O: 200 status ok/degraded, uptime, adapters map, degraded_count (:604-612)
    end
```

## AB-02.10 supervisord program lifecycle
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
        program bootstrap flake.nix:2638, startsecs=0 at flake.nix:2643
        program bootstrap-seal flake.nix:2668, startsecs=0 at flake.nix:2673, timeout 120s at flake.nix:2675
        program tailscale-up flake.nix:2787, startsecs=0 at flake.nix:2793
        program tmux-autostart flake.nix:2966, startsecs=0 at flake.nix:2972
    end note
    note right of Running
        DOC-DRIFT most programs set no startretries in flake.nix
        and rely on the supervisord built-in default startretries=3, autorestart=true
        e.g. management-api flake.nix:2652, nostr-relay flake.nix:2734 and flake.nix:2746
        exceptions agentbox-mcp-hub startretries=2 at flake.nix:3302
        and sidestr-producer startretries=5 at flake.nix:3112, a pin mismatch stops at FATAL
    end note
    note left of Starting
        startsecs varies by program
        agentbox-sessions flake.nix:2910, startsecs=5 at flake.nix:2917
        nip98-proxy flake.nix:2923, startsecs=3 at flake.nix:2930
        code-server flake.nix:2850, startsecs=5 at flake.nix:2868
        xwayland-session flake.nix:2545, startsecs=5 at flake.nix:2551
    end note
```

## AB-02.11 tmux-autostart window layout
```mermaid
flowchart TB
    START["program:tmux-autostart<br/>flake.nix:2966, priority=95<br/>runs config/tmux-autostart.sh"]
    ENV["fish conf.d/agentbox-runtime.fish<br/>entrypoint-unified.sh:3985-4004<br/>sources RUNTIME_ENV_FILE, fallback to durable copy<br/>every new tmux window's fish shell sources it on start"]
    W0["window 0 Claude<br/>tmux-autostart.sh:291<br/>tab0-bridge injection target<br/>CLAUDE_CONFIG_DIR=/home/devuser/.claude (:292), AGENTBOX_PROFILE=tab0 (:294)"]
    W1["window 1 Agent<br/>tmux-autostart.sh:317<br/>agent execution workspace"]
    W2["window 2 Services<br/>tmux-autostart.sh:324<br/>supervisorctl status (:325)"]
    W3["window 3 Build<br/>tmux-autostart.sh:330"]
    W4["window 4 Logs<br/>tmux-autostart.sh:338<br/>supervisorctl tail -f management-api (:339), split pane"]
    W5["window 5 System<br/>tmux-autostart.sh:345<br/>systemscape restart loop (:349), split with btm/htop"]
    W6["window 6 VNC<br/>tmux-autostart.sh:362<br/>host shell, display :1 port 5901 status"]
    W7["window 7 Git<br/>tmux-autostart.sh:375<br/>git status in PROJECT dir"]
    W8["window 8 Sessions<br/>tmux-autostart.sh:383<br/>agentbox-sessions pick (:384-385) or a disabled notice (:387),<br/>attach lands here when the binary exists (:440-441)"]
    W9["window 9 Notes<br/>tmux-autostart.sh:102 _notes_window, called :396<br/>ADR-2029 Rune markdown TUI over vault"]

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
    G3 -->|"absent"| REFUSE3["refuse: rebuild or cargo install<br/>tmux-autostart.sh:175-187<br/>DEBT: the hint installs aka-rider/rune v1.4.0 (:183),<br/>not the DreamLab fork the image bakes"]
    G3 -->|"present"| G4
    G3 -.->|"candidates that fail --version"| STALE["reported as stale loaders, skipped<br/>tmux-autostart.sh:170-173"]
    G4 -->|"not writable"| REFUSE4["refuse: not launching degraded<br/>tmux-autostart.sh:201-217"]
    G4 -->|"writable"| LAUNCH
```

## AB-02.12 profile isolation — per-profile HOME and CLAUDE_CONFIG_DIR (ADR-2007)
```mermaid
sequenceDiagram
    autonumber
    participant E as entrypoint-unified.sh<br/>Phase 4 entrypoint-unified.sh:957-958
    participant AM as agentbox-manifest provision-stacks<br/>services/agentbox-manifest/src/stacks.rs:187 build_profile
    participant FS as WORKSPACE/profiles/STACK
    participant SEED as session-provision.mjs<br/>scripts/session-provision.mjs:203
    participant WRAP as harness wrapper<br/>config/harness-wrappers/zai.sh|openrouter.sh

    E->>AM: agentbox-manifest provision-stacks, runs as root Phase 4 (entrypoint-unified.sh:958)
    loop for each stack in STACKS_JSON (stacks.rs:371-372)
        AM->>FS: build_profile writes root=WORKSPACE/profiles/STACK (stacks.rs:188-189)
        AM->>FS: symlink profiles/STACK/projects and /workspace (stacks.rs:199-200)
        AM->>FS: write .env with AGENT_STACK=STACK (stacks.rs:191-201)
        opt Claude-hosted profile
            AM->>FS: hooks start from learning_hooks (stacks.rs:290), then add the decline-recorder entries on Stop, SubagentStop and UserPromptSubmit when the floor gate is on (stacks.rs:293-308, ADR-2144)
            AM->>FS: add voice-paned record on Stop and Notification when voicelink record_events is on (stacks.rs:311-323, ADR-2151)
            AM->>FS: add session-state.cjs on UserPromptSubmit, Stop and Notification when the session plane is on (stacks.rs:324-334, ADR-2153)
            AM->>FS: add the activity-sink hooks when [integrations.activity_sink] is usable (stacks.rs:335-337, ADR-2146)
            AM->>FS: write .claude/settings.json with the assembled hooks (stacks.rs:339-357)
        end
    end
    Note over E: chown -R 1000:1000 WORKSPACE/profiles after provision-stacks (entrypoint-unified.sh:967)
    SEED->>FS: provision profiles/openrouter/.claude/settings.local.json ANTHROPIC_BASE_URL/AUTH_TOKEN (session-provision.mjs:16-34, run by the daemon at AB-02.18)
    SEED->>FS: provision profiles/zai/.claude/settings.local.json (session-provision.mjs:36-53)
    Note right of SEED: ADR-043 D4.1 — a distinct AGENTBOX_PROFILE per session<br/>yields a distinct persisted did:nostr identity
    Note over WRAP: at session launch the wrapper pins HOME=PROFILE and<br/>CLAUDE_CONFIG_DIR=PROFILE/.claude (see AB-02.13)
    Note over E,WRAP: DIVERGENCE — profile isolation routes configuration under<br/>ONE OS user devuser — ADR-2007 line 40 says harnesses are isolated<br/>by directory, not by OS user — it is NOT an OS access boundary
```

## AB-02.13 harness wrapper invocation — Z.AI / OpenRouter redirect assertion
```mermaid
sequenceDiagram
    autonumber
    participant SP as agentbox-sessions tmux session<br/>services/agentbox-sessions/src/contract.rs:79 wrapper kind
    participant W as zai.sh / openrouter.sh<br/>config/harness-wrappers/zai.sh:1
    participant PV as provider_url_validate<br/>config/harness-wrappers/_provider-url.sh:71-72
    participant SET as settings.local.json<br/>WORKSPACE/profiles/SLUG/.claude
    participant C as claude binary

    SP->>W: argv is the wrapper path alone (config/session-plane/launch.json:15), exec zai.sh (SLUG=zai EXPECT_HOST=z.ai) or openrouter.sh (EXPECT_HOST=openrouter.ai, openrouter.sh:26-28)
    W->>SET: check PROFILE dir and SETTINGS file exist (zai.sh:92-100)
    alt profile or settings.local.json missing
        W-->>SP: _die fatal, exit 1 (zai.sh:41-53,92-100)
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
        W->>W: export HOME=PROFILE, CLAUDE_CONFIG_DIR=PROFILE/.claude (zai.sh:131-132)
        W->>W: export ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY="" (zai.sh:133-135)
        W->>W: export AGENTBOX_PROFILE default SLUG (zai.sh:138)
        opt W == openrouter.sh (openrouter.sh:141-156)
            W->>SET: read "model" from settings.local.json, export as ANTHROPIC_MODEL<br/>and every tier alias, caller value wins (openrouter.sh:141-150)
            Note right of W: RESOLVED 2026-09-26 (ADR-2007/ADR-2111 re-verification) —<br/>settings.local.json is a project-level file CLAUDE_CONFIG_DIR does not read,<br/>so the model pin sat unread and OpenRouter billed Opus instead of the<br/>configured model — also pins CLAUDE_CODE_PROMPT_CACHE_TTL=1h (openrouter.sh:156)
        end
        W-->>SP: echo credential-free confirmation line (zai.sh:142)
        W->>C: exec claude "$@" (zai.sh:143)
    end
    Note over W: DOC-DRIFT (confirmed still open) — agentbox/docs/BASELINE-container.md:276<br/>still says ADR-2007 is partial and the wrapper host assertion is substring-based —<br/>code (zai.sh:114-128, _provider-url.sh) has since implemented full URL parsing,<br/>closed 2026-09-05 per the code's own ADR-2007 closeout comments
```

## AB-02.14 VAULT env propagation — resolve to PID1 to supervised programs
```mermaid
sequenceDiagram
    autonumber
    participant TOML as agentbox.toml vault section<br/>entrypoint-unified.sh:61-96
    participant VR as _ab_vault_resolve<br/>entrypoint-unified.sh:96
    participant PID1 as supervisord PID1 env<br/>entrypoint-unified.sh:1286 exec
    participant PROG as supervised programs<br/>flake.nix program blocks

    VR->>TOML: _ab_toml_val vault root/pages/format/tui/repo/working/transcripts (:101,126-145)
    alt vault section absent, VAULT_ROOT empty (entrypoint-unified.sh:102)
        VR-->>VR: echo "[vault] disabled — no [vault] in agentbox.toml" (:106)
        Note right of VR: fail-loud absent-vault branch — every consumer<br/>disables itself rather than indexing a stale tree
        alt AGENTBOX_VAULT_LEGACY_PATHS=1 opt-in (entrypoint-unified.sh:110)
            VR-->>VR: RETAIN deprecated ONTOLOGY_PAGES_DIR (:111-113)
        else no opt-in
            VR-->>VR: ONTOLOGY_PAGES_DIR="" cleared, warns once (:118-122)
        end
    else VAULT_ROOT resolved (entrypoint-unified.sh:126)
        VR->>VR: VAULT_REPO from [vault].repo, else derived from VAULT_ROOT<br/>(knowledge/working to parent, else root itself) (:152-158)
        alt VAULT_REPO/ontology/vocabulary.yaml absent (entrypoint-unified.sh:159)
            VR-->>VR: warn, VAULT_REPO="" (governed vault writes then refuse) (:160-161)
        end
        VR->>VR: export VAULT_ROOT/VAULT_REPO/VAULT_PAGES/VAULT_FORMAT/VAULT_TUI/<br/>VAULT_WORKING_ROOT/VAULT_WORKING_PAGES/VAULT_TRANSCRIPTS/HP_LOOM_SYNC/LOOM_URL/LOOM_HP_HOST (:163-164)
        VR->>VR: export ONTOLOGY_PAGES_DIR default VAULT_PAGES, derived (:170-173)
        Note right of VR: DIVERGENCE — vault ENABLED but an explicit<br/>ONTOLOGY_PAGES_DIR differing from VAULT_PAGES<br/>still OVERRIDES it for legacy consumers (:170-171)
    end
    VR->>PID1: exec supervisord inherits VAULT_ROOT/REPO/PAGES/FORMAT/TUI/WORKING/TRANSCRIPTS (entrypoint-unified.sh:1286)
    PID1->>PROG: every program child inherits PID1 env at spawn, no VAULT_* set per-program in flake.nix
    Note over PID1,PROG: exceptions since custody W0 and W3 - docker-read-proxy and serve-identity<br/>start through env -i and see only the variables named on their command line<br/>(flake.nix:2810, flake.nix:2835)
```

## AB-02.16 seccomp profile — supplemental denylist
```mermaid
flowchart TB
    COMPOSE["docker-compose.yml:128-130<br/>security_opt no-new-privileges:true<br/>seccomp=./config/seccomp-agentbox.json"]
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

## AB-02.17 shared MCP hub: bounded wait, then a loud FATAL (ADR-2034, ADR-2104)
```mermaid
sequenceDiagram
    autonumber
    participant SV as supervisord PID1
    participant HUB as program:agentbox-mcp-hub<br/>flake.nix:3294 priority=205 (:3303)
    participant WAIT as wait_for_config<br/>services/agentbox-mcp/src/hub/mod.rs:252
    participant BOOT as program:bootstrap Stage B<br/>config/entrypoint-unified.sh:3360
    participant FS as /run/agentbox/mcp-hub.json

    SV->>HUB: spawn priority=205 while Stage B is still in phase 7, file absent (flake.nix:3303)
    HUB->>WAIT: agentbox-mcp hub --wait-config-secs 120 (flake.nix:3295, default in agentbox-mcp/src/main.rs:59)
    loop poll every 500 ms (hub/mod.rs:287), log every 15 s (hub/mod.rs:284)
        WAIT->>FS: path.exists
    end
    BOOT->>FS: agentbox-manifest mcp-hub-project writes mcp-hub.json (entrypoint-unified.sh:3364-3371)
    BOOT->>FS: chown 1000:1000 and chmod 600 the projection (entrypoint-unified.sh:3374-3375)
    BOOT->>SV: supervisorctl status agentbox-mcp-hub (entrypoint-unified.sh:3377)
    alt RUNNING, waiting or serving an older config
        BOOT->>SV: supervisorctl restart agentbox-mcp-hub (:3380)
    else FATAL, EXITED, STOPPED or BACKOFF
        BOOT->>SV: supervisorctl start agentbox-mcp-hub (:3383)
    end
    BOOT->>FS: then agentbox-manifest mcp-mirror-server copies the final claude-flow entry into ~/.claude/.claude.json, idempotent, fail-open (entrypoint-unified.sh:3389-3397)
    alt config present before the timeout
        WAIT-->>HUB: Ok, HubConfig::load then serve on loopback port 9720 (hub/mod.rs:297-298)
        HUB->>HUB: a non-loopback bind is refused, not downgraded (hub/mod.rs:300-301)
    else 120 s elapsed
        WAIT-->>SV: error naming the projection and the gate, then bail (hub/mod.rs:259-282)
        Note over SV,HUB: startsecs=130 is LONGER than the wait (flake.nix:3301),<br/>so the exit is a FAILED START, retried startretries=2 (flake.nix:3302)<br/>and then parked FATAL by autorestart=unexpected (flake.nix:2813)
    end
    Note over HUB,FS: INVARIANT - the hub is loopback-only and never published (hub/mod.rs:300)
    Note over SV,BOOT: ADR-2104 - the old unbounded 600 s wait under autorestart=true made<br/>supervisorctl status read RUNNING for three days over a port never bound,<br/>docs/adr/ADR-2104-direct-control-over-mcp.md:23
```

**Invariant:** a hub that never receives its projection now fails the start rather than idling, because `startsecs=130` outlives the 120 s wait (`../project/agentbox/flake.nix:3301`, `../project/agentbox/services/agentbox-mcp/src/main.rs:59`).

**Debt:** nine MCP servers are hub-routed and exactly one has a call site in this repository, so a silent hub cost nothing that was noticed for three days (`../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md:30`).

## AB-02.18 session plane — provision, bind, reconcile, bounded restarts (ADR-2153)
```mermaid
sequenceDiagram
    autonumber
    participant SV as supervisord<br/>flake.nix:2910
    participant D as agentbox-sessions daemon<br/>services/agentbox-sessions/src/main.rs:156
    participant P as Plane<br/>services/agentbox-sessions/src/plane.rs:292
    participant R as reconcile decide<br/>services/agentbox-sessions/src/reconcile.rs:65
    participant WT as worktree ensure<br/>services/agentbox-sessions/src/worktree.rs:80
    participant TM as tmux server<br/>services/agentbox-sessions/src/tmux.rs:132

    SV->>D: spawn program:agentbox-sessions, user=devuser, startsecs 5, priority 47 (flake.nix:2913, :2917-2918)
    alt interaction_plane.enabled is false
        D-->>SV: one line and exit 0, no restart loop (agentbox-sessions/src/main.rs:160-161)
    end
    D->>TM: wait up to 120 s for the operator's tmux server (agentbox-sessions/src/main.rs:163, plane.rs:487)
    D->>P: provision() runs node scripts/session-provision.mjs, 60 s cap (agentbox-sessions/src/main.rs:166, plane.rs:505)
    D->>D: bind /run/agentbox-sessions/sessions.sock, see AB-16.7 (agentbox-sessions/src/main.rs:167)
    D->>WT: reap_legacy() once, AoE leftovers under PROJECT-worktrees (agentbox-sessions/src/main.rs:189, worktree.rs:144)
    loop each cfg.interval (agentbox-sessions/src/main.rs:195, :203)
        D->>P: reconcile_once(false) (plane.rs:292)
        P->>TM: list sessions, only those carrying @agentbox-sessions=1 are owned (tmux.rs:21)
        P->>R: decide(wanted fingerprint, seen pane, dirty, restarts) (plane.rs:321)
        alt no session for the seed
            R-->>P: Create, no dirty or backoff check (reconcile.rs:66)
        else name held by a session we do not own
            R-->>P: Hold Conflict, never touched (reconcile.rs:67-69)
        else fingerprint drift, dead pane or a trust-prompt blocker, checked in that order
            alt worktree seed with tracked or untracked changes
                R-->>P: Hold DirtyWorktree (reconcile.rs:82-86)
            else three restarts inside the hour
                R-->>P: Hold Backoff (reconcile.rs:88-90, :17-18)
            else recreate allowed
                R-->>P: Recreate (reconcile.rs:70-78, :91)
            end
        else live matching pane with a sign-in blocker
            R-->>P: Hold Blocked, reported and left alone (reconcile.rs:75-77)
        else healthy and matching the fingerprint
            R-->>P: Keep (reconcile.rs:80)
        end
        opt Create or an allowed Recreate leads to start()
            P->>WT: cut or reuse wt/aoe-SLUG, a plain directory is moved aside as .orphan-STAMP (plane.rs:268, worktree.rs:86-90)
            P->>TM: trust seeding, kill the old pane, new_session abs_SLUG (plane.rs:270-275, :276)
        end
    end
    Note over D,TM: the reaper removes a legacy worktree only when clean, no commit beyond main and every ignored path regenerable (worktree.rs:184)
```

**Invariant:** ownership is a marker, not a name: only sessions carrying `@agentbox-sessions=1` are ever recreated, so `agentbox:0` (tab0), a leftover `aoe_*` session and operator sessions are never touched, and a worktree seed holding work in progress is held rather than recreated (`../project/agentbox/services/agentbox-sessions/src/tmux.rs:21`, `../project/agentbox/services/agentbox-sessions/src/reconcile.rs:67-69`, `../project/agentbox/services/agentbox-sessions/src/reconcile.rs:82-86`).
**Invariant:** trust seeding moved into the daemon: `trust::seed` writes the root store, every wrapper-profile store that already exists and the Codex config (`../project/agentbox/services/agentbox-sessions/src/trust.rs:34-39`, `../project/agentbox/services/agentbox-sessions/src/trust.rs:102`), while `trust-seed.cjs` still writes only the root store and an existing profile store (`config/hooks/trust-seed.cjs:78-86`, `config/hooks/trust-seed.cjs:91-95`).
**Open:** ADR-2153 reads `accepted`, `partial`, `live` (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`) and calls its cutover "implementation evidence, not a cold-boot acceptance claim" (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:144`); the unused `aoe-profiles` volume stays mounted for recovery (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:145-148`).

## AB-02.19 session-provision.mjs run-as-script guard — realpath fix carried over from the AoE seeder
```mermaid
flowchart TB
    ENTRY["daemon provision() spawns node scripts/session-provision.mjs<br/>services/agentbox-sessions/src/plane.rs:505"]
    SYMLINK["/opt/agentbox/scripts is a Nix-store symlink<br/>in the baked image"]
    OLDCHECK["OLD seeder guard: path.resolve(argv[1]) === fileURLToPath(import.meta.url)<br/>compares the symlink path to the resolved store path"]
    OLDRESULT["mismatch -&gt; invokedDirectly=false<br/>main() never called, exit 0 silently<br/>observed 2026-09-06: the AoE seeder ran but provisioned nothing"]
    NEWCHECK["NOW: real(path.resolve(argv[1])) === real(fileURLToPath(import.meta.url))<br/>session-provision.mjs:211"]
    REALPATHOR["real(p) = fs.realpathSync(p), catch -&gt; p<br/>session-provision.mjs:210"]
    NEWRESULT["both sides resolve through the symlink to the same<br/>Nix store path -&gt; invokedDirectly=true -&gt; main() runs<br/>provisionOpenRouter, provisionZai, provisionOpenCode (session-provision.mjs:205)"]

    ENTRY --> SYMLINK --> OLDCHECK --> OLDRESULT
    SYMLINK --> NEWCHECK
    NEWCHECK --> REALPATHOR --> NEWRESULT

    NOTE1["INVARIANT — the guard moved with the provisioning code when the seeder was deleted;<br/>tests/cli/opencode-loom.test.mjs imports the module (:6) and spawns it as a script (:125)<br/>by a plain path, so the run-as-script branch is tested but the Nix-store symlink case is not"]
    NEWRESULT --- NOTE1
```

## AB-02.20 launch contract wrapper kind — router seed joins openrouter/zai (ADR-2080, ADR-2153)
```mermaid
flowchart TB
    K["Contract::kind<br/>services/agentbox-sessions/src/contract.rs:68"]
    K -->|"slug has config/harness-wrappers/SLUG.sh and tool is claude or custom:SLUG<br/>contract.rs:69-72"| W["kind wrapper, argv is the wrapper path<br/>config/session-plane/launch.json:13-15<br/>trust claude-profile, blockers include trust, login and API 401/403 (launch.json:16-17)"]
    K -->|"otherwise the tool name selects the template"| T["claude, codex, antigravity, opencode templates<br/>contract.rs:73-75"]
    W --> OR["openrouter -&gt; openrouter.sh"]
    W --> ZA["zai -&gt; zai.sh"]
    W --> RO["router -&gt; router.sh (ADR-2080)"]
    OR -.-> NOTE1["there is no detectAs alias any more: blocked comes from the contract's blocker patterns,<br/>busy and idle from the agent hook or the age settle (status.rs:53), so no AoE heuristic stands in for them"]
    RO -.-> NOTE2["router is its own program (config/harness-wrappers/router.sh)<br/>EXTERNAL: the router.sh console itself and its wiring are owned by AB-29"]
```

- `Contract::kind` resolves each `[[interaction_plane.session_seeds]]` slug that has a `config/harness-wrappers/<slug>.sh` and a `claude` or `custom:<slug>` tool to the `wrapper` template (`../project/agentbox/services/agentbox-sessions/src/contract.rs:68-72`), so the `router` seed (`tool = "custom:router"`, `../project/agentbox/agentbox.toml:2184`) launches exactly like `openrouter` and `zai`.
- **Drift:** two comments still describe the deleted seeder: `agentbox.toml` says `custom:router` is "resolved by the seeder's WRAPPER_SLUGS" (`../project/agentbox/agentbox.toml:2184`) and `router.sh` says it is "registered as the AoE custom_agents.router entry" (`../project/agentbox/config/harness-wrappers/router.sh:5-6`); see AB-01.11, AB-05.11 for the `[model_routing.neural]` gate that bakes `router.sh`'s artefact directory.

## AB-02.22 bootstrap.done now means the promised projections exist (ADR-2104)
```mermaid
sequenceDiagram
    autonumber
    participant SEAL as program:bootstrap-seal<br/>config/seal-bootstrap.sh:15
    participant CP as _check_projections<br/>seal-bootstrap.sh:152
    participant AM as agentbox-manifest toml-bool<br/>seal-bootstrap.sh:147
    participant FS as /run/agentbox
    participant RDY as GET /ready<br/>management-api/server.js:468

    SEAL->>CP: after every AGENTBOX_REQUIRED_FOR_READINESS program is RUNNING
    Note over CP: rows are gate:path pairs, default<br/>resources.mcp_hub.enabled:/run/agentbox/mcp-hub.json and<br/>system_one.enabled:/run/agentbox/system-one.json (ADR-2139)<br/>seal-bootstrap.sh:141
    loop each promised projection
        CP->>AM: toml-bool --path <gate>
        alt gate off or manifest unreadable
            AM-->>CP: skip the row (seal-bootstrap.sh:144-146)
        else gate on
            CP->>FS: poll for the projected file up to 300 s (seal-bootstrap.sh:137)
        end
    end
    alt a promised projection is missing
        CP-->>SEAL: log BootstrapProjectionMissing naming gate and path (seal-bootstrap.sh:169-174)
        SEAL-->>FS: exit 1, sentinel NEVER written (seal-bootstrap.sh:181-184)
    else all present
        SEAL->>FS: write bootstrap.done atomically via tmp and mv
        RDY->>FS: sentinel present, /ready can answer 200
    end
    Note over SEAL,RDY: ADR-2104 - on 2026-09-18 the entrypoint died under set -e two blocks<br/>before the hub projection and the sentinel was written anyway,<br/>so /ready said 200 over a boot that had skipped three projections,<br/>docs/adr/ADR-2104-direct-control-over-mcp.md:27
```

**Invariant:** the sentinel is written only after every gate-on projection in `_PROJECTIONS` exists, so `bootstrap.done` means more than "the daemons came up" (`../project/agentbox/config/seal-bootstrap.sh:181`).

**Open:** the projection table carries two rows today (`mcp_hub` and, since ADR-2139, `system_one`), and nothing forces a new boot-written file into it (`../project/agentbox/config/seal-bootstrap.sh:141`).

## AB-02.23 registry reconciliation at boot: skills, agents, commands (ADR-2092)
```mermaid
sequenceDiagram
    autonumber
    participant B as program:bootstrap Stage B<br/>config/entrypoint-unified.sh:3726
    participant RS as reconcile-skills.sh<br/>scripts/reconcile-skills.sh:18
    participant PR as project-skill-roots.mjs<br/>scripts/project-skill-roots.mjs:61
    participant RA as reconcile-agents.sh<br/>scripts/reconcile-agents.sh:52 SUPERSEDED_BASE
    participant RC as reconcile-commands.sh<br/>scripts/reconcile-commands.sh:33 SUPERSEDED_BASE
    participant FS as host-mounted config roots

    B->>RS: CLAUDE_SKILLS_DIR ~/.claude/skills, manifest registered-skills.txt (:3728-3733)
    RS->>FS: symlink each registered name to the baked tree (reconcile-skills.sh:63-67)
    B->>RS: second pass for Codex, ~/.codex/skills with codex-registered-skills.txt (:3740-3750)
    B->>PR: SKILL_ROOT_TARGETS workspace and workspace/project (:3759-3764)
    PR->>FS: retire an unregistered non-overlay dir to an IN-ROOT .superseded (project-skill-roots.mjs:131)
    B->>RA: AGENT_ROOT_TARGETS plus registered-agents.txt (:3779-3785)
    RA->>FS: retire a vendor-marked or category-filed agent to SUPERSEDED_BASE, OUTSIDE every scanned root (reconcile-agents.sh:212,218)
    RA->>FS: KEPT unmanaged - a flat hand-written agent with no vendor marker is reported, not retired (reconcile-agents.sh:220)
    Note over RA,RC: RESOLVED — the sidecar used to be an in-root .superseded/, but Claude Code<br/>scans agent/command roots recursively including dot-dirs, so 93 retired agents and<br/>139 retired commands were STILL loading as `.superseded:*` — both reconcilers now<br/>retire OUTSIDE every root (an agentbox-superseded dir beside the primary root's<br/>parent) and migrate any legacy in-root sidecar out on every boot, self-healing<br/>(reconcile-agents.sh:19-24,200 / reconcile-commands.sh:19-22,79-105)
    B->>RC: COMMAND_ROOT_TARGETS three roots, registered-commands.txt (:3797-3802)
    RC->>FS: prune-only sweep, kept names stay, the rest move to SUPERSEDED_BASE (reconcile-commands.sh:127)
    Note over B,FS: every reconciler is fail-open with a trailing true,<br/>so a broken registry never blocks the boot (:3733, :3785, :3802)
```

**What it shows.** The four boot-phase reconcilers that make the visible skill, agent and command sets a function of checked-in manifests rather than of whatever a vendor installer last dumped into the host mounts.
**Why it is this way.** ADR-2092: the agent roots were ungoverned, so `ruflo init` and `aqe init --auto` accreted 97 agents across two roots, 74 duplicated and 37 byte-divergent, with the nested root shadowing the user root (`config/entrypoint-unified.sh:3762-3769`).

**Invariant:** retirement is always to a recoverable sidecar, never a delete, in all three reconcilers — for agents and commands that sidecar now sits OUTSIDE every scanned root (`../project/agentbox/scripts/reconcile-agents.sh:94-105`, `../project/agentbox/scripts/reconcile-commands.sh:13-14`).

**Invariant:** a hand-written agent placed flat in the root with no vendor marker is preserved and reported, so experiments survive the sweep (`../project/agentbox/scripts/reconcile-agents.sh:220`).

**Debt:** the registries are four separate manifests in three formats (12 agents, 1 command, 20 Claude skills and 17 Codex skills), each reconciled by its own script with its own retirement rules (`../project/agentbox/agents/registered-agents.txt:1`, `../project/agentbox/config/registered-commands.txt:1`).

## AB-02.24 the role_isolation boot branch — one image, two supervisor configs (ADR-2122)
```mermaid
flowchart TB
    START["Stage A as root<br/>entrypoint-unified.sh:227 PATH sanitised to store and system dirs<br/>before the stage dispatch, in BOTH modes"]
    READ{"AGENTBOX_ROLE_ISOLATION<br/>entrypoint-unified.sh:354 _ab_toml_bool security role_isolation<br/>schema default false"}
    READY{"image can honour it?<br/>role-custody.sh:59 ab_role_isolation_ready<br/>roles config and delivery plan readable"}
    UNAV["ROLE-ISOLATION-UNAVAILABLE logged, flag forced to 0<br/>entrypoint-unified.sh:365-366"]
    subgraph OFF["role_isolation = 0 (shipped default)"]
        O1["/run/secrets chowned to devuser 0700<br/>entrypoint-unified.sh:396-398"]
        O2["host docker socket widened o+rw for devuser<br/>entrypoint-unified.sh:598-600"]
        O3["nostr.key written 0400 devuser<br/>entrypoint-unified.sh:1111-1120"]
        O4["exec supervisord -c /etc/supervisord.conf<br/>entrypoint-unified.sh:1272, :1286"]
        O5["Stage B guard /tmp/.agentbox-root<br/>entrypoint-unified.sh:245"]
        O1 --> O2 --> O3 --> O4 --> O5
    end
    subgraph ON["role_isolation = 1"]
        I1["/run/secrets kept root 0711, its own mount<br/>role-custody.sh:89 ab_secrets_root_prepare"]
        I2["ROLE vars written to /run/secrets/role/NAME 0400 and unset<br/>entrypoint-unified.sh:431 _ab_role_env_capture"]
        I3["socket left as found, ok or degraded docker-socket recorded<br/>entrypoint-unified.sh:598 _ab_docker_socket_access"]
        I4["volume and env secrets delivered per role from role-secrets.tsv<br/>role-custody.sh:218 ab_role_secrets_deliver"]
        I5["any ROLE var still in PID1 env logged as a LEAK and unset<br/>entrypoint-unified.sh:475 _ab_role_env_scrub"]
        I6["exec supervisord -c /etc/supervisord.roles.conf<br/>role-custody.sh:48 ab_supervisord_conf_pick"]
        I7["Stage B guard /run/secrets/.root-guard<br/>role-custody.sh:36 ab_root_state_dir_pick"]
        I1 --> I2 --> I3 --> I4 --> I5 --> I6 --> I7
    end
    CLAIM["Stage B claims the guard with an atomic mkdir<br/>a supervisorctl replay is a no-op exit 0<br/>entrypoint-unified.sh:266 _ab_stage_b_claim"]
    START --> READ
    READ -->|"0"| OFF
    READ -->|"1"| READY
    READY -->|"no"| UNAV --> OFF
    READY -->|"yes"| ON
    OFF --> CLAIM
    ON --> CLAIM
    ON -.-> RO["devuser shells get DOCKER_HOST=/run/docker-ro.sock<br/>GET-only, exec and run answer 403<br/>docker-read-proxy.cjs:134-136 drops to 65534 after bind"]
    OFF -.-> SIDOFF["serve-identity and docker-read-proxy print one line and exit 0<br/>identity_port/mod.rs:93-96, docker-read-proxy.cjs:144-146"]
```

**What it shows.** The single decision the custody change adds to the boot: `[security].role_isolation` is read once in Stage A and exported, and it selects between today's boot and the isolated one. Both supervisor configs, the role accounts and the delivery plan ship in every image, so the flag is boot-class: a restart flips it, a rebuild does not. The PATH sanitiser and the Stage B one-shot guard sit outside the branch and apply in both modes.
**Why it is this way.** The custody design (ADR-2122, X-1 step 1) closes three bypasses: root running devuser-planted code, devuser driving the host Docker daemon, and role secrets inherited by every process from PID 1. The first is closed unconditionally. The other two change what devuser can do, so they sit behind a flag that ships off until the owner's boot rehearsal passes. Per-role detail, the identity port and the rehearsal are in AB-36.

**Invariant:** root's boot PATH holds only `/nix/store` entries and the image's fixed system directories, in both modes, so a binary planted in the devuser-owned workspace cargo bin can no longer run as root (`../project/agentbox/config/entrypoint-unified.sh:212`, `../project/agentbox/config/entrypoint-unified.sh:227`). The cargo bin reaches devuser shells only, appended (`../project/agentbox/config/entrypoint-unified.sh:3898`).

**Invariant:** Stage B runs once per container start: a `supervisorctl start bootstrap` replay finds `stage-b.claimed` already made and exits 0 without running anything, and a missing or squatted guard directory makes Stage B refuse (`../project/agentbox/config/entrypoint-unified.sh:266-270`, `config/entrypoint-unified.sh:1315-1319`).

**Invariant:** devuser is not a member of group `root`, unconditionally: the baked group file now reads `root:x:0:` with no members, and undoing it needs a rebuild (`../project/agentbox/flake.nix:4044`).

**Invariant (under role_isolation):** no ROLE variable is in PID 1's environment when supervisord starts, so no supervised program inherits a signing key. The capture writes each to a 0400 file and unsets it, and a final scrub logs and unsets any that reappear (`../project/agentbox/config/entrypoint-unified.sh:419`, `../project/agentbox/config/entrypoint-unified.sh:1285`). With the flag off both functions return at their first line, and PID 1 still carries every `.env` key.

**Invariant (under role_isolation):** the entrypoint does not widen the host Docker socket for devuser (`../project/agentbox/config/entrypoint-unified.sh:601-601`). With the flag off it still runs `chmod o+rw` on every boot.

**Debt:** `[program:docker-read-proxy]` has no `user=` and drops privilege itself. Its comment still names a W1 follow-up to move it into `lib/role-accounts.nix` with its own account (`flake.nix:2808-2809`), but the role registry W1 shipped is `config/role-accounts.json` (`../project/agentbox/config/role-accounts.json:3`), and it has no row for the proxy.

**Drift:** `docs/BASELINE-container.md:77` calls `[security].role_isolation` boot-class, but the manifest catalogue entry is `apply_class: 'rebuild'` (`management-api/lib/system-manifest.js:97-99`) and the entrypoint comment says rebuild-class since ER-9 because the devuser sudo drop is baked (`config/entrypoint-unified.sh:351`, `flake.nix:4052-4059`).
