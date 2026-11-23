---
id: AB-02
title: Boot sequence, supervision tree and readiness
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2003, ADR-2007, ADR-2028, ADR-2029, ADR-2034, ADR-2063, ADR-2080, ADR-2092, ADR-2104, ADR-2122, ADR-2140, ADR-2144, ADR-2146, ADR-2151, ADR-2153, ADR-2210]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/config/lib/role-custody.sh
  - ../project/agentbox/config/lib/nip98-config.sh
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/config/docker-read-proxy.cjs
  - ../project/agentbox/config/role-accounts.json
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/management-api/lib/system-manifest.js
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
  - ../project/agentbox/skills/registered-skills.txt
  - ../project/agentbox/skills/codex-registered-skills.txt
  - ../project/agentbox/agents/registered-agents.txt
  - ../project/agentbox/services/agentbox-manifest/src/mcp.rs
  - ../project/agentbox/config/registered-commands.txt
  - ../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
---
Split 2026-11-23: AB-02.9 to AB-02.14 and AB-02.16 moved to [AB-45](45-boot-readiness-lifecycle-profiles-and-seccomp.md); AB-02.8 and AB-02.17 to AB-02.24 moved to [AB-46](46-boot-mcp-hub-session-plane-and-role-isolation.md). The old IDs resolve through the aliases declared there.

## AB-02.1 boot phases 1-3 — vault resolution, directories, sovereign identity
```mermaid
sequenceDiagram
    autonumber
    participant D as Docker/PID1
    participant E as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1
    participant TV as _ab_toml_val<br/>entrypoint-unified.sh:44
    participant FS as rootfs/tmpfs
    participant NPB as nostr-pod-bridge<br/>entrypoint-unified.sh:764

    D->>E: exec entrypoint-unified.sh<br/>set -euo pipefail (entrypoint-unified.sh:17)
    E->>E: PATH = _ab_root_path_sanitise(PATH) — /nix/store and fixed system dirs only, no workspace path (entrypoint-unified.sh:175, entrypoint-unified.sh:190)
    E->>E: source config/lib/role-custody.sh when readable, definitions only (entrypoint-unified.sh:239-242)
    Note over E: ADR-2028 vault resolve runs before Phase 1<br/>every supervised program inherits PID1 env
    E->>TV: _ab_vault_resolve() (entrypoint-unified.sh:89) calls _ab_toml_val vault root (entrypoint-unified.sh:90)
    TV->>FS: awk anchored-section parse of AGENTBOX_CONFIG (/etc/agentbox.toml)
    alt VAULT_ROOT empty (entrypoint-unified.sh:91)
        E->>E: unset VAULT_* export AGENTBOX_VAULT_ENABLED=0 (entrypoint-unified.sh:92-94)
        E-->>D: echo "[vault] disabled — no [vault] in agentbox.toml" (entrypoint-unified.sh:95), return (entrypoint-unified.sh:96)
        Note over E: no ONTOLOGY_PAGES_DIR branch any more - the pre-vault variable and its<br/>AGENTBOX_VAULT_LEGACY_PATHS opt-in were removed 2026-10-31 (entrypoint-unified.sh:77-84)
    else VAULT_ROOT set (entrypoint-unified.sh:98)
        E->>E: VAULT_PAGES = root plus the optional pages folder, where absent, . and ./ all mean the root itself (entrypoint-unified.sh:102-104, ADR-OM-007 flatten), HP_LOOM_SYNC default check from vault.loom_sync (entrypoint-unified.sh:114)<br/>export VAULT_ROOT/VAULT_PAGES/VAULT_FORMAT/VAULT_TUI/VAULT_WORKING_ROOT/VAULT_TRANSCRIPTS/HP_LOOM_SYNC/LOOM_URL/LOOM_HP_HOST AGENTBOX_VAULT_ENABLED=1 (entrypoint-unified.sh:135-136)
    end
    E->>E: AGENTBOX_ROLE_ISOLATION = _ab_toml_bool security role_isolation, exported to PID1 (entrypoint-unified.sh:317)
    alt role_isolation = 1 and ab_role_isolation_ready (entrypoint-unified.sh:324)
        E->>FS: ab_secrets_root_prepare /run/secrets root 0711, guard moves to /run/secrets/.root-guard (entrypoint-unified.sh:325-326)
    else flag on but roles config or plan missing
        E-->>D: ROLE-ISOLATION-UNAVAILABLE, boot continues as role_isolation = 0 (entrypoint-unified.sh:328-329)
    end
    E->>FS: _ab_root_state_dir_prepare AB_ROOT_STATE_DIR — arms Stage B's one-shot guard (entrypoint-unified.sh:335)
    E-->>D: echo "[1/8] Preparing runtime directories..." (entrypoint-unified.sh:338)
    E->>FS: mkdir -p WORKSPACE, RUVECTOR_DATA_DIR, SOLID_POD_ROOT, /run/secrets... (entrypoint-unified.sh:339-350)
    opt role_isolation off (entrypoint-unified.sh:359)
        E->>FS: chmod 0700 /run/secrets, chown 1000:1000 (entrypoint-unified.sh:360-361)
    end
    E->>E: _ab_role_env_capture — flag on, each ROLE var to /run/secrets/role/NAME 0400 and unset (entrypoint-unified.sh:460)
    E->>FS: _ab_pg_credential_prepare - RUVECTOR_PG_PASSWORD written once to /run/secrets/ruvector-pg/password 0600 devuser,<br/>public HOST/PORT/DATABASE/USER and PASSWORD_FILE exported, the password and any CONNINFO unset (entrypoint-unified.sh:473-507)
    E->>FS: chown 1000:1000 volume roots, non-recursive (entrypoint-unified.sh:583-591)
    E->>FS: _ab_docker_socket_access — flag off chmod o+rw the host socket, flag on leave it and record ok or degraded docker-socket (entrypoint-unified.sh:606, entrypoint-unified.sh:632)
    E-->>D: echo "[2/8] Bootstrapping sovereign mesh identity..." (entrypoint-unified.sh:749)
    E->>NPB: nostr-pod-bridge bootstrap (entrypoint-unified.sh:764)
    Note right of NPB: k256 + nostr-bbs-core keypair, pod ACL/DID docs,<br/>writes /run/agentbox/identity.env 0600 — self-gates on<br/>sovereign_mesh.enabled, silent on success
    E-->>D: echo "[3/8] Ensuring workspace defaults..." (entrypoint-unified.sh:785)
    E->>FS: mkdir WORKSPACE/agents if absent (entrypoint-unified.sh:787)
    E->>FS: ln -sf /home/devuser/.claude WORKSPACE/.claude (entrypoint-unified.sh:800)
    opt DREAM_CMD_SRC exists and no dream.md (entrypoint-unified.sh:809-818)
        E->>FS: cp dream.md to /home/devuser/.claude/commands/ (entrypoint-unified.sh:818)
    end
    opt skill-creator not yet registered (entrypoint-unified.sh:834)
        E->>E: agentbox-manifest plugin-register --key skill-creator@claude-plugins-official (entrypoint-unified.sh:834-839)
    end
    opt codex plugin baked and not registered (entrypoint-unified.sh:859)
        E->>E: agentbox-manifest plugin-register --key codex@openai-codex (entrypoint-unified.sh:859-864)
    end
    opt voicelink enabled and record_events, Codex config present (entrypoint-unified.sh:907-913)
        E->>FS: reconcile a top-level notify line naming voice-paned into the Codex config, an operator-set notify is never replaced (entrypoint-unified.sh:913-940, ADR-2151)
    end
```

## AB-02.2 boot phases 4-6 — provisioning, artifact validation, supervisord exec, Stage B closure probes
```mermaid
sequenceDiagram
    autonumber
    participant E as entrypoint-unified.sh<br/>config/entrypoint-unified.sh:970-971
    participant AM as agentbox-manifest<br/>entrypoint-unified.sh:971
    participant FS as rootfs/tmpfs
    participant SV as supervisord<br/>entrypoint-unified.sh:1299
    participant B as [program:bootstrap] Stage B<br/>entrypoint-unified.sh:1304

    E-->>E: echo "[4/8] Provisioning agent stacks..." (entrypoint-unified.sh:970)
    E->>AM: agentbox-manifest provision-stacks (entrypoint-unified.sh:971), which adds the session-state hooks to each profile when the session plane is on (stacks.rs:327)
    E->>FS: chown -R 1000:1000 WORKSPACE/profiles (entrypoint-unified.sh:980)
    E-->>E: echo "[5/8] Validating runtime closure..." (entrypoint-unified.sh:987)
    E->>FS: bash validate-artifacts.sh (entrypoint-unified.sh:988)
    alt validate-artifacts.sh fails (entrypoint-unified.sh:988)
        E-->>E: fatal BootstrapFailed, exit 1 (entrypoint-unified.sh:989-992)
    end
    E->>FS: mkdir /run/agentbox /run/agentbox/hooks (entrypoint-unified.sh:1004) — bootstrap-seal writes here later
    E->>FS: when interaction_plane.enabled, mkdir /run/agentbox-sessions, chown 1000:1000, chmod 0755 — the daemon binds its socket in it (entrypoint-unified.sh:999-1002, ADR-2153)
    Note over E: ADR-2153 retired the N-05 AoE token-directory check here (mode 700 owner 1000, the N-05-VIOLATION marker): there is no token file, authorisation is the socket peer uid, see AB-16.7
    Note over E,FS: ADR-2040 — code-server and jupyter-lab credential minting<br/>runs here (entrypoint-unified.sh:1021-1095), between the session-directory prep and the identity.env source<br/>below. Not re-diagrammed here — see AB-06.7 and AB-07.9
    E->>FS: source /run/agentbox/identity.env into PID1 env (entrypoint-unified.sh:1105-1106)
    Note over E: under role_isolation identity.env carries public lines only, no AGENTBOX_NSEC<br/>or AGENTBOX_BRIDGE_SK (entrypoint-unified.sh:1108-1111), see AB-36
    E->>FS: _ab_role_key_file_own — flag on, the bridge key file goes to ab-identity 0400 (entrypoint-unified.sh:1112)
    alt AGENTBOX_BRIDGE_SK set and role_isolation off (entrypoint-unified.sh:1125)
        E->>FS: write /run/secrets/nostr.key 0400 devuser, unset AGENTBOX_BRIDGE_SK from env (entrypoint-unified.sh:1124-1133)
    end
    Note over E: ADR-2028 D3 — ontology PUSH cache refresh (Phase 5c)
    alt ONTO_PAGES = VAULT_PAGES only (entrypoint-unified.sh:1151) is empty (entrypoint-unified.sh:1163)
        E-->>E: echo "[5c/8] ontology PUSH cache refresh skipped ([vault] disabled)" (entrypoint-unified.sh:1164)
    else ONTO_PAGES dir exists and builder present (entrypoint-unified.sh:1165)
        E->>FS: run_as_devuser node ontology-index-build.js (entrypoint-unified.sh:1167-1169)
    end
    Note over E: Phase 5d(ii) — vault CLI liveness gate (ADR-2107/2108):<br/>the corpus's only programmatic door now, so a missing/broken<br/>binary is fail-LOUD, never fail-fatal (entrypoint-unified.sh:1173-1192)
    alt AGENTBOX_VAULT_ENABLED=1 (entrypoint-unified.sh:1195)
        E->>FS: /opt/agentbox/bin/vault --version liveness probe (entrypoint-unified.sh:1196-1198)
        alt probe succeeds
            E-->>E: echo "[5d/8] vault OK — version (bin)" (entrypoint-unified.sh:1199)
        else binary present, probe fails
            E-->>E: echo "[5d/8] vault FAILED its liveness probe — rebuild the image" (entrypoint-unified.sh:1201-1204)
        else binary absent
            E-->>E: echo "[5d/8] vault MISSING — [vault].cli on, no binary baked" (entrypoint-unified.sh:1205-1208)
        end
    else vault disabled
        E-->>E: echo "[5d/8] [vault] liveness gate skipped ([vault] disabled)" (entrypoint-unified.sh:1211)
    end
    Note over E: Phase 5d(iii) — HP Loom reconcile (ADR-2140), never waited on (entrypoint-unified.sh:1214-1219)
    alt AGENTBOX_VAULT_ENABLED=1 and HP_LOOM_SYNC not off (entrypoint-unified.sh:1222)
        E-->>E: echo the previous receipt, state at ts, detail, from hp-loom-reconcile.json (entrypoint-unified.sh:1227)
        alt /opt/agentbox/scripts/hp-loom-reconcile.sh executable (entrypoint-unified.sh:1229)
            E->>FS: background run_as_devuser, timeout 1800, --mode check or full, log to hp-loom-reconcile.log (entrypoint-unified.sh:1230-1231)
            E-->>E: echo "[5d/8] HP Loom reconcile started in the background" (entrypoint-unified.sh:1232)
        else script missing
            E-->>E: echo "[5d/8] HP Loom reconcile SKIPPED - rebuild the image" (entrypoint-unified.sh:1234)
        end
    else loom_sync off or vault disabled
        E-->>E: echo "[5d/8] HP Loom reconcile skipped" (entrypoint-unified.sh:1238)
    end
    opt AGENTBOX_TAB0_BRIDGE_SUPERVISED=1 and BRIDGE_TOKEN unset (entrypoint-unified.sh:1252)
        E->>FS: generate/read BRIDGE_TOKEN, write 0600 secrets file (entrypoint-unified.sh:1253-1268)
    E->>E: eval agentbox-manifest system-one-project --format shell into PID1 env before supervisord, so sibling services see the typed-decision slot (entrypoint-unified.sh:1271-1275, ADR-2139)
    end
    opt role_isolation = 1 (entrypoint-unified.sh:1285)
        E->>FS: ab_role_secrets_deliver /etc/agentbox/role-secrets.tsv into /run/secrets/role (entrypoint-unified.sh:1287)
        E->>FS: append ok or degraded secrets-delivery to /run/secrets/role-isolation.state (entrypoint-unified.sh:1288-1294)
        E->>E: _AB_SUPERVISORD_CONF = ab_supervisord_conf_pick 1, the roles config (entrypoint-unified.sh:1293)
    end
    E-->>E: echo "[5b/8] Starting supervisord..." (entrypoint-unified.sh:1296)
    E->>E: _ab_role_env_scrub — flag on, any ROLE var still in PID1 env is a LEAK, unset (entrypoint-unified.sh:1298)
    E->>SV: exec supervisord -c _AB_SUPERVISORD_CONF -n (entrypoint-unified.sh:1299)
    Note over E,SV: Stage A process image is REPLACED by supervisord (exec) —<br/>PID1 is now supervisord, not the shell
    SV->>B: spawn [program:bootstrap] with AGENTBOX_BOOTSTRAP_STAGE=B (flake.nix, not this file)
    B->>B: re-export WORKSPACE/RUVECTOR_DATA_DIR/SOLID_POD_ROOT/RUVECTOR_PORT/AGENTBOX_CONFIG/SHARED_PROJECTS_ROOT defaults (entrypoint-unified.sh:1311-1316)
    alt AGENTBOX_VAULT_ENABLED unset (entrypoint-unified.sh:1319)
        B->>B: _ab_vault_resolve() again (standalone-invocation fallback) (entrypoint-unified.sh:1319)
    end
    B->>B: _ab_stage_b_claim AB_ROOT_STATE_DIR — atomic mkdir stage-b.claimed (entrypoint-unified.sh:1328)
    alt claim made
        B-->>SV: Stage B claimed, one-shot (entrypoint-unified.sh:1330)
    else already claimed this container start
        B-->>SV: no-op exit 0, a supervisorctl replay runs nothing (entrypoint-unified.sh:1331)
    else guard dir missing or not root 0700
        B-->>SV: REFUSED exit 1 (entrypoint-unified.sh:1332)
    end
    B-->>SV: echo "[6/8] Validating pre-packaged service closures..." (entrypoint-unified.sh:1346)
    loop _probe_closure for management-api, mcp, gated toolchains (entrypoint-unified.sh:1348-1361)
        B->>FS: test -d node_modules under each closure dir
        alt node_modules missing (entrypoint-unified.sh:1357)
            B-->>SV: fatal MissingArtifactDetected, exit 1 (entrypoint-unified.sh:1357-1360)
        end
    end
    B-->>SV: echo "[6/8] Service closures OK." (entrypoint-unified.sh:1387)
```

## AB-02.3 boot phases 7-8 — ruflo plugins, manifest projection, runtime-env publish
```mermaid
sequenceDiagram
    autonumber
    participant B as [program:bootstrap] Stage B<br/>entrypoint-unified.sh:1461
    participant AM as agentbox-manifest<br/>entrypoint-unified.sh:2275
    participant MCP as .mcp.json<br/>entrypoint-unified.sh:1557
    participant FS as rootfs/tmpfs
    participant RT as runtime-env.sh<br/>entrypoint-unified.sh:4083
    participant TS as trust-seed.cjs<br/>run once from entrypoint-unified.sh:1818

    B-->>B: echo "[7/8] Bootstrapping ruflo plugins..." (entrypoint-unified.sh:1461)
    B->>FS: mkdir ~/.claude-flow/plugins, /var/cache/ruflo-plugins (entrypoint-unified.sh:1463-1467)
    Note over B: the claude-flow config template that carried the Postgres password is gone<br/>(entrypoint-unified.sh:1469-1472), nothing writes ~/.claude-flow/config.json's memory block now
    Note over B: PRD-018/ADR-036 D6 — read RuVector memory gate flags<br/>via _ab_toml_bool memory_learning.* (entrypoint-unified.sh:1495-1507), fail-open all-off
    B->>MCP: ensure .mcp.json points at ruvector-mcp.cjs with RUVECTOR_PG_HOST/PORT/DATABASE/USER<br/>and RUVECTOR_PG_PASSWORD_FILE, never a password, idempotent (entrypoint-unified.sh:1525-1590)
    B->>MCP: inject RuVector memory gates, including evidence_refs and read_counts (entrypoint-unified.sh:1592-1675)
    opt browser-gpu sidecar reachable, its /health answers 200 or 503 while Chrome restarts (entrypoint-unified.sh:2248-2251)
        B->>AM: agentbox-manifest mcp-set-server --name browser-gpu (entrypoint-unified.sh:2253)
    end
    B->>AM: agentbox-manifest mcp-reconcile-aqe --provider "$_MR_PROVIDER_ARG" (entrypoint-unified.sh:2275)
    opt _MR_ENABLED=1 — ADR-041 model routing (entrypoint-unified.sh:2285)
        B->>AM: run_as_devuser agentbox-manifest model-routing-project --manifest AGENTBOX_CONFIG --workspace WORKSPACE<br/>(entrypoint-unified.sh:2285-2289)
        Note right of AM: projects [model_routing.routes] into every<br/>.agentic-qe/llm-config.json under the workspace,<br/>a runtime file the repository does not track
    end
    Note over B: ADR-069, ADR-2210 — interaction_plane.proxy projected every boot, and never left as an absent file (entrypoint-unified.sh:2333-2352)
    opt nip98-config.sh readable, else the older direct call (entrypoint-unified.sh:2343,entrypoint-unified.sh:2347-2351)
        B->>AM: ab_nip98_project sources the helper and calls agentbox-manifest nip98-config --manifest AGENTBOX_CONFIG --out .agentbox/nip98-proxy-config.json<br/>(entrypoint-unified.sh:2345-2346, nip98-config.sh:24)
        B->>FS: section absent writes the explicit empty config, then chown 1000:1000 and chmod 600 (nip98-config.sh:28-33)
    end
    Note over B: DEBT - the proxy program starts alongside bootstrap (flake.nix:2878-2884), so it can serve the previous boot's file until this block runs,<br/>and a failed projection leaves that last-good file serving, or no file so the door stays closed (nip98-config.sh:21,nip98-config.sh:27)
    Note over B: hook registration reconciled into root settings.json every boot, each fail-open (|| true)
    B->>FS: decline-recorder on Stop, SubagentStop and UserPromptSubmit when claude_code.floor is on, default on<br/>(entrypoint-unified.sh:2101-2160, ADR-2144)
    B->>FS: voice-paned record on Stop and Notification when voicelink.enabled and record_events<br/>(entrypoint-unified.sh:2162-2204, ADR-2151)
    B->>FS: session-state.cjs hook on UserPromptSubmit, Stop and Notification into root settings.json when<br/>interaction_plane.enabled, pruned when off (entrypoint-unified.sh:1950-1962, ADR-2153)
    B->>FS: Codex config.toml notify line set to session-state.cjs codex, keeping --voice when voicelink records, restored when<br/>off (entrypoint-unified.sh:1927-1946)
    B->>FS: activity-sink hooks when [integrations.activity_sink] has a usable url (entrypoint-unified.sh:2889-2975, ADR-2146)
    B->>AM: agentbox-manifest instructions-project, with --managed-out /etc/claude-code/CLAUDE.md when the floor is on,<br/>--no-managed otherwise (entrypoint-unified.sh:3030-3032, ADR-2144)
    Note over B: ADR-2139 vendor slot — _ab_toml_bool system_one enabled (entrypoint-unified.sh:2684) selects system-one-project<br/>(entrypoint-unified.sh:2687-2697), else sso-project (entrypoint-unified.sh:2714-2716)
    opt [system_one].enabled and the projection is non-empty (entrypoint-unified.sh:2686, entrypoint-unified.sh:2689)
        B->>AM: agentbox-manifest system-one-project --format json, then shell and plugin-config (entrypoint-unified.sh:2687,<br/>entrypoint-unified.sh:2695-2697)
        B->>FS: atomic tmp-and-mv /run/agentbox/system-one.json, the proxy's only config (entrypoint-unified.sh:2692-2693)
    end
    Note over B: unusable slot writes a disabled stub and falls back to sso-project, so the seal's promised file always exists<br/>(entrypoint-unified.sh:2699-2707)
    Note over B: RESOLVED — the ontology-bridge MCP registration that used to sit<br/>here (gated ENABLE_ONTOLOGY) is GONE (ADR-2107/2108): agents reach the<br/>corpus only through the vault CLI and the Loom over HTTP (entrypoint-unified.sh:2358-2361)
    opt AGENTBOX_TRUST_SEED not 0 and node present (entrypoint-unified.sh:1818)
        B->>TS: node trust-seed.cjs marks workspace root and worktrees trusted, once (entrypoint-unified.sh:1818-1820)
        Note right of TS: DOC-DRIFT resolved — deliberately NOT a SessionStart hook any<br/>more (it walked ~1,170 paths per session start, avg 3.2s, and raced<br/>Claude Code's own ~/.claude.json writes) — hook shim reconcile below prunes<br/>any stale registration
    end
    Note over B: after the hub projection (see AB-46.2) the claude-flow entry is mirrored into user scope<br/>(entrypoint-unified.sh:3460-3468)
    B->>AM: agentbox-manifest mcp-mirror-server --from .mcp.json --to ~/.claude/.claude.json --name claude-flow<br/>(entrypoint-unified.sh:3466-3467, mcp.rs:275)
    B->>FS: RuVector credential scrub - every boot removes any RUVECTOR_PG_CONNINFO or memory.password an earlier boot<br/>left in .mcp.json, ~/.claude.json, Codex config.toml, ~/.claude-flow/config.json and their backups,<br/>atomic rewrite, fail-open (entrypoint-unified.sh:3907-4065)
    B-->>B: echo "[8/8] Publishing environment hints..." (entrypoint-unified.sh:4070)
    B->>RT: cat RUNTIME_ENV_FILE=/run/agentbox/runtime-env.sh heredoc (entrypoint-unified.sh:4083-4170)
    Note right of RT: role_isolation on adds DOCKER_HOST=unix:///run/docker-ro.sock,<br/>the GET-only read proxy, to devuser shells (entrypoint-unified.sh:4078-4081)
    Note right of RT: exports WORKSPACE, RUVECTOR_PG_HOST/PORT/DATABASE/USER/PASSWORD_FILE (entrypoint-unified.sh:4088-4092),<br/>VAULT_ROOT/PAGES/FORMAT/TUI/WORKING_ROOT/TRANSCRIPTS, AGENTBOX_INTERACTION_PLANE_*,<br/>CUDA_PATH and friends are appended after the heredoc (entrypoint-unified.sh:4216-4226) (see AB-45.3)
    B->>FS: _ab_quarantine_stale_cargo_bins, ELF bins in WORKSPACE/.cargo/bin whose<br/>/nix/store loader was collected move to .stale-loader/, numbered, never deleted (entrypoint-unified.sh:4179-4197)
    Note right of FS: ADR-2029 D4 follow-up (2026-10-01) - .cargo/bin outlives rebuilds, so a dead loader<br/>once shadowed baked rune and agentbox-hook. Custody W0 now APPENDS it, devuser shells only,<br/>never root (entrypoint-unified.sh:4163-4165). Loader path read from bytes, nothing executed<br/>(entrypoint-unified.sh:4185-4187)
    B->>FS: ln -sf RUNTIME_ENV_FILE /etc/profile.d/agentbox-runtime.sh best-effort (entrypoint-unified.sh:4243)
    B->>FS: cp RUNTIME_ENV_FILE to durable WORKSPACE/.agentbox-runtime-env.sh (entrypoint-unified.sh:4246)
    B->>FS: write fish conf.d/agentbox-runtime.fish sourcing the env file (entrypoint-unified.sh:4252-4271)
```


## AB-02.4 supervision tree — core and identity programs
```mermaid
flowchart TB
    BOOT["program:bootstrap<br/>flake.nix:2878<br/>no user= line -&gt; runs as root<br/>priority=5 autorestart=false one-shot"]
    MGMT["program:management-api<br/>flake.nix:2892<br/>user=devuser bind 0.0.0.0:ENV_MANAGEMENT_API_PORT default 9090<br/>priority=20 REQUIRED_FOR_READINESS=true"]
    CRED["program:claude-cred-sync (ADR-2118)<br/>flake.nix:3535<br/>gate toolchains.claude_code, converges ~/.claude/.credentials.json<br/>both ways with the host bind; exits 0 (stopped) if the bind is absent<br/>user=devuser priority=30 autorestart=unexpected"]
    SEAL["program:bootstrap-seal<br/>flake.nix:2908<br/>user=devuser priority=99 autorestart=false one-shot<br/>writes /run/agentbox/bootstrap.done, timeout 120s"]
    SOLID["program:solid-pod<br/>flake.nix:2920<br/>gate sovereign_mesh.enabled and local-solid-rs active<br/>user=devuser priority=30 REQUIRED_FOR_READINESS=true"]
    HTTPSB["program:https-bridge<br/>flake.nix:2937<br/>gate sovereign_mesh.enabled and https_bridge<br/>user=devuser priority=32<br/>self-signed TLS to plain-HTTP target, https-bridge/https-proxy.js<br/>full cert-provisioning and forwarding flow: see AB-30.4, AB-30.4"]
    GATERELAY{"podBridgeEnabled? ADR-2003<br/>flake.nix:1890 relayLocal and pod_bridge"}
    NRELAY1["program:nostr-relay native<br/>flake.nix:2976<br/>bind 127.0.0.1:7777 (AGENTBOX_RELAY_BIND)<br/>user=devuser, ab-identity under role_isolation priority=35 REQUIRED_FOR_READINESS=false"]
    NRELAY2["program:nostr-relay nostr-rs-relay<br/>flake.nix:2988<br/>config /etc/agentbox/nostr-relay.toml<br/>user=devuser, ab-identity under role_isolation priority=35 REQUIRED_FOR_READINESS=false"]
    NGW["program:nostr-gateway<br/>flake.nix:2651<br/>user=devuser, ab-gateway under role_isolation priority=234<br/>off switch AGENTBOX_NOSTR_GATEWAY=0"]
    DRP["program:docker-read-proxy (custody W0)<br/>flake.nix:3041<br/>no user= -&gt; starts root, env -i, binds /run/docker-ro.sock,<br/>then drops itself to uid 65534 plus the socket's group<br/>flag off: one line, exit 0, EXITED is expected priority=20"]
    SID["program:serve-identity (custody W3)<br/>flake.nix:3063<br/>env -i, nostr-pod-bridge serve-identity, the identity port<br/>user=devuser here, ab-identity in supervisord.roles.conf<br/>flag off: exits 0 without reading a key priority=30"]
    ISO["/etc/supervisord.roles.conf<br/>flake.nix:4255 agentbox-manifest role-accounts isolate<br/>a pure function of this config and config/role-accounts.json:<br/>only user= and environment= of role programs differ"]
    TSD["program:tailscaled<br/>flake.nix:3014<br/>gate networking.tailscale, no user= -&gt; root<br/>socket /var/run/tailscale priority=15"]
    TSU["program:tailscale-up<br/>flake.nix:3029<br/>gate networking.tailscale, no user= -&gt; root<br/>priority=16 autorestart=false one-shot"]

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
    DRP -.-> N1["DEBT: the comment still says TODO(custody W1) move into the role registry<br/>lib/role-accounts.nix (flake.nix:3050), but the registry is<br/>config/role-accounts.json and the proxy has no account in it"]
```

## AB-02.6 supervision tree — gated toolchain programs
```mermaid
flowchart TB
    QGIS["program:qgis-mcp<br/>flake.nix:2594<br/>gate spatial.qgis, TCP proxy to gui-tools-service:9877<br/>user=devuser priority=230"]
    BLEND["program:blender-mcp<br/>flake.nix:2621<br/>gate spatial.blender, bridges 127.0.0.1:9876 to external GUI sidecar<br/>user=devuser priority=231"]
    JLAB["program:jupyter-lab<br/>flake.nix:2694<br/>gate data_science.jupyter, bind 0.0.0.0:8888 — RESOLVED ADR-2040,<br/>JUPYTER_TOKEN now minted at boot (see AB-07.9), no longer tokenless<br/>user=devuser priority=232"]
    IMGM["program:imagemagick-mcp<br/>flake.nix:2951<br/>gate media.imagemagick<br/>user=devuser priority=210"]
    COMFY["program:comfyui-builtin<br/>flake.nix:3116<br/>gate media.comfyui_builtin, bind 127.0.0.1:8188<br/>user=devuser priority=220"]
    OPF["program:opf-router<br/>flake.nix:3001<br/>gate privacyFilterEnabled, OPF_PORT default 9092<br/>user=devuser priority=240"]
    DREAM["program:dream-engine<br/>flake.nix:3139<br/>gate dreamEngineEnabled (flake.nix:1940), LOOM_URL default a LAN address, port 8084<br/>user=devuser priority=230"]
    SIDEP["program:sidestr-producer<br/>flake.nix:3374<br/>gate sidechain.enabled (flake.nix:424), shipped true (agentbox.toml:2030, re-enabled on recovered validated prefixes after the ENOSPC failure, agentbox.toml:2029) so the producer block is emitted; runs the baked upstream, refuses on a pin mismatch<br/>user=devuser, ab-sidestr-dreamlab under role_isolation priority=260 startretries=5"]
    SIDEM["program:sidestr-mirror<br/>flake.nix:3392<br/>gate sidechain.mirror, only with enabled (flake.nix:425)<br/>user=devuser priority=261"]
    SIDEF["program:sidestr-faucet<br/>flake.nix:3409<br/>gate sidechain.faucet, only with enabled (flake.nix:426)<br/>user=devuser, ab-faucet-dreamlab under role_isolation priority=262"]
    CODES["program:code-server<br/>flake.nix:3092<br/>gate toolchains.code_server, bind 0.0.0.0:8080 — RESOLVED ADR-2040,<br/>auth password with a boot-minted credential (see AB-07.9), not auth none<br/>user=devuser priority=50"]

    SO1["program:system-one-proxy (ADR-2139)<br/>flake.nix:3284<br/>gate system_one.enabled (flake.nix:417), typed-decision proxy (loopback; configured 0.0.0.0:8094 on the docker networks when system_one.network, agentbox.toml:915-916, live with the next image),<br/>reads only /run/agentbox/system-one.json, vendor keys from env<br/>user=devuser priority=60 autorestart=unexpected"]
    SEO["program:seo-watch-cron<br/>flake.nix:3331<br/>gate seo_watch.enabled (flake.nix:410), supercronic over the website repo crontab<br/>user=devuser priority=250"]
    TGC["program:target-gc (ADR-2138)<br/>flake.nix:3623<br/>gate resources.target_gc.enabled (flake.nix:345), idle Cargo target sweeper<br/>user=devuser priority=246"]
    OPF -.->|"privacy filter mode gate"| DREAM
    SIDEP -->|"block files"| SIDEM
    SIDEP -->|"local producer"| SIDEF
    DREAM -.->|"ZAI_ANTHROPIC_API_KEY inherited from PID1; the memory DB password is read from<br/>the 0600 file named by RUVECTOR_PG_PASSWORD_FILE, flake.nix:3134-3138"| RVNOTE["note: secrets never written<br/>into generated supervisor text"]
    UCRON["program:user-cron<br/>flake.nix:3305<br/>gate user_cron.enabled (flake.nix:411), shipped true (agentbox.toml:1311), supercronic over<br/>~/workspace/.local/etc/agentbox/user.crontab, idle until the file exists<br/>user=devuser priority=250"]
```

## AB-02.7 supervision tree — desktop stack and interaction plane
```mermaid
flowchart TB
    GATESTACK{"desktop.stack ADR-2003<br/>flake.nix:301-302, Nix if/else-if/else"}
    HYPR["program:hyprland<br/>flake.nix:2771<br/>user=devuser priority=40"]
    XWAY["program:xwayland-session<br/>flake.nix:2785<br/>user=devuser priority=41"]
    WAYVNC["program:wayvnc<br/>flake.nix:2796<br/>bind 0.0.0.0:5901 user=devuser priority=42"]
    XORG["program:xorg-nvidia<br/>flake.nix:2807<br/>no user= -&gt; root priority=40"]
    I3A["program:i3wm xorg-nvidia branch<br/>flake.nix:2817<br/>user=devuser priority=41"]
    X11VNC["program:x11vnc<br/>flake.nix:2828<br/>bind 0.0.0.0:5901 user=devuser priority=42"]
    XVNC["program:xvnc<br/>flake.nix:2839<br/>rfbport 5901, SecurityTypes None, user=devuser (flake.nix:2841) priority=40"]
    I3B["program:i3wm i3-x11 default branch<br/>flake.nix:2849<br/>user=devuser priority=41"]
    SESS["program:agentbox-sessions<br/>flake.nix:3153<br/>gate interaction_plane.enabled (flake.nix:521), no TCP listener, unix socket /run/agentbox-sessions/sessions.sock<br/>user=devuser priority=47 startsecs=5"]
    NIP98["program:nip98-proxy<br/>flake.nix:3166<br/>bind 0.0.0.0:9096, named operator routes only, unknown routes 404 (proxy.mjs), environment NIP98_PROXY_REQUIRE_CONFIG=1 at flake.nix:3170 so an absent config file refuses every identity, and a changed one exits 0 for a restart (proxy.mjs:1112-1115)<br/>user=devuser, ab-ingress under role_isolation priority=46"]
    TAB0["program:tab0-bridge<br/>flake.nix:3196<br/>gate sovereign_mesh.enabled<br/>user=devuser priority=236"]
    TMUX["program:tmux-autostart<br/>flake.nix:3209<br/>user=devuser priority=95 autorestart=false one-shot"]

    GATESTACK -->|"hyprland-wayland"| HYPR --> XWAY --> WAYVNC
    GATESTACK -->|"xorg-nvidia"| XORG --> I3A --> X11VNC
    GATESTACK -->|"i3-x11 default"| XVNC --> I3B
    TAB0 -->|"session socket"| SESS
    NIP98 -.->|"/mgmt/* forwarded"| MGMTREF["management-api<br/>see AB-02.4"]
    TAB0 -.->|"AGENTBOX_TAB0_BRIDGE_SUPERVISED=1"| TMUX
```
