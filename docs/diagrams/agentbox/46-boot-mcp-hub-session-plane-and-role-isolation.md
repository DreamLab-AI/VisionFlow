---
id: AB-46
title: Boot — shared MCP hub, session plane, launch contract, registry reconciliation and role isolation
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2034, ADR-2063, ADR-2080, ADR-2092, ADR-2104, ADR-2122, ADR-2153]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/config/lib/role-custody.sh
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
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/scripts/reconcile-skills.sh
  - ../project/agentbox/scripts/project-skill-roots.mjs
  - ../project/agentbox/scripts/reconcile-agents.sh
  - ../project/agentbox/scripts/reconcile-commands.sh
  - ../project/agentbox/skills/registered-skills.txt
  - ../project/agentbox/skills/codex-registered-skills.txt
  - ../project/agentbox/agents/registered-agents.txt
  - ../project/agentbox/config/registered-commands.txt
  - ../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
aliases:
  AB-02.8: AB-46.1
  AB-02.17: AB-46.2
  AB-02.18: AB-46.3
  AB-02.19: AB-46.4
  AB-02.20: AB-46.5
  AB-02.22: AB-46.6
  AB-02.23: AB-46.7
  AB-02.24: AB-46.8
---
Split from AB-02 on 2026-11-23: PID1 and the bootstrap seal, and the later boot-time additions (formerly AB-02.8 and AB-02.17 to AB-02.24). Old panel IDs resolve through the `aliases:` map above; the stamp is AB-02's, and the panels were not re-read.

## AB-46.1 PID1 root, priority ordering and bootstrap-seal
```mermaid
sequenceDiagram
    autonumber
    participant SV as supervisord PID1<br/>flake.nix:2863 supervisord section nodaemon=true
    participant BOOT as program:bootstrap<br/>flake.nix:2878 no user= line means root
    participant MGMT as program:management-api<br/>flake.nix:2892 user=devuser flake.nix:2895 priority=20 flake.nix:2899
    participant SOLID as program:solid-pod<br/>flake.nix:2920 user=devuser flake.nix:2923 priority=30 flake.nix:2927
    participant SEAL as program:bootstrap-seal<br/>agentbox/config/seal-bootstrap.sh priority=99
    participant SC as supervisorctl status<br/>seal-bootstrap.sh:100

    Note over SV: supervisord itself inherits root from the exec'd<br/>entrypoint-unified.sh Stage A (entrypoint-unified.sh:1299)
    SV->>BOOT: spawn priority=5 flake.nix:2884, environment AGENTBOX_BOOTSTRAP_STAGE=B (flake.nix:2880)
    Note right of BOOT: no user= line — Stage B (phases 6-8) runs as ROOT,<br/>needed for chown/mkdir under devuser volumes
    SV->>MGMT: spawn priority=20 flake.nix:2899, user=devuser (flake.nix:2895)
    SV->>SOLID: spawn priority=30, user=devuser, gated sovereign_mesh.enabled (flake.nix:2918-2920)
    Note over SV: every program below priority=99 launches in ascending<br/>priority order but does not block on prior RUNNING state
    Note over SV,BOOT: RESOLVED ADR-2063 (2026-09-05) - a program that needs a file Stage B writes later<br/>waits for it with a bounded timeout instead of crashing into FATAL (see AB-46.2)
    SV->>SEAL: spawn priority=99 last, user=devuser (flake.nix:2910-2915)
    SEAL->>SEAL: _required_programs() awk-scans /etc/supervisord.conf<br/>for AGENTBOX_REQUIRED_FOR_READINESS=true blocks (seal-bootstrap.sh:42-72)
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

## AB-46.2 shared MCP hub: bounded wait, then a loud FATAL (ADR-2034, ADR-2104)
```mermaid
sequenceDiagram
    autonumber
    participant SV as supervisord PID1
    participant HUB as program:agentbox-mcp-hub<br/>flake.nix:3563 priority=205 (flake.nix:3572)
    participant WAIT as wait_for_config<br/>services/agentbox-mcp/src/hub/mod.rs:252
    participant BOOT as program:bootstrap Stage B<br/>config/entrypoint-unified.sh:3431
    participant FS as /run/agentbox/mcp-hub.json

    SV->>HUB: spawn priority=205 while Stage B is still in phase 7, file absent (flake.nix:3572)
    HUB->>WAIT: agentbox-mcp hub --wait-config-secs 120 (flake.nix:3564, default in agentbox-mcp/src/main.rs:59)
    loop poll every 500 ms (hub/mod.rs:287), log every 15 s (hub/mod.rs:284)
        WAIT->>FS: path.exists
    end
    BOOT->>FS: agentbox-manifest mcp-hub-project writes mcp-hub.json (entrypoint-unified.sh:3431-3434)
    BOOT->>FS: chown 1000:1000 and chmod 600 the projection (entrypoint-unified.sh:3440-3441)
    BOOT->>SV: supervisorctl status agentbox-mcp-hub (entrypoint-unified.sh:3448)
    alt RUNNING, waiting or serving an older config
        BOOT->>SV: supervisorctl restart agentbox-mcp-hub (entrypoint-unified.sh:3451)
    else FATAL, EXITED, STOPPED or BACKOFF
        BOOT->>SV: supervisorctl start agentbox-mcp-hub (entrypoint-unified.sh:3454)
    end
    BOOT->>FS: then agentbox-manifest mcp-mirror-server copies the final claude-flow entry into ~/.claude/.claude.json, idempotent, fail-open (entrypoint-unified.sh:3460-3468)
    alt config present before the timeout
        WAIT-->>HUB: Ok, HubConfig::load then serve on loopback port 9720 (hub/mod.rs:297-298)
        HUB->>HUB: a non-loopback bind is refused, not downgraded (hub/mod.rs:300-301)
    else 120 s elapsed
        WAIT-->>SV: error naming the projection and the gate, then bail (hub/mod.rs:259-282)
        Note over SV,HUB: startsecs=130 is LONGER than the wait (flake.nix:3570),<br/>so the exit is a FAILED START, retried startretries=2 (flake.nix:3571)<br/>and then parked FATAL by autorestart=unexpected (flake.nix:3569)
    end
    Note over HUB,FS: INVARIANT - the hub is loopback-only and never published (hub/mod.rs:300)
    Note over SV,BOOT: ADR-2104 - the old unbounded 600 s wait under autorestart=true made<br/>supervisorctl status read RUNNING for three days over a port never bound,<br/>docs/adr/ADR-2104-direct-control-over-mcp.md:23
```

**Invariant:** a hub that never receives its projection now fails the start rather than idling, because `startsecs=130` outlives the 120 s wait (`../project/agentbox/flake.nix:3570`, `../project/agentbox/services/agentbox-mcp/src/main.rs:59`).

**Debt:** nine MCP servers are hub-routed and exactly one has a call site in this repository, so a silent hub cost nothing that was noticed for three days (`../project/agentbox/docs/adr/ADR-2104-direct-control-over-mcp.md:30`).

## AB-46.3 session plane — provision, bind, reconcile, bounded restarts (ADR-2153)
```mermaid
sequenceDiagram
    autonumber
    participant SV as supervisord<br/>flake.nix:3153
    participant D as agentbox-sessions daemon<br/>services/agentbox-sessions/src/main.rs:156
    participant P as Plane<br/>services/agentbox-sessions/src/plane.rs:328
    participant R as reconcile decide<br/>services/agentbox-sessions/src/reconcile.rs:65
    participant WT as worktree ensure<br/>services/agentbox-sessions/src/worktree.rs:80
    participant TM as tmux server<br/>services/agentbox-sessions/src/tmux.rs:183

    SV->>D: spawn program:agentbox-sessions, user=devuser, startsecs 5, priority 47 (flake.nix:3156, flake.nix:3160-3161)
    alt interaction_plane.enabled is false
        D-->>SV: one line and exit 0, no restart loop (agentbox-sessions/src/main.rs:160-161)
    end
    D->>TM: wait up to 120 s for the operator's tmux server (agentbox-sessions/src/main.rs:163, plane.rs:523)
    D->>P: provision() runs node scripts/session-provision.mjs, 60 s cap (agentbox-sessions/src/main.rs:166, plane.rs:541)
    D->>D: bind /run/agentbox-sessions/sessions.sock, see AB-16.7 (agentbox-sessions/src/main.rs:167)
    D->>WT: reap_legacy() once, AoE leftovers under PROJECT-worktrees (agentbox-sessions/src/main.rs:189, worktree.rs:238)
    loop each cfg.interval (agentbox-sessions/src/main.rs:195, agentbox-sessions/src/main.rs:203)
        D->>P: reconcile_once(false) (plane.rs:328)
        P->>TM: list sessions, only those carrying @agentbox-sessions=1 are owned (tmux.rs:23)
        P->>R: decide(wanted fingerprint, seen pane, dirty, restarts) (plane.rs:357)
        alt no session for the seed
            R-->>P: Create, no dirty or backoff check (reconcile.rs:66)
        else name held by a session we do not own
            R-->>P: Hold Conflict, never touched (reconcile.rs:67-69)
        else fingerprint drift, dead pane or a trust-prompt blocker, checked in that order
            alt worktree seed with tracked or untracked changes
                R-->>P: Hold DirtyWorktree (reconcile.rs:82-86)
            else three restarts inside the hour
                R-->>P: Hold Backoff (reconcile.rs:88-90, reconcile.rs:17-18)
            else recreate allowed
                R-->>P: Recreate (reconcile.rs:70-78, reconcile.rs:91)
            end
        else live matching pane with a sign-in blocker
            R-->>P: Hold Blocked, reported and left alone (reconcile.rs:75-77)
        else healthy and matching the fingerprint
            R-->>P: Keep (reconcile.rs:80)
        end
        opt Create or an allowed Recreate leads to start()
            P->>WT: adopt a registered legacy wt/aoe-SLUG into wt/abs-SLUG before ensure, preserving edits (plane.rs:303, worktree.rs:117-134)
            Note over P,WT: another session inside the old path defers the start, the seed's own old pane may be closed first (plane.rs:277-286)
            P->>WT: ensure current wt/abs-SLUG, a plain directory moves aside as .orphan-STAMP (plane.rs:304, worktree.rs:86-90)
            P->>TM: trust seeding, kill the old pane, new_session abs_SLUG (plane.rs:306-311, plane.rs:312)
        end
    end
    Note over D,TM: the reaper removes a legacy worktree only when clean, no commit beyond main and every ignored path regenerable (worktree.rs:278)
```

**Invariant:** ownership is a marker, not a name: only sessions carrying `@agentbox-sessions=1` are ever recreated, so `agentbox:0` (tab0), a leftover `aoe_*` session and operator sessions are never touched, and a worktree seed holding work in progress is held rather than recreated (`../project/agentbox/services/agentbox-sessions/src/tmux.rs:23`, `../project/agentbox/services/agentbox-sessions/src/reconcile.rs:67-69`, `../project/agentbox/services/agentbox-sessions/src/reconcile.rs:82-86`).
**Invariant:** trust seeding moved into the daemon: `trust::seed` writes the root store, every wrapper-profile store that already exists and the Codex config (`../project/agentbox/services/agentbox-sessions/src/trust.rs:34-39`, `../project/agentbox/services/agentbox-sessions/src/trust.rs:102`), while `trust-seed.cjs` still writes only the root store and an existing profile store (`config/hooks/trust-seed.cjs:78-86`, `config/hooks/trust-seed.cjs:91-95`).
**Acceptance boundary:** ADR-2153 remains accepted, partial and live (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`). Its 14 November source check records worktree adoption and passing offline tests, without a new live provider round trip (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:73-75`); its 15 November check finds only a SystemScape comment changed in `tmux-autostart.sh`, with the session-plane commands and tmux ownership unchanged and the live round trip again not rerun (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:64-71`). The old profile records were archived and the mount removed from source; deleting the host volume remains an operator step after the rebuild (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:93-97`).

## AB-46.4 session-provision.mjs run-as-script guard — realpath fix carried over from the AoE seeder
```mermaid
flowchart TB
    ENTRY["daemon provision() spawns node scripts/session-provision.mjs<br/>services/agentbox-sessions/src/plane.rs:541"]
    SYMLINK["/opt/agentbox/scripts is a Nix-store symlink<br/>in the baked image"]
    OLDCHECK["OLD seeder guard: path.resolve(argv[1]) === fileURLToPath(import.meta.url)<br/>compares the symlink path to the resolved store path"]
    OLDRESULT["mismatch -&gt; invokedDirectly=false<br/>main() never called, exit 0 silently<br/>observed 2026-09-06: the AoE seeder ran but provisioned nothing"]
    NEWCHECK["NOW: real(path.resolve(argv[1])) === real(fileURLToPath(import.meta.url))<br/>session-provision.mjs:211"]
    REALPATHOR["real(p) = fs.realpathSync(p), catch -&gt; p<br/>session-provision.mjs:210"]
    NEWRESULT["both sides resolve through the symlink to the same<br/>Nix store path -&gt; invokedDirectly=true -&gt; main() runs<br/>provisionOpenRouter, provisionZai, provisionOpenCode (session-provision.mjs:205)"]

    ENTRY --> SYMLINK --> OLDCHECK --> OLDRESULT
    SYMLINK --> NEWCHECK
    NEWCHECK --> REALPATHOR --> NEWRESULT

    NOTE1["INVARIANT — the guard moved with the provisioning code when the seeder was deleted;<br/>tests/cli/opencode-loom.test.mjs imports the module (opencode-loom.test.mjs:6) and spawns it as a script (opencode-loom.test.mjs:125)<br/>by a plain path, so the run-as-script branch is tested but the Nix-store symlink case is not"]
    NEWRESULT --- NOTE1
```

## AB-46.5 launch contract wrapper kind — router seed joins openrouter/zai (ADR-2080, ADR-2153)
```mermaid
flowchart TB
    K["Contract::kind<br/>services/agentbox-sessions/src/contract.rs:72"]
    K -->|"slug has config/harness-wrappers/SLUG.sh and tool is claude or custom:SLUG<br/>contract.rs:73-76"| W["kind wrapper, argv is the wrapper path<br/>config/session-plane/launch.json:14-16<br/>trust claude-profile, blockers include trust, login and API 401/403 (launch.json:17-18)"]
    K -->|"otherwise the tool name selects the template"| T["claude, codex, antigravity, opencode templates<br/>contract.rs:77-79"]
    W --> OR["openrouter -&gt; openrouter.sh"]
    W --> ZA["zai -&gt; zai.sh"]
    W --> RO["router -&gt; router.sh (ADR-2080)"]
    OR -.-> NOTE1["there is no detectAs alias any more: blocked comes from the contract's blocker patterns,<br/>busy and idle from the agent hook or the age settle (status.rs:53), so no AoE heuristic stands in for them"]
    RO -.-> NOTE2["router is its own program (config/harness-wrappers/router.sh)<br/>EXTERNAL: the router.sh console itself and its wiring are owned by AB-29"]
```

- `Contract::kind` resolves each `[[interaction_plane.session_seeds]]` slug that has a `config/harness-wrappers/<slug>.sh` and a `claude` or `custom:<slug>` tool to the `wrapper` template (`../project/agentbox/services/agentbox-sessions/src/contract.rs:72-76`), so the `router` seed (`tool = "custom:router"`, `../project/agentbox/agentbox.toml:2296`) launches exactly like `openrouter` and `zai`.
- **Residual comment drift:** `agentbox.toml` still says `custom:router` is "resolved by the seeder's WRAPPER_SLUGS" (`../project/agentbox/agentbox.toml:2296`). The wrapper now correctly names the native session plane (`../project/agentbox/config/harness-wrappers/router.sh:5-8`); see AB-29 for its launch flow.

## AB-46.6 bootstrap.done now means the promised projections exist (ADR-2104)
```mermaid
sequenceDiagram
    autonumber
    participant SEAL as program:bootstrap-seal<br/>config/seal-bootstrap.sh:15
    participant CP as _check_projections<br/>seal-bootstrap.sh:152
    participant AM as agentbox-manifest toml-bool<br/>seal-bootstrap.sh:147
    participant FS as /run/agentbox
    participant RDY as GET /ready<br/>management-api/server.js:473

    SEAL->>CP: after every AGENTBOX_REQUIRED_FOR_READINESS program is RUNNING
    Note over CP: rows are gate:path pairs, default<br/>resources.mcp_hub.enabled:/run/agentbox/mcp-hub.json and<br/>system_one.enabled:/run/agentbox/system-one.json (ADR-2139)<br/>seal-bootstrap.sh:141
    loop each promised projection
        CP->>AM: toml-bool --path <gate>
        alt gate off or manifest unreadable
            AM-->>CP: skip the row (seal-bootstrap.sh:145-147, seal-bootstrap.sh:158)
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

## AB-46.7 registry reconciliation at boot: skills, agents, commands (ADR-2092)
```mermaid
sequenceDiagram
    autonumber
    participant B as program:bootstrap Stage B<br/>config/entrypoint-unified.sh:3829
    participant RS as reconcile-skills.sh<br/>scripts/reconcile-skills.sh:18
    participant PR as project-skill-roots.mjs<br/>scripts/project-skill-roots.mjs:61
    participant RA as reconcile-agents.sh<br/>scripts/reconcile-agents.sh:52 SUPERSEDED_BASE
    participant RC as reconcile-commands.sh<br/>scripts/reconcile-commands.sh:33 SUPERSEDED_BASE
    participant FS as host-mounted config roots

    B->>RS: CLAUDE_SKILLS_DIR ~/.claude/skills, manifest registered-skills.txt (entrypoint-unified.sh:3826-3829)
    RS->>FS: symlink each registered name to the baked tree (reconcile-skills.sh:63-67)
    B->>RS: second pass for Codex, ~/.codex/skills with codex-registered-skills.txt (entrypoint-unified.sh:3835-3845)
    B->>PR: SKILL_ROOT_TARGETS workspace and workspace/project (entrypoint-unified.sh:3858-3866)
    PR->>FS: retire an unregistered non-overlay dir to an IN-ROOT .superseded (project-skill-roots.mjs:131)
    B->>RA: AGENT_ROOT_TARGETS plus registered-agents.txt (entrypoint-unified.sh:3882-3888)
    RA->>FS: retire a vendor-marked or category-filed agent to SUPERSEDED_BASE, OUTSIDE every scanned root (reconcile-agents.sh:212,218)
    RA->>FS: KEPT unmanaged - a flat hand-written agent with no vendor marker is reported, not retired (reconcile-agents.sh:220)
    Note over RA,RC: RESOLVED — the sidecar used to be an in-root .superseded/, but Claude Code<br/>scans agent/command roots recursively including dot-dirs, so 93 retired agents and<br/>139 retired commands were STILL loading as `.superseded:*` — both reconcilers now<br/>retire OUTSIDE every root (an agentbox-superseded dir beside the primary root's<br/>parent) and migrate any legacy in-root sidecar out on every boot, self-healing<br/>(reconcile-agents.sh:19-24,200 / reconcile-commands.sh:19-22,79-105)
    B->>RC: COMMAND_ROOT_TARGETS three roots, registered-commands.txt (entrypoint-unified.sh:3900-3905)
    RC->>FS: prune-only sweep, kept names stay, the rest move to SUPERSEDED_BASE (reconcile-commands.sh:127)
    Note over B,FS: every reconciler is fail-open with a trailing true,<br/>so a broken registry never blocks the boot (entrypoint-unified.sh:3829, entrypoint-unified.sh:3887, entrypoint-unified.sh:3903)
```

**What it shows.** The four boot-phase reconcilers that make the visible skill, agent and command sets a function of checked-in manifests rather than of whatever a vendor installer last dumped into the host mounts.
**Why it is this way.** ADR-2092: the agent roots were ungoverned, so `ruflo init` and `aqe init --auto` accreted 97 agents across two roots, 74 duplicated and 37 byte-divergent, with the nested root shadowing the user root (`config/entrypoint-unified.sh:3868-3876`).

**Invariant:** retirement is always to a recoverable sidecar, never a delete, in all three reconcilers — for agents and commands that sidecar now sits OUTSIDE every scanned root (`../project/agentbox/scripts/reconcile-agents.sh:94-105`, `../project/agentbox/scripts/reconcile-commands.sh:13-14`).

**Invariant:** a hand-written agent placed flat in the root with no vendor marker is preserved and reported, so experiments survive the sweep (`../project/agentbox/scripts/reconcile-agents.sh:220`).

**Debt:** the registries are four separate manifests in three formats (13 agents, 1 command, 22 Claude skills and 17 Codex skills; the entrypoint comment still says it "links the 12 registered agents", `config/entrypoint-unified.sh:3877`), with the skill counts taken from `skills/registered-skills.txt` and `skills/codex-registered-skills.txt`, each reconciled by its own script with its own retirement rules (`../project/agentbox/agents/registered-agents.txt:1`, `../project/agentbox/config/registered-commands.txt:1`).

## AB-46.8 the role_isolation boot branch — one image, two supervisor configs (ADR-2122)
```mermaid
flowchart TB
    START["Stage A as root<br/>entrypoint-unified.sh:190 PATH sanitised to store and system dirs<br/>before the stage dispatch, in BOTH modes"]
    READ{"AGENTBOX_ROLE_ISOLATION<br/>entrypoint-unified.sh:317 _ab_toml_bool security role_isolation<br/>schema default false"}
    READY{"image can honour it?<br/>role-custody.sh:59 ab_role_isolation_ready<br/>roles config and delivery plan readable"}
    UNAV["ROLE-ISOLATION-UNAVAILABLE logged, flag forced to 0<br/>entrypoint-unified.sh:328-329"]
    subgraph OFF["role_isolation = 0 (shipped default)"]
        O1["/run/secrets chowned to devuser 0700<br/>entrypoint-unified.sh:359-361"]
        O2["host docker socket widened o+rw for devuser<br/>entrypoint-unified.sh:609-611"]
        O3["nostr.key written 0400 devuser<br/>entrypoint-unified.sh:1124-1133"]
        O4["exec supervisord -c /etc/supervisord.conf<br/>entrypoint-unified.sh:1285, entrypoint-unified.sh:1299"]
        O5["Stage B guard /tmp/.agentbox-root<br/>entrypoint-unified.sh:208"]
        O1 --> O2 --> O3 --> O4 --> O5
    end
    subgraph ON["role_isolation = 1"]
        I1["/run/secrets kept root 0711, its own mount<br/>role-custody.sh:89 ab_secrets_root_prepare"]
        I2["ROLE vars written to /run/secrets/role/NAME 0400 and unset<br/>entrypoint-unified.sh:394 _ab_role_env_capture"]
        I3["socket left as found, ok or degraded docker-socket recorded<br/>entrypoint-unified.sh:606 _ab_docker_socket_access"]
        I4["volume and env secrets delivered per role from role-secrets.tsv<br/>role-custody.sh:218 ab_role_secrets_deliver"]
        I5["any ROLE var still in PID1 env logged as a LEAK and unset<br/>entrypoint-unified.sh:438 _ab_role_env_scrub"]
        I6["exec supervisord -c /etc/supervisord.roles.conf<br/>role-custody.sh:48 ab_supervisord_conf_pick"]
        I7["Stage B guard /run/secrets/.root-guard<br/>role-custody.sh:36 ab_root_state_dir_pick"]
        I1 --> I2 --> I3 --> I4 --> I5 --> I6 --> I7
    end
    CLAIM["Stage B claims the guard with an atomic mkdir<br/>a supervisorctl replay is a no-op exit 0<br/>entrypoint-unified.sh:229 _ab_stage_b_claim"]
    START --> READ
    READ -->|"0"| OFF
    READ -->|"1"| READY
    READY -->|"no"| UNAV --> OFF
    READY -->|"yes"| ON
    OFF --> CLAIM
    ON --> CLAIM
    ON -.-> RO["devuser shells get DOCKER_HOST=/run/docker-ro.sock<br/>GET-only, exec and run answer 403<br/>docker-read-proxy.cjs:134-136 drops to 65534 after bind"]
    OFF -.-> SIDOFF["serve-identity and docker-read-proxy print one line and exit 0<br/>identity_port/mod.rs:95-98, docker-read-proxy.cjs:144-146"]
```

**What it shows.** The single decision the custody change adds to the boot: `[security].role_isolation` is read once in Stage A and exported, and it selects between today's boot and the isolated one. Both supervisor configs, the role accounts and the delivery plan ship in every image, so the supervisor branch is chosen at boot; the devuser sudo drop, however, is baked from the flag at build time (`../project/agentbox/flake.nix:4290-4297`), so the flag is rebuild-class end to end: an image built off and booted on reports `degraded:devuser-sudo` until rebuilt (see the Drift below). The PATH sanitiser and the Stage B one-shot guard sit outside the branch and apply in both modes.
**Why it is this way.** The custody design (ADR-2122, X-1 step 1) closes three bypasses: root running devuser-planted code, devuser driving the host Docker daemon, and role secrets inherited by every process from PID 1. The first is closed unconditionally. The other two change what devuser can do, so they sit behind a flag that ships off until the owner's boot rehearsal passes. Per-role detail, the identity port and the rehearsal are in AB-36.

**Invariant:** root's boot PATH holds only `/nix/store` entries and the image's fixed system directories, in both modes, so a binary planted in the devuser-owned workspace cargo bin can no longer run as root (`../project/agentbox/config/entrypoint-unified.sh:175`, `../project/agentbox/config/entrypoint-unified.sh:190`). The cargo bin reaches devuser shells only, appended (`../project/agentbox/config/entrypoint-unified.sh:4165`).

**Invariant:** Stage B runs once per container start: a `supervisorctl start bootstrap` replay finds `stage-b.claimed` already made and exits 0 without running anything, and a missing or squatted guard directory makes Stage B refuse (`../project/agentbox/config/entrypoint-unified.sh:229-233`, `config/entrypoint-unified.sh:1328-1332`).

**Invariant:** devuser is not a member of group `root`, unconditionally: the baked group file now reads `root:x:0:` with no members, and undoing it needs a rebuild (`../project/agentbox/flake.nix:4282`).

**Invariant (under role_isolation):** no ROLE variable is in PID 1's environment when supervisord starts, so no supervised program inherits a signing key. The capture writes each to a 0400 file and unsets it, and a final scrub logs and unsets any that reappear (`../project/agentbox/config/entrypoint-unified.sh:382`, `../project/agentbox/config/entrypoint-unified.sh:1298`). With the flag off both functions return at their first line, and PID 1 still carries every `.env` key.

**Invariant (under role_isolation):** the entrypoint does not widen the host Docker socket for devuser (`../project/agentbox/config/entrypoint-unified.sh:609`). With the flag off it still runs `chmod o+rw` on every boot.

**Debt:** `[program:docker-read-proxy]` has no `user=` and drops privilege itself. Its comment still names a W1 follow-up to move it into `lib/role-accounts.nix` with its own account (`flake.nix:3050-3051`), but the role registry W1 shipped is `config/role-accounts.json` (`../project/agentbox/config/role-accounts.json:3`), and it has no row for the proxy.

**Drift:** `docs/BASELINE-container.md:115` calls `[security].role_isolation` boot-class, but the manifest catalogue entry is `apply_class: 'rebuild'` (`management-api/lib/system-manifest.js:97-99`) and the entrypoint comment says rebuild-class since ER-9 because the devuser sudo drop is baked (`config/entrypoint-unified.sh:314`, `flake.nix:4290-4297`).
