---
id: AB-47
title: The agentbox.sh CLI — dispatch, ruvector, health and model-router
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2037, ADR-2038, ADR-2080]
sources:
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/scripts/ruvector-sidecar-update.sh
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/scripts/recall-fixtures/recall-fixture.v1.json
  - ../project/agentbox/scripts/ruvector-recall-harness.mjs
  - ../project/agentbox/skills/mcp.json
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/config/instructions/
verified_commit: 1d26ebe4259797a724d63628777995c40add1173
aliases:
  AB-05.6: AB-47.2
  AB-05.7: AB-47.3
  AB-05.11: AB-47.4
---
Split from AB-05 on 2026-11-23 (formerly AB-05.6, AB-05.7 and AB-05.11), plus the successor to the frozen AB-05.5. Stamped at agentbox 1d26ebe, not AB-05's 59b1aca: agentbox.sh gained one usage line at line 62, so every agentbox.sh anchor in the moved panels moved down one line and was re-derived against 1d26ebe by content, not by offset alone. Old panel IDs resolve through the `aliases:` map above.

## AB-47.1 agentbox.sh top-level subcommand dispatch — config now on the allowlist
```mermaid
flowchart LR
    ARG["arg parse loop<br/>agentbox.sh:2045-2066"] --> AL{"subcommand in the<br/>allowlist at agentbox.sh:2055?"}
    AL -->|no| ERR["Unknown command then usage then exit 1<br/>agentbox.sh:2061-2064"]
    AL -->|yes| CASE["execute case CMD<br/>agentbox.sh:2868-2910"]
    CASE --> G1["access: ssh agentbox.sh:2869, vnc agentbox.sh:2870, browser agentbox.sh:2871,<br/>code agentbox.sh:2872, api agentbox.sh:2873, all agentbox.sh:2874, status agentbox.sh:2875, ip agentbox.sh:2876"]
    CASE --> G2["lifecycle: up agentbox.sh:2882, down agentbox.sh:2883, build agentbox.sh:2884,<br/>NEW prepare agentbox.sh:2885, NEW activate agentbox.sh:2886 (ADR-2132),<br/>rebuild agentbox.sh:2887, update agentbox.sh:2888, NEW security agentbox.sh:2889 (ADR-2137, see the Debt below)"]
    CASE --> G3["provisioning: provision agentbox.sh:2877, setup agentbox.sh:2878,<br/>start-browser agentbox.sh:2879, migrate-workspace agentbox.sh:2906,<br/>migrate-claude-home ADR-2118 agentbox.sh:2907, preflight agentbox.sh:2908"]
    CASE --> G4["data: backup agentbox.sh:2880, restore agentbox.sh:2881,<br/>ruvector agentbox.sh:2890, ruvnet-brain agentbox.sh:2891"]
    CASE --> G5["observe: logs agentbox.sh:2892, shell agentbox.sh:2893, health agentbox.sh:2894"]
    CASE --> G6["sidecars: browsercontainer agentbox.sh:2895, gui-tools agentbox.sh:2896,<br/>NEW isaac-sim agentbox.sh:2897 (ADR-2149), openmed agentbox.sh:2898, voice agentbox.sh:2899,<br/>NEW config agentbox.sh:2900 (plan, validate), systemone agentbox.sh:2901 ADR-2094,<br/>model-router agentbox.sh:2902 ADR-2080, xr-runtime agentbox.sh:2903, android agentbox.sh:2904,<br/>NEW concat agentbox.sh:2905 gated AGENTBOX_ENABLE_CONCAT=1 agentbox.sh:617-619"]
    G2 --> RB["cmd_rebuild agentbox.sh:1086 now delegates to<br/>scripts/runtime-delivery.sh rebuild — prepare a candidate<br/>image, then scoped activation, without recreating sidecars<br/>or volumes (ADR-2132); agentbox.sh prepare agentbox.sh:2885 and<br/>activate agentbox.sh:2886 call the same flow's halves"]
    G4 --> RV["cmd_ruvector agentbox.sh:1042<br/>exec bash scripts/ruvector-sidecar-update.sh agentbox.sh:1049 — see AB-05.6"]
    G5 --> HL["cmd_health agentbox.sh:1132 — see AB-05.7"]
    G5 --> SH["cmd_shell agentbox.sh:1112"]
    SH -.-> RES1["ADR-2038 closed: cmd_shell agentbox.sh:1129 now uses<br/>cd /home/devuser/workspace/profiles/${profile} && exec fish — the old<br/>finding was that it execed the retired literal /workspace/profiles/PROFILE<br/>path (agentbox/CLAUDE.md 'Runtime model gotchas'); every other profile<br/>path in the script already used SCRIPT_DIR/workspace/profiles"]
```

**Invariant (successor to AB-05.5, true at 1d26ebe):** `config` is on the allowlist (`../project/agentbox/agentbox.sh:2055`) and in the usage text (`../project/agentbox/agentbox.sh:62`), so the `config` arm (`../project/agentbox/agentbox.sh:2900`) is reachable and runs `cmd_config` (`../project/agentbox/agentbox.sh:2448-2454`): `plan` and `validate` only, anything else exits 2.

**Drift (resolved, `security` subcommand):** the `security` arm used to be unreachable because the allowlist omitted it; the allowlist at this stamp lists `security` (`../project/agentbox/agentbox.sh:2055`) and the arm runs `scripts/security-scan.sh` (`../project/agentbox/agentbox.sh:2889`), which `cmd_up --build` and `cmd_build` also call directly (`../project/agentbox/agentbox.sh:766`, `agentbox.sh:941`).

## AB-47.2 agentbox.sh ruvector — a dispatch table split across two files
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant AB as cmd_ruvector<br/>agentbox.sh:1042
    participant SC as ruvector-sidecar-update.sh<br/>scripts/ruvector-sidecar-update.sh:1841-1861 dispatch
    participant H as scripts/ruvector-recall-harness.mjs

    OP->>AB: ./agentbox.sh ruvector <subcmd> [args]
    AB->>SC: exec bash SCRIPT_DIR/scripts/ruvector-sidecar-update.sh "$@" (agentbox.sh:1049)
    Note over AB: image pin lives in agentbox.toml [integrations.ruvector_external] and is mirrored into docker-compose.yml (agentbox.sh:1043-1048)
    alt status | check | test (ruvector-sidecar-update.sh:1842-1844)
        SC-->>OP: sidecar state, pinned-vs-Docker-Hub comparison
    else update (ruvector-sidecar-update.sh:1845)
        SC->>SC: dump then pg_basebackup snapshot then candidate rehearsal then swap
    else rollback (ruvector-sidecar-update.sh:1846)
        SC->>SC: restore previous image plus datadir from the recorded snapshot
    else migrate-trajectories | repair-namespaces | backfill-embeddings | archive-legacy | aggregate-effectiveness | build-metadata-gin (ruvector-sidecar-update.sh:1847-1852)
        SC-->>OP: DRY-RUN by default — each needs --yes plus its manifest flag
    else recall (ruvector-sidecar-update.sh:1853)
        SC->>SC: cmd_recall (ruvector-sidecar-update.sh:1248) — require_prod_running then node present
        SC->>SC: resolve governed MCP env via mcp_env_pairs from .mcp.json (ruvector-sidecar-update.sh:1254)
        SC->>H: run_recall_harness (ruvector-sidecar-update.sh:1265) runs env ENVP node "$RECALL_HARNESS" "$@" (ruvector-sidecar-update.sh:1210), default scripts/ruvector-recall-harness.mjs (ruvector-sidecar-update.sh:1201)
        H-->>OP: harness sets its own exit code — PASS 0, FAIL non-zero (ruvector-sidecar-update.sh:1259)
    else reindex | reader-role | hba-harden (ruvector-sidecar-update.sh:1854-1856, ADR-2133)
        SC-->>OP: serial HNSW rebuild with recall gate before and after, reader role, pg_hba hardening<br/>since d09926697 a pre-swap screen scores each index with the other hidden and keeps the current one<br/>unless a rebuild scores no worse, two build attempts (ruvector-sidecar-update.sh:1322, ruvector-sidecar-update.sh:1406, ruvector-sidecar-update.sh:1421-1434, ruvector-sidecar-update.sh:1438-1442)
    else anything else
        SC-->>OP: die unknown subcommand (ruvector-sidecar-update.sh:1860)
    end
    Note over SC,H: recall is READ-ONLY, no gate of its own — fixture scripts/recall-fixtures/recall-fixture.v1.json is frozen and checked in (ruvector-sidecar-update.sh:1257)
    Note over H: classes self-recall@10, true-recall@10 vs forced exact scan, exact-token — median-of-3 no-regression band (ruvector-sidecar-update.sh:1258-1259)
    Note over H: artefact lands in backups/ruvector-sidecar/recall-runs/<utc>.json (ruvector-sidecar-update.sh:1260) — retrieval-geometry gate boundary, see AB-20
    Note over AB,SC: DOC-DRIFT — RESOLVED: the usage text at agentbox.sh:51 lists recall,<br/>which ruvector-sidecar-update.sh:1853 implements and ruvector-sidecar-update.sh:1860 advertises
    Note over AB,SC: RESOLVED ADR-2038: recall is in the ruvector subcommand list at<br/>agentbox.sh:51 and has a usage example at agentbox.sh:105
```

## AB-47.3 agentbox.sh health — exit-code contract
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant SH as cmd_health<br/>agentbox.sh:1132
    participant H as GET /health<br/>server.js:626-639
    participant M as GET /v1/meta<br/>localhost:9090

    OP->>SH: ./agentbox.sh health [--json]
    SH->>H: curl -sf HEALTH_URL (agentbox.sh:632 and agentbox.sh:1143)
    alt curl fails
        SH-->>OP: ERROR could not reach — exit 1 (agentbox.sh:1145-1147)
    else --json passed
        SH-->>OP: raw JSON then exit 0 (agentbox.sh:1149-1151)
    else jq absent
        SH-->>OP: warning plus raw response (agentbox.sh:1223-1224)
    else pretty path
        SH->>SH: degraded = jq '.adapters // {} | select(.value != healthy and != off)' (agentbox.sh:1160-1164)
        SH->>SH: degraded_count = jq '.degraded_count // 0' (agentbox.sh:1165)
        SH->>SH: print adapter/<slot> lines from .adapters (agentbox.sh:1169-1177)
        SH->>M: curl /v1/meta then read observability.metrics_endpoint (agentbox.sh:1187-1189)
        M-->>SH: metrics endpoint
        SH->>SH: print first 5 non-comment metric lines (agentbox.sh:1195-1202)
        alt degraded non-empty OR degraded_count > 0
            SH-->>OP: exit 1 (agentbox.sh:1219-1220)
        else
            SH-->>OP: exit 0
        end
    end
    Note over SH,H: RESOLVED ADR-2037 — the old finding was that /health emits<br/>status, uptime, image_hash, manifest_checksum, adapters, adapter_failures,<br/>adapter_failure_files, degraded_count, note (server.js:628-638) with no services key, so a stale jq '.services // {}' read<br/>left the exit-1 branch unreachable. cmd_health now derives failure from<br/>.adapters (a slot fails when neither "healthy" nor "off", agentbox.sh:1160-1164)<br/>plus .degraded_count (agentbox.sh:1165) — exit 1 at agentbox.sh:1219-1220<br/>is reachable, and the condition now also fails on speech_failed (agentbox.sh:1206-1219).<br/>Same fix as AB-04.16
    Note over SH: BASELINE-container Adapter spine stage 4 claims agentbox.sh health exits non-zero if any slot gauge is 0 — it never reads the agentbox_adapter_health gauge at all. See AB-04.16
    Note over H: /health self-describes as human-inspection-only and points orchestrators at /ready (server.js:637)
```

## AB-47.4 agentbox.sh model-router — ADR-2080 CLI dispatch, gated by [model_routing.neural]
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant AB as cmd_model_router<br/>agentbox.sh:2397
    participant FETCH as scripts/model-router-fetch.sh
    participant CONSOLE as config/model-router/console.mjs
    participant WRAP as config/harness-wrappers/router.sh

    OP->>AB: ./agentbox.sh model-router <sub> [args]
    alt sub == fetch (agentbox.sh:2402)
        AB->>FETCH: exec scripts/model-router-fetch.sh "$@"
        Note right of FETCH: populates $WORKSPACE/.agentbox/model-router,<br/>hash-verified against config/model-router/artefacts.json — pre-rebuild fallback
    else sub == check (agentbox.sh:2403)
        AB->>FETCH: exec scripts/model-router-fetch.sh --check "$@"
    else sub == status (agentbox.sh:2404, default when sub omitted agentbox.sh:2399)
        AB->>CONSOLE: exec node console.mjs --status "$@"
    else sub == route (agentbox.sh:2405-2406)
        AB->>AB: require at least 1 positional arg or usage+exit 2
        AB->>CONSOLE: exec node console.mjs --once "<task>" "$@"
    else sub == console (agentbox.sh:2407)
        AB->>WRAP: exec config/harness-wrappers/router.sh "$@"
        Note right of WRAP: same wrapper the agentbox-sessions `router` seed<br/>execs — see AB-29.3
    else unknown (agentbox.sh:2408-2417)
        AB-->>OP: usage heredoc, exit 2
    end

    Note over AB,CONSOLE: EXTERNAL — console.mjs's own request/response flow to the OpenRouter<br/>provider and the KRR/k-NN router logic is AB-29 (model-routing-neural topic), not here
    Note over AB: cmd_model_router itself is UNGATED — the CLI runs regardless of<br/>[model_routing.neural].enabled, fetch/status/console fail loud if the<br/>rebuild-baked or fetch-populated artefact dir is absent — see AB-01.11
```
