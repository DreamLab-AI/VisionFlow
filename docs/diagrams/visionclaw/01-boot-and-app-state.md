---
id: VC-01
title: Server boot, AppState construction and the full route table
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2004, ADR-2005, ADR-2007, ADR-2008, ADR-2026, ADR-2037, ADR-2038, ADR-2045, ADR-2053]
sources:
  - ../project/src/main.rs
  - ../project/src/app_state.rs
  - ../project/src/handlers/mod.rs
  - ../project/src/handlers/consolidated_health_handler.rs
  - ../project/src/handlers/api_handler/mod.rs
  - ../project/src/handlers/api_handler/graph/mod.rs
  - ../project/src/handlers/api_handler/files/mod.rs
  - ../project/src/handlers/api_handler/bots/mod.rs
  - ../project/src/handlers/api_handler/analytics/mod.rs
  - ../project/src/handlers/api_handler/ontology/mod.rs
  - ../project/src/handlers/api_handler/ontology_physics/mod.rs
  - ../project/src/handlers/api_handler/semantic_forces.rs
  - ../project/src/handlers/graph_state_handler.rs
  - ../project/src/handlers/graph_export_handler.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/handlers/physics_handler.rs
  - ../project/src/handlers/schema_handler.rs
  - ../project/src/handlers/semantic_handler.rs
  - ../project/src/handlers/natural_language_query_handler.rs
  - ../project/src/handlers/semantic_pathfinding_handler.rs
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/handlers/validation_handler.rs
  - ../project/src/handlers/workspace_handler.rs
  - ../project/src/handlers/pages_handler.rs
  - ../project/src/handlers/kpi_handler.rs
  - ../project/src/handlers/trace_handler.rs
  - ../project/src/handlers/insight_loop_handler.rs
  - ../project/src/handlers/liveness_harness_handler.rs
  - ../project/src/handlers/ingest_writeback_handler.rs
  - ../project/src/handlers/metrics_handler.rs
  - ../project/src/handlers/multi_mcp_websocket_handler.rs
  - ../project/src/handlers/bots_visualization_handler.rs
  - ../project/src/handlers/ontology_derived_handler.rs
  - ../project/src/handlers/ontology_class_count_handler.rs
  - ../project/src/handlers/ontology_agent_handler.rs
  - ../project/src/handlers/admin_rbac_handler.rs
  - ../project/src/handlers/admin_sync_handler.rs
  - ../project/src/handlers/nostr_handler.rs
  - ../project/src/handlers/broker_inbox_handler.rs
  - ../project/src/handlers/decision_handler.rs
  - ../project/src/handlers/enrichment_proposals_handler.rs
  - ../project/src/handlers/briefing_handler.rs
  - ../project/src/handlers/memory_flash_handler.rs
  - ../project/src/handlers/image_gen_handler.rs
  - ../project/src/handlers/solid_proxy_handler.rs
  - ../project/src/handlers/ragflow_handler.rs
  - ../project/src/handlers/quic_transport_handler.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/src/services/liveness_harness.rs
  - ../project/src/services/canary_nostr_tap.rs
  - ../project/src/config/security_profile.rs
  - ../project/src/handlers/pay_handler.rs
  - ../project/src/services/github/config.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/src/services/kpi_compute.rs
  - ../project/src/services/role_store.rs
  - ../project/src/utils/auth.rs
  - ../project/tests/resd_class_count_route.rs
  - ../project/src/services/ontology_pull.rs
  - ../project/Cargo.toml
  - ../project/src/adapters/mod.rs
verified_commit: 36bb64e1e
---

## VC-01.1 main() phase 1 — hygiene, logging, settings, stores
```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>src/main.rs:172
    participant EH as enforce_release_env_hygiene<br/>src/main.rs:118 real / :169 stub
    participant VE as validate_required_env_vars<br/>src/main.rs:61
    participant TL as telemetry logger<br/>src/main.rs:266
    participant DB as dev-mode banner<br/>src/main.rs:274-290
    participant CF as AppFullSettings::new<br/>src/main.rs:292
    participant ST as stores<br/>Oxigraph + SQLite

    rect rgb(240,232,232)
    Note over M,EH: refusal band — nothing has bound a socket yet
    M->>EH: enforce_release_env_hygiene() (src/main.rs:195)
    alt release build and a dev env var or --allow-skip-auth is present
        EH-->>M: eprintln FATAL then exit(1) argv / exit(2) env
        Note over EH: ADR-2026 + ADR-2037 — detail see VC-09.3
    end
    end
    M->>VE: validate_required_env_vars()
    alt APP_ENV == "production" and SYSTEM_NETWORK_PORT missing
        VE-->>M: Err — hard fail
    else
        VE-->>M: Ok (warns on MANAGEMENT_API_KEY / JWT_SECRET / CORS_ALLOWED_ORIGINS)
    end
    M->>TL: init logger, dir from TELEMETRY_LOG_DIR (src/main.rs:266)
    opt cfg any(debug_assertions, feature="dev-auth") and dev_full_bypass_active()
        M->>DB: warn banner x5 — VISIONCLAW_DEV_MODE=1 LAN-LOCAL AUTH BYPASS ACTIVE
        Note over DB: src/utils/auth.rs:100 dev_full_bypass_active, prints DEV_MODE_PUBKEY<br/>ADR-2039 — every request is granted dev-admin, no NIP-98/token/peer check
    end
    M->>CF: AppFullSettings::new()
    CF->>CF: SETTINGS_FILE_PATH, default "/app/settings.yaml" (src/main.rs:296)
    Note over CF: YAML in snake_case, JSON out camelCase — serde alias asserted src/main.rs:300-320
    alt load fails
        CF-->>M: Err — boot aborts
    end
    M->>ST: DATA_DIR (src/main.rs:356) — Oxigraph store plus per-domain SQLite files
    Note over ST: INVARIANT (BASELINE Invariants) — persistence is Oxigraph (data/oxigraph)<br/>plus per-domain SQLite under DATA_DIR. One Oxigraph store is shared by the<br/>ontology and graph repositories. No networked graph DB. ADR-2004.
    Note over ST: DIVERGENCE (BASELINE 2026-09-04 persistence closeout) — shared Oxigraph ownership<br/>establishes no cross-store transaction, no actor reload consistency, no restore correctness
```

## VC-01.2 main() phase 2 — service graph and RBAC owner bootstrap
```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>src/main.rs:172
    participant AS as AppState::new<br/>src/app_state.rs:413
    participant GH as GitHub client + ContentAPI<br/>src/services/github/config.rs
    participant SY as GithubSyncService<br/>src/services/github_sync_service.rs
    participant RS as RoleStore<br/>src/services/role_store.rs
    participant SV as domain services
    participant PR as assert_effective_profile_or_exit<br/>src/config/security_profile.rs:604-612

    M->>AS: AppState::new(...) — see VC-01.5 for actor start order
    AS-->>M: AppState (liveness_harness, kpi_compute_service, sqlite_kpi_repository, addrs)
    M->>GH: build client from GITHUB_OWNER / GITHUB_REPO / GITHUB_BRANCH / GITHUB_BASE_PATH(S)
    Note over GH: token const GITHUB_TOKEN_ENV = PRIVATE_REPO_GITHUB_PAT (src/services/github/config.rs:23, read :31)<br/>legacy alias GITHUB_TOKEN_ENV_LEGACY :33 — register see VC-09.11
    M->>SY: GithubSyncService — corpus ingest, see VC-21
    RS->>RS: bootstrap_owner_from_env — RBAC_OWNER_PUBKEY_ENV (src/services/role_store.rs:643)
    alt no Owner assigned and RBAC_ALLOW_OWNERLESS unset
        RS-->>M: boot FAILS — "RBAC: no Owner assigned and RBAC_ALLOW_OWNERLESS not set" (src/main.rs:755)
        Note over RS: fail-closed, ADR-2026 — const RBAC_ALLOW_OWNERLESS_ENV src/services/role_store.rs:33
    else RBAC_ALLOW_OWNERLESS=1
        RS-->>M: warn and continue with no Owner (src/main.rs:742)
    end
    M->>RS: set_global_role_store(RoleStore) (src/main.rs:766)
    par pre-server service construction
        M->>SV: SchemaService, NaturalLanguageQueryService, PathfindingService
    and
        M->>SV: OntologyQueryService, OntologyMutationService, DecisionService
    and
        M->>SV: BriefingService, NostrPublisher, ValidationService, PhysicsService
    and
        M->>SV: presence_handler_state (PRD-008 XR), KpiComputeService, LivenessHarness
    end
    opt feature solid-pod-embed
        M->>SV: init_solid_state (src/main.rs:841)
        M->>SV: ontology_pull::spawn_boot_pull(Arc::clone(&solid_state.storage)) (src/main.rs:848)
        Note over SV: ADR-2106 — pulls the published `ontology-latest` release INTO the embedded pod at<br/>boot and re-checks on ONTOLOGY_PULL_INTERVAL_SECS (default 3600s). Fail-open: any error is<br/>logged and the pod keeps what it held. Detail see VC-01.15.
        M->>SV: pay config/ledger/exchange (src/main.rs:854)
        Note over SV: async init hoisted out of the worker factory so worker threads never block
    end
    M->>PR: assert_effective_profile_or_exit(EnvSnapshot::from_process(), BuildIdentity::current(), today) (src/main.rs:883)
    Note over PR: ADR-2038 boot-time profile assertion — runs BEFORE bind. Detail see VC-09.4
    PR-->>M: EffectiveProfile logged as summary + observed_flags (src/main.rs:889-893)
```

## VC-01.3 HttpServer worker factory and middleware stack order
```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>src/main.rs:903
    participant W as worker closure<br/>src/main.rs:903-1183
    participant A as App::new<br/>src/main.rs:978
    participant REQ as inbound request

    Note over M,W: HttpServer::new(move || ...) runs the closure once per worker — .workers(4) (src/main.rs:1185)
    M->>W: spawn 4 workers
    W->>W: build Cors — CORS_ALLOWED_ORIGINS (src/main.rs:905), ALLOW_INSECURE_DEFAULTS (src/main.rs:909) compile-time gated
    Note over W: ADR-06 D1 — a release binary cannot widen CORS via env, it must set CORS_ALLOWED_ORIGINS
    W->>A: App::new()
    A->>A: .wrap(Logger::default()) (src/main.rs:979)
    A->>A: .wrap(cors) (src/main.rs:980)
    A->>A: .wrap(Compress::default()) (src/main.rs:981)
    A->>A: .wrap(TimeoutMiddleware::with_config(30s, override /api/admin/sync 600s)) (src/main.rs:982-985)
    Note over A: actix applies .wrap in REVERSE registration order — the LAST wrap is OUTERMOST.<br/>So an inbound request meets TimeoutMiddleware first and Logger last.
    rect rgb(232,240,232)
    Note over A: /api scope only — src/main.rs:1054-1180
    A->>A: .wrap(PublicDemoGuard::from_env()) (src/main.rs:1058)
    A->>A: .wrap(RbacGate::from_env()) (src/main.rs:1065)
    end
    rect rgb(232,236,244)
    Note over A: /api/settings scope only — src/main.rs:1071-1074
    A->>A: .wrap(RateLimit::per_minute(60)) (src/main.rs:1072)
    end
    Note over A: /api/graph carries its own scope limiter RateLimit::per_minute(600) plus<br/>tighter 120/min per-resource wraps — src/handlers/api_handler/graph/mod.rs:1493 and :1515-1529
    M->>M: .bind(&bind_address) (src/main.rs:1184) then .run() (src/main.rs:1186)
    REQ->>A: request order = Timeout → Compress → Cors → Logger → [PublicDemoGuard → RbacGate] → extractor → handler
    Note over REQ,A: request-time behaviour of each middleware see VC-03
```

## VC-01.4 app_data registry — what each worker gets injected
```mermaid
flowchart TB
    subgraph CORE["core state — src/main.rs:988-997"]
        C1["settings_data"]
        C2["GitHub client"]
        C3["ContentAPI"]
        C4["app_state_data (AppState)"]
        C5["LivenessHarness<br/>web::Data::from(app_state_data.liveness_harness)<br/>backs /api/canary/*"]
        C6["KpiComputeService<br/>backs /api/kpi/{summary,lineage}"]
        C7["pre_read_ws_settings_data"]
        C8["metrics_handler::ProcessStartTime<br/>src/main.rs:997"]
    end
    subgraph ADDRS["actor addresses — src/main.rs:999-1003"]
        A1["graph_service_addr"]
        A2["settings_addr"]
        A3["metadata_addr"]
        A4["client_manager_addr"]
        A5["workspace_addr"]
    end
    subgraph SERVICES["domain services — src/main.rs:1004-1019"]
        S1["schema_service"]
        S2["nl_query_service"]
        S3["pathfinding_service"]
        S4["NostrService (or NostrService::default)"]
        S5["feature_access"]
        S6["github_sync_service"]
        S7["ontology_query_service"]
        S8["ontology_mutation_service"]
        S9["decision_service"]
        S10["settings_repo_data"]
        S11["briefing_service"]
        S12["nostr_publisher"]
        S13["validation_service"]
        S14["physics_service"]
        S15["presence_handler_state (PRD-008)"]
    end
    subgraph FEAT["feature solid-pod-embed — src/main.rs:1021-1033"]
        F1["solid_state"]
        F2["pay_config_data"]
        F3["pay_ledger_data"]
        F4["pay_exchange_data"]
    end
    CORE --> ADDRS --> SERVICES --> FEAT
    N["handlers extract each as web::Data of that type<br/>handler families see VC-04 and VC-05"]
    SERVICES --- N
```

## VC-01.5 AppState::new — actor start order and the coordinator re-bind
```mermaid
sequenceDiagram
    autonumber
    participant AS as AppState::new<br/>src/app_state.rs:413
    participant CC as ClientCoordinatorActor<br/>src/app_state.rs:709
    participant AB as AgentBeamActor<br/>src/app_state.rs:722
    participant MD as MetadataActor<br/>src/app_state.rs:805
    participant GS as GraphServiceSupervisor<br/>src/app_state.rs:810
    participant GPU as GPU actor group<br/>src/app_state.rs:963-977
    participant SE as settings_actor<br/>src/app_state.rs:1152
    participant PEERS as peer actors

    Note over AS: pub struct AppState src/app_state.rs:296
    AS->>CC: ClientCoordinatorActor.start() → client_manager_addr
    AS->>AB: AgentBeamActor::new(client_manager_addr.clone()).start()
    AS->>MD: MetadataActor::new(MetadataStore::new()).start() → metadata_addr
    AS->>GS: GraphServiceSupervisor::new(graph_adapter.clone()).start() → graph_service_addr
    AS->>GS: SetClientCoordinatorAddr { ... } (src/app_state.rs:821)
    Note over CC,GS: INVARIANT (BASELINE Invariants) — the live ClientCoordinatorActor must be the one<br/>clients register with (registry not empty). The re-bind at :820 is what enforces it.
    rect rgb(236,236,244)
    Note over GPU: GPU actor group — GPUManagerActor only. RESOLVED ADR-2053: the standalone<br/>ShortestPathActor and ConnectedComponentsActor spawns were removed. They were never sent<br/>a SharedGPUContext (ResourceSupervisor distributes it only to the subsystem supervisors),<br/>so every /api/analytics pathfinding route addressed a GPU-blind pair.
    AS->>GPU: GPUManagerActor::new().start()
    AS->>GPU: gpu_manager.do_send(SetNodeSSSP { node_sssp }) — ADR-031 D2b, wire slot 28<br/>forwarded GPUManagerActor to GraphAnalyticsSupervisor to the SUPERVISED ShortestPathActor
    Note over GPU: DOC-DRIFT — this block carries NO #[cfg(feature = "gpu")] gate. app_state.rs itself<br/>carries no solid-pod-embed cfg gates either — the only cfg(feature) sites for that flag are in<br/>src/main.rs (:840, :847 — ADR-2106 ontology_pull::spawn_boot_pull, :853, :1022, :1028). It works<br/>because gpu is in the DEFAULT feature set (Cargo.toml:250 default = gpu, ontology,<br/>persistence-oxigraph, solid-pod-embed) while the adapter layer IS gated (src/adapters/mod.rs:19,<br/>:37) — so the actor start and its adapters are gated inconsistently. GPU internals see VC-10.
    end
    AS->>SE: settings_actor.start() → settings_addr
    par peer actors
        AS->>PEERS: AgentMonitorActor::new(claude_flow_client, graph_service_addr).start() (:1184)
    and
        AS->>PEERS: ProtectedSettingsActor::new(ProtectedSettings::default()).start() (:1197)
    and
        AS->>PEERS: WorkspaceActor::new().start() → workspace_addr (:1200)
    and
        AS->>PEERS: OntologyActor.start() (:1213)
    and
        AS->>PEERS: TaskOrchestratorActor::new(mgmt_client).start() (:1263)
    end
    alt ElevationActor::new(...) returns Some (src/app_state.rs:1271-1289)
        AS->>PEERS: let _ = actor.start() — ACSP knowledge-elevation panel live
        Note over PEERS: GOV-7 ADR-130 — Some(ontology_repository) is passed so the EL++ consistency<br/>gate is armed. A None here would fail the gate CLOSED, blocking approvals.
    else None
        AS->>AS: log disabled — needs ELEVATION_ACTOR_ENABLED plus FORUM_RELAY_URL plus ACSP_PANEL_NOSTR_PRIVKEY
    end
    alt VoiceInterfaceActor::new(task_orchestrator_addr, speech_service) returns Some (:1294-1305)
        AS->>PEERS: let _ = actor.start() — ADR-110 voice to settings-assistant bridge
    else None
        AS->>AS: log disabled (no speech service)
    end
    Note over AS,PEERS: full supervision tree and per-actor message surfaces see VC-02
```

## VC-01.6 Feature and flag gates that change the boot shape
```mermaid
sequenceDiagram
    autonumber
    participant B as build identity
    participant M as main<br/>src/main.rs
    participant AS as AppState::new<br/>src/app_state.rs

    alt cfg(not(any(debug_assertions, feature="dev-auth"))) — production artefact
        B->>M: enforce_release_env_hygiene = the real impl (src/main.rs:118)
        Note over B,M: ADR-2037 — the dev bypass codepaths are #[cfg]-stripped, they do not exist in the binary
    else cfg(any(debug_assertions, feature="dev-auth")) — development artefact
        B->>M: enforce_release_env_hygiene = no-op stub (src/main.rs:169)
        B->>M: dev banner block compiled in (src/main.rs:274-290)
    end
    alt feature solid-pod-embed
        M->>M: init_solid_state (src/main.rs:841)
        M->>M: ontology_pull::spawn_boot_pull (src/main.rs:848 — ADR-2106, see VC-01.15) and pay state (src/main.rs:854)
        M->>M: .app_data(solid_state) (src/main.rs:1021-1023)
        M->>M: .app_data(pay_*) and .configure(pay_handler::configure_pay_routes) (src/main.rs:1025-1033)
        Note over M: routes are mounted UNCONDITIONALLY here and stay inert until PAY_ENABLED=true<br/>src/handlers/pay_handler.rs:95 and :892 — .info reports disabled, gated routes 403
        M->>M: pub use solid_proxy_handler::init_solid_state (src/handlers/mod.rs:123)
        M->>M: pub mod pay_handler + configure_pay_routes (src/handlers/mod.rs:130-133)
    else feature absent
        Note over M: /pay/* is not registered at all and init_solid_state is not exported
    end
    opt ELEVATION_ACTOR_ENABLED / DECISION_ELEVATION_ENABLED
        AS->>AS: elevation and decision-elevation actors start or stay down
    end
    Note over M,AS: the complete env-flag register with per-flag file:line see VC-09.8 to VC-09.14
```

## VC-01.7 Health and readiness composition
```mermaid
sequenceDiagram
    autonumber
    participant K as k8s or Docker probe
    participant R as actix router
    participant L as liveness_probe<br/>src/handlers/consolidated_health_handler.rs
    participant D as readiness_probe<br/>src/handlers/consolidated_health_handler.rs
    participant U as unified_health_check<br/>src/handlers/consolidated_health_handler.rs
    participant P as check_physics_simulation

    Note over K,R: root probes are registered OUTSIDE the /api scope, so no RbacGate and no PublicDemoGuard
    K->>R: GET /healthz (src/main.rs:1037)
    R->>L: liveness_probe
    K->>R: GET /readyz (src/main.rs:1038)
    R->>D: readiness_probe
    Note over R: configure_routes (src/handlers/consolidated_health_handler.rs:474-489) ALSO registers<br/>/healthz and /readyz a second time inside /api, plus the /health scope
    K->>R: GET /api/health
    R->>U: unified_health_check (:477)
    K->>R: GET /api/health/physics
    R->>P: check_physics_simulation (:478)
    K->>R: POST /api/health/mcp/start → start_mcp_relay (:481)
    K->>R: GET /api/health/mcp/logs → get_mcp_logs (:482)
    Note over R: /api/health also collides with api_handler::config's own /health route<br/>(src/handlers/api_handler/mod.rs:121) — first registration wins, see VC-01.10
    Note over K,P: INVARIANT (BASELINE Invariants) — backend 4000, nginx 3001, Vite 5173,<br/>all inside visionclaw_container
```

## VC-01.8 Post-bind background tasks and graceful shutdown
```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>src/main.rs:1188
    participant SH as server_handle<br/>src/main.rs:1188
    participant WD as run_kg_watchdog<br/>src/services/liveness_harness.rs
    participant TAP as run_agent_event_tap<br/>src/services/kpi_compute.rs
    participant CT as CanaryNostrTap<br/>src/services/canary_nostr_tap.rs
    participant SIG as signal handlers<br/>src/main.rs:1234-1248

    M->>SH: let server_handle = server.handle()
    par detached background tasks
        M->>WD: tokio::spawn(run_kg_watchdog(harness, self_url, interval))
        Note over WD: VISIONCLAW_SELF_URL default http://127.0.0.1:{port} (src/main.rs:1195)<br/>VISIONCLAW_KG_WATCHDOG_SECS default 30 (src/main.rs:1197)
        loop every VISIONCLAW_KG_WATCHDOG_SECS (default 30s)
            WD->>M: GET /api/health on itself — this server IS the KG backend
            WD->>WD: drive kg_backend_up gauge, fire CANARY-VC-RESA-KG on every transition
        end
    and
        M->>TAP: tokio::spawn(run_agent_event_tap(kpi_repo)) (src/main.rs:1217)
        Note over TAP: REC-4 ADR-130 D5 — subscribes to the process-global /wss/agent-events hub,<br/>one volume row per envelope. Augmentation-Ratio numerator. Fail-open on lagged/closed.
    and
        alt CANARY_TAP_RELAY_URL is set
            M->>CT: CanaryNostrTap::from_env(harness) then tokio::spawn(tap.run())
            Note over CT: RES-a / ADR-130 D3 — lets Nostr-only repos fire canaries they cannot POST over HTTP<br/>feeds the SAME LivenessHarness.observe path, fail-open
        else unset
            M->>M: log "canary Nostr tap not started"
        end
    end
    M->>SIG: signal(SignalKind::terminate) and signal(SignalKind::interrupt)
    SIG->>SIG: tokio::select! on SIGTERM or SIGINT
    SIG->>SH: server_handle.stop(true) — graceful
    M->>M: server.await then "HTTP server stopped"
```

## VC-01.9 Route table 1 — root scope, WebSocket upgrades and OpenAPI
```mermaid
flowchart LR
    ROOT["App::new root — src/main.rs:1035-1052"]
    ROOT --> H1["GET /healthz<br/>consolidated_health_handler::liveness_probe<br/>src/main.rs:1037"]
    ROOT --> H2["GET /readyz<br/>consolidated_health_handler::readiness_probe<br/>src/main.rs:1038"]
    ROOT --> W1["GET /wss<br/>socket_flow_handler<br/>src/main.rs:1039"]
    ROOT --> W2["GET /wss/agent-events<br/>agent_events::agent_events_ws<br/>src/main.rs:1041 — ADR-059 s1 authenticated inbound agent_action ingest"]
    ROOT --> W3["GET /ws/speech<br/>speech_socket_handler<br/>src/main.rs:1042"]
    ROOT --> W4["GET /ws/mcp-relay<br/>mcp_relay_handler<br/>src/main.rs:1043"]
    ROOT --> W5["GET /ws/client-messages<br/>client_messages_handler::websocket_client_messages<br/>src/main.rs:1045"]
    ROOT --> W6["GET /ws/presence<br/>ws_presence<br/>src/main.rs:1047 — PRD-008 s5.3 Quest 3 multi-user sync"]
    ROOT --> O1["GET /swagger-ui/{_:.*}<br/>SwaggerUi<br/>src/main.rs:1049-1052"]
    ROOT --> O2["GET /api-docs/openapi.json<br/>openapi::ApiDoc::openapi<br/>src/main.rs:1051"]
    ROOT --> PAY["/pay/* — feature solid-pod-embed only<br/>pay_handler::configure_pay_routes<br/>src/main.rs:1033"]
    N1["these routes sit OUTSIDE the /api scope, so PublicDemoGuard and RbacGate do NOT apply<br/>WS auth including the ?token= query path see VC-03.2"]
    ROOT --- N1
    N2["DOC-DRIFT — src/handlers/quic_transport_handler.rs no longer defines QuicTransportServer.<br/>ADR-2066 removed it as dead code (constructed nowhere, routed nowhere, only re-exported at<br/>src/handlers/mod.rs). Only PostcardNodeUpdate/PostcardBatchUpdate remain (quic_transport_handler.rs:1-9),<br/>imported directly by fastwebsockets_handler.rs — not re-exported through handlers/mod.rs at all<br/>(handlers/mod.rs:111-117). It still has no configure fn and is registered nowhere in main.rs."]
    ROOT --- N2
```

## VC-01.10 Route table 2 — /api scope registration ORDER
```mermaid
flowchart TB
    S["web::scope('/api') — src/main.rs:1054"]
    S --> G1["wrap PublicDemoGuard::from_env — src/main.rs:1058"]
    G1 --> G2["wrap RbacGate::from_env — src/main.rs:1065"]
    G2 --> R1["1. POST /api/client-logs → client_log_handler::handle_client_logs<br/>src/main.rs:1067 — registered early to avoid scope conflicts, RBAC-allowlisted"]
    R1 --> R2["2. admin_rbac_handler::configure_routes<br/>src/main.rs:1069"]
    R2 --> R3["3. scope /settings + RateLimit::per_minute(60)<br/>settings::api::configure_routes — src/main.rs:1071-1074"]
    R3 --> R4["4. configure_ontology_derived_routes<br/>src/main.rs:1081"]
    R4 --> R5["5. configure_ontology_class_count_routes<br/>src/main.rs:1090"]
    R5 --> R6["6. api_handler::config — src/main.rs:1091"]
    R6 --> R7["7. workspace_handler::config — src/main.rs:1092"]
    R7 --> R8["8. admin_sync_handler::configure_routes — src/main.rs:1093"]
    R8 --> R9["9. validation_handler::config — src/main.rs:1094"]
    R9 --> R10["10. hexagonal group — see VC-01.12"]
    R10 --> R11["11. content group — see VC-01.13"]
    ORD["INVARIANT registration order — actix matches scopes in registration order by path segment<br/>and does NOT fall through a matched scope prefix. The broad /ontology scope inside<br/>api_handler::ontology::config would shadow /ontology/derived and /ontology/class-count to 404,<br/>so both MUST register before api_handler::config. Comments src/main.rs:1075-1089.<br/>Guarded by tests/resd_class_count_route.rs"]
    R4 --- ORD
    R5 --- ORD
    ORD2["same hazard inside /graph — actix claims the prefix for the FIRST web::scope('/graph'),<br/>so mixed-auth must live in ONE scope with per-resource .wrap()<br/>src/handlers/api_handler/graph/mod.rs:1488-1491"]
    R6 --- ORD2
```

## VC-01.11 Route table 3 — api_handler::config fan-out
```mermaid
flowchart LR
    C["api_handler::config<br/>src/handlers/api_handler/mod.rs:120"]
    C --> A1["GET /api/health → health_check (:121)"]
    C --> A2["GET /api/config → get_app_config (:122) — CQRS LoadAllSettings via hexser"]
    C --> F["files::config<br/>scope /files — src/handlers/api_handler/files/mod.rs:312<br/>POST /process, GET /get_content/{filename}<br/>POST /refresh_graph, POST /update_graph"]
    C --> G["graph::config — see VC-01.14"]
    C --> GS["graph_state_handler::config<br/>scope /graph — src/handlers/graph_state_handler.rs:394<br/>GET /state, GET /statistics, POST /nodes, GET|PUT|DELETE /nodes/{id}<br/>POST /edges, PUT /edges/{id}, POST /positions/batch"]
    C --> B["bots::config<br/>scope /bots — src/handlers/api_handler/bots/mod.rs:11<br/>GET|POST /data, POST /update, POST /initialize-swarm, GET /status<br/>GET /agents, POST /spawn-agent-hybrid, POST /submit-task, POST /interrupt<br/>GET /task-status/{id}, DELETE /remove-task/{id}"]
    C --> AN["analytics::config<br/>scope /analytics — src/handlers/api_handler/analytics/mod.rs:169<br/>POST clustering/{focus,cancel,dbscan}, community/detect, anomaly/{detect,toggle}<br/>sssp/{params,compute,toggle}, feature-flags<br/>GET params, constraints, stats, gpu-{metrics,status,features}<br/>clustering/status, anomaly/{current,config}, insights, insights/realtime<br/>sssp/status, dashboard-status, health-check, feature-flags"]
    C --> NO["nostr_handler::config<br/>scope /auth/nostr — src/handlers/nostr_handler.rs:53 — see VC-05"]
    C --> RF["ragflow_handler::config<br/>scope /ragflow — src/handlers/ragflow_handler.rs:620<br/>POST /session, /message, /chat, /session/enhanced<br/>GET /history/{session_id}, /history/enhanced/{session_id}"]
    C --> CO["constraints_handler::config<br/>scope /constraints — src/handlers/constraints_handler.rs:13<br/>POST /define, /apply, /remove, /validate — GET /list"]
    C --> ON["ontology::config<br/>scope /ontology — src/handlers/api_handler/ontology/mod.rs:1654 — see VC-20"]
    C --> OP["ontology_physics::configure_routes<br/>scope /ontology-physics — src/handlers/api_handler/ontology_physics/mod.rs:502<br/>POST /enable, /disable — GET /constraints, /trust-status — PUT /weights"]
    C --> SF["semantic_forces::config<br/>scope /semantic-forces — src/handlers/api_handler/semantic_forces.rs:677<br/>POST /dag/configure, /collision/configure — GET /hierarchy-levels, /config"]
    N["WS-0 — the duplicate /ontology scope in ontology_handler::config was collapsed<br/>into api_handler::ontology::config (src/handlers/api_handler/mod.rs:125-128)<br/>settings_handler::config is disabled in favour of settings::api (:132-133)"]
    C --- N
```

## VC-01.12 Route table 4 — hexagonal, health and MCP groups
```mermaid
flowchart LR
    S["/api scope — src/main.rs:1100-1118"]
    S --> P["configure_physics_routes<br/>scope /physics — src/handlers/physics_handler.rs:411<br/>POST start, stop, optimize, step, forces/apply, nodes/pin, nodes/unpin<br/>POST parameters, reset, settle-mode — GET status, settle-mode"]
    S --> SC["configure_schema_routes<br/>scope /schema — src/handlers/schema_handler.rs:268<br/>GET '', /llm-context, /node-types, /edge-types<br/>GET /node-types/{type}, /edge-types/{type}"]
    S --> NL["configure_nl_query_routes<br/>scope /nl-query — src/handlers/natural_language_query_handler.rs:235<br/>POST /translate, /explain, /validate — GET /examples"]
    S --> PF["configure_pathfinding_routes<br/>scope /pathfinding — src/handlers/semantic_pathfinding_handler.rs:117<br/>POST /semantic-path, /query-traversal, /chunk-traversal"]
    S --> SM["configure_semantic_routes<br/>scope /semantic — src/handlers/semantic_handler.rs:242<br/>POST /communities, /centrality, /shortest-path, generate-constraints, /cache/invalidate<br/>GET /statistics"]
    S --> IN["REMOVED ADR-2066 — configure_inference_routes<br/>the /api/inference scope was deleted, comment block at src/main.rs:1105<br/>every handler extracted an InferenceService that was never<br/>registered as app data, so each route 500'd at the extractor<br/>live reasoning path is GitHubSyncService::run_post_sync_reasoning"]
    S --> HE["consolidated_health_handler::configure_routes<br/>scope /health — src/handlers/consolidated_health_handler.rs:474<br/>GET '', /physics — scope /mcp POST /start, GET /logs<br/>plus a second /healthz and /readyz at :488-489"]
    S --> ME["metrics_handler::configure_routes<br/>GET /api/metrics — src/handlers/metrics_handler.rs:92"]
    S --> MM["configure_multi_mcp_routes<br/>scope /multi-mcp — src/handlers/multi_mcp_websocket_handler.rs:911<br/>GET /ws, GET /status, POST /refresh"]
    N["port and adapter hops behind these routes see VC-07 — GPU internals see VC-10"]
    S --- N
```

## VC-01.13 Route table 5 — content, governance and observability groups
```mermaid
flowchart LR
    S["/api scope — src/main.rs:1120-1178"]
    S --> PG["scope /pages + pages_handler::config<br/>src/main.rs:1120, src/handlers/pages_handler.rs:148 — GET ''"]
    S --> BO["scope /bots + api_handler::bots::config<br/>src/main.rs:1121"]
    S --> BV["bots_visualization_handler::configure_routes<br/>scope /visualization — src/handlers/bots_visualization_handler.rs:500<br/>GET /agents/ws, GET snapshot, POST initialize<br/>plus POST /bots/mock-agents at :513"]
    S --> GE["configure_graph_export_routes<br/>scope /graph-export — src/handlers/graph_export_handler.rs:319<br/>POST '', /share, /publish — GET /shared/{id}, /stats — DELETE /shared/{id}"]
    S --> OA["configure_ontology_agent_routes<br/>scope /ontology-agent — src/handlers/ontology_agent_handler.rs:434<br/>POST /discover, /read, /query, /traverse, /validate — GET /status<br/>nested scope /propose POST '' — see VC-05"]
    S --> DE["configure_decision_routes<br/>scope /decisions — src/handlers/decision_handler.rs:323<br/>GET /{urn}/trace — nested scope /record POST '' — PRD-022 W-B / ADR-048"]
    S --> SO["configure_solid_routes<br/>src/handlers/solid_proxy_handler.rs:1752<br/>re-exported at src/handlers/mod.rs:121 — see VC-05"]
    S --> IG["configure_image_gen_routes<br/>scope /image-gen — src/handlers/image_gen_handler.rs:779<br/>GET /health, /status/{job_id} — POST /submit, /agent-submit"]
    S --> BR["configure_briefing_routes<br/>scope /briefs — src/handlers/briefing_handler.rs:118<br/>POST '' submit_brief, POST /{brief_id}/debrief"]
    S --> MF["configure_memory_flash_routes<br/>POST /api/memory-flash and the batch route<br/>src/handlers/memory_flash_handler.rs:134-137"]
    S --> EP["configure_enrichment_proposals_routes<br/>POST /api/enrichment-proposals/{id}/decide<br/>src/handlers/enrichment_proposals_handler.rs:540"]
    S --> BI["configure_broker_inbox_routes<br/>scope /broker — src/handlers/broker_inbox_handler.rs:164<br/>GET /inbox, GET /cases/{id}, POST decide_as_operator — WS-12"]
    S --> IW["configure_ingest_writeback_routes<br/>POST /api/ingest/writeback — src/handlers/ingest_writeback_handler.rs:103 — GOV-4"]
    S --> LV["configure_liveness_routes<br/>scope /canary — src/handlers/liveness_harness_handler.rs:212<br/>POST /register, POST /observe/{canary_id}, GET /status — RES-a"]
    S --> KP["configure_kpi_routes<br/>scope /kpi — src/handlers/kpi_handler.rs:49<br/>GET /summary, GET /lineage/{snapshot_id} — REC-4 ADR-043"]
    S --> IL["configure_insight_loop_routes<br/>scope /insight-loop — src/handlers/insight_loop_handler.rs:91<br/>GET /trace, GET /trace/{case_id} — REC-10 PRD-023 WP-12"]
    S --> TR["configure_trace_routes<br/>GET /api/trace — src/handlers/trace_handler.rs:76<br/>REC-11 joins agent-events and broker decisions on did:nostr"]
    S --> LA["configure_layout_routes<br/>scope /layout — src/handlers/layout_handler.rs:282<br/>GET /modes, /status, /zones — POST /mode, /radial, /zones, /reset — ADR-031"]
    N["handler internals for this group see VC-04 and VC-05"]
    S --- N
```

## VC-01.14 Route table 6 — /api/graph mixed-auth scope and /api/settings
```mermaid
flowchart TB
    G["web::scope('/graph') — src/handlers/api_handler/graph/mod.rs:1492"]
    G --> RL["wrap RateLimit::per_minute(600) — scope default, public reads (:1493)"]
    RL --> P1["GET /api/graph/data → get_graph_data (:1495)"]
    RL --> P2["GET /api/graph/data/paginated → get_paginated_graph_data (:1496)"]
    RL --> P3["GET /api/graph/positions → get_graph_positions (:1497)"]
    RL --> P4["GET /api/graph/fold → fold::get_fold_plan (:1501) — Wave 3, read-only plan"]
    RL --> P5["GET /api/graph/auto-balance-notifications → get_auto_balance_notifications (:1502)"]
    RL --> T1["resource /node/{id}/relations + wrap RateLimit::per_minute(120)<br/>GET → get_node_relations (:1513-1516)"]
    RL --> T2["resource /node/{id}/expand + wrap RateLimit::per_minute(120)<br/>POST → expand_node (:1518-1521)"]
    RL --> T3["resource /query/pattern + wrap RateLimit::per_minute(120)<br/>POST → query_pattern (:1526-1529)"]
    RL --> A1["resource /update + wrap RequireAuth::power_user()<br/>POST → update_graph (:1537-1540)"]
    RL --> A2["resource /refresh + wrap RequireAuth::authenticated()<br/>POST → refresh_graph (:1544-1547)"]
    N1["S2 escalation — /update triggers a full bulk reload (re-fetch, re-process, rebuild from<br/>AddNodesFromMetadata). Destructive and expensive, so power_user (Admin), not Authenticated.<br/>/refresh only reads back GetGraphData, so any authenticated user may call it. Comments :1531-1543"]
    A1 --- N1
    N2["the Graph2VR reads scan every edge, so their 120/min per-resource ceiling stacks UNDER<br/>the 600/min scope limiter and is the stricter gate. Comment :1507-1512"]
    T1 --- N2
    SET["web::scope('/settings') + RateLimit::per_minute(60) — src/main.rs:1071-1074<br/>settings::api::configure_routes src/settings/api/settings_routes.rs:1736<br/>GET|PUT physics, constraints, rendering, node-filter, quality-gates, visual<br/>POST physics/reset-layout — GET all — POST|GET profiles — GET|DELETE profiles/{id}<br/>nested scope /user GET|PUT /filter (:1734-1736)"]
    SET2["settings round-trip and the OptimizedSettings/ProtectedSettings actors see VC-06"]
    SET --- SET2
```

## VC-01.15 ADR-2106 — the embedded pod pulls the published ontology at boot
```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>src/main.rs:848
    participant SB as spawn_boot_pull<br/>src/services/ontology_pull.rs:384
    participant PO as pull_once<br/>src/services/ontology_pull.rs:290
    participant GH as GitHub release<br/>ontology-latest
    participant ST as Storage (embedded pod)

    Note over M,SB: feature solid-pod-embed only — spawned right after init_solid_state,<br/>BEFORE pay state construction (src/main.rs:840-854)
    M->>SB: ontology_pull::spawn_boot_pull(Arc::clone(&solid_state.storage))
    SB->>SB: OntologyPullConfig::from_env (src/services/ontology_pull.rs:73-90)<br/>ONTOLOGY_PULL_URL, _ENABLED (default true), _TIMEOUT_SECS (default 120), _INTERVAL_SECS (default 3600)
    alt ONTOLOGY_PULL_ENABLED=false|0
        SB-->>M: return — logs "ontology pull disabled" (src/services/ontology_pull.rs:390), nothing spawned
    else enabled
        SB->>SB: tokio::spawn(async move { loop { ... } }) (src/services/ontology_pull.rs:393-408)
        loop every ONTOLOGY_PULL_INTERVAL_SECS (0 disables re-checks, one-shot)
            SB->>PO: pull_once(&fetch, storage, &cfg)
            PO->>GH: GET index.jsonld (MANIFEST, src/services/ontology_pull.rs:45)
            alt fetch fails or manifest missing visionflow:buildSha
                PO-->>SB: Err — fail-open, pod keeps what it held (src/services/ontology_pull.rs:12-16, :205-208)
            else visionflow:buildSha unchanged from the pod's own
                PO-->>SB: Ok(NoChange) — one small GET, nothing else fetched
            else build moved
                PO->>GH: GET SHA256SUMS (SUMS, src/services/ontology_pull.rs:46)
                PO->>GH: GET each of the five pod resources
                PO->>PO: verify each file's digest against SHA256SUMS
                alt any digest mismatch or missing sum entry
                    PO-->>SB: Err — write nothing (src/services/ontology_pull.rs:127)
                else all verified
                    PO->>ST: ensure containers, write a public-read WAC ACL at<br/>/public/ontology/.acl ONLY IF ABSENT (operator edits survive, :345, :354)
                    PO->>ST: write content, then the manifest LAST
                    PO-->>SB: Ok(Updated { build_sha, classes, triples })
                end
            end
        end
    end
    Note over PO,ST: fail-open throughout — any error is logged (log_outcome, src/services/ontology_pull.rs:411)<br/>and the pod keeps whatever it already held. Ten unit tests cover the sequence against<br/>MemoryBackend and a map-backed fetcher.
    Note over M,GH: rationale — deploy-jss PUTs to the in-process pod, unreachable from a hosted<br/>GitHub Actions runner (no self-hosted runner exists). Delivery is inverted — ontology-publish.yml's<br/>publish-release job attaches the five pod resources + SHA256SUMS to the rolling release<br/>ontology-latest, and the server now PULLS from it instead. See ES-09.13 (EXTERNAL) for the release job.
```

