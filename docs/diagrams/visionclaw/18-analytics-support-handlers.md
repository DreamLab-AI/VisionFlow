---
id: VC-18
title: Analytics support handlers and the analytics WebSocket
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2007, ADR-2009, ADR-2059, ADR-2138, ADR-2140]
sources:
  - ../project/src/handlers/api_handler/analytics/mod.rs
  - ../project/src/handlers/api_handler/analytics/feature_flags_handlers.rs
  - ../project/src/handlers/api_handler/analytics/insights_handlers.rs
  - ../project/src/handlers/api_handler/analytics/params_handlers.rs
  - ../project/src/handlers/api_handler/analytics/performance_handlers.rs
  - ../project/src/handlers/api_handler/analytics/stress_handlers.rs
  - ../project/src/handlers/api_handler/analytics/websocket_integration.rs
  - ../project/src/handlers/api_handler/analytics/state.rs
  - ../project/src/handlers/api_handler/analytics/types.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/src/handlers/api_handler/ontology/mod.rs
  - ../project/src/actors/gpu/analytics_telemetry.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/stress_majorization_actor.rs
  - ../project/src/models/simulation_params.rs
  - ../project/src/handlers/api_handler/analytics/cycles_handlers.rs
  - ../project/src/layout/scc.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/src/app_state.rs
  - ../project/docs/adr/ADR-2138-directed-cycles-are-a-server-analysis-with-one-adjacency.md
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
---

## For developers

These are the analytics support handlers: feature flags, parameters, performance, stress majorization, insights, the analytics WebSocket and the cycles report. Several of them reach ForceComputeActor stubs, and the register records which ones answer success without having any effect.

## For the business

These routes back the analytics panels and their settings. Where a route reports success without effect, the panel shows a change that never happened.

## VC-18.1 /analytics scope — auth wrapper and route table

```mermaid
flowchart LR
    SCOPE["web::scope(/analytics)<br/>analytics/mod.rs:196"]
    W1["wrap RequireAuth::authenticated().mutations_only()<br/>analytics/mod.rs:197"]
    WS["web::resource(/ws) wrap RequireAuth::authenticated()<br/>analytics/mod.rs:302-305"]

    subgraph POSTS["Authenticated mutations - POST"]
        P1["/params :199 update_analytics_params"]
        P2["/constraints :200 update_constraints"]
        P3["/focus :201 set_focus"]
        P4["/kernel-mode :202 set_kernel_mode"]
        P5["/feature-flags :236 update_feature_flags"]
        P6["/stress-majorization/trigger :221"]
        P7["/stress-majorization/reset-safety :225"]
        P8["/stress-majorization/params :229"]
        P9["/stress-majorization/configure :233"]
        P11["/sssp/toggle :216 with RateLimit::per_minute(60) :217"]
        P10["clustering, community, anomaly/detect, sssp/compute, pagerank, pathfinding<br/>see VC-15, anomaly/toggle :211"]
    end
    subgraph GETS["Public read-only telemetry - GET"]
        G1["/params :270 get_analytics_params"]
        G2["/constraints :271 get_constraints"]
        G3["/stats :272 get_performance_stats"]
        G4["/gpu-metrics :273 get_gpu_metrics"]
        G5["/gpu-status :274 get_gpu_status"]
        G6["/gpu-features :275 get_gpu_features"]
        G7["/insights :283 get_ai_insights"]
        G8["/insights/realtime :284 get_realtime_insights"]
        G9["/dashboard-status :295 get_dashboard_status"]
        G10["/health-check :296 get_health_check"]
        G11["/feature-flags :297 get_feature_flags"]
        G12["/stress-majorization/stats :288"]
        G13["/stress-majorization/config :292"]
        G14["/sssp/status :285 get_sssp_status"]
        G15["/cycles :286 cycles_handlers::get_cycles, see VC-18.10"]
    end

    SCOPE --> W1
    W1 --> POSTS
    W1 --> GETS
    SCOPE --> WS

    N1["INVARIANT ADR-2009 mutations_only() means every POST needs an authenticated pubkey while GETs stay public as harmless operational telemetry (rationale analytics/mod.rs:189-194)"]
    N2["The /ws resource takes the STRICTER wrap - RequireAuth::authenticated() with no mutations_only(), so even the read-only stream requires auth"]
    N3["update_feature_flags additionally takes an AuthenticatedUser extractor feature_flags_handlers.rs:34 - belt and braces on top of the scope wrap"]
    W1 -.- N1
    WS -.- N2
    P5 -.- N3
```

## VC-18.2 Feature flags — get, update, and what they actually gate

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant FH as feature_flags_handlers<br/>src/handlers/api_handler/analytics/feature_flags_handlers.rs:20
    participant ST as FEATURE_FLAGS static<br/>analytics/state.rs:15
    participant SS as sssp_handlers<br/>analytics/sssp_handlers.rs:171
    participant ON as ontology check_feature_enabled<br/>api_handler/ontology/mod.rs:330

    C->>FH: GET /analytics/feature-flags (route reg analytics/mod.rs:297)
    FH->>FH: reported_flags feature_flags_handlers.rs:13-18
    FH->>ST: ontology_validation from FEATURE_FLAGS.lock() feature_flags_handlers.rs:15
    FH->>SS: sssp_integration from stored_sssp_enabled feature_flags_handlers.rs:16, the stored physics.ssspEnabled, true if the settings actor does not answer (sssp_handlers.rs:171-179)
    FH-->>C: flags plus a two-entry description map (feature_flags_handlers.rs:26-29)
    C->>FH: POST /analytics/feature-flags (route reg analytics/mod.rs:236)
    Note over FH: requires AuthenticatedUser extractor feature_flags_handlers.rs:34 on top of the scope wrap
    FH->>ST: write ONLY ontology_validation feature_flags_handlers.rs:41
    Note over FH: a body sssp_integration is accepted and ignored (feature_flags_handlers.rs:40), the store owns it
    FH-->>C: success plus the reported flags (feature_flags_handlers.rs:42-48)

    rect rgb(250,228,228)
        Note over ST,ON: What the flags actually do
        ON->>ST: if !flags.ontology_validation (api_handler/ontology/mod.rs:333)
        Note over ON: ontology_validation is the ONLY flag read as a real GATE - it short-circuits ontology validation with an ErrorResponse
        Note over SS: sssp_integration is display-only and mirrors the store, the FEATURE_FLAGS copy is a placeholder (types.rs:314-317)
    end
    Note over SS: toggle_sssp (sssp_handlers.rs:112) builds sssp_toggle_patch (sssp_handlers.rs:16) and sends it through apply_physics_patch (sssp_handlers.rs:126, settings_routes.rs:617) - store, GPU actors and SQLite together
    Note over ST: RESOLVED ADR-2059 - seven inert flags removed (types.rs:307-317), struct keeps two fields (types.rs:319-323)
    Note over ST: OPEN ADR-2059 - ontology_validation in FEATURE_FLAGS is still process-global, unpersisted and shared across all users (state.rs:15), default true (types.rs:338)
```

## VC-18.3 Params handlers — analytics params, constraints, focus, kernel mode

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant PH as params_handlers<br/>src/handlers/api_handler/analytics/params_handlers.rs:17
    participant SET as OptimizedSettingsActor
    participant GPU as ForceComputeActor<br/>get_gpu_compute_addr app_state.rs:1600

    C->>PH: GET /analytics/params (route reg analytics/mod.rs:270)
    PH->>SET: send(GetSettings) params_handlers.rs:20
    alt settings actor error or mailbox error
        PH-->>C: 500 success=false (params_handlers.rs:22-41)
    else settings returned
        PH->>PH: create_default_analytics_params params_handlers.rs:44, fn :344 ignores the settings and returns fixed builder values (params_handlers.rs:349-354)
        PH-->>C: 200 AnalyticsParamsResponse
    end

    C->>PH: POST /analytics/params (route reg analytics/mod.rs:199)
    PH->>GPU: send(UpdateVisualAnalyticsParams) params_handlers.rs:63
    Note over GPU: the ForceComputeActor handler is a logged no-op returning Ok (force_compute_actor.rs:3375-3385), so the POST answers success and changes nothing

    C->>PH: GET /analytics/constraints (route reg analytics/mod.rs:271)
    PH->>GPU: send(GetConstraints) params_handlers.rs:112
    Note over GPU: always Err "Constraints should be handled by ConstraintActor" (force_compute_actor.rs:3388-3393)
    PH-->>C: 200 success=true with an EMPTY default ConstraintSet (params_handlers.rs:129-133) - the error is masked
    C->>PH: POST /analytics/constraints (route reg analytics/mod.rs:200)
    PH->>GPU: send(UpdateConstraints{constraint_data}) params_handlers.rs:159
    Note over GPU: parsed and cached for the next physics step (force_compute_actor.rs:3396-3413)
    PH->>GPU: send(GetConstraints) params_handlers.rs:163 to echo back the applied set, which always fails
    PH-->>C: falls through to 503 "GPU compute service not available" (params_handlers.rs:194-199) although the update was applied
    Note over GPU: residency change re-derives ENABLE_CONSTRAINTS next step - see VC-11.5 and VC-16.6

    C->>PH: POST /analytics/focus (route reg analytics/mod.rs:201)
    PH->>PH: set_focus params_handlers.rs:203 builds a FocusRegion
    PH->>GPU: send(UpdateVisualAnalyticsParams) params_handlers.rs:296, the same no-op handler

    C->>PH: POST /analytics/kernel-mode (route reg analytics/mod.rs:202)
    PH->>PH: set_kernel_mode params_handlers.rs:357 maps legacy, standard, dual_graph, advanced, visual_analytics (params_handlers.rs:365-378)
    PH->>GPU: send(SetComputeMode{mode}) params_handlers.rs:383
    Note over GPU: real - sets compute_mode and resyncs unified_params (force_compute_actor.rs:2825-2839)
    Note over PH,GPU: get_gpu_compute_addr returns Option, a None answers 503 rather than panicking (params_handlers.rs:97-104)
```

## VC-18.4 Performance handlers — stats, GPU metrics, status and features

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant PF as performance_handlers<br/>src/handlers/api_handler/analytics/performance_handlers.rs:55
    participant TEL as analytics_telemetry<br/>src/actors/gpu/analytics_telemetry.rs:190
    participant GPU as ForceComputeActor<br/>get_gpu_compute_addr app_state.rs:1600

    C->>PF: GET /analytics/stats (route reg analytics/mod.rs:272)
    PF->>GPU: get_performance_stats :55 sends GetPhysicsStats performance_handlers.rs:60
    C->>PF: GET /analytics/gpu-metrics (route reg analytics/mod.rs:273)
    PF->>PF: get_gpu_metrics performance_handlers.rs:149
    PF->>TEL: snapshot analytics_telemetry.rs:190 and total_cpu_fallbacks analytics_telemetry.rs:218 (call performance_handlers.rs:159-162)
    Note over TEL: per-kernel GPU-vs-fallback counters are PROCESS-GLOBAL, recorded by the GPU analytics actors via record_execution analytics_telemetry.rs:144
    alt gpu addr present
        PF->>GPU: send(GetGPUMetrics) performance_handlers.rs:167
        alt Ok(Ok(metrics))
            PF-->>C: metrics plus analytics_execution (performance_handlers.rs:168-173)
            Note over GPU: the metrics are hard-coded zeros (force_compute_actor.rs:3357-3372)
        else Ok(Err(e))
            PF-->>C: 500 with analytics_execution (performance_handlers.rs:175-182)
        else mailbox error
            PF-->>C: 503 WITHOUT analytics_execution (performance_handlers.rs:184-186)
        end
    else no GPU
        PF-->>C: 503 WITHOUT analytics_execution (performance_handlers.rs:189-193)
    end
    C->>PF: GET /analytics/gpu-status (route reg analytics/mod.rs:274)
    PF->>PF: get_gpu_status performance_handlers.rs:197
    C->>PF: GET /analytics/gpu-features (route reg analytics/mod.rs:275)
    PF->>PF: get_gpu_features performance_handlers.rs:281
    Note over TEL: INVARIANT a non-zero total_cpu_fallbacks means the zero-fallback intent has been violated this process lifetime (performance_handlers.rs:157-158) - the observable signal for the trust gap in VC-15.13
```

## VC-18.5 Stress-majorization handlers — they reach ForceComputeActor stubs

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SH as stress_handlers<br/>src/handlers/api_handler/analytics/stress_handlers.rs:12
    participant FCA as ForceComputeActor<br/>get_gpu_compute_addr app_state.rs:1600
    participant SMA as StressMajorizationActor<br/>src/actors/gpu/stress_majorization_actor.rs:311

    C->>SH: POST /stress-majorization/trigger (route reg analytics/mod.rs:221)
    SH->>FCA: send(TriggerStressMajorization) stress_handlers.rs:17
    FCA-->>SH: Err "should be handled by StressMajorizationActor" (force_compute_actor.rs:3457-3466)
    SH-->>C: 400 success=false (stress_handlers.rs:22-27)
    C->>SH: GET /stress-majorization/stats (route reg analytics/mod.rs:288)
    SH->>FCA: send(GetStressMajorizationStats) stress_handlers.rs:43
    FCA-->>SH: Err (force_compute_actor.rs:3469-3482)
    C->>SH: POST /stress-majorization/reset-safety (route reg analytics/mod.rs:225)
    SH->>FCA: send(ResetStressMajorizationSafety) stress_handlers.rs:70
    FCA-->>SH: Err (force_compute_actor.rs:3485-3497)
    C->>SH: POST /stress-majorization/params (route reg analytics/mod.rs:229)
    SH->>FCA: build UpdateStressMajorizationParams stress_handlers.rs:98 then send stress_handlers.rs:102
    FCA-->>SH: Ok after a log line, nothing forwarded (force_compute_actor.rs:3500-3510)
    C->>SH: POST /stress-majorization/configure (route reg analytics/mod.rs:233)
    SH->>FCA: send(config) stress_handlers.rs:131
    FCA->>FCA: learning_rate rewrites unified_params.temperature, momentum rewrites unified_params.damping (force_compute_actor.rs:3729-3739)
    C->>SH: GET /stress-majorization/config (route reg analytics/mod.rs:292)
    SH->>FCA: send(GetStressMajorizationConfig) stress_handlers.rs:158
    FCA-->>SH: config derived from unified_params plus fixed values (force_compute_actor.rs:3758-3783)
    Note over SMA: StressMajorizationActor has real handlers (stress_majorization_actor.rs:311, :330, :343, :406, :471) and errs "GPU not available for stress majorization" without a context (stress_majorization_actor.rs:95), but no /analytics route reaches it
    Note over SH,FCA: All six handlers guard on get_gpu_compute_addr() being Some and answer 503 otherwise (stress_handlers.rs:120)
    Note over SMA: ADR-2060 bit5 ENABLE_STRESS_MAJORIZATION is RESERVED - stress majorization is CPU-side per project/src/models/simulation_params.rs:77 and is not a GPU force channel
```

## VC-18.6 Insights handlers

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant IH as insights_handlers<br/>src/handlers/api_handler/analytics/insights_handlers.rs:33
    participant GS as GraphServiceSupervisor
    participant ST as CLUSTERING_TASKS and ANOMALY_STATE<br/>analytics/state.rs:9
    participant GPU as ForceComputeActor

    C->>IH: GET /analytics/insights (route reg analytics/mod.rs:283)
    IH->>GS: current_graph, an unreadable graph means nothing graph-derived is said (insights_handlers.rs:39-42)
    IH->>IH: gate_for_graph for the caller's pubkey (insights_handlers.rs:43-48)
    IH->>ST: CLUSTERING_TASKS.lock() insights_handlers.rs:50 and ANOMALY_STATE.lock() insights_handlers.rs:51
    IH->>IH: insight_scope, a gated caller gets the view's counts and the latest clusters less their hidden members (insights_handlers.rs:16-31, :57-58)
    IH-->>C: derived AI insights
    C->>IH: GET /analytics/insights/realtime (route reg analytics/mod.rs:284)
    IH->>GS: get_realtime_insights insights_handlers.rs:170 reads current_graph, a failure is a 500 (insights_handlers.rs:176-181)
    IH->>IH: the density is the caller's view's, gate_for_graph then insight_scope (insights_handlers.rs:182-188)
    Note over IH: OPEN a view of exactly one node makes the density denominator zero, so the reply carries a non-finite density (insights_handlers.rs:196-198)
    IH->>ST: CLUSTERING_TASKS.lock() insights_handlers.rs:190 and ANOMALY_STATE.lock() insights_handlers.rs:191
    opt gpu addr present insights_handlers.rs:240
        IH->>GPU: send(GetPhysicsStats) insights_handlers.rs:242, a gpu_failure_count above zero adds a warning (insights_handlers.rs:245-252)
    end
    C->>IH: GET /analytics/dashboard-status (route reg analytics/mod.rs:295)
    IH->>IH: gpu_available = get_gpu_compute_addr().is_some() insights_handlers.rs:271
    IH->>ST: CLUSTERING_TASKS.lock() insights_handlers.rs:272 and ANOMALY_STATE.lock() insights_handlers.rs:273
    C->>IH: GET /analytics/health-check (route reg analytics/mod.rs:296)
    IH-->>C: get_health_check insights_handlers.rs:331 health summary
    Note over ST: ANOMALY_STATE is the AGENT-HEALTH heuristic fed by MCP telemetry via /anomaly/toggle - it is NOT node_analytics.anomaly, which comes from the GPU LOF kernel (analytics/mod.rs:208-211)
    Note over IH,ST: All four insights endpoints are pure reads over graph data plus process-global state - none dispatch a GPU kernel<br/>INVARIANT since 2026-11 the two graph-derived ones answer over the caller's view, insights_handlers.rs:12-15
```

## VC-18.7 Analytics WebSocket — connect, subscribe, stream

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as RequireAuth::authenticated()<br/>analytics/mod.rs:304
    participant WSH as gpu_analytics_websocket<br/>analytics/websocket_integration.rs:501
    participant A as GpuAnalyticsWebSocket actor<br/>websocket_integration.rs:95

    C->>RT: GET /analytics/ws upgrade
    alt not authenticated
        Note over RT: rejected before the upgrade - the stream is auth-gated even though it is read-only
    else authenticated
        RT->>WSH: gpu_analytics_websocket (websocket_integration.rs:501)
        Note over WSH: the handler checks auth again, ?token= is accepted only in debug or dev-auth builds (websocket_integration.rs:520, :535-556)
        WSH->>A: ws::start(GpuAnalyticsWebSocket::new(app_state)) websocket_integration.rs:560
        A->>A: Actor::started websocket_integration.rs:384
        A->>C: message_type "connected" with clientId and a capabilities map websocket_integration.rs:390-405
        Note over A: capabilities advertise gpuMetrics, clusteringProgress, anomalyAlerts, insightsUpdates, realTimeUpdates
        A->>A: start_periodic_updates websocket_integration.rs:409, interval fixed ONCE from the prefs clamp websocket_integration.rs:357-362, default 5000 ms websocket_integration.rs:90
        loop every interval
            A->>A: ctx.run_interval(interval) websocket_integration.rs:364
            A->>C: send_gpu_metrics, send_clustering_progress, send_anomaly_alerts, send_insights_update websocket_integration.rs:373-376
            alt heartbeat older than 60s websocket_integration.rs:365-366
                Note over A: client considered dead and the session is stopped websocket_integration.rs:369
            end
        end
    end
    Note over A: MIN_UPDATE_INTERVAL_MS 100 websocket_integration.rs:11 and MAX_UPDATE_INTERVAL_MS 60000 websocket_integration.rs:12 bound the push cadence
    Note over A: This stream is JSON over text frames - it is NOT the binary position socket on /wss (see VC-13)
```

## VC-18.8 Analytics WebSocket — inbound message handling

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as GpuAnalyticsWebSocket StreamHandler<br/>websocket_integration.rs:420

    C->>A: ws::Message::Text
    A->>A: heartbeat = Instant::now() :424
    alt JSON parses as AnalyticsWebSocketMessage :426
        alt message_type == "updateSubscriptions"
            A->>A: parse SubscriptionPreferences :433
            A->>A: clamp update_interval_ms to MIN 100 and MAX 60000 :435-437, store prefs :438
            A->>C: "subscriptionsUpdated" echoing the clamped prefs :444-451
            Note over A: TENSION the running timer is not re-armed - start_periodic_updates is called only from started (websocket_integration.rs:409), so a changed interval is echoed but not applied
        else message_type == "requestImmediateUpdate"
            A->>C: send_gpu_metrics + send_clustering_progress + send_anomaly_alerts + send_insights_update :455-458
        else message_type == "ping"
            A->>C: "pong" with a server timestamp :461-468
        else unknown type
            Note over A: warn and ignore, no error frame is returned :471-473
        end
    else malformed JSON
        Note over A: parse failure is logged :476-477, the frame is dropped, the session stays open
    end
    Note over A: Ping and Pong frames also refresh the heartbeat :481-487, Close or a protocol error stops the actor :488-495
    Note over A: Message envelope AnalyticsWebSocketMessage :16 carries message_type, data, timestamp and an optional client_id
```

## VC-18.9 Analytics process-global state

```mermaid
classDiagram
    class FEATURE_FLAGS {
        +static_Lazy_Arc_Mutex state_rs_L15
        +ontology_validation_GATE_retained()
        +sssp_integration_PLACEHOLDER_reported_from_store()
        +seven_inert_flags_REMOVED_ADR_2059()
    }
    class ANOMALY_STATE {
        +static_Lazy_Arc_Mutex state_rs_L12
        +agent_health_heuristic_from_MCP_telemetry()
        +written_by_anomaly_toggle mod_L183
        +NOT_node_analytics_anomaly()
    }
    class CLUSTERING_TASKS {
        +static_Lazy_Arc_Mutex state_rs_L9
        +read_by_insights_handlers L20_L151_L233()
    }
    class AnalyticsTelemetry {
        +snapshot()
        +total_cpu_fallbacks()
        +record_execution()
        +process_global_per_kernel_counters()
    }
    FEATURE_FLAGS --> ANOMALY_STATE : both re-exported at mod_rs_L43
    CLUSTERING_TASKS --> ANOMALY_STATE : read together by insights

    note for FEATURE_FLAGS "ADR-2059: only ontology_validation gates; sssp_integration is reported from physics.ssspEnabled (feature_flags_handlers.rs:11-16)"
    note for FEATURE_FLAGS "OPEN ADR-2059: still process-global, unpersisted, shared (state.rs:15)"
    note for ANOMALY_STATE "Agent-health heuristic, distinct from the GPU LOF anomaly score on the wire (analytics/mod.rs:210)"
    note for AnalyticsTelemetry "Non-zero total_cpu_fallbacks violates the zero-fallback intent (analytics_telemetry.rs:218)"
```

## VC-18.10 GET /analytics/cycles — directed-cycle report on the CPU

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:286
    participant CY as get_cycles<br/>analytics/cycles_handlers.rs:60
    participant GS as GraphServiceSupervisor (graph_service_addr)
    participant SC as cycle_report<br/>src/layout/scc.rs:69

    C->>RT: GET /analytics/cycles?edgeType=subsumption|all|type
    Note over RT: a GET, so the scope's mutations_only wrap leaves it public like /api/graph/data
    RT->>CY: get_cycles(CyclesQuery)
    CY->>GS: send(GetGraphData) cycles_handlers.rs:65
    alt graph unavailable or mailbox error
        CY-->>C: error_json Graph unavailable cycles_handlers.rs:69 and cycles_handlers.rs:73
    else graph returned
        CY->>CY: select_edges(&graph.edges, edgeType) cycles_handlers.rs:48, default keeps only Edge::asserts_subsumption edges cycles_handlers.rs:52
        CY->>SC: cycle_report(node_ids, edges) cycles_handlers.rs:82
        SC->>SC: directed_adjacency then tarjan SCCs, cyclic nodes sorted, multi-node components largest first (scc.rs:70-94)
        CY-->>C: 200 CyclesResponse nodeCount, edgeCount, edgeFilter, cyclicNodeIds, components (cycles_handlers.rs:83-89)
    end
    Note over CY: CPU-only and recomputed per request, no GPU kernel and no cache, so desktop, XR and agents all read one answer
    Note over CY: the default edge set is the same predicate the GPU DAG ranker uses (see VC-16.8), but the ranker does not call layout::scc - only cycles_handlers.rs:16 and the CPU layout engine use it
```

**Tension (ADR-2138 vs code):** resolved at 7c953a40b — ADR-2138 was `proposed` / `none` / `inactive` while `GET /analytics/cycles` was already registered and served. At the pin its frontmatter is `decision_status: proposed`, `implementation_status: complete`, `activation_status: staged` (`docs/adr/ADR-2138-directed-cycles-are-a-server-analysis-with-one-adjacency.md:5-7`), so the ledger agrees with the code (`src/handlers/api_handler/analytics/mod.rs:286`, `src/handlers/api_handler/analytics/cycles_handlers.rs:60`). Ownership and activation are still open. The decision stays `proposed` because the record leaves acceptance to the owner. Activation stays `staged` until the XR Shapes tab first runs on a headset against a live graph (its `review_trigger`, `docs/adr/ADR-2138-directed-cycles-are-a-server-analysis-with-one-adjacency.md:13`).

**Debt:** resolved — `toggle_sssp` now writes through `apply_physics_patch` (`src/handlers/api_handler/analytics/sssp_handlers.rs:126`, `src/settings/api/settings_routes.rs:617`), and ADR-2140 records the item as closed (`docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:44`).

**Debt:** every `/analytics/stress-majorization/*` route sends to `ForceComputeActor`, because `get_gpu_compute_addr` returns that actor (`src/app_state.rs:1600`). Its handlers are stubs. Trigger, stats and reset return `Err`; params returns `Ok` without forwarding (`src/actors/gpu/force_compute_actor.rs:3457-3510`). The real `StressMajorizationActor` handlers (`src/actors/gpu/stress_majorization_actor.rs:311`) are unreachable from HTTP. Closing this needs the routes to address the supervised stress actor, or the routes to be removed.

**Tension (ADR-2140):** `POST /stress-majorization/configure` rewrites `unified_params.temperature` and `unified_params.damping` in `ForceComputeActor` (`src/actors/gpu/force_compute_actor.rs:3729-3739`) outside the stored physics document. That is a second write path for a physics value. This trace does not establish whether the value reaches the device before the next resync from the store.

**Debt:** `POST /analytics/params` and `POST /analytics/focus` reach a no-op handler that answers `Ok` (`src/actors/gpu/force_compute_actor.rs:3375-3385`). `GET /analytics/constraints` masks a guaranteed `Err` as an empty successful set (`src/handlers/api_handler/analytics/params_handlers.rs:129-133`). `POST /analytics/constraints` applies the update and then answers 503 (`src/handlers/api_handler/analytics/params_handlers.rs:194-199`). `GetGPUMetrics` returns hard-coded zeros (`src/actors/gpu/force_compute_actor.rs:3357-3372`).
