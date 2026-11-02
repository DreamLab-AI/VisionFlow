---
id: VC-04
title: Handler internals — graph, state, and domain route families
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2005, ADR-2007, ADR-2011, ADR-2111, ADR-2140]
sources:
  - ../project/src/handlers/api_handler/graph/mod.rs
  - ../project/src/handlers/api_handler/files/mod.rs
  - ../project/src/handlers/bots_handler.rs
  - ../project/src/handlers/api_handler/bots/mod.rs
  - ../project/src/handlers/api_handler/analytics/mod.rs
  - ../project/src/handlers/api_handler/ontology_physics/mod.rs
  - ../project/src/handlers/api_handler/semantic_forces.rs
  - ../project/src/handlers/ragflow_handler.rs
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/handlers/graph_export_handler.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/handlers/physics_handler.rs
  - ../project/src/application/physics_service.rs
  - ../project/src/adapters/actix_physics_adapter.rs
  - ../project/crates/visionclaw-domain/src/ports/gpu_physics_adapter.rs
  - ../project/src/handlers/kpi_handler.rs
  - ../project/src/services/kpi_compute.rs
  - ../project/src/adapters/sqlite_kpi_repository.rs
  - ../project/src/services/file_service.rs
  - ../project/src/services/ragflow_service.rs
  - ../project/src/services/provenance_trace.rs
  - ../project/src/services/intent_match.rs
  - ../project/docs/adr/ADR-2111-re-sequence-rgb-for-bridged-assets-and-delete-the-host-payment-store.md
  - ../project/src/handlers/consolidated_health_handler.rs
  - ../project/src/handlers/liveness_harness_handler.rs
  - ../project/src/utils/agent_key.rs
  - ../project/src/services/liveness_harness.rs
  - ../project/src/handlers/trace_handler.rs
  - ../project/src/handlers/validation_handler.rs
  - ../project/src/handlers/schema_handler.rs
  - ../project/src/handlers/semantic_handler.rs
  - ../project/src/handlers/workspace_handler.rs
  - ../project/src/handlers/pages_handler.rs
  - ../project/src/handlers/client_log_handler.rs
  - ../project/src/handlers/client_messages_handler.rs
  - ../project/src/handlers/pay_handler.rs
  - ../project/src/handlers/quic_transport_handler.rs
  - ../project/src/handlers/mod.rs
  - ../project/src/handlers/api_handler/mod.rs
  - ../project/src/main.rs
  - ../project/src/actors/gpu/analytics_telemetry.rs
  - ../project/src/handlers/api_handler/analytics/performance_handlers.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/src/utils/binary_protocol.rs
  - ../project/src/utils/validation/sanitization.rs
  - ../project/src/handlers/fastwebsockets_handler.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/agent_beam_actor.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/crates/visionclaw-domain/src/types/layout.rs
  - ../project/src/middleware/rbac_gate.rs
  - ../project/src/utils/nip98.rs
  - ../project/src/services/mcp_relay_manager.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/utils/auth.rs
verified_commit: 86cdd76355823a51bb64d503c005fe277726e724
---

## VC-04.1 api_handler graph — read path (data, paginated, positions, fold, relations, expand, pattern)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as get_graph_data<br/>src/handlers/api_handler/graph/mod.rs:217
    participant QH as graph_query_handlers<br/>AppState CQRS handler set
    participant GPU as ForceComputeActor<br/>fetch_settlement, graph/mod.rs:152
    participant POP as PopulationFilter::parse<br/>graph/mod.rs:120

    C->>H: GET /api/graph/data?graph_type=&exclude_linked_pages=
    par CQRS queries via execute_in_thread
        H->>QH: get_graph_data.handle(GetGraphData)
    and
        H->>QH: get_node_map.handle(GetNodeMap)
    and
        H->>QH: get_physics_state.handle(GetPhysicsState)
    and
        H->>GPU: fetch_settlement — get_gpu_compute_addr().send(GetSettlementState)
        alt GPU actor absent or no tick yet
            GPU-->>H: None — fall back to run-state (is_settled = not running)
        else telemetry present
            GPU-->>H: Some(SettlementSnapshot)
        end
    end
    H->>POP: PopulationFilter::parse(graph_type) — Agent / Knowledge / Ontology bit-mirrors (graph/mod.rs:106-146)
    alt query.graph_type set
        H->>H: filter nodes by population.matches(node_type, metadata)
    else absent
        H->>H: no population filtering
    end
    opt exclude_linked_pages=true
        H->>H: drop nodes whose origin resolves to linked_page (graph/mod.rs:298-309)
    end
    alt all three CQRS queries Ok
        H-->>C: 200 GraphResponseWithPositions{nodes,edges,metadata,settlement_state}
    else any thread execution error
        H-->>C: 500 Internal server error
    else any handler Err
        H-->>C: 500 Failed to retrieve graph data
    end
    Note over H: GET /api/graph/data/paginated (graph/mod.rs:361) is the SAME CQRS GetGraphData<br/>read plus offset/limit slicing — GraphQuery{page,page_size} — see graph/mod.rs:364
    Note over H: settlement_state now takes its running flag from the GPU snapshot (snap.physics_running, graph/mod.rs:328-336), not from the CQRS GetPhysicsState result,<br/>and GET /api/graph/settlement (route graph/mod.rs:1526, get_physics_state_summary graph/mod.rs:210) returns the compact {settlementState, telemetryLive, physicsRunning} view<br/>telemetryLive false means the numbers are run-state fallbacks, not measurements (graph/mod.rs:188-206)
    Note over H: GET /api/graph/positions (graph/mod.rs:632) skips CQRS entirely — reads<br/>ForceComputeActor.send(GetCurrentPositions) directly, 503 if gpu_addr is None
    Note over H: GET /api/graph/fold -> fold::get_fold_plan (graph/mod.rs:1533, module fold) —<br/>read-only Wave-3 fold-ladder plan, not expanded here (see governing doc fold section)
    Note over H: GET /api/graph/auto-balance-notifications (graph/mod.rs:597) — CQRS<br/>GetAutoBalanceNotifications{since_timestamp} via execute_in_thread, same Ok/Err split
```

## VC-04.2 api_handler graph — node/{id} relations, expand, pattern-query (RateLimit 120/min)
```mermaid
sequenceDiagram
    autonumber
    participant C as XR client (visual query builder)
    participant GR as get_node_relations<br/>src/handlers/api_handler/graph/mod.rs:1022
    participant EX as expand_node<br/>graph/mod.rs:1053
    participant QP as query_pattern<br/>graph/mod.rs:1490
    participant SNAP as fetch_graph_snapshot<br/>graph/mod.rs:997 — actor read, GetGraphStateActor then GetGraphData, NOT the CQRS handler

    C->>GR: GET /api/graph/node/{id}/relations
    GR->>GR: node_id = path & NODE_ID_MASK (src/utils/binary_protocol.rs)
    GR->>SNAP: fetch_graph_snapshot(&state)
    alt snapshot fetch fails
        SNAP-->>GR: Err
        GR-->>C: 500 Failed to retrieve graph data
    else node_id not in graph_data.nodes
        GR-->>C: 404 Node {id} not found
    else found
        GR->>GR: aggregate_relations(&edges, node_id)
        GR-->>C: 200 relations response
    end
    C->>EX: POST /api/graph/node/{id}/expand {limit, edge_type, direction}
    EX->>EX: node_id = path & NODE_ID_MASK — limit = clamp_expand_limit(req.limit)
    EX->>SNAP: fetch_graph_snapshot(&state)
    alt node not found
        EX-->>C: 404 Node {id} not found
    else found
        EX->>EX: expand_neighbours(edges, node_id, edge_type, direction, limit, node_index lookup)
        EX-->>C: 200 neighbours, heaviest-weight first, capped
    end
    C->>QP: POST /api/graph/query/pattern {triples:[{src,edgeType,tgt}]}
    Note over QP: pattern variables are JSON strings by convention #quot;?vN#quot —<br/>concrete ids are JSON numbers, masked with NODE_ID_MASK (graph/mod.rs:1114-1120)
    QP->>SNAP: fetch_graph_snapshot(&state) — SAME in-memory typed graph as /relations and /expand
    Note over SNAP: it skips the CQRS GetGraphData query handler on purpose: that sync handler spins a fresh<br/>Tokio runtime per call, a DoS lever on a public route (graph/mod.rs:988-996)
    Note over QP: deliberately NOT translated to SPARQL/Oxigraph — that store holds only<br/>the OWL ontology, not graph node/edge instances (graph/mod.rs:1109-1112)
    alt match_pattern Ok
        QP-->>C: 200 pattern bindings over the visible graph
    else empty pattern or empty variable name
        QP-->>C: 400 {error} (graph/mod.rs:1508-1510)
    end
    Note over GR,QP: all three sit UNDER RateLimit::per_minute(120) resource wraps<br/>(graph/mod.rs:1545-1561), stacked under the scope-wide 600/min limiter
```

## VC-04.3 api_handler graph — update (power_user bulk reload) and refresh (authenticated read-back)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RA as RequireAuth::power_user<br/>src/handlers/api_handler/graph/mod.rs:1571
    participant UG as update_graph<br/>graph/mod.rs:502
    participant FS as FileService<br/>src/services/file_service.rs
    participant MD as MetadataActor<br/>UpdateMetadata
    participant GS as GraphServiceSupervisor<br/>AddNodesFromMetadata
    participant RG as RequireAuth::authenticated<br/>graph/mod.rs:1578
    participant RF as refresh_graph<br/>graph/mod.rs:464
    participant QH as graph_query_handlers.get_graph_data

    rect rgb(240,225,225)
    Note over RA,GS: S2 escalation — POST /api/graph/update triggers a FULL bulk reload<br/>(re-fetch, re-process, rebuild). Destructive and expensive -> power_user (Admin) only (graph/mod.rs:1563-1575)
    C->>RA: POST /api/graph/update
    alt caller is not power_user
        RA-->>C: 403 Forbidden
    else power_user
        RA->>UG: call handler
        UG->>UG: FileService::load_or_create_metadata()
        alt load fails
            UG-->>C: 500 Failed to load metadata
        end
        UG->>UG: settings_addr.send(GetSettings)
        alt settings fetch fails
            UG-->>C: 500 Failed to retrieve application settings
        end
        UG->>FS: fetch_and_process_files(content_api, settings, &mut metadata)
        alt processed_files empty
            UG-->>C: 200 success — No updates needed
        else new files
            UG->>MD: metadata_addr.send(UpdateMetadata{metadata})
            UG->>GS: graph_service_addr.send(AddNodesFromMetadata{metadata})
            alt Ok(Ok(()))
                UG-->>C: 200 Graph updated with {n} new files
            end
        end
    end
    end
    rect rgb(225,240,225)
    Note over RG,QH: /refresh only reads back GetGraphData — any authenticated user may call it (graph/mod.rs:1576-1579)
    C->>RG: POST /api/graph/refresh
    RG->>RF: call handler
    RF->>QH: execute_in_thread(get_graph_data.handle(GetGraphData))
    alt Ok(Ok(graph_data))
        RF-->>C: 200 {success, message, data: GraphResponse{nodes,edges,metadata}}
    else Ok(Err(e))
        RF-->>C: 500 Failed to retrieve current graph data
    else thread error
        RF-->>C: 500 Internal server error
    end
    end
```

**Drift (RESOLVED, graph_state_handler removed in project commit 061d8cbf8):** this topic used to draw a second `/api/graph` family, `graph_state_handler` (nine routes: GET `/state`, GET `/statistics`, POST `/nodes`, GET, PUT and DELETE `/nodes/{id}`, POST `/edges`, PUT `/edges/{id}`, POST `/positions/batch`). Each handler ran a hexser CQRS handler over `state.graph_adapter`, the Oxigraph graph repository, directly, with no `GraphServiceSupervisor` message and so no actor notify, a third write path beside the actor path and the `graph_query_handlers` read path. The family never ran: `graph::config` registers `web::scope("/graph")` first (`src/handlers/api_handler/mod.rs:113`, comment at `src/handlers/api_handler/graph/mod.rs:1519`), actix claims the prefix for that first scope and answers 404 for every route of a later same-prefix scope, and none of the nine paths exists in the first. A same-order actix test confirmed 200 for the first scope and 404 for the second. The module, its `pub mod` and `.configure(...)` registrations and `tests/graph_state_integration.rs` were deleted. The surviving `/api/graph` surface is the single scope drawn in VC-04.1 to VC-04.3: `/settlement`, `/data`, `/data/paginated`, `/positions`, `/fold`, `/auto-balance-notifications`, `/node/{id}/relations`, `/node/{id}/expand`, `/query/pattern`, `/update`, `/refresh` (`src/handlers/api_handler/graph/mod.rs:1523`).

## VC-04.4 `files` scope — process/get_content/refresh_graph/update_graph
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant FP as fetch_and_process_files<br/>src/handlers/api_handler/files/mod.rs:13
    participant FS as FileService<br/>src/services/file_service.rs
    participant MD as MetadataActor<br/>UpdateMetadata
    participant GS as GraphServiceSupervisor<br/>AddNodesFromMetadata
    participant GPU as GPU compute actor<br/>GetNodeData (feature gpu)
    participant GC as get_file_content<br/>files/mod.rs:141

    C->>FP: POST /api/files/process
    FP->>FP: FileService::load_or_create_metadata()
    alt load fails
        FP-->>C: 500 Failed to initialize metadata
    end
    FP->>FP: settings_addr.send(GetSettings)
    alt settings fetch fails
        FP-->>C: 500 Failed to retrieve application settings
    end
    FP->>FS: fetch_and_process_files(content_api, settings, &mut metadata_store)
    alt Err
        FP-->>C: 500 Error processing files
    else Ok(processed_files)
        FP->>MD: metadata_addr.send(UpdateMetadata{metadata_store})
        FP->>FP: FileService::save_metadata(&metadata_store)
        alt save fails
            FP-->>C: 500 Failed to save metadata
        end
        FP->>GS: graph_service_addr.send(AddNodesFromMetadata{metadata_store})
        alt Ok(Ok(()))
            opt cfg(feature = "gpu")
                FP->>GPU: gpu_addr.send(GetNodeData) — best-effort, errors only logged
            end
            FP-->>C: 200 {status:success, processed_files}
        else Ok(Err(e)) or mailbox Err
            FP-->>C: 500 Failed to build graph
        end
    end
    C->>GC: GET /api/files/get_content/{filename}
    GC->>GC: reject filename containing ".." or leading "/" or NUL byte
    alt traversal detected
        GC-->>C: 400 Invalid file name
    end
    GC->>GC: canonicalize MARKDOWN_DIR then join(filename), canonicalize again
    alt file missing/unreadable
        GC-->>C: 404 File not found or unreadable
    else canonical path escapes MARKDOWN_DIR
        GC-->>C: 400 Invalid file name (files/mod.rs:181-190, second traversal check post-canonicalize)
    else
        GC-->>C: 200 body: file contents (plain text, not JSON)
    end
    Note over C: POST /api/files/refresh_graph (files/mod.rs:204) and POST /api/files/update_graph (files/mod.rs:243)<br/>are a DIFFERENT pair from /api/graph/refresh and /api/graph/update (VC-04.3) — same names,<br/>same GraphServiceSupervisor messages (GetGraphData / AddNodesFromMetadata), no RequireAuth<br/>wrap here (only the scope-wide RbacGate mutating classification applies)
    Note over GS: DOC-DRIFT — files::update_graph (files/mod.rs:243) calls AddNodesFromMetadata with metadata<br/>freshly loaded from disk via load_or_create_metadata, NOT the request body — the route accepts<br/>no JSON payload despite being a POST that #quot;updates#quot the graph
```

## VC-04.5 `bots` scope — data, initialize-swarm, spawn-agent-hybrid, submit/interrupt/status/remove task
```mermaid
sequenceDiagram
    autonumber
    participant C as Client / AgentDetailPanel
    participant GD as get_bots_data<br/>src/handlers/bots_handler.rs:138
    participant UD as update_bots_graph<br/>bots_handler.rs:110
    participant IH as initialize_hive_mind_swarm<br/>bots_handler.rs:160
    participant SA as spawn_agent_hybrid<br/>bots_handler.rs:290
    participant ST as submit_task / interrupt_task / get_task_status / remove_task<br/>bots_handler.rs:589,681,779,486
    participant GS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:1815<br/>GetBotsGraphData, :2031 UpdateBotsGraph
    participant AB as AgentBeamActor<br/>src/actors/agent_beam_actor.rs:246<br/>UpdateBotsGraph, :267 GetBotsGraphData
    participant TO as TaskOrchestratorActor<br/>state.get_task_orchestrator_addr()

    Note over C,TO: routes registered at src/handlers/api_handler/bots/mod.rs:11-28, including<br/>POST /settings-command (process_settings_command, bots_handler.rs:397), not expanded here
    C->>GD: GET /api/bots/data
    GD->>GD: agent_monitor_addr.send(GetChainPaymentsView),<br/>default on error (bots_handler.rs:142-146)
    GD->>GS: graph_service_addr.send(GetBotsGraphData) (bots_handler.rs:148)
    GS->>AB: forward when AgentBeamActor is registered,<br/>else Err AgentBeamActor not registered (graph_service_supervisor.rs:1823-1834)
    AB-->>GS: roster view from the feed,<br/>payment edges applied (agent_beam_actor.rs:272-284)
    alt Ok(Ok(view))
        GD-->>C: 200 {success, nodes, edges, chain} (bots_handler.rs:152-157)
    else any error
        GD-->>C: 200 with an empty GraphData view (bots_handler.rs:148-151),<br/>never a static fallback
    end
    C->>UD: POST /api/bots/data or /api/bots/update (AuthenticatedUser required)
    UD->>GS: do_send(UpdateBotsGraph{source: Rest, agents, targets: []})<br/>fire-and-forget (bots_handler.rs:117-121)
    GS->>AB: do_send, or warn when AgentBeamActor is not registered<br/>(graph_service_supervisor.rs:2035-2038)
    AB->>AB: feed.apply then broadcast 0x23<br/>presence/state frames (agent_beam_actor.rs:249-264)
    UD->>GS: send(GetBotsGraphData), empty view on error (bots_handler.rs:122-125)
    UD-->>C: 200 BotsResponse{success, "Roster accepted", nodes, edges} (bots_handler.rs:127-132)
    C->>IH: POST /api/bots/initialize-swarm (AuthenticatedUser)
    IH->>IH: build task string from custom_prompt or<br/>topology/strategy/max_agents/agent_types/enable_neural (bots_handler.rs:170-199)
    IH->>TO: send(CreateTask{agent, task, provider: PRIMARY_PROVIDER env default gemini,<br/>claude_flow_agent_id:None}) (bots_handler.rs:210-224)
    alt Ok(Ok(task_response))
        IH->>IH: CURRENT_SWARM_ID.write() = Some(task_id) (bots_handler.rs:233-234)
        IH-->>C: 202 Accepted {task_id, topology, strategy, agent_types, ...}
    else Ok(Err(e)) or mailbox Err
        IH-->>C: 500 Failed to create task / Actor communication error
    end
    C->>SA: POST /api/bots/spawn-agent-hybrid (AuthenticatedUser)
    SA->>TO: send(CreateTask{agent: agent_type, task, provider,<br/>claude_flow_agent_id:None}) (bots_handler.rs:300)
    SA-->>C: 202 Accepted or 500 (same Ok/Err split as initialize-swarm)
    C->>ST: POST /api/bots/submit-task (AuthenticatedUser)
    alt task empty after trim
        ST-->>C: 400 "task is empty" (bots_handler.rs:595-600)
    else
        ST->>TO: send(CreateTask) then observe CANARY-VC-D2-STEER<br/>(bots_handler.rs:616-647)
        ST-->>C: 202 {success, taskId, agentType, provider} or 500
    end
    C->>ST: POST /api/bots/interrupt {agentId} (AuthenticatedUser)
    ST->>TO: send(InterruptAgentTask{id}) resolves a task id<br/>or a swarm agent id (bots_handler.rs:700-704)
    alt Ok(Ok(resolved_task_id))
        ST-->>C: 200 {success, taskId, requestedId}
    else Ok(Err(InterruptError::Unresolved))
        ST-->>C: 422 {interruptible:false, resolution:unresolved}<br/>(bots_handler.rs:732-752)
    else other error
        ST-->>C: 500
    end
    C->>ST: GET /api/bots/task-status/{id}
    ST->>TO: send(GetTaskStatus{task_id})
    alt Ok(Ok(status))
        ST-->>C: 200 status
    else Ok(Err(e))
        ST-->>C: 404 {status:not_found, error} (bots_handler.rs:793-800)
    end
    C->>ST: DELETE /api/bots/remove-task/{id} (AuthenticatedUser)
    ST->>TO: send(StopTask{task_id}) (bots_handler.rs:494-498)
    alt Ok(Ok(()))
        ST-->>C: 200 TaskResponse{success:true}
    else Ok(Err(e)) or mailbox Err
        ST-->>C: 500 TaskResponse{success:false, error}
    end
    Note over GD,ST: GET /api/bots/status -> get_bots_connection_status (bots_handler.rs:268,<br/>reads state.bots_client.get_status())<br/>GET /api/bots/agents -> get_bots_agents (bots_handler.rs:275) -> fetch_hive_mind_agents reads<br/>state.bots_client.get_agents_snapshot() (bots_handler.rs:92-95), not AgentMonitorActor
```

**Drift (RESOLVED at 86cdd7635):** the `BOTS_GRAPH` static fallback store this panel used to draw is gone. `GET /api/bots/data` and the REST roster write both go through `GraphServiceSupervisor` to `AgentBeamActor` (ADR-2142 amendment, project commit `15acb9219`): the write is fire-and-forget and the read answers with an empty view on any error (`src/handlers/bots_handler.rs:117-125`, `:148-151`).

## VC-04.6 `analytics` scope — GPU-touching compute triggers and telemetry reads
```mermaid
sequenceDiagram
    autonumber
    participant C as Client (authenticated for POST)
    participant W as RequireAuth::authenticated().mutations_only()<br/>src/handlers/api_handler/analytics/mod.rs:169
    participant SS as compute_sssp<br/>sssp_handlers.rs:205
    participant AN as run_anomaly_detection<br/>analytics/mod.rs:92
    participant GM as GPUManagerActor / ShortestPathActor<br/>see VC-10
    participant GT as analytics_telemetry<br/>src/actors/gpu/analytics_telemetry.rs
    participant PM as get_gpu_metrics<br/>performance_handlers.rs:149

    Note over W: mutations_only() gates every state-mutating POST at authenticated(), and the read-only metric GETs carry no handler auth (analytics/mod.rs:161-166)<br/>they are anonymous only when the /api RbacGate allows public reads (RBAC_PUBLIC_READS=1, src/middleware/rbac_gate.rs:135-142, :172-177)
    Note over SS: the sibling toggle_sssp (sssp_handlers.rs:112) is now a physics patch: sssp_toggle_patch names only ssspEnabled and ssspAlpha (sssp_handlers.rs:16-22)<br/>and apply_physics_patch merges, validates, stores, pushes to the GPU actors and persists (sssp_handlers.rs:125-126, src/settings/api/settings_routes.rs:602-611)<br/>a validation failure answers 400, a store failure 500 (sssp_handlers.rs:145-166). The route is rate-limited at 60 per minute (analytics/mod.rs:187-191)
    Note over SS: the hard-coded /sssp/params routes are deleted (ADR-2140), GET /sssp/status reads the stored physics document (sssp_handlers.rs:182-193, test analytics/mod.rs:289-291)
    C->>W: POST /api/analytics/sssp/compute {sourceNode}
    W->>SS: call handler (mutation, requires auth)
    SS->>SS: source_node = body.sourceNode as u32, default 0
    SS->>GM: graph_service_addr.send(ComputeShortestPaths{source_node_id})
    alt Ok(Ok(_))
        SS-->>C: 200 {success:true, sourceNode}
    else Ok(Err(e)) or mailbox Err
        SS-->>C: 500 error
    end
    rect rgb(232,236,244)
    Note over AN,GM: GPU/CPU-fallback boundary — internals of the compute kernel see VC-10, VC-15
    C->>W: POST /api/analytics/anomaly/detect {method,k_neighbors,radius,feature_data,threshold}
    W->>AN: call handler (mutation)
    AN->>GM: anomaly::run_gpu_anomaly_detection(app_state, method, k_neighbors, radius, feature_data, threshold)
    GM-->>AN: Vec<anomaly>
    AN-->>C: 200 {success, anomalies, total, method}
    end
    C->>W: GET /api/analytics/gpu-metrics (read, no handler auth wrap)
    W->>PM: call handler
    PM->>GT: analytics_telemetry::snapshot() + total_cpu_fallbacks() — process-global counters (:154-162)
    alt gpu_addr present
        PM->>GM: gpu_addr.send(GetGPUMetrics)
        alt Ok(Ok(metrics))
            PM-->>C: 200 {metrics, analytics_execution} (performance_handlers.rs:168-173)
        else Ok(Err(e))
            PM-->>C: 500 {error, gpu_initialized:false, analytics_execution} (performance_handlers.rs:175-182)
        else mailbox Err
            PM-->>C: 503 GPU compute actor unavailable, no counters (performance_handlers.rs:184-187)
        end
    else no GPU actor
        PM-->>C: 503 GPU compute not available, no counters (performance_handlers.rs:189-193)
    end
    Note over PM: the in-code comment (performance_handlers.rs:154-158) says the counters ride every response including failure branches,<br/>but the two 503 branches carry none, so total_cpu_fallbacks is hidden exactly when the GPU actor is missing
    Note over C,PM: full route table (47 method routes counting the authenticated /ws upgrade: clustering, community, pagerank, pathfinding, stress-<br/>majorization, feature-flags, dashboard-status, health-check) registered at analytics/mod.rs:167-279, see VC-18
```

**Drift:** the comment at `src/handlers/api_handler/analytics/performance_handlers.rs:154-158` promises the per-kernel counters on every response, including failure branches. The mailbox-error and no-GPU branches return 503 through `service_unavailable!` without them (`src/handlers/api_handler/analytics/performance_handlers.rs:184-193`). Close it by attaching `analytics_execution` to both 503 bodies, or by narrowing the comment.

## VC-04.7 `ontology-physics` and `semantic-forces` — constraint application and the read-only semantic-forces views
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant EN as enable_ontology_physics<br/>src/handlers/api_handler/ontology_physics/mod.rs:107
    participant OA as OntologyActor<br/>state.ontology_actor_addr, GetOntologyReport
    participant GMA as GPUManagerActor<br/>state.gpu_manager_addr, ApplyOntologyConstraints
    participant TS as get_trust_status<br/>ontology_physics/mod.rs:441
    participant SF as semantic_forces handlers<br/>get_hierarchy_levels src/handlers/api_handler/semantic_forces.rs:27

    C->>EN: POST /api/ontology-physics/enable {ontologyId, mergeMode}
    EN->>EN: check_ontology_feature() — feature gate
    EN->>EN: merge_mode = replace|merge|add_if_no_conflict
    alt merge_mode invalid string
        EN-->>C: 400 Invalid merge mode
    end
    alt ontology_actor_addr is None
        EN-->>C: 503 Ontology actor not available
    end
    EN->>OA: send(GetOntologyReport{report_id: Some(ontology_id)})
    alt Ok(Ok(Some(report)))
        alt gpu_manager_addr is None
            EN-->>C: 503 GPU manager not available
        else present
            EN->>EN: build ConstraintSet from report.violations (Semantic kind,<br/>weight by Severity Error=1.0/Warning=0.6/Info=0.3)
            EN->>GMA: send(ApplyOntologyConstraints{constraint_set, merge_mode, graph_id:0})
            alt Ok(Ok(()))
                EN-->>C: 200 {success, activeConstraints, ontologyId, mergeMode}
            else Err
                EN-->>C: 500 error
            end
        end
    else Ok(Ok(None))
        EN-->>C: 400 "Ontology not found — Run validation first" (ontology_physics/mod.rs:243-252)
    else Ok(Err) or mailbox Err
        EN-->>C: 500 report retrieval / actor communication failed (ontology_physics/mod.rs:254-266)
    end
    Note over EN: disable_ontology_physics (ontology_physics/mod.rs:387) mirrors this with an empty/removed<br/>ConstraintSet via the SAME ApplyOntologyConstraints message
    C->>TS: GET /api/ontology-physics/trust-status
    TS->>TS: provenance_emitter::count_shapes_loaded / count_provenance_triples (ontology_repository.store())
    TS->>TS: shacl_gate::global_gate_mode() — writePaths honours mode, readPaths always advisory
    TS-->>C: 200 {status, shacl{shapesLoaded,engine:shape-driven,gateModes,w3cEnforcement},<br/>provenance{triplesStored,appendOnly:true}, federation{status:deferred, PRD-022 WS-3}}
    Note over TS: DIVERGENCE — federation.status is hardcoded #quot;deferred#quot — SPARQL federation is<br/>relay-mediated future work, not wired in this commit (ontology_physics/mod.rs:492-495)
    C->>SF: GET /api/semantic-forces/hierarchy-levels
    alt gpu_manager_addr is None
        SF-->>C: 500 GPU manager not initialized (semantic_forces.rs:30-36)
    else present
        SF->>GMA: send(GetHierarchyLevels)
        SF-->>C: 200 {maxLevel, nodeLevels, levelCounts} or 500 per Ok/Err (semantic_forces.rs:38-61)
    end
    C->>SF: GET /api/semantic-forces/config
    SF->>SF: read stored physics setting from sqlite_settings_repository, return its semantic group, default on a miss (semantic_forces.rs:66-80)
    C->>SF: POST /api/semantic-forces/hierarchy/recalculate (AuthenticatedUser)
    SF->>GMA: send(RecalculateHierarchy) (semantic_forces.rs:84-98)
    Note over SF: Force PARAMETERS are no longer configured here: the /dag/configure and /collision/configure routes are gone.<br/>They live in the central physics settings semantic group (PUT /api/settings/physics), validated and applied by that one path (semantic_forces.rs:3-6, ADR-2140)<br/>The relationship-type registry stays: GET/POST /relationship-types, GET/PUT /relationship-types/uri, POST /relationship-types/reload (semantic_forces.rs:393-424)
    Note over SF: DRIFT: docs/diagrams/visionclaw/01-boot-and-app-state.md still lists POST /dag/configure and /collision/configure for this scope<br/>whereas the route table registers neither (src/handlers/api_handler/semantic_forces.rs:393-425)
    Note over C: `constraints` scope (src/handlers/constraints_handler.rs) shares this settings-then-GPU-push<br/>shape — POST /api/constraints/define (constraints_handler.rs:22) validates, then merges<br/>{visualisation.graphs.{knowledge,visionclaw}.physics.computeMode = 2} into app_settings<br/>(constraints_handler.rs:66) BEFORE a best-effort GPU push (constraint failures there are only<br/>warn-logged, never fail the request, constraints_handler.rs:100) — POST /validate<br/>(validate_constraint_definition, constraints_handler.rs:282 per VC-16.6) is the one route in that scope<br/>with NO settings or GPU round-trip, pure request validation
```

## VC-04.8 `ragflow` scope — external HTTP boundary (RAGFlow API)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SM as send_message<br/>src/handlers/ragflow_handler.rs:52
    participant CS as create_session<br/>ragflow_handler.rs:139
    participant RS as RAGFlowService<br/>src/services/ragflow_service.rs:102
    participant TTS as speech_service.text_to_speech<br/>state.speech_service

    C->>CS: POST /api/ragflow/session {userId}
    CS->>RS: create session (state.ragflow_service)
    alt state.ragflow_service is None
        CS-->>C: 503 RAGFlow service is not available
    end
    CS-->>C: 200 CreateSessionResponse{success, session_id}
    C->>SM: POST /api/ragflow/message {question, stream, sessionId, enableTts}
    alt state.ragflow_service is None
        SM-->>C: 503 RAGFlow service is not available
    end
    SM->>SM: session_id = request.session_id or state.ragflow_session_id (fallback)
    rect rgb(225,225,245)
    Note over RS: PROCESS BOUNDARY — external HTTP call, credentials from env
    SM->>RS: send_message(session_id, question, false, None, stream.unwrap_or(true))
    RS->>RS: RAGFlowService::new reads RAGFLOW_API_KEY, RAGFLOW_API_BASE_URL and RAGFLOW_AGENT_ID through config_from (ragflow_service.rs:115-142),<br/>every request then carries Authorization Bearer (ragflow_service.rs:172)
    Note over RS: an absent or blank key is RAGFlowError::NotConfigured (ragflow_service.rs:122) and boot logs a warning, so state.ragflow_service stays None and the 503 branches above fire (main.rs:431).<br/>A key without base URL or agent id is a ParseError (ragflow_service.rs:124-130)
    end
    alt Ok(response_stream)
        opt enable_tts
            SM->>TTS: actix_web::rt::spawn — text_to_speech(question, SpeechOptions::default())
        end
        loop per streamed answer chunk
            SM->>SM: map answer -> Bytes{answer, success:true} JSON
            opt enable_tts and chunk non-empty
                SM->>TTS: spawn text_to_speech(answer_clone, ...) fire-and-forget
            end
        end
        SM-->>C: 200 HttpResponse::Ok().streaming(mapped_stream)
    else Err(e)
        SM-->>C: error_json! Failed to send message
    end
    Note over C,RS: /ragflow/chat and /ragflow/session/enhanced (ragflow_handler.rs:631,635) are the<br/>same external-boundary shape with an enhanced request/response envelope, /chat runs<br/>ValidationService::validate_ragflow_chat on the payload (ragflow_handler.rs:304)
    Note over C,RS: GET /history/{session_id} (ragflow_handler.rs:638) is NOT local: get_session_history makes an outbound<br/>GET to RAGFlow with the Bearer key (src/services/ragflow_service.rs:312-328), so the read path crosses the same boundary
```

## VC-04.11 `graph-export` scope — rate limit, serialization, shared-link lifecycle
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant EX as export_graph<br/>src/handlers/graph_export_handler.rs:109
    participant SH as share_graph<br/>graph_export_handler.rs:148
    participant GET as get_shared_graph<br/>graph_export_handler.rs:194
    participant PUB as publish_graph<br/>graph_export_handler.rs:267
    participant DEL as delete_shared_graph<br/>graph_export_handler.rs:277
    participant H as GraphExportHandler<br/>graph_export_handler.rs:29 web::Data — rate limiter, shared_graphs map, serialization_service

    Note over C,H: the whole /graph-export scope is wrapped in RequireAuth::authenticated() (graph_export_handler.rs:321)
    C->>EX: POST /api/graph-export {} (ExportRequest)
    EX->>H: check_rate_limit(client_ip)
    alt remaining_exports == 0
        EX-->>C: 429 Rate limit exceeded
    else check errors
        EX-->>C: 500 Rate limit check failed
    end
    EX->>H: get_current_graph(&app_state)
    EX->>H: serialization_service.export_graph(&graph, &request)
    alt Ok
        EX-->>C: 200 export_response (json/gexf/graphml/csv/dot per ExportFormat)
    else Err
        EX-->>C: 500 Export failed
    end
    C->>SH: POST /api/graph-export/share {} (ShareRequest)
    SH->>H: check_rate_limit then get_current_graph (same as export_graph)
    SH->>H: serialization_service.create_shared_graph(&graph, &request)
    alt Ok((shared_graph, share_response))
        SH->>H: shared_graphs.write().insert(shared_graph.id, shared_graph)
        SH-->>C: 200 share_response
    else Err
        SH-->>C: 500 Failed to create shared graph
    end
    C->>GET: GET /api/graph-export/shared/{id}?password=
    GET->>GET: Uuid::parse_str(id)
    alt malformed id
        GET-->>C: 400 Invalid share ID format
    end
    GET->>H: shared_graphs.read().get(&share_id)
    alt not found
        GET-->>C: 404 Shared graph not found
    else is_expired()
        GET-->>C: 410 Gone — Shared graph has expired
    else access_limit_reached()
        GET-->>C: 403 Access limit reached
    else password_hash Some and query password missing/wrong
        GET-->>C: 401 Password required / Invalid password
    else ok
        GET->>H: shared_graphs.write().increment_access()
        GET->>GET: std::fs::read(shared_graph.file_path)
        GET-->>C: 200 body — content-type by ExportFormat, Content-Encoding gzip if compressed
    end
    C->>PUB: POST /api/graph-export/publish
    PUB-->>C: 501 Not Implemented — Graph publishing not yet implemented (:272-274)
    Note over PUB: DIVERGENCE — the route is registered and reachable but the handler is a stub<br/>returning 501 unconditionally, no publish path exists in this commit
    C->>DEL: DELETE /api/graph-export/shared/{id}
    DEL->>DEL: Uuid::parse_str(id), 400 on a malformed id (graph_export_handler.rs:281-286)
    DEL->>H: shared_graphs.write().remove(&share_id)
    alt Some(graph)
        DEL->>DEL: std::fs::remove_file(file_path) — failure only warn-logged
        DEL-->>C: 200 {message, deleted_id}
    else None
        DEL-->>C: 404 Shared graph not found
    end
    Note over C,DEL: GET /api/graph-export/stats (get_export_stats, graph_export_handler.rs:308) is a second stub: it answers 501<br/>"Export statistics not yet implemented" unconditionally (graph_export_handler.rs:309-311)
```

**Tension:** `configure_routes` builds a fresh `GraphExportHandler` on every call (`src/handlers/graph_export_handler.rs:315`), and that handler owns its `shared_graphs` map and rate-limit map as fresh `Arc`s (`src/handlers/graph_export_handler.rs:38-45`). `configure_routes` runs inside the `HttpServer::new` factory (`src/main.rs:972`, `:1196`), which actix calls once per worker, and the server runs four workers (`src/main.rs:1265`). Each worker therefore holds its own share map and its own export budget: a share created on one worker answers 404 on the other three, and the 20-per-hour limit is per worker, not per client. Source tracing only; not exercised at runtime. Close it by building the handler once outside the factory and cloning the `web::Data` into each worker, as `semantic_handler.rs` does for its limiter (`src/handlers/semantic_handler.rs:264-282`).

## VC-04.12 `layout` scope — mode/radial/zones/reset; the mode is the stored physics.layoutMode (ADR-2140)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SM as set_layout_mode<br/>src/handlers/layout_handler.rs:45
    participant PP as apply_physics_patch<br/>src/settings/api/settings_routes.rs:606 — the one physics write path
    participant GPU as GPU compute actor<br/>UpdateSimulationParams / SetRadialLayout / ResetPositions / GetSettlementState
    participant RL as set_radial_layout<br/>layout_handler.rs:198
    participant RS as reset_layout<br/>layout_handler.rs:330

    C->>SM: POST /api/layout/mode {mode, transitionMs} (AuthenticatedUser, RateLimit 60/min layout_handler.rs:368-372)
    SM->>SM: LayoutMode::parse(mode), camelCase or quality-gate name (layout_handler.rs:64)
    alt unknown mode string
        SM-->>C: 400 "unknown layout mode" (layout_handler.rs:66-71), no silent ForceDirected
    end
    alt stored_physics().layout_mode == mode
        SM->>SM: nothing written, the repeated page-load call must not rewrite, broadcast or wake the GPU (layout_handler.rs:76-90)
    else changed
        SM->>PP: apply_physics_patch({layoutMode: mode}, auth.pubkey) (layout_handler.rs:92-98)
        PP->>PP: merge, validate_physics_settings, validate_for_gpu, UpdateSettings (settings_routes.rs:636-679)
        PP->>GPU: UpdateSimulationParams built from the stored physics (settings_routes.rs:684-694)
        alt PhysicsPatchError::BadRequest
            SM-->>C: 400 from e.into_response() (layout_handler.rs:101)
        else PhysicsPatchError::Internal
            SM->>SM: persisted = Err(e), warn logged (layout_handler.rs:102-105)
        end
    end
    alt mode.is_gpu_resident() (ForceDirected, Radial, Clustered, Hierarchical, DagLeftRight, TypeClustering, crates/visionclaw-domain/src/types/layout.rs:108-118)
        SM-->>C: 200 {success:true, positions:[]} or 200 {success:false, error} when persisted is Err (layout_handler.rs:112-125)
    else Spectral / Temporal (CPU one-shot)
        SM->>SM: a failed write answers 200 {success:false} before any placement (layout_handler.rs:130-136)
        SM->>SM: no GPU round-trip — graph_service_addr.send(GetGraphData) then compute_layout(mode, nodes, edges) (layout_handler.rs:139-177)
        SM-->>C: 200 {success:true, positions:[{id,x,y,z}]}
    end
    C->>RL: POST /api/layout/radial {mode, focusNode} (no auth extractor in the handler)
    RL->>RL: unknown radial mode answers 200 {success:false} (layout_handler.rs:217-227)
    RL->>GPU: send(SetRadialLayout{mode, focus_node}) (layout_handler.rs:230-232)
    RL-->>C: 200 {success:true|false} per Ok/Err or missing GPU actor (layout_handler.rs:233-262)
    C->>RS: POST /api/layout/reset
    RS->>GPU: send(ResetPositions) (layout_handler.rs:333-334)
    RS-->>C: 200 {success:true|false, message} (layout_handler.rs:335-358)
    Note over C,RS: GET /modes (layout_handler.rs:365) reports the stored mode and the seven live modes, GET /status (layout_handler.rs:374) folds GetSettlementState telemetry (layout_handler.rs:285-293)<br/>POST /zones (layout_handler.rs:375) only counts the body (TODO at layout_handler.rs:317), GET /zones (layout_handler.rs:376) returns an empty list
    Note over SM,RS: every failure branch of /mode, /radial and /reset answers HTTP 200 with success:false except the two 400s above, so a client must read the body, not the status
```

**Drift (RESOLVED at 86cdd7635):** this panel used to draw `set_layout_mode` sending `SetLayoutMode` straight to the GPU actor. The mode is now one stored field, `physics.layoutMode`, written through `apply_physics_patch` like `PUT /api/settings/physics` (`src/handlers/layout_handler.rs:13-22`, `:92-98`; project commits `58b05852d`, `db3ab82fa`).

**Tension:** `set_radial_layout` (`src/handlers/layout_handler.rs:198-201`) and `reset_layout` (`src/handlers/layout_handler.rs:330`) take no `AuthenticatedUser`, while `set_layout_mode` does (`src/handlers/layout_handler.rs:46`). They still sit under the `/api` RbacGate, which demands `WriteGraph` for any POST (`src/middleware/rbac_gate.rs:179-189`), so the difference is only in the handler's own posture.

## VC-04.13 `physics` scope — port/adapter hop, PhysicsService to ActixPhysicsAdapter to PhysicsOrchestratorActor
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SS as start_simulation<br/>src/handlers/physics_handler.rs:126
    participant PS as PhysicsService<br/>src/application/physics_service.rs:51
    participant PA as ActixPhysicsAdapter<br/>src/adapters/actix_physics_adapter.rs:23
    participant POA as PhysicsOrchestratorActor<br/>InitializePhysicsMessage / ComputeForcesMessage
    participant EB as EventBus<br/>SimulationStartedEvent

    C->>SS: POST /api/physics/start {timeStep,damping,springConstant,...} (power_user)
    SS->>SS: user.require_power_user()?
    alt not power_user
        SS-->>C: 403 Forbidden
    end
    SS->>SS: graph_access::current_graph(&graph_service) reads the graph through GraphServiceSupervisor (physics_handler.rs:133), then builds PhysicsParameters + SimulationParams
    Note over SS: max_iterations is clamped to MAX_PHYSICS_ITERATIONS 10000 (physics_handler.rs:124, :161)<br/>INVARIANT: every mutating physics route calls require_power_user first, including stop, step, forces/apply, nodes/pin, nodes/unpin, parameters, reset, optimize, physics_handler.rs:132,187,230,261,279,299,320,334,368
    SS->>PS: physics_service.start_simulation(graph, sim_params)
    Note over PS,PA: hexagonal port — PhysicsService holds Arc~RwLock~dyn GpuPhysicsAdapter~~<br/>(visionclaw_domain::ports::gpu_physics_adapter::GpuPhysicsAdapter trait)
    PS->>PA: adapter.initialize(graph.clone(), physics_params)
    alt actor_addr is None (first call)
        PA->>POA: PhysicsOrchestratorActor::new(SimulationParams::default(), None, Some(graph)).start()
    end
    PA->>POA: send(InitializePhysicsMessage::new(graph, params))
    PA-->>PS: Ok(())
    PS->>PS: simulation_id = format!("sim-{uuid}")
    PS->>EB: event_bus.publish(SimulationStartedEvent{simulation_id, physics_profile, node_count})
    alt Ok(simulation_id)
        SS-->>C: 200 StartSimulationResponse{simulation_id, status:started}
    else Err(e)
        SS-->>C: 500 Failed to start simulation
    end
    Note over PA,POA: compute_forces() / update_positions() / step() / simulate_until_convergence()<br/>each wrap ONE actor message (ComputeForcesMessage, UpdatePositionsMessage, PhysicsStepMessage,<br/>SimulateUntilConvergenceMessage) behind a tokio::time::timeout(self.timeout) —<br/>timeout maps to GpuPhysicsAdapterError::ComputationError("Actor communication timeout")
    Note over C,POA: POST /api/physics/stop, /optimize, /step, /forces/apply, /nodes/pin,<br/>/nodes/unpin, /parameters, /reset all route through the SAME PhysicsService port<br/>to ONE ActixPhysicsAdapter to the SAME PhysicsOrchestratorActor — GET /status (physics_handler.rs:196)<br/>bypasses the port entirely and reads app_state.get_gpu_compute_addr().is_some() directly
    Note over PA,POA: the adapter starts its own PhysicsOrchestratorActor with gpu_compute_addr None (src/adapters/actix_physics_adapter.rs:121-127,<br/>constructor src/actors/physics_orchestrator_actor.rs:195-198), and main.rs builds the service on a fresh ActixPhysicsAdapter::new() (src/main.rs:855-857)
```

**Tension:** the `/api/physics` port drives a private `PhysicsOrchestratorActor` that has no GPU compute address (`src/adapters/actix_physics_adapter.rs:121-127`, `src/actors/physics_orchestrator_actor.rs:195-198`, `src/main.rs:855`). The comment in `layout_handler.rs` relies on this ("reachable but harmless … nothing they send reaches the device", `src/handlers/layout_handler.rs:79-84`). So `/api/physics/start` and `/parameters` return success and publish `SimulationStartedEvent` without changing the simulation the clients see, which runs in the GPU actors behind `apply_physics_patch` (VC-04.12). Source tracing only. Close it by routing the port to the live GPU actors or retiring the mutating `/api/physics` routes.

## VC-04.14 `kpi` scope — compute-on-read, agent-event volume tap, SQLite lineage
```mermaid
sequenceDiagram
    autonumber
    participant C as Client (KPI dashboard)
    participant SU as summary<br/>src/handlers/kpi_handler.rs:25
    participant KC as KpiComputeService::compute_and_persist<br/>src/services/kpi_compute.rs:387
    participant KR as SqliteKpiRepository<br/>src/adapters/sqlite_kpi_repository.rs:274
    participant EN as enrichment_repo<br/>decisions_since
    participant LH as LivenessHarness<br/>observe(CANARY-VC-REC4-KPI)
    participant TAP as run_agent_event_tap<br/>kpi_compute.rs:646
    participant HUB as agent_events::hub<br/>subscribe()

    C->>SU: GET /api/kpi/summary
    SU->>KC: compute_and_persist()
    rect rgb(232,240,232)
    Note over KC,KR: process boundary — SQLite (kpi_agent_events, kpi_snapshots, kpi_lineage tables)
    KC->>KR: count_agent_events_since(window_start) — KPI_WINDOW_MS lookback
    KC->>EN: enrichment_repo.decisions_since(window_start), then decided_cases_since (kpi_compute.rs:399-413)
    KC->>KR: trajectories_since(window_start) (kpi_compute.rs:414-418)
    KC->>KC: decided_cases_with_intent(decided_rows, trajectories) (kpi_compute.rs:270), then hitl_precision(cases) — warranted divided by decided (kpi_compute.rs:213, call :419-420)
    KC->>KC: augmentation_ratio(agent_volume, escalation_volume) -> (value, confidence)
    KC->>KR: insert_snapshot_with_lineage(ar_snapshot, ar_lineage) -> ar_id
    KC->>KC: trust_variance(outcomes) — Gini-Simpson dispersion, 30-day rolling
    KC->>KR: insert_snapshot_with_lineage(tv_snapshot, tv_lineage) -> tv_id
    end
    KC->>LH: harness.observe(CANARY_REC4_KPI, evidence) — best-effort, warn on failure (kpi_compute.rs:504-513)
    KC->>KR: insert_snapshot_with_lineage(hitl_snapshot, hitl_lineage) — only when decided is above zero (kpi_compute.rs:554)
    KC->>KC: assemble four tiles (kpi_compute.rs:622) — Augmentation Ratio, Trust Variance and<br/>HITL Precision computed, Mesh Velocity still KpiTile::awaiting (kpi_compute.rs:545)
    Note over KC: INVARIANT (DDD invariant 8, ADR-2110) — a case resolved by the reserved system<br/>identity system:whelk-gate is excluded from BOTH terms of HITL Precision, so a window of<br/>gate rejections reports no value rather than a precision no human earned<br/>is_system_actor kpi_compute.rs:143, SYSTEM_WHELK_GATE kpi_compute.rs:140
    Note over KC: INVARIANT (ADR-2110 auditor counter-example, b2baa2d16) — a case id must occupy<br/>WHOLE delimited URN segments before a trajectory is correlated to a case, because the URNs<br/>come off an agent-controlled envelope, so a substring match would let one agent attach a<br/>mismatched intent to another agent's case — urn_names_case kpi_compute.rs:244 shares<br/>urn_names_segment with the trace verdict, intent_match.rs:163
    alt Ok(summary)
        SU-->>C: 200 KpiSummary (four tiles)
    else Err(e)
        SU-->>C: 500 {error}
    end
    C->>SU: GET /api/kpi/lineage/{snapshot_id}
    SU->>KC: lineage_for(snapshot_id)
    KC->>KR: lineage_for(snapshot_id) — DERIVED_FROM trail (WP-8 AC3)
    SU-->>C: 200 {snapshot_id, lineage}
    par background volume tap — src/main.rs:1297 tokio::spawn(run_agent_event_tap(kpi_repo))
        TAP->>HUB: subscribe() — same seam the render actor uses
        loop rx.recv().await — never returns, fail-open on lagged/closed channel
            HUB-->>TAP: AgentEventEnvelope
            TAP->>TAP: derive agent_did — did:nostr from pubkey (x-only hex) or source_urn fallback
            TAP->>KR: record_agent_trajectory(NewAgentTrajectory{...}) — one row per envelope
        end
    end
    Note over KC: REC-4 ADR-130 D5 — numerator is wss/agent-events window count, denominator<br/>is ACSP escalation volume (enrichment_decisions) — see VC-24, VC-25 for the elevation<br/>and insight-loop producers of these source events
```

## VC-04.16 `consolidated_health` — unified_health_check, physics probe, MCP relay controls
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant UH as unified_health_check<br/>src/handlers/consolidated_health_handler.rs:63
    participant SYS as check_system_metrics<br/>consolidated_health_handler.rs:104
    participant MCP as check_mcp_metrics<br/>consolidated_health_handler.rs
    participant PH as check_physics_simulation<br/>consolidated_health_handler.rs:282
    participant SR as start_mcp_relay<br/>consolidated_health_handler.rs:427
    participant GL as get_mcp_logs<br/>consolidated_health_handler.rs:443

    C->>UH: GET /api/health
    UH->>UH: app_state.get_degraded_reason() — e.g. Oxigraph store init failure
    alt degraded reason present
        UH->>UH: health_status = degraded, push reason
    end
    UH->>SYS: check_system_metrics(&mut health_status, &mut issues)
    SYS->>SYS: sysinfo System::new_all().refresh_all() — cpu/memory/disk/gpu
    alt cpu_usage or memory_usage or disk_usage exceeds HIGH_*_THRESHOLD
        SYS->>SYS: health_status = degraded, push issue
    end
    UH->>MCP: check_mcp_metrics().await — computed ONCE (D5)
    Note over UH,MCP: D5 fix — mcp_status_label derives from the SAME check as the mcp block<br/>so /api/health reports one MCP truth, not two disagreeing fields (:75-78)
    UH->>UH: check_service_metrics(&app_state, mcp_status_label, ...)
    alt health_status healthy but issues non-empty
        UH->>UH: health_status = degraded
    end
    UH-->>C: 200 HealthResponse{status,timestamp,issues,system,services,mcp}
    C->>PH: GET /api/health/physics
    PH-->>C: 200 physics simulation health detail
    C->>SR: POST /api/health/mcp/start
    SR-->>C: 200/error — relay start result
    C->>GL: GET /api/health/mcp/logs?...
    GL-->>C: 200 LogQuery-filtered MCP relay logs
    Note over C,GL: root /healthz and /readyz (consolidated_health_handler.rs:457 liveness_probe, :463 readiness_probe) sit<br/>OUTSIDE /api (no RbacGate) — configure_routes (consolidated_health_handler.rs:474) ALSO re-registers them at<br/>/api/healthz and /api/readyz (consolidated_health_handler.rs:488-489 per VC-01.7) — same handlers, second registration
    Note over SR,GL: /api/health/* is on the RbacGate public allowlist (src/middleware/rbac_gate.rs:55-61), and neither MCP route has a handler auth wrap (consolidated_health_handler.rs:479-483)
```

**Tension:** `POST /api/health/mcp/start` and `GET /api/health/mcp/logs` are reachable without any credential. The whole `api/health` prefix is public in the RbacGate (`src/middleware/rbac_gate.rs:55-61`, `:154-159`), and the routes carry no `RequireAuth` (`src/handlers/consolidated_health_handler.rs:479-483`). The start route runs `docker exec -d multi-agent-container … npm run mcp:start` when the relay is not running (`src/services/mcp_relay_manager.rs:171-187`), then blocks its actix worker thread for two seconds with `std::thread::sleep` (`src/services/mcp_relay_manager.rs:194`). The logs route returns the tail of `/tmp/mcp-server.log` (`src/services/mcp_relay_manager.rs:210-219`). An anonymous caller can therefore start a process in another container, read its log, and stall one of the four workers per request. Source tracing only; the `PublicDemoGuard` refuses mutating methods only when `PUBLIC_DEMO=read-only` (`src/main.rs:1127-1130`). Close it by moving the two MCP routes under an authenticated scope, or off `/api/health`.

## VC-04.17 `canary` scope (`liveness_harness_handler`) — X-Agent-Key gate, register/observe/status
```mermaid
sequenceDiagram
    autonumber
    participant C as Reporting repo / Godot dev client
    participant AU as canary_write_authorised<br/>src/handlers/liveness_harness_handler.rs:38 release / :52 dev
    participant RG as register<br/>liveness_harness_handler.rs:100
    participant OB as observe<br/>liveness_harness_handler.rs:138
    participant ST as status<br/>liveness_harness_handler.rs:233
    participant LH as LivenessHarness<br/>src/services/liveness_harness.rs

    C->>RG: POST /api/canary/register {canaryId,description,kind,ownerRepo,wave,shaAtRegistration}
    RG->>AU: canary_write_authorised(req)
    alt release build (no debug_assertions, no dev-auth)
        AU->>AU: expected = VISIONCLAW_AGENT_KEY env — provided = X-Agent-Key header
        AU->>AU: crate::utils::agent_key::check_agent_key(expected, provided) (agent_key.rs:13) — subtle ConstantTimeEq on the key bytes (agent_key.rs:15),<br/>a length mismatch rejects at once because the length is not secret
        alt env unset/empty OR header missing/mismatched
            AU-->>RG: false — FAIL CLOSED
        end
    else debug or dev-auth build
        AU-->>RG: true — unauthenticated dev flow preserved (:49-51)
    end
    alt not authorised
        RG-->>C: 401 {error: missing or invalid X-Agent-Key}
    else authorised
        RG->>LH: harness.register(&CanaryRegistration{...sha_at_registration default current_sha()})
        alt Ok(())
            RG-->>C: 200 {registered:true, canary_id, kind, owner_repo, sha_at_registration}
        else Err(e)
            RG-->>C: 500 store_error_body(e)
        end
    end
    C->>OB: POST /api/canary/observe/{canary_id} {evidence}
    OB->>AU: canary_write_authorised(req) — SAME gate
    alt not authorised
        OB-->>C: 401
    else authorised
        OB->>LH: harness.observe(&canary_id, &evidence)
        alt Ok(fire_id)
            OB-->>C: 200 {fired:true, canary_id, fire_id, sha}
        else Err(NotFound)
            OB-->>C: 404 {fired:false, error: unknown canary}
        else Err(e)
            OB-->>C: 500 store_error_body(e)
        end
    end
    C->>ST: GET /api/canary/status (no auth gate — read-only)
    ST->>LH: harness.status()
    ST-->>C: 200 {kg_backend_up, sha, canaries} (status_body, liveness_harness_handler.rs:209-228)
    Note over ST: Opt-in: the body gains a jev_triage hypothesis only when the opt-in Jev option is on (VISIONCLAW_JEV_HEALTH_TRIAGE=1) and the state is degraded,<br/>it never feeds armed, fired or kg_backend_up (liveness_harness_handler.rs:165-179, :205-208)
    Note over AU: ADR-06 D11 insecure-defaults posture — fail-closed release gate mirrors<br/>socket_flow_handler::http_handler::is_insecure_defaults_allowed and settings::auth_extractor
    Note over C,AU: the canary scope is mounted inside /api (src/main.rs:1238), so the RbacGate runs first and a POST needs WriteGraph (src/middleware/rbac_gate.rs:179-189)<br/>verify_access accepts only the dev bypasses, NIP-98 or legacy session headers, never X-Agent-Key (src/utils/auth.rs:211-218, :259-265, :329-353)
```

**Tension:** a release-build reporting repo that sends only `X-Agent-Key` never reaches `canary_write_authorised`. The `/api` RbacGate classifies `POST /api/canary/*` as a mutation needing `WriteGraph` (`src/middleware/rbac_gate.rs:179-189`) and resolves identity through `verify_access`, which has no `X-Agent-Key` path (`src/utils/auth.rs:191-195`, `:259-265`, `:329-353`), so the write is refused with 401 at the gate. The panel's handler-level gate is correct, but in release it sits behind a second credential the caller may not hold. Dev builds with `VISIONCLAW_DEV_MODE=1` grant every request (`src/utils/auth.rs:211-218`), which hides this. Source tracing only. Close it by allowlisting the canary write routes in the gate and keeping the `X-Agent-Key` check, or by documenting that a canary writer must also present NIP-98.

## VC-04.18 `GET /api/trace` — REC-11 unified provenance join (did:nostr key)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant UT as unified_trace<br/>src/handlers/trace_handler.rs:35
    participant PTS as ProvenanceTraceService<br/>src/services/provenance_trace.rs
    participant KR as sqlite_kpi_repository<br/>state.sqlite_kpi_repository
    participant ER as sqlite_enrichment_repository<br/>state.sqlite_enrichment_repository
    participant LH as liveness_harness<br/>observe(CANARY_REC11_TRACE)

    C->>UT: GET /api/trace?agent=did:nostr:...&window_ms=
    UT->>PTS: ProvenanceTraceService::new(sqlite_enrichment_repository, sqlite_kpi_repository)
    Note over PTS,ER: read-time JOIN over stores that already exist — NOT a new store (ADR-130)
    UT->>UT: window = window_ms or TRACE_WINDOW_MS (30-day default)
    UT->>PTS: service.query(window, agent)
    PTS->>KR: agent-events / hook-trajectory rows (kpi_agent_events)
    PTS->>ER: broker decision rows (enrichment_decisions)
    Note over PTS: pod git-marks (solid-pod-rs) source is default-off — reported under<br/>sources_absent, incorporated only when a --features git pod supplies them
    alt Ok(trace)
        opt trace.joins_multiple_source_kinds()
            UT->>LH: liveness_harness.observe(CANARY_REC11_TRACE, evidence) — best-effort, debug-logged on skip
            Note over LH: INVARIANT (REC-11 acceptance) — fires ONLY on observed live traffic that<br/>genuinely joins >= 2 live source kinds under one did:nostr, never synthetic
        end
        UT-->>C: 200 ProvenanceTrace{sourcesPresent,sourcesAbsent,totalRecords,joins,maxJoinSpan}
        Note over PTS: FR5.2 (EXP-AC-005) — each agent-event record now carries the agent's<br/>DECLARED intent verbatim and an intent_match verdict (provenance_trace.rs:203-211).<br/>Decision and git-mark records carry neither. A null verdict means NO CLAIM WAS MADE,<br/>which is not the same as a claim that failed — intent_match.rs:214
    else Err(e)
        UT-->>C: 500 {error}
    end
```

## VC-04.19 `validation` scope — schema dry-run test and static capability stats
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant VP as validate_payload<br/>src/handlers/validation_handler.rs:344
    participant VS as ValidationService<br/>validation_handler.rs:11 — settings/physics/ragflow/bots/swarm schemas
    participant SAN as Sanitizer::sanitize_json<br/>src/utils/validation/sanitization.rs
    participant GS as get_validation_stats<br/>validation_handler.rs:382

    C->>VP: POST /api/validation/test/{type} {payload}
    VP->>VP: validation_type = path {type}, extract_client_id(req)
    alt type not in settings|physics|ragflow|bots|swarm
        VP-->>C: 400 invalid_validation_type
    else recognised
        VP->>VS: validate_{type}(&payload)
        VS->>SAN: Sanitizer::sanitize_json(&mut sanitized_payload)
        VS->>VS: schema.validate(&sanitized_payload, &mut ValidationContext)
        VS->>VS: validate_{type}_custom(&sanitized_payload) — type-specific extra rules
        alt Ok(sanitized_payload)
            VP-->>C: 200 {status:valid, sanitized_payload, validation_type}
        else Err(error)
            VP-->>C: error.to_http_response() (DetailedValidationError)
        end
    end
    C->>GS: GET /api/validation/stats
    GS-->>C: 200 static capability descriptor — supported_endpoints, security_features<br/>(input_sanitization, schema_validation, rate_limiting, xss/sql-injection/path-traversal prevention)
    Note over VS: ValidationService is registered as app data once (src/main.rs:846, :1086), and it also runs on a real request path:<br/>EnhancedRagFlowHandler builds its own instance and validates every /ragflow/chat payload (src/handlers/ragflow_handler.rs:203, :304)
    Note over VS: no free-function validators module (validate_iri, check_sql_injection) exists under src/ at this revision, so it is not what this service calls
```

**Drift (corrected):** this panel used to say `ValidationService` is the consumer of a free-function `validators::{validate_iri, check_sql_injection, …}` module and is reachable only through the dry-run test route. Neither holds at `86cdd7635`: no such functions exist under `src/`, and `EnhancedRagFlowHandler` runs `validate_ragflow_chat` on the live `/ragflow/chat` path (`src/handlers/ragflow_handler.rs:203`, `:304`).

## VC-04.20 `schema` scope — GraphServiceSupervisor read, SchemaService cache refresh
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant GS as get_schema<br/>src/handlers/schema_handler.rs:70
    participant GSA as GraphServiceSupervisor<br/>web::Data~Addr~GraphServiceSupervisor~~ (schema_handler.rs:73)
    participant SS as SchemaService<br/>web::Data~Arc~SchemaService~~
    participant LC as get_llm_context<br/>schema_handler.rs:126
    participant NT as get_node_types / get_edge_types<br/>schema_handler.rs:149,177

    C->>GS: GET /api/schema (AuthenticatedUser)
    GS->>GSA: graph_state_actor.send(GetGraphData)
    alt Ok(Ok(graph_data))
        GS->>SS: schema_service.update_schema(&graph_data) — refresh cache from live graph
        GS->>SS: get_schema() and get_llm_context()
        GS-->>C: 200 SchemaResponse{schema, llm_context}
    else Ok(Err(e))
        GS-->>C: 500 Failed to retrieve graph data
    else mailbox Err
        GS-->>C: 500 Actor communication error
    end
    C->>LC: GET /api/schema/llm-context (AuthenticatedUser)
    LC->>SS: schema_service.get_llm_context() — cached, no actor round-trip
    LC-->>C: 200 text/plain — human-readable schema description for LLM consumption
    C->>NT: GET /api/schema/node-types (and /edge-types) (AuthenticatedUser)
    NT->>SS: read cached type/count table
    NT-->>C: 200 {node_types:[{node_type,count},...]}
    Note over GSA: DIVERGENCE — another graph-data read path beside api_handler::graph's graph_query_handlers CQRS read (VC-04.1)<br/>and its GraphStateActor snapshot read (VC-04.2) — schema_handler reads via a directly-injected Addr~GraphServiceSupervisor~<br/>(the parameter is still named graph_state_actor, schema_handler.rs:73)
    Note over C,NT: GET /schema/node-types/{type} and /edge-types/{type} (schema_handler.rs:203,239) are the SAME<br/>cached-lookup shape narrowed to one type name, an unknown type answers error_json (500), not 404 (schema_handler.rs:220-225)
```

## VC-04.22 `semantic` — graph-analysis service, and the removed inference stack
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant DC as detect_communities<br/>src/handlers/semantic_handler.rs:75
    participant SEM as SemanticService<br/>web::Data~Arc~SemanticService~~
    participant RI as REMOVED ADR-2066<br/>was src/handlers/inference_handler.rs

    C->>DC: POST /api/semantic/communities {algorithm, minClusterSize}
    DC->>DC: graph_access::current_graph(&graph_service) — reads through web::Data~Addr~GraphServiceSupervisor~~, not the old shared Arc RwLock GraphData (semantic_handler.rs:80)
    DC->>SEM: semantic_service.initialize(graph)
    alt init fails
        DC-->>C: 500 Failed to initialize
    end
    DC->>DC: algorithm = louvain|label_propagation|connected_components|hierarchical (default louvain)
    DC->>SEM: detect_communities(CommunityDetectionRequest{algorithm,min_cluster_size})
    alt Ok(result)
        DC-->>C: 200 CommunitiesResponse{clusters,cluster_sizes,modularity,computation_time_ms}
    else Err(e)
        DC-->>C: 500 Failed to detect communities
    end
    Note over DC,SEM: /centrality, /shortest-path, /constraints/generate (compute_centrality :114,<br/>compute_shortest_path :167, generate_constraints :197, routes semantic_handler.rs:293-301) share the SAME<br/>read-graph-then-initialize-then-compute shape — /cache/invalidate (:242) and<br/>GET /statistics (:227) skip the graph read entirely

    Note over SEM: INVARIANT: caller-supplied work is bounded: PageRank iterations 1000, top_k 1000, constraints 10000 (semantic_handler.rs:256,258,260,285), damping clamped to 0..1 (semantic_handler.rs:129),<br/>and the /semantic scope is rate-limited at 30 per minute per verified pubkey, IP when unauthenticated (semantic_handler.rs:262-292)
    Note over SEM: SemanticService is now registered as app data (src/main.rs:864-869, :1088), and before ER-25 every /api/semantic route returned 500 at the extractor
C--xRI: POST /api/inference/run — route no longer registered
    Note over RI: REMOVED ADR-2066 — the whole Phase 7 inference stack was deleted as dead code.<br/>src/handlers/inference_handler.rs, src/application/inference_service.rs and<br/>src/events/inference_triggers.rs are gone, and the registration (comment block at<br/>src/main.rs:1178-1182) was removed with them. Removal rationale recorded at src/handlers/mod.rs:46-51
    Note over RI: root cause — all seven handlers extracted web::Data of Arc RwLock InferenceService<br/>but InferenceService was never registered as app data anywhere, so every<br/>/api/inference/* route 500'd at the extractor. The live reasoning path is<br/>GitHubSyncService::run_post_sync_reasoning — see VC-20.3
```

## VC-04.23 `workspace` scope — RequireAuth + RateLimit(60/min), WorkspaceActor CRUD with timeout
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant W as scope /workspace<br/>src/handlers/workspace_handler.rs:32
    participant LW as list_workspaces<br/>workspace_handler.rs:48
    participant CW as create_workspace<br/>workspace_handler.rs:172
    participant WA as WorkspaceActor<br/>send_with_default_timeout / .send()

    Note over W: registered as .wrap(RequireAuth::authenticated()) then .wrap(RateLimit::per_minute(60)) (:35-36)<br/>actix runs the last-registered wrap outermost, so the rate limiter sees the request before the auth check
    C->>W: any /api/workspace/* request
    W->>W: RateLimit::per_minute(60), then RequireAuth::authenticated()
    C->>LW: GET /api/workspace/list?page=&pageSize=&sortBy=&sortDirection=
    LW->>LW: WorkspaceQuery::validate() (validator crate)
    alt validation fails
        LW-->>C: 400 {success:false, message, workspaces:[], total_count:0}
    end
    LW->>WA: send_with_default_timeout(&workspace_actor, GetWorkspaces{query}, "Workspace")
    alt Ok(Ok(response))
        LW-->>C: 200 WorkspaceListResponse
    else Ok(Err(e))
        LW-->>C: 500 WorkspaceListResponse::error
    else Err(ActorTimeoutError::Timeout{duration,actor_type})
        LW-->>C: 504 Gateway Timeout — request timeout
    else Err(other)
        LW-->>C: 500 Service temporarily unavailable
    end
    C->>CW: POST /api/workspace/create {CreateWorkspaceRequest} (AuthenticatedUser)
    CW->>CW: payload.validate()
    alt invalid
        CW-->>C: 400 WorkspaceResponse::error
    end
    CW->>WA: workspace_actor.send(CreateWorkspace{request}) — plain .send(), no timeout wrapper here
    alt Ok(Ok(workspace))
        CW-->>C: 201 WorkspaceResponse::success(workspace)
    else Ok(Err(e)) or mailbox Err
        CW-->>C: 500 WorkspaceResponse::error
    end
    Note over WA: GET /{id} (get_workspace), PUT /{id} (update_workspace), DELETE /{id}<br/>(delete_workspace, soft delete), POST /{id}/favorite (toggle_favorite_workspace),<br/>POST /{id}/archive (archive_workspace) and GET /count (get_workspace_count) each<br/>send ONE matching WorkspaceActor message (GetWorkspace/UpdateWorkspace/DeleteWorkspace/<br/>ToggleFavoriteWorkspace/ArchiveWorkspace/GetWorkspaceCount), only list and get use the<br/>timeout wrapper with its 504 branch (workspace_handler.rs:74, :133), the rest use plain .send()
```

## VC-04.24 `GET /api/pages` — metadata fan-out over the content API
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant GP as get_pages<br/>src/handlers/pages_handler.rs:21
    participant SE as settings_addr<br/>GetSettings
    participant MD as metadata_addr<br/>GetMetadata
    participant CA as content_api<br/>app_state.content_api (per-file fetch)

    C->>GP: GET /api/pages
    GP->>SE: settings_addr.send(GetSettings)
    alt mailbox error or Err
        GP-->>C: 500 Settings actor mailbox error / settings error
    end
    GP->>MD: metadata_addr.send(GetMetadata)
    alt mailbox error or Err
        GP-->>C: 500 Metadata actor mailbox error
    end
    GP->>GP: is_debug_enabled() gates verbose logging only
    par one future per metadata entry
        GP->>CA: get_file_metadata_extended(file_name) per entry, a failure keeps the page with modified 0 — join_all(futures) (pages_handler.rs:55-98)
    end
    GP-->>C: 200 {pages: [PageInfo{id,title,path,parent,modified}, ...]}
    Note over GP: no auth wrap on this route — /api/pages sits under the scope-wide RbacGate only (VC-03.6), a GET that is anonymous<br/>only when RBAC_PUBLIC_READS=1 (src/middleware/rbac_gate.rs:135-142, :172-177) and otherwise needs ReadOnly
```

**Tension:** `get_pages` starts one `get_file_metadata_extended` call per metadata entry and awaits them all with an unbounded `join_all` (`src/handlers/pages_handler.rs:55-98`), with no cache and no concurrency cap. With public reads on, each anonymous `GET /api/pages` fans out one upstream content-API request per page in the corpus. Source tracing only; the cost per call depends on the content API configured. Close it with a bounded `buffer_unordered` and a short-lived cache.

## VC-04.25 `client-logs` (REST) and `client-messages` (WebSocket) — telemetry ingest and outbound stream
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CL as handle_client_logs<br/>src/handlers/client_log_handler.rs:39
    participant TL as telemetry logger<br/>get_telemetry_logger()
    participant WS as websocket_client_messages<br/>src/handlers/client_messages_handler.rs:106
    participant CW as ClientMessagesWs actor<br/>client_messages_handler.rs:10

    C->>CL: POST /api/client-logs {logs:[...], sessionId, timestamp}
    alt logs.len() > MAX_LOG_ENTRIES (1000)
        CL-->>C: 413 Too many log entries
    end
    CL->>CL: client_session_id = X-Session-ID header or payload.sessionId
    CL->>CL: correlation_id = Uuid::parse_str(client_session_id) or new_v4() fallback
    opt telemetry logger present
        CL->>TL: TelemetryEvent::new(...).with_client_session_id(...).with_metadata(log_count,...)
    end
    CL->>CL: append entries to /app/logs/client.log
    CL-->>C: 200 {status:success}
    Note over CL: registered EARLY in the /api scope (src/main.rs:1139) specifically to avoid<br/>scope-registration-order conflicts (VC-01.10) — RBAC-allowlisted (src/middleware/rbac_gate.rs:57)

    C->>WS: GET /ws/client-messages (Upgrade: websocket)
    WS->>WS: token = Authorization Bearer, plus the ?token= query only in debug or dev-auth builds (client_messages_handler.rs:137-151)
    alt token empty or missing
        WS-->>C: 401 {error: Authentication required} — client_ip logged as SECURITY warning (client_messages_handler.rs:166-169)
    else no NostrService configured
        WS-->>C: 401 fail closed (client_messages_handler.rs:173-175)
    else nostr_service.get_session(token) is None
        WS-->>C: 401 "token does not name a live, unexpired session" (client_messages_handler.rs:177-179)
    else live session
        WS->>CW: ws::start(ClientMessagesWs::new(app_state), req, stream) (client_messages_handler.rs:182)
        CW->>C: 101 Switching Protocols, text {type:init, status:connected}
        CW->>CW: start_heartbeat() — ctx.run_interval(30s) ping, 90s no-pong -> ctx.stop()
        CW->>CW: start_message_stream() — ctx.run_interval(100ms) drains app_state.client_message_rx
        loop every 100ms while messages queued
            CW->>C: text {type:client_message, content, timestamp, session_id, agent_id}
        end
        loop client frames
            C->>CW: Ping/Pong/Text/Binary/Close
            alt Binary
                CW->>CW: warn — binary messages not supported on this stream
            end
        end
    end
    Note over WS: RESOLVED (ADR-2044, ADR-2058) — the token is resolved through NostrService::get_session, which enforces AUTH_TOKEN_EXPIRY,<br/>and the query fallback is compiled out of release (comment client_messages_handler.rs:113-135, code :136-180)
    Note over CW: every socket drains the ONE shared app_state.client_message_rx with try_recv (client_messages_handler.rs:35-39),<br/>so with two sockets open each message reaches whichever socket drains first, not every client
```

**Drift (corrected):** this panel used to say the `/ws/client-messages` upgrade checked only that a token was present, so any non-empty string passed. At `86cdd7635` the handler resolves the token with `NostrService::get_session` and fails closed on unknown, expired or absent tokens (`src/handlers/client_messages_handler.rs:166-179`).

**Tension:** `start_message_stream` drains the single process-wide `client_message_rx` with `try_recv` on a 100 ms interval in every connected socket (`src/handlers/client_messages_handler.rs:34-52`). The receivers compete, so a message goes to one socket, chosen by timing, and is never filtered by its `session_id`. Source tracing only. Close it with a broadcast channel or a per-session route.

## VC-04.27 `pay` scope — HTTP 402 balance/debit gate, mounted with the embedded-Pod feature, inert until PAY_ENABLED
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CFG as VcPayConfig::from_env<br/>src/handlers/pay_handler.rs:98 — PAY_ENABLED, PAY_COST_SATS, PAY_LEDGER_DIR
    participant INFO as pay_info_handler<br/>pay_handler.rs:465 GET /pay/.info
    participant BAL as pay_balance_handler<br/>pay_handler.rs:477
    participant RES as pay_resource_handler<br/>pay_handler.rs:530
    participant DEP as pay_deposit_handler<br/>pay_handler.rs:504
    participant ST as FsPaymentStore<br/>get_balance and debit
    Note over BAL,RES: extract_caller_pubkey (pay_handler.rs:404) now calls the shared verifier validate_nip98_token_bound with BodyBinding::NoBody for these GETs (pay_handler.rs:413-418)<br/>so each token is single-use in the process-wide replay cache (src/utils/nip98.rs:164-168, claim :573)<br/>and the debit receipt headers are built with HeaderName::from_bytes, skipping a bad pair instead of panicking (pay_handler.rs:567-575), project commit 5fa0e1325

    Note over CFG: embedded-Pod routes mount at src/main.rs:1105 (VC-01) at the App root, outside the /api RbacGate — inert until<br/>PAY_ENABLED=true, gated handler-by-handler rather than by a scope-level middleware
    C->>INFO: GET /pay/.info (always reachable, no gate)
    INFO-->>C: 200 {enabled, methods:[lightning], cost_tiers} — reports the REAL enabled flag
    C->>BAL: GET /pay/.balance
    alt !config.enabled
        BAL-->>C: 403 Payment system is disabled
    else enabled
        BAL->>BAL: extract_caller_pubkey(req, None) — NIP-98 Authorization header (pay_handler.rs:488)
        alt no pubkey
            BAL-->>C: 401 Authentication required
        else
            BAL->>BAL: did = pubkey_to_did(pubkey)
            BAL->>ST: store.get_balance(&did)
            BAL-->>C: 200 balance_response(did, balance, cost_sats)
        end
    end
    C->>RES: GET /pay/{resource_path}
    alt !config.enabled
        RES-->>C: 403 Payment system is disabled
    else enabled and authenticated
        RES->>RES: cost = config.cost_for_endpoint(#quot;/{resource}#quot) — P2-05 per-endpoint cost tier<br/>(inference=cost*10, image-gen=cost*100, analytics=cost*5, default=cost_sats)
        RES->>ST: store.debit(&did, cost)
        alt Ok(remaining)
            RES-->>C: 200 {resource,charged,balance:remaining,unit:sat} with payment headers (pay_handler.rs:558-577)
        else PaymentError::InsufficientBalance
            RES-->>C: 402 payment_required_body (pay_handler.rs:579-581)
        else other store error
            RES-->>C: 500 Payment store error (pay_handler.rs:583-585)
        end
        Note over RES: resource proxying is a STUB — this confirms and charges but never<br/>forwards to the underlying resource handler (pay_handler.rs:527-529)
    Note over CFG,ST: PROPOSED, NOT LIVE — ADR-2111 (decision_status proposed, implementation_status<br/>none, 2026-09-21) would DELETE FsPaymentStore and the whole /pay/* route set and replace<br/>them with a thin proxy to the agentbox wallet surface, carrying the caller's NIP-98 identity<br/>through, so this repo stops owning a sats ledger. Everything drawn above is the LIVE state<br/>today — docs/adr/ADR-2111-re-sequence-rgb-for-bridged-assets-and-delete-the-host-payment-store.md:38<br/>Re-qualified 2026-10-01: the sidestr-rs parity release does not implement it, FsPaymentStore<br/>and /pay/* stay host-owned and the record stays proposed (ADR-2111-re-sequence-rgb-for-bridged-assets-and-delete-the-host-payment-store.md:83, :95-96)
    end
    C->>DEP: POST /pay/.deposit
    DEP-->>C: 403 when disabled, else 501 Not Implemented — #quot;Contact the server operator for manual funding#quot (:505-516)
    Note over DEP: DIVERGENCE — Lightning deposit is a stub in this commit, spec link only (webledgers.org)
```

## VC-04.28 `quic_transport_handler` — DOC-DRIFT: the QUIC transport was removed under ADR-2066, two Postcard structs survive
```mermaid
flowchart TB
    Q["quic_transport_handler.rs (101 lines) — now holds ONLY PostcardNodeUpdate and<br/>PostcardBatchUpdate plus their BinaryNodeData round-trip impls (quic_transport_handler.rs:22-75)"]
    FIELDS["PostcardNodeUpdate: id,x,y,z,vx,vy,vz,cluster_id,anomaly_score,community_id (:22-36)<br/>PostcardBatchUpdate: frame_id,timestamp_ms,nodes:Vec~PostcardNodeUpdate~ (:70-75)<br/>round-trip impls From/Into BinaryNodeData (:38-67), one #[cfg(test)] unit test (:82-100)"]
    IMP["imported directly by (fastwebsockets_handler.rs:34)<br/>`use super::quic_transport_handler::{PostcardBatchUpdate, PostcardNodeUpdate}`<br/>— never by any HTTP/WS route in this file"]
    MODRS["src/handlers/mod.rs:116-122 — module doc comment records the removal;<br/>NO pub use re-export of anything from this module exists"]
    NONE["NO configure fn, no route, scope, or .service() call anywhere in src/main.rs<br/>registers this module — it never had one even before the removal"]
    Q --> FIELDS --> IMP
    Q --> MODRS --> NONE
    D1["DOC-DRIFT (pre-dates this verification window, commit 35c2448a8) — the previously-diagrammed<br/>QuicTransportServer, QuicServerConfig, QuicClientSession, ControlMessage, TopologyNode,<br/>TopologyEdge, PostcardDeltaUpdate and the encode/decode/calculate_deltas free functions do<br/>NOT exist in this file any more. ADR-2066 removed the whole QUIC/WebTransport server as dead<br/>code — constructed nowhere, routed nowhere, only ever re-exported (never called). Only the two<br/>Postcard wire-format structs survive, kept because fastwebsockets_handler.rs's own<br/>postcard-serialized position broadcasts import them directly."]
    NONE --- D1
```
