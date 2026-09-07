---
id: VC-27
title: Agent estate integration — MCP relay, discovery, monitoring, ingest
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/IDENTIFIER-taxonomy.md
adrs: [ADR-2025, ADR-2058, ADR-2090, ADR-2091, ADR-2094]
sources:
  - ../project/src/services/bots_client.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/utils/mcp_tcp_client.rs
  - ../project/src/client/mod.rs
  - ../project/src/services/mcp_relay_manager.rs
  - ../project/src/handlers/mcp_relay_handler.rs
  - ../project/src/services/multi_mcp_agent_discovery.rs
  - ../project/src/handlers/multi_mcp_websocket_handler.rs
  - ../project/src/actors/multi_mcp_visualization_actor.rs
  - ../project/src/actors/agent_monitor_actor.rs
  - ../project/src/actors/task_orchestrator_actor.rs
  - ../project/src/services/management_api_client.rs
  - ../project/src/app_state.rs
  - ../project/src/services/agent_visualization_protocol.rs
  - ../project/src/services/agent_visualization_processor.rs
  - ../project/src/handlers/bots_visualization_handler.rs
  - ../project/src/handlers/memory_flash_handler.rs
  - ../project/src/agent_events/ingest.rs
  - ../project/src/agent_events/hub.rs
  - ../project/src/agent_events/schema.rs
  - ../project/src/agent_events/provenance.rs
  - ../project/src/services/acsp/client.rs
  - ../project/src/main.rs
verified_commit: dd82a07b0
---

## VC-27.1 BotsClient — legacy `:9500` MCP-TCP poller (superseded path)

```mermaid
sequenceDiagram
    autonumber
    participant Caller as caller<br/>src/services/bots_client.rs:138
    participant BC as BotsClient<br/>src/services/bots_client.rs:113
    participant MCP as McpTcpClient<br/>utils/mcp_tcp_client.rs:24, test_connection() :772,<br/>initialize_session() :785, query_agent_list() :291
    participant GSS as GraphServiceSupervisor<br/>actors/graph_service_supervisor.rs:421

    Caller->>BC: connect(_bots_url) - bots_client.rs:144
    BC->>MCP: test_connection() - :150
    alt server unreachable
        MCP-->>BC: Ok(false) or Err
        BC-->>Caller: Err("MCP server is not reachable") - :164-168
    else reachable
        BC->>MCP: initialize_session() - :155
        BC->>BC: start_polling() - :176
        loop every 2s (tokio interval, :187,189)
            BC->>MCP: query_agent_list() - :192
            alt agents non-empty
                BC->>BC: Agent::from(mcp_agent) map - :197-198
                BC->>BC: agents.write().await = converted - :200-203
                opt graph_service_addr set
                    BC->>GSS: do_send(UpdateBotsGraph{agents}) - :205-214
                end
            else empty list
                BC->>BC: clear stored agents if non-empty - :216-220
            else query_agent_list Err
                BC->>BC: debug log, keep stale snapshot - :223-225
            end
        end
    end
    Note over BC,MCP: RESOLVED ADR-2088 (estate) - get_status() misreported on THREE axes, not one: host<br/>"agentic-workstation", port 9090 and an unconditional connected=true. It now reports<br/>self.mcp_client.host/.port (:242-243, values resolved in BotsClient::new() :118-121) and a real<br/>AtomicBool connection state (:237,241) set from the actual test_connection() outcome (:236-244).<br/>Two tokio tests cover it.
    Note over Caller,MCP: agent_events/ingest.rs:14-19 marks this port-9500 snapshot path<br/>as untouched by design (not legacy/deprecated) - agent_action events use a separate /wss/agent-events ingest (see VC-27.13, RESOLVED ADR-2084)
    Note over MCP: DOC-DRIFT (audit-rust.md correction): a SECOND file also named mcp_tcp_client.rs<br/>exists at src/client/mcp_tcp_client.rs, defining McpTelemetryClient. grep confirms it is dead -<br/>src/client/mod.rs:3 re-exports it but nothing else in src/ constructs or calls it. The live MCP-TCP<br/>hop is exclusively utils/mcp_tcp_client.rs::McpTcpClient shown above (used by bots_client.rs,<br/>ontology_class_index.rs, multi_mcp_agent_discovery.rs)
```

## VC-27.2 McpRelayManager — multi-agent-container lifecycle via docker exec

```mermaid
sequenceDiagram
    autonumber
    participant Caller as ensure_mcp_ready()<br/>src/services/mcp_relay_manager.rs:277
    participant MRM as McpRelayManager<br/>mcp_relay_manager.rs:43
    participant CB as CircuitBreaker<br/>failure_threshold=3, recovery_timeout=30s (:45-53)
    participant Docker as docker exec<br/>multi-agent-container

    Caller->>MRM: check_mcp_container() - :265
    Docker-->>MRM: docker ps -q -f name=multi-agent-container
    alt container absent
        MRM-->>Caller: Err("multi-agent-container is not running") - :279
    else container present
        Caller->>MRM: ensure_relay_running() - :160
        MRM->>MRM: health_manager.check_service_now("mcp-relay") - :161
        MRM->>CB: execute(check_relay_status_internal) - :64-73
        CB->>Docker: exec multi-agent-container pgrep -f mcp-server - mcp_relay_manager.rs:92-94
        alt pgrep succeeds (already running)
            Docker-->>MRM: status.success()=true
            MRM-->>Caller: Ok(()) - :175-177
        else not running
            MRM->>Docker: exec -d multi-agent-container bash -c 'cd /app then npm run mcp:start' - :181-190
            MRM->>MRM: std::thread::sleep(2s) - :197
            MRM->>Docker: re-check pgrep - :199
            alt now running
                MRM-->>Caller: Ok(())
            else still failing
                MRM-->>Caller: Err("MCP relay started but not running") - :202
            end
        end
        Caller->>Caller: tokio::time::sleep(1s) - :286
    end
    par background health loop
        loop every 30s (start_health_monitoring, :244-247)
            MRM->>MRM: health_manager.check_service_now("mcp-relay") - :249
        end
    end
    Note over MRM,CB: INVARIANT: CircuitBreakerConfig failure_rate_threshold=0.5, success_threshold=2,<br/>half_open_max_requests=3, minimum_request_threshold=5 (:45-53)
```

## VC-27.3 MCPRelayActor — `/ws/mcp-relay` session lifecycle (connect/retry/teardown)

```mermaid
sequenceDiagram
    autonumber
    participant Client as WS client
    participant H as mcp_relay_handler()<br/>src/handlers/mcp_relay_handler.rs:442
    participant A as MCPRelayActor<br/>mcp_relay_handler.rs:39, Actor impl :200
    participant O as Orchestrator WS<br/>ORCHESTRATOR_WS_URL default ws://multi-agent-container:3002/ws (:77-78)

    Client->>H: GET /ws/mcp-relay (upgrade) - main.rs:1043
    H->>H: extract Bearer token or ?token= - :452-463
    alt token empty
        H-->>Client: 401 "Authentication required" - :474-476
        Note over H: SECURITY: logged but not yet enforced on all clients (:447-450)
    else token present
        H->>A: ws::start(MCPRelayActor::new()) - :479
        A->>A: started() - register health endpoint, run_interval 30s ping+check (:209-233)
        A->>A: run_interval 60s circuit-breaker stats log (:235-251)
        A->>O: connect_to_orchestrator() - circuit_breaker.execute(connect_async, timeout) (:94-113)
        alt connect ok
            O-->>A: ws_stream split into tx/rx - :121-125
            A->>A: do_send(SetOrchestratorTx(tx)) - :125
            loop forward orchestrator->client (rx.next())
                O-->>A: Text/Binary/Ping/Close - :133-173
                A->>Client: ctx.text(msg) or ctx.binary(msg) - :279,289
            end
        else connect fails or times out
            A->>A: retry_delay = min(5s * 2^(attempts-1), 60s) - :186-189
            A->>A: sleep(retry_delay) then do_send("retry") - :192-193
            A->>A: connect_to_orchestrator() again - :276-277
        end
    end
    Client->>A: ws::Message::Text (JSON)
    alt type == "ping"
        A-->>Client: {"type":"pong", timestamp} - :320-328
    else forward to orchestrator
        alt orchestrator_tx set and healthy
            A->>O: send Text/Binary, 5s timeout (:355-372,404-421)
        else unhealthy or absent
            A-->>Client: {"type":"error", message} - :337-346,375-384
        end
    end
    Client->>A: ws::Message::Close
    A->>A: ctx.stop() - :427
    A->>A: stopped() logs - :256-258
```

## VC-27.4 MultiMcpAgentDiscovery — per-server agent + tool discovery

```mermaid
sequenceDiagram
    autonumber
    participant Caller as start_discovery()<br/>src/services/multi_mcp_agent_discovery.rs:207
    participant D as MultiMcpAgentDiscovery<br/>multi_mcp_agent_discovery.rs:62
    participant CF as claude-flow server<br/>host=CLAUDE_FLOW_HOST port=MCP_TCP_PORT default 9500 (:129-133)
    participant RS as ruv-swarm server<br/>host=RUV_SWARM_HOST port=RUV_SWARM_PORT default 9501 (:146-150)
    participant DAA as daa server<br/>host=DAA_HOST port=DAA_PORT default 9502 (:163-167)

    Caller->>D: initialize_default_servers() - :121-179
    D->>D: insert claude-flow/ruv-swarm/daa McpServerConfig - :124-173
    Caller->>D: start_discovery() - :207
    loop while discovery_running (tokio::spawn, :225-304)
        par concurrent per enabled server (:235-295)
            D->>CF: discover_server_agents -> discover_claude_flow_agents - :335-336,346-407
            CF-->>D: query_server_info / query_agent_list / query_swarm_status - :376,397,420
            D->>RS: discover_ruv_swarm_agents - :337,434-526
            RS-->>D: same three-call pattern, server_type=RuvSwarm - :464,490,513
            D->>DAA: discover_daa_agents - :338,527-609
            DAA-->>D: same pattern, server_type=Daa - :557,582,602
        end
        alt discover_server_agents Ok
            D->>D: insert server_info/agents/topology, successful_discoveries+=1 - :246-271
        else Err (connect/timeout)
            D->>D: failed_discoveries+=1, is_connected=false - :278-291
        end
        D->>D: tokio::time::sleep(sleep_ms) - :303
    end
    Note over D,CF: RESOLVED ADR-2083 (estate) - WIRED, not removed: the per-server values are deliberate, so the<br/>loop now sleeps the minimum interval across ENABLED servers (select_discovery_interval_ms :232), with a named<br/>MIN_DISCOVERY_INTERVAL_MS floor so a misconfigured 0 cannot spin it, and a named fallback when no server is<br/>enabled. The flat 1000ms sleep is gone (:299-303).
    Note over CF,DAA: supported_tools fallback differs per server when query_server_info fails:<br/>claude-flow=[agent_list,swarm_status,server_info] (multi_mcp_agent_discovery.rs:387-391),<br/>ruv-swarm and daa each have their own equivalent fallback list in their own discover_*_agents function
```

## VC-27.5 MultiMcpVisualizationWs — `/multi-mcp/ws` session and opcodes

```mermaid
sequenceDiagram
    autonumber
    participant Client as WS client
    participant H as multi_mcp_visualization_ws()<br/>src/handlers/multi_mcp_websocket_handler.rs:823
    participant Ws as MultiMcpVisualizationWs<br/>multi_mcp_websocket_handler.rs:473 (Actor::started)

    Client->>H: GET /multi-mcp/ws (upgrade) - configure_multi_mcp_routes :911-921
    H->>H: require Bearer/token or 401 - :831-906
    H->>Ws: ws::start(MultiMcpVisualizationWs::new) - :908
    Ws->>Ws: started() - start_heartbeat, register MONITORED_SERVICES health endpoints,<br/>start_health_monitor (ADR-2094), start_position_updates - :473-500
    Ws->>Ws: run_interval 60s recovery-if-idle-300s + circuit stats log - :502-524
    Ws->>Ws: send_discovery_data(ctx) - :526
    loop position updates (PerformanceMode: HighFreq=16ms/Normal=100ms/Low=1000ms, OnDemand=none) - :127-135,176-178
        Ws->>Ws: do_send(RequestAgentUpdate)
    end
    loop heartbeat every 5s (start_heartbeat) - :187-200
        alt no pong for >30s
            Ws-->>Client: ctx.close(None) - :194
        else
            Ws-->>Client: ctx.ping(b"ping") - :198
        end
    end
    Client->>Ws: Text "ping" (plain)
    Ws-->>Client: "pong" - :546-549
    Client->>Ws: Text JSON {action, data} - ClientRequest :677-680
    alt action == configure
        Ws->>Ws: handle_client_config(ClientConfig{subscription_filters,performance_mode}) - :367-389
    else action == request_discovery
        Ws->>Ws: handle_discovery_request(ctx) - :391-404
    else action == request_agents
        Ws->>Ws: do_send(RequestAgentUpdate), degrade under open circuit breaker - :567-603
    else action == request_performance
        alt has_healthy_services() true
            Ws->>Ws: do_send(RequestPerformanceUpdate) - :619
        else degraded
            Ws-->>Client: cached "performance_data" status=degraded - :606-617
        end
    else action == request_topology
        Ws->>Ws: do_send(RequestTopologyUpdate{swarm_id}) - :622-632
    else unknown action
        Ws-->>Client: send_error_response("Unknown action: ...") - :633-639
    end
    Client->>Ws: ws::Message::Close
    Ws->>Ws: log final circuit-breaker stats, ctx.close(reason) - :646-663
    Note over Ws: RESOLVED ADR-2094 (2026-09-05): has_healthy_services (:253) is a pure atomic read of a cached verdict<br/>one monitor task started at connection init publishes it and stops when the client drops (start_health_monitor :209) - no per-call spawn
    Note over H,Ws: RESOLVED ADR-2091 (supersedes an earlier DOC-DRIFT finding that they merely returned<br/>fiction): GET /multi-mcp/status and POST /multi-mcp/refresh are DELETED, not just hardcoded -<br/>configure_multi_mcp_routes only registers /ws now (:911-921). /status served a hardcoded<br/>two-server JSON literal (claude-flow is_connected:true agent_count:4, never queried) and<br/>/refresh echoed "initiated" while never calling MultiMcpAgentDiscovery::start_discovery - both<br/>took an unused _app_state. Real discovery state lives in multi_mcp_agent_discovery.rs (see VC-27.4).
```

## VC-27.6 MultiMcpVisualizationActor — message set and periodic ticks

```mermaid
classDiagram
    class MultiMcpVisualizationActor {
        +HashMap(String,McpServerConfig) mcp_servers
        +HashMap(String,Position) agent_positions
        +HashMap(String,AgentInit) agents
        +HashMap(String,ConnectionInit) connections
        +HashMap(String,McpServerMetrics) server_metrics
        +LayoutAlgorithm layout_algorithm
        +Duration update_interval = 33ms
        +Vec~Recipient~ subscribers
        +SwarmTopologyData topology_data
        +GlobalPerformanceMetrics global_metrics
    }
    class MultiMcpVisualizationMessage {
        <<enum, rtype Result-unit-String>>
        Initialize servers,layout,physics,visual
        UpdateAgentPositions server_id,positions,timestamp
        AddAgent server_id,agent,position
        RemoveAgent server_id,agent_id
        UpdateAgentStatus server_id,agent_id,status,metadata
        AddConnection connection
        RemoveConnection connection_id
        UpdateServerMetrics server_id,metrics
        Subscribe recipient
        Unsubscribe recipient
        ChangeLayout algorithm
        AnalyzeTopology
        GetVisualizationState
        Reset
    }
    class MultiMcpVisualizationResponse {
        <<enum, rtype unit>>
        VisualizationState agents,positions,connections,servers,metrics,topology,global_metrics
        TopologyAnalysis topology_data,recommendations
        PerformanceMetrics global_metrics,server_metrics
    }
    class LayoutAlgorithm {
        <<enum, default ForceDirected>>
        ForceDirected attraction_strength,repulsion_strength,damping_factor
        Hierarchical server_separation,layer_height,node_spacing
        Circular radius_base,radius_increment,angular_spacing
        Grid grid_spacing,cluster_size,padding
    }
    MultiMcpVisualizationActor ..> MultiMcpVisualizationMessage : Handler impl at line 264
    MultiMcpVisualizationActor ..> MultiMcpVisualizationResponse : constructs
    MultiMcpVisualizationActor --> LayoutAlgorithm : layout_algorithm field
    note "run_interval 33ms update_visualization (245-247), 10s analyze_topology (249-251), 5s collect_global_metrics (253-255), all from started() at 242-256"
    note "RESOLVED ADR-2089 (estate): Subscribe and Unsubscribe both had zero senders, so the whole broadcast path was dead. 188 lines removed."
```

## VC-27.7 AgentMonitorActor — Management API poll loop and debounce

```mermaid
sequenceDiagram
    autonumber
    participant Sup as AppState / supervisor
    participant AM as AgentMonitorActor<br/>src/actors/agent_monitor_actor.rs:169-203 (struct), new() :247-314
    participant MAC as ManagementApiClient<br/>host=MANAGEMENT_API_HOST port=MANAGEMENT_API_PORT default 9090 (:253-258)
    participant GSS as GraphServiceSupervisor

    Sup->>AM: started() - is_connected=true, do_send(InitializeActor) - :434-441
    AM->>AM: handle(InitializeActor) - run_later(100ms) poll_agent_statuses + schedule_next_poll - :448-463
    loop self-rescheduling poll (schedule_next_poll, :415-421)
        AM->>MAC: tokio::join!(list_tasks(), get_system_status()) - :333-334
        alt tasks_result Ok
            AM->>AM: task_to_agent_status per active task - :376-380
            AM->>AM: do_send(ProcessAgentStatuses{agents,telemetry}) - :382
        else tasks_result Err
            AM->>AM: do_send(RecordPollFailure) - :386
        end
        AM->>AM: next_poll_delay() - base 15s (:304), doubles per consecutive_poll_failures (max shift 5), capped 90s - :404-411
    end
    AM->>AM: handle(ProcessAgentStatuses) - :568
    opt agents empty and MOCK_AGENTS=true/1
        AM->>AM: build_mock_swarm_agents() 5 mock agents - :466-567
    end
    AM->>AM: golden-angle spiral position per agent, poll_offset round-robin (ADR-031 item 1) - :596-634
    AM->>AM: decide_bots_graph_emit(count, last_nonempty, consecutive_empty) - call :640-644, fn body :126-167
    alt roster non-empty (fn :131-137)
        AM->>GSS: do_send(UpdateBotsGraph{agents}) - :648-655
    else roster empty and consecutive_empty < EMPTY_CONFIRM_THRESHOLD=2 (fn :140-148,159-165)
        AM->>AM: suppress emit - debounce a transient blip - :656-662
    else roster empty and confirmed (2nd consecutive empty) (fn :150-158)
        AM->>GSS: do_send(UpdateBotsGraph{agents: []}) - clears once - :648-655
    end
    Sup->>AM: TaskStatusChanged{agent_type,running_task_count} (from TaskOrchestratorActor, ADR-031 item 3) - handler :717-728
    AM->>AM: poll_agent_statuses(ctx) immediate re-poll - :726
    Note over AM,MAC: INVARIANT: idle cadence is 15s (not 3s) to share agentbox's per-key rate-limit<br/>bucket with task creation - backoff cap 90s exceeds agentbox's 60s continueExceeding window (:304,394-411)
    Note over AM,GSS: DIVERGENCE (roster-clobber fix): an empty Management API poll is "no information"<br/>not "all agents died" - only a confirmed 2nd consecutive empty poll clears the graph (:197-201,126-167)
```

## VC-27.8 TaskOrchestratorActor — CreateTask/Interrupt/Drain message handlers

```mermaid
sequenceDiagram
    autonumber
    participant H as VisionClaw API handler
    participant TO as TaskOrchestratorActor<br/>src/actors/task_orchestrator_actor.rs:47, new() :66
    participant MAC as ManagementApiClient<br/>services/management_api_client.rs
    participant AM as AgentMonitorActor (agent_monitor_addr)

    TO->>TO: started() - do_send(InitializeActor) - :136-141
    TO->>TO: handle(InitializeActor) - run_interval 300s cache cleanup (Completed/Failed >5min old) - :158-177

    H->>TO: CreateTask{agent,task,provider,claude_flow_agent_id} - :185-198
    alt accepting_tasks == false (draining, ADR-031 item 7)
        TO-->>H: Err("Task creation rejected: actor is draining") - :301-306
    else running_count >= max_concurrent_tasks (MAX_CONCURRENT_TASKS env, default 20)
        TO-->>H: Err("At capacity: N/max tasks running") - ADR-031 item 2 (:308-330)
    else capacity available
        TO->>MAC: create_task_with_retry - max_retries=3, retry_delay=2s*attempt (:75-76,90-131)
        loop up to 3 attempts
            MAC-->>TO: Err -> sleep(retry_delay * attempts), retry - :113-127
        end
        alt final attempt Ok
            TO->>TO: active_tasks.insert(task_id, TaskState{status:Running,...}) - :367-380
            opt agent_monitor_addr set
                TO->>AM: do_send(TaskStatusChanged{agent_type,running_task_count}) - ADR-031 item 3 (:390-395)
            end
            TO-->>H: Ok(TaskResponse)
        else exhausted retries
            TO-->>H: Err(e.to_string()) - :359,399
        end
    end

    H->>TO: InterruptAgentTask{id} - :219-236
    alt id is a known local task_id
        TO->>TO: resolved = id (fast path, no round-trip) - :453,457-458
    else id not local
        TO->>MAC: list_tasks() - :464
        alt task_id match or claude_flow_agent_id match found
            TO->>TO: resolved = matched task_id - :479-487
        else no match
            TO-->>H: Err(InterruptError::Unresolved) - :488-490
        end
    end
    TO->>MAC: stop_task(resolved) - :493-497
    TO-->>H: Ok(resolved) or Err(InterruptError::Stop)

    H->>TO: DrainTasksBeforeShutdown{timeout_secs} - :621-626
    TO->>TO: accepting_tasks = false - :642
    loop every 1s until deadline (:646-663)
        alt running == 0
            TO->>TO: ctx.stop() - all tasks drained - :654-655
        else deadline exceeded
            TO->>TO: ctx.stop() - drain timeout, remaining tasks abandoned - :657-661
        end
    end
    Note over TO: DIVERGENCE: InterruptAgentTask deliberately never matches the role-label `agent`<br/>field ("coder"/"researcher") - only task_id or claude_flow_agent_id, to avoid stopping the wrong task (:474-478)
    Note over TO,AM: INVARIANT (ADR-031 item 3): every CreateTask success pushes TaskStatusChanged so<br/>AgentMonitorActor re-polls immediately rather than waiting its 15s idle cadence (see VC-27.7)
```

## VC-27.9 ManagementApiClient — agentbox management-api HTTP calls

```mermaid
sequenceDiagram
    autonumber
    participant Boot as AppState::new<br/>src/app_state.rs:1252-1262
    participant MAC as ManagementApiClient<br/>src/services/management_api_client.rs:27, new() :180-199
    participant API as agentbox management-api<br/>base_url = http://MANAGEMENT_API_HOST:MANAGEMENT_API_PORT (default agentic-workstation:9090)

    Boot->>Boot: validate_security_env_vars() - :82-172
    alt MANAGEMENT_API_KEY unset, insecure-default-listed, or <16 chars
        Boot->>Boot: log SECURITY CONFIGURATION ERROR, panic on Err - :140-162
    else key valid
        Boot->>MAC: ManagementApiClient::new(host, port, mgmt_api_key) - :1262, client.rs:180
        MAC->>MAC: reqwest Client::builder().timeout(30s).connect_timeout(10s) - :183-187
    end

    MAC->>API: POST /v1/tasks (create_task_with_context) - Authorization: Bearer api_key - :244-291
    alt status 202/200
        API-->>MAC: TaskResponse{task_id,...} - :295-305
    else other status
        MAC-->>MAC: Err(ApiError(text, status)) - :306-312
    else transport failure
        MAC-->>MAC: Err(NetworkError) - :291
    end

    MAC->>API: GET /v1/tasks/{task_id} (get_task_status) - :315-343
    MAC->>API: GET /v1/tasks (list_tasks) - :345-373
    MAC->>API: DELETE /v1/tasks/{task_id} (stop_task) - :375-400
    MAC->>API: GET /v1/status (get_system_status) - :402-430
    MAC->>API: POST /v1/briefs (create_brief) - :433-482
    MAC->>API: POST /v1/briefs/{id}/execute (execute_brief) - :489-533
    MAC->>API: POST /v1/briefs/{id}/debrief (create_debrief) - :539-580
    MAC->>API: GET /health (health_check, no auth header) - :586-597
    Note over MAC,API: every call above shares the same alt: 200/2xx Ok(json) else Err(ApiError(body,status)),<br/>and Err(NetworkError) on transport failure (repeated at each call site, e.g. :328-342,388-399)
    Note over Boot,MAC: RESOLVED ADR-2094 (2026-09-05): AgentMonitorActor::new calls the same validate_security_env_vars AppState uses (app_state.rs:82)<br/>a missing or weak MANAGEMENT_API_KEY is a boot error and the client is an Option, never an empty-string key (agent_monitor_actor.rs:235-244,264-292)
```

## VC-27.10 agent_visualization_protocol — outbound wire message envelope

```mermaid
classDiagram
    class AgentVisualizationMessage {
        <<enum>>
        Initialize InitializeMessage
        PositionUpdate PositionUpdateMessage
        StateUpdate StateUpdateMessage
        ConnectionUpdate ConnectionUpdateMessage
        MetricsUpdate MetricsUpdateMessage
    }
    class InitializeMessage {
        +i64 timestamp
        +String swarm_id
        +Option~String~ session_uuid
        +String topology
        +List~AgentInit~ agents
        +List~ConnectionInit~ connections
        +VisualConfig visual_config
        +PhysicsConfig physics_config
        +HashMap_String_Position positions
    }
    class PositionUpdateMessage {
        +i64 timestamp
        +List~PositionUpdate~ positions
    }
    class StateUpdateMessage {
        +i64 timestamp
        +List~AgentStateUpdate~ updates
    }
    class ConnectionUpdateMessage {
        +i64 timestamp
        +List~ConnectionInit~ added
        +List~String~ removed
        +List~ConnectionStateUpdate~ updated
    }
    class MetricsUpdateMessage {
        +i64 timestamp
        +SwarmMetrics overall
        +List~AgentMetrics~ agent_metrics
    }
    AgentVisualizationMessage --> InitializeMessage : serde rename init
    AgentVisualizationMessage --> PositionUpdateMessage : serde rename positions
    AgentVisualizationMessage --> StateUpdateMessage : serde rename state
    AgentVisualizationMessage --> ConnectionUpdateMessage : serde rename connections
    AgentVisualizationMessage --> MetricsUpdateMessage : serde rename metrics
    note "top-level enum is serde(tag = type),<br/>internally tagged (protocol.rs 6-23)"
    note "AgentInit (44-69): id,name,agent_type,<br/>status,color,shape,size,health,cpu,memory,<br/>activity,tasks_active,tasks_completed,<br/>success_rate,tokens,token_rate,<br/>capabilities List~String~,created_at i64"
    note "PositionUpdate (89-98): id,x,y,z f32<br/>plus vx,vy,vz Option~f32~"
    note "ConnectionInit (72-80): id,source,target,<br/>strength,flow_rate,color,active bool"
    note "AgentStateUpdate (107-123): id plus<br/>status,health,cpu,memory,activity,<br/>tasks_active,current_task all Option~T~<br/>- a partial differential update"
```

## VC-27.11 AgentVisualizationProcessor — `/api/visualization/agents/ws` init/refresh

```mermaid
sequenceDiagram
    autonumber
    participant Client as WS client
    participant H as agent_visualization_ws()<br/>src/handlers/bots_visualization_handler.rs:214
    participant Ws as AgentVisualizationWs<br/>bots_visualization_handler.rs:17, Actor :93
    participant Proto as AgentVisualizationProtocol<br/>services/agent_visualization_protocol.rs:434, new() :457
    participant Proc as AgentVisualizationProcessor<br/>services/agent_visualization_processor.rs:162, new() :168

    Client->>H: GET /api/visualization/agents/ws - configure_routes :515-531
    H->>Ws: ws::start(AgentVisualizationWs::new) - :219
    Ws->>Ws: started() - do_send(InitConnection), start_heartbeat, start_position_updates - :96-104
    Ws->>Ws: handle(InitConnection) -> send_init_state(ctx) - :132-134
    Ws->>Proto: create_init_message("swarm-001","hierarchical", agents=Vec::new()) - :58-59
    Proto->>Proc: create_visualization_packet(agents, swarm_id, topology) - agent_visualization_protocol.rs:543 calling agent_visualization_processor.rs:308
    Proc->>Proc: process_agents() - color/shape/animation via get_visual_properties, spherical fallback position, glow_intensity - :175-257
    Proc->>Proc: create_connections(), create_clusters() - :365-389,391-420
    Proc-->>Proto: AgentVisualizationData{swarm,agents,connections,physics_config,...}
    Proto-->>Ws: init_json (AgentInit list mapped from VisualizedAgent) - agent_visualization_protocol.rs:549-585
    Ws-->>Client: ctx.text(init_json) - :62
    loop position updates every 16ms (bots_visualization_handler.rs:69-78)
        Ws->>Proto: create_position_update() - :74
        opt buffered updates present
            Ws-->>Client: ctx.text(update_json)
        end
    end
    loop heartbeat every 5s (bots_visualization_handler.rs:80-90)
        alt no pong for >10s
            Ws-->>Client: ctx.stop() - :82-85
        else
            Ws-->>Client: ctx.ping(b"ping") - :88
        end
    end
    Client->>Ws: Text {action} - :164-193
    alt action == refresh
        Ws->>Ws: send_init_state(ctx) again - :177-179
    else action == pause_updates or resume_updates
        Ws->>Ws: sets self.paused (ADR-2066 addendum), then debug log - :180-183,184-187
    else unknown action
        Ws->>Ws: warn "Unknown client action" - :188-190
    end
    Note over Ws,Proto: PROPOSED ADR-2066 addendum: send_init_state still reports an empty roster, now explicit rather than<br/>disguised - the fake get_real_agent_data() helper is deleted. A real source exists (bots_client.get_agents_snapshot,<br/>bots_client.rs:231) but Agent lacks the profile, task counts, success_rate and timestamp AgentStatus requires, so the<br/>mapping needs a decided contract rather than invented defaults.
    Note over Ws: RESOLVED ADR-2066 addendum: the actor now carries a paused flag - pause_updates and resume_updates<br/>set it and the 16ms run_interval returns early while it is set, so the opcodes do what they advertise.
```

## VC-27.12 memory_flash_handler — `/api/memory-flash` RuVector access broadcast

```mermaid
sequenceDiagram
    autonumber
    participant Caller as RuVector-aware caller
    participant H as handle_memory_flash()<br/>src/handlers/memory_flash_handler.rs:41
    participant HB as handle_memory_flash_batch()<br/>memory_flash_handler.rs:103
    participant CC as ClientCoordinatorActor
    participant Ws as all connected WS clients

    Caller->>H: POST /api/memory-flash {key,namespace,action} - MemoryFlashRequest :16-23
    H->>H: build MemoryFlashBroadcast{type=memory_flash, data{key,namespace,action,timestamp}} - :44-59
    H->>CC: send(BroadcastMessage{message: json}) - :63-65
    alt actor Ok(Ok(()))
        H-->>Caller: 200 {ok:true} - :72
    else actor Ok(Err(e))
        H-->>Caller: 200 {ok:true, warn:e} - :74-76
    else mailbox Err(e)
        H-->>Caller: 500 {ok:false, error} - :78-83
    else serialization Err
        H-->>Caller: 500 {ok:false, error: serialization failed} - :87-93
    end
    CC->>Ws: fan out memory_flash JSON to every registered client

    Caller->>HB: POST /api/memory-flash/batch {events:[...]} - MemoryFlashBatchRequest :99-101
    loop each event in body.events (:113-127)
        HB->>HB: build MemoryFlashBroadcast per event, shared timestamp - :107-122
        HB->>CC: do_send(BroadcastMessage{message: json}) - :124
    end
    HB-->>Caller: 200 {ok:true, count} - :130
    Note over H,CC: routes mounted at /api/memory-flash and /api/memory-flash/batch via<br/>configure_routes (:133-139), configured inside the /api scope (main.rs:1171)
```

## VC-27.13 `/wss/agent-events` ingest — schema validation, hub fan-out, provenance

```mermaid
sequenceDiagram
    autonumber
    participant AB as agentbox management-api<br/>Note: see ES-02
    participant H as agent_events_ws()<br/>src/agent_events/ingest.rs:297-318
    participant Ws as AgentEventsIngestWs<br/>ingest.rs:155, Actor :193
    participant Sch as AgentActionNotification<br/>agent_events/schema.rs
    participant Prov as provenance::record<br/>agent_events/provenance.rs:96
    participant Hub as agent_events::hub<br/>agent_events/hub.rs:22 (HUB_CAPACITY=256)

    rect rgb(240,230,255)
    Note over AB,H: TRUST BOUNDARY - server-to-server ingest (agentbox pushes notifications/agent_action)
    AB->>H: GET /wss/agent-events (upgrade), subprotocol vc-agent-events.v1 - ingest.rs:54,297-318
    H->>H: authenticate() - Bearer or ?token= via NostrService::get_session - :257-294
    alt token valid
        H->>H: session_pubkey = Some(user.pubkey) - :275
    else token invalid and ALLOW_INSECURE_DEFAULTS unset
        H-->>AB: 401 Invalid or expired authentication token - :283-285
    else token invalid and ALLOW_INSECURE_DEFAULTS set (debug/dev-auth builds only)
        H-->>H: warn, accept unauthenticated (session_pubkey=None) - :276-282
    else no token and ALLOW_INSECURE_DEFAULTS set
        H-->>H: warn, accept unauthenticated - :287-290
    else no token, insecure defaults disallowed
        H-->>AB: 401 Authentication required - :291-293
    end
    end
    H->>Ws: WsResponseBuilder::new(AgentEventsIngestWs, harness).start() - :311-317
    AB->>Ws: Text frame (JSON-RPC notifications/agent_action) - :207
    Ws->>Sch: process_frame(text) - serde_json::from_str + is_canonical() - :100-102
    alt parse fails (Err(_))
        Ws-->>AB: {"error":"malformed_json"} - :229-232
    else parses but not canonical (Ok(_), wrong method or version<3)
        Ws-->>AB: {"error":"non_canonical_envelope"} - :225-228
    else canonical - process_frame itself calls provenance+hub before returning IngestOutcome::Published
        Ws->>Prov: provenance::record(&event) - classify() + record_crossings() - ingest.rs:113, provenance.rs:96-103
        Prov->>Prov: classify by pubkey: 64-hex Attributed, malformed hex Malformed, absent Anonymous - provenance.rs:66-72
        Prov->>Prov: record_crossings: cross_from_agentbox(source_urn), cross_from_agentbox(target_urn) - ADR-2025 closed kind-map (uri.rs), provenance.rs:90-93
        Ws->>Hub: hub::publish(event) - broadcast::Sender, drops oldest under backpressure - ingest.rs:138, hub.rs:32-34
        opt event.has_ctc() true (typed CTC field populated)
            Ws->>Ws: fire_ctc_canary() - one-shot CANARY-VC-REC3-CTC via LivenessHarness - ingest.rs:173-190,221-223
        end
        Ws-->>AB: no ack frame - debug log only, always published regardless of provenance status - :216-220
    end
    Ws->>Ws: Ping/Pong/Close handled - pong echo, ctx.close+stop - :239-246
    Note over Ws,Hub: RESOLVED ADR-2084: ingest.rs module doc (ingest.rs:14-19) once carried stale framing about<br/>this socket relationship to the legacy port-9500 bots_client snapshot path (VC-27.1) - the doc now correctly<br/>states that path is untouched by design (state snapshots, polled every 2s) while this socket carries the<br/>disjoint agent_action payload - no replacement for the port-9500 path exists yet
    Note over Prov: SECURITY: ProvenanceStatus::Attributed means a well-formed pubkey was asserted,<br/>NOT that a signature was verified - the wire carries no sig field (provenance.rs:29-38)
```
