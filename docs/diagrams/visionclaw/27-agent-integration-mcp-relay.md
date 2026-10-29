---
id: VC-27
title: Agent estate integration — MCP relay, discovery, monitoring, ingest
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/IDENTIFIER-taxonomy.md
adrs: [ADR-2025, ADR-2058, ADR-2090, ADR-2091, ADR-2094, ADR-2142]
sources:
  - ../project/src/services/bots_client.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/utils/mcp_tcp_client.rs
  - ../project/src/client/mod.rs
  - ../project/src/client/mcp_tcp_client.rs
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
  - ../project/src/agent_events/ingest.rs
  - ../project/src/agent_events/hub.rs
  - ../project/src/agent_events/schema.rs
  - ../project/src/agent_events/provenance.rs
  - ../project/src/actors/agent_demo_director.rs
  - ../project/src/handlers/api_handler/agents_demo.rs
  - ../project/src/main.rs
verified_commit: {visionclaw: 19417bdff5fc22053a8a0cc185797b0311521d10}
---

## VC-27.1 BotsClient — legacy `:9500` MCP-TCP poller (superseded path)

```mermaid
sequenceDiagram
    autonumber
    participant Caller as caller<br/>src/services/bots_client.rs:138
    participant BC as BotsClient<br/>src/services/bots_client.rs:113
    participant MCP as McpTcpClient<br/>utils/mcp_tcp_client.rs:24, test_connection() :778,<br/>initialize_session() :791, query_agent_list() :293
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
    Note over MCP: DOC-DRIFT (audit-rust.md correction, gap #2): a SECOND file also named mcp_tcp_client.rs<br/>exists at src/client/mcp_tcp_client.rs, defining McpTelemetryClient (struct src/client/mcp_tcp_client.rs:10,<br/>impl :17). grep confirms it is dead - src/client/mod.rs:1,3 declares and re-exports it but nothing else in<br/>src/ constructs or calls it. The live MCP-TCP hop is exclusively utils/mcp_tcp_client.rs::McpTcpClient<br/>shown above (used by bots_client.rs, ontology_class_index.rs, multi_mcp_agent_discovery.rs)
```

## VC-27.2 McpRelayManager — multi-agent-container lifecycle via docker exec

```mermaid
sequenceDiagram
    autonumber
    participant Caller as ensure_mcp_ready()<br/>src/services/mcp_relay_manager.rs:280
    participant MRM as McpRelayManager<br/>mcp_relay_manager.rs:41
    participant CB as CircuitBreaker<br/>failure_threshold=3, recovery_timeout=30s (:43-51)
    participant Docker as docker exec<br/>multi-agent-container

    Caller->>MRM: check_mcp_container() - :262
    Docker-->>MRM: docker ps -q -f name=multi-agent-container
    alt container absent
        MRM-->>Caller: Err("multi-agent-container is not running") - :282
    else container present
        Caller->>MRM: ensure_relay_running() - :157
        MRM->>MRM: health_manager.check_service_now("mcp-relay") - :158
        MRM->>CB: execute(check_relay_status_internal) - :61-70
        CB->>Docker: exec multi-agent-container pgrep -f mcp-server - mcp_relay_manager.rs:89-91
        alt pgrep succeeds (already running)
            Docker-->>MRM: status.success()=true
            MRM-->>Caller: Ok(()) - :172-174
        else not running
            MRM->>Docker: exec -d multi-agent-container bash -c 'cd /app then npm run mcp:start' - :178-187
            MRM->>MRM: std::thread::sleep(2s) - :194
            MRM->>Docker: re-check pgrep - :196
            alt now running
                MRM-->>Caller: Ok(())
            else still failing
                MRM-->>Caller: Err("MCP relay started but not running") - :199
            end
        end
        Caller->>Caller: tokio::time::sleep(1s) - :289
    end
    par background health loop
        loop every 30s (start_health_monitoring, :241-244)
            MRM->>MRM: health_manager.check_service_now("mcp-relay") - :246
        end
    end
    Note over MRM,CB: INVARIANT: CircuitBreakerConfig failure_rate_threshold=0.5, success_threshold=2,<br/>half_open_max_requests=3, minimum_request_threshold=5 (:43-51)
```

## VC-27.3 MCPRelayActor — `/ws/mcp-relay` session lifecycle (connect/retry/teardown)

```mermaid
sequenceDiagram
    autonumber
    participant Client as WS client
    participant H as mcp_relay_handler()<br/>src/handlers/mcp_relay_handler.rs:438
    participant A as MCPRelayActor<br/>mcp_relay_handler.rs:39, Actor impl :200
    participant O as Orchestrator WS<br/>ORCHESTRATOR_WS_URL default ws://multi-agent-container:3002/ws (:77-78)

    Client->>H: GET /ws/mcp-relay (upgrade) - main.rs:1115
    H->>H: header_token = Authorization Bearer - :475-480 (the ?token= carrier is dev-only :482-494, a release build drops it :496-505)
    alt no token
        H-->>Client: 401 "Authentication required" via the unauthorised closure - :459-467, :507-510
    else NostrService absent or get_session finds no live session
        H-->>Client: 401 fail-closed - :514-516, :518-520
        Note over H: RESOLVED ADR-2090 - the old note here said the token check was logged but not enforced.<br/>The upgrade now resolves the token through NostrService::get_session before ws::start (:445-452), see VC-05.12
    else token names a live session
        H->>A: ws::start(MCPRelayActor::new()) - :523
        A->>A: started() - register the orchestrator health endpoint :209-219, run_interval 30s ping+check (:221-232)
        A->>A: run_interval 60s circuit-breaker stats log (:234-250)
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
            A->>A: connect_to_orchestrator() again - :275-276
        end
    end
    Client->>A: ws::Message::Text (JSON)
    alt type == "ping"
        A-->>Client: {"type":"pong", timestamp} - :320-326
    else forward to orchestrator
        alt orchestrator_tx set and healthy
            A->>O: send Text (:350-370) or Binary (:399-419), 5s timeout
        else unhealthy or absent
            A-->>Client: {"type":"error", message} - :333-342,373-380
        end
    end
    Client->>A: ws::Message::Close
    A->>A: ctx.stop() - :424
    A->>A: stopped() logs - :255-257
```

## VC-27.4 MultiMcpAgentDiscovery — per-server agent + tool discovery, polled only while a socket is attached

```mermaid
sequenceDiagram
    autonumber
    participant Sync as sync_running<br/>src/services/multi_mcp_agent_discovery.rs:184
    participant D as MultiMcpAgentDiscovery<br/>multi_mcp_agent_discovery.rs:101
    participant CF as claude-flow server<br/>host=claude_flow_host(): CLAUDE_FLOW_HOST, then MCP_HOST, then multi-agent-container (:144-148, :220) port=MCP_TCP_PORT default 9500 (:221-224)
    participant RS as ruv-swarm server<br/>host=RUV_SWARM_HOST port=RUV_SWARM_PORT default 9501 (:237-241)
    participant DAA as daa server<br/>host=DAA_HOST port=DAA_PORT default 9502 (:254-258)

    Note over Sync,D: DOC-DRIFT (RESOLVED 2026-10-12, project 6a1a74b8b/4f4345503) - this diagram used to show start_discovery() driven by an unnamed Caller, because nothing constructed the service.<br/>AppState::new now builds it and runs initialize_default_servers (src/app_state.rs:1166-1168), and every /multi-mcp/ws socket attaches and calls sync_running (VC-05.13)
    Sync->>D: initialize_default_servers() once at boot - :212-270 (called at src/app_state.rs:1168)
    D->>D: insert claude-flow :215-230 (enabled :225), ruv-swarm :232-247 (enabled only if env_configured :242), daa :249-264 (enabled only if env_configured :259) - env_configured :134-137
    Sync->>D: attach() :170 and detach() :175 move the subscriber count synchronously, sync_running :184-193 starts or stops the loop to match
    D->>D: start_discovery :298 - epoch bumped :307, wake notified :308, so a quick restart leaves one loop
    loop while running and this loop's epoch is current (tokio::spawn multi_mcp_agent_discovery.rs:324-434, condition :327)
        par concurrent per enabled server (multi_mcp_agent_discovery.rs:336-410)
            D->>CF: discover_server_agents :452 -> discover_claude_flow_agents - :472, :482-568
            CF-->>D: query_server_info / query_agent_list / query_swarm_status - :512,533,556
            D->>RS: discover_ruv_swarm_agents - :473, :570-661
            RS-->>D: same three-call pattern, server_type=RuvSwarm - :600,626,649
            D->>DAA: discover_daa_agents - :474, :663-750
            DAA-->>D: same pattern, server_type=Daa - :693,718,738
        end
        alt discover_server_agents Ok
            D->>D: insert server_info/agents/topology, successful_discoveries+=1 - :348-378
        else Err (connect/timeout)
            D->>D: failed_discoveries+=1, is_connected=false, entry inserted if absent so a first-cycle failure is visible - :379-406
        end
        D->>D: cycles += 1 :413, epoch recheck :417, then select between sleep(sleep_ms) and wake - :425-428
    end
    D->>D: stop_discovery :437-450 - running=false, epoch bumped :441, caches cleared :446-448
    Note over D,CF: RESOLVED ADR-2083 (estate) - WIRED, not removed: the per-server values are deliberate, so the<br/>loop now sleeps the minimum interval across ENABLED servers (select_discovery_interval_ms :46, called :333), with a named<br/>MIN_DISCOVERY_INTERVAL_MS floor so a misconfigured 0 cannot spin it, and a named fallback when no server is<br/>enabled. The flat 1000ms sleep is gone (:425-428).
    Note over CF,DAA: supported_tools fallback differs per server when query_server_info fails:<br/>claude-flow=[agent_list,swarm_status,server_info] (multi_mcp_agent_discovery.rs:523-527),<br/>ruv-swarm (:614) and daa (:707) each have their own equivalent fallback list in their own discover_*_agents function
    rect rgb(255,235,235)
    Note over D,DAA: OPEN: ruv-swarm and daa have no confirmed endpoint in this estate and stay disabled unless RUV_SWARM_* or DAA_* are set (:242, :259).<br/>None of this was observed against a live claude-flow hub
    Note over D,DAA: RESOLVED 2026-10-12 (project 9f98e3787): discovery now resolves the claude-flow host as BotsClient does (CLAUDE_FLOW_HOST, then MCP_HOST, then multi-agent-container), so both pollers target the same host on port 9500. The host is resolved by claude_flow_host (multi_mcp_agent_discovery.rs:144-148, used at :220) in the same order as bots_client.rs:120-122, so the old localhost fallback is gone.
    Note over D,DAA: DEBT: DiscoveryStats total_discoveries :91, total_agents_discovered :94, servers_online :97 and servers_offline :98 are declared but never written<br/>by this loop, which updates only successful_discoveries :367, failed_discoveries :383 and last_discovery_time :368<br/>The WS frame no longer reads those stored fields: build_discovery_frame derives total_discoveries as successful+failed, total_agents_discovered as agents.len() and servers_online/offline from the frame's own server list (multi_mcp_websocket_handler.rs:393-402)
    end
```

## VC-27.5 MultiMcpVisualizationWs — `/multi-mcp/ws` session and opcodes

```mermaid
sequenceDiagram
    autonumber
    participant Client as WS client
    participant H as multi_mcp_visualization_ws()<br/>src/handlers/multi_mcp_websocket_handler.rs:776
    participant Ws as MultiMcpVisualizationWs<br/>multi_mcp_websocket_handler.rs:410 (Actor::started)
    participant DS as MultiMcpAgentDiscovery<br/>src/services/multi_mcp_agent_discovery.rs:101

    Client->>H: GET /multi-mcp/ws (upgrade) - configure_multi_mcp_routes :867-878
    H->>H: require Authorization Bearer and a live session or 401 - :783-858
    H->>Ws: ws::start(MultiMcpVisualizationWs::new(app_state.multi_mcp_discovery.clone())) - :860-864
    Ws->>Ws: started() - start_heartbeat :413, discovery.attach() :419, spawn sync_running :421, start_health_monitor :422 (ADR-2094), start_position_updates :439 - :410-439
    Note over Ws: started() snapshots discovery.cycle_count() into last_frame_cycle (multi_mcp_websocket_handler.rs:426), so only a cycle that completes after this socket attached can produce a frame -<br/>a stale or empty cache is never replayed as the first frame
    Ws->>DS: attach() moves the subscriber count, sync_running starts the poll loop - multi_mcp_agent_discovery.rs:170, :184-193 (VC-27.4)
    Ws->>Ws: run_interval 1s - cycle_count() above last_frame_cycle triggers send_discovery_data - :431-437
    Ws->>Ws: run_interval 60s recovery-if-idle-300s + circuit stats log - :441-463
    Note over Ws: DOC-DRIFT (RESOLVED 2026-10-12, project 6a1a74b8b/4f4345503) - started() used to register MONITORED_SERVICES health endpoints and call send_discovery_data(ctx) on connect.<br/>Both are gone: the localhost port 8080+i registration no longer exists and nothing is sent at connect, because an empty cache would read as no agents (comment :428-430)
    loop position updates (PerformanceMode: HighFreq=16ms/Normal=100ms/Low=1000ms, OnDemand=none) - multi_mcp_websocket_handler.rs:107-116, :146-155
        Ws->>Ws: do_send(RequestAgentUpdate)
    end
    loop heartbeat every 5s (start_heartbeat) - multi_mcp_websocket_handler.rs:158-171
        alt no pong for >30s
            Ws-->>Client: ctx.close(None) - :165
        else
            Ws-->>Client: ctx.ping(b"ping") - :169
        end
    end
    Client->>Ws: Text "ping" (plain)
    Ws-->>Client: "pong" - :486-489
    Client->>Ws: Text JSON {action, data} - ClientRequest :617-620
    alt action == configure
        Ws->>Ws: handle_client_config(ClientConfig{subscription_filters,performance_mode}) - :311-333
    else action == request_discovery
        Ws->>Ws: handle_discovery_request(ctx) - :335-348, rate-limited to 1 per second :338-344
        Ws->>Ws: send_discovery_data - :249-309
        alt cached verdict or fresh get_server_statuses() says no server connected
            Ws-->>Client: {type:error, message:"No connected MCP servers"} - :257-264, :282-290
        else at least one connected
            Ws-->>Client: {type:discovery_data, servers, agents, topology, stats} filtered by SubscriptionFilters - build_discovery_frame :357-405
        end
    else action == request_agents
        Ws->>Ws: reads CircuitBreaker stats then do_send(RequestAgentUpdate) in both arms - :507-543
    else action == request_performance
        alt has_healthy_services() true
            Ws->>Ws: do_send(RequestPerformanceUpdate) - :559
        else degraded
            Ws-->>Client: cached "performance_data" status=degraded - :546-557
        end
    else action == request_topology
        Ws->>Ws: do_send(RequestTopologyUpdate{swarm_id}) - :562-572
    else unknown action
        Ws-->>Client: send_error_response("Unknown action: ...") - :573-579
    end
    Client->>Ws: ws::Message::Close
    Ws->>Ws: log final circuit-breaker stats, ctx.close(reason) - :586-603
    Ws->>DS: stopped() - detach() :468, spawn sync_running :470, the loop stops when the last socket leaves and its caches are cleared (multi_mcp_agent_discovery.rs:437-450)
    Note over Ws: RESOLVED ADR-2094 (2026-09-05): has_healthy_services (:226) is a pure atomic read of a cached verdict<br/>one monitor task started at connection init publishes it and stops when the client drops (start_health_monitor :180) - no per-call spawn
    Note over Ws: DOC-DRIFT (RESOLVED 2026-10-12, project 6a1a74b8b/4f4345503) - the monitor used to probe localhost port 8080+i through a HealthCheckManager, services that were not the ones discovery polls.<br/>It now folds the discovery service's own per-server is_connected (get_server_statuses :202, fold_service_health :208), one task per connection as before
    Note over H,Ws: RESOLVED ADR-2091 (supersedes an earlier DOC-DRIFT finding that they merely returned<br/>fiction): GET /multi-mcp/status and POST /multi-mcp/refresh are DELETED, not just hardcoded -<br/>configure_multi_mcp_routes only registers /ws now (:867-878). /status served a hardcoded<br/>two-server JSON literal (claude-flow is_connected:true agent_count:4, never queried) and<br/>/refresh echoed "initiated"
    rect rgb(255,235,235)
    Note over Ws,DS: DEBT: request_agents, request_performance and request_topology never reach the discovery service - the RequestAgentUpdate, RequestPerformanceUpdate and<br/>RequestTopologyUpdate handlers only log at debug (multi_mcp_websocket_handler.rs:669-671, :694-699, :705-710). The ADR-2091 addendum names the request_agents circuit-breaker path as a simulation left unchanged
    Note over Ws,DS: OPEN: no client in client/src or xr-client consumes the discovery_data frame, so its shape is provisional and defined only by build_discovery_frame (multi_mcp_websocket_handler.rs:357-405).<br/>Nothing was observed against a live hub, and ruv-swarm and daa have no confirmed endpoint (VC-27.4)
    end
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
    participant AM as AgentMonitorActor<br/>src/actors/agent_monitor_actor.rs:199-233 (struct), new() :279-346
    participant MAC as ManagementApiClient<br/>host=MANAGEMENT_API_HOST port=MANAGEMENT_API_PORT default 9090 (:251-256)
    participant GSS as GraphServiceSupervisor

    Sup->>AM: started() - is_connected=true, do_send(InitializeActor) - :431-438
    AM->>AM: handle(InitializeActor) - run_later(100ms) poll_agent_statuses + schedule_next_poll - :445-460
    loop self-rescheduling poll (schedule_next_poll, :412-418)
        AM->>MAC: tokio::join!(list_tasks(), get_system_status()) - :331-332
        alt tasks_result Ok
            AM->>AM: task_to_agent_status per active task - :373-377
            AM->>AM: do_send(ProcessAgentStatuses{agents,telemetry}) - :379
        else tasks_result Err
            AM->>AM: do_send(RecordPollFailure) - :383
        end
        AM->>AM: next_poll_delay() - base 15s (:336), doubles per consecutive_poll_failures (max shift 5), capped 90s - :445-462
    end
    AM->>AM: handle(ProcessAgentStatuses) - :631
    opt agents empty and MOCK_AGENTS=true/1
        AM->>AM: build_mock_swarm_agents() 5 mock agents - :526-625
    end
    AM->>AM: golden-angle spiral position per agent, poll_offset round-robin (ADR-031 item 1) - :593-631
    AM->>AM: decide_bots_graph_emit(count, last_nonempty, consecutive_empty) - call :701-705, fn body :156-197
    alt roster non-empty (fn :131-137)
        AM->>GSS: do_send(UpdateBotsGraph{agents}) - :709-716
    else roster empty and consecutive_empty < EMPTY_CONFIRM_THRESHOLD=2 (fn :140-148,159-165)
        AM->>AM: suppress emit - debounce a transient blip - :653-659
    else roster empty and confirmed (2nd consecutive empty) (fn :150-158)
        AM->>GSS: do_send(UpdateBotsGraph{agents: []}) - clears once - :709-716
    end
    Sup->>AM: TaskStatusChanged{agent_type,running_task_count} (from TaskOrchestratorActor, ADR-031 item 3) - handler :714-725
    AM->>AM: poll_agent_statuses(ctx) immediate re-poll - :857
    Note over AM,MAC: INVARIANT: idle cadence is 15s (not 3s) to share agentbox's per-key rate-limit<br/>bucket with task creation - backoff cap 90s exceeds agentbox's 60s continueExceeding window (:336,445-462)
    Note over AM,GSS: DIVERGENCE (roster-clobber fix): an empty Management API poll is "no information"<br/>not "all agents died" - only a confirmed 2nd consecutive empty poll clears the graph (:225-229,156-197)
    Note over MAC: every ManagementApiClient endpoint shares one Result contract - NetworkError on<br/>transport failure, ApiError(body,status) on non-2xx, DeserializationError on a bad body<br/>(e.g. list_tasks management_api_client.rs:358-386, stop_task :388-413, create_brief<br/>:446-497) — health_check (:599-610) is the one exception - GET /health with no auth<br/>header, collapses straight to Ok(status==200), no ApiError variant
    Note over AM,MAC: ADR-2094: AgentMonitorActor::new validates MANAGEMENT_API_KEY through the SAME<br/>validate_security_env_vars AppState::new calls (app_state.rs:82-174, pub(crate) so there is one policy,<br/>not a laxer copy) - fail-closed by default (panics without a valid key, agent_monitor_actor.rs:313-323),<br/>except an ALLOW_INSECURE_DEFAULTS-gated dev/debug build instead disables the client (Option::None, never<br/>an empty-string key) rather than aborting boot (:301-312)
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

## VC-27.11 AgentVisualizationProcessor — `/api/visualization/agents/ws` init/refresh

```mermaid
sequenceDiagram
    autonumber
    participant Client as WS client
    participant H as agent_visualization_ws()<br/>src/handlers/bots_visualization_handler.rs:203
    participant Ws as AgentVisualizationWs<br/>bots_visualization_handler.rs:17, Actor :92
    participant Proto as AgentVisualizationProtocol<br/>services/agent_visualization_protocol.rs:429, new() :451
    participant Proc as AgentVisualizationProcessor<br/>services/agent_visualization_processor.rs:162, new() :168

    Client->>H: GET /api/visualization/agents/ws - configure_routes :504-520
    H->>Ws: ws::start(AgentVisualizationWs::new) - :208
    Ws->>Ws: started() - do_send(InitConnection), start_heartbeat, start_position_updates - :95-103
    Ws->>Ws: handle(InitConnection) -> send_init_state(ctx) - :131-133
    Ws->>Proto: create_init_message("swarm-001","hierarchical", agents=Vec::new()) - :57-58
    Proto->>Proc: create_visualization_packet(agents, swarm_id, topology) - agent_visualization_protocol.rs:535 calling agent_visualization_processor.rs:308
    Proc->>Proc: process_agents() - color/shape/animation via get_visual_properties, spherical fallback position, glow_intensity - :175-257
    Proc->>Proc: create_connections(), create_clusters() - :365-389,391-420
    Proc-->>Proto: AgentVisualizationData{swarm,agents,connections,physics_config,...}
    Proto-->>Ws: init_json (AgentInit list mapped from VisualizedAgent) - agent_visualization_protocol.rs:541-577
    Ws-->>Client: ctx.text(init_json) - :61
    loop position updates every 16ms (bots_visualization_handler.rs:68-77)
        Ws->>Proto: create_position_update() - :73
        opt buffered updates present
            Ws-->>Client: ctx.text(update_json)
        end
    end
    loop heartbeat every 5s (bots_visualization_handler.rs:79-89)
        alt no pong for >10s
            Ws-->>Client: ctx.stop() - :81-84
        else
            Ws-->>Client: ctx.ping(b"ping") - :87
        end
    end
    Client->>Ws: Text {action} - :165-184
    alt action == refresh
        Ws->>Ws: send_init_state(ctx) again - :168-170
    else action == pause_updates or resume_updates
        Ws->>Ws: sets self.paused (ADR-2066 addendum), then debug log - :171-174,184-187
    else unknown action
        Ws->>Ws: warn "Unknown client action" - :179-181
    end
    Note over Ws,Proto: PROPOSED ADR-2066 addendum: send_init_state still reports an empty roster, now explicit rather than<br/>disguised - the fake get_real_agent_data() helper is deleted. A real source exists (bots_client.get_agents_snapshot,<br/>bots_client.rs:231) but Agent lacks the profile, task counts, success_rate and timestamp AgentStatus requires, so the<br/>mapping needs a decided contract rather than invented defaults.
    Note over Ws: RESOLVED ADR-2066 addendum: the actor now carries a paused flag - pause_updates and resume_updates<br/>set it and the 16ms run_interval returns early while it is set, so the opcodes do what they advertise.
    Note over Proto: agent_visualization_protocol::AgentVisualizationMessage is internally tagged<br/>(serde(tag="type"), agent_visualization_protocol.rs:6-23) - the wire "type" field is the serde rename, not the<br/>Rust variant name: Initialize->"init", PositionUpdate->"positions", StateUpdate->"state",<br/>ConnectionUpdate->"connections", MetricsUpdate->"metrics". AgentStateUpdate (agent_visualization_protocol.rs:107-123)<br/>is a partial differential update - every field but id is Option~T~, and its status string's recognised values<br/>(idle/busy/active/initializing/terminating/offline/error, plus XR-swarm blocked/done) map to the XR client's<br/>status halo via render_store::agent_status_code (busy/active/working->working, blocked/error->blocked,<br/>done/terminating/offline->done, else idle)
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

## VC-27.14 AgentDemoDirector — one server-side synthetic producer (ADR-2142)

```mermaid
sequenceDiagram
    autonumber
    participant C as web or XR client
    participant H as set_demo<br/>src/handlers/api_handler/agents_demo.rs:30
    participant D as AgentDemoDirector<br/>src/actors/agent_demo_director.rs:386
    participant G as GraphStateActor
    participant CC as ClientCoordinatorActor
    participant HUB as agent event hub and KPI tap

    C->>H: POST /api/agents/demo {running: true} - authenticated, route agents_demo.rs:51-57
    H->>D: StartAgentDemo - agents_demo.rs:36
    D->>G: GetGraphData - agent_demo_director.rs:469
    G-->>D: graph, reduced by demo_graph_from to public non-agent nodes - agent_demo_director.rs:496-519
    alt no usable node
        D-->>H: Err - the graph has no nodes the demo agents can work on - agent_demo_director.rs:477-479
        H-->>C: 409 Conflict - agents_demo.rs:38
    else director built
        D-->>H: DemoStatus running with the six reserved agent ids - agent_demo_director.rs:409-414
        H-->>C: 200 - agents_demo.rs:37
    end
    loop every 250 ms TICK while a director exists - agent_demo_director.rs:394, :427
        D->>D: DemoDirector.tick advances stagger, work, follow-an-edge and rest per agent - agent_demo_director.rs:212, :432
        D->>CC: BroadcastAgentActionFrame of the encoded 0x23 action events - agent_demo_director.rs:434-435
    end
    Note over D,HUB: INVARIANT: the demo never enters the agent event hub - a synthetic event there would be persisted by the KPI tap.<br/>Frames go to the client coordinator only, the same door real agent actions leave through<br/>src/actors/agent_demo_director.rs:3-6
    Note over D: the six agents carry reserved ids 0x80000000 | 0xD001..0xD006 and nothing in a frame says demo, so provenance lives in code alone<br/>src/actors/agent_demo_director.rs:35, :47, :205-207
    C->>H: POST /api/agents/demo {running: false} - agents_demo.rs:43
    H->>D: StopAgentDemo drops the director, frames stop and clients' evidence expiry retires the agents - agent_demo_director.rs:451-454
    Note over C,H: GET /api/agents/demo takes no AuthenticatedUser extractor at handler level and returns the director's state, so either client's button reflects a demo the other started<br/>src/handlers/api_handler/agents_demo.rs:20-27
```

**What it shows:** the shared agents demo as a server-side producer of agent-action frames that is started and stopped through one authenticated endpoint and fans out through the client coordinator, bypassing the agent-event hub that VC-27.13 describes.

**Why it is this way:** ADR-2142 (2026-10-14) replaced two client-side demo producers, each feeding only its own screen, with one director both clients control; it stays out of the hub so synthetic events never reach durable records. The decision is accepted and implemented, with activation staged.
