this is some of the code for our back end system which runs in a docker and talks to a multi-agent-container to assign tasks sent from the clients, with a UUID per task. Communication with the docker allows detailed monitoring and logs to be sent onward to the clients (some via the force directed graph on gpu). We have RADICALLY simplified and refactored the communication with the agent docker. We need a very detailed refactoring plan for the rust code to align it with the agentic docker management seen in the attached code.

First check that the agent docker in /multi-agent-docker and it's associated z ai docker have naming and networking suitable for discovery by the main visionflow container.


Detailed Refactoring Plan
This plan is structured in five phases to systematically decouple the Rust backend from the old architecture and align it with the new, simplified agentic docker management.
Phase 1: Decouple Task Management & Remove Docker Control
Objective: Eliminate all direct Docker interaction from the Rust backend. All task creation and management must be delegated to the management-api via HTTP.
Create a Management API Client:
File: src/services/management_api_client.rs (New File)
Action: Implement a new Rust client using reqwest to interact with the multi-agent-container's Management API on port 9090.
Methods to Implement:
create_task(agent: &str, task: &str, provider: &str) -> Result<String, Error>: Sends a POST request to /v1/tasks and returns the taskId.
get_task_status(task_id: &str) -> Result<TaskStatus, Error>: Sends a GET request to /v1/tasks/:taskId.
list_tasks() -> Result<Vec<TaskInfo>, Error>: Sends a GET request to /v1/tasks.
stop_task(task_id: &str) -> Result<(), Error>: Sends a DELETE request to /v1/tasks/:taskId.
get_system_status() -> Result<SystemStatus, Error>: Sends a GET request to /v1/status.
Configuration: The client should read MANAGEMENT_API_HOST (e.g., multi-agent-container) and MANAGEMENT_API_PORT (e.g., 9090) from environment variables.
Create a Task Orchestrator Actor:
File: src/actors/task_orchestrator_actor.rs (New File)
Action: Create a new Actix actor that wraps the ManagementApiClient. This actor will manage the asynchronous lifecycle of tasks submitted by the Rust backend.
Responsibilities:
Handle messages like CreateTask, GetTaskStatus.
Manage API keys for authenticating with the management-api.
Implement retry logic for HTTP requests.
Optionally, poll for task completion status in the background and notify other actors.
Remove Redundant Docker Utilities:
Action: Delete the following files and modules, as their functionality is now entirely handled by the management-api within the Docker container:
src/utils/docker_hive_mind.rs
src/services/mcp_session_bridge.rs
src/services/session_correlation_bridge.rs
Action: Remove these modules from src/utils/mod.rs and src/services/mod.rs.
Update AppState:
File: src/app_state.rs
Action:
Remove the mcp_session_bridge and session_correlation_bridge fields.
Add a new field: pub task_orchestrator_addr: Addr<TaskOrchestratorActor>.
In AppState::new(), start the new TaskOrchestratorActor and store its address.
Phase 2: Simplify Agent Monitoring (claude_flow_actor.rs)
Objective: Refactor ClaudeFlowActorTcp into a lean, dedicated monitoring actor whose sole purpose is to poll the MCP TCP server (port 9500) for agent statuses.
Rename and Refocus the Actor:
File: src/actors/claude_flow_actor.rs
Action: Rename ClaudeFlowActorTcp to AgentMonitorActor. This clarifies its new, focused role. Update the reference in src/actors/mod.rs.
Eliminate Sub-Actors and Complex Connection Management:
File: src/actors/agent_monitor_actor.rs (formerly claude_flow_actor.rs)
Action: Remove the tcp_actor: Option<Addr<TcpConnectionActor>> and jsonrpc_client: Option<Addr<JsonRpcClient>> fields.
Action: Delete the initialize_sub_actors method. The actor will no longer manage its own sub-actors for TCP and JSON-RPC.
Action: Remove the Handler implementations for TcpConnectionEvent and MCPSessionReady.
Streamline the Polling Mechanism:
File: src/actors/agent_monitor_actor.rs
Action: Modify the poll_agent_statuses method to be the only communication point. It should directly use the create_mcp_client utility from src/utils/mcp_tcp_client.rs within a tokio::spawn block, just as it currently does as a fallback. This becomes the primary logic.
Action: The actor's state should be simplified to is_connected and agent_cache. The main loop in started() will periodically call poll_agent_statuses.
Remove Obsolete Handlers and Logic:
Action: Remove the Handler implementations for InitializeSwarm, GetSwarmStatus, and any other CallTool-based messages. These actions are now handled by the management-api.
Action: Remove the mcp_agent_to_status and parse_legacy_response functions. The new TCP server provides a clean MultiMcpAgentStatus list, which should be the single source of truth. The actor should only need to convert MultiMcpAgentStatus to the internal AgentStatus.
Delete Unused Communication Actors:
Action: Delete the following files, as they are replaced by the direct client usage in AgentMonitorActor:
src/actors/jsonrpc_client.rs
src/actors/tcp_connection_actor.rs
Action: Remove their modules from src/actors/mod.rs.
Phase 3: Refactor API Handlers (bots_handler.rs)
Objective: Reroute all task-related HTTP requests from the Rust backend's API to use the new TaskOrchestratorActor, effectively turning the Rust backend into a proxy for the management-api.
Refactor initialize_hive_mind_swarm:
File: src/handlers/bots_handler.rs
Action: Completely rewrite this function.
New Logic:
Instead of using DockerHiveMind or McpSessionBridge, it should parse the InitializeSwarmRequest.
It will then create a CreateTask message containing the agent type ("claude-flow"), task description (from the request), and provider.
Send this message to the TaskOrchestratorActor stored in AppState.
Await the response, which should be the taskId from the management-api.
Return a 202 Accepted response to the client with the taskId.
Refactor spawn_agent_hybrid:
File: src/handlers/bots_handler.rs
Action: Rewrite this function.
New Logic: This handler is now redundant. The management-api handles spawning. This endpoint should be deprecated or refactored to be a specific type of task creation. For now, it can be simplified to call the TaskOrchestratorActor with a predefined task description like "Spawn agent of type {agent_type}".
Update or Remove Task Control Handlers:
File: src/handlers/bots_handler.rs
Action: The management-api exposes DELETE /v1/tasks/:taskId to stop a task.
Refactor remove_task to send a StopTask { task_id } message to the TaskOrchestratorActor.
The pause_task and resume_task handlers are now obsolete as the management-api does not support them. They should be removed. Update src/handlers/api_handler/bots/mod.rs to remove the routes.
Phase 4: Consolidate Graph State Management
Objective: Ensure a single, clear data flow for graph updates and remove monolithic actor patterns.
Introduce GraphStateActor:
File: src/actors/graph_state_actor.rs (New File, based on provided code)
Action: Create this actor to be the sole owner of GraphData (both for the knowledge graph and the bots graph). It will handle all CRUD operations on nodes and edges.
Refactor GraphServiceSupervisor:
File: src/actors/graph_service_supervisor.rs
Action: Transition from the TransitionalGraphSupervisor to the full GraphServiceSupervisor pattern.
New Logic:
The supervisor should spawn and manage four distinct child actors: GraphStateActor, PhysicsOrchestratorActor, SemanticProcessorActor, and ClientCoordinatorActor.
It will no longer hold a GraphServiceActor. Instead, it will route messages. For example, UpdateBotsGraph messages from AgentMonitorActor should be routed to the GraphStateActor. UpdateNodePositions from the PhysicsOrchestratorActor should also go to the GraphStateActor.
This removes the monolithic GraphServiceActor and promotes separation of concerns.
Update Data Flow for Graph Updates:
AgentMonitorActor: Will now send UpdateBotsGraph to the GraphServiceSupervisor, which forwards it to the GraphStateActor.
PhysicsOrchestratorActor: Will receive graph data from GraphStateActor for simulation. After a physics step, it will send position updates back to GraphStateActor.
ClientCoordinatorActor: Will receive position updates from GraphStateActor (or be notified of changes) to broadcast to clients.
Phase 5: Final Cleanup and Validation
Objective: Remove any remaining legacy code and validate the new architecture.
Code Removal:
Action: Delete the src/actors/graph_actor.rs file, as its responsibilities have been distributed among GraphStateActor, PhysicsOrchestratorActor, etc.
Action: Review all files in src/services and src/utils for any code that is now unused and delete it. Pay special attention to mcp_connection.rs and hybrid_fault_tolerance.rs.
Action: Clean up src/actors/messages.rs to remove messages that are no longer sent or handled.
Configuration Update:
Action: Review all [DEPRECATED] and [REMOVED] comments in the provided code files and perform the indicated actions.
Action: Update environment variable handling. The Rust backend now only needs variables for its own server and the connection details for the management-api and mcp-tcp-server.
Testing Strategy:
Unit Tests: Write unit tests for the new ManagementApiClient to ensure it correctly formats HTTP requests.
Integration Tests:
Test the full task creation flow: Client -> Rust API -> TaskOrchestratorActor -> Management API.
Test the full monitoring flow: multi-agent-container -> AgentMonitorActor -> GraphStateActor -> ClientCoordinatorActor -> Client.
End-to-End Tests: Run the full system and verify that creating a task via the Rust backend's API results in agents appearing on the force-directed graph.