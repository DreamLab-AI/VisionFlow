this is some of the code for our back end system which runs in a docker and talks to a multi-agent-container to assign tasks sent from the clients, with a UUID per task. Communication with the docker allows detailed monitoring and logs to be sent onward to the clients (some via the force directed graph on gpu). We have RADICALLY simplified and refactored the communication with the agent docker. We need a very detailed refactoring plan for the rust code to align it with the agentic docker management seen in the attached code.

First check that the agent docker in /multi-agent-docker and it's associated z ai docker have naming and networking suitable for discovery by the main visionflow container described in /src


Detailed Refactoring Plan
This plan is structured in five phases to systematically decouple the Rust backend from the old architecture and align it with the new, simplified agentic docker management.

**PHASE 0: Container Discovery & Networking - ✅ COMPLETE**
- ✅ Fixed container naming: multi-agent-container → agentic-workstation in 3 files
- ✅ Updated docker-compose.yml environment variables (both dev and prod)
- ✅ Configured docker_ragflow network for both VisionFlow and multi-agent containers
- ✅ Network connectivity validated between containers

**PHASE 1: Decouple Task Management & Remove Docker Control - ✅ COMPLETE**
✅ Objective: Eliminate all direct Docker interaction from the Rust backend. All task creation and management must be delegated to the management-api via HTTP.
✅ Create a Management API Client:
✅ File: src/services/management_api_client.rs (New File) - CREATED
✅ Action: Implement a new Rust client using reqwest to interact with the agentic-workstation's Management API on port 9090.
✅ Methods Implemented:
✅ create_task(agent: &str, task: &str, provider: &str) -> Result<String, Error>: Sends a POST request to /v1/tasks and returns the taskId.
✅ get_task_status(task_id: &str) -> Result<TaskStatus, Error>: Sends a GET request to /v1/tasks/:taskId.
✅ list_tasks() -> Result<Vec<TaskInfo>, Error>: Sends a GET request to /v1/tasks.
✅ stop_task(task_id: &str) -> Result<(), Error>: Sends a DELETE request to /v1/tasks/:taskId.
✅ get_system_status() -> Result<SystemStatus, Error>: Sends a GET request to /v1/status.
✅ Configuration: The client reads MANAGEMENT_API_HOST and MANAGEMENT_API_PORT from environment variables.
✅ Create a Task Orchestrator Actor:
✅ File: src/actors/task_orchestrator_actor.rs (New File) - CREATED
✅ Action: Created Actix actor that wraps the ManagementApiClient.
✅ Responsibilities:
✅ Handles messages: CreateTask, GetTaskStatus, StopTask, ListActiveTasks, GetSystemStatus
✅ Manages API keys for authenticating with the management-api
✅ Implements retry logic (3 attempts with exponential backoff)
✅ Caches task state and periodically cleans up old tasks
✅ Remove Redundant Docker Utilities:
✅ Action: Deleted legacy files:
✅ src/utils/docker_hive_mind.rs
✅ src/services/mcp_session_bridge.rs
✅ src/services/session_correlation_bridge.rs
✅ Action: Removed these modules from src/utils/mod.rs and src/services/mod.rs
✅ Update AppState:
✅ File: src/app_state.rs
✅ Action:
✅ Removed the mcp_session_bridge and session_correlation_bridge fields
✅ Added new field: pub task_orchestrator_addr: Addr<TaskOrchestratorActor>
✅ In AppState::new(), started TaskOrchestratorActor and stored its address
✅ Added getter method: get_task_orchestrator_addr()

**PHASE 2: Simplify Agent Monitoring (claude_flow_actor.rs) - ✅ COMPLETE**
✅ Objective: Refactor ClaudeFlowActorTcp into a lean, dedicated monitoring actor whose sole purpose is to poll the MCP TCP server (port 9500) for agent statuses.
✅ Rename and Refocus the Actor:
✅ File: src/actors/agent_monitor_actor.rs (renamed from claude_flow_actor.rs)
✅ Action: Renamed ClaudeFlowActorTcp to AgentMonitorActor. Updated reference in src/actors/mod.rs.
✅ Eliminate Sub-Actors and Complex Connection Management:
✅ File: src/actors/agent_monitor_actor.rs
✅ Action: Removed tcp_actor and jsonrpc_client fields
✅ Action: Deleted initialize_sub_actors method
✅ Action: Removed Handler implementations for TcpConnectionEvent and MCPSessionReady
✅ Streamline the Polling Mechanism:
✅ File: src/actors/agent_monitor_actor.rs
✅ Action: Modified poll_agent_statuses to be the only communication point, using create_mcp_client utility directly
✅ Action: Simplified actor state to is_connected and agent_cache
✅ Remove Obsolete Handlers and Logic:
✅ Action: Removed Handler implementations for InitializeSwarm, GetSwarmStatus, and CallTool-based messages
✅ Action: Removed mcp_agent_to_status and parse_legacy_response functions
✅ Delete Unused Communication Actors:
✅ Action: Deleted files:
✅ src/actors/jsonrpc_client.rs
✅ src/actors/tcp_connection_actor.rs
✅ Action: Removed their modules from src/actors/mod.rs

**PHASE 3: Refactor API Handlers (bots_handler.rs) - ✅ COMPLETE**
✅ Objective: Reroute all task-related HTTP requests from the Rust backend's API to use the new TaskOrchestratorActor.
✅ Refactor initialize_hive_mind_swarm:
✅ File: src/handlers/bots_handler.rs
✅ Action: Completely rewrote this function
✅ New Logic:
✅ Parses InitializeSwarmRequest
✅ Creates CreateTask message with agent type, task description, and provider
✅ Sends message to TaskOrchestratorActor
✅ Returns 202 Accepted response with taskId
✅ Refactor spawn_agent_hybrid:
✅ File: src/handlers/bots_handler.rs
✅ Action: Rewrote this function
✅ New Logic: Simplified to call TaskOrchestratorActor with task description "Spawn {agent_type} agent for swarm {swarm_id}"
✅ Update or Remove Task Control Handlers:
✅ File: src/handlers/bots_handler.rs
✅ Action: Refactored remove_task to send StopTask message to TaskOrchestratorActor
✅ Action: Removed pause_task and resume_task handlers (Management API doesn't support pause/resume)
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
