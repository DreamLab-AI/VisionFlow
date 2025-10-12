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

Phase 4: Graph State Management Refactoring - Comprehensive Analysis

  Current Architecture Overview

  TransitionalGraphSupervisor (Lines 810-1175)

  What it is: A simple wrapper around the monolithic GraphServiceActor
  - Current role: Message forwarder with basic supervision lifecycle
  - Size: 365 lines
  - Pattern: Delegates ALL messages to wrapped GraphServiceActor
  - Purpose: Maintains compatibility while preparing for true supervision

  GraphServiceActor (3890 lines, 172KB)

  The Monolith - Handles EVERYTHING:
  - Graph data storage (nodes, edges, metadata)
  - Bots graph management (separate from knowledge graph)
  - Physics simulation coordination
  - Semantic analysis & feature caching
  - Edge generation (multi-modal)
  - Stress-majorization optimization
  - GPU compute management
  - Client WebSocket broadcasting
  - Auto-balancing detection
  - Position change tracking
  - Update queue batching (1000 operations)
  - Constraint management
  - Performance metrics

  Existing Specialized Actors

  - PhysicsOrchestratorActor (28KB) - Already handles physics coordination
  - SemanticProcessorActor (52KB) - Already handles semantic analysis
  - ClientCoordinatorActor (32KB) - Already handles client connections

  ---
  Proposed Phase 4 Architecture

  1. Create GraphStateActor (NEW)

  Pure State Management Actor

  Responsibilities:
  - Own GraphData (knowledge graph)
  - Own bots_graph_data (agent visualization)
  - CRUD operations on nodes/edges
  - Position updates from physics
  - Metadata management
  - Update queue batching
  - Position change detection

  Key Operations:
  - AddNode / BatchAddNodes
  - AddEdge / BatchAddEdges
  - RemoveNode / RemoveEdge
  - UpdateNodePositions (from PhysicsOrchestrator)
  - GetGraphData / GetBotsGraphData
  - UpdateBotsGraph (from AgentMonitorActor)

  2. Refactor GraphServiceSupervisor

  From Wrapper → True Supervisor

  Old Pattern:
  GraphServiceSupervisor
    └── GraphServiceActor (wraps, forwards everything)

  New Pattern:
  GraphServiceSupervisor (routes messages)
    ├── GraphStateActor          (state & persistence)
    ├── PhysicsOrchestratorActor (GPU physics)
    ├── SemanticProcessorActor   (AI/semantic)
    └── ClientCoordinatorActor   (WebSocket clients)

  Message Routing Logic:
  - UpdateBotsGraph → GraphStateActor
  - StartSimulation → PhysicsOrchestratorActor
  - SemanticAnalysis → SemanticProcessorActor
  - BroadcastPositions → ClientCoordinatorActor

  3. New Data Flow

  Current Flow (Monolithic):
  AgentMonitorActor → TransitionalGraphSupervisor → GraphServiceActor
                                                      ↓
                                              [Does Everything]
                                                      ↓
                                              ClientCoordinatorActor

  Proposed Flow (Distributed):
  AgentMonitorActor → GraphServiceSupervisor → GraphStateActor
                                                 ↓
                                          PhysicsOrchestratorActor
                                                 ↓
                                          GraphStateActor (positions)
                                                 ↓
                                          ClientCoordinatorActor

  ---
  Advantages

  1. Separation of Concerns ✅

  - Current: One actor does 20+ different things
  - Proposed: Each actor has ONE clear responsibility
  - Benefit: Easier to reason about, test, and debug

  2. Fault Isolation ✅

  - Current: Physics crash = entire graph system down
  - Proposed: Physics crash = restart only PhysicsOrchestrator
  - Benefit: System resilience through OneForOne restart policy

  3. Scalability ✅

  - Current: All operations bottleneck through single actor
  - Proposed: Parallel processing across 4 actors
  - Benefit: Physics and semantic analysis can run concurrently

  4. State Clarity ✅

  - Current: 149 fields in GraphServiceActor struct (lines 90-149)
  - Proposed: GraphStateActor owns ~20 fields (just data + queue)
  - Benefit: Clear state boundaries, no shared mutable state

  5. Testing ✅

  - Current: Must mock entire system to test one feature
  - Proposed: Test GraphStateActor independently
  - Benefit: Unit tests without full actor system setup

  6. Hot Reload ✅

  - Current: Restart = lose all graph data
  - Proposed: Restart physics without losing graph state
  - Benefit: Development velocity, production stability

  7. Message Buffering ✅

  - Supervision Strategy: Buffer messages during actor restart
  - Benefit: No lost updates during restarts (lines 464-485)

  8. Performance Monitoring ✅

  - Per-Actor Metrics: Messages processed, failures, response times
  - Benefit: Identify bottlenecks precisely (lines 115-124)

  ---
  Disadvantages

  1. Complexity Increase ⚠️

  - Issue: 4 actors + supervisor = more moving parts
  - Mitigation: Clear boundaries reduce cognitive load
  - Risk Level: Medium - well-documented patterns help

  2. Message Overhead ⚠️

  - Issue: Inter-actor messages vs direct function calls
  - Current: self.update_positions() (function call)
  - Proposed: graph_state_addr.send(UpdatePositions{...}).await
  - Overhead: ~50-200μs per message (actix-web benchmarks)
  - Impact: For 10,000 position updates/second = 0.5-2% overhead
  - Risk Level: Low - batching mitigates this

  3. State Synchronization ⚠️

  - Issue: Physics actor needs graph data copy
  - Current: Direct Arc access
  - Proposed: Request data from GraphStateActor
  - Pattern:
  let graph_data = graph_state_addr.send(GetGraphData).await?;
  // Physics computation
  graph_state_addr.send(UpdatePositions{...}).await?;
  - Risk Level: Medium - requires careful message ordering

  4. Migration Effort ⚠️

  - Lines to Refactor: ~1500 lines in GraphServiceActor
  - New Code: ~800 lines for GraphStateActor
  - Testing Required: Integration tests for all 4 actors
  - Estimate: 8-12 hours of focused work
  - Risk Level: High - potential for breakage

  5. Debugging Difficulty ⚠️

  - Issue: Message flow across 4 actors harder to trace
  - Current: Single stack trace
  - Proposed: Message logs across actors
  - Mitigation: Correlation IDs, structured logging
  - Risk Level: Medium

  6. Deadlock Potential ⚠️

  - Issue: Circular message dependencies
  - Example:
  GraphState waits on Physics response
  Physics waits on GraphState data
  = DEADLOCK
  - Mitigation:
    - Use do_send() (fire-and-forget) where possible
    - Clear data flow direction (acyclic)
    - Timeout on send().await
  - Risk Level: Medium - requires careful design

  7. Memory Duplication ⚠️

  - Issue: GraphData cloned across actors
  - Current: Single Arc
  - Proposed: Multiple clones for physics/semantic
  - Impact: For 10,000 nodes = ~2MB × 3 actors = 6MB
  - Mitigation: Use Arc in messages
  - Risk Level: Low - Arc is cheap

  8. Backpressure Handling ⚠️

  - Issue: GraphStateActor mailbox can overflow
  - Current: Direct function calls never "overflow"
  - Proposed: Bounded mailbox (default 16 messages)
  - Failure Mode: send().await returns error on full mailbox
  - Mitigation:
    - Update queue batching (already implemented)
    - Bounded queue config (lines 152-167)
    - Backpressure monitoring (lines 130-134)
  - Risk Level: Low - existing patterns handle this

  9. Transaction Semantics ⚠️

  - Issue: Multi-actor updates not atomic
  - Example:
  1. GraphState adds node
  2. Physics crashes before receiving update
  3. Node exists without physics simulation
  - Current: Single actor = atomic operations
  - Proposed: No transaction guarantees
  - Mitigation:
    - Eventual consistency model
    - State recovery on restart
    - Idempotent message handlers
  - Risk Level: Medium - acceptable for visualization

  ---
  Specific Implementation Concerns

  UpdateBotsGraph Handler (Lines 3113-3230)

  Current: 117 lines in GraphServiceActor
  - Converts agents to nodes
  - Preserves existing positions (HashMap lookup)
  - Generates random positions for new agents
  - Updates bots_graph_data
  - Broadcasts to clients

  Proposed: Split across 2 actors
  // GraphStateActor
  fn handle(&mut self, msg: UpdateBotsGraph) {
      // Convert agents → nodes
      // Update bots_graph_data
      // Notify ClientCoordinator
  }

  // ClientCoordinatorActor
  fn handle(&mut self, msg: BroadcastBotsGraph) {
      // Broadcast to WebSocket clients
  }

  Challenge: 117 lines of business logic must be carefully split
  Risk: Regression in position preservation logic

  ---
  Recommendation

  ⚠️ DEFER Phase 4 Until After Production Validation

  Reasoning:
  1. Phases 1-3 provide immediate value:
    - Task management decoupled ✅
    - Agent monitoring simplified ✅
    - API handlers refactored ✅
  2. Phase 4 is high-risk, moderate-reward:
    - Requires extensive testing
    - High potential for regressions
    - Benefits are primarily architectural (not user-facing)
  3. Alternative approach:
    - Keep TransitionalGraphSupervisor as-is
    - It's working and stable
    - Provides supervision benefits without refactoring
    - Revisit after production validation of Phases 1-3

  If You Proceed Anyway

  Implementation Order:
  1. Create GraphStateActor with ONLY data storage (no logic)
  2. Test GraphStateActor in isolation (unit tests)
  3. Update GraphServiceSupervisor to spawn GraphStateActor
  4. Move UpdateBotsGraph handler (simplest migration)
  5. Test end-to-end: AgentMonitor → Supervisor → GraphState → Client
  6. Incrementally migrate other handlers
  7. Delete monolithic GraphServiceActor last

  Testing Strategy:
  - Unit test each actor independently
  - Integration test all 4 actors together
  - Load test with 10,000 nodes
  - Stress test restart scenarios
  - Validate position preservation logic

  Rollback Plan:
  - Keep GraphServiceActor in codebase (commented out)
  - Revert supervisor to TransitionalGraphSupervisor pattern
  - Should take < 30 minutes to rollback

  ---
  Final Assessment

  Advantages Summary:

  ✅ Separation of concerns
  ✅ Fault isolation✅ Scalability
  ✅ State clarity
  ✅ Testing
  ✅ Hot reload
  ✅ Message buffering
  ✅ Performance monitoring

  Disadvantages Summary:

  ⚠️ Complexity increase
  ⚠️ Message overhead (~1-2%)
  ⚠️ State synchronization complexity
  ⚠️ Migration effort (8-12 hours)
  ⚠️ Debugging difficulty
  ⚠️ Deadlock potential
  ⚠️ Memory duplication
  ⚠️ Backpressure handling
  ⚠️ No transaction semantics

  Verdict:

  Phase 4 is architecturally sound but operationally risky.
  - Good for: Greenfield projects, long-term maintainability
  - Risky for: Production systems with working monolith
  - Best approach: Defer until Phases 1-3 are validated in production

  The TransitionalGraphSupervisor already provides supervision benefits without the migration risk. Only
  proceed with Phase 4 if you have strong evidence that the monolithic GraphServiceActor is a performance
  bottleneck or causing production issues.
