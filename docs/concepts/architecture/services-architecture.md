# Services Architecture - WebXR Knowledge Graph Platform

**Version:** 1.0.0
**Last Updated:** 2025-11-04
**Status:** Production

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [API Service Layer](#api-service-layer)
3. [Handler Architecture](#handler-architecture)
4. [Worker System](#worker-system)
5. [Repository Pattern](#repository-pattern)
6. [Service Layer](#service-layer)
7. [Actor System](#actor-system)
8. [CQRS Implementation](#cqrs-implementation)
9. [Dependency Injection](#dependency-injection)
10. [Data Flow Diagrams](#data-flow-diagrams)
11. [Performance Characteristics](#performance-characteristics)
12. [Code Examples](#code-examples)

---

## Architecture Overview

### System Architecture

The WebXR Knowledge Graph Platform follows a **layered hexagonal architecture** with **CQRS** (Command Query Responsibility Segregation) patterns:

```
┌────────────────────────────────────────────────────────────────┐
│                    HTTP/WebSocket Clients                       │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                  API Service Layer (Actix-Web)                  │
│  - REST Endpoints (52+ handlers)                                │
│  - WebSocket Endpoints (8 concurrent handlers)                  │
│  - Authentication & Authorization                               │
│  - Request/Response Serialization                               │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│              Application Service Layer (CQRS)                   │
│  - GraphApplicationService                                      │
│  - SettingsApplicationService                                   │
│  - OntologyApplicationService                                   │
│  - PhysicsApplicationService                                    │
│  Command Bus / Query Bus / Event Bus                            │
└─────────┬─────────────────┬───────────────────┬────────────────┘
          │                 │                   │
┌─────────▼─────────┐ ┌─────▼───────┐ ┌────────▼──────────┐
│  Domain Services  │ │ Actor System │ │ Background Workers│
│  - GitHub Sync    │ │ - Graph Actor│ │ - Async Tasks     │
│  - Semantic       │ │ - GPU Actors │ │ - Job Queues      │
│  - Reasoning      │ │ - Settings   │ │ - Event Handlers  │
│  - Speech/Voice   │ │ - Workspace  │ │ - Sync Workers    │
└─────────┬─────────┘ └─────┬───────┘ └────────┬──────────┘
          │                 │                   │
┌─────────▼─────────────────▼───────────────────▼────────────────┐
│              Repository/Adapter Layer (Ports)                   │
│  - Neo4jAdapter (Knowledge Graph)                               │
│  - UnifiedOntologyRepository (SQLite)                           │
│  - ActorGraphRepository (In-Memory)                             │
│  - Neo4jSettingsRepository                                      │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                    Data Storage Layer                           │
│  - Neo4j Graph Database (Primary KG)                            │
│  - SQLite (Ontology OWL Classes)                                │
│  - In-Memory Actor State                                        │
└────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Hexagonal Architecture (Ports & Adapters)**
   - Core business logic isolated from infrastructure
   - Dependency inversion: adapters depend on ports
   - Testable through mock adapters

2. **CQRS (Command Query Responsibility Segregation)**
   - Commands for writes (Directives)
   - Queries for reads
   - Separate optimization strategies

3. **Actor Model (Actix)**
   - Concurrent message-passing
   - State isolation
   - Fault tolerance through supervision

4. **Event-Driven Architecture**
   - Domain events for cross-cutting concerns
   - Event bus for decoupled communication
   - Async event processing

5. **Repository Pattern**
   - Abstract data access
   - Single source of truth
   - Transaction management

---

## API Service Layer

### HTTP Server Configuration

**Framework:** Actix-Web 4.x
**Bind Address:** `0.0.0.0:4000` (configurable via `SYSTEM_NETWORK_PORT`)
**Workers:** 4 concurrent worker threads
**Middleware Stack:**

```rust
// src/main.rs (lines 373-385)
App::new()
    .wrap(middleware::Logger::default())          // Request logging
    .wrap(cors)                                    // CORS (allow all)
    .wrap(middleware::Compress::default())         // Response compression
    .wrap(TimeoutMiddleware::new(Duration::from_secs(30))) // 30s timeout
```

### Middleware Components

#### 1. TimeoutMiddleware

**Purpose:** Prevent request starvation
**Implementation:** `src/middleware/mod.rs`

```rust
pub struct TimeoutMiddleware {
    timeout: Duration,
}

impl TimeoutMiddleware {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}
```

**Behavior:**
- Aborts requests exceeding 30 seconds
- Returns HTTP 504 Gateway Timeout
- Prevents database connection exhaustion

#### 2. CORS Configuration

**Policy:** Permissive (development mode)

```rust
let cors = Cors::default()
    .allow_any_origin()      // Accept requests from any origin
    .allow_any_method()      // GET, POST, PUT, DELETE, PATCH, OPTIONS
    .allow_any_header()      // All headers allowed
    .max_age(3600)           // Preflight cache: 1 hour
    .supports_credentials(); // Allow cookies/auth headers
```

**Production Recommendation:** Restrict origins to specific domains.

#### 3. Compression Middleware

**Algorithm:** Automatic (Brotli, Gzip, Deflate)
**Threshold:** 1KB minimum response size
**Compression Level:** Default (Compress::default())

### REST API Structure

**Base Path:** `/api`
**Total Endpoints:** 52+ handler functions
**API Style:** RESTful with snake_case paths

#### Core API Scopes

```rust
// src/main.rs (lines 410-426)
web::scope("/api")
    // Settings management (new Neo4j-backed SettingsActor)
    .service(web::scope("/settings").configure(webxr::settings::api::configure_routes))

    // Main API handlers (graph, files, bots, analytics)
    .configure(api_handler::config)

    // Workspace management (file upload, organization)
    .configure(workspace_handler::config)

    // GitHub sync administration
    .configure(admin_sync_handler::configure_routes)

    // Static pages and templates
    .service(web::scope("/pages").configure(pages_handler::config))

    // Bot orchestration and visualization
    .service(web::scope("/bots").configure(api_handler::bots::config))
    .configure(bots_visualization_handler::configure_routes)

    // Graph data export (JSON, GraphML, GEXF)
    .configure(graph_export_handler::configure_routes)

    // Client-side error logging
    .route("/client-logs", web::post().to(client_log_handler::handle_client_logs))
```

### WebSocket Endpoints

**Total WebSocket Handlers:** 8 concurrent endpoints

```rust
// src/main.rs (lines 405-409)
// Primary graph visualization WebSocket (binary protocol)
.route("/wss", web::get().to(socket_flow_handler))

// Speech recognition and synthesis WebSocket
.route("/ws/speech", web::get().to(speech_socket_handler))

// MCP (Model Context Protocol) relay for AI agents
.route("/ws/mcp-relay", web::get().to(mcp_relay_handler))

// Client-to-server message bus WebSocket
.route("/ws/client-messages", web::get().to(websocket_client_messages))
```

**WebSocket Protocol Details:**

| Endpoint | Protocol | Compression | Heartbeat | Max Connections |
|----------|----------|-------------|-----------|-----------------|
| `/wss` | Binary (custom) | Delta + Gzip | 5s | Unlimited |
| `/ws/speech` | JSON | None | 10s | 100/client |
| `/ws/mcp-relay` | JSON | None | 30s | 10/client |
| `/ws/client-messages` | JSON | None | 15s | Unlimited |

### Request/Response Flow

#### Typical REST Request Flow

```
1. Client → HTTP Request → Actix-Web Server
2. Logger Middleware → Request logged
3. CORS Middleware → Origin validation
4. Compression Middleware → Accept-Encoding check
5. Timeout Middleware → Start timeout timer
6. Router → Match route path
7. Handler Function → Extract AppState, Query, Body
8. Handler → Send message to Actor/Service
9. Actor/Service → Process via CQRS (Command/Query)
10. Repository → Database query/mutation
11. Repository → Return result
12. Handler → Serialize response (JSON)
13. Compression Middleware → Compress response
14. Logger Middleware → Log response
15. Client ← HTTP Response
```

#### WebSocket Request Flow

```
1. Client → WebSocket Upgrade Request
2. Handler → Create WebSocket actor
3. Handler → Inject AppState into actor context
4. Actor → Start heartbeat timer
5. Actor → Subscribe to relevant event streams
6. Client ↔ Actor → Bidirectional messages
7. Actor → Process messages via message handlers
8. Actor → Forward to GraphServiceActor/SettingsActor
9. Actor → Receive updates from event bus
10. Actor → Send updates to client
11. Heartbeat → Check client alive every interval
12. Client disconnects → Actor stops → Cleanup
```

---

## Handler Architecture

### Handler Organization

**Total Handlers:** 52 Rust source files in `src/handlers/`

#### Handler Categories

```
src/handlers/
├── api_handler/                  # REST API handlers (modular)
│   ├── mod.rs                    # Main API config
│   ├── analytics/                # GPU analytics endpoints
│   │   ├── anomaly.rs            # Anomaly detection
│   │   ├── clustering.rs         # Community detection
│   │   ├── community.rs          # Graph clustering
│   │   └── websocket_integration.rs # Real-time analytics
│   ├── bots/                     # Bot orchestration
│   │   └── mod.rs                # Bot CRUD and commands
│   ├── constraints/              # Physics constraints
│   │   └── mod.rs                # Constraint management
│   ├── files/                    # File management
│   │   └── mod.rs                # Upload, list, process
│   ├── graph/                    # Graph data endpoints
│   │   └── mod.rs                # Graph CRUD, pagination
│   ├── ontology/                 # OWL ontology management
│   │   └── mod.rs                # Classes, properties, axioms
│   ├── quest3/                   # VR/XR Quest 3 integration
│   │   └── mod.rs                # XR session management
│   ├── settings/                 # Settings CRUD
│   │   └── mod.rs                # Settings persistence
│   └── visualisation/            # Visualization configs
│       └── mod.rs                # Rendering settings
├── admin_sync_handler.rs         # GitHub sync admin
├── bots_handler.rs               # Bot lifecycle
├── bots_visualization_handler.rs # Bot graph rendering
├── client_log_handler.rs         # Client error logs
├── client_messages_handler.rs    # Client message bus
├── clustering_handler.rs         # Graph clustering
├── consolidated_health_handler.rs # Health checks
├── constraints_handler.rs        # Physics constraints
├── cypher_query_handler.rs       # [DEPRECATED]
├── graph_export_handler.rs       # Export formats
├── graph_state_handler.rs        # Graph state sync
├── inference_handler.rs          # Reasoning inference
├── mcp_relay_handler.rs          # MCP WebSocket
├── multi_mcp_websocket_handler.rs # Multi-agent MCP
├── nostr_handler.rs              # Nostr protocol
├── ontology_handler.rs           # Ontology CRUD
├── pages_handler.rs              # Static pages
├── perplexity_handler.rs         # Perplexity AI
├── physics_handler.rs            # Physics simulation
├── pipeline_admin_handler.rs     # [DEPRECATED]
├── ragflow_handler.rs            # RAGFlow chat
├── realtime_websocket_handler.rs # Real-time updates
├── semantic_handler.rs           # Semantic search
├── settings_handler.rs           # [DEPRECATED - use settings::api]
├── settings_validation_fix.rs    # Settings validation
├── socket_flow_handler.rs        # Main graph WebSocket
├── speech_socket_handler.rs      # Speech WebSocket
├── utils.rs                      # Handler utilities
├── validation_handler.rs         # Input validation
├── websocket_settings_handler.rs # Settings WebSocket
├── websocket_utils.rs            # WebSocket utilities
└── workspace_handler.rs          # Workspace management
```

### Handler Patterns

#### 1. REST Handler Pattern (CQRS)

**Example:** `src/handlers/api_handler/graph/mod.rs`

```rust
use crate::application::graph::queries::GetGraphData;
use crate::handlers::utils::execute_in_thread;
use hexser::QueryHandler;

pub async fn get_graph_data(
    state: web::Data<AppState>,
    _req: HttpRequest,
) -> impl Responder {
    info!("Received request for graph data (CQRS Phase 1D)");

    // Clone query handler from AppState
    let graph_handler = state.graph_query_handlers.get_graph_data.clone();

    // Execute query in thread pool (blocking SQLite operations)
    let result = execute_in_thread(move || {
        graph_handler.handle(GetGraphData)
    }).await;

    match result {
        Ok(Ok(graph_data)) => {
            debug!("Returning {} nodes, {} edges",
                graph_data.nodes.len(), graph_data.edges.len());
            ok_json!(graph_data)
        }
        Ok(Err(e)) => {
            error!("CQRS query failed: {}", e);
            error_json!("Failed to retrieve graph data")
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            error_json!("Internal server error")
        }
    }
}
```

**Key Points:**
- Uses CQRS query handlers from `AppState`
- Executes blocking operations in thread pool
- Structured error handling with HTTP status codes
- Macro-based response helpers (`ok_json!`, `error_json!`)

#### 2. Actor Message Handler Pattern

**Example:** `src/handlers/settings_handler.rs` (legacy, but pattern still used)

```rust
use crate::actors::messages::UpdateSettings;

pub async fn update_settings(
    state: web::Data<AppState>,
    settings_json: web::Json<serde_json::Value>,
) -> impl Responder {
    info!("Updating settings via OptimizedSettingsActor");

    // Send message to settings actor
    let result = state.settings_addr
        .send(UpdateSettings {
            settings: settings_json.into_inner(),
        })
        .await;

    match result {
        Ok(Ok(())) => {
            info!("Settings updated successfully");
            ok_json!({"success": true, "message": "Settings updated"})
        }
        Ok(Err(e)) => {
            error!("Settings actor error: {}", e);
            error_json!("Failed to update settings")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed")
        }
    }
}
```

**Key Points:**
- Direct actor messaging via `Addr<T>.send()`
- Double Result unwrapping (`Ok(Ok())`)
- Actor errors vs mailbox errors

#### 3. WebSocket Handler Pattern

**Example:** `src/handlers/socket_flow_handler.rs`

```rust
use actix::prelude::*;
use actix_web_actors::ws;

pub struct GraphWebSocket {
    heartbeat: Instant,
    client_id: String,
    app_state: web::Data<AppState>,
}

impl Actor for GraphWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket client {} connected", self.client_id);

        // Start heartbeat
        self.start_heartbeat(ctx);

        // Subscribe to graph updates
        self.subscribe_to_graph_events(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("WebSocket client {} disconnected", self.client_id);
        self.app_state.decrement_connections();
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for GraphWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                // Handle JSON message
                self.handle_text_message(text, ctx);
            }
            Ok(ws::Message::Binary(bin)) => {
                // Handle binary protocol message
                self.handle_binary_message(bin, ctx);
            }
            Ok(ws::Message::Ping(msg)) => {
                self.heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.heartbeat = Instant::now();
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => {}
        }
    }
}
```

**Key Points:**
- Implements `Actor` trait for lifecycle management
- Implements `StreamHandler` for message processing
- Heartbeat mechanism for connection health
- Binary and text protocol support

### Response Macros

**Location:** `src/macros.rs` (implied from usage)

```rust
// Success responses
ok_json!(data)           // HTTP 200 with JSON body
created_json!(data)      // HTTP 201 Created
accepted!()              // HTTP 202 Accepted
no_content!()            // HTTP 204 No Content

// Error responses
bad_request!(msg)        // HTTP 400 Bad Request
unauthorized!(msg)       // HTTP 401 Unauthorized
forbidden!(msg)          // HTTP 403 Forbidden
not_found!(msg)          // HTTP 404 Not Found
conflict!(msg)           // HTTP 409 Conflict
error_json!(msg)         // HTTP 500 Internal Server Error
service_unavailable!()   // HTTP 503 Service Unavailable
payload_too_large!()     // HTTP 413 Payload Too Large
too_many_requests!()     // HTTP 429 Too Many Requests
```

---

## Worker System

### Background Task Architecture

The platform uses multiple strategies for background/async work:

1. **Tokio Tasks** - Short-lived async operations
2. **Actor Messages** - Long-running stateful operations
3. **Event Handlers** - Event-driven workflows
4. **GitHub Sync Worker** - Continuous sync

### 1. Tokio Spawn Tasks

**Used For:** Non-blocking I/O, HTTP requests, file operations

```rust
// src/app_state.rs (lines 219-263)
// GitHub sync spawned as background task during startup
let sync_handle = tokio::spawn(async move {
    info!("🔄 Background GitHub sync task spawned successfully");

    match sync_service_clone.sync_graphs().await {
        Ok(stats) => {
            info!("✅ GitHub sync complete! Duration: {:?}", stats.duration);
            info!("  📊 Total files: {}", stats.total_files);
            info!("  🔗 KG files: {}", stats.kg_files_processed);
            info!("  🏛️ Ontology files: {}", stats.ontology_files_processed);

            // Notify graph actor to reload synced data
            if let Some(graph_addr) = &*graph_service_addr_clone_for_sync.lock().await {
                graph_addr.do_send(ReloadGraphFromDatabase);
            }
        }
        Err(e) => {
            error!("❌ Background GitHub sync failed: {}", e);
        }
    }
});

// Monitor task with timeout
tokio::spawn(async move {
    let timeout_duration = Duration::from_secs(300); // 5 min timeout
    match tokio::time::timeout(timeout_duration, sync_handle).await {
        Ok(join_result) => match join_result {
            Ok(_) => info!("GitHub sync completed successfully"),
            Err(e) if e.is_cancelled() => error!("GitHub sync CANCELLED"),
            Err(e) if e.is_panic() => error!("GitHub sync PANICKED: {:?}", e),
            _ => error!("GitHub sync failed"),
        },
        Err(_) => error!("GitHub sync TIMED OUT after {:?}", timeout_duration),
    }
});
```

**Characteristics:**
- Non-blocking startup (server starts immediately)
- 5-minute timeout protection
- Panic recovery and logging
- State synchronization with actors

### 2. Actor-Based Workers

**Used For:** State management, scheduled tasks, supervision

#### WorkspaceActor

**Purpose:** Manage workspace files and organization
**Location:** `src/actors/workspace_actor.rs`

```rust
pub struct WorkspaceActor {
    workspace_path: PathBuf,
    file_cache: HashMap<String, CachedFile>,
}

// Message handlers
impl Handler<ListWorkspaceFiles> for WorkspaceActor {
    type Result = ResponseActFuture<Self, Result<Vec<FileInfo>, String>>;

    fn handle(&mut self, _msg: ListWorkspaceFiles, _ctx: &mut Context<Self>) -> Self::Result {
        let workspace_path = self.workspace_path.clone();

        Box::pin(
            async move {
                // Scan workspace directory
                let mut files = Vec::new();
                let entries = tokio::fs::read_dir(&workspace_path).await?;
                // ... process entries
                Ok(files)
            }
            .into_actor(self)
        )
    }
}
```

#### TaskOrchestratorActor

**Purpose:** Orchestrate multi-agent workflows via Management API
**Location:** `src/actors/task_orchestrator_actor.rs`

```rust
pub struct TaskOrchestratorActor {
    management_api_client: ManagementApiClient,
    active_tasks: HashMap<String, TaskState>,
}

impl Handler<CreateTaskWorkflow> for TaskOrchestratorActor {
    type Result = ResponseActFuture<Self, Result<String, String>>;

    fn handle(&mut self, msg: CreateTaskWorkflow, _ctx: &mut Context<Self>) -> Self::Result {
        let client = self.management_api_client.clone();

        Box::pin(
            async move {
                // Create task via Management API
                let task_id = client.create_task(msg.workflow).await?;

                // Poll for completion
                loop {
                    let status = client.get_task_status(&task_id).await?;
                    if status.is_complete() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }

                Ok(task_id)
            }
            .into_actor(self)
        )
    }
}
```

#### AgentMonitorActor

**Purpose:** Poll MCP (Model Context Protocol) agents
**Location:** `src/actors/agent_monitor_actor.rs`

```rust
pub struct AgentMonitorActor {
    claude_flow_client: ClaudeFlowClient,
    graph_service_addr: Addr<TransitionalGraphSupervisor>,
    poll_interval: Duration,
}

impl Actor for AgentMonitorActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Context<Self>) {
        info!("AgentMonitorActor started - polling MCP agents");

        // Schedule periodic polling
        ctx.run_interval(self.poll_interval, |act, _ctx| {
            act.poll_agents();
        });
    }
}

impl AgentMonitorActor {
    fn poll_agents(&mut self) {
        let client = self.claude_flow_client.clone();

        actix::spawn(async move {
            match client.get_swarm_status().await {
                Ok(status) => {
                    debug!("MCP agents: {} active, {} idle",
                        status.active_count, status.idle_count);
                }
                Err(e) => {
                    warn!("Failed to poll MCP agents: {}", e);
                }
            }
        });
    }
}
```

### 3. Event-Driven Workers

**Event Bus:** `src/events/bus.rs`
**Event Handlers:** `src/events/handlers/`

```
src/events/handlers/
├── audit_handler.rs          # Audit log writer
├── graph_handler.rs          # Graph event processor
├── notification_handler.rs   # User notifications
└── ontology_handler.rs       # Ontology sync
```

#### Event Flow

```rust
// src/events/bus.rs
pub struct EventBus {
    subscribers: HashMap<String, Vec<Box<dyn EventHandler>>>,
}

impl EventBus {
    pub async fn publish(&self, event: DomainEvent) {
        if let Some(handlers) = self.subscribers.get(&event.event_type) {
            for handler in handlers {
                tokio::spawn({
                    let handler = handler.clone();
                    let event = event.clone();
                    async move {
                        if let Err(e) = handler.handle(event).await {
                            error!("Event handler error: {}", e);
                        }
                    }
                });
            }
        }
    }
}
```

#### Example: GraphHandler

```rust
// src/events/handlers/graph_handler.rs
pub struct GraphEventHandler {
    graph_repository: Arc<ActorGraphRepository>,
}

#[async_trait]
impl EventHandler for GraphEventHandler {
    async fn handle(&self, event: DomainEvent) -> Result<(), String> {
        match event.event_type.as_str() {
            "node.created" => {
                // Trigger physics recalculation
                self.graph_repository.recalculate_forces().await?;
            }
            "node.deleted" => {
                // Clean up orphaned edges
                self.graph_repository.clean_orphaned_edges().await?;
            }
            "graph.synced" => {
                // Notify WebSocket clients
                // (broadcast via ClientCoordinatorActor)
            }
            _ => {}
        }
        Ok(())
    }
}
```

### 4. GitHub Sync Service

**Purpose:** Continuous synchronization of GitHub repositories
**Location:** `src/services/github_sync_service.rs`

#### Architecture

```rust
pub struct GitHubSyncService {
    content_api: Arc<EnhancedContentAPI>,
    neo4j_adapter: Arc<Neo4jAdapter>,
    ontology_repository: Arc<UnifiedOntologyRepository>,
    pipeline_service: Option<Arc<OntologyPipelineService>>,
}

impl GitHubSyncService {
    pub async fn sync_graphs(&self) -> Result<SyncStats, Box<dyn Error>> {
        let start = Instant::now();
        let mut stats = SyncStats::default();

        // 1. Fetch repository tree from GitHub
        let tree = self.content_api.get_repository_tree().await?;
        stats.total_files = tree.len();

        // 2. Process markdown files in parallel
        let mut handles = Vec::new();
        for file in tree.iter() {
            if file.path.ends_with(".md") {
                let handle = self.process_file(file.clone());
                handles.push(handle);
            }
        }

        // 3. Await all file processing tasks
        let results = futures::future::join_all(handles).await;

        for result in results {
            match result {
                Ok(FileType::KnowledgeGraph) => stats.kg_files_processed += 1,
                Ok(FileType::Ontology) => stats.ontology_files_processed += 1,
                Err(e) => stats.errors.push(e.to_string()),
            }
        }

        stats.duration = start.elapsed();
        Ok(stats)
    }

    async fn process_file(&self, file: GitHubFile) -> Result<FileType, Box<dyn Error>> {
        // 1. Download file content
        let content = self.content_api.get_file_content(&file.path).await?;

        // 2. Parse file type from frontmatter
        let file_type = self.detect_file_type(&content)?;

        // 3. Process based on type
        match file_type {
            FileType::KnowledgeGraph => {
                self.process_kg_file(&content).await?;
            }
            FileType::Ontology => {
                self.process_ontology_file(&content).await?;
            }
        }

        Ok(file_type)
    }
}
```

**Sync Process:**

1. **Fetch Repository Tree** (1 API call)
   - List all files in repository
   - Filter `.md` files

2. **Parallel File Processing** (N concurrent)
   - Download file content
   - Parse frontmatter and content
   - Detect file type (KG vs Ontology)

3. **Knowledge Graph Files**
   - Extract nodes and edges
   - Store in Neo4j via `Neo4jAdapter`
   - Trigger graph recalculation

4. **Ontology Files**
   - Extract OWL classes and properties
   - Store in SQLite via `UnifiedOntologyRepository`
   - Trigger reasoning inference

5. **Semantic Physics Pipeline**
   - Resolve OWL class IRIs to node IDs
   - Apply semantic forces
   - Update graph layout

---

## Repository Pattern

### Repository Abstraction

**Ports (Traits):** `src/ports/`

```
src/ports/
├── graph_repository.rs       # GraphRepository trait
├── ontology_repository.rs    # OntologyRepository trait
└── settings_repository.rs    # SettingsRepository trait
```

### Repository Implementations

#### 1. Neo4jAdapter (Primary Knowledge Graph)

**Location:** `src/adapters/neo4j_adapter.rs`
**Database:** Neo4j Graph Database
**Port:** `GraphRepository` (implied)

```rust
pub struct Neo4jAdapter {
    graph: Arc<Graph>,
    config: Neo4jConfig,
}

impl Neo4jAdapter {
    pub async fn new(config: Neo4jConfig) -> Result<Self, Neo4jError> {
        let graph = Arc::new(Graph::new(
            &config.uri,
            &config.user,
            &config.password,
        ).await?);

        Ok(Self { graph, config })
    }

    // Node operations
    pub async fn create_node(&self, node: &Node) -> Result<u32, Neo4jError> {
        let query = neo4rs::query(
            "CREATE (n:Node {
                metadata_id: $metadata_id,
                label: $label,
                position_x: $position_x,
                position_y: $position_y,
                position_z: $position_z,
                velocity_x: $velocity_x,
                velocity_y: $velocity_y,
                velocity_z: $velocity_z
            })
            RETURN id(n) as node_id"
        )
        .param("metadata_id", &node.metadata_id)
        .param("label", &node.label)
        .param("position_x", node.data.position.x)
        .param("position_y", node.data.position.y)
        .param("position_z", node.data.position.z)
        .param("velocity_x", node.data.velocity.x)
        .param("velocity_y", node.data.velocity.y)
        .param("velocity_z", node.data.velocity.z);

        let mut result = self.graph.execute(query).await?;
        let row = result.next().await?.ok_or(Neo4jError::NoResults)?;
        let node_id: i64 = row.get("node_id")?;

        Ok(node_id as u32)
    }

    pub async fn get_node(&self, node_id: u32) -> Result<Option<Node>, Neo4jError> {
        let query = neo4rs::query(
            "MATCH (n:Node) WHERE id(n) = $node_id RETURN n"
        )
        .param("node_id", node_id as i64);

        let mut result = self.graph.execute(query).await?;

        if let Some(row) = result.next().await? {
            let neo4j_node: neo4rs::Node = row.get("n")?;
            let node = self.neo4j_node_to_node(neo4j_node)?;
            Ok(Some(node))
        } else {
            Ok(None)
        }
    }

    // Batch operations for performance
    pub async fn batch_create_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>, Neo4jError> {
        let mut txn = self.graph.start_txn().await?;
        let mut node_ids = Vec::new();

        for node in nodes {
            let query = neo4rs::query(
                "CREATE (n:Node {
                    metadata_id: $metadata_id,
                    label: $label
                })
                RETURN id(n) as node_id"
            )
            .param("metadata_id", &node.metadata_id)
            .param("label", &node.label);

            let mut result = txn.execute(query).await?;
            let row = result.next().await?.ok_or(Neo4jError::NoResults)?;
            let node_id: i64 = row.get("node_id")?;
            node_ids.push(node_id as u32);
        }

        txn.commit().await?;
        Ok(node_ids)
    }

    // Edge operations
    pub async fn create_edge(&self, edge: &Edge) -> Result<u32, Neo4jError> {
        let query = neo4rs::query(
            "MATCH (source:Node), (target:Node)
             WHERE id(source) = $source_id AND id(target) = $target_id
             CREATE (source)-[r:LINKS_TO {
                 label: $label,
                 weight: $weight
             }]->(target)
             RETURN id(r) as edge_id"
        )
        .param("source_id", edge.source as i64)
        .param("target_id", edge.target as i64)
        .param("label", &edge.label)
        .param("weight", edge.weight.unwrap_or(1.0));

        let mut result = self.graph.execute(query).await?;
        let row = result.next().await?.ok_or(Neo4jError::NoResults)?;
        let edge_id: i64 = row.get("edge_id")?;

        Ok(edge_id as u32)
    }

    // Graph queries
    pub async fn get_all_nodes(&self) -> Result<Vec<Node>, Neo4jError> {
        let query = neo4rs::query("MATCH (n:Node) RETURN n");
        let mut result = self.graph.execute(query).await?;

        let mut nodes = Vec::new();
        while let Some(row) = result.next().await? {
            let neo4j_node: neo4rs::Node = row.get("n")?;
            let node = self.neo4j_node_to_node(neo4j_node)?;
            nodes.push(node);
        }

        Ok(nodes)
    }

    // Shortest path queries
    pub async fn compute_shortest_path(
        &self,
        source_id: u32,
        target_id: u32,
    ) -> Result<Vec<u32>, Neo4jError> {
        let query = neo4rs::query(
            "MATCH (source:Node), (target:Node),
             path = shortestPath((source)-[*]-(target))
             WHERE id(source) = $source_id AND id(target) = $target_id
             RETURN [node in nodes(path) | id(node)] as node_ids"
        )
        .param("source_id", source_id as i64)
        .param("target_id", target_id as i64);

        let mut result = self.graph.execute(query).await?;

        if let Some(row) = result.next().await? {
            let node_ids: Vec<i64> = row.get("node_ids")?;
            Ok(node_ids.into_iter().map(|id| id as u32).collect())
        } else {
            Ok(Vec::new())
        }
    }
}
```

**Performance Characteristics:**
- Connection pooling: 10 connections
- Query timeout: 30 seconds
- Batch inserts: 1000 nodes/transaction
- Index on `Node.metadata_id` for fast lookups

#### 2. UnifiedOntologyRepository (SQLite)

**Location:** `src/repositories/unified_ontology_repository.rs`
**Database:** SQLite (`data/unified.db`)
**Port:** `OntologyRepository`

```rust
pub struct UnifiedOntologyRepository {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl UnifiedOntologyRepository {
    pub fn new(db_path: &str) -> Result<Self, Box<dyn Error>> {
        let conn = rusqlite::Connection::open(db_path)?;

        // Create tables if not exist
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS owl_classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iri TEXT UNIQUE NOT NULL,
                label TEXT,
                description TEXT,
                parent_iri TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_owl_classes_iri ON owl_classes(iri);
            CREATE INDEX IF NOT EXISTS idx_owl_classes_parent ON owl_classes(parent_iri);

            CREATE TABLE IF NOT EXISTS owl_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iri TEXT UNIQUE NOT NULL,
                label TEXT,
                domain_iri TEXT,
                range_iri TEXT,
                property_type TEXT CHECK(property_type IN ('object', 'data', 'annotation')),
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_owl_properties_iri ON owl_properties(iri);"
        )?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn insert_owl_class(&self, class: &OwlClass) -> Result<i64, Box<dyn Error>> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO owl_classes (iri, label, description, parent_iri)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(iri) DO UPDATE SET
                label = excluded.label,
                description = excluded.description,
                parent_iri = excluded.parent_iri",
            rusqlite::params![
                &class.iri,
                &class.label,
                &class.description,
                &class.parent_iri,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    pub fn get_owl_class(&self, iri: &str) -> Result<Option<OwlClass>, Box<dyn Error>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT iri, label, description, parent_iri FROM owl_classes WHERE iri = ?1"
        )?;

        let class = stmt.query_row([iri], |row| {
            Ok(OwlClass {
                iri: row.get(0)?,
                label: row.get(1)?,
                description: row.get(2)?,
                parent_iri: row.get(3)?,
            })
        }).optional()?;

        Ok(class)
    }

    pub fn list_owl_classes(&self) -> Result<Vec<OwlClass>, Box<dyn Error>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT iri, label, description, parent_iri
             FROM owl_classes
             ORDER BY label"
        )?;

        let classes = stmt.query_map([], |row| {
            Ok(OwlClass {
                iri: row.get(0)?,
                label: row.get(1)?,
                description: row.get(2)?,
                parent_iri: row.get(3)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        Ok(classes)
    }

    // Graph serialization for visualization
    pub fn load_ontology_graph(&self) -> Result<Arc<GraphData>, Box<dyn Error>> {
        let conn = self.conn.lock().unwrap();

        // Load all classes as nodes
        let mut stmt = conn.prepare(
            "SELECT id, iri, label FROM owl_classes"
        )?;

        let nodes: Vec<Node> = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let iri: String = row.get(1)?;
            let label: String = row.get(2)?;

            Ok(Node {
                id: id as u32,
                metadata_id: iri.clone(),
                label,
                node_type: Some("owl:Class".to_string()),
                // ... other fields
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        // Load subclass relationships as edges
        let mut stmt = conn.prepare(
            "SELECT id, parent_iri FROM owl_classes WHERE parent_iri IS NOT NULL"
        )?;

        let edges: Vec<Edge> = stmt.query_map([], |row| {
            let child_id: i64 = row.get(0)?;
            let parent_iri: String = row.get(1)?;

            // Find parent node ID
            let parent_id = self.find_class_id_by_iri(&parent_iri)?;

            Ok(Edge {
                source: parent_id,
                target: child_id as u32,
                label: "rdfs:subClassOf".to_string(),
                // ... other fields
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        Ok(Arc::new(GraphData {
            nodes,
            edges,
            metadata: HashMap::new(),
        }))
    }
}
```

**Performance Characteristics:**
- Single-threaded (Mutex-protected)
- Query timeout: None (local file)
- Batch inserts: Not optimized (use transactions)
- Indexes on `iri` and `parent_iri`

#### 3. ActorGraphRepository (In-Memory)

**Location:** `src/adapters/actor_graph_repository.rs`
**Storage:** Actor mailbox (message-passing)
**Port:** `GraphRepository` (CQRS queries)

```rust
pub struct ActorGraphRepository {
    graph_actor: Addr<GraphServiceActor>,
}

impl ActorGraphRepository {
    pub fn new(graph_actor: Addr<GraphServiceActor>) -> Self {
        Self { graph_actor }
    }
}

#[async_trait]
impl GraphRepository for ActorGraphRepository {
    async fn get_graph_data(&self) -> Result<GraphData, String> {
        self.graph_actor
            .send(GetGraphData)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }

    async fn get_node_map(&self) -> Result<HashMap<u32, PhysicsNode>, String> {
        self.graph_actor
            .send(GetNodeMap)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }

    async fn get_physics_state(&self) -> Result<PhysicsState, String> {
        self.graph_actor
            .send(GetPhysicsState)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }
}
```

**Characteristics:**
- Zero-copy (actors share Arc references)
- Non-blocking (message-passing)
- Actor supervision and fault tolerance
- Bounded mailbox (100k messages)

#### 4. Neo4jSettingsRepository

**Location:** `src/adapters/neo4j_settings_repository.rs`
**Database:** Neo4j (settings stored as properties)
**Port:** `SettingsRepository`

```rust
pub struct Neo4jSettingsRepository {
    graph: Arc<Graph>,
}

impl Neo4jSettingsRepository {
    pub async fn new(config: Neo4jSettingsConfig) -> Result<Self, Box<dyn Error>> {
        let graph = Arc::new(Graph::new(
            &config.uri,
            &config.user,
            &config.password,
        ).await?);

        // Ensure settings node exists
        let query = neo4rs::query(
            "MERGE (s:Settings {id: 'app_settings'})
             RETURN s"
        );
        graph.execute(query).await?;

        Ok(Self { graph })
    }
}

#[async_trait]
impl SettingsRepository for Neo4jSettingsRepository {
    async fn load_all_settings(&self) -> Result<Option<AppFullSettings>, Box<dyn Error>> {
        let query = neo4rs::query(
            "MATCH (s:Settings {id: 'app_settings'})
             RETURN s.settings_json as settings"
        );

        let mut result = self.graph.execute(query).await?;

        if let Some(row) = result.next().await? {
            let settings_json: String = row.get("settings")?;
            let settings: AppFullSettings = serde_json::from_str(&settings_json)?;
            Ok(Some(settings))
        } else {
            Ok(None)
        }
    }

    async fn save_all_settings(&self, settings: &AppFullSettings) -> Result<(), Box<dyn Error>> {
        let settings_json = serde_json::to_string(settings)?;

        let query = neo4rs::query(
            "MATCH (s:Settings {id: 'app_settings'})
             SET s.settings_json = $settings_json,
                 s.updated_at = timestamp()"
        )
        .param("settings_json", settings_json);

        self.graph.execute(query).await?;
        Ok(())
    }

    async fn get_setting(&self, path: &str) -> Result<Option<serde_json::Value>, Box<dyn Error>> {
        // Load full settings and extract path
        let settings = self.load_all_settings().await?;

        if let Some(settings) = settings {
            let value = serde_json::to_value(settings)?;
            let result = self.extract_json_path(&value, path)?;
            Ok(result)
        } else {
            Ok(None)
        }
    }

    async fn update_setting(
        &self,
        path: &str,
        value: serde_json::Value,
    ) -> Result<(), Box<dyn Error>> {
        // Load, modify, save pattern
        let mut settings = self.load_all_settings().await?
            .unwrap_or_default();

        self.set_json_path(&mut settings, path, value)?;
        self.save_all_settings(&settings).await?;

        Ok(())
    }
}
```

**Performance Considerations:**
- Loads entire settings object for path queries (not optimal)
- Single settings node (no concurrency issues)
- JSON serialization overhead
- Network latency for each operation

---

## Service Layer

### Service Organization

**Total Services:** 42 domain services in `src/services/`

#### Service Categories

```
src/services/
├── Domain Services (Core Logic)
│   ├── github_sync_service.rs        # GitHub repository sync
│   ├── ontology_pipeline_service.rs  # Ontology processing
│   ├── semantic_analyzer.rs          # Semantic search
│   ├── edge_generation.rs            # Graph edge inference
│   ├── edge_classifier.rs            # Edge type classification
│   └── file_service.rs               # File management
│
├── External Integration Services
│   ├── github/                       # GitHub API client
│   │   ├── api.rs                    # REST API
│   │   ├── content_enhanced.rs       # Content fetching
│   │   ├── pr.rs                     # Pull requests
│   │   └── types.rs                  # Type definitions
│   ├── perplexity_service.rs         # Perplexity AI
│   ├── ragflow_service.rs            # RAGFlow chat
│   ├── nostr_service.rs              # Nostr protocol
│   ├── speech_service.rs             # Speech synthesis
│   └── bots_client.rs                # Bot orchestration
│
├── Ontology Services
│   ├── owl_validator.rs              # OWL validation
│   ├── ontology_converter.rs         # Format conversion
│   ├── ontology_reasoner.rs          # Logical reasoning
│   ├── ontology_reasoning_service.rs # Reasoning orchestration
│   └── ontology_enrichment_service.rs # Metadata enrichment
│
├── Visualization Services
│   ├── topology_visualization_engine.rs # Graph layout
│   ├── agent_visualization_processor.rs # Agent rendering
│   └── agent_visualization_protocol.rs  # Protocol handling
│
├── MCP (Model Context Protocol)
│   ├── mcp_relay_manager.rs          # MCP relay
│   ├── multi_mcp_agent_discovery.rs  # Agent discovery
│   └── real_mcp_integration_bridge.rs # Integration layer
│
├── Parser Services
│   └── parsers/
│       ├── knowledge_graph_parser.rs # KG markdown parsing
│       └── ontology_parser.rs        # OWL markdown parsing
│
└── Utility Services
    ├── graph_serialization.rs        # Graph format conversion
    ├── settings_watcher.rs           # Hot-reload watcher
    ├── settings_broadcast.rs         # Settings push
    ├── voice_context_manager.rs      # Voice context
    ├── voice_tag_manager.rs          # Voice tags
    └── management_api_client.rs      # Management API
```

### Key Services Deep Dive

#### 1. GitHubSyncService

**Purpose:** Synchronize GitHub repositories into knowledge graph
**Dependencies:**
- `EnhancedContentAPI` (GitHub content fetching)
- `Neo4jAdapter` (knowledge graph storage)
- `UnifiedOntologyRepository` (ontology storage)
- `OntologyPipelineService` (semantic processing)

**Public API:**

```rust
impl GitHubSyncService {
    // Create new service with dependencies
    pub fn new(
        content_api: Arc<EnhancedContentAPI>,
        neo4j_adapter: Arc<Neo4jAdapter>,
        ontology_repository: Arc<UnifiedOntologyRepository>,
    ) -> Self;

    // Set optional pipeline service for semantic physics
    pub fn set_pipeline_service(&mut self, pipeline: Arc<OntologyPipelineService>);

    // Main sync operation - synchronize all graphs
    pub async fn sync_graphs(&self) -> Result<SyncStats, Box<dyn Error>>;

    // Sync specific file path
    pub async fn sync_file(&self, path: &str) -> Result<(), Box<dyn Error>>;

    // Check if sync is needed (compares GitHub commit SHA)
    pub async fn check_sync_status(&self) -> Result<SyncStatus, Box<dyn Error>>;
}
```

**Sync Algorithm:**

```rust
pub async fn sync_graphs(&self) -> Result<SyncStats, Box<dyn Error>> {
    // 1. Fetch repository tree from GitHub
    let tree = self.content_api.get_repository_tree().await?;

    // 2. Filter markdown files
    let markdown_files: Vec<_> = tree.iter()
        .filter(|f| f.path.ends_with(".md"))
        .collect();

    // 3. Process files in parallel batches
    let chunks = markdown_files.chunks(10); // 10 files per batch

    for chunk in chunks {
        let handles: Vec<_> = chunk.iter()
            .map(|file| self.process_file(file.clone()))
            .collect();

        futures::future::join_all(handles).await;
    }

    // 4. Trigger semantic physics pipeline
    if let Some(pipeline) = &self.pipeline_service {
        pipeline.process_all_nodes().await?;
    }

    // 5. Return statistics
    Ok(stats)
}

async fn process_file(&self, file: GitHubFile) -> Result<FileType, Box<dyn Error>> {
    // 1. Download content
    let content = self.content_api.get_file_content(&file.path).await?;

    // 2. Parse frontmatter
    let (frontmatter, body) = self.parse_markdown(&content)?;

    // 3. Detect file type
    let file_type = frontmatter.get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("knowledge_graph");

    // 4. Process based on type
    match file_type {
        "knowledge_graph" => {
            // Parse as knowledge graph nodes/edges
            let parser = KnowledgeGraphParser::new();
            let (nodes, edges) = parser.parse(&content)?;

            // Store in Neo4j
            self.neo4j_adapter.batch_create_nodes(nodes).await?;
            self.neo4j_adapter.batch_create_edges(edges).await?;
        }
        "ontology" => {
            // Parse as OWL ontology
            let parser = OntologyParser::new();
            let (classes, properties) = parser.parse(&content)?;

            // Store in SQLite
            for class in classes {
                self.ontology_repository.insert_owl_class(&class)?;
            }
            for property in properties {
                self.ontology_repository.insert_owl_property(&property)?;
            }
        }
        _ => {
            warn!("Unknown file type: {}", file_type);
        }
    }

    Ok(file_type)
}
```

#### 2. OntologyPipelineService

**Purpose:** Apply semantic physics forces based on OWL ontology
**Dependencies:**
- `Neo4jAdapter` (graph repository for node lookups)

**Semantic Physics Concepts:**

- **Ontological Gravity:** Nodes of same OWL class attract
- **Semantic Repulsion:** Nodes of different classes repel
- **Hierarchical Layering:** Subclass relationships create vertical forces

**Public API:**

```rust
impl OntologyPipelineService {
    pub fn new(config: SemanticPhysicsConfig) -> Self;

    // Set graph repository for IRI → node ID resolution
    pub fn set_graph_repository(&mut self, repo: Arc<Neo4jAdapter>);

    // Process all nodes with semantic forces
    pub async fn process_all_nodes(&self) -> Result<(), Box<dyn Error>>;

    // Process single node
    pub async fn process_node(&self, node_id: u32) -> Result<SemanticForces, Box<dyn Error>>;

    // Compute semantic similarity between two nodes
    pub async fn compute_similarity(
        &self,
        node_a: u32,
        node_b: u32,
    ) -> Result<f32, Box<dyn Error>>;
}
```

**Processing Algorithm:**

```rust
pub async fn process_all_nodes(&self) -> Result<(), Box<dyn Error>> {
    let repo = self.graph_repository.as_ref()
        .ok_or("Graph repository not set")?;

    // 1. Load all nodes from Neo4j
    let nodes = repo.get_all_nodes().await?;

    // 2. Group nodes by OWL class IRI
    let mut class_groups: HashMap<String, Vec<u32>> = HashMap::new();

    for node in &nodes {
        if let Some(owl_class) = &node.owl_class_iri {
            class_groups.entry(owl_class.clone())
                .or_default()
                .push(node.id);
        }
    }

    // 3. Apply semantic forces
    for node in &nodes {
        let forces = self.compute_semantic_forces(node, &class_groups).await?;

        // Apply forces to node velocity
        repo.update_node_velocity(
            node.id,
            forces.total_force,
        ).await?;
    }

    Ok(())
}

async fn compute_semantic_forces(
    &self,
    node: &Node,
    class_groups: &HashMap<String, Vec<u32>>,
) -> Result<SemanticForces, Box<dyn Error>> {
    let mut forces = SemanticForces::default();

    if let Some(owl_class) = &node.owl_class_iri {
        // Attraction to same-class nodes
        if let Some(same_class_nodes) = class_groups.get(owl_class) {
            for &other_id in same_class_nodes {
                if other_id != node.id {
                    let attraction = self.compute_attraction(node.id, other_id).await?;
                    forces.add_force(attraction);
                }
            }
        }

        // Repulsion from different-class nodes
        for (other_class, other_nodes) in class_groups {
            if other_class != owl_class {
                for &other_id in other_nodes {
                    let repulsion = self.compute_repulsion(node.id, other_id).await?;
                    forces.add_force(repulsion);
                }
            }
        }

        // Hierarchical forces (subclass relationships)
        if let Some(parent_class) = self.get_parent_class(owl_class).await? {
            let parent_force = self.compute_hierarchical_force(node, &parent_class).await?;
            forces.add_force(parent_force);
        }
    }

    Ok(forces)
}

async fn compute_attraction(&self, node_a: u32, node_b: u32) -> Result<Vec3, Box<dyn Error>> {
    // Spring force: F = k * (distance - rest_length)
    let pos_a = self.get_node_position(node_a).await?;
    let pos_b = self.get_node_position(node_b).await?;

    let delta = pos_b - pos_a;
    let distance = delta.magnitude();

    if distance < 0.01 {
        return Ok(Vec3::zero());
    }

    let rest_length = self.config.same_class_distance;
    let spring_constant = self.config.attraction_strength;

    let force_magnitude = spring_constant * (distance - rest_length);
    let force_direction = delta / distance;

    Ok(force_direction * force_magnitude)
}
```

**Configuration:**

```rust
pub struct SemanticPhysicsConfig {
    pub same_class_distance: f32,      // Target distance for same-class nodes
    pub attraction_strength: f32,       // Spring constant for attraction
    pub repulsion_strength: f32,        // Coulomb constant for repulsion
    pub hierarchical_strength: f32,     // Force strength for subclass relationships
    pub damping_factor: f32,            // Velocity damping (0-1)
}

impl Default for SemanticPhysicsConfig {
    fn default() -> Self {
        Self {
            same_class_distance: 100.0,
            attraction_strength: 0.5,
            repulsion_strength: 1000.0,
            hierarchical_strength: 0.3,
            damping_factor: 0.95,
        }
    }
}
```

#### 3. RAGFlowService

**Purpose:** AI chat with RAG (Retrieval Augmented Generation)
**External API:** RAGFlow HTTP API
**Authentication:** API key

**Public API:**

```rust
impl RAGFlowService {
    pub async fn new(settings: Arc<RwLock<AppFullSettings>>) -> Result<Self, Box<dyn Error>>;

    // Start new chat session
    pub async fn create_session(&self) -> Result<String, Box<dyn Error>>;

    // Send message and get response
    pub async fn send_message(
        &self,
        session_id: &str,
        message: &str,
    ) -> Result<ChatResponse, Box<dyn Error>>;

    // Get conversation history
    pub async fn get_history(
        &self,
        session_id: &str,
    ) -> Result<Vec<ChatMessage>, Box<dyn Error>>;

    // Upload document to knowledge base
    pub async fn upload_document(
        &self,
        file_path: &str,
    ) -> Result<String, Box<dyn Error>>;
}
```

**Implementation:**

```rust
pub async fn send_message(
    &self,
    session_id: &str,
    message: &str,
) -> Result<ChatResponse, Box<dyn Error>> {
    let url = format!("{}/api/v1/chats/{}/completions",
        self.base_url, session_id);

    let body = serde_json::json!({
        "message": message,
        "stream": false,
        "conversation_id": session_id,
    });

    let response = self.client
        .post(&url)
        .header("Authorization", format!("Bearer {}", self.api_key))
        .json(&body)
        .timeout(Duration::from_secs(60))
        .send()
        .await?;

    if !response.status().is_success() {
        let error_body = response.text().await?;
        return Err(format!("RAGFlow API error: {}", error_body).into());
    }

    let response_json: serde_json::Value = response.json().await?;

    let chat_response = ChatResponse {
        answer: response_json["answer"].as_str()
            .ok_or("Missing answer field")?
            .to_string(),
        references: response_json["references"]
            .as_array()
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect())
            .unwrap_or_default(),
    };

    Ok(chat_response)
}
```

#### 4. SpeechService

**Purpose:** Text-to-speech synthesis with Kokoro TTS
**External API:** Kokoro HTTP API
**Features:** Multi-voice, real-time streaming

**Public API:**

```rust
impl SpeechService {
    pub fn new(settings: Arc<RwLock<AppFullSettings>>) -> Self;

    // Synthesize text to audio file
    pub async fn synthesize(
        &self,
        text: &str,
        voice: &str,
        output_path: &str,
    ) -> Result<(), Box<dyn Error>>;

    // Stream audio chunks for real-time playback
    pub async fn synthesize_stream(
        &self,
        text: &str,
        voice: &str,
    ) -> Result<impl Stream<Item = Result<Bytes, Box<dyn Error>>>, Box<dyn Error>>;

    // List available voices
    pub async fn list_voices(&self) -> Result<Vec<VoiceInfo>, Box<dyn Error>>;
}
```

**Streaming Implementation:**

```rust
pub async fn synthesize_stream(
    &self,
    text: &str,
    voice: &str,
) -> Result<impl Stream<Item = Result<Bytes, Box<dyn Error>>>, Box<dyn Error>> {
    let url = format!("{}/api/tts/stream", self.base_url);

    let body = serde_json::json!({
        "text": text,
        "voice": voice,
        "speed": 1.0,
        "format": "wav",
    });

    let response = self.client
        .post(&url)
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("Kokoro API error: {}", response.status()).into());
    }

    // Return byte stream
    let stream = response.bytes_stream()
        .map_err(|e| Box::new(e) as Box<dyn Error>);

    Ok(stream)
}
```

---

## Actor System

### Actor Architecture

**Framework:** Actix 0.13
**Actor Model:** Message-passing concurrency
**Total Actors:** 33 actor types

#### Actor Organization

```
src/actors/
├── Core System Actors
│   ├── graph_service_supervisor.rs      # Graph lifecycle supervisor
│   ├── graph_actor.rs                   # [Supervised by above]
│   ├── client_coordinator_actor.rs      # WebSocket client manager
│   ├── metadata_actor.rs                # Metadata cache
│   ├── optimized_settings_actor.rs      # Settings management
│   ├── workspace_actor.rs               # Workspace file manager
│   └── ontology_actor.rs                # Ontology operations
│
├── GPU Computation Actors (feature = "gpu")
│   └── gpu/
│       ├── gpu_manager_actor.rs         # GPU resource manager
│       ├── force_compute_actor.rs       # Physics force calculation
│       ├── stress_majorization_actor.rs # Graph layout optimization
│       ├── clustering_actor.rs          # Community detection
│       ├── anomaly_detection_actor.rs   # Anomaly detection
│       ├── constraint_actor.rs          # Constraint satisfaction
│       └── ontology_constraint_actor.rs # Ontology-based constraints
│
├── Monitoring Actors
│   ├── agent_monitor_actor.rs           # MCP agent polling
│   └── task_orchestrator_actor.rs       # Task workflow orchestration
│
├── Visualization Actors
│   ├── multi_mcp_visualization_actor.rs # Multi-agent visualization
│   └── physics_orchestrator_actor.rs    # Physics simulation coordination
│
└── Actor Utilities
    ├── mod.rs                           # Actor exports
    ├── messages.rs                      # Message definitions
    ├── graph_messages.rs                # Graph-specific messages
    ├── lifecycle.rs                     # Lifecycle utilities
    └── supervisor.rs                    # Supervisor patterns
```

### Actor Supervision

**Supervision Strategy:** One-for-one restart

```rust
// src/actors/graph_service_supervisor.rs
pub struct TransitionalGraphSupervisor {
    graph_actor: Option<Addr<GraphServiceActor>>,
    client_coordinator: Option<Addr<ClientCoordinatorActor>>,
    neo4j_adapter: Arc<Neo4jAdapter>,
}

impl Actor for TransitionalGraphSupervisor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Context<Self>) {
        info!("TransitionalGraphSupervisor started");

        // Spawn supervised GraphServiceActor
        let graph_actor = GraphServiceActor::new(
            self.neo4j_adapter.clone(),
        ).start();

        self.graph_actor = Some(graph_actor);

        // Start periodic health checks
        ctx.run_interval(Duration::from_secs(10), |act, _ctx| {
            act.check_health();
        });
    }
}

impl TransitionalGraphSupervisor {
    fn check_health(&mut self) {
        if let Some(ref graph_actor) = self.graph_actor {
            // Check if actor is alive
            if graph_actor.connected() {
                debug!("GraphServiceActor health: OK");
            } else {
                error!("GraphServiceActor health: DEAD - restarting");

                // Restart actor
                let new_actor = GraphServiceActor::new(
                    self.neo4j_adapter.clone(),
                ).start();

                self.graph_actor = Some(new_actor);

                // Notify client coordinator of new actor address
                if let Some(ref client_coord) = self.client_coordinator {
                    client_coord.do_send(UpdateGraphServiceAddress {
                        addr: new_actor,
                    });
                }
            }
        }
    }
}
```

### Key Actors Deep Dive

#### 1. GraphServiceActor

**Purpose:** Manage in-memory graph state and physics simulation
**Location:** `src/actors/graph_actor.rs` (not shown, but supervised)

**State:**

```rust
pub struct GraphServiceActor {
    // Graph data
    nodes: HashMap<u32, PhysicsNode>,
    edges: Vec<Edge>,
    metadata: HashMap<String, Metadata>,

    // Physics state
    simulation_running: bool,
    kinetic_energy: f32,
    stable_frame_count: u32,
    is_settled: bool,

    // Configuration
    simulation_params: SimulationParams,

    // Dependencies
    neo4j_adapter: Arc<Neo4jAdapter>,
    gpu_manager: Option<Addr<GPUManagerActor>>,
}
```

**Message Handlers:**

```rust
// Query messages (read-only)
impl Handler<GetGraphData> for GraphServiceActor {
    type Result = ResponseFuture<Result<GraphData, String>>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        let nodes: Vec<Node> = self.nodes.values()
            .map(|pn| pn.to_node())
            .collect();

        let graph_data = GraphData {
            nodes,
            edges: self.edges.clone(),
            metadata: self.metadata.clone(),
        };

        Box::pin(async move { Ok(graph_data) })
    }
}

impl Handler<GetNodeMap> for GraphServiceActor {
    type Result = Result<HashMap<u32, PhysicsNode>, String>;

    fn handle(&mut self, _msg: GetNodeMap, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(self.nodes.clone())
    }
}

impl Handler<GetPhysicsState> for GraphServiceActor {
    type Result = Result<PhysicsState, String>;

    fn handle(&mut self, _msg: GetPhysicsState, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(PhysicsState {
            is_settled: self.is_settled,
            stable_frame_count: self.stable_frame_count,
            kinetic_energy: self.kinetic_energy,
        })
    }
}

// Command messages (mutations)
impl Handler<UpdateNodePosition> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePosition, _ctx: &mut Context<Self>) -> Self::Result {
        if let Some(node) = self.nodes.get_mut(&msg.node_id) {
            node.data.set_position(msg.position);

            // Reset settlement state
            self.is_settled = false;
            self.stable_frame_count = 0;

            Ok(())
        } else {
            Err(format!("Node {} not found", msg.node_id))
        }
    }
}

impl Handler<ReloadGraphFromDatabase> for GraphServiceActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, _msg: ReloadGraphFromDatabase, _ctx: &mut Context<Self>) -> Self::Result {
        let neo4j = self.neo4j_adapter.clone();

        Box::pin(
            async move {
                // Load nodes from Neo4j
                let nodes = neo4j.get_all_nodes().await
                    .map_err(|e| format!("Failed to load nodes: {}", e))?;

                // Load edges from Neo4j
                let edges = neo4j.get_all_edges().await
                    .map_err(|e| format!("Failed to load edges: {}", e))?;

                Ok((nodes, edges))
            }
            .into_actor(self)
            .map(|result, act, _ctx| {
                match result {
                    Ok((nodes, edges)) => {
                        // Replace in-memory state
                        act.nodes = nodes.into_iter()
                            .map(|n| (n.id, PhysicsNode::from_node(n)))
                            .collect();
                        act.edges = edges;

                        info!("Reloaded {} nodes, {} edges from database",
                            act.nodes.len(), act.edges.len());
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })
        )
    }
}

// Simulation tick
impl Handler<SimulationTick> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, _msg: SimulationTick, ctx: &mut Context<Self>) {
        if !self.simulation_running {
            return;
        }

        // 1. Compute forces (CPU or GPU)
        if let Some(ref gpu_manager) = self.gpu_manager {
            // Use GPU for force computation
            self.compute_forces_gpu(gpu_manager, ctx);
        } else {
            // Fallback to CPU
            self.compute_forces_cpu();
        }

        // 2. Integrate velocities and positions
        self.integrate_physics();

        // 3. Check settlement
        self.check_settlement();

        // 4. Broadcast updates to WebSocket clients
        self.broadcast_updates();

        // 5. Schedule next tick
        ctx.run_later(Duration::from_millis(16), |_act, ctx| {
            ctx.address().do_send(SimulationTick);
        });
    }
}
```

**Physics Integration:**

```rust
fn integrate_physics(&mut self) {
    let dt = 0.016; // 16ms = 60 FPS

    for node in self.nodes.values_mut() {
        // Semi-implicit Euler integration
        // v(t+dt) = v(t) + a(t) * dt
        // p(t+dt) = p(t) + v(t+dt) * dt

        let acceleration = node.force / node.mass;
        node.data.velocity += acceleration * dt;

        // Apply damping
        node.data.velocity *= self.simulation_params.damping;

        // Update position
        node.data.position += node.data.velocity * dt;

        // Reset force accumulator
        node.force = Vec3::zero();
    }

    // Compute kinetic energy for settlement detection
    self.kinetic_energy = self.nodes.values()
        .map(|n| 0.5 * n.mass * n.data.velocity.magnitude_squared())
        .sum();
}

fn check_settlement(&mut self) {
    let threshold = self.simulation_params.settlement_threshold;

    if self.kinetic_energy < threshold {
        self.stable_frame_count += 1;

        if self.stable_frame_count >= 60 {
            // 1 second of stability at 60 FPS
            self.is_settled = true;
            self.simulation_running = false;
            info!("Graph simulation settled (KE: {})", self.kinetic_energy);
        }
    } else {
        self.stable_frame_count = 0;
        self.is_settled = false;
    }
}
```

#### 2. ClientCoordinatorActor

**Purpose:** Manage WebSocket clients and broadcast updates
**Location:** `src/actors/client_coordinator_actor.rs`

**State:**

```rust
pub struct ClientCoordinatorActor {
    clients: HashMap<String, Recipient<WebSocketMessage>>,
    graph_service_addr: Option<Addr<TransitionalGraphSupervisor>>,
    broadcast_interval: Duration,
}
```

**Message Handlers:**

```rust
impl Handler<RegisterClient> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: RegisterClient, _ctx: &mut Context<Self>) {
        info!("Registering WebSocket client: {}", msg.client_id);
        self.clients.insert(msg.client_id, msg.recipient);
    }
}

impl Handler<UnregisterClient> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: UnregisterClient, _ctx: &mut Context<Self>) {
        info!("Unregistering WebSocket client: {}", msg.client_id);
        self.clients.remove(&msg.client_id);
    }
}

impl Handler<BroadcastGraphUpdate> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: BroadcastGraphUpdate, _ctx: &mut Context<Self>) {
        let message = WebSocketMessage::GraphUpdate(msg.update);

        // Broadcast to all connected clients
        let mut disconnected_clients = Vec::new();

        for (client_id, recipient) in &self.clients {
            if recipient.do_send(message.clone()).is_err() {
                warn!("Client {} disconnected", client_id);
                disconnected_clients.push(client_id.clone());
            }
        }

        // Clean up disconnected clients
        for client_id in disconnected_clients {
            self.clients.remove(&client_id);
        }

        debug!("Broadcasted update to {} clients", self.clients.len());
    }
}

impl Handler<SetGraphServiceAddress> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: SetGraphServiceAddress, ctx: &mut Context<Self>) {
        info!("Updated graph service address for broadcast loop");
        self.graph_service_addr = Some(msg.addr);

        // Start broadcast loop if not already running
        if !ctx.state().is_some() {
            self.start_broadcast_loop(ctx);
        }
    }
}
```

**Broadcast Loop:**

```rust
impl ClientCoordinatorActor {
    fn start_broadcast_loop(&self, ctx: &mut Context<Self>) {
        ctx.run_interval(self.broadcast_interval, |act, ctx| {
            if let Some(ref graph_addr) = act.graph_service_addr {
                // Request graph update from graph service
                let graph_addr_clone = graph_addr.clone();
                let clients_count = act.clients.len();

                if clients_count == 0 {
                    return; // No clients to broadcast to
                }

                actix::spawn(async move {
                    match graph_addr_clone.send(GetGraphData).await {
                        Ok(Ok(graph_data)) => {
                            // Broadcast to clients via BroadcastGraphUpdate
                            ctx.address().do_send(BroadcastGraphUpdate {
                                update: GraphUpdate {
                                    nodes: graph_data.nodes,
                                    edges: graph_data.edges,
                                    timestamp: SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .unwrap()
                                        .as_millis() as u64,
                                },
                            });
                        }
                        Ok(Err(e)) => {
                            error!("Failed to get graph data: {}", e);
                        }
                        Err(e) => {
                            error!("Actor mailbox error: {}", e);
                        }
                    }
                });
            }
        });
    }
}
```

#### 3. OptimizedSettingsActor

**Purpose:** Manage application settings with Neo4j persistence
**Location:** `src/actors/optimized_settings_actor.rs`

**State:**

```rust
pub struct OptimizedSettingsActor {
    settings_repository: Arc<dyn SettingsRepository>,
    graph_service_addr: Option<Addr<TransitionalGraphSupervisor>>,
    cache: Option<AppFullSettings>,
}
```

**Message Handlers:**

```rust
impl Handler<GetSettings> for OptimizedSettingsActor {
    type Result = ResponseActFuture<Self, Result<AppFullSettings, String>>;

    fn handle(&mut self, _msg: GetSettings, _ctx: &mut Context<Self>) -> Self::Result {
        // Check cache first
        if let Some(ref cached) = self.cache {
            return Box::pin(future::ready(Ok(cached.clone())).into_actor(self));
        }

        // Load from repository
        let repo = self.settings_repository.clone();

        Box::pin(
            async move {
                repo.load_all_settings().await
                    .map_err(|e| format!("Failed to load settings: {}", e))?
                    .ok_or_else(|| "No settings found".to_string())
            }
            .into_actor(self)
            .map(|result, act, _ctx| {
                match result {
                    Ok(settings) => {
                        // Cache settings
                        act.cache = Some(settings.clone());
                        Ok(settings)
                    }
                    Err(e) => Err(e),
                }
            })
        )
    }
}

impl Handler<UpdateSettings> for OptimizedSettingsActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: UpdateSettings, _ctx: &mut Context<Self>) -> Self::Result {
        let repo = self.settings_repository.clone();
        let graph_addr = self.graph_service_addr.clone();

        Box::pin(
            async move {
                // Save to repository
                repo.save_all_settings(&msg.settings).await
                    .map_err(|e| format!("Failed to save settings: {}", e))?;

                // Update graph physics parameters if graph service available
                if let Some(graph_addr) = graph_addr {
                    let physics_settings = msg.settings.visualisation.graphs.logseq.physics.clone();
                    let params = SimulationParams::from(&physics_settings);

                    graph_addr.do_send(UpdateSimulationParams { params });
                }

                Ok(())
            }
            .into_actor(self)
            .map(|result, act, _ctx| {
                match result {
                    Ok(()) => {
                        // Update cache
                        act.cache = Some(msg.settings.clone());
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            })
        )
    }
}

impl Handler<InvalidateCache> for OptimizedSettingsActor {
    type Result = ();

    fn handle(&mut self, _msg: InvalidateCache, _ctx: &mut Context<Self>) {
        info!("Settings cache invalidated");
        self.cache = None;
    }
}
```

---

## CQRS Implementation

### CQRS Architecture

**Pattern:** Command Query Responsibility Segregation
**Library:** Custom implementation inspired by `hexser` crate

```
API Handler
    ↓
Application Service (CommandBus / QueryBus)
    ↓
Command/Query Handler
    ↓
Repository/Adapter
    ↓
Database
```

### Command Bus

**Location:** `src/cqrs/bus.rs`

```rust
pub struct CommandBus {
    handlers: HashMap<TypeId, Box<dyn Any>>,
}

impl CommandBus {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    pub fn register<C, H>(&mut self, handler: H)
    where
        C: Command + 'static,
        H: CommandHandler<C> + 'static,
    {
        let type_id = TypeId::of::<C>();
        self.handlers.insert(type_id, Box::new(handler));
    }

    pub async fn execute<C>(&self, command: C) -> Result<C::Output, Box<dyn Error>>
    where
        C: Command + 'static,
    {
        let type_id = TypeId::of::<C>();

        let handler = self.handlers.get(&type_id)
            .ok_or("No handler registered for command")?;

        let handler = handler.downcast_ref::<Box<dyn CommandHandler<C>>>()
            .ok_or("Handler type mismatch")?;

        handler.handle(command).await
    }
}
```

### Query Bus

**Location:** `src/cqrs/bus.rs`

```rust
pub struct QueryBus {
    handlers: HashMap<TypeId, Box<dyn Any>>,
}

impl QueryBus {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    pub fn register<Q, H>(&mut self, handler: H)
    where
        Q: Query + 'static,
        H: QueryHandler<Q> + 'static,
    {
        let type_id = TypeId::of::<Q>();
        self.handlers.insert(type_id, Box::new(handler));
    }

    pub async fn execute<Q>(&self, query: Q) -> Result<Q::Output, Box<dyn Error>>
    where
        Q: Query + 'static,
    {
        let type_id = TypeId::of::<Q>();

        let handler = self.handlers.get(&type_id)
            .ok_or("No handler registered for query")?;

        let handler = handler.downcast_ref::<Box<dyn QueryHandler<Q>>>()
            .ok_or("Handler type mismatch")?;

        handler.handle(query).await
    }
}
```

### Example: Settings Domain

#### Queries

```rust
// src/application/settings/queries.rs

#[derive(Debug, Clone)]
pub struct LoadAllSettings;

#[derive(Debug, Clone)]
pub struct GetSetting {
    pub path: String,
}

pub struct LoadAllSettingsHandler {
    repository: Arc<dyn SettingsRepository>,
}

impl LoadAllSettingsHandler {
    pub fn new(repository: Arc<dyn SettingsRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl QueryHandler for LoadAllSettingsHandler {
    type Query = LoadAllSettings;
    type Output = Option<AppFullSettings>;

    async fn handle(&self, _query: Self::Query) -> Result<Self::Output, Box<dyn Error>> {
        self.repository.load_all_settings().await
    }
}

pub struct GetSettingHandler {
    repository: Arc<dyn SettingsRepository>,
}

#[async_trait]
impl QueryHandler for GetSettingHandler {
    type Query = GetSetting;
    type Output = Option<serde_json::Value>;

    async fn handle(&self, query: Self::Query) -> Result<Self::Output, Box<dyn Error>> {
        self.repository.get_setting(&query.path).await
    }
}
```

#### Directives (Commands)

```rust
// src/application/settings/directives.rs

#[derive(Debug, Clone)]
pub struct SaveAllSettings {
    pub settings: AppFullSettings,
}

#[derive(Debug, Clone)]
pub struct UpdateSetting {
    pub path: String,
    pub value: serde_json::Value,
}

pub struct SaveAllSettingsHandler {
    repository: Arc<dyn SettingsRepository>,
}

#[async_trait]
impl DirectiveHandler for SaveAllSettingsHandler {
    type Directive = SaveAllSettings;
    type Output = ();

    async fn handle(&self, directive: Self::Directive) -> Result<Self::Output, Box<dyn Error>> {
        self.repository.save_all_settings(&directive.settings).await
    }
}

pub struct UpdateSettingHandler {
    repository: Arc<dyn SettingsRepository>,
}

#[async_trait]
impl DirectiveHandler for UpdateSettingHandler {
    type Directive = UpdateSetting;
    type Output = ();

    async fn handle(&self, directive: Self::Directive) -> Result<Self::Output, Box<dyn Error>> {
        self.repository.update_setting(&directive.path, directive.value).await
    }
}
```

#### Application Service

```rust
// src/application/services.rs

pub struct SettingsApplicationService {
    command_bus: Arc<RwLock<CommandBus>>,
    query_bus: Arc<RwLock<QueryBus>>,
    event_bus: Arc<RwLock<EventBus>>,
}

impl SettingsApplicationService {
    pub fn new(
        command_bus: Arc<RwLock<CommandBus>>,
        query_bus: Arc<RwLock<QueryBus>>,
        event_bus: Arc<RwLock<EventBus>>,
    ) -> Self {
        Self {
            command_bus,
            query_bus,
            event_bus,
        }
    }

    pub async fn load_settings(&self) -> Result<Option<AppFullSettings>, Box<dyn Error>> {
        let query_bus = self.query_bus.read().await;
        query_bus.execute(LoadAllSettings).await
    }

    pub async fn save_settings(&self, settings: AppFullSettings) -> Result<(), Box<dyn Error>> {
        let command_bus = self.command_bus.read().await;
        command_bus.execute(SaveAllSettings { settings: settings.clone() }).await?;

        // Publish event
        let event_bus = self.event_bus.read().await;
        event_bus.publish(DomainEvent {
            event_type: "settings.updated".to_string(),
            payload: serde_json::to_value(&settings)?,
            timestamp: SystemTime::now(),
        }).await;

        Ok(())
    }
}
```

---

## Dependency Injection

### AppState Structure

**Location:** `src/app_state.rs`

```rust
#[derive(Clone)]
pub struct AppState {
    // Actors
    pub graph_service_addr: Addr<TransitionalGraphSupervisor>,
    pub settings_addr: Addr<OptimizedSettingsActor>,
    pub metadata_addr: Addr<MetadataActor>,
    pub client_manager_addr: Addr<ClientCoordinatorActor>,
    pub workspace_addr: Addr<WorkspaceActor>,
    pub task_orchestrator_addr: Addr<TaskOrchestratorActor>,
    pub agent_monitor_addr: Addr<AgentMonitorActor>,
    pub ontology_actor_addr: Option<Addr<OntologyActor>>,

    // GPU actors (conditional compilation)
    #[cfg(feature = "gpu")]
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>,

    // Repositories
    pub neo4j_adapter: Arc<Neo4jAdapter>,
    pub ontology_repository: Arc<UnifiedOntologyRepository>,
    pub settings_repository: Arc<dyn SettingsRepository>,
    pub graph_repository: Arc<ActorGraphRepository>,

    // CQRS handlers
    pub graph_query_handlers: GraphQueryHandlers,

    // CQRS buses
    pub command_bus: Arc<RwLock<CommandBus>>,
    pub query_bus: Arc<RwLock<QueryBus>>,
    pub event_bus: Arc<RwLock<EventBus>>,

    // Application services
    pub app_services: ApplicationServices,

    // External services
    pub github_client: Arc<GitHubClient>,
    pub content_api: Arc<ContentAPI>,
    pub perplexity_service: Option<Arc<PerplexityService>>,
    pub ragflow_service: Option<Arc<RAGFlowService>>,
    pub speech_service: Option<Arc<SpeechService>>,
    pub nostr_service: Option<web::Data<NostrService>>,
    pub bots_client: Arc<BotsClient>,

    // Feature flags
    pub feature_access: web::Data<FeatureAccess>,

    // State
    pub ragflow_session_id: String,
    pub active_connections: Arc<AtomicUsize>,
    pub debug_enabled: bool,

    // Message channels
    pub client_message_tx: mpsc::UnboundedSender<ClientMessage>,
    pub client_message_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<ClientMessage>>>,

    // Pipeline service
    pub ontology_pipeline_service: Option<Arc<OntologyPipelineService>>,
}
```

### Initialization Flow

```rust
// src/app_state.rs (lines 132-628)
impl AppState {
    pub async fn new(
        settings: AppFullSettings,
        github_client: Arc<GitHubClient>,
        content_api: Arc<ContentAPI>,
        perplexity_service: Option<Arc<PerplexityService>>,
        ragflow_service: Option<Arc<RAGFlowService>>,
        speech_service: Option<Arc<SpeechService>>,
        ragflow_session_id: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 1. Initialize repositories
        let settings_repository = Arc::new(
            Neo4jSettingsRepository::new(Neo4jSettingsConfig::default()).await?
        );

        let ontology_repository = Arc::new(
            tokio::task::spawn_blocking(||
                UnifiedOntologyRepository::new("data/unified.db")
            ).await??
        );

        let neo4j_adapter = Arc::new(
            Neo4jAdapter::new(Neo4jConfig::default()).await?
        );

        // 2. Initialize pipeline service
        let mut pipeline_service = OntologyPipelineService::new(
            SemanticPhysicsConfig::default()
        );
        pipeline_service.set_graph_repository(neo4j_adapter.clone());
        let ontology_pipeline_service = Some(Arc::new(pipeline_service));

        // 3. Initialize GitHub sync service
        let enhanced_content_api = Arc::new(EnhancedContentAPI::new(github_client.clone()));
        let mut github_sync_service = GitHubSyncService::new(
            enhanced_content_api,
            neo4j_adapter.clone(),
            ontology_repository.clone(),
        );
        github_sync_service.set_pipeline_service(
            ontology_pipeline_service.clone().unwrap()
        );

        // 4. Start background GitHub sync
        tokio::spawn(async move {
            match github_sync_service.sync_graphs().await {
                Ok(stats) => info!("GitHub sync complete: {:?}", stats),
                Err(e) => error!("GitHub sync failed: {}", e),
            }
        });

        // 5. Initialize actors
        let client_manager_addr = ClientCoordinatorActor::new().start();
        let metadata_addr = MetadataActor::new(MetadataStore::new()).start();

        let graph_service_addr = TransitionalGraphSupervisor::new(
            Some(client_manager_addr.clone()),
            None,
            neo4j_adapter.clone(),
        ).start();

        // 6. Initialize CQRS components
        let graph_actor_addr = graph_service_addr
            .send(GetGraphServiceActor)
            .await?
            .ok_or("GraphServiceActor not initialized")?;

        let graph_repository = Arc::new(ActorGraphRepository::new(graph_actor_addr));

        let graph_query_handlers = GraphQueryHandlers {
            get_graph_data: Arc::new(GetGraphDataHandler::new(graph_repository.clone())),
            get_node_map: Arc::new(GetNodeMapHandler::new(graph_repository.clone())),
            // ... other handlers
        };

        let command_bus = Arc::new(RwLock::new(CommandBus::new()));
        let query_bus = Arc::new(RwLock::new(QueryBus::new()));
        let event_bus = Arc::new(RwLock::new(EventBus::new()));

        let app_services = ApplicationServices {
            graph: GraphApplicationService::new(
                command_bus.clone(),
                query_bus.clone(),
                event_bus.clone(),
            ),
            // ... other services
        };

        // 7. Initialize remaining actors
        let settings_actor = OptimizedSettingsActor::with_actors(
            settings_repository.clone(),
            Some(graph_service_addr.clone()),
            None,
        )?;
        let settings_addr = settings_actor.start();

        let workspace_addr = WorkspaceActor::new().start();
        let task_orchestrator_addr = TaskOrchestratorActor::new(mgmt_client).start();
        let agent_monitor_addr = AgentMonitorActor::new(
            claude_flow_client,
            graph_service_addr.clone(),
        ).start();

        // 8. Initialize message channels
        let (client_message_tx, client_message_rx) = mpsc::unbounded_channel();

        Ok(Self {
            graph_service_addr,
            settings_addr,
            metadata_addr,
            client_manager_addr,
            workspace_addr,
            task_orchestrator_addr,
            agent_monitor_addr,
            ontology_actor_addr: None,
            neo4j_adapter,
            ontology_repository,
            settings_repository,
            graph_repository,
            graph_query_handlers,
            command_bus,
            query_bus,
            event_bus,
            app_services,
            github_client,
            content_api,
            perplexity_service,
            ragflow_service,
            speech_service,
            nostr_service: None,
            feature_access: web::Data::new(FeatureAccess::from_env()),
            ragflow_session_id,
            active_connections: Arc::new(AtomicUsize::new(0)),
            bots_client: Arc::new(BotsClient::with_graph_service(graph_service_addr.clone())),
            debug_enabled: crate::utils::logging::is_debug_enabled(),
            client_message_tx,
            client_message_rx: Arc::new(tokio::sync::Mutex::new(client_message_rx)),
            ontology_pipeline_service,
            #[cfg(feature = "gpu")]
            gpu_manager_addr: None,
        })
    }
}
```

### Dependency Injection into Handlers

```rust
// src/main.rs (lines 387-402)
.app_data(settings_data.clone())
.app_data(web::Data::new(github_client.clone()))
.app_data(web::Data::new(content_api.clone()))
.app_data(app_state_data.clone())
.app_data(pre_read_ws_settings_data.clone())
.app_data(web::Data::new(app_state_data.graph_service_addr.clone()))
.app_data(web::Data::new(app_state_data.settings_addr.clone()))
.app_data(web::Data::new(app_state_data.metadata_addr.clone()))
.app_data(web::Data::new(app_state_data.client_manager_addr.clone()))
.app_data(web::Data::new(app_state_data.workspace_addr.clone()))
.app_data(app_state_data.nostr_service.clone().unwrap_or_default())
.app_data(app_state_data.feature_access.clone())
.app_data(web::Data::new(github_sync_service.clone()))
.app_data(settings_actor_data.clone())
```

**Handler Extraction:**

```rust
pub async fn handler(
    state: web::Data<AppState>,                    // Main app state
    github_client: web::Data<GitHubClient>,        // Specific service
    settings_addr: web::Data<Addr<SettingsActor>>, // Specific actor
    query: web::Query<QueryParams>,                // Query parameters
    body: web::Json<RequestBody>,                  // JSON body
    req: HttpRequest,                              // Raw request
) -> impl Responder {
    // Handler logic
}
```

---

## Data Flow Diagrams

### 1. Graph Data Query Flow

```
Client
  │
  │ GET /api/graph/data
  │
  ▼
HTTP Handler (get_graph_data)
  │
  │ Extract AppState
  │
  ▼
CQRS Query Handler
  │
  │ GetGraphData query
  │
  ▼
ActorGraphRepository
  │
  │ Send GetGraphData message
  │
  ▼
GraphServiceActor
  │
  │ Return in-memory graph data
  │
  ▼
Query Handler
  │
  │ Serialize to JSON
  │
  ▼
HTTP Handler
  │
  │ ok_json!(data)
  │
  ▼
Client
```

### 2. Settings Update Flow

```
Client
  │
  │ POST /api/settings
  │ { path: "physics.damping", value: 0.95 }
  │
  ▼
HTTP Handler (update_setting)
  │
  │ Extract AppState
  │
  ▼
SettingsApplicationService
  │
  │ UpdateSetting directive
  │
  ▼
CommandBus
  │
  │ Execute handler
  │
  ▼
UpdateSettingHandler
  │
  │ Repository update
  │
  ▼
Neo4jSettingsRepository
  │
  │ Load full settings
  │ Modify path
  │ Save full settings
  │
  ▼
Neo4j Database
  │
  │ MATCH (s:Settings {id: 'app_settings'})
  │ SET s.settings_json = $json
  │
  ▼
CommandBus
  │
  │ Publish SettingsUpdated event
  │
  ▼
EventBus
  │
  │ Notify subscribers
  │
  ▼
OptimizedSettingsActor
  │
  │ Invalidate cache
  │
  ▼
GraphServiceActor
  │
  │ Apply new physics parameters
  │
  ▼
Client (Success Response)
```

### 3. GitHub Sync Flow

```
Server Startup
  │
  │ AppState::new()
  │
  ▼
Initialize GitHubSyncService
  │
  │ Dependencies injected
  │
  ▼
Spawn Background Task
  │
  │ tokio::spawn(sync_graphs())
  │
  ▼
Fetch GitHub Repository Tree
  │
  │ GitHub API call
  │
  ▼
Filter Markdown Files
  │
  │ *.md files only
  │
  ▼
Parallel File Processing
  │
  ├─────┬─────┬─────┐
  │     │     │     │
  ▼     ▼     ▼     ▼
File1 File2 File3 ... FileN
  │     │     │     │
  │ Download content
  │ Parse frontmatter
  │ Detect type (KG vs Ontology)
  │
  ├─────┴─────┴─────┘
  │
  ▼
Knowledge Graph Files
  │
  │ Parse nodes and edges
  │
  ▼
Neo4jAdapter.batch_create_nodes()
Neo4jAdapter.batch_create_edges()
  │
  │ Cypher queries
  │
  ▼
Neo4j Database
  │
  │
  ▼
Ontology Files
  │
  │ Parse OWL classes
  │
  ▼
UnifiedOntologyRepository.insert_owl_class()
  │
  │ SQLite INSERT
  │
  ▼
SQLite Database (unified.db)
  │
  │
  ▼
Semantic Physics Pipeline
  │
  │ OntologyPipelineService.process_all_nodes()
  │
  ▼
Apply Semantic Forces
  │
  │ Attraction/repulsion based on OWL classes
  │
  ▼
Update Node Velocities in Neo4j
  │
  │
  ▼
Notify GraphServiceActor
  │
  │ ReloadGraphFromDatabase message
  │
  ▼
GraphServiceActor
  │
  │ Reload nodes/edges from Neo4j
  │ Reset physics simulation
  │
  ▼
Broadcast to WebSocket Clients
  │
  │
  ▼
Clients Receive Updated Graph
```

### 4. WebSocket Real-Time Update Flow

```
Client
  │
  │ WebSocket connection: /wss
  │
  ▼
socket_flow_handler
  │
  │ Create GraphWebSocket actor
  │
  ▼
GraphWebSocket Actor Started
  │
  │ Register with ClientCoordinatorActor
  │
  ▼
ClientCoordinatorActor
  │
  │ Add client to broadcast list
  │ Start broadcast loop (if not running)
  │
  ▼
Broadcast Loop (every 100ms)
  │
  │ Check if clients exist
  │
  ▼
Request Graph Data
  │
  │ Send GetGraphData to GraphServiceActor
  │
  ▼
GraphServiceActor
  │
  │ Return current graph state
  │
  ▼
ClientCoordinatorActor
  │
  │ BroadcastGraphUpdate message
  │
  ▼
Iterate All Clients
  │
  ├─────┬─────┬─────┐
  │     │     │     │
  ▼     ▼     ▼     ▼
Client1 Client2 Client3 ... ClientN
  │     │     │     │
  │ Send WebSocketMessage::GraphUpdate
  │
  │ Serialize to binary protocol
  │ Apply delta compression
  │
  ▼
WebSocket Transport
  │
  │
  ▼
Client Browser
  │
  │ Decode binary message
  │ Apply delta to local state
  │ Render updated graph
```

---

## Performance Characteristics

### Request Latency

| Endpoint | P50 | P95 | P99 | Notes |
|----------|-----|-----|-----|-------|
| GET /api/health | 1ms | 2ms | 5ms | No I/O |
| GET /api/config | 10ms | 50ms | 100ms | Neo4j settings query |
| GET /api/graph/data | 50ms | 200ms | 500ms | In-memory actor query |
| GET /api/graph/data/paginated | 30ms | 150ms | 300ms | Pagination reduces size |
| POST /api/graph/update | 2s | 5s | 10s | GitHub fetch + processing |
| GET /api/ontology/classes | 20ms | 100ms | 200ms | SQLite query |
| POST /api/settings | 100ms | 300ms | 600ms | Neo4j write + cache invalidation |

### Throughput

| Scenario | Requests/sec | Concurrent Users | Bottleneck |
|----------|--------------|------------------|------------|
| Health checks | 10,000+ | N/A | CPU bound |
| Graph data queries | 500 | 100 | Actor mailbox |
| Settings queries | 200 | 50 | Neo4j connections |
| Graph updates | 10 | 5 | GitHub API rate limit |
| WebSocket broadcasts | 100 clients | 100 | Network bandwidth |

### Memory Usage

| Component | Baseline | Per Client | Peak | Notes |
|-----------|----------|------------|------|-------|
| App State | 50 MB | - | - | Initialized once |
| GraphServiceActor | 100 MB | - | 500 MB | Depends on graph size |
| WebSocket Client | - | 1 MB | 5 MB | Buffered messages |
| Neo4j Connection Pool | 20 MB | - | 50 MB | 10 connections |
| SQLite Repository | 5 MB | - | 20 MB | Cached queries |
| Total (100 clients) | 200 MB | 1 MB | 700 MB | |

### Database Performance

#### Neo4j Queries

```cypher
-- Simple node fetch (1-5ms)
MATCH (n:Node) WHERE id(n) = $node_id RETURN n

-- All nodes query (10-50ms for 1000 nodes)
MATCH (n:Node) RETURN n

-- Shortest path (5-100ms depending on graph size)
MATCH (source:Node), (target:Node),
      path = shortestPath((source)-[*]-(target))
WHERE id(source) = $source_id AND id(target) = $target_id
RETURN path

-- Batch insert (100ms for 1000 nodes in single transaction)
UNWIND $nodes AS node
CREATE (n:Node {
    metadata_id: node.metadata_id,
    label: node.label,
    position_x: node.position.x,
    position_y: node.position.y,
    position_z: node.position.z
})
```

**Optimization Strategies:**
- Index on `Node.metadata_id`
- Connection pooling (10 connections)
- Batch inserts (1000 per transaction)
- Query result caching in actors

#### SQLite Queries

```sql
-- Class lookup (1-5ms)
SELECT * FROM owl_classes WHERE iri = ?

-- All classes (5-20ms for 500 classes)
SELECT * FROM owl_classes ORDER BY label

-- Insert with conflict resolution (1-10ms)
INSERT INTO owl_classes (iri, label, description)
VALUES (?, ?, ?)
ON CONFLICT(iri) DO UPDATE SET
    label = excluded.label,
    description = excluded.description
```

**Optimization Strategies:**
- Index on `iri` (UNIQUE)
- Index on `parent_iri` (foreign key)
- Single-threaded (Mutex) to avoid WAL conflicts
- Use transactions for batch inserts

### Actor Message Throughput

| Actor | Messages/sec | Mailbox Capacity | Latency (avg) |
|-------|--------------|------------------|---------------|
| GraphServiceActor | 1000 | 100k | 1ms |
| ClientCoordinatorActor | 500 | 10k | 2ms |
| SettingsActor | 200 | 1k | 5ms |
| TaskOrchestratorActor | 50 | 1k | 100ms |

**Bottlenecks:**
- GraphServiceActor: Physics computation (CPU-bound)
- ClientCoordinatorActor: WebSocket send operations (I/O-bound)
- SettingsActor: Neo4j database writes (Network-bound)

### Concurrency

**Actix-Web Workers:** 4 threads
**Tokio Runtime:** Default (CPU cores)
**Neo4j Connection Pool:** 10 connections
**Actor Mailboxes:** Bounded (prevents memory exhaustion)

**Concurrency Model:**
```
[HTTP Request] → [Actix Worker 1] → [Tokio Runtime] → [Actor]
[HTTP Request] → [Actix Worker 2] → [Tokio Runtime] → [Actor]
[HTTP Request] → [Actix Worker 3] → [Tokio Runtime] → [Actor]
[HTTP Request] → [Actix Worker 4] → [Tokio Runtime] → [Actor]
```

---

## Code Examples

### Example 1: Creating a New REST Endpoint

**Requirement:** Add endpoint to get node count

**1. Define Query:**

```rust
// src/application/graph/queries.rs
#[derive(Debug, Clone)]
pub struct GetNodeCount;

pub struct GetNodeCountHandler {
    repository: Arc<ActorGraphRepository>,
}

impl GetNodeCountHandler {
    pub fn new(repository: Arc<ActorGraphRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl QueryHandler for GetNodeCountHandler {
    type Query = GetNodeCount;
    type Output = usize;

    async fn handle(&self, _query: Self::Query) -> Result<Self::Output, Box<dyn Error>> {
        let graph_data = self.repository.get_graph_data().await?;
        Ok(graph_data.nodes.len())
    }
}
```

**2. Add Handler to AppState:**

```rust
// src/app_state.rs
pub struct GraphQueryHandlers {
    // ... existing handlers
    pub get_node_count: Arc<GetNodeCountHandler>,
}

// In AppState::new():
let graph_query_handlers = GraphQueryHandlers {
    // ... existing
    get_node_count: Arc::new(GetNodeCountHandler::new(graph_repository.clone())),
};
```

**3. Create HTTP Handler:**

```rust
// src/handlers/api_handler/graph/mod.rs
pub async fn get_node_count(state: web::Data<AppState>) -> impl Responder {
    info!("Fetching node count");

    let handler = state.graph_query_handlers.get_node_count.clone();
    let result = execute_in_thread(move || handler.handle(GetNodeCount)).await;

    match result {
        Ok(Ok(count)) => ok_json!({"count": count}),
        Ok(Err(e)) => {
            error!("Failed to get node count: {}", e);
            error_json!("Failed to retrieve node count")
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            error_json!("Internal server error")
        }
    }
}
```

**4. Register Route:**

```rust
// src/handlers/api_handler/graph/mod.rs
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/graph")
            // ... existing routes
            .route("/count", web::get().to(get_node_count))
    );
}
```

### Example 2: Adding a Background Worker

**Requirement:** Periodic cleanup of orphaned edges

**1. Define Actor:**

```rust
// src/actors/edge_cleanup_actor.rs
use actix::prelude::*;

pub struct EdgeCleanupActor {
    neo4j_adapter: Arc<Neo4jAdapter>,
    interval: Duration,
}

impl EdgeCleanupActor {
    pub fn new(neo4j_adapter: Arc<Neo4jAdapter>, interval: Duration) -> Self {
        Self {
            neo4j_adapter,
            interval,
        }
    }

    async fn cleanup_orphaned_edges(&self) -> Result<usize, Box<dyn Error>> {
        let query = neo4rs::query(
            "MATCH (n:Node)-[r:LINKS_TO]->(m:Node)
             WHERE n IS NULL OR m IS NULL
             DELETE r
             RETURN count(r) as deleted_count"
        );

        let mut result = self.neo4j_adapter.graph.execute(query).await?;
        let row = result.next().await?.ok_or("No results")?;
        let count: i64 = row.get("deleted_count")?;

        Ok(count as usize)
    }
}

impl Actor for EdgeCleanupActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Context<Self>) {
        info!("EdgeCleanupActor started - interval: {:?}", self.interval);

        // Schedule periodic cleanup
        ctx.run_interval(self.interval, |act, _ctx| {
            let neo4j = act.neo4j_adapter.clone();

            actix::spawn(async move {
                match act.cleanup_orphaned_edges().await {
                    Ok(count) => {
                        if count > 0 {
                            info!("Cleaned up {} orphaned edges", count);
                        }
                    }
                    Err(e) => {
                        error!("Edge cleanup failed: {}", e);
                    }
                }
            });
        });
    }
}
```

**2. Initialize in AppState:**

```rust
// src/app_state.rs
impl AppState {
    pub async fn new(...) -> Result<Self, Box<dyn Error>> {
        // ... existing initialization

        // Start edge cleanup actor
        let cleanup_interval = Duration::from_secs(3600); // 1 hour
        let _edge_cleanup_addr = EdgeCleanupActor::new(
            neo4j_adapter.clone(),
            cleanup_interval,
        ).start();

        // ... rest of initialization
    }
}
```

### Example 3: Adding Event-Driven Logic

**Requirement:** Send notification when graph sync completes

**1. Define Event:**

```rust
// src/events/domain_events.rs
#[derive(Debug, Clone, Serialize)]
pub struct GraphSyncCompletedEvent {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub duration_ms: u64,
}

impl Into<DomainEvent> for GraphSyncCompletedEvent {
    fn into(self) -> DomainEvent {
        DomainEvent {
            event_type: "graph.sync_completed".to_string(),
            payload: serde_json::to_value(&self).unwrap(),
            timestamp: SystemTime::now(),
        }
    }
}
```

**2. Create Event Handler:**

```rust
// src/events/handlers/notification_handler.rs
pub struct NotificationHandler {
    // Could inject notification service
}

#[async_trait]
impl EventHandler for NotificationHandler {
    async fn handle(&self, event: DomainEvent) -> Result<(), String> {
        if event.event_type == "graph.sync_completed" {
            let sync_event: GraphSyncCompletedEvent = serde_json::from_value(event.payload)
                .map_err(|e| format!("Failed to parse event: {}", e))?;

            info!("📧 Sending notification: Graph sync completed with {} nodes",
                sync_event.total_nodes);

            // TODO: Actually send notification (email, webhook, etc.)

            Ok(())
        } else {
            Ok(())
        }
    }
}
```

**3. Register Handler:**

```rust
// src/events/bus.rs
impl EventBus {
    pub fn register_handlers(&mut self) {
        self.subscribe(
            "graph.sync_completed",
            Box::new(NotificationHandler::new()),
        );
    }
}
```

**4. Publish Event:**

```rust
// src/services/github_sync_service.rs
impl GitHubSyncService {
    pub async fn sync_graphs(&self) -> Result<SyncStats, Box<dyn Error>> {
        let start = Instant::now();

        // ... sync logic

        let duration_ms = start.elapsed().as_millis() as u64;

        // Publish event
        if let Some(event_bus) = &self.event_bus {
            let event = GraphSyncCompletedEvent {
                total_nodes: stats.total_nodes,
                total_edges: stats.total_edges,
                duration_ms,
            };

            event_bus.publish(event.into()).await;
        }

        Ok(stats)
    }
}
```

---

## Cross-References

- **Database Schemas:** [04-database-schemas.md](04-database-schemas.md)
- **Error Codes:**  (TODO)
- **WebSocket Protocol:**  (TODO)
- **CQRS Architecture:** [hexagonal-cqrs-architecture.md](hexagonal-cqrs-architecture.md)
- **GPU Acceleration:** 
- **Ontology Pipeline:** [ontology-reasoning-pipeline.md](ontology-reasoning-pipeline.md)

---

**Document Status:** Production
**Maintainer:** Architecture Team
**Review Cycle:** Quarterly
