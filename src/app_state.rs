use actix::prelude::*;
use actix_web::web;
use log::{debug, info, warn};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::RwLock;

// Neo4j feature imports - now the primary graph repository
use crate::adapters::neo4j_adapter::{Neo4jAdapter, Neo4jConfig};

// CQRS Phase 1D: Graph domain imports
use crate::adapters::actor_graph_repository::ActorGraphRepository;
use crate::application::graph::*;

// CQRS Phase 4: Command/Query/Event buses and Application Services
use crate::cqrs::{CommandBus, QueryBus};
use crate::events::EventBus;

use crate::actors::gpu;
use crate::actors::gpu::GPUContextBus;
use crate::actors::graph_service_supervisor::GraphServiceSupervisor;
use crate::actors::ontology_actor::OntologyActor;
use crate::actors::GPUManagerActor;
use crate::actors::{
    AgentMonitorActor, ClientCoordinatorActor, MetadataActor, OptimizedSettingsActor,
    ProtectedSettingsActor, TaskOrchestratorActor, WorkspaceActor,
};
use crate::config::feature_access::FeatureAccess;
use crate::config::AppFullSettings; 
use crate::models::metadata::MetadataStore;
use crate::models::protected_settings::{ApiKeys, NostrUser, ProtectedSettings};
use crate::services::bots_client::BotsClient;
use crate::services::github::content_enhanced::EnhancedContentAPI;
use crate::services::github::{ContentAPI, GitHubClient};
use crate::services::github_sync_service::GitHubSyncService;
use crate::services::management_api_client::ManagementApiClient;
use crate::services::nostr_service::NostrService;
use crate::services::perplexity_service::PerplexityService;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::speech_service::SpeechService;
use crate::utils::client_message_extractor::ClientMessage;
use tokio::sync::mpsc;
use tokio::time::Duration;

// Repository trait imports for hexagonal architecture
use crate::adapters::neo4j_settings_repository::Neo4jSettingsRepository;
use crate::adapters::neo4j_ontology_repository::{Neo4jOntologyRepository, Neo4jOntologyConfig};
use crate::ports::settings_repository::SettingsRepository;

/// SECURITY: List of known insecure default values that must be rejected
/// Note: Do NOT include empty string - use separate length check instead
const INSECURE_DEFAULT_KEYS: &[&str] = &[
    "change-this-secret-key",
    "changeme",
    "secret",
    "password",
    "admin",
    "test",
    "default",
    "your-secret-key",
    "your-api-key",
    "replace-me",
    "xxx",
];

/// Validates all security-critical environment variables at startup.
/// This function enforces strict security requirements:
/// - MANAGEMENT_API_KEY must be set and not contain insecure default values
/// - JWT_SECRET (if used) must be set and not contain insecure default values
/// # Panics
/// Panics if any security-critical environment variable is missing or insecure.
/// This is intentional to prevent the application from starting in an insecure state.
/// # Returns
/// The validated MANAGEMENT_API_KEY value on success.
fn validate_security_env_vars() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut errors: Vec<String> = Vec::new();

    // Validate MANAGEMENT_API_KEY
    let mgmt_api_key = match std::env::var("MANAGEMENT_API_KEY") {
        Ok(key) => {
            let key_lower = key.to_lowercase();
            if INSECURE_DEFAULT_KEYS.iter().any(|&insecure| !insecure.is_empty() && (key_lower == insecure || key_lower.contains(insecure))) {
                errors.push(format!(
                    "MANAGEMENT_API_KEY contains an insecure default value. \
                     Please set a strong, unique API key (minimum 32 characters recommended)."
                ));
                None
            } else if key.len() < 16 {
                errors.push(format!(
                    "MANAGEMENT_API_KEY is too short ({} chars). \
                     Please use a key with at least 16 characters (32+ recommended).",
                    key.len()
                ));
                None
            } else {
                Some(key)
            }
        }
        Err(_) => {
            errors.push(
                "MANAGEMENT_API_KEY environment variable is not set. \
                 This is required for secure API authentication. \
                 Please set MANAGEMENT_API_KEY to a strong, unique value."
                    .to_string(),
            );
            None
        }
    };

    // Validate JWT_SECRET if it exists (optional but must be secure if set)
    if let Ok(jwt_secret) = std::env::var("JWT_SECRET") {
        let jwt_lower = jwt_secret.to_lowercase();
        if INSECURE_DEFAULT_KEYS.iter().any(|&insecure| !insecure.is_empty() && (jwt_lower == insecure || jwt_lower.contains(insecure))) {
            errors.push(format!(
                "JWT_SECRET contains an insecure default value. \
                 Please set a strong, unique secret (minimum 32 characters recommended)."
            ));
        } else if jwt_secret.len() < 32 {
            warn!(
                "[Security] JWT_SECRET is shorter than recommended ({} chars). \
                 Consider using at least 32 characters for production.",
                jwt_secret.len()
            );
        }
    }

    // If there are any security errors, log them clearly and panic
    if !errors.is_empty() {
        eprintln!("=========================================================");
        eprintln!("  SECURITY CONFIGURATION ERROR - APPLICATION CANNOT START");
        eprintln!("=========================================================");
        for (i, error) in errors.iter().enumerate() {
            eprintln!("  {}. {}", i + 1, error);
        }
        eprintln!("---------------------------------------------------------");
        eprintln!("  Required environment variables:");
        eprintln!("    - MANAGEMENT_API_KEY: Strong unique API key (16+ chars)");
        eprintln!("    - JWT_SECRET (optional): Strong unique secret (32+ chars)");
        eprintln!("---------------------------------------------------------");
        eprintln!("  Example secure configuration:");
        eprintln!("    export MANAGEMENT_API_KEY=$(openssl rand -hex 32)");
        eprintln!("    export JWT_SECRET=$(openssl rand -hex 32)");
        eprintln!("=========================================================");

        return Err(format!(
            "Security configuration failed: {} error(s). See logs above for details.",
            errors.len()
        )
        .into());
    }

    // Return the validated key
    let key = mgmt_api_key.expect("Key validated but None - logic error");
    info!(
        "[Security] MANAGEMENT_API_KEY validated successfully ({}*** chars)",
        key.len()
    );

    Ok(key)
}

// CQRS Phase 1D: Graph query handlers struct
#[derive(Clone)]
pub struct GraphQueryHandlers {
    pub get_graph_data: Arc<GetGraphDataHandler>,
    pub get_node_map: Arc<GetNodeMapHandler>,
    pub get_physics_state: Arc<GetPhysicsStateHandler>,
    pub get_auto_balance_notifications: Arc<GetAutoBalanceNotificationsHandler>,
    pub get_bots_graph_data: Arc<GetBotsGraphDataHandler>,
    pub get_constraints: Arc<GetConstraintsHandler>,
    pub get_equilibrium_status: Arc<GetEquilibriumStatusHandler>,
    pub compute_shortest_paths: Arc<ComputeShortestPathsHandler>,
}

// Phase 7: GPU Subsystem Decomposition
// Independent subsystems that receive GPU context via event bus

/// Physics simulation subsystem actors
#[derive(Clone)]
pub struct PhysicsSubsystem {
    pub force_compute: Option<Addr<gpu::ForceComputeActor>>,
    pub stress_major: Option<Addr<gpu::StressMajorizationActor>>,
    pub constraint: Option<Addr<gpu::ConstraintActor>>,
}

/// Analytics and ML subsystem actors
#[derive(Clone)]
pub struct AnalyticsSubsystem {
    pub clustering: Option<Addr<gpu::ClusteringActor>>,
    pub anomaly: Option<Addr<gpu::AnomalyDetectionActor>>,
    pub pagerank: Option<Addr<gpu::PageRankActor>>,
}

/// Graph algorithm subsystem actors
#[derive(Clone)]
pub struct GraphSubsystem {
    pub shortest_path: Option<Addr<gpu::ShortestPathActor>>,
    pub components: Option<Addr<gpu::ConnectedComponentsActor>>,
}

impl Default for PhysicsSubsystem {
    fn default() -> Self {
        Self {
            force_compute: None,
            stress_major: None,
            constraint: None,
        }
    }
}

impl Default for AnalyticsSubsystem {
    fn default() -> Self {
        Self {
            clustering: None,
            anomaly: None,
            pagerank: None,
        }
    }
}

impl Default for GraphSubsystem {
    fn default() -> Self {
        Self {
            shortest_path: None,
            components: None,
        }
    }
}

impl PhysicsSubsystem {
    /// Check if any physics actors are initialized
    pub fn is_initialized(&self) -> bool {
        self.force_compute.is_some() || self.stress_major.is_some() || self.constraint.is_some()
    }

    /// Get count of active actors
    pub fn active_count(&self) -> usize {
        [
            self.force_compute.is_some(),
            self.stress_major.is_some(),
            self.constraint.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }
}

impl AnalyticsSubsystem {
    /// Check if any analytics actors are initialized
    pub fn is_initialized(&self) -> bool {
        self.clustering.is_some() || self.anomaly.is_some() || self.pagerank.is_some()
    }

    /// Get count of active actors
    pub fn active_count(&self) -> usize {
        [
            self.clustering.is_some(),
            self.anomaly.is_some(),
            self.pagerank.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }
}

impl GraphSubsystem {
    /// Check if any graph algorithm actors are initialized
    pub fn is_initialized(&self) -> bool {
        self.shortest_path.is_some() || self.components.is_some()
    }

    /// Get count of active actors
    pub fn active_count(&self) -> usize {
        [self.shortest_path.is_some(), self.components.is_some()]
            .iter()
            .filter(|&&x| x)
            .count()
    }
}

#[derive(Clone)]
pub struct AppState {
    pub graph_service_addr: Addr<GraphServiceSupervisor>,
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>,
    /// ForceComputeActor address - populated asynchronously after GPU initialization
    /// Use `get_gpu_compute_addr().await` to access this safely
    pub gpu_compute_addr: Arc<RwLock<Option<Addr<gpu::ForceComputeActor>>>>,
    pub stress_majorization_addr: Option<Addr<gpu::StressMajorizationActor>>,
    pub shortest_path_actor: Option<Addr<gpu::ShortestPathActor>>,
    pub connected_components_actor: Option<Addr<gpu::ConnectedComponentsActor>>,

    // Phase 7: Decomposed GPU subsystems (direct access, bypassing GPUManagerActor)
    pub physics: PhysicsSubsystem,
    pub analytics: AnalyticsSubsystem,
    pub graph_ops: GraphSubsystem,

    // Event bus for GPU context distribution to independent subsystems
    pub gpu_context_bus: Arc<GPUContextBus>,

    pub settings_repository: Arc<dyn SettingsRepository>,

    // Concrete Neo4j settings repository for user-specific operations (filters, etc.)
    pub neo4j_settings_repository: Arc<Neo4jSettingsRepository>,

    // Neo4j is now the primary knowledge graph repository
    pub neo4j_adapter: Arc<Neo4jAdapter>,

    // Neo4j ontology repository (replaces UnifiedOntologyRepository)
    pub ontology_repository: Arc<Neo4jOntologyRepository>,

    pub graph_repository: Arc<ActorGraphRepository>,
    pub graph_query_handlers: GraphQueryHandlers,
    
    pub command_bus: Arc<RwLock<CommandBus>>,
    pub query_bus: Arc<RwLock<QueryBus>>,
    pub event_bus: Arc<RwLock<EventBus>>,
    
    
    pub settings_addr: Addr<OptimizedSettingsActor>,
    pub protected_settings_addr: Addr<ProtectedSettingsActor>,
    pub metadata_addr: Addr<MetadataActor>,
    pub client_manager_addr: Addr<ClientCoordinatorActor>,
    pub agent_monitor_addr: Addr<AgentMonitorActor>,
    pub workspace_addr: Addr<WorkspaceActor>,
    pub ontology_actor_addr: Option<Addr<OntologyActor>>,
    pub github_client: Arc<GitHubClient>,
    pub content_api: Arc<ContentAPI>,
    pub perplexity_service: Option<Arc<PerplexityService>>,
    pub ragflow_service: Option<Arc<RAGFlowService>>,
    pub speech_service: Option<Arc<SpeechService>>,
    pub nostr_service: Option<web::Data<NostrService>>,
    pub feature_access: web::Data<FeatureAccess>,
    pub ragflow_session_id: String,
    pub active_connections: Arc<AtomicUsize>,
    pub bots_client: Arc<BotsClient>,
    pub task_orchestrator_addr: Addr<TaskOrchestratorActor>,
    pub debug_enabled: bool,
    pub client_message_tx: mpsc::UnboundedSender<ClientMessage>,
    pub client_message_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<ClientMessage>>>,
    pub ontology_pipeline_service: Option<Arc<crate::services::ontology_pipeline_service::OntologyPipelineService>>,
}

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
        info!("[AppState::new] Initializing actor system");
        tokio::time::sleep(Duration::from_millis(50)).await;


        info!("[AppState::new] Creating repository adapters for hexagonal architecture");

        // Phase 3: Using Neo4j settings repository
        use crate::adapters::neo4j_settings_repository::Neo4jSettingsConfig;
        let settings_config = Neo4jSettingsConfig::default();
        let neo4j_settings_repository = Arc::new(
            Neo4jSettingsRepository::new(settings_config)
                .await
                .map_err(|e| format!("Failed to create Neo4j settings repository: {}", e))?,
        );
        // Keep both trait object and concrete type for different use cases
        let settings_repository: Arc<dyn SettingsRepository> = neo4j_settings_repository.clone();

        info!("[AppState::new] Creating Neo4j ontology repository...");
        let ontology_config = Neo4jOntologyConfig::default();
        let ontology_repository: Arc<Neo4jOntologyRepository> = Arc::new(
            Neo4jOntologyRepository::new(ontology_config)
                .await
                .map_err(|e| format!("Failed to create Neo4j ontology repository: {}", e))?,
        );

        info!("[AppState::new] Neo4j ontology repository initialized successfully");
        info!("[AppState::new] Database and settings service initialized successfully");
        info!(
            "[AppState::new] IMPORTANT: UI now connects directly to database via SettingsService"
        );

        // Neo4j is now the primary graph repository
        let neo4j_adapter = {
            info!("[AppState::new] Initializing Neo4j as primary knowledge graph repository");
            let config = Neo4jConfig::default();
            let adapter = Neo4jAdapter::new(config).await
                .map_err(|e| format!("Failed to initialize Neo4j adapter: {}", e))?;
            info!("✅ Neo4j adapter initialized successfully");
            Arc::new(adapter)
        };

        // Create ontology pipeline service with semantic physics
        info!("[AppState::new] Creating ontology pipeline service");
        let mut pipeline_service = crate::services::ontology_pipeline_service::OntologyPipelineService::new(
            crate::services::ontology_pipeline_service::SemanticPhysicsConfig::default()
        );

        // CRITICAL: Set graph repository for IRI → node ID resolution
        pipeline_service.set_graph_repository(neo4j_adapter.clone());

        let ontology_pipeline_service = Some(Arc::new(pipeline_service));



        info!("[AppState::new] Initializing GitHubSyncService for data ingestion");

        let enhanced_content_api = Arc::new(EnhancedContentAPI::new(github_client.clone()));
        let mut github_sync_service = GitHubSyncService::new(
            enhanced_content_api,
            neo4j_adapter.clone(),
            ontology_repository.clone(),
        );

        // Connect pipeline service to GitHub sync
        if let Some(ref pipeline) = ontology_pipeline_service {
            github_sync_service.set_pipeline_service(pipeline.clone());
            info!("[AppState::new] Ontology pipeline connected to GitHub sync");
        }

        let github_sync_service = Arc::new(github_sync_service);

        info!("[AppState::new] Starting GitHub data sync in background (non-blocking)...");

        let sync_service_clone = github_sync_service.clone();

        // Will be initialized before spawn
        let graph_service_addr_ref: std::sync::Arc<tokio::sync::Mutex<Option<Addr<GraphServiceSupervisor>>>> =
            std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let graph_service_addr_clone_for_sync = graph_service_addr_ref.clone();

        let sync_handle = tokio::spawn(async move {
            info!("🔄 Background GitHub sync task spawned successfully");
            info!("🔄 Task ID: {:?}", std::thread::current().id());
            info!("🔄 Starting sync_graphs() execution...");



            info!("📡 Calling sync_service.sync_graphs()...");
            let sync_start = std::time::Instant::now();

            match sync_service_clone.sync_graphs().await {
                Ok(stats) => {
                    let elapsed = sync_start.elapsed();
                    info!("✅ GitHub sync complete! (elapsed: {:?})", elapsed);
                    info!("  📊 Total files scanned: {}", stats.total_files);
                    info!("  🔗 Knowledge graph files: {}", stats.kg_files_processed);
                    info!("  🏛️  Ontology files: {}", stats.ontology_files_processed);
                    info!("  ⏱️  Duration: {:?}", stats.duration);
                    if !stats.errors.is_empty() {
                        warn!("  ⚠️  Errors encountered: {}", stats.errors.len());
                        for (i, error) in stats.errors.iter().enumerate().take(5) {
                            warn!("    {}. {}", i + 1, error);
                        }
                        if stats.errors.len() > 5 {
                            warn!("    ... and {} more errors", stats.errors.len() - 5);
                        }
                    }

                    // Load synced data into graph actor (if it's ready)
                    if let Some(graph_addr) = &*graph_service_addr_clone_for_sync.lock().await {
                        info!("📥 [GitHub Sync] Notifying GraphServiceActor to reload synced data...");
                        graph_addr.do_send(crate::actors::messages::ReloadGraphFromDatabase);
                        info!("✅ [GitHub Sync] Reload notification sent to GraphServiceActor");
                    } else {
                        info!("ℹ️  [GitHub Sync] Graph service not yet initialized - will load on startup");
                    }
                }
                Err(e) => {
                    let elapsed = sync_start.elapsed();
                    log::error!("❌ Background GitHub sync failed after {:?}: {}", elapsed, e);
                    log::error!("❌ Error details: {:?}", e);
                    log::error!("⚠️  Databases may have partial data - use manual import API if needed");
                }
            }
        });

        
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            info!("👀 GitHub sync monitor: Checking task status...");

            
            let timeout_duration = Duration::from_secs(300); 
            match tokio::time::timeout(timeout_duration, sync_handle).await {
                Ok(join_result) => {
                    match join_result {
                        Ok(_sync_result) => {
                            info!("👀 GitHub sync monitor: Task completed successfully");
                        }
                        Err(join_error) => {
                            if join_error.is_cancelled() {
                                log::error!("👀 GitHub sync monitor: Task was CANCELLED");
                            } else if join_error.is_panic() {
                                log::error!("👀 GitHub sync monitor: Task PANICKED");
                                log::error!("👀 JoinError details: {:?}", join_error);
                            } else {
                                log::error!("👀 GitHub sync monitor: Task failed with unknown error");
                                log::error!("👀 JoinError: {:?}", join_error);
                            }
                        }
                    }
                }
                Err(_timeout_error) => {
                    log::error!("👀 GitHub sync monitor: Task TIMED OUT after {:?}", timeout_duration);
                    log::error!("👀 This likely indicates a deadlock or infinite loop in sync_graphs()");
                }
            }

            info!("👀 GitHub sync monitor: Monitoring complete");
        });

        info!("[AppState::new] GitHub sync running in background with enhanced monitoring, proceeding with actor initialization");


        info!("[AppState::new] Starting ClientCoordinatorActor");
        let mut client_coordinator = ClientCoordinatorActor::new();
        client_coordinator.set_neo4j_repository(neo4j_settings_repository.clone());
        let client_manager_addr = client_coordinator.start();


        let physics_settings = settings.visualisation.graphs.logseq.physics.clone();

        info!("[AppState::new] Starting MetadataActor");
        let metadata_addr = MetadataActor::new(MetadataStore::new()).start();


        info!("[AppState::new] Starting GraphServiceSupervisor (refactored architecture)");








        let graph_service_addr = GraphServiceSupervisor::new(neo4j_adapter.clone()).start();

        // Neo4j feature is now required - removed legacy SQLite path

        // Store graph service address in Arc for GitHub sync task to use
        let graph_service_addr_clone = graph_service_addr.clone();
        tokio::spawn(async move {
            let mut addr_guard = graph_service_addr_ref.lock().await;
            *addr_guard = Some(graph_service_addr_clone);
            info!("[AppState::new] GitHub sync task notified - graph service address available");
        });


        info!("[AppState::new] Retrieving GraphStateActor from GraphServiceSupervisor for CQRS");
        let graph_actor_addr = graph_service_addr
            .send(crate::actors::messages::GetGraphStateActor)
            .await
            .map_err(|e| format!("Failed to send GetGraphStateActor message: {}", e))?
            .ok_or_else(|| "GraphStateActor not initialized in supervisor".to_string())?;

        info!("[AppState::new] Creating Neo4j graph repository adapter (CQRS Phase 2: Direct Query)");
        // Professional, scalable approach: Query Neo4j directly with intelligent caching
        let neo4j_graph_repository = Arc::new(crate::adapters::Neo4jGraphRepository::new(neo4j_adapter.graph().clone()));

        // Create ActorGraphRepository using the graph actor
        let graph_repository = Arc::new(crate::adapters::ActorGraphRepository::new(graph_actor_addr.clone()));

        // Load existing data from Neo4j into repository cache on startup
        info!("[AppState::new] Loading graph data from Neo4j into repository cache...");
        neo4j_graph_repository.load_graph().await
            .map_err(|e| format!("Failed to load graph from Neo4j: {:?}", e))?;

        // Get node count by calling the trait method through the GraphRepository trait
        let node_count = {
            use crate::ports::graph_repository::GraphRepository;
            graph_repository.get_graph().await
                .map(|g| g.nodes.len())
                .unwrap_or(0)
        };
        info!("[AppState::new] ✅ Graph data loaded from Neo4j ({} nodes)", node_count);

        info!("[AppState::new] Initializing CQRS query handlers for graph domain");
        let graph_query_handlers = GraphQueryHandlers {
            get_graph_data: Arc::new(GetGraphDataHandler::new(graph_repository.clone())),
            get_node_map: Arc::new(GetNodeMapHandler::new(graph_repository.clone())),
            get_physics_state: Arc::new(GetPhysicsStateHandler::new(graph_repository.clone())),
            get_auto_balance_notifications: Arc::new(GetAutoBalanceNotificationsHandler::new(
                graph_repository.clone(),
            )),
            get_bots_graph_data: Arc::new(GetBotsGraphDataHandler::new(graph_repository.clone())),
            get_constraints: Arc::new(GetConstraintsHandler::new(graph_repository.clone())),
            get_equilibrium_status: Arc::new(GetEquilibriumStatusHandler::new(
                graph_repository.clone(),
            )),
            compute_shortest_paths: Arc::new(ComputeShortestPathsHandler::new(
                graph_repository.clone(),
            )),
        };

        
        info!("[AppState::new] Initializing CQRS buses (Phase 4)");
        let command_bus = Arc::new(RwLock::new(CommandBus::new()));
        let query_bus = Arc::new(RwLock::new(QueryBus::new()));
        let event_bus = Arc::new(RwLock::new(EventBus::new()));

        // Register event handlers on the EventBus
        {
            use crate::events::handlers::{AuditEventHandler, NotificationEventHandler, GraphEventHandler, OntologyEventHandler};

            let bus = event_bus.write().await;
            let audit_handler = Arc::new(AuditEventHandler::new("global-audit"));
            let notification_handler = Arc::new(NotificationEventHandler::new("global-notifications"));
            let graph_handler = Arc::new(GraphEventHandler::new("global-graph"));
            let ontology_handler = Arc::new(OntologyEventHandler::new("global-ontology"));
            bus.subscribe(audit_handler).await;
            bus.subscribe(notification_handler).await;
            bus.subscribe(graph_handler).await;
            bus.subscribe(ontology_handler).await;
            info!("[AppState::new] EventBus handlers registered: AuditEventHandler, NotificationEventHandler, GraphEventHandler, OntologyEventHandler");
        }

        // Log warnings for settings that are present in config but not yet wired
        {
            let net = &settings.system.network;
            let sec = &settings.system.security;

            if net.enable_tls {
                warn!("[Settings] system.network.enableTls is true but TLS termination is not implemented server-side. Use a reverse proxy for TLS.");
            }
            if net.enable_http2 {
                warn!("[Settings] system.network.enableHttp2 is true but HTTP/2 is not implemented server-side. Use a reverse proxy for HTTP/2.");
            }
            if net.enable_rate_limiting {
                warn!("[Settings] system.network.enableRateLimiting is true but rate limiting middleware is not yet wired. These values have no effect: rateLimitRequests={}, rateLimitWindow={}s",
                    net.rate_limit_requests, net.rate_limit_window);
            }
            if net.enable_metrics {
                warn!("[Settings] system.network.enableMetrics is true but metrics endpoint on port {} is not yet implemented.", net.metrics_port);
            }
            if sec.enable_request_validation {
                warn!("[Settings] system.security.enableRequestValidation is true but request validation middleware is not yet wired.");
            }
        }

        info!("[AppState::new] Linking ClientCoordinatorActor to GraphServiceSupervisor for settling fix");
        
        let graph_supervisor_clone = graph_service_addr.clone();
        let client_manager_clone = client_manager_addr.clone();
        actix::spawn(async move {
            
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            // Set the GraphServiceSupervisor address in ClientManagerActor
            info!("Setting GraphServiceSupervisor address in ClientManagerActor");
            client_manager_clone
                .do_send(crate::actors::messages::SetGraphServiceAddress { addr: graph_supervisor_clone.clone() });
        });


        let (gpu_manager_addr, stress_majorization_addr, shortest_path_actor, connected_components_actor) = {
            info!("[AppState::new] Starting GPUManagerActor (modular architecture)");
            let gpu_manager = GPUManagerActor::new().start();

            // P2 Feature: Initialize ShortestPathActor and ConnectedComponentsActor
            info!("[AppState::new] Starting ShortestPathActor and ConnectedComponentsActor for P2 features");
            let shortest_path = gpu::ShortestPathActor::new().start();
            let connected_components = gpu::ConnectedComponentsActor::new().start();

            // Extract StressMajorizationActor from GPUManagerActor's child actors
            // Note: The actor is spawned by GPUManagerActor, so we'll retrieve it after initialization
            info!("[AppState::new] StressMajorizationActor will be available after GPU initialization");

            (Some(gpu_manager), None, Some(shortest_path), Some(connected_components))
        };


        // Create shared gpu_compute_addr that will be populated asynchronously
        let gpu_compute_addr: Arc<RwLock<Option<Addr<gpu::ForceComputeActor>>>> =
            Arc::new(RwLock::new(None));

        {
            use crate::actors::messages::{InitializeGPUConnection, GetForceComputeActor};

            info!("[AppState] Initializing GPU connection with GPUManagerActor for proper message delegation");
            if let Some(ref gpu_manager) = gpu_manager_addr {
                graph_service_addr.do_send(InitializeGPUConnection {
                    gpu_manager: Some(gpu_manager.clone()),
                });

                // Spawn async task to get ForceComputeActor address after actors are ready
                let gpu_manager_clone = gpu_manager.clone();
                let gpu_compute_addr_clone = gpu_compute_addr.clone();

                actix::spawn(async move {
                    // Wait for GPUManagerActor and PhysicsSupervisor to fully initialize
                    // This delay allows the actor hierarchy to be established:
                    // GPUManagerActor -> ResourceSupervisor -> GPU init -> PhysicsSupervisor -> ForceComputeActor
                    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

                    // Retry loop to get ForceComputeActor address
                    let max_retries = 10;
                    let retry_delay = tokio::time::Duration::from_millis(500);

                    for attempt in 1..=max_retries {
                        debug!("[AppState] Querying GPUManagerActor for ForceComputeActor (attempt {}/{})",
                               attempt, max_retries);

                        match gpu_manager_clone.send(GetForceComputeActor).await {
                            Ok(Ok(force_compute_actor)) => {
                                info!("[AppState] Successfully obtained ForceComputeActor address on attempt {}", attempt);
                                let mut guard = gpu_compute_addr_clone.write().await;
                                *guard = Some(force_compute_actor);
                                info!("[AppState] ForceComputeActor address stored - GPU physics now available via AppState");
                                return;
                            }
                            Ok(Err(e)) => {
                                debug!("[AppState] ForceComputeActor not ready yet: {} (attempt {}/{})",
                                       e, attempt, max_retries);
                            }
                            Err(e) => {
                                warn!("[AppState] Failed to communicate with GPUManagerActor: {} (attempt {}/{})",
                                      e, attempt, max_retries);
                            }
                        }

                        if attempt < max_retries {
                            tokio::time::sleep(retry_delay).await;
                        }
                    }

                    warn!("[AppState] Failed to obtain ForceComputeActor after {} attempts - HTTP handlers will use fallback paths",
                          max_retries);
                });
            } else {
                warn!("[AppState] GPUManagerActor not available - GPU physics will be disabled");
            }
        }

        info!("[AppState::new] Starting OptimizedSettingsActor with repository injection (hexagonal architecture)");

        // Phase 3: Using Neo4j settings repository for actor (reusing config from above)
        let actor_config = Neo4jSettingsConfig::default();
        let actor_settings_repository = Arc::new(
            Neo4jSettingsRepository::new(actor_config)
                .await
                .map_err(|e| format!("Failed to create Neo4j actor settings repository: {}", e))?,
        );

        let settings_actor = OptimizedSettingsActor::with_actors(
            actor_settings_repository,
            Some(graph_service_addr.clone()),
            None,
        )
        .map_err(|e| {
            log::error!("Failed to create OptimizedSettingsActor: {}", e);
            e
        })?;
        let settings_addr = settings_actor.start();

        
        info!("[AppState::new] Starting settings hot-reload watcher");
        
        
        
        
        
        
        
        
        
        
        info!(
            "[AppState::new] Settings hot-reload watcher DISABLED (was causing database deadlocks)"
        );

        info!("[AppState::new] Starting AgentMonitorActor for MCP monitoring");
        let mcp_host =
            std::env::var("MCP_HOST").unwrap_or_else(|_| "agentic-workstation".to_string());
        let mcp_port = std::env::var("MCP_TCP_PORT")
            .unwrap_or_else(|_| "9500".to_string())
            .parse::<u16>()
            .unwrap_or(9500);

        info!(
            "[AppState::new] AgentMonitorActor will poll MCP at {}:{}",
            mcp_host, mcp_port
        );
        let claude_flow_client =
            crate::types::claude_flow::ClaudeFlowClient::new(mcp_host, mcp_port);
        let agent_monitor_addr =
            AgentMonitorActor::new(claude_flow_client, graph_service_addr.clone()).start();

        
        
        
        let sim_params =
            crate::models::simulation_params::SimulationParams::from(&physics_settings);

        let update_msg = crate::actors::messages::UpdateSimulationParams { params: sim_params };


        graph_service_addr.do_send(update_msg.clone());


        if let Some(ref _gpu_addr) = gpu_manager_addr {


        }

        info!("[AppState::new] Starting ProtectedSettingsActor");
        let protected_settings_addr =
            ProtectedSettingsActor::new(ProtectedSettings::default()).start();

        info!("[AppState::new] Starting WorkspaceActor");
        let workspace_addr = WorkspaceActor::new().start();

        info!("[AppState::new] Starting OntologyActor");
        let ontology_actor_addr = {
            let mut ontology_actor = OntologyActor::new();
            // Wire GPU manager for constraint pipeline (Fix #2)
            if let Some(ref gpu_mgr) = gpu_manager_addr {
                ontology_actor.set_gpu_manager_addr(gpu_mgr.clone());
                info!("[AppState] OntologyActor wired to GPUManagerActor for constraint pipeline");
            }
            // Wire client coordinator for WebSocket broadcasts (Fix #8)
            ontology_actor.set_client_manager_addr(client_manager_addr.clone());
            info!("[AppState] OntologyActor initialized successfully");
            Some(ontology_actor.start())
        };

        info!("[AppState::new] Initializing BotsClient with graph service");
        let bots_client = Arc::new(BotsClient::with_graph_service(graph_service_addr.clone()));

        info!("[AppState::new] Initializing TaskOrchestratorActor with Management API");
        let mgmt_api_host = std::env::var("MANAGEMENT_API_HOST")
            .unwrap_or_else(|_| "agentic-workstation".to_string());
        let mgmt_api_port = std::env::var("MANAGEMENT_API_PORT")
            .unwrap_or_else(|_| "9090".to_string())
            .parse::<u16>()
            .unwrap_or(9090);
        // SECURITY: Validate all security-critical environment variables at startup
        let mgmt_api_key = validate_security_env_vars()?;

        let mgmt_client = ManagementApiClient::new(mgmt_api_host, mgmt_api_port, mgmt_api_key);
        let task_orchestrator_addr = TaskOrchestratorActor::new(mgmt_client).start();

        
        
        info!("[AppState] GPU manager will self-initialize when needed");


        info!("[AppState::new] Actor system initialization complete (GPU initialization sent earlier)");

        
        let debug_enabled = crate::utils::logging::is_debug_enabled();

        info!("[AppState::new] Debug mode enabled: {}", debug_enabled);

        
        let (client_message_tx, client_message_rx) = mpsc::unbounded_channel::<ClientMessage>();
        info!("[AppState::new] Client message channel created");

        // Phase 7: Initialize GPU context bus for event-based distribution
        info!("[AppState::new] Creating GPU context bus for subsystem distribution");
        let gpu_context_bus = Arc::new(GPUContextBus::new());

        // Phase 7: Initialize decomposed GPU subsystems
        // These will receive GPU context via the event bus when GPUManagerActor initializes
        let physics = PhysicsSubsystem {
            force_compute: None,
            stress_major: None,
            constraint: None,
        };

        let analytics = AnalyticsSubsystem {
            clustering: None,
            anomaly: None,
            pagerank: None,
        };

        // Graph ops subsystem initialized with the actors we already created
        let graph_ops = GraphSubsystem {
            shortest_path: shortest_path_actor.clone(),
            components: connected_components_actor.clone(),
        };

        info!("[AppState::new] GPU subsystems initialized (physics={}, analytics={}, graph_ops={})",
            physics.active_count(), analytics.active_count(), graph_ops.active_count());

        let state = Self {
            graph_service_addr,
            gpu_manager_addr,
            gpu_compute_addr,  // Now Arc<RwLock<Option<...>>>, populated asynchronously
            stress_majorization_addr,
            shortest_path_actor,
            connected_components_actor,

            // Phase 7: Decomposed subsystems
            physics,
            analytics,
            graph_ops,
            gpu_context_bus,

            settings_repository,
            neo4j_settings_repository,

            neo4j_adapter,

            ontology_repository,

            graph_repository,
            graph_query_handlers,

            command_bus,
            query_bus,
            event_bus,


            settings_addr,
            protected_settings_addr,
            metadata_addr,
            client_manager_addr,
            agent_monitor_addr,
            workspace_addr,
            ontology_actor_addr,
            github_client,
            content_api,
            perplexity_service,
            ragflow_service,
            speech_service,
            nostr_service: None,
            feature_access: web::Data::new(FeatureAccess::from_env()),
            ragflow_session_id,
            active_connections: Arc::new(AtomicUsize::new(0)),
            bots_client,
            task_orchestrator_addr,
            debug_enabled,
            client_message_tx,
            client_message_rx: Arc::new(tokio::sync::Mutex::new(client_message_rx)),
            ontology_pipeline_service,
        };

        // Validate optional actor addresses
        info!("[AppState::new] Validating actor initialization");
        let validation_report = state.validate();
        validation_report.log();

        if !validation_report.is_valid() {
            return Err(format!("AppState validation failed: {:?}", validation_report.errors).into());
        }

        info!("[AppState::new] ✅ All validation checks passed");

        Ok(state)
    }

    /// Validate that all optional actors and services are properly initialized
    /// based on feature flags and environment configuration.
    pub fn validate(&self) -> crate::validation::ValidationReport {
        use crate::validation::*;
        let mut report = ValidationReport::new();

        // GPU-related actors
        {
            report.add(ValidationItem {
                name: "GPUManagerActor".to_string(),
                expected: true,
                present: self.gpu_manager_addr.is_some(),
                severity: Severity::Warning,
                reason: "GPU feature is enabled".to_string(),
            });

            // gpu_compute_addr is populated asynchronously - check via try_read
            let gpu_compute_present = self.gpu_compute_addr
                .try_read()
                .map(|guard| guard.is_some())
                .unwrap_or(false);
            report.add(ValidationItem {
                name: "gpu_compute_addr".to_string(),
                expected: false,
                present: gpu_compute_present,
                severity: Severity::Info,
                reason: "Initialized asynchronously after GPU manager starts".to_string(),
            });

            report.add(ValidationItem {
                name: "stress_majorization_addr".to_string(),
                expected: false,
                present: self.stress_majorization_addr.is_some(),
                severity: Severity::Info,
                reason: "Initialized after GPU manager starts".to_string(),
            });
        }

        // Ontology actor
        {
            let present = self.ontology_actor_addr.is_some();
            report.add(ValidationItem {
                name: "OntologyActor".to_string(),
                expected: true,
                present,
                severity: Severity::Warning,
                reason: "Ontology feature is enabled".to_string(),
            });
        }

        // Perplexity service (environment-dependent)
        let perplexity_expected = env_is_set("PERPLEXITY_API_KEY");
        report.add(ValidationItem {
            name: "PerplexityService".to_string(),
            expected: perplexity_expected,
            present: self.perplexity_service.is_some(),
            severity: if perplexity_expected { Severity::Warning } else { Severity::Info },
            reason: if perplexity_expected {
                "PERPLEXITY_API_KEY is set".to_string()
            } else {
                "PERPLEXITY_API_KEY not set".to_string()
            },
        });

        // RAGFlow service (environment-dependent)
        let ragflow_expected = env_is_set("RAGFLOW_API_KEY");
        report.add(ValidationItem {
            name: "RAGFlowService".to_string(),
            expected: ragflow_expected,
            present: self.ragflow_service.is_some(),
            severity: if ragflow_expected { Severity::Warning } else { Severity::Info },
            reason: if ragflow_expected {
                "RAGFLOW_API_KEY is set".to_string()
            } else {
                "RAGFLOW_API_KEY not set".to_string()
            },
        });

        // Speech service (environment-dependent)
        let speech_expected = env_is_set("SPEECH_SERVICE_ENABLED");
        report.add(ValidationItem {
            name: "SpeechService".to_string(),
            expected: speech_expected,
            present: self.speech_service.is_some(),
            severity: if speech_expected { Severity::Warning } else { Severity::Info },
            reason: if speech_expected {
                "SPEECH_SERVICE_ENABLED is set".to_string()
            } else {
                "SPEECH_SERVICE_ENABLED not set".to_string()
            },
        });

        // Nostr service (set later via set_nostr_service)
        report.add(ValidationItem {
            name: "NostrService".to_string(),
            expected: false,
            present: self.nostr_service.is_some(),
            severity: Severity::Info,
            reason: "Set later via set_nostr_service()".to_string(),
        });

        // Ontology pipeline service
        report.add(ValidationItem {
            name: "OntologyPipelineService".to_string(),
            expected: true,
            present: self.ontology_pipeline_service.is_some(),
            severity: Severity::Warning,
            reason: "Required for semantic physics".to_string(),
        });

        report
    }

    pub fn increment_connections(&self) -> usize {
        self.active_connections.fetch_add(1, Ordering::SeqCst)
    }

    pub fn decrement_connections(&self) -> usize {
        self.active_connections
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if current > 0 { Some(current - 1) } else { None }
            })
            .unwrap_or(0)
    }

    pub async fn get_api_keys(&self, pubkey: &str) -> ApiKeys {
        use crate::actors::protected_settings_actor::GetApiKeys;
        self.protected_settings_addr
            .send(GetApiKeys {
                pubkey: pubkey.to_string(),
            })
            .await
            .unwrap_or_else(|_| ApiKeys::default())
    }

    pub async fn get_nostr_user(&self, pubkey: &str) -> Option<NostrUser> {
        if let Some(nostr_service) = &self.nostr_service {
            nostr_service.get_user(pubkey).await
        } else {
            None
        }
    }

    pub async fn validate_nostr_session(&self, pubkey: &str, token: &str) -> bool {
        if let Some(nostr_service) = &self.nostr_service {
            nostr_service.validate_session(pubkey, token).await
        } else {
            false
        }
    }

    pub async fn update_nostr_user_api_keys(
        &self,
        pubkey: &str,
        api_keys: ApiKeys,
    ) -> Result<NostrUser, String> {
        if let Some(nostr_service) = &self.nostr_service {
            nostr_service
                .update_user_api_keys(pubkey, api_keys)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err("Nostr service not initialized".to_string())
        }
    }

    pub fn set_nostr_service(&mut self, service: NostrService) {
        self.nostr_service = Some(web::Data::new(service));
    }

    pub fn is_power_user(&self, pubkey: &str) -> bool {
        self.feature_access.is_power_user(pubkey)
    }

    pub fn can_sync_settings(&self, pubkey: &str) -> bool {
        self.feature_access.can_sync_settings(pubkey)
    }

    pub fn has_feature_access(&self, pubkey: &str, feature: &str) -> bool {
        self.feature_access.has_feature_access(pubkey, feature)
    }

    pub fn get_available_features(&self, pubkey: &str) -> Vec<String> {
        self.feature_access.get_available_features(pubkey)
    }

    pub fn get_client_manager_addr(&self) -> &Addr<ClientCoordinatorActor> {
        &self.client_manager_addr
    }

    pub fn get_graph_service_addr(&self) -> &Addr<GraphServiceSupervisor> {
        &self.graph_service_addr
    }

    pub fn get_settings_addr(&self) -> &Addr<OptimizedSettingsActor> {
        &self.settings_addr
    }

    pub fn get_metadata_addr(&self) -> &Addr<MetadataActor> {
        &self.metadata_addr
    }

    pub fn get_workspace_addr(&self) -> &Addr<WorkspaceActor> {
        &self.workspace_addr
    }

    pub fn get_ontology_actor_addr(&self) -> Option<&Addr<OntologyActor>> {
        self.ontology_actor_addr.as_ref()
    }

    pub fn get_task_orchestrator_addr(&self) -> &Addr<TaskOrchestratorActor> {
        &self.task_orchestrator_addr
    }

    /// Get the ForceComputeActor address asynchronously.
    /// Returns None if the address hasn't been initialized yet or GPU is not available.
    pub async fn get_gpu_compute_addr(&self) -> Option<Addr<gpu::ForceComputeActor>> {
        self.gpu_compute_addr.read().await.clone()
    }

    /// Try to get the ForceComputeActor address synchronously (non-blocking).
    /// Returns None if the lock is held or the address isn't available.
    pub fn try_get_gpu_compute_addr(&self) -> Option<Addr<gpu::ForceComputeActor>> {
        self.gpu_compute_addr
            .try_read()
            .ok()
            .and_then(|guard| guard.clone())
    }
}
