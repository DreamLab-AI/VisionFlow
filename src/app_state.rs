use actix::prelude::*;
use actix_web::web;
use log::{info, warn};
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
use crate::application::{
    GraphApplicationService, OntologyApplicationService, PhysicsApplicationService,
    SettingsApplicationService,
};
use crate::cqrs::{CommandBus, QueryBus};
use crate::events::EventBus;

#[cfg(feature = "gpu")]
use crate::actors::gpu;
use crate::actors::graph_service_supervisor::GraphServiceSupervisor;
#[cfg(feature = "ontology")]
use crate::actors::ontology_actor::OntologyActor;
#[cfg(feature = "gpu")]
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
use crate::ports::settings_repository::SettingsRepository;
use crate::repositories::UnifiedOntologyRepository;

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

// CQRS Phase 4: Application Services
#[derive(Clone)]
pub struct ApplicationServices {
    pub graph: GraphApplicationService,
    pub settings: SettingsApplicationService,
    pub ontology: OntologyApplicationService,
    pub physics: PhysicsApplicationService,
}

#[derive(Clone)]
pub struct AppState {
    pub graph_service_addr: Addr<GraphServiceSupervisor>,
    #[cfg(feature = "gpu")]
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>,
    #[cfg(feature = "gpu")]
    pub gpu_compute_addr: Option<Addr<gpu::ForceComputeActor>>,
    #[cfg(feature = "gpu")]
    pub stress_majorization_addr: Option<Addr<gpu::StressMajorizationActor>>, 
    

    pub settings_repository: Arc<dyn SettingsRepository>,

    // Neo4j is now the primary knowledge graph repository
    pub neo4j_adapter: Arc<Neo4jAdapter>,

    pub ontology_repository: Arc<UnifiedOntologyRepository>,

    pub graph_repository: Arc<ActorGraphRepository>,
    pub graph_query_handlers: GraphQueryHandlers,
    
    pub command_bus: Arc<RwLock<CommandBus>>,
    pub query_bus: Arc<RwLock<QueryBus>>,
    pub event_bus: Arc<RwLock<EventBus>>,
    
    pub app_services: ApplicationServices,
    
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
        let settings_repository: Arc<dyn SettingsRepository> = Arc::new(
            Neo4jSettingsRepository::new(settings_config)
                .await
                .map_err(|e| format!("Failed to create Neo4j settings repository: {}", e))?,
        );

        info!("[AppState::new] Creating unified ontology repository in blocking context...");
        let ontology_repository: Arc<UnifiedOntologyRepository> =
            tokio::task::spawn_blocking(|| UnifiedOntologyRepository::new("data/unified.db"))
                .await
                .map_err(|e| format!("Failed to spawn blocking task: {}", e))?
                .map_err(|e| format!("Failed to create unified ontology repository: {}", e))
                .map(Arc::new)?;

        info!("[AppState::new] Repository adapters initialized successfully (via spawn_blocking)");
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
        let client_manager_addr = ClientCoordinatorActor::new().start();


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

        info!("[AppState::new] Creating graph repository adapter (CQRS Phase 1D)");
        let graph_repository = Arc::new(ActorGraphRepository::new(graph_actor_addr));

        // Load existing data from database into graph actor on startup
        info!("[AppState::new] Loading graph data from database into GraphServiceActor...");
        graph_service_addr.do_send(crate::actors::messages::ReloadGraphFromDatabase);
        info!("[AppState::new] ✅ Initial graph reload request sent");

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

        
        info!("[AppState::new] Initializing application services (Phase 4)");
        let app_services = ApplicationServices {
            graph: GraphApplicationService::new(
                command_bus.clone(),
                query_bus.clone(),
                event_bus.clone(),
            ),
            settings: SettingsApplicationService::new(
                command_bus.clone(),
                query_bus.clone(),
                event_bus.clone(),
            ),
            ontology: OntologyApplicationService::new(
                command_bus.clone(),
                query_bus.clone(),
                event_bus.clone(),
            ),
            physics: PhysicsApplicationService::new(
                command_bus.clone(),
                query_bus.clone(),
                event_bus.clone(),
            ),
        };

        
        info!("[AppState::new] Linking ClientCoordinatorActor to GraphServiceSupervisor for settling fix");
        
        let graph_supervisor_clone = graph_service_addr.clone();
        let client_manager_clone = client_manager_addr.clone();
        actix::spawn(async move {
            
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            
            if let Ok(Some(graph_actor)) = graph_supervisor_clone
                .send(crate::actors::messages::GetGraphStateActor)
                .await
            {
                info!("Retrieved GraphServiceActor from supervisor, setting in ClientManagerActor");
                client_manager_clone
                    .do_send(crate::actors::messages::SetGraphServiceAddress { addr: graph_actor });
            } else {
                warn!("Could not retrieve GraphServiceActor from supervisor");
            }
        });

        
        #[cfg(feature = "gpu")]
        let (gpu_manager_addr, stress_majorization_addr) = {
            info!("[AppState::new] Starting GPUManagerActor (modular architecture)");
            let gpu_manager = GPUManagerActor::new().start();

            // Extract StressMajorizationActor from GPUManagerActor's child actors
            // Note: The actor is spawned by GPUManagerActor, so we'll retrieve it after initialization
            info!("[AppState::new] StressMajorizationActor will be available after GPU initialization");

            (Some(gpu_manager), None) // stress_majorization_addr will be set later
        };

        
        #[cfg(feature = "gpu")]
        {
            use crate::actors::messages::InitializeGPUConnection;
            
            info!("[AppState] Initializing GPU connection with GPUManagerActor for proper message delegation");
            if let Some(ref gpu_manager) = gpu_manager_addr {
                graph_service_addr.do_send(InitializeGPUConnection {
                    gpu_manager: Some(gpu_manager.clone()),
                });
            } else {
                warn!("[AppState] GPUManagerActor not available - GPU physics will be disabled");
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            info!("[AppState] GPU feature disabled - running in CPU-only mode");
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

        
        #[cfg(feature = "gpu")]
        if let Some(ref _gpu_addr) = gpu_manager_addr {
            
            
        }

        info!("[AppState::new] Starting ProtectedSettingsActor");
        let protected_settings_addr =
            ProtectedSettingsActor::new(ProtectedSettings::default()).start();

        info!("[AppState::new] Starting WorkspaceActor");
        let workspace_addr = WorkspaceActor::new().start();

        info!("[AppState::new] Starting OntologyActor");
        #[cfg(feature = "ontology")]
        let ontology_actor_addr = {
            info!("[AppState] OntologyActor initialized successfully");
            Some(OntologyActor::new().start())
        };

        #[cfg(not(feature = "ontology"))]
        let ontology_actor_addr = None;

        info!("[AppState::new] Initializing BotsClient with graph service");
        let bots_client = Arc::new(BotsClient::with_graph_service(graph_service_addr.clone()));

        info!("[AppState::new] Initializing TaskOrchestratorActor with Management API");
        let mgmt_api_host = std::env::var("MANAGEMENT_API_HOST")
            .unwrap_or_else(|_| "agentic-workstation".to_string());
        let mgmt_api_port = std::env::var("MANAGEMENT_API_PORT")
            .unwrap_or_else(|_| "9090".to_string())
            .parse::<u16>()
            .unwrap_or(9090);
        let mgmt_api_key = std::env::var("MANAGEMENT_API_KEY").unwrap_or_else(|_| {
            warn!("[AppState] MANAGEMENT_API_KEY not set, using default");
            "change-this-secret-key".to_string()
        });

        let mgmt_client = ManagementApiClient::new(mgmt_api_host, mgmt_api_port, mgmt_api_key);
        let task_orchestrator_addr = TaskOrchestratorActor::new(mgmt_client).start();

        
        
        info!("[AppState] GPU manager will self-initialize when needed");


        info!("[AppState::new] Actor system initialization complete (GPU initialization sent earlier)");

        
        let debug_enabled = crate::utils::logging::is_debug_enabled();

        info!("[AppState::new] Debug mode enabled: {}", debug_enabled);

        
        let (client_message_tx, client_message_rx) = mpsc::unbounded_channel::<ClientMessage>();
        info!("[AppState::new] Client message channel created");

        Ok(Self {
            graph_service_addr,
            #[cfg(feature = "gpu")]
            gpu_manager_addr,
            #[cfg(feature = "gpu")]
            gpu_compute_addr: None,
            #[cfg(feature = "gpu")]
            stress_majorization_addr,

            settings_repository,

            neo4j_adapter,

            ontology_repository,

            graph_repository,
            graph_query_handlers,

            command_bus,
            query_bus,
            event_bus,

            app_services,

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
        })
    }

    pub fn increment_connections(&self) -> usize {
        self.active_connections.fetch_add(1, Ordering::SeqCst)
    }

    pub fn decrement_connections(&self) -> usize {
        self.active_connections.fetch_sub(1, Ordering::SeqCst)
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
}
