use actix::prelude::*;
use actix_web::web;
use log::{info, warn};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[cfg(feature = "gpu")]
use crate::actors::gpu;
use crate::actors::graph_service_supervisor::TransitionalGraphSupervisor;
#[cfg(feature = "ontology")]
use crate::actors::ontology_actor::OntologyActor;
#[cfg(feature = "gpu")]
use crate::actors::GPUManagerActor;
use crate::actors::{
    AgentMonitorActor, ClientCoordinatorActor, MetadataActor, OptimizedSettingsActor,
    ProtectedSettingsActor, TaskOrchestratorActor, WorkspaceActor,
};
use crate::config::feature_access::FeatureAccess;
use crate::config::AppFullSettings; // Renamed for clarity, ClientFacingSettings removed
use crate::models::metadata::MetadataStore;
use crate::models::protected_settings::{ApiKeys, NostrUser, ProtectedSettings};
use crate::services::bots_client::BotsClient;
use crate::services::database_service::DatabaseService;
use crate::services::github::{ContentAPI, GitHubClient};
use crate::services::management_api_client::ManagementApiClient;
use crate::services::nostr_service::NostrService;
use crate::services::perplexity_service::PerplexityService;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::settings_service::SettingsService;
use crate::services::speech_service::SpeechService;
use crate::utils::client_message_extractor::ClientMessage;
#[cfg(feature = "gpu")]
use cudarc::driver::CudaDevice;
use tokio::sync::mpsc;
use tokio::time::Duration;

// Repository trait imports for hexagonal architecture
use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
use crate::adapters::sqlite_settings_repository::SqliteSettingsRepository;
use crate::ports::settings_repository::SettingsRepository;

#[derive(Clone)]
pub struct AppState {
    pub graph_service_addr: Addr<TransitionalGraphSupervisor>,
    #[cfg(feature = "gpu")]
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>, // Modular GPU manager system
    #[cfg(feature = "gpu")]
    pub gpu_compute_addr: Option<Addr<gpu::ForceComputeActor>>, // Force compute actor for physics
    // HEXAGONAL ARCHITECTURE: Repository adapters
    // Settings uses trait object (non-generic handlers)
    pub settings_repository: Arc<dyn SettingsRepository>,
    // Knowledge graph and ontology use concrete types (generic handlers)
    pub knowledge_graph_repository: Arc<SqliteKnowledgeGraphRepository>,
    pub ontology_repository: Arc<SqliteOntologyRepository>,
    // Database-backed settings (legacy - for backward compatibility)
    pub db_service: Arc<DatabaseService>,
    pub settings_service: Arc<SettingsService>,
    // Legacy actor-based settings (will be phased out)
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

        // CRITICAL: Initialize database FIRST - this is the new source of truth for settings
        info!("[AppState::new] Initializing SQLite database (NEW architecture)");
        let db_path =
            std::env::var("DATABASE_PATH").unwrap_or_else(|_| "data/visionflow.db".to_string());
        let db_service = Arc::new(
            DatabaseService::new(&db_path)
                .map_err(|e| format!("Failed to create database: {}", e))?,
        );

        // Initialize database schema
        info!("[AppState::new] Initializing database schema");
        db_service
            .initialize_schema()
            .map_err(|e| format!("Failed to initialize schema: {}", e))?;

        // Save current settings to database (migration from YAML/in-memory)
        info!("[AppState::new] Migrating settings to database");
        db_service
            .save_all_settings(&settings)
            .map_err(|e| format!("Failed to save settings to database: {}", e))?;

        // Create settings service (provides direct access to database for handlers)
        info!("[AppState::new] Creating SettingsService (UI → Database direct connection)");
        let settings_service = Arc::new(
            SettingsService::new(db_service.clone())
                .map_err(|e| format!("Failed to create settings service: {}", e))?,
        );

        // HEXAGONAL ARCHITECTURE: Create repository adapters (port implementations)
        info!("[AppState::new] Creating repository adapters for hexagonal architecture");

        // Settings repository as trait object (handlers accept Arc<dyn SettingsRepository>)
        let settings_repository: Arc<dyn SettingsRepository> =
            Arc::new(SqliteSettingsRepository::new(db_service.clone()));

        // Knowledge graph and ontology repositories use spawn_blocking to avoid blocking tokio runtime
        // during schema creation (12+ CREATE statements can take 500ms-3s)
        info!("[AppState::new] Creating knowledge graph repository in blocking context...");
        let knowledge_graph_repository: Arc<SqliteKnowledgeGraphRepository> =
            tokio::task::spawn_blocking(|| {
                SqliteKnowledgeGraphRepository::new("data/knowledge_graph.db")
            })
            .await
            .map_err(|e| format!("Failed to spawn blocking task: {}", e))?
            .map_err(|e| format!("Failed to create knowledge graph repository: {}", e))
            .map(Arc::new)?;

        info!("[AppState::new] Creating ontology repository in blocking context...");
        let ontology_repository: Arc<SqliteOntologyRepository> =
            tokio::task::spawn_blocking(|| {
                SqliteOntologyRepository::new("data/ontology.db")
            })
            .await
            .map_err(|e| format!("Failed to spawn blocking task: {}", e))?
            .map_err(|e| format!("Failed to create ontology repository: {}", e))
            .map(Arc::new)?;

        info!("[AppState::new] Repository adapters initialized successfully (via spawn_blocking)");
        info!("[AppState::new] Database and settings service initialized successfully");
        info!(
            "[AppState::new] IMPORTANT: UI now connects directly to database via SettingsService"
        );

        // Start actors
        info!("[AppState::new] Starting ClientCoordinatorActor");
        let client_manager_addr = ClientCoordinatorActor::new().start();

        // Extract physics settings from logseq graph before moving settings
        let physics_settings = settings.visualisation.graphs.logseq.physics.clone();

        info!("[AppState::new] Starting MetadataActor");
        let metadata_addr = MetadataActor::new(MetadataStore::new()).start();

        // Create GraphServiceSupervisor instead of the monolithic GraphServiceActor
        info!("[AppState::new] Starting GraphServiceSupervisor (refactored architecture)");
        // CUDA initialization moved to GPU compute actor to avoid blocking actor startup
        // #[cfg(feature = "gpu")]
        // {
        //     let _device = CudaDevice::new(0).map_err(|e| {
        //         log::error!("Failed to create CUDA device: {}", e);
        //         format!("CUDA initialization failed: {}", e)
        //     })?;
        // }
        let graph_service_addr = TransitionalGraphSupervisor::new(
            Some(client_manager_addr.clone()),
            None, // GPU manager will be linked later
        )
        .start();

        // WEBSOCKET SETTLING FIX: Set graph service supervisor address in client manager for force broadcasts
        info!("[AppState::new] Linking ClientCoordinatorActor to TransitionalGraphSupervisor for settling fix");
        // Get the internal GraphServiceActor from the supervisor and set it in ClientManagerActor
        let graph_supervisor_clone = graph_service_addr.clone();
        let client_manager_clone = client_manager_addr.clone();
        actix::spawn(async move {
            // Wait a moment for supervisor to initialize
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            // Get the internal GraphServiceActor address
            if let Ok(Some(graph_actor)) = graph_supervisor_clone
                .send(crate::actors::messages::GetGraphServiceActor)
                .await
            {
                info!("Retrieved GraphServiceActor from supervisor, setting in ClientManagerActor");
                client_manager_clone
                    .do_send(crate::actors::messages::SetGraphServiceAddress { addr: graph_actor });
            } else {
                warn!("Could not retrieve GraphServiceActor from supervisor");
            }
        });

        // Create the modular GPU manager system
        #[cfg(feature = "gpu")]
        let gpu_manager_addr = {
            info!("[AppState::new] Starting GPUManagerActor (modular architecture)");
            Some(GPUManagerActor::new().start())
        };

        // Initialize the connection between GraphServiceSupervisor and GPUManagerActor
        #[cfg(feature = "gpu")]
        {
            use crate::actors::messages::InitializeGPUConnection;
            // Send the GPUManagerActor address to GraphServiceSupervisor for proper message routing
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
        // Create settings repository adapter for OptimizedSettingsActor (separate from CQRS repositories)
        let actor_settings_repository = Arc::new(SqliteSettingsRepository::new(db_service.clone()));

        let settings_actor = OptimizedSettingsActor::with_actors(
            actor_settings_repository,
            Some(graph_service_addr.clone()),
            None, // Legacy GPU compute actor removed
        )
        .map_err(|e| {
            log::error!("Failed to create OptimizedSettingsActor: {}", e);
            e
        })?;
        let settings_addr = settings_actor.start();

        // Start settings hot-reload watcher
        info!("[AppState::new] Starting settings hot-reload watcher");
        let settings_db_path = std::env::var("SETTINGS_DB_PATH")
            .unwrap_or_else(|_| "data/settings.db".to_string());
        let settings_watcher =
            crate::services::settings_watcher::SettingsWatcher::new(settings_db_path, settings_addr.clone());
        tokio::spawn(async move {
            if let Err(e) = settings_watcher.start().await {
                log::error!("Settings watcher failed to start: {}", e);
            }
        });
        info!("[AppState::new] Settings hot-reload watcher started successfully");

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

        // Send initial physics settings to both GraphServiceSupervisor and GPUComputeActor
        // Use the From trait to convert PhysicsSettings to SimulationParams
        // This ensures all fields are properly set from settings.yaml
        let sim_params =
            crate::models::simulation_params::SimulationParams::from(&physics_settings);

        let update_msg = crate::actors::messages::UpdateSimulationParams { params: sim_params };

        // Send to GraphServiceSupervisor
        graph_service_addr.do_send(update_msg.clone());

        // Send to GPUManagerActor if available
        #[cfg(feature = "gpu")]
        if let Some(ref _gpu_addr) = gpu_manager_addr {
            // TODO: GPUManagerActor needs to handle UpdateSimulationParams
            // gpu_addr.do_send(update_msg);
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

        // GPU initialization will be handled later by the actors themselves
        // This avoids the tokio runtime panic during initialization
        info!("[AppState] GPU manager will self-initialize when needed");

        // Schedule GPU initialization to happen after actor system is ready
        #[cfg(feature = "gpu")]
        if let Some(ref gpu_manager) = gpu_manager_addr {
            use crate::actors::messages::InitializeGPUConnection;
            let init_msg = InitializeGPUConnection {
                gpu_manager: Some(gpu_manager.clone()),
            };
            graph_service_addr.do_send(init_msg);
            info!("[AppState] Sent GPU initialization message to GraphServiceSupervisor");
        }

        info!("[AppState::new] Actor system initialization complete");

        // Read debug state from settings (can be overridden by env var)
        let debug_enabled = crate::utils::logging::is_debug_enabled();

        info!("[AppState::new] Debug mode enabled: {}", debug_enabled);

        // Create client message channel for agent -> user communication
        let (client_message_tx, client_message_rx) = mpsc::unbounded_channel::<ClientMessage>();
        info!("[AppState::new] Client message channel created");

        Ok(Self {
            graph_service_addr,
            #[cfg(feature = "gpu")]
            gpu_manager_addr,
            #[cfg(feature = "gpu")]
            gpu_compute_addr: None, // Will be set by GPUManagerActor
            // HEXAGONAL ARCHITECTURE: Repository trait objects
            settings_repository,
            knowledge_graph_repository,
            ontology_repository,
            // NEW: Database-backed settings (direct UI connection)
            db_service,
            settings_service,
            // Legacy actor-based settings
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

    pub fn get_graph_service_addr(&self) -> &Addr<TransitionalGraphSupervisor> {
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
