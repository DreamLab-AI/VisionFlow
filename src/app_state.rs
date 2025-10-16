use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use actix::prelude::*;
use actix_web::web;
use log::{info, warn};

use crate::actors::{OptimizedSettingsActor, MetadataActor, ClientCoordinatorActor, ProtectedSettingsActor, AgentMonitorActor, WorkspaceActor, TaskOrchestratorActor};
use crate::actors::graph_state_actor::GraphStateActor;
use crate::actors::semantic_processor_actor::SemanticProcessorActor;
use crate::actors::physics_orchestrator_actor::PhysicsOrchestratorActor;
#[cfg(feature = "gpu")]
use crate::actors::GPUManagerActor;
#[cfg(feature = "ontology")]
use crate::actors::ontology_actor::OntologyActor;
#[cfg(feature = "gpu")]
use crate::actors::gpu;
#[cfg(feature = "gpu")]
use cudarc::driver::CudaDevice;
use crate::config::AppFullSettings;
use tokio::time::Duration;
use crate::config::feature_access::FeatureAccess;
use crate::models::metadata::MetadataStore;
use crate::models::protected_settings::{ProtectedSettings, ApiKeys, NostrUser};
use crate::services::github::{GitHubClient, ContentAPI};
use crate::services::perplexity_service::PerplexityService;
use crate::services::speech_service::SpeechService;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::nostr_service::NostrService;
use crate::services::bots_client::BotsClient;
use crate::services::management_api_client::ManagementApiClient;
use tokio::sync::mpsc;
use crate::utils::client_message_extractor::ClientMessage;

#[derive(Clone)]
pub struct AppState {
    pub graph_state_addr: Addr<GraphStateActor>,
    pub semantic_processor_addr: Addr<SemanticProcessorActor>,
    pub physics_orchestrator_addr: Addr<PhysicsOrchestratorActor>,
    #[cfg(feature = "gpu")]
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>,
    #[cfg(feature = "gpu")]
    pub gpu_compute_addr: Option<Addr<gpu::ForceComputeActor>>,
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
        info!("[AppState::new] Initializing actor system with Hexagonal Architecture");
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Extract physics settings from logseq graph before moving settings
        let physics_settings = settings.visualisation.graphs.logseq.physics.clone();

        info!("[AppState::new] Starting MetadataActor");
        let metadata_addr = MetadataActor::new(MetadataStore::new()).start();

        // Initialize Hexagonal Architecture actors
        info!("[AppState::new] Starting GraphStateActor (Domain Layer)");
        let graph_state_addr = GraphStateActor::new().start();

        info!("[AppState::new] Starting SemanticProcessorActor (Domain Layer)");
        let semantic_processor_addr = SemanticProcessorActor::new(None).start();

        // Create the modular GPU manager system
        #[cfg(feature = "gpu")]
        let (gpu_manager_addr, gpu_compute_addr) = {
            info!("[AppState::new] Initializing GPU subsystem");
            let _device = CudaDevice::new(0).map_err(|e| {
                log::error!("Failed to create CUDA device: {}", e);
                format!("CUDA initialization failed: {}", e)
            })?;

            info!("[AppState::new] Starting GPUManagerActor");
            let gpu_manager = GPUManagerActor::new().start();

            // Request ForceComputeActor address from GPUManagerActor
            info!("[AppState::new] Requesting ForceComputeActor address from GPUManagerActor");
            let gpu_compute = match gpu_manager.send(crate::actors::messages::GetForceComputeActor).await {
                Ok(Ok(addr)) => {
                    info!("[AppState::new] ForceComputeActor address received");
                    Some(addr)
                }
                Ok(Err(e)) => {
                    warn!("[AppState::new] Failed to get ForceComputeActor address: {}", e);
                    None
                }
                Err(e) => {
                    warn!("[AppState::new] Mailbox error getting ForceComputeActor: {}", e);
                    None
                }
            };

            (Some(gpu_manager), gpu_compute)
        };

        #[cfg(not(feature = "gpu"))]
        {
            info!("[AppState] GPU feature disabled - running in CPU-only mode");
        }

        // Convert physics settings to simulation parameters
        let sim_params = crate::models::simulation_params::SimulationParams::from(&physics_settings);

        // Start PhysicsOrchestratorActor with graph state dependency
        info!("[AppState::new] Starting PhysicsOrchestratorActor with dependencies");
        #[cfg(feature = "gpu")]
        let physics_orchestrator_addr = {
            let graph_data = match graph_state_addr.send(crate::actors::messages::GetGraphData).await {
                Ok(Ok(data)) => Some(data),
                _ => None,
            };
            PhysicsOrchestratorActor::new(
                sim_params.clone(),
                gpu_compute_addr.clone(),
                graph_data,
            ).start()
        };

        #[cfg(not(feature = "gpu"))]
        let physics_orchestrator_addr = {
            let graph_data = match graph_state_addr.send(crate::actors::messages::GetGraphData).await {
                Ok(Ok(data)) => Some(data),
                _ => None,
            };
            PhysicsOrchestratorActor::new(
                sim_params.clone(),
                None,
                graph_data,
            ).start()
        };

        info!("[AppState::new] Starting ClientCoordinatorActor");
        let client_manager_addr = ClientCoordinatorActor::new().start();

        info!("[AppState::new] Starting OptimizedSettingsActor");
        let settings_actor = OptimizedSettingsActor::with_actors(
            None, // GraphServiceSupervisor replaced by Hexagonal Architecture
            None, // Legacy GPU compute actor removed
        ).map_err(|e| {
            log::error!("Failed to create OptimizedSettingsActor: {}", e);
            e
        })?;
        let settings_addr = settings_actor.start();

        info!("[AppState::new] Starting AgentMonitorActor for MCP monitoring");
        let mcp_host = std::env::var("MCP_HOST")
            .unwrap_or_else(|_| "agentic-workstation".to_string());
        let mcp_port = std::env::var("MCP_TCP_PORT")
            .unwrap_or_else(|_| "9500".to_string())
            .parse::<u16>()
            .unwrap_or(9500);

        info!("[AppState::new] AgentMonitorActor will poll MCP at {}:{}", mcp_host, mcp_port);
        let claude_flow_client = crate::types::claude_flow::ClaudeFlowClient::new(
            mcp_host,
            mcp_port
        );
        let agent_monitor_addr = AgentMonitorActor::new(
            claude_flow_client,
            graph_state_addr.clone()
        ).start();

        // Send initial physics settings to PhysicsOrchestratorActor
        let update_msg = crate::actors::messages::UpdateSimulationParams {
            params: sim_params,
        };
        physics_orchestrator_addr.do_send(update_msg.clone());

        // Send to GPUManagerActor if available
        #[cfg(feature = "gpu")]
        if let Some(ref gpu_addr) = gpu_manager_addr {
            gpu_addr.do_send(update_msg);
        }

        info!("[AppState::new] Starting ProtectedSettingsActor");
        let protected_settings_addr = ProtectedSettingsActor::new(ProtectedSettings::default()).start();

        info!("[AppState::new] Starting WorkspaceActor");
        let workspace_addr = WorkspaceActor::new().start();

        info!("[AppState::new] Starting OntologyActor");
        #[cfg(feature = "ontology")]
        let ontology_actor_addr = {
            let ontology_actor = OntologyActor::new().start();

            // Connect OntologyActor to PhysicsOrchestratorActor for constraint generation
            #[cfg(feature = "ontology")]
            physics_orchestrator_addr.do_send(crate::actors::physics_orchestrator_actor::SetOntologyActor {
                addr: ontology_actor.clone(),
            });

            info!("[AppState] OntologyActor initialized and connected to PhysicsOrchestratorActor");
            Some(ontology_actor)
        };

        #[cfg(not(feature = "ontology"))]
        let ontology_actor_addr = None;

        info!("[AppState::new] Initializing BotsClient with graph state");
        let bots_client = Arc::new(BotsClient::with_graph_service(graph_state_addr.clone()));

        info!("[AppState::new] Initializing TaskOrchestratorActor with Management API");
        let mgmt_api_host = std::env::var("MANAGEMENT_API_HOST")
            .unwrap_or_else(|_| "agentic-workstation".to_string());
        let mgmt_api_port = std::env::var("MANAGEMENT_API_PORT")
            .unwrap_or_else(|_| "9090".to_string())
            .parse::<u16>()
            .unwrap_or(9090);
        let mgmt_api_key = std::env::var("MANAGEMENT_API_KEY")
            .unwrap_or_else(|_| {
                warn!("[AppState] MANAGEMENT_API_KEY not set, using default");
                "change-this-secret-key".to_string()
            });

        let mgmt_client = ManagementApiClient::new(mgmt_api_host, mgmt_api_port, mgmt_api_key);
        let task_orchestrator_addr = TaskOrchestratorActor::new(mgmt_client).start();

        // Initialize GPU connection between actors
        #[cfg(feature = "gpu")]
        if let Some(ref gpu_manager) = gpu_manager_addr {
            use crate::actors::messages::InitializeGPUConnection;
            info!("[AppState] Initializing GPU connection between Hexagonal Architecture actors");
            graph_state_addr.do_send(InitializeGPUConnection {
                gpu_manager: Some(gpu_manager.clone()),
            });
        }

        info!("[AppState::new] Hexagonal Architecture initialization complete");

        // Read debug state from settings (can be overridden by env var)
        let debug_enabled = crate::utils::logging::is_debug_enabled();

        info!("[AppState::new] Debug mode enabled: {}", debug_enabled);

        // Create client message channel for agent -> user communication
        let (client_message_tx, client_message_rx) = mpsc::unbounded_channel::<ClientMessage>();
        info!("[AppState::new] Client message channel created");

        Ok(Self {
            graph_state_addr,
            semantic_processor_addr,
            physics_orchestrator_addr,
            #[cfg(feature = "gpu")]
            gpu_manager_addr,
            #[cfg(feature = "gpu")]
            gpu_compute_addr,
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
        self.protected_settings_addr.send(GetApiKeys {
            pubkey: pubkey.to_string(),
        }).await.unwrap_or_else(|_| ApiKeys::default())
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

    pub async fn update_nostr_user_api_keys(&self, pubkey: &str, api_keys: ApiKeys) -> Result<NostrUser, String> {
        if let Some(nostr_service) = &self.nostr_service {
            nostr_service.update_user_api_keys(pubkey, api_keys)
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

    pub fn get_graph_state_addr(&self) -> &Addr<GraphStateActor> {
        &self.graph_state_addr
    }

    pub fn get_semantic_processor_addr(&self) -> &Addr<SemanticProcessorActor> {
        &self.semantic_processor_addr
    }

    pub fn get_physics_orchestrator_addr(&self) -> &Addr<PhysicsOrchestratorActor> {
        &self.physics_orchestrator_addr
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

    #[cfg(feature = "gpu")]
    pub fn get_gpu_manager_addr(&self) -> Option<&Addr<GPUManagerActor>> {
        self.gpu_manager_addr.as_ref()
    }
}
