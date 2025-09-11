use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use actix::prelude::*;
use actix_web::web;
use log::{info};

use crate::actors::{GraphServiceActor, SettingsActor, MetadataActor, ClientManagerActor, GPUManagerActor, ProtectedSettingsActor, ClaudeFlowActor};
use crate::actors::gpu;
use cudarc::driver::CudaDevice;
use crate::config::AppFullSettings; // Renamed for clarity, ClientFacingSettings removed
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

#[derive(Clone)]
pub struct AppState {
    pub graph_service_addr: Addr<GraphServiceActor>,
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>, // Modular GPU manager system
    pub gpu_compute_addr: Option<Addr<gpu::ForceComputeActor>>, // Force compute actor for physics
    pub settings_addr: Addr<SettingsActor>,
    pub protected_settings_addr: Addr<ProtectedSettingsActor>,
    pub metadata_addr: Addr<MetadataActor>,
    pub client_manager_addr: Addr<ClientManagerActor>,
    pub claude_flow_addr: Addr<ClaudeFlowActor>,
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
    pub debug_enabled: bool,
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

        // Start actors
        info!("[AppState::new] Starting ClientManagerActor");
        let client_manager_addr = ClientManagerActor::new().start();

        // Extract physics settings from logseq graph before moving settings
        let physics_settings = settings.visualisation.graphs.logseq.physics.clone();
        
        info!("[AppState::new] Starting MetadataActor");
        let metadata_addr = MetadataActor::new(MetadataStore::new()).start();

        // Create GraphServiceActor first, but without GPU compute address initially
        info!("[AppState::new] Starting GraphServiceActor");
        let _device = CudaDevice::new(0).map_err(|e| {
            log::error!("Failed to create CUDA device: {}", e);
            format!("CUDA initialization failed: {}", e)
        })?;
        let graph_service_addr = GraphServiceActor::new(
            client_manager_addr.clone(),
            None, // GPU compute actor will be created and linked later
            None // SettingsActor address will be set later
        ).start();
        
        // Create the modular GPU manager system
        info!("[AppState::new] Starting GPUManagerActor (modular architecture)");
        let gpu_manager_addr = Some(GPUManagerActor::new().start());
        
        // Store the GPU manager address in the graph service actor for physics
        // Note: We'll need to update GraphServiceActor to use GPUManagerActor instead
        use crate::actors::messages::StoreGPUComputeAddress;
        // TODO: Update this message type to StoreGPUManagerAddress
        // For now, passing None since graph actor needs updating
        graph_service_addr.do_send(StoreGPUComputeAddress {
            addr: None,
        });

        info!("[AppState::new] Starting SettingsActor with actor addresses for physics forwarding");
        let settings_actor = SettingsActor::with_actors(
            Some(graph_service_addr.clone()),
            None, // Legacy GPU compute actor removed
        ).map_err(|e| {
            log::error!("Failed to create SettingsActor: {}", e);
            e
        })?;
        let settings_addr = settings_actor.start();
        
        info!("[AppState::new] Starting ClaudeFlowActor (TCP)");
        // Create ClaudeFlowClient for MCP connection on port 9500
        // Use multi-agent-container since VisionFlow runs in a separate container
        let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
            .or_else(|_| std::env::var("MCP_HOST"))
            .unwrap_or_else(|_| "multi-agent-container".to_string());
        let claude_flow_port = std::env::var("MCP_TCP_PORT")
            .unwrap_or_else(|_| "9500".to_string())
            .parse::<u16>()
            .unwrap_or(9500);
        
        info!("[AppState::new] Connecting ClaudeFlowActor to {}:{}", claude_flow_host, claude_flow_port);
        let claude_flow_client = crate::types::claude_flow::ClaudeFlowClient::new(
            claude_flow_host,
            claude_flow_port
        );
        let claude_flow_addr = ClaudeFlowActor::new(
            claude_flow_client,
            graph_service_addr.clone()
        ).start();
        
        // Send initial physics settings to both GraphServiceActor and GPUComputeActor
        // Use the From trait to convert PhysicsSettings to SimulationParams
        // This ensures all fields are properly set from settings.yaml
        let sim_params = crate::models::simulation_params::SimulationParams::from(&physics_settings);
        
        let update_msg = crate::actors::messages::UpdateSimulationParams {
            params: sim_params,
        };
        
        // Send to GraphServiceActor
        graph_service_addr.do_send(update_msg.clone());
        
        // Send to GPUManagerActor if available
        if let Some(ref gpu_addr) = gpu_manager_addr {
            // TODO: GPUManagerActor needs to handle UpdateSimulationParams
            // gpu_addr.do_send(update_msg);
        }

        info!("[AppState::new] Starting ProtectedSettingsActor");
        let protected_settings_addr = ProtectedSettingsActor::new(ProtectedSettings::default()).start();

        info!("[AppState::new] Initializing BotsClient with graph service");
        let bots_client = Arc::new(BotsClient::with_graph_service(graph_service_addr.clone()));

        // Initialize GPU with delay to ensure graph data is ready
        if let Some(ref gpu_addr) = gpu_manager_addr {
            let gpu_addr_clone = gpu_addr.clone();
            let graph_addr_clone = graph_service_addr.clone();
            
            // Spawn a task to initialize GPU after a delay
            tokio::spawn(async move {
                // Wait for graph service to be ready and data to be loaded
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                
                log::info!("[AppState] Initializing GPU manager with graph data");
                
                // Get initial graph data
                use crate::actors::messages::{GetGraphData};
                match graph_addr_clone.send(GetGraphData).await {
                    Ok(Ok(graph_data)) => {
                        log::info!("[AppState] Got graph data with {} nodes for GPU manager", graph_data.nodes.len());
                        // TODO: Send initialization to GPU manager
                        // gpu_addr_clone.do_send(InitializeGPUManager { graph: graph_data });
                    }
                    Ok(Err(e)) => {
                        log::error!("[AppState] Failed to get graph data for GPU init: {}", e);
                    }
                    Err(e) => {
                        log::error!("[AppState] Failed to send GetGraphData message: {}", e);
                    }
                }
            });
        }

        info!("[AppState::new] Actor system initialization complete");

        // Read debug state from settings (can be overridden by env var)
        let debug_enabled = crate::utils::logging::is_debug_enabled();
        
        info!("[AppState::new] Debug mode enabled: {}", debug_enabled);

        Ok(Self {
            graph_service_addr,
            gpu_manager_addr,
            gpu_compute_addr: None, // Will be set by GPUManagerActor
            settings_addr,
            protected_settings_addr,
            metadata_addr,
            client_manager_addr,
            claude_flow_addr,
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
            debug_enabled,
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

    pub fn get_client_manager_addr(&self) -> &Addr<ClientManagerActor> {
        &self.client_manager_addr
    }

    pub fn get_graph_service_addr(&self) -> &Addr<GraphServiceActor> {
        &self.graph_service_addr
    }

    pub fn get_settings_addr(&self) -> &Addr<SettingsActor> {
        &self.settings_addr
    }

    pub fn get_metadata_addr(&self) -> &Addr<MetadataActor> {
        &self.metadata_addr
    }
}
