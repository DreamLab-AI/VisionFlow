// Rebuild: KE velocity fix applied
use webxr::actors::messages::{UpdateMetadata, BuildGraphFromMetadata, AddNodesFromMetadata, UpdateGraphData};
use webxr::services::nostr_service::NostrService;
use webxr::{
    config::AppFullSettings, // Import AppFullSettings only
    handlers::{
        api_handler,
        bots_visualization_handler,
        client_log_handler,
        client_messages_handler,
        graph_export_handler,
        mcp_relay_handler::mcp_relay_handler,
        nostr_handler,
        pages_handler,
        socket_flow_handler::{socket_flow_handler, PreReadSocketSettings}, // Import PreReadSocketSettings
        speech_socket_handler::speech_socket_handler,
        // DEPRECATED: hybrid_health_handler removed
        workspace_handler,
    },
    services::speech_service::SpeechService,
    services::{
        file_service::FileService,
        // graph_service::GraphService removed - now using GraphServiceSupervisor
        github::{ContentAPI, GitHubClient, GitHubConfig},
        ragflow_service::RAGFlowService, // ADDED IMPORT
    },
    // DEPRECATED: docker_hive_mind, HybridHealthManager removed
    utils::mcp_connection::MCPConnectionPool,
    AppState,
};

use actix_cors::Cors;
use actix_web::{middleware, web, App, HttpServer};
// DEPRECATED: std::future imports removed (were for ErrorRecoveryMiddleware)
// DEPRECATED: Actix dev imports removed (were for ErrorRecoveryMiddleware)
// DEPRECATED: LocalBoxFuture import removed (was for ErrorRecoveryMiddleware)
// use actix_files::Files; // Removed unused import
use dotenvy::dotenv;
use log::{debug, error, info, warn};
use std::sync::Arc;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::RwLock;
use tokio::time::Duration;
use webxr::middleware::TimeoutMiddleware;
use webxr::telemetry::agent_telemetry::init_telemetry_logger;
use webxr::utils::advanced_logging::init_advanced_logging;
use webxr::utils::logging::init_logging;

// DEPRECATED: ErrorRecoveryMiddleware removed - NetworkRecoveryManager deleted
/*
/// Simple error recovery middleware that integrates with NetworkRecoveryManager
pub struct ErrorRecoveryMiddleware {
    recovery_manager: Option<Arc<webxr::utils::hybrid_fault_tolerance::NetworkRecoveryManager>>,
}

impl ErrorRecoveryMiddleware {
    pub fn new() -> Self {
        Self {
            recovery_manager: None,
        }
    }

    pub fn with_recovery_manager(recovery_manager: Arc<webxr::utils::hybrid_fault_tolerance::NetworkRecoveryManager>) -> Self {
        Self {
            recovery_manager: Some(recovery_manager),
        }
    }
}

impl<S, B> Transform<S, ServiceRequest> for ErrorRecoveryMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = ActixError;
    type InitError = ();
    type Transform = ErrorRecoveryMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(ErrorRecoveryMiddlewareService {
            service,
            recovery_manager: self.recovery_manager.clone(),
        }))
    }
}

pub struct ErrorRecoveryMiddlewareService<S> {
    service: S,
    recovery_manager: Option<Arc<webxr::utils::hybrid_fault_tolerance::NetworkRecoveryManager>>,
}

impl<S, B> Service<ServiceRequest> for ErrorRecoveryMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = ActixError;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let fut = self.service.call(req);
        let recovery_manager = self.recovery_manager.clone();

        Box::pin(async move {
            match fut.await {
                Ok(response) => Ok(response),
                Err(error) => {
                    // Log the error
                    error!("Request failed: {}", error);

                    // If we have a recovery manager, we could trigger recovery here
                    if let Some(_manager) = recovery_manager {
                        warn!("Error recovery manager available but not implementing specific recovery for this error type");
                    }

                    // Return the original error for now
                    Err(error)
                }
            }
        })
    }
}
*/

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Make dotenv optional since env vars can come from Docker
    dotenv().ok();

    // Initialize logging with env_logger (reads RUST_LOG environment variable)
    init_logging()?;

    // Initialize advanced logging system
    if let Err(e) = init_advanced_logging() {
        error!("Failed to initialize advanced logging: {}", e);
    } else {
        info!("Advanced logging system initialized successfully");
    }

    // Initialize telemetry logger
    // Check if we're in Docker or local development
    let log_dir = if std::path::Path::new("/app/logs").exists() {
        "/app/logs".to_string()
    } else if std::path::Path::new("/workspace/ext/logs").exists() {
        "/workspace/ext/logs".to_string()
    } else {
        // Fallback to temp directory for development
        std::env::temp_dir()
            .join("webxr_telemetry")
            .to_string_lossy()
            .to_string()
    };

    let log_dir = std::env::var("TELEMETRY_LOG_DIR").unwrap_or(log_dir);

    if let Err(e) = init_telemetry_logger(&log_dir, 100) {
        error!("Failed to initialize telemetry logger: {}", e);
    } else {
        info!("Telemetry logger initialized with directory: {}", log_dir);
    }

    // Load settings
    let settings = match AppFullSettings::new() {
        Ok(s) => {
            info!(
                "✅ AppFullSettings loaded successfully from: {}",
                std::env::var("SETTINGS_FILE_PATH")
                    .unwrap_or_else(|_| "/app/settings.yaml".to_string())
            );

            // Test JSON serialization to verify camelCase output works
            match serde_json::to_string(&s.visualisation.rendering) {
                Ok(json_output) => {
                    info!(
                        "✅ SERDE ALIAS FIX WORKS! JSON serialization (camelCase): {}",
                        json_output
                    );

                    // Verify the JSON contains camelCase fields, not snake_case
                    if json_output.contains("ambientLightIntensity")
                        && !json_output.contains("ambient_light_intensity")
                    {
                        info!("✅ CONFIRMED: JSON uses camelCase field names for REST API compatibility");
                    }

                    // Log some key values that were loaded from snake_case YAML
                    info!("✅ CONFIRMED: Values loaded from snake_case YAML:");
                    info!(
                        "   - ambient_light_intensity -> {}",
                        s.visualisation.rendering.ambient_light_intensity
                    );
                    info!(
                        "   - enable_ambient_occlusion -> {}",
                        s.visualisation.rendering.enable_ambient_occlusion
                    );
                    info!(
                        "   - background_color -> {}",
                        s.visualisation.rendering.background_color
                    );
                    info!("🎉 SERDE ALIAS FIX IS WORKING: YAML (snake_case) loads successfully, JSON serializes as camelCase!");
                }
                Err(e) => {
                    error!("❌ JSON serialization failed: {}", e);
                }
            }

            Arc::new(RwLock::new(s)) // Now holds Arc<RwLock<AppFullSettings>>
        }
        Err(e) => {
            error!("❌ Failed to load AppFullSettings: {:?}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to initialize AppFullSettings: {:?}", e),
            ));
        }
    };

    // GPU compute is now handled by the GPUComputeActor
    info!("GPU compute will be initialized by GPUComputeActor when needed");

    debug!("Successfully loaded AppFullSettings"); // Updated log message

    info!("Starting WebXR application...");
    debug!("main: Beginning application startup sequence.");

    // Create web::Data instances first
    // This now holds Data<Arc<RwLock<AppFullSettings>>>
    let settings_data = web::Data::new(settings.clone());

    // Initialize services
    let github_config = match GitHubConfig::from_env() {
        Ok(config) => config,
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load GitHub config: {}", e),
            ))
        }
    };

    // GitHubClient::new might need adjustment if it expects client-facing Settings
    // Assuming it can work with AppFullSettings for now.
    let github_client = match GitHubClient::new(github_config, settings.clone()).await {
        Ok(client) => Arc::new(client),
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to initialize GitHub client: {}", e),
            ))
        }
    };

    let content_api = Arc::new(ContentAPI::new(github_client.clone()));

    // Initialize speech service
    // SpeechService::new might need adjustment if it expects client-facing Settings
    let speech_service = {
        let service = SpeechService::new(settings.clone());
        Some(Arc::new(service))
    };

    // Initialize RAGFlow Service
    info!("[main] Attempting to initialize RAGFlowService...");
    let ragflow_service_option = match RAGFlowService::new(settings.clone()).await {
        Ok(service) => {
            info!("[main] RAGFlowService::new SUCCEEDED. Service instance created.");
            Some(Arc::new(service))
        }
        Err(e) => {
            error!("[main] RAGFlowService::new FAILED. Error: {}", e);
            None
        }
    };

    if ragflow_service_option.is_some() {
        info!("[main] ragflow_service_option is Some after RAGFlowService::new attempt.");
    } else {
        error!("[main] ragflow_service_option is None after RAGFlowService::new attempt. Chat functionality will be unavailable.");
    }

    // Initialize app state asynchronously
    // AppState::new now receives AppFullSettings directly (not Arc<RwLock<>>)
    let settings_value = {
        let settings_read = settings.read().await;
        settings_read.clone()
    };

    let mut app_state = match AppState::new(
        settings_value,
        github_client.clone(),
        content_api.clone(),
        None,                   // Perplexity placeholder
        ragflow_service_option, // Pass the initialized RAGFlow service
        speech_service,
        "default_session".to_string(), // RAGFlow session ID placeholder
    )
    .await
    {
        Ok(state) => {
            info!("[main] AppState::new completed successfully");
            state
        }
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to initialize app state: {}", e),
            ))
        }
    };

    info!("[main] About to initialize Nostr service");
    // Initialize Nostr service
    nostr_handler::init_nostr_service(&mut app_state);
    info!("[main] Nostr service initialized");

    // DEPRECATED: HybridHealthManager removed - use TaskOrchestratorActor
    // Docker exec architecture replaced by HTTP Management API

    // Skip bots orchestrator connection during startup to prevent blocking
    // Connection will be established on-demand when bots features are used
    info!("Skipping bots orchestrator connection during startup (will connect on-demand)");

    // First, try to load existing metadata without waiting for GitHub download
    info!("Loading existing metadata for quick initialization");
    let metadata_store = FileService::load_or_create_metadata().map_err(|e| {
        error!("Failed to load existing metadata: {}", e);
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })?;

    if metadata_store.is_empty() {
        warn!("No metadata found, starting with empty metadata store");
        // Continue with empty metadata - the app can still function
    }

    info!("Loaded {} items from metadata store", metadata_store.len());

    // Spawn background task to fetch GitHub data after server starts
    info!("Spawning background task to fetch and process GitHub markdown files");
    let content_api_clone = content_api.clone();
    let settings_clone = settings.clone();
    let metadata_addr_clone = app_state.metadata_addr.clone();
    let graph_service_addr_clone = app_state.graph_service_addr.clone();

    tokio::spawn(async move {
        // Wait a bit for the server to fully initialize
        info!("Background GitHub sync: Waiting 5 seconds for server initialization...");
        tokio::time::sleep(Duration::from_secs(5)).await;

        info!("Background GitHub sync: Starting markdown synchronization process");

        // Get settings for FileService
        let file_service = FileService::new(settings_clone.clone());
        info!("Background GitHub sync: FileService created");

        // Create a mutable copy for the fetch operation
        let mut metadata_store_copy = match FileService::load_or_create_metadata() {
            Ok(store) => {
                info!(
                    "Background GitHub sync: Loaded metadata store with {} existing entries",
                    store.len()
                );
                store
            }
            Err(e) => {
                error!("Background GitHub sync: Failed to load metadata: {}", e);
                return;
            }
        };

        info!("Background GitHub sync: Starting fetch_and_process_files...");
        match file_service
            .fetch_and_process_files(
                content_api_clone,
                settings_clone.clone(),
                &mut metadata_store_copy,
            )
            .await
        {
            Ok(processed_files) => {
                info!(
                    "Background GitHub sync: Successfully processed {} markdown files",
                    processed_files.len()
                );

                if processed_files.is_empty() {
                    warn!("Background GitHub sync: No files were processed - check GitHub configuration");
                } else {
                    let file_names: Vec<String> = processed_files
                        .iter()
                        .map(|pf| pf.file_name.clone())
                        .collect();
                    info!("Background GitHub sync: Processed files: {:?}", file_names);
                }

                // Update metadata actor
                info!("Background GitHub sync: Updating metadata actor...");
                if let Err(e) = metadata_addr_clone
                    .send(UpdateMetadata {
                        metadata: metadata_store_copy.clone(),
                    })
                    .await
                {
                    error!(
                        "Background GitHub sync: Failed to update metadata actor: {}",
                        e
                    );
                } else {
                    info!("Background GitHub sync: Metadata actor updated successfully");
                }

                // Save metadata to disk
                info!("Background GitHub sync: Saving metadata to disk...");
                if let Err(e) = FileService::save_metadata(&metadata_store_copy) {
                    error!("Background GitHub sync: Failed to save metadata: {}", e);
                } else {
                    info!("Background GitHub sync: Metadata saved successfully");
                }

                // Update graph with new data
                use webxr::actors::messages::AddNodesFromMetadata;
                info!("Background GitHub sync: Updating graph with new data...");
                match graph_service_addr_clone
                    .send(AddNodesFromMetadata {
                        metadata: metadata_store_copy,
                    })
                    .await
                {
                    Ok(Ok(())) => {
                        info!("Background GitHub sync: Graph updated successfully with new GitHub data");
                    }
                    Ok(Err(e)) => {
                        error!("Background GitHub sync: Failed to update graph: {}", e);
                    }
                    Err(e) => {
                        error!("Background GitHub sync: Actor communication error: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Background GitHub sync: Error processing files: {}", e);
                error!("Background GitHub sync: Check GitHub token and repository configuration");
            }
        }

        info!("Background GitHub sync: Process completed");
    });

    info!("[main] Background GitHub sync task spawned, continuing main thread...");

    // Skip metadata actor update during startup to prevent blocking
    // The background GitHub sync task will update metadata when ready
    info!("Skipping metadata actor update during startup (will be updated by background task)");

    // Build initial graph from metadata and initialize GPU compute
    info!("Building initial graph from existing metadata for physics simulation");

    // First, try to load pre-computed graph data with positions
    let graph_data_option = match FileService::load_graph_data() {
        Ok(graph_data) => {
            if let Some(graph) = graph_data {
                info!(
                    "Loaded pre-computed graph data with {} nodes and {} edges",
                    graph.nodes.len(),
                    graph.edges.len()
                );
                Some(graph)
            } else {
                info!("No pre-computed graph data found, will build from metadata");
                None
            }
        }
        Err(e) => {
            error!("Error loading graph data: {}", e);
            None
        }
    };

    // Use GraphServiceSupervisor to build or update the graph
    use std::sync::Arc as StdArc;
    use webxr::actors::messages::{BuildGraphFromMetadata, UpdateGraphData};

    // Use fire-and-forget for all graph initialization to avoid blocking startup
    if let Some(graph_data) = graph_data_option {
        // If we have pre-computed graph data, send it directly to the GraphServiceSupervisor
        info!("Triggering async graph data load with {} nodes and {} edges", graph_data.nodes.len(), graph_data.edges.len());
        app_state.graph_service_addr.do_send(UpdateGraphData {
            graph_data: StdArc::new(graph_data),
        });
        info!("Pre-computed graph data load triggered (async)");
    } else {
        // Build graph from metadata using fire-and-forget to avoid blocking startup
        info!("Triggering async graph build from {} metadata entries", metadata_store.len());
        app_state.graph_service_addr.do_send(BuildGraphFromMetadata {
            metadata: metadata_store.clone(),
        });
        info!("Graph build triggered (async)");
    }

    info!("Starting HTTP server...");

    // Start simulation in GraphServiceSupervisor (Second start attempt commented out for debugging stack overflow)
    // use webxr::actors::messages::StartSimulation;
    // if let Err(e) = app_state.graph_service_addr.send(StartSimulation).await {
    //     error!("Failed to start simulation in GraphServiceSupervisor: {}", e);
    //     return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to start simulation: {}", e)));
    // }
    // info!("Simulation started in GraphServiceSupervisor (Second start attempt commented out)");
    info!("Skipping redundant StartSimulation message to GraphServiceSupervisor for debugging stack overflow. Simulation should already be running from supervisor's started() method.");

    // Create web::Data after all initialization is complete
    let app_state_data = web::Data::new(app_state);
    // DEPRECATED: hybrid_health_manager_data, mcp_session_bridge, session_correlation_bridge removed

    // Start the server - read from environment (set by docker-compose)
    let bind_address = std::env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("SYSTEM_NETWORK_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(4000);
    let bind_address = format!("{}:{}", bind_address, port);

    // Pre-read WebSocket settings for SocketFlowServer
    let pre_read_ws_settings = {
        let s = settings.read().await;
        PreReadSocketSettings {
            min_update_rate: s.system.websocket.min_update_rate,
            max_update_rate: s.system.websocket.max_update_rate,
            motion_threshold: s.system.websocket.motion_threshold,
            motion_damping: s.system.websocket.motion_damping,
            heartbeat_interval_ms: s.system.websocket.heartbeat_interval, // Assuming these exist
            heartbeat_timeout_ms: s.system.websocket.heartbeat_timeout,   // Assuming these exist
        }
    };
    let pre_read_ws_settings_data = web::Data::new(pre_read_ws_settings);

    info!("Starting HTTP server on {}", bind_address);

    info!("main: All services and actors initialized. Configuring HTTP server.");
    let server =
        HttpServer::new(move || {
            let cors = Cors::default()
                .allow_any_origin()
                .allow_any_method()
                .allow_any_header()
                .max_age(3600)
                .supports_credentials();

            let app = App::new()
            .wrap(middleware::Logger::default())
            .wrap(cors)
            .wrap(middleware::Compress::default())
            .wrap(TimeoutMiddleware::new(Duration::from_secs(30))) // Add 30s timeout middleware
            // DEPRECATED: ErrorRecoveryMiddleware removed
            // Pass AppFullSettings wrapped in Data
            .app_data(settings_data.clone())
            .app_data(web::Data::new(github_client.clone()))
            .app_data(web::Data::new(content_api.clone()))
            .app_data(app_state_data.clone()) // Add the complete AppState
            .app_data(pre_read_ws_settings_data.clone()) // Add pre-read WebSocket settings
            // Register actor addresses for handler access
            .app_data(web::Data::new(app_state_data.graph_service_addr.clone()))
            .app_data(web::Data::new(app_state_data.settings_addr.clone()))
            .app_data(web::Data::new(app_state_data.metadata_addr.clone()))
            .app_data(web::Data::new(app_state_data.client_manager_addr.clone()))
            .app_data(web::Data::new(app_state_data.workspace_addr.clone()))
            .app_data(app_state_data.nostr_service.clone().unwrap_or_else(|| web::Data::new(NostrService::default()))) // Provide default if None
            .app_data(app_state_data.feature_access.clone())
            // DEPRECATED: hybrid_health_manager_data, mcp_session_bridge, session_correlation_bridge removed
            .route("/wss", web::get().to(socket_flow_handler)) // Changed from /ws to /wss
            .route("/ws/speech", web::get().to(speech_socket_handler))
            .route("/ws/mcp-relay", web::get().to(mcp_relay_handler)) // Legacy MCP relay endpoint
            // DEPRECATED: hybrid health routes removed
            .route("/ws/client-messages", web::get().to(client_messages_handler::websocket_client_messages)) // Agent -> User messages
            .service(
                web::scope("/api") // Add /api prefix for these routes
                    .configure(api_handler::config) // This will now serve /api/user-settings etc.
                    .configure(workspace_handler::config) // Add workspace routes under /api/workspace
                    .service(web::scope("/pages").configure(pages_handler::config))
                    .service(web::scope("/bots").configure(api_handler::bots::config)) // This will now serve /api/bots/data and /api/bots/update
                    .configure(bots_visualization_handler::configure_routes) // Agent visualization endpoints
                    .configure(graph_export_handler::configure_routes) // Graph export and sharing endpoints
                    .route("/client-logs", web::post().to(client_log_handler::handle_client_logs)) // Client browser logs endpoint
                    // DEPRECATED: hybrid health routes removed
            );

            app
        })
        .bind(&bind_address)?
        .workers(4) // Explicitly set the number of worker threads
        .run();

    let server_handle = server.handle();

    // Set up signal handlers
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    tokio::spawn(async move {
        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM signal");
            }
            _ = sigint.recv() => {
                info!("Received SIGINT signal");
            }
        }
        info!("Initiating graceful shutdown");
        server_handle.stop(true).await;
    });

    info!("main: HTTP server startup sequence complete. Server is now running.");
    server.await?;

    info!("HTTP server stopped");
    Ok(())
}
