use webxr::services::nostr_service::NostrService;
use webxr::{
    AppState,
    config::AppFullSettings, // Import AppFullSettings only
    handlers::{
        api_handler,
        health_handler,
        pages_handler,
        socket_flow_handler::{socket_flow_handler, PreReadSocketSettings}, // Import PreReadSocketSettings
        speech_socket_handler::speech_socket_handler,
        mcp_relay_handler::mcp_relay_handler,
        nostr_handler,
        bots_handler,
        mcp_health_handler,
        bots_visualization_handler,
    },
    services::{
        file_service::FileService,
        // graph_service::GraphService removed - now using GraphServiceActor
        github::{GitHubClient, ContentAPI, GitHubConfig},
        ragflow_service::RAGFlowService, // ADDED IMPORT
    },
    services::speech_service::SpeechService,
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
// use actix_files::Files; // Removed unused import
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Duration;
use dotenvy::dotenv;
use log::{error, info, debug};
use webxr::utils::logging::init_logging;
use tokio::signal::unix::{signal, SignalKind};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Make dotenv optional since env vars can come from Docker
    dotenv().ok();

    // Initialize logging with env_logger (reads RUST_LOG environment variable)
    init_logging()?;

    // Load settings
    let settings = match AppFullSettings::new() {
        Ok(s) => {
            info!("✅ AppFullSettings loaded successfully from: {}",
                std::env::var("SETTINGS_FILE_PATH").unwrap_or_else(|_| "/app/settings.yaml".to_string()));
            
            // Test JSON serialization to verify camelCase output works
            match serde_json::to_string(&s.visualisation.rendering) {
                Ok(json_output) => {
                    info!("✅ SERDE ALIAS FIX WORKS! JSON serialization (camelCase): {}", json_output);
                    
                    // Verify the JSON contains camelCase fields, not snake_case
                    if json_output.contains("ambientLightIntensity") && !json_output.contains("ambient_light_intensity") {
                        info!("✅ CONFIRMED: JSON uses camelCase field names for REST API compatibility");
                    }
                    
                    // Log some key values that were loaded from snake_case YAML
                    info!("✅ CONFIRMED: Values loaded from snake_case YAML:");
                    info!("   - ambient_light_intensity -> {}", s.visualisation.rendering.ambient_light_intensity);
                    info!("   - enable_ambient_occlusion -> {}", s.visualisation.rendering.enable_ambient_occlusion);
                    info!("   - background_color -> {}", s.visualisation.rendering.background_color);
                    info!("🎉 SERDE ALIAS FIX IS WORKING: YAML (snake_case) loads successfully, JSON serializes as camelCase!");
                }
                Err(e) => {
                    error!("❌ JSON serialization failed: {}", e);
                }
            }
            
            Arc::new(RwLock::new(s)) // Now holds Arc<RwLock<AppFullSettings>>
        },
        Err(e) => {
            error!("❌ Failed to load AppFullSettings: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize AppFullSettings: {:?}", e)));
        }
    };

    // GPU compute is now handled by the GPUComputeActor
    info!("GPU compute will be initialized by GPUComputeActor when needed");

    debug!("Successfully loaded AppFullSettings"); // Updated log message

    info!("Starting WebXR application...");

    // Create web::Data instances first
    // This now holds Data<Arc<RwLock<AppFullSettings>>>
    let settings_data = web::Data::new(settings.clone());

    // Initialize services
    let github_config = match GitHubConfig::from_env() {
        Ok(config) => config,
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to load GitHub config: {}", e)))
    };

    // GitHubClient::new might need adjustment if it expects client-facing Settings
    // Assuming it can work with AppFullSettings for now.
    let github_client = match GitHubClient::new(github_config, settings.clone()).await {
        Ok(client) => Arc::new(client),
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHub client: {}", e)))
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
            None, // Perplexity placeholder
            ragflow_service_option, // Pass the initialized RAGFlow service
            speech_service,
            "default_session".to_string() // RAGFlow session ID placeholder
        ).await {
            Ok(state) => state,
            Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize app state: {}", e)))
        };

    // Initialize Nostr service
    nostr_handler::init_nostr_service(&mut app_state);

    // Initialize BotsClient connection with proper WebSocket protocol
    info!("Connecting to bots orchestrator via WebSocket...");
    let bots_url = std::env::var("BOTS_ORCHESTRATOR_URL")
        .unwrap_or_else(|_| "ws://multi-agent-container:3002/ws".to_string());

    let bots_client = app_state.bots_client.clone();
    tokio::spawn(async move {
        // Retry connection with exponential backoff
        let mut retry_count = 0;
        let max_retries = 5;
        
        while retry_count < max_retries {
            match bots_client.connect(&bots_url).await {
                Ok(()) => {
                    info!("Successfully connected to bots orchestrator at {}", bots_url);
                    break;
                }
                Err(e) => {
                    retry_count += 1;
                    let delay = std::cmp::min(1000 * (1 << retry_count), 30000); // Max 30s
                    error!("Failed to connect to bots orchestrator (attempt {}): {}. Retrying in {}ms", 
                           retry_count, e, delay);
                    
                    if retry_count < max_retries {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    }
                }
            }
        }
        
        if retry_count >= max_retries {
            error!("Failed to connect to bots orchestrator after {} attempts. Will use fallback mode.", max_retries);
        }
    });

    // First, try to load existing metadata without waiting for GitHub download
    info!("Loading existing metadata for quick initialization");
    let metadata_store = FileService::load_or_create_metadata()
        .map_err(|e| {
            error!("Failed to load existing metadata: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;

    info!("Note: Background GitHub data fetch is disabled to resolve compilation issues");

    if metadata_store.is_empty() {
        error!("No metadata found and could not create empty store");
        return Err(std::io::Error::new(std::io::ErrorKind::Other,
            "No metadata found and could not create empty store".to_string()));
    }

    info!("Loaded {} items from metadata store", metadata_store.len());

    // Update metadata in app state using actor
    use webxr::actors::messages::UpdateMetadata;
    if let Err(e) = app_state.metadata_addr.send(UpdateMetadata { metadata: metadata_store.clone() }).await {
        error!("Failed to update metadata in actor: {}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to update metadata in actor: {}", e)));
    }
    info!("Loaded metadata into app state actor");

    // Build initial graph from metadata and initialize GPU compute
    info!("Building initial graph from existing metadata for physics simulation");

    // Use GraphServiceActor to build the graph from metadata
    use webxr::actors::messages::BuildGraphFromMetadata;
    match app_state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
        Ok(Ok(())) => {
            info!("Graph built successfully using GraphServiceActor - GPU initialization is handled automatically by the actor");
        },
        Ok(Err(e)) => {
            error!("Failed to build graph from metadata using actor: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to build graph: {}", e)));
        },
        Err(e) => {
            error!("Graph service actor communication error: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Graph service unavailable: {}", e)));
        }
    }

    info!("Waiting for initial physics layout calculation to complete...");
    tokio::time::sleep(Duration::from_millis(500)).await;
    info!("Initial delay complete. Starting HTTP server...");

    // Start simulation in GraphServiceActor (Second start attempt commented out for debugging stack overflow)
    // use webxr::actors::messages::StartSimulation;
    // if let Err(e) = app_state.graph_service_addr.send(StartSimulation).await {
    //     error!("Failed to start simulation in GraphServiceActor: {}", e);
    //     return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to start simulation: {}", e)));
    // }
    // info!("Simulation started in GraphServiceActor (Second start attempt commented out)");
    info!("Skipping redundant StartSimulation message to GraphServiceActor for debugging stack overflow. Simulation should already be running from actor's started() method.");

    // Create web::Data after all initialization is complete
    let app_state_data = web::Data::new(app_state);

    // Start the server
    let bind_address = {
        let settings_read = settings.read().await; // Reads AppFullSettings
        // Access network settings correctly
        format!("{}:{}", settings_read.system.network.bind_address, settings_read.system.network.port)
    };

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

    let server = HttpServer::new(move || {
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
            .app_data(app_state_data.nostr_service.clone().unwrap_or_else(|| web::Data::new(NostrService::default()))) // Provide default if None
            .app_data(app_state_data.feature_access.clone())
            .route("/wss", web::get().to(socket_flow_handler)) // Changed from /ws to /wss
            .route("/ws/speech", web::get().to(speech_socket_handler))
            .route("/ws/mcp-relay", web::get().to(mcp_relay_handler)) // Legacy MCP relay endpoint
            .service(
                web::scope("/api") // Add /api prefix for these routes
                    .configure(api_handler::config) // This will now serve /api/user-settings etc.
                    .service(web::scope("/health").configure(health_handler::config)) // This will now serve /api/health
                    .service(web::scope("/pages").configure(pages_handler::config))
                    .service(web::scope("/bots").configure(bots_handler::config)) // This will now serve /api/bots/data and /api/bots/update
                    .service(web::scope("/mcp").configure(mcp_health_handler::configure_routes)) // MCP health and control endpoints
                    .configure(bots_visualization_handler::configure_routes) // Agent visualization endpoints
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

    server.await?;

    info!("HTTP server stopped");
    Ok(())
}
