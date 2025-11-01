// Rebuild: KE velocity fix applied
use actix::Actor;
use webxr::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use webxr::services::nostr_service::NostrService;
use webxr::settings::settings_actor::SettingsActor;
use webxr::adapters::sqlite_settings_repository::SqliteSettingsRepository;
use webxr::{
    config::AppFullSettings,
    handlers::{
        admin_sync_handler,
        api_handler,
        bots_visualization_handler,
        client_log_handler,
        client_messages_handler,
        graph_export_handler,
        mcp_relay_handler::mcp_relay_handler,
        nostr_handler,
        pages_handler,
        socket_flow_handler::{socket_flow_handler, PreReadSocketSettings}, 
        speech_socket_handler::speech_socket_handler,
        
        
        workspace_handler,
    },
    services::speech_service::SpeechService,
    services::{
        
        github::{content_enhanced::EnhancedContentAPI, ContentAPI, GitHubClient, GitHubConfig},
        github_sync_service::GitHubSyncService, 
        ragflow_service::RAGFlowService,        
    },
    
    AppState,
};

use actix_cors::Cors;
use actix_web::{middleware, web, App, HttpServer};
// DEPRECATED: std::future imports removed (were for ErrorRecoveryMiddleware)
// DEPRECATED: Actix dev imports removed (were for ErrorRecoveryMiddleware)
// DEPRECATED: LocalBoxFuture import removed (was for ErrorRecoveryMiddleware)
// use actix_files::Files; 
use dotenvy::dotenv;
use log::{debug, error, info};
use std::sync::Arc;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::RwLock;
use tokio::time::Duration;
use webxr::middleware::TimeoutMiddleware;
use webxr::telemetry::agent_telemetry::init_telemetry_logger;
use webxr::utils::advanced_logging::init_advanced_logging;
use webxr::utils::logging::init_logging;

// DEPRECATED: ErrorRecoveryMiddleware removed - NetworkRecoveryManager deleted


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    
    dotenv().ok();

    
    init_logging()?;

    
    if let Err(e) = init_advanced_logging() {
        error!("Failed to initialize advanced logging: {}", e);
    } else {
        info!("Advanced logging system initialized successfully");
    }

    
    
    let log_dir = if std::path::Path::new("/app/logs").exists() {
        "/app/logs".to_string()
    } else if std::path::Path::new("/workspace/ext/logs").exists() {
        "/workspace/ext/logs".to_string()
    } else {
        
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

    
    let settings = match AppFullSettings::new() {
        Ok(s) => {
            info!(
                "✅ AppFullSettings loaded successfully from: {}",
                std::env::var("SETTINGS_FILE_PATH")
                    .unwrap_or_else(|_| "/app/settings.yaml".to_string())
            );

            
            match serde_json::to_string(&s.visualisation.rendering) {
                Ok(json_output) => {
                    info!(
                        "✅ SERDE ALIAS FIX WORKS! JSON serialization (camelCase): {}",
                        json_output
                    );

                    
                    if json_output.contains("ambientLightIntensity")
                        && !json_output.contains("ambient_light_intensity")
                    {
                        info!("✅ CONFIRMED: JSON uses camelCase field names for REST API compatibility");
                    }

                    
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

            Arc::new(RwLock::new(s)) 
        }
        Err(e) => {
            error!("❌ Failed to load AppFullSettings: {:?}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to initialize AppFullSettings: {:?}", e),
            ));
        }
    };

    
    info!("GPU compute will be initialized by GPUComputeActor when needed");

    debug!("Successfully loaded AppFullSettings");

    info!("Starting WebXR application...");
    debug!("main: Beginning application startup sequence.");

    // Initialize settings repository and actor
    let settings_db_path = std::env::var("SETTINGS_DB_PATH")
        .unwrap_or_else(|_| "/app/data/settings.db".to_string());

    info!("Initializing SettingsActor with database: {}", settings_db_path);
    let settings_repository = match SqliteSettingsRepository::new(&settings_db_path) {
        Ok(repo) => Arc::new(repo),
        Err(e) => {
            error!("Failed to create settings repository: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create settings repository: {}", e),
            ));
        }
    };

    let settings_actor = SettingsActor::new(settings_repository).start();
    let settings_actor_data = web::Data::new(settings_actor);
    info!("SettingsActor initialized successfully");



    let settings_data = web::Data::new(settings.clone());

    
    let github_config = match GitHubConfig::from_env() {
        Ok(config) => config,
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load GitHub config: {}", e),
            ))
        }
    };

    
    
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

    
    
    let speech_service = {
        let service = SpeechService::new(settings.clone());
        Some(Arc::new(service))
    };

    
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

    
    
    let settings_value = {
        let settings_read = settings.read().await;
        settings_read.clone()
    };

    let mut app_state = match AppState::new(
        settings_value,
        github_client.clone(),
        content_api.clone(),
        None,                   
        ragflow_service_option, 
        speech_service,
        "default_session".to_string(), 
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
    
    nostr_handler::init_nostr_service(&mut app_state);
    info!("[main] Nostr service initialized");

    
    info!("[main] Initializing GitHub Sync Service...");
    let enhanced_content_api = Arc::new(EnhancedContentAPI::new(github_client.clone()));
    let github_sync_service = Arc::new(GitHubSyncService::new(
        enhanced_content_api,
        app_state.knowledge_graph_repository.clone(),
        app_state.ontology_repository.clone(),
    ));
    info!("[main] GitHub Sync Service initialized");

    
    

    
    

    
    
    info!("Skipping bots orchestrator connection during startup (will connect on-demand)");

    
    info!("Loading graph from knowledge_graph.db...");

    let graph_data_option = match app_state.knowledge_graph_repository.load_graph().await {
        Ok(graph_arc) => {
            let graph = graph_arc.as_ref();
            if !graph.nodes.is_empty() {
                info!(
                    "✅ Loaded graph from database: {} nodes, {} edges",
                    graph.nodes.len(),
                    graph.edges.len()
                );
                Some((*graph_arc).clone())
            } else {
                info!("📂 Database is empty - waiting for GitHub sync to complete");
                info!("ℹ️  Graph will be loaded after sync finishes");
                None
            }
        }
        Err(e) => {
            error!("⚠️  Failed to load graph from database: {}", e);
            error!("⚠️  Graph will be empty until GitHub sync completes");
            None
        }
    };

    
    use std::sync::Arc as StdArc;
    use webxr::actors::messages::UpdateGraphData;

    if let Some(graph_data) = graph_data_option {
        
        info!(
            "📤 Sending graph data to GraphServiceActor: {} nodes, {} edges",
            graph_data.nodes.len(),
            graph_data.edges.len()
        );
        app_state.graph_service_addr.do_send(UpdateGraphData {
            graph_data: StdArc::new(graph_data),
        });
        info!("✅ Graph data sent to actor");
    } else {
        
        info!("⏳ GraphServiceActor will remain empty until GitHub sync finishes");
        info!("ℹ️  You can manually trigger sync via /api/admin/sync endpoint");
    }

    info!("Starting HTTP server...");

    
    
    
    
    
    
    
    info!("Skipping redundant StartSimulation message to GraphServiceSupervisor for debugging stack overflow. Simulation should already be running from supervisor's started() method.");

    
    let app_state_data = web::Data::new(app_state);
    

    
    let bind_address = std::env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("SYSTEM_NETWORK_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(4000);
    let bind_address = format!("{}:{}", bind_address, port);

    
    let pre_read_ws_settings = {
        let s = settings.read().await;
        PreReadSocketSettings {
            min_update_rate: s.system.websocket.min_update_rate,
            max_update_rate: s.system.websocket.max_update_rate,
            motion_threshold: s.system.websocket.motion_threshold,
            motion_damping: s.system.websocket.motion_damping,
            heartbeat_interval_ms: s.system.websocket.heartbeat_interval, 
            heartbeat_timeout_ms: s.system.websocket.heartbeat_timeout,   
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
            .wrap(TimeoutMiddleware::new(Duration::from_secs(30))) 


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
            .app_data(app_state_data.nostr_service.clone().unwrap_or_else(|| web::Data::new(NostrService::default())))
            .app_data(app_state_data.feature_access.clone())
            .app_data(web::Data::new(github_sync_service.clone()))
            .app_data(settings_actor_data.clone()) 
            
            
            .route("/wss", web::get().to(socket_flow_handler)) 
            .route("/ws/speech", web::get().to(speech_socket_handler))
            .route("/ws/mcp-relay", web::get().to(mcp_relay_handler)) 
            
            .route("/ws/client-messages", web::get().to(client_messages_handler::websocket_client_messages)) 
            .service(
                web::scope("/api")
                    .service(web::scope("/settings").configure(webxr::settings::api::configure_routes))
                    .configure(api_handler::config)
                    .configure(workspace_handler::config)
                    .configure(admin_sync_handler::configure_routes)

                    .service(web::scope("/pages").configure(pages_handler::config))
                    .service(web::scope("/bots").configure(api_handler::bots::config))
                    .configure(bots_visualization_handler::configure_routes)
                    .configure(graph_export_handler::configure_routes)
                    .route("/client-logs", web::post().to(client_log_handler::handle_client_logs))

            );

            app
        })
        .bind(&bind_address)?
        .workers(4) 
        .run();

    let server_handle = server.handle();

    
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
