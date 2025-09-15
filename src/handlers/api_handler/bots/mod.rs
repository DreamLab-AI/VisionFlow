use actix_web::web;
use crate::handlers::bots_handler::{
    update_bots_data as bots_update,
    get_bots_data as bots_get,
    disconnect_multi_agent,
};

// Re-export the handlers
pub use crate::handlers::bots_handler::{
    update_bots_data,
    get_bots_data,
    get_bots_positions,
    initialize_swarm,
    initialize_multi_agent,
    check_mcp_connection,
    disconnect_multi_agent as disconnect_handler,
    spawn_agent,
};

// Configure bots API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/bots")
            .route("/data", web::get().to(bots_get))
            .route("/data", web::post().to(bots_update))
            .route("/update", web::post().to(bots_update))
            .route("/initialize-swarm", web::post().to(initialize_swarm))
            .route("/initialize-multi-agent", web::post().to(initialize_multi_agent))
            .route("/mcp-status", web::get().to(check_mcp_connection))
            .route("/disconnect-multi-agent", web::post().to(disconnect_multi_agent))
            .route("/spawn-agent", web::post().to(spawn_agent))
    );
}