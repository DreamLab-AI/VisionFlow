use actix_web::web;
use crate::handlers::bots_handler::{
    update_bots_graph as bots_update,
    get_bots_data as bots_get,
    initialize_hive_mind_swarm as initialize_swarm,
    get_bots_connection_status,
    get_bots_agents,
};

// Configure bots API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/bots")
            .route("/data", web::get().to(bots_get))
            .route("/data", web::post().to(bots_update))
            .route("/update", web::post().to(bots_update))
            .route("/initialize-swarm", web::post().to(initialize_swarm))
            .route("/status", web::get().to(get_bots_connection_status))
            .route("/agents", web::get().to(get_bots_agents))
    );
}