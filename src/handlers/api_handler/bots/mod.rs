use actix_web::web;
use crate::handlers::bots_handler::{
    update_bots_graph as bots_update,
    get_bots_data as bots_get,
    initialize_hive_mind_swarm as initialize_swarm,
    get_bots_connection_status,
    get_bots_agents,
    spawn_agent_hybrid,
    remove_task,
    pause_task,
    resume_task,
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
            .route("/spawn-agent-hybrid", web::post().to(spawn_agent_hybrid))
            .route("/remove-task/{id}", web::delete().to(remove_task))
            .route("/pause-task/{id}", web::post().to(pause_task))
            .route("/resume-task/{id}", web::post().to(resume_task))
    );
}