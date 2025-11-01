pub mod admin_sync_handler;
// DEPRECATED: admin_bridge_handler removed (legacy ontology bridge)
pub mod api_handler;
pub mod bots_handler;
pub mod bots_visualization_handler;
pub mod client_log_handler;
pub mod client_messages_handler;
pub mod clustering_handler;
pub mod consolidated_health_handler; 
pub mod constraints_handler;
pub mod graph_export_handler;
pub mod graph_state_handler; 
                             
pub mod mcp_relay_handler;
pub mod multi_mcp_websocket_handler;
pub mod nostr_handler;
pub mod ontology_handler; 
pub mod pages_handler;
pub mod perplexity_handler;
pub mod ragflow_handler;
pub mod realtime_websocket_handler;
pub mod settings_handler; 
                          
pub mod settings_validation_fix;
pub mod socket_flow_handler;
pub mod speech_socket_handler;
pub mod utils; 
pub mod validation_handler;
pub mod websocket_settings_handler;
pub mod workspace_handler;

// Phase 5: Hexagonal architecture handlers
pub mod physics_handler;
pub mod semantic_handler;

pub use physics_handler::configure_routes as configure_physics_routes;
pub use semantic_handler::configure_routes as configure_semantic_routes;

// Phase 7: Inference handler
pub mod inference_handler;

pub use inference_handler::configure_routes as configure_inference_routes;

#[cfg(test)]
pub mod tests;
