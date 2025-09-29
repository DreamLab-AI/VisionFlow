pub mod api_handler;
pub mod consolidated_health_handler; // Replaces health_handler and mcp_health_handler
pub mod pages_handler;
pub mod perplexity_handler;
pub mod ragflow_handler;
pub mod settings_handler;
pub mod settings_paths;
pub mod socket_flow_handler;
pub mod speech_socket_handler;
pub mod nostr_handler;
pub mod bots_handler;
pub mod mcp_relay_handler;
pub mod bots_visualization_handler;
pub mod multi_mcp_websocket_handler;
pub mod clustering_handler;
pub mod constraints_handler;
pub mod validation_handler;
pub mod graph_state_handler;
pub mod settings_validation_fix;
pub mod hybrid_health_handler;
pub mod workspace_handler;
pub mod graph_export_handler;
pub mod realtime_websocket_handler;
pub mod websocket_settings_handler;
pub mod client_log_handler;

#[cfg(test)]
pub mod tests;
