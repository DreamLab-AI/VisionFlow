// REMOVED: advanced_gpu_compute module - functionality moved to unified_gpu_compute
// REMOVED: gpu_compute module - legacy implementation replaced by unified_gpu_compute
pub mod audio_processor;
pub mod binary_protocol;
pub mod edge_data;
pub mod gpu_diagnostics;
pub mod gpu_safety;
pub mod logging;
pub mod advanced_logging;
pub mod memory_bounds;
pub mod mcp_connection;
pub mod mcp_tcp_client;
pub mod network;
pub mod resource_monitor;
pub mod socket_flow_constants;
pub mod socket_flow_messages;
pub mod unified_gpu_compute;
pub mod ptx;
#[cfg(test)]
mod ptx_tests;
#[cfg(test)]
mod gpu_compute_tests;
pub mod validation;
pub mod docker_hive_mind;
pub mod hybrid_fault_tolerance;
pub mod hybrid_performance_optimizer;
pub mod gpu_memory;
pub mod async_improvements;
pub mod websocket_heartbeat;
pub mod handler_commons;
pub mod standard_websocket_messages;
pub mod realtime_integration;
pub mod client_message_extractor;
pub mod session_log_monitor;
