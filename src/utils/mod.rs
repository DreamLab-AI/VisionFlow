// REMOVED: advanced_gpu_compute module - functionality moved to unified_gpu_compute
// REMOVED: gpu_compute module - legacy implementation replaced by unified_gpu_compute
pub mod actor_timeout;
pub mod advanced_logging;
pub mod async_improvements;
pub mod audio_processor;
pub mod binary_protocol;
pub mod client_message_extractor;
pub mod edge_data;
#[cfg(all(test, feature = "gpu"))]
mod gpu_compute_tests;
#[cfg(feature = "gpu")]
pub mod gpu_diagnostics;
#[cfg(feature = "gpu")]
pub mod gpu_memory;
pub mod gpu_safety;
pub mod handler_commons;
// pub mod hybrid_fault_tolerance; 
// pub mod hybrid_performance_optimizer; 
pub mod logging;
pub mod mcp_connection;
pub mod mcp_tcp_client;
pub mod memory_bounds;
pub mod network;
#[cfg(feature = "gpu")]
pub mod ptx;
#[cfg(all(test, feature = "gpu"))]
mod ptx_tests;
pub mod realtime_integration;
pub mod resource_monitor;
pub mod session_log_monitor;
pub mod socket_flow_constants;
pub mod socket_flow_messages;
pub mod standard_websocket_messages;
#[cfg(feature = "gpu")]
pub mod unified_gpu_compute;
pub mod validation;
pub mod websocket_heartbeat;
