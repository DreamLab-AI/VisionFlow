// REMOVED: advanced_gpu_compute module - functionality moved to unified_gpu_compute
// REMOVED: gpu_compute module - legacy implementation replaced by unified_gpu_compute
pub mod audio_processor;
pub mod binary_protocol;
// REMOVED: pub mod edge_data; - unused struct
pub mod gpu_diagnostics;
pub mod gpu_safety;
pub mod logging;
pub mod memory_bounds;
pub mod mcp_connection;
pub mod network;
pub mod resource_monitor;
// REMOVED: pub mod socket_flow_constants; - unused constants
pub mod socket_flow_messages;
pub mod unified_gpu_compute;
pub mod validation;
