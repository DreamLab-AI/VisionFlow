//! GPU computation modules for visual analytics and high-performance graph processing
//!
//! All GPU modules now include comprehensive safety measures, bounds checking,
//! and error handling by default.

// Canonical GPU type definitions (AUTHORITATIVE)
pub mod types;

// Primary safe implementations (formerly safe_*)
pub mod streaming_pipeline;
pub mod visual_analytics;

// REMOVED: hybrid_sssp module - contained only stub implementations, archived to archive/legacy_code_2025_11_03/

// GPU conversion utilities
pub mod conversion_utils;

// Unified GPU memory management
pub mod memory_manager;
pub mod dynamic_buffer_manager; // Legacy - use memory_manager instead

// Canonical type exports (AUTHORITATIVE SOURCE)
pub use types::{BinaryNodeData, RenderData};

// Primary exports (safe by default)
pub use visual_analytics::{
    IsolationLayer, PerformanceMetrics, TSEdge, TSNode, Vec4, VisualAnalyticsBuilder,
    VisualAnalyticsEngine, VisualAnalyticsGPU, VisualAnalyticsParams,
};

pub use streaming_pipeline::{
    ClientConnection, ClientLOD, ClientStats, CompressedEdge, DeltaCompressor, FrameBuffer,
    PipelineStats, SimplifiedNode, StreamMessage, StreamingPipeline,
};

// REMOVED: Hybrid SSSP exports - module contained only stub implementations

// GPU conversion utilities exports
pub use conversion_utils::{
    allocate_gpu_buffer, calculate_buffer_size, calculate_memory_footprint,
    extract_position_3d, extract_position_vec4, from_gpu_buffer, get_element_count,
    gpu_buffer_to_nodes, gpu_to_positions, gpu_to_positions_4d, nodes_to_gpu_buffer,
    positions_4d_to_gpu, positions_to_gpu, to_gpu_buffer, validate_buffer_size,
    validate_buffer_stride, validate_render_data, ConversionError, GpuNode,
};

// Unified memory management exports (NEW - recommended)
pub use memory_manager::{
    BufferConfig, BufferStats, GpuBuffer, GpuMemoryManager, MemoryStats,
};
