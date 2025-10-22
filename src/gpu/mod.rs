//! GPU computation modules for visual analytics and high-performance graph processing
//!
//! All GPU modules now include comprehensive safety measures, bounds checking,
//! and error handling by default.

// Primary safe implementations (formerly safe_*)
pub mod streaming_pipeline;
pub mod visual_analytics;

// Hybrid CPU-WASM/GPU SSSP implementation
pub mod hybrid_sssp;

// Primary exports (safe by default)
pub use visual_analytics::{
    IsolationLayer, PerformanceMetrics, RenderData, TSEdge, TSNode, Vec4, VisualAnalyticsBuilder,
    VisualAnalyticsEngine, VisualAnalyticsGPU, VisualAnalyticsParams,
};

pub use streaming_pipeline::{
    ClientConnection, ClientLOD, ClientStats, CompressedEdge, DeltaCompressor, FrameBuffer,
    PipelineStats, SimplifiedNode, StreamMessage, StreamingPipeline,
};

// Hybrid SSSP exports
pub use hybrid_sssp::{HybridSSPConfig, HybridSSPExecutor, HybridSSPResult, SSPMetrics};
