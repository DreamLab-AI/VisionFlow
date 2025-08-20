//! GPU computation modules for visual analytics and high-performance graph processing
//! 
//! All GPU modules now include comprehensive safety measures, bounds checking,
//! and error handling by default.

// Primary safe implementations (formerly safe_*)
pub mod streaming_pipeline;
pub mod visual_analytics;

// Primary exports (safe by default)
pub use visual_analytics::{
    VisualAnalyticsGPU, VisualAnalyticsParams, VisualAnalyticsBuilder, VisualAnalyticsEngine,
    TSNode, TSEdge, IsolationLayer, Vec4, RenderData, PerformanceMetrics
};

pub use streaming_pipeline::{
    StreamingPipeline, SimplifiedNode, CompressedEdge, ClientLOD, FrameBuffer,
    ClientConnection, ClientStats, PipelineStats, DeltaCompressor, StreamMessage
};