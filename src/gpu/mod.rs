//! GPU computation modules for visual analytics and high-performance graph processing
//! 
//! All GPU modules now include comprehensive safety measures, bounds checking,
//! and error handling by default.

// Primary safe implementations (formerly safe_*)
pub mod streaming_pipeline;
pub mod visual_analytics;

// Legacy implementations (retained for compatibility)
pub mod streaming_pipeline_legacy;
pub mod visual_analytics_legacy;

// Primary exports (safe by default)
pub use visual_analytics::{
    VisualAnalyticsGPU, VisualAnalyticsParams, VisualAnalyticsBuilder, VisualAnalyticsEngine,
    TSNode, TSEdge, IsolationLayer, Vec4, RenderData, PerformanceMetrics
};

pub use streaming_pipeline::{
    StreamingPipeline, SimplifiedNode, CompressedEdge, ClientLOD, FrameBuffer,
    ClientConnection, ClientStats, PipelineStats, DeltaCompressor, StreamMessage
};

// Legacy compatibility exports (unsafe, deprecated)
#[deprecated(note = "Use the safe versions from visual_analytics module instead")]
pub mod legacy {
    #[allow(unused_imports)]
    pub use crate::gpu::visual_analytics_legacy::{
        Vec4 as LegacyVec4, TSNode as LegacyTSNode, TSEdge as LegacyTSEdge, 
        IsolationLayer as LegacyIsolationLayer, VisualAnalyticsParams as LegacyVisualAnalyticsParams,
        VisualAnalyticsGPU as LegacyVisualAnalyticsGPU, RenderData as LegacyRenderData
    };
    #[allow(unused_imports)]
    pub use crate::gpu::streaming_pipeline_legacy::*;
}