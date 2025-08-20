//! GPU computation modules for visual analytics and high-performance graph processing

pub mod streaming_pipeline;
pub mod safe_streaming_pipeline;
pub mod visual_analytics;
pub mod safe_visual_analytics;

pub use visual_analytics::{
    VisualAnalyticsGPU, VisualAnalyticsParams, VisualAnalyticsBuilder, VisualAnalyticsEngine,
    TSNode, TSEdge, IsolationLayer, Vec4, RenderData, PerformanceMetrics
};