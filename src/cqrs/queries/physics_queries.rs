// src/cqrs/queries/physics_queries.rs
//! GPU Physics Queries
//!
//! Read operations for GPU physics adapter status and metrics.

use crate::cqrs::types::Query;
use crate::ports::gpu_physics_adapter::{GpuDeviceInfo, PhysicsStatistics};

#[derive(Debug, Clone)]
pub struct GetGpuStatusQuery;

impl Query for GetGpuStatusQuery {
    type Result = GpuDeviceInfo;

    fn name(&self) -> &'static str {
        "GetGpuStatus"
    }
}

#[derive(Debug, Clone)]
pub struct GetPhysicsStatisticsQuery;

impl Query for GetPhysicsStatisticsQuery {
    type Result = PhysicsStatistics;

    fn name(&self) -> &'static str {
        "GetPhysicsStatistics"
    }
}

#[derive(Debug, Clone)]
pub struct ListGpuDevicesQuery;

impl Query for ListGpuDevicesQuery {
    type Result = Vec<GpuDeviceInfo>;

    fn name(&self) -> &'static str {
        "ListGpuDevices"
    }
}

#[derive(Debug, Clone)]
pub struct GetPerformanceMetricsQuery {
    pub metric_type: PerformanceMetricType,
}

#[derive(Debug, Clone)]
pub enum PerformanceMetricType {
    StepTime,
    Energy,
    MemoryUsage,
    CacheHitRate,
    All,
}

impl Query for GetPerformanceMetricsQuery {
    type Result = PerformanceMetrics;

    fn name(&self) -> &'static str {
        "GetPerformanceMetrics"
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub average_step_time_ms: Option<f32>,
    pub average_energy: Option<f32>,
    pub gpu_memory_used_mb: Option<f32>,
    pub cache_hit_rate: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct IsGpuAvailableQuery;

impl Query for IsGpuAvailableQuery {
    type Result = bool;

    fn name(&self) -> &'static str {
        "IsGpuAvailable"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_names() {
        let query = GetGpuStatusQuery;
        assert_eq!(query.name(), "GetGpuStatus");

        let query = GetPhysicsStatisticsQuery;
        assert_eq!(query.name(), "GetPhysicsStatistics");

        let query = ListGpuDevicesQuery;
        assert_eq!(query.name(), "ListGpuDevices");
    }
}
