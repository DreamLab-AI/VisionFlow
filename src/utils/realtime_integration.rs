// Real-time WebSocket Integration Layer
// Connects existing analytics, optimization, and export systems with WebSocket broadcasting

use log::{debug, error, info};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::handlers::realtime_websocket_handler::{
    broadcast_analysis_progress, broadcast_export_progress, broadcast_export_ready,
    broadcast_optimization_update, broadcast_workspace_update,
};

// Integration traits for different systems
#[async_trait::async_trait]
pub trait AnalyticsProgressReporter: Send + Sync {
    async fn report_progress(&self, analysis_id: &str, progress: f64, stage: &str, operation: &str);
    async fn report_completion(&self, analysis_id: &str, results: Value, success: bool);
    async fn report_error(&self, analysis_id: &str, error: &str);
}

#[async_trait::async_trait]
pub trait OptimizationProgressReporter: Send + Sync {
    async fn report_progress(
        &self,
        optimization_id: &str,
        progress: f64,
        algorithm: &str,
        iteration: u64,
        metrics: Value,
    );
    async fn report_completion(&self, optimization_id: &str, results: Value);
}

#[async_trait::async_trait]
pub trait ExportProgressReporter: Send + Sync {
    async fn report_progress(&self, export_id: &str, progress: f64, stage: &str, format: &str);
    async fn report_completion(
        &self,
        export_id: &str,
        download_url: String,
        size: u64,
        format: &str,
    );
}

#[async_trait::async_trait]
pub trait WorkspaceEventReporter: Send + Sync {
    async fn report_workspace_change(&self, workspace_id: &str, operation: &str, changes: Value);
    async fn report_user_activity(&self, workspace_id: &str, user_id: &str, action: &str);
}

// Concrete implementation of the progress reporters
pub struct RealtimeProgressReporter {
    graph_id: Option<String>,
}

impl RealtimeProgressReporter {
    pub fn new(graph_id: Option<String>) -> Self {
        Self { graph_id }
    }
}

#[async_trait::async_trait]
impl AnalyticsProgressReporter for RealtimeProgressReporter {
    async fn report_progress(
        &self,
        analysis_id: &str,
        progress: f64,
        stage: &str,
        operation: &str,
    ) {
        debug!(
            "Reporting analysis progress: {} - {}% - {}",
            analysis_id, progress, stage
        );

        broadcast_analysis_progress(
            analysis_id.to_string(),
            self.graph_id.clone(),
            progress,
            stage.to_string(),
            operation.to_string(),
            None,
        )
        .await;
    }

    async fn report_completion(&self, analysis_id: &str, results: Value, success: bool) {
        info!(
            "Analysis {} completed with success: {}",
            analysis_id, success
        );

        
        let _message = json!({
            "type": "analysis_complete",
            "data": {
                "analysis_id": analysis_id,
                "graph_id": self.graph_id,
                "results": results,
                "success": success,
                "processing_time": 0.0,
                "timestamp": chrono::Utc::now().timestamp_millis()
            }
        });

        
        
        broadcast_analysis_progress(
            analysis_id.to_string(),
            self.graph_id.clone(),
            100.0,
            "completed".to_string(),
            "Analysis finished".to_string(),
            Some(results),
        )
        .await;
    }

    async fn report_error(&self, analysis_id: &str, error: &str) {
        error!("Analysis {} failed: {}", analysis_id, error);

        broadcast_analysis_progress(
            analysis_id.to_string(),
            self.graph_id.clone(),
            0.0,
            "error".to_string(),
            format!("Error: {}", error),
            Some(json!({"error": error})),
        )
        .await;
    }
}

#[async_trait::async_trait]
impl OptimizationProgressReporter for RealtimeProgressReporter {
    async fn report_progress(
        &self,
        optimization_id: &str,
        progress: f64,
        algorithm: &str,
        iteration: u64,
        metrics: Value,
    ) {
        debug!(
            "Reporting optimization progress: {} - {}% - iteration {}",
            optimization_id, progress, iteration
        );

        broadcast_optimization_update(
            optimization_id.to_string(),
            self.graph_id.clone(),
            progress,
            algorithm.to_string(),
            iteration,
            1000, 
            metrics,
        )
        .await;
    }

    async fn report_completion(&self, optimization_id: &str, results: Value) {
        info!("Optimization {} completed", optimization_id);

        
        let algorithm = results
            .get("algorithm")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let confidence = results
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let performance_gain = results
            .get("performance_gain")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let clusters = results
            .get("clusters")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        
        broadcast_optimization_update(
            optimization_id.to_string(),
            self.graph_id.clone(),
            100.0,
            algorithm,
            1000, 
            1000,
            json!({
                "confidence": confidence,
                "performance_gain": performance_gain,
                "clusters": clusters,
                "completed": true
            }),
        )
        .await;
    }
}

#[async_trait::async_trait]
impl ExportProgressReporter for RealtimeProgressReporter {
    async fn report_progress(&self, export_id: &str, progress: f64, stage: &str, format: &str) {
        debug!(
            "Reporting export progress: {} - {}% - {}",
            export_id, progress, stage
        );

        broadcast_export_progress(
            export_id.to_string(),
            self.graph_id.clone(),
            format.to_string(),
            progress,
            stage.to_string(),
        )
        .await;
    }

    async fn report_completion(
        &self,
        export_id: &str,
        download_url: String,
        size: u64,
        format: &str,
    ) {
        info!("Export {} completed: {} bytes", export_id, size);

        broadcast_export_ready(
            export_id.to_string(),
            self.graph_id.clone(),
            format.to_string(),
            download_url,
            size,
        )
        .await;
    }
}

#[async_trait::async_trait]
impl WorkspaceEventReporter for RealtimeProgressReporter {
    async fn report_workspace_change(&self, workspace_id: &str, operation: &str, changes: Value) {
        info!("Workspace {} changed: {}", workspace_id, operation);

        broadcast_workspace_update(
            workspace_id.to_string(),
            changes,
            operation.to_string(),
            None, 
        )
        .await;
    }

    async fn report_user_activity(&self, workspace_id: &str, user_id: &str, action: &str) {
        debug!(
            "User activity in workspace {}: {} - {}",
            workspace_id, user_id, action
        );

        
        broadcast_workspace_update(
            workspace_id.to_string(),
            json!({"user_activity": {"user_id": user_id, "action": action}}),
            "user_activity".to_string(),
            Some(user_id.to_string()),
        )
        .await;
    }
}

// Factory functions for creating reporters
pub fn create_analytics_reporter(
    graph_id: Option<String>,
) -> Arc<dyn AnalyticsProgressReporter + Send + Sync> {
    Arc::new(RealtimeProgressReporter::new(graph_id))
}

pub fn create_optimization_reporter(
    graph_id: Option<String>,
) -> Arc<dyn OptimizationProgressReporter + Send + Sync> {
    Arc::new(RealtimeProgressReporter::new(graph_id))
}

pub fn create_export_reporter(
    graph_id: Option<String>,
) -> Arc<dyn ExportProgressReporter + Send + Sync> {
    Arc::new(RealtimeProgressReporter::new(graph_id))
}

pub fn create_workspace_reporter() -> Arc<dyn WorkspaceEventReporter + Send + Sync> {
    Arc::new(RealtimeProgressReporter::new(None))
}

// Global state for managing active operations
pub struct OperationTracker {
    pub active_analyses: Arc<RwLock<std::collections::HashMap<String, String>>>,
    pub active_optimizations: Arc<RwLock<std::collections::HashMap<String, String>>>,
    pub active_exports: Arc<RwLock<std::collections::HashMap<String, String>>>,
}

impl OperationTracker {
    pub fn new() -> Self {
        Self {
            active_analyses: Arc::new(RwLock::new(std::collections::HashMap::new())),
            active_optimizations: Arc::new(RwLock::new(std::collections::HashMap::new())),
            active_exports: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub async fn start_analysis(&self, analysis_id: String, graph_id: String) {
        let mut analyses = self.active_analyses.write().await;
        analyses.insert(analysis_id.clone(), graph_id);
        info!("Started tracking analysis: {}", analysis_id);
    }

    pub async fn finish_analysis(&self, analysis_id: &str) {
        let mut analyses = self.active_analyses.write().await;
        if analyses.remove(analysis_id).is_some() {
            info!("Finished tracking analysis: {}", analysis_id);
        }
    }

    pub async fn start_optimization(&self, optimization_id: String, graph_id: String) {
        let mut optimizations = self.active_optimizations.write().await;
        optimizations.insert(optimization_id.clone(), graph_id);
        info!("Started tracking optimization: {}", optimization_id);
    }

    pub async fn finish_optimization(&self, optimization_id: &str) {
        let mut optimizations = self.active_optimizations.write().await;
        if optimizations.remove(optimization_id).is_some() {
            info!("Finished tracking optimization: {}", optimization_id);
        }
    }

    pub async fn start_export(&self, export_id: String, graph_id: String) {
        let mut exports = self.active_exports.write().await;
        exports.insert(export_id.clone(), graph_id);
        info!("Started tracking export: {}", export_id);
    }

    pub async fn finish_export(&self, export_id: &str) {
        let mut exports = self.active_exports.write().await;
        if exports.remove(export_id).is_some() {
            info!("Finished tracking export: {}", export_id);
        }
    }

    pub async fn get_active_operations(&self) -> Value {
        let analyses = self.active_analyses.read().await;
        let optimizations = self.active_optimizations.read().await;
        let exports = self.active_exports.read().await;

        json!({
            "analyses": analyses.len(),
            "optimizations": optimizations.len(),
            "exports": exports.len(),
            "total": analyses.len() + optimizations.len() + exports.len()
        })
    }
}

// Global operation tracker instance
use lazy_static::lazy_static;

lazy_static! {
    pub static ref OPERATION_TRACKER: OperationTracker = OperationTracker::new();
}

// Helper macros for easy progress reporting
#[macro_export]
macro_rules! report_analysis_progress {
    ($analysis_id:expr, $progress:expr, $stage:expr, $operation:expr) => {
        if let Some(graph_id) =
            $crate::utils::realtime_integration::get_graph_id_for_analysis($analysis_id).await
        {
            let reporter =
                $crate::utils::realtime_integration::create_analytics_reporter(Some(graph_id));
            reporter
                .report_progress($analysis_id, $progress, $stage, $operation)
                .await;
        }
    };
}

#[macro_export]
macro_rules! report_optimization_progress {
    ($optimization_id:expr, $progress:expr, $algorithm:expr, $iteration:expr, $metrics:expr) => {
        if let Some(graph_id) =
            $crate::utils::realtime_integration::get_graph_id_for_optimization($optimization_id)
                .await
        {
            let reporter =
                $crate::utils::realtime_integration::create_optimization_reporter(Some(graph_id));
            reporter
                .report_progress(
                    $optimization_id,
                    $progress,
                    $algorithm,
                    $iteration,
                    $metrics,
                )
                .await;
        }
    };
}

#[macro_export]
macro_rules! report_export_progress {
    ($export_id:expr, $progress:expr, $stage:expr, $format:expr) => {
        if let Some(graph_id) =
            $crate::utils::realtime_integration::get_graph_id_for_export($export_id).await
        {
            let reporter =
                $crate::utils::realtime_integration::create_export_reporter(Some(graph_id));
            reporter
                .report_progress($export_id, $progress, $stage, $format)
                .await;
        }
    };
}

// Helper functions to get graph IDs from operation IDs
pub async fn get_graph_id_for_analysis(analysis_id: &str) -> Option<String> {
    let analyses = OPERATION_TRACKER.active_analyses.read().await;
    analyses.get(analysis_id).cloned()
}

pub async fn get_graph_id_for_optimization(optimization_id: &str) -> Option<String> {
    let optimizations = OPERATION_TRACKER.active_optimizations.read().await;
    optimizations.get(optimization_id).cloned()
}

pub async fn get_graph_id_for_export(export_id: &str) -> Option<String> {
    let exports = OPERATION_TRACKER.active_exports.read().await;
    exports.get(export_id).cloned()
}
