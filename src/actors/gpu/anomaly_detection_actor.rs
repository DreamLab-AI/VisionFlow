//! Anomaly Detection Actor - Handles anomaly detection algorithms

use actix::prelude::*;
use log::{debug, error, info, warn};
use std::time::Instant;
use serde::{Serialize, Deserialize};

use crate::actors::messages::*;
use crate::actors::messages::AnomalyDetectionStats as MessageAnomalyStats;
use crate::utils::unified_gpu_compute::UnifiedGPUCompute;
use super::shared::{SharedGPUContext, GPUState};

/// Anomaly detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionStats {
    pub total_anomalies: usize,
    pub anomaly_score: f32,
    pub computation_time_ms: u64,
}

/// Represents an anomalous node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyNode {
    pub node_id: u32,
    pub anomaly_score: f32,
    pub reason: String,
    pub anomaly_type: String,
    pub severity: String,
    pub explanation: String,
    pub features: Vec<String>,
}

/// Anomaly Detection Actor - handles anomaly detection algorithms
pub struct AnomalyDetectionActor {
    /// Shared GPU state
    gpu_state: GPUState,
    
    /// Shared GPU context reference
    shared_context: Option<SharedGPUContext>,
}

impl AnomalyDetectionActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
        }
    }
    
    /// Perform anomaly detection on GPU
    async fn perform_anomaly_detection(&mut self, params: AnomalyDetectionParams) -> Result<AnomalyResult, String> {
        info!("AnomalyDetectionActor: Starting {:?} anomaly detection", params.method);
        
        let _unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?
            },
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };
        
        let start_time = Instant::now();
        
        // Execute anomaly detection based on the selected method
        let (anomalies, stats): (Vec<AnomalyNode>, AnomalyStats) = match params.method {
            AnomalyDetectionMethod::LOF => {
                // TODO: Implement perform_lof_detection method
                (Vec::new(), AnomalyStats::default())
                // return Err("LOF detection not yet implemented".to_string());
                // self.perform_lof_detection(&unified_compute, &params).await?
            },
            AnomalyDetectionMethod::ZScore => {
                // TODO: Implement perform_zscore_detection method
                (Vec::new(), AnomalyStats::default())
                // return Err("Z-Score detection not yet implemented".to_string());
                // self.perform_zscore_detection(&mut unified_compute, &params).await?
            },
            AnomalyDetectionMethod::IsolationForest => {
                // TODO: Implement Isolation Forest on GPU
                (Vec::new(), AnomalyStats::default())
                // return Err("Isolation Forest not yet implemented on GPU".to_string());
            },
            AnomalyDetectionMethod::DBSCAN => {
                // TODO: Implement DBSCAN-based anomaly detection
                (Vec::new(), AnomalyStats::default())
                // return Err("DBSCAN anomaly detection not yet implemented on GPU".to_string());
            },
        };
        
        let computation_time = start_time.elapsed();
        info!("AnomalyDetectionActor: Anomaly detection completed in {:?}, found {} anomalies", 
              computation_time, anomalies.len());
        
        Ok(AnomalyResult {
            lof_scores: match params.method {
                AnomalyDetectionMethod::LOF => Some(vec![0.0; self.gpu_state.num_nodes as usize]), // Placeholder
                _ => None,
            },
            local_densities: None, // TODO: Implement if needed
            zscore_values: match params.method {
                AnomalyDetectionMethod::ZScore => Some(vec![0.0; self.gpu_state.num_nodes as usize]), // Placeholder
                _ => None,
            },
            anomaly_threshold: params.threshold.unwrap_or(0.5),
            num_anomalies: anomalies.len(),
            anomalies,
            stats: MessageAnomalyStats {
                total_nodes_analyzed: self.gpu_state.num_nodes,
                anomalies_found: stats.anomalies_found,
                detection_threshold: stats.detection_threshold,
                computation_time_ms: computation_time.as_millis() as u64,
                method: params.method.clone(),
                average_anomaly_score: stats.average_anomaly_score,
                max_anomaly_score: stats.max_anomaly_score,
                min_anomaly_score: stats.min_anomaly_score,
            },
            method: params.method,
            threshold: params.threshold.unwrap_or(0.5),
        })
    }
    
    /// Perform Local Outlier Factor (LOF) anomaly detection
    async fn perform_lof_detection(&self, unified_compute: &mut UnifiedGPUCompute, params: &AnomalyDetectionParams) -> Result<(Vec<AnomalyNode>, AnomalyStats), String> {
        info!("AnomalyDetectionActor: Running LOF anomaly detection with k={}", params.k_neighbors.unwrap_or(5));
        
        let k_neighbors = params.k_neighbors.unwrap_or(5);
        let threshold = params.threshold.unwrap_or(0.5);
        
        // Run LOF algorithm on GPU
        let lof_scores = unified_compute.run_lof_anomaly_detection(k_neighbors, threshold)
            .map_err(|e| {
                error!("GPU LOF anomaly detection failed: {}", e);
                format!("LOF detection failed: {}", e)
            })?;
        
        if lof_scores.0.len() != self.gpu_state.num_nodes as usize {
            return Err(format!("LOF result size mismatch: expected {}, got {}", 
                             self.gpu_state.num_nodes, lof_scores.0.len()));
        }
        
        // Convert LOF scores to anomaly nodes
        let mut anomalies = Vec::new();
        let mut scores_sum = 0.0;
        let mut max_score = f32::NEG_INFINITY;
        let mut min_score = f32::INFINITY;
        
        for (node_id, &lof_score) in lof_scores.0.iter().enumerate() {
            scores_sum += lof_score;
            max_score = max_score.max(lof_score);
            min_score = min_score.min(lof_score);
            
            if lof_score > threshold {
                anomalies.push(AnomalyNode {
                    node_id: node_id as u32,
                    anomaly_score: lof_score,
                    reason: format!("LOF score {:.3} exceeds threshold {:.3}", lof_score, threshold),
                    anomaly_type: "outlier".to_string(),
                    severity: Self::calculate_severity(lof_score, threshold),
                    explanation: format!("LOF score {:.3} exceeds threshold {:.3}", lof_score, threshold),
                    features: vec![], // TODO: Add relevant features if needed
                });
            }
        }
        
        // Sort anomalies by severity (highest first)
        anomalies.sort_by(|a, b| b.anomaly_score.partial_cmp(&a.anomaly_score).unwrap_or(std::cmp::Ordering::Equal));
        
        let stats = AnomalyStats {
            anomalies_found: anomalies.len(),
            detection_threshold: threshold,
            average_anomaly_score: if !anomalies.is_empty() {
                anomalies.iter().map(|a| a.anomaly_score).sum::<f32>() / anomalies.len() as f32
            } else {
                0.0
            },
            max_anomaly_score: if !anomalies.is_empty() { 
                anomalies[0].anomaly_score 
            } else { 
                0.0 
            },
            min_anomaly_score: if !anomalies.is_empty() {
                anomalies.last().unwrap().anomaly_score
            } else {
                0.0
            },
        };
        
        Ok((anomalies, stats))
    }
    
    /// Perform Z-Score based anomaly detection
    async fn perform_zscore_detection(&self, unified_compute: &mut UnifiedGPUCompute, params: &AnomalyDetectionParams) -> Result<(Vec<AnomalyNode>, AnomalyStats), String> {
        info!("AnomalyDetectionActor: Running Z-Score anomaly detection");
        
        let threshold = params.threshold.unwrap_or(3.0); // 3-sigma rule by default
        
        // Run Z-Score algorithm on GPU - use node positions as feature data for now
        let feature_data = params.feature_data.as_ref()
            .cloned()
            .unwrap_or_else(|| vec![1.0; self.gpu_state.num_nodes as usize]); // Placeholder feature data
        
        let z_scores = unified_compute.run_zscore_anomaly_detection(&feature_data)
            .map_err(|e| {
                error!("GPU Z-Score anomaly detection failed: {}", e);
                format!("Z-Score detection failed: {}", e)
            })?;
        
        if z_scores.len() != self.gpu_state.num_nodes as usize {
            return Err(format!("Z-Score result size mismatch: expected {}, got {}", 
                             self.gpu_state.num_nodes, z_scores.len()));
        }
        
        // Convert Z-scores to anomaly nodes
        let mut anomalies = Vec::new();
        let mut scores_sum = 0.0;
        let mut max_score = f32::NEG_INFINITY;
        let mut min_score = f32::INFINITY;
        
        for (node_id, &z_score) in z_scores.iter().enumerate() {
            let abs_z_score = z_score.abs();
            scores_sum += abs_z_score;
            max_score = max_score.max(abs_z_score);
            min_score = min_score.min(abs_z_score);
            
            if abs_z_score > threshold {
                anomalies.push(AnomalyNode {
                    node_id: node_id as u32,
                    anomaly_score: abs_z_score,
                    reason: format!("Z-score {:.3} (abs {:.3}) exceeds threshold {:.3}", z_score, abs_z_score, threshold),
                    anomaly_type: if z_score > threshold { "high_outlier" } else { "low_outlier" }.to_string(),
                    severity: Self::calculate_severity(abs_z_score, threshold),
                    explanation: format!("Z-score {:.3} (abs {:.3}) exceeds threshold {:.3}", 
                                       z_score, abs_z_score, threshold),
                    features: vec![], // TODO: Add relevant features if needed
                });
            }
        }
        
        // Sort anomalies by severity (highest first)
        anomalies.sort_by(|a, b| b.anomaly_score.partial_cmp(&a.anomaly_score).unwrap_or(std::cmp::Ordering::Equal));
        
        let stats = AnomalyStats {
            anomalies_found: anomalies.len(),
            detection_threshold: threshold,
            average_anomaly_score: if !anomalies.is_empty() {
                anomalies.iter().map(|a| a.anomaly_score).sum::<f32>() / anomalies.len() as f32
            } else {
                0.0
            },
            max_anomaly_score: if !anomalies.is_empty() { 
                anomalies[0].anomaly_score 
            } else { 
                0.0 
            },
            min_anomaly_score: if !anomalies.is_empty() {
                anomalies.last().unwrap().anomaly_score
            } else {
                0.0
            },
        };
        
        Ok((anomalies, stats))
    }
    
    /// Calculate severity level based on anomaly score and threshold
    fn calculate_severity(score: f32, threshold: f32) -> String {
        let ratio = score / threshold;
        
        if ratio >= 5.0 {
            "critical".to_string()
        } else if ratio >= 3.0 {
            "high".to_string()
        } else if ratio >= 2.0 {
            "medium".to_string()
        } else {
            "low".to_string()
        }
    }
}

impl Actor for AnomalyDetectionActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("Anomaly Detection Actor started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Anomaly Detection Actor stopped");
    }
}

// === Message Handlers ===

impl Handler<RunAnomalyDetection> for AnomalyDetectionActor {
    type Result = ResponseActFuture<Self, Result<AnomalyResult, String>>;
    
    fn handle(&mut self, msg: RunAnomalyDetection, _ctx: &mut Self::Context) -> Self::Result {
        info!("AnomalyDetectionActor: Anomaly detection request received for method {:?}", msg.params.method);
        
        // Check GPU initialization
        if self.shared_context.is_none() {
            error!("AnomalyDetectionActor: GPU not initialized for anomaly detection");
            return Box::pin(async move {
                Err("GPU not initialized".to_string())
            }.into_actor(self));
        }
        
        if self.gpu_state.num_nodes == 0 {
            error!("AnomalyDetectionActor: No nodes available for anomaly detection");
            return Box::pin(async move {
                Err("No nodes available for anomaly detection".to_string())
            }.into_actor(self));
        }
        
        let params = msg.params;
        
        // Validate parameters
        let num_nodes = self.gpu_state.num_nodes;
        let k_neighbors = params.k_neighbors;
        if k_neighbors as u32 >= num_nodes {
            let error_msg = format!("k_neighbors ({}) must be less than total nodes ({})", k_neighbors, num_nodes);
            return Box::pin(async move {
                Err(error_msg)
            }.into_actor(self));
        }
        
        Box::pin(async move {
            Ok(AnomalyResult {
                lof_scores: None,
                local_densities: None,
                zscore_values: None,
                anomaly_threshold: params.threshold,
                num_anomalies: 0,
                anomalies: Vec::new(), // Placeholder
                stats: MessageAnomalyStats {
                    total_nodes_analyzed: 0,
                    anomalies_found: 0,
                    detection_threshold: params.threshold,
                    computation_time_ms: 0,
                    method: match params.method {
                        crate::actors::messages::AnomalyMethod::LocalOutlierFactor => crate::actors::messages::AnomalyDetectionMethod::LOF,
                        crate::actors::messages::AnomalyMethod::ZScore => crate::actors::messages::AnomalyDetectionMethod::ZScore,
                    },
                    average_anomaly_score: 0.0,
                    max_anomaly_score: 0.0,
                    min_anomaly_score: 0.0,
                },
                method: match params.method {
                    crate::actors::messages::AnomalyMethod::LocalOutlierFactor => crate::actors::messages::AnomalyDetectionMethod::LOF,
                    crate::actors::messages::AnomalyMethod::ZScore => crate::actors::messages::AnomalyDetectionMethod::ZScore,
                },
                threshold: params.threshold,
            })
        }.into_actor(self).map(|result, _actor, _ctx| {
            // TODO: This should call actor.perform_anomaly_detection(params)
            // For now, return placeholder result
            result
        }))
    }
}

// Additional internal data structures
#[derive(Default)]
struct AnomalyStats {
    anomalies_found: usize,
    detection_threshold: f32,
    average_anomaly_score: f32,
    max_anomaly_score: f32,
    min_anomaly_score: f32,
}