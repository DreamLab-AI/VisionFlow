//! Anomaly Detection Actor - Handles anomaly detection algorithms

use actix::prelude::*;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::shared::{GPUState, SharedGPUContext};
use crate::actors::messages::AnomalyDetectionStats as MessageAnomalyStats;
use crate::actors::messages::*;
use crate::utils::unified_gpu_compute::UnifiedGPUCompute;

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionStats {
    pub total_anomalies: usize,
    pub anomaly_score: f32,
    pub computation_time_ms: u64,
}

///
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

///
pub struct AnomalyDetectionActor {
    
    gpu_state: GPUState,

    
    shared_context: Option<Arc<SharedGPUContext>>,
}

impl AnomalyDetectionActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
        }
    }

    
    async fn perform_anomaly_detection(
        &mut self,
        params: AnomalyDetectionParams,
    ) -> Result<AnomalyResult, String> {
        info!(
            "AnomalyDetectionActor: Starting {:?} anomaly detection",
            params.method
        );

        let mut unified_compute = match &self.shared_context {
            Some(ctx) => ctx
                .unified_compute
                .lock()
                .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?,
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };

        let start_time = Instant::now();

        
        let (anomalies, stats): (Vec<AnomalyNode>, AnomalyStats) = match params.method {
            AnomalyDetectionMethod::LOF => {
                self.perform_lof_detection(&mut unified_compute, &params)
                    .await?
            }
            AnomalyDetectionMethod::ZScore => {
                self.perform_zscore_detection(&mut unified_compute, &params)
                    .await?
            }
            AnomalyDetectionMethod::IsolationForest => {
                self.perform_isolation_forest_detection(&mut unified_compute, &params)
                    .await?
            }
            AnomalyDetectionMethod::DBSCAN => {
                self.perform_dbscan_anomaly_detection(&mut unified_compute, &params)
                    .await?
            }
        };

        let computation_time = start_time.elapsed();
        info!(
            "AnomalyDetectionActor: Anomaly detection completed in {:?}, found {} anomalies",
            computation_time,
            anomalies.len()
        );

        Ok(AnomalyResult {
            lof_scores: match params.method {
                AnomalyDetectionMethod::LOF => {
                    
                    let lof_scores: Vec<f32> = anomalies
                        .iter()
                        .enumerate()
                        .map(|(idx, anomaly)| {
                            if anomaly.node_id == idx as u32 {
                                anomaly.anomaly_score
                            } else {
                                
                                anomalies
                                    .iter()
                                    .find(|a| a.node_id == idx as u32)
                                    .map(|a| a.anomaly_score)
                                    .unwrap_or(0.0)
                            }
                        })
                        .collect();
                    
                    let mut full_scores = vec![0.0; self.gpu_state.num_nodes as usize];
                    for anomaly in &anomalies {
                        if (anomaly.node_id as usize) < full_scores.len() {
                            full_scores[anomaly.node_id as usize] = anomaly.anomaly_score;
                        }
                    }
                    Some(full_scores)
                }
                _ => None,
            },
            local_densities: None,
            zscore_values: match params.method {
                AnomalyDetectionMethod::ZScore => {
                    
                    let mut full_scores = vec![0.0; self.gpu_state.num_nodes as usize];
                    for anomaly in &anomalies {
                        if (anomaly.node_id as usize) < full_scores.len() {
                            full_scores[anomaly.node_id as usize] = anomaly.anomaly_score;
                        }
                    }
                    Some(full_scores)
                }
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

    
    async fn perform_lof_detection(
        &self,
        unified_compute: &mut UnifiedGPUCompute,
        params: &AnomalyDetectionParams,
    ) -> Result<(Vec<AnomalyNode>, AnomalyStats), String> {
        info!(
            "AnomalyDetectionActor: Running LOF anomaly detection with k={}",
            params.k_neighbors.unwrap_or(5)
        );

        let k_neighbors = params.k_neighbors.unwrap_or(5);
        let threshold = params.threshold.unwrap_or(0.5);

        
        let lof_scores = unified_compute
            .run_lof_anomaly_detection(k_neighbors, threshold)
            .map_err(|e| {
                error!("GPU LOF anomaly detection failed: {}", e);
                format!("LOF detection failed: {}", e)
            })?;

        if lof_scores.0.len() != self.gpu_state.num_nodes as usize {
            return Err(format!(
                "LOF result size mismatch: expected {}, got {}",
                self.gpu_state.num_nodes,
                lof_scores.0.len()
            ));
        }

        
        let mut anomalies = Vec::new();
        let mut _scores_sum = 0.0;
        let mut _max_score = f32::NEG_INFINITY;
        let mut _min_score = f32::INFINITY;

        for (node_id, &lof_score) in lof_scores.0.iter().enumerate() {
            _scores_sum += lof_score;
            _max_score = _max_score.max(lof_score);
            _min_score = _min_score.min(lof_score);

            if lof_score > threshold {
                anomalies.push(AnomalyNode {
                    node_id: node_id as u32,
                    anomaly_score: lof_score,
                    reason: format!(
                        "LOF score {:.3} exceeds threshold {:.3}",
                        lof_score, threshold
                    ),
                    anomaly_type: "outlier".to_string(),
                    severity: Self::calculate_severity(lof_score, threshold),
                    explanation: format!(
                        "LOF score {:.3} exceeds threshold {:.3}",
                        lof_score, threshold
                    ),
                    features: vec![
                        "lof_score".to_string(),
                        "local_density".to_string(),
                        "reachability".to_string(),
                    ],
                });
            }
        }

        
        anomalies.sort_by(|a, b| {
            b.anomaly_score
                .partial_cmp(&a.anomaly_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
                anomalies.last().expect("Expected non-empty collection").anomaly_score
            } else {
                0.0
            },
        };

        Ok((anomalies, stats))
    }

    
    async fn perform_zscore_detection(
        &self,
        unified_compute: &mut UnifiedGPUCompute,
        params: &AnomalyDetectionParams,
    ) -> Result<(Vec<AnomalyNode>, AnomalyStats), String> {
        info!("AnomalyDetectionActor: Running Z-Score anomaly detection");

        let threshold = params.threshold.unwrap_or(3.0); 

        
        let feature_data = params.feature_data.as_ref().cloned().unwrap_or_else(|| {
            
            (0..self.gpu_state.num_nodes)
                .map(|i| {
                    let base_val = (i as f32 + 1.0) / self.gpu_state.num_nodes as f32;
                    
                    base_val + (i as f32).sin() * 0.1 + (i as f32).cos() * 0.05
                })
                .collect()
        });

        let z_scores = unified_compute
            .run_zscore_anomaly_detection(&feature_data)
            .map_err(|e| {
                error!("GPU Z-Score anomaly detection failed: {}", e);
                format!("Z-Score detection failed: {}", e)
            })?;

        if z_scores.len() != self.gpu_state.num_nodes as usize {
            return Err(format!(
                "Z-Score result size mismatch: expected {}, got {}",
                self.gpu_state.num_nodes,
                z_scores.len()
            ));
        }

        
        let mut anomalies = Vec::new();
        let mut _scores_sum = 0.0;
        let mut _max_score = f32::NEG_INFINITY;
        let mut _min_score = f32::INFINITY;

        for (node_id, &z_score) in z_scores.iter().enumerate() {
            let abs_z_score = z_score.abs();
            _scores_sum += abs_z_score;
            _max_score = _max_score.max(abs_z_score);
            _min_score = _min_score.min(abs_z_score);

            if abs_z_score > threshold {
                anomalies.push(AnomalyNode {
                    node_id: node_id as u32,
                    anomaly_score: abs_z_score,
                    reason: format!(
                        "Z-score {:.3} (abs {:.3}) exceeds threshold {:.3}",
                        z_score, abs_z_score, threshold
                    ),
                    anomaly_type: if z_score > threshold {
                        "high_outlier"
                    } else {
                        "low_outlier"
                    }
                    .to_string(),
                    severity: Self::calculate_severity(abs_z_score, threshold),
                    explanation: format!(
                        "Z-score {:.3} (abs {:.3}) exceeds threshold {:.3}",
                        z_score, abs_z_score, threshold
                    ),
                    features: vec!["z_score".to_string(), "statistical_deviation".to_string()],
                });
            }
        }

        
        anomalies.sort_by(|a, b| {
            b.anomaly_score
                .partial_cmp(&a.anomaly_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
                anomalies.last().expect("Expected non-empty collection").anomaly_score
            } else {
                0.0
            },
        };

        Ok((anomalies, stats))
    }

    
    async fn perform_isolation_forest_detection(
        &self,
        unified_compute: &mut UnifiedGPUCompute,
        params: &AnomalyDetectionParams,
    ) -> Result<(Vec<AnomalyNode>, AnomalyStats), String> {
        info!("AnomalyDetectionActor: Running Isolation Forest anomaly detection");

        let threshold = params.threshold.unwrap_or(0.5);
        let num_trees = 100; 

        
        let (pos_x, pos_y, pos_z) = unified_compute
            .get_node_positions()
            .map_err(|e| format!("Failed to get node positions: {}", e))?;

        
        let mut features = Vec::new();
        for i in 0..self.gpu_state.num_nodes as usize {
            features.extend_from_slice(&[pos_x[i], pos_y[i], pos_z[i]]);
        }

        
        let isolation_scores = self.compute_isolation_scores(&features, num_trees);

        if isolation_scores.len() != self.gpu_state.num_nodes as usize {
            return Err(format!(
                "Isolation Forest result size mismatch: expected {}, got {}",
                self.gpu_state.num_nodes,
                isolation_scores.len()
            ));
        }

        
        let mut anomalies = Vec::new();
        let mut _scores_sum = 0.0;
        let mut _max_score = f32::NEG_INFINITY;
        let mut _min_score = f32::INFINITY;

        for (node_id, &score) in isolation_scores.iter().enumerate() {
            _scores_sum += score;
            _max_score = _max_score.max(score);
            _min_score = _min_score.min(score);

            if score > threshold {
                anomalies.push(AnomalyNode {
                    node_id: node_id as u32,
                    anomaly_score: score,
                    reason: format!(
                        "Isolation score {:.3} exceeds threshold {:.3}",
                        score, threshold
                    ),
                    anomaly_type: "isolated_outlier".to_string(),
                    severity: Self::calculate_severity(score, threshold),
                    explanation: format!(
                        "Isolation Forest score {:.3} indicates anomalous behavior",
                        score
                    ),
                    features: vec!["position".to_string()],
                });
            }
        }

        
        anomalies.sort_by(|a, b| {
            b.anomaly_score
                .partial_cmp(&a.anomaly_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
                anomalies.last().expect("Expected non-empty collection").anomaly_score
            } else {
                0.0
            },
        };

        Ok((anomalies, stats))
    }

    
    async fn perform_dbscan_anomaly_detection(
        &self,
        unified_compute: &mut UnifiedGPUCompute,
        params: &AnomalyDetectionParams,
    ) -> Result<(Vec<AnomalyNode>, AnomalyStats), String> {
        info!("AnomalyDetectionActor: Running DBSCAN anomaly detection");

        let eps = params.threshold.unwrap_or(50.0); 
        let min_pts = 3; 

        
        let cluster_labels = unified_compute
            .run_dbscan_clustering(eps, min_pts)
            .map_err(|e| format!("DBSCAN clustering failed: {}", e))?;

        if cluster_labels.len() != self.gpu_state.num_nodes as usize {
            return Err(format!(
                "DBSCAN result size mismatch: expected {}, got {}",
                self.gpu_state.num_nodes,
                cluster_labels.len()
            ));
        }

        
        let mut anomalies = Vec::new();
        let mut _noise_count = 0;

        for (node_id, &label) in cluster_labels.iter().enumerate() {
            if label == -1 {
                
                _noise_count += 1;
                let anomaly_score = 1.0; 

                anomalies.push(AnomalyNode {
                    node_id: node_id as u32,
                    anomaly_score,
                    reason: format!(
                        "Node classified as noise by DBSCAN (eps={:.2}, min_pts={})",
                        eps, min_pts
                    ),
                    anomaly_type: "noise_outlier".to_string(),
                    severity: "high".to_string(),
                    explanation:
                        "DBSCAN identified this node as noise (not belonging to any cluster)"
                            .to_string(),
                    features: vec!["spatial_isolation".to_string()],
                });
            }
        }

        let threshold = 0.5; 
        let stats = AnomalyStats {
            anomalies_found: anomalies.len(),
            detection_threshold: threshold,
            average_anomaly_score: if !anomalies.is_empty() { 1.0 } else { 0.0 },
            max_anomaly_score: if !anomalies.is_empty() { 1.0 } else { 0.0 },
            min_anomaly_score: if !anomalies.is_empty() { 1.0 } else { 0.0 },
        };

        Ok((anomalies, stats))
    }

    
    fn compute_isolation_scores(&self, features: &[f32], num_trees: usize) -> Vec<f32> {
        let num_nodes = self.gpu_state.num_nodes as usize;
        let feature_dim = 3; 
        let mut isolation_scores = vec![0.0f32; num_nodes];

        let mut rng = rand::thread_rng();

        
        for _tree in 0..num_trees {
            let mut path_lengths = vec![0.0f32; num_nodes];

            
            for node_idx in 0..num_nodes {
                let node_features = &features[node_idx * feature_dim..(node_idx + 1) * feature_dim];
                path_lengths[node_idx] = self.compute_isolation_path_length(
                    node_features,
                    features,
                    feature_dim,
                    &mut rng,
                );
            }

            
            let max_depth = (num_nodes as f32).log2().ceil() as usize;
            for node_idx in 0..num_nodes {
                let normalized_score = 1.0 - (path_lengths[node_idx] / max_depth as f32);
                isolation_scores[node_idx] += normalized_score;
            }
        }

        
        for score in &mut isolation_scores {
            *score /= num_trees as f32;
        }

        isolation_scores
    }

    
    fn compute_isolation_path_length(
        &self,
        point: &[f32],
        all_features: &[f32],
        feature_dim: usize,
        rng: &mut rand::rngs::ThreadRng,
    ) -> f32 {
        let _num_nodes = all_features.len() / feature_dim;
        let max_depth = 10; 

        self.isolation_path_recursive(point, all_features, feature_dim, 0, max_depth, rng)
    }

    
    fn isolation_path_recursive(
        &self,
        point: &[f32],
        features: &[f32],
        feature_dim: usize,
        depth: usize,
        max_depth: usize,
        rng: &mut rand::rngs::ThreadRng,
    ) -> f32 {
        use rand::Rng;

        if depth >= max_depth || features.len() < feature_dim * 2 {
            return depth as f32;
        }

        
        let split_feature = rng.gen_range(0..feature_dim);

        
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for node_idx in 0..(features.len() / feature_dim) {
            let feature_val = features[node_idx * feature_dim + split_feature];
            min_val = min_val.min(feature_val);
            max_val = max_val.max(feature_val);
        }

        if max_val <= min_val {
            return depth as f32;
        }

        let split_val = rng.gen_range(min_val..max_val);

        
        if point[split_feature] < split_val {
            
            let mut left_features = Vec::new();
            for node_idx in 0..(features.len() / feature_dim) {
                let node_features = &features[node_idx * feature_dim..(node_idx + 1) * feature_dim];
                if node_features[split_feature] < split_val {
                    left_features.extend_from_slice(node_features);
                }
            }
            self.isolation_path_recursive(
                point,
                &left_features,
                feature_dim,
                depth + 1,
                max_depth,
                rng,
            )
        } else {
            
            let mut right_features = Vec::new();
            for node_idx in 0..(features.len() / feature_dim) {
                let node_features = &features[node_idx * feature_dim..(node_idx + 1) * feature_dim];
                if node_features[split_feature] >= split_val {
                    right_features.extend_from_slice(node_features);
                }
            }
            self.isolation_path_recursive(
                point,
                &right_features,
                feature_dim,
                depth + 1,
                max_depth,
                rng,
            )
        }
    }

    
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
        info!(
            "AnomalyDetectionActor: Anomaly detection request received for method {:?}",
            msg.params.method
        );

        
        if self.shared_context.is_none() {
            error!("AnomalyDetectionActor: GPU not initialized for anomaly detection");
            return Box::pin(
                async move { Err("GPU not initialized".to_string()) }.into_actor(self),
            );
        }

        if self.gpu_state.num_nodes == 0 {
            error!("AnomalyDetectionActor: No nodes available for anomaly detection");
            return Box::pin(
                async move { Err("No nodes available for anomaly detection".to_string()) }
                    .into_actor(self),
            );
        }

        let params = msg.params;

        
        let num_nodes = self.gpu_state.num_nodes;
        let k_neighbors = params.k_neighbors;
        if k_neighbors as u32 >= num_nodes {
            let error_msg = format!(
                "k_neighbors ({}) must be less than total nodes ({})",
                k_neighbors, num_nodes
            );
            return Box::pin(async move { Err(error_msg) }.into_actor(self));
        }

        
        let internal_params = AnomalyDetectionParams {
            method: match params.method {
                crate::actors::messages::AnomalyMethod::LocalOutlierFactor => {
                    AnomalyDetectionMethod::LOF
                }
                crate::actors::messages::AnomalyMethod::ZScore => AnomalyDetectionMethod::ZScore,
            },
            threshold: Some(params.threshold),
            k_neighbors: Some(params.k_neighbors),
            window_size: Some(100), 
            feature_data: None,
        };

        
        let start_time = std::time::Instant::now();

        
        let result = match &self.shared_context {
            Some(ctx) => {
                match ctx.unified_compute.lock() {
                    Ok(mut unified_compute) => {
                        match internal_params.method {
                            AnomalyDetectionMethod::LOF => {
                                
                                let k_neighbors = internal_params.k_neighbors.unwrap_or(5);
                                let threshold = internal_params.threshold.unwrap_or(0.5);

                                match unified_compute
                                    .run_lof_anomaly_detection(k_neighbors, threshold)
                                {
                                    Ok(lof_result) => {
                                        let lof_scores = lof_result.0;
                                        let mut anomalies = Vec::new();

                                        for (node_id, &score) in lof_scores.iter().enumerate() {
                                            if score > threshold {
                                                anomalies.push(crate::actors::gpu::anomaly_detection_actor::AnomalyNode {
                                                        node_id: node_id as u32,
                                                        anomaly_score: score,
                                                        reason: format!("LOF score {:.3} exceeds threshold {:.3}", score, threshold),
                                                        anomaly_type: "outlier".to_string(),
                                                        severity: if score > threshold * 3.0 { "high" } else { "medium" }.to_string(),
                                                        explanation: format!("LOF anomaly detected with score {:.3}", score),
                                                        features: vec!["lof_score".to_string()],
                                                    });
                                            }
                                        }

                                        Ok((Some(lof_scores), anomalies))
                                    }
                                    Err(e) => Err(format!("GPU LOF detection failed: {}", e)),
                                }
                            }
                            AnomalyDetectionMethod::ZScore => {
                                
                                let feature_data: Vec<f32> = (0..self.gpu_state.num_nodes)
                                    .map(|i| (i as f32 + 1.0) / self.gpu_state.num_nodes as f32)
                                    .collect();

                                match unified_compute.run_zscore_anomaly_detection(&feature_data) {
                                    Ok(z_scores) => {
                                        let threshold = internal_params.threshold.unwrap_or(3.0);
                                        let mut anomalies = Vec::new();

                                        for (node_id, &score) in z_scores.iter().enumerate() {
                                            let abs_score = score.abs();
                                            if abs_score > threshold {
                                                anomalies.push(crate::actors::gpu::anomaly_detection_actor::AnomalyNode {
                                                        node_id: node_id as u32,
                                                        anomaly_score: abs_score,
                                                        reason: format!("Z-score {:.3} exceeds threshold {:.3}", abs_score, threshold),
                                                        anomaly_type: "statistical_outlier".to_string(),
                                                        severity: if abs_score > threshold * 2.0 { "high" } else { "medium" }.to_string(),
                                                        explanation: format!("Statistical anomaly detected with Z-score {:.3}", score),
                                                        features: vec!["z_score".to_string()],
                                                    });
                                            }
                                        }

                                        Ok((Some(z_scores), anomalies))
                                    }
                                    Err(e) => Err(format!("GPU Z-Score detection failed: {}", e)),
                                }
                            }
                            AnomalyDetectionMethod::DBSCAN => {
                                
                                let eps = internal_params.threshold.unwrap_or(50.0);
                                let min_pts = 3;

                                match unified_compute.run_dbscan_clustering(eps, min_pts) {
                                    Ok(cluster_labels) => {
                                        let mut anomalies = Vec::new();

                                        for (node_id, &label) in cluster_labels.iter().enumerate() {
                                            if label == -1 {
                                                
                                                anomalies.push(crate::actors::gpu::anomaly_detection_actor::AnomalyNode {
                                                        node_id: node_id as u32,
                                                        anomaly_score: 1.0,
                                                        reason: format!("Node classified as noise by DBSCAN (eps={:.2})", eps),
                                                        anomaly_type: "spatial_outlier".to_string(),
                                                        severity: "high".to_string(),
                                                        explanation: "DBSCAN identified this node as noise (not belonging to any cluster)".to_string(),
                                                        features: vec!["spatial_isolation".to_string()],
                                                    });
                                            }
                                        }

                                        Ok((None, anomalies))
                                    }
                                    Err(e) => Err(format!("GPU DBSCAN detection failed: {}", e)),
                                }
                            }
                            _ => Err("Unsupported anomaly detection method".to_string()),
                        }
                    }
                    Err(e) => Err(format!("Failed to acquire GPU compute lock: {}", e)),
                }
            }
            None => Err("GPU context not initialized".to_string()),
        };

        let computation_time = start_time.elapsed();

        let final_result = match result {
            Ok((scores, anomalies)) => {
                let anomalies_count = anomalies.len();
                let avg_score = if !anomalies.is_empty() {
                    anomalies.iter().map(|a| a.anomaly_score).sum::<f32>() / anomalies.len() as f32
                } else {
                    0.0
                };
                let max_score = anomalies
                    .iter()
                    .map(|a| a.anomaly_score)
                    .fold(0.0, f32::max);
                let min_score = anomalies
                    .iter()
                    .map(|a| a.anomaly_score)
                    .fold(f32::INFINITY, f32::min);

                info!("AnomalyDetectionActor: GPU {:?} detection completed in {:?}, found {} anomalies",
                          internal_params.method, computation_time, anomalies_count);

                Ok(AnomalyResult {
                    lof_scores: if matches!(internal_params.method, AnomalyDetectionMethod::LOF) {
                        scores.clone()
                    } else {
                        None
                    },
                    local_densities: None,
                    zscore_values: if matches!(
                        internal_params.method,
                        AnomalyDetectionMethod::ZScore
                    ) {
                        scores
                    } else {
                        None
                    },
                    anomaly_threshold: internal_params.threshold.unwrap_or(0.5),
                    num_anomalies: anomalies_count,
                    anomalies,
                    stats: crate::actors::messages::AnomalyDetectionStats {
                        total_nodes_analyzed: self.gpu_state.num_nodes,
                        anomalies_found: anomalies_count,
                        detection_threshold: internal_params.threshold.unwrap_or(0.5),
                        computation_time_ms: computation_time.as_millis() as u64,
                        method: internal_params.method.clone(),
                        average_anomaly_score: avg_score,
                        max_anomaly_score: max_score,
                        min_anomaly_score: min_score,
                    },
                    method: internal_params.method.clone(),
                    threshold: internal_params.threshold.unwrap_or(0.5),
                })
            }
            Err(e) => {
                error!("AnomalyDetectionActor: GPU detection failed: {}", e);
                Err(e)
            }
        };

        Box::pin(async move { final_result }.into_actor(self))
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

///
impl Handler<SetSharedGPUContext> for AnomalyDetectionActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        info!("AnomalyDetectionActor: Received SharedGPUContext from ResourceActor");
        self.shared_context = Some(msg.context);
        
        info!("AnomalyDetectionActor: SharedGPUContext stored successfully");
        Ok(())
    }
}
