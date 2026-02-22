

use std::collections::HashMap;
use log::{debug, error, info, warn};
use rand::Rng;
use uuid::Uuid;
use chrono::Utc;

use super::{Anomaly, AnomalyStats, ANOMALY_STATE};
use crate::AppState;
use crate::actors::messages::{RunAnomalyDetection, AnomalyParams, AnomalyMethod};
use crate::utils::result_helpers::safe_json_number;

pub async fn run_gpu_anomaly_detection(
    app_state: &actix_web::web::Data<AppState>,
    method: &str,
    k_neighbors: Option<i32>,
    radius: Option<f32>,
    feature_data: Option<Vec<f32>>,
    threshold: Option<f32>,
) -> Result<Vec<Anomaly>, String> {
    info!("Running GPU anomaly detection with method: {}", method);

    // GPU compute address is now Arc<RwLock<Option<...>>> - use async accessor
    let gpu_addr = app_state.get_gpu_compute_addr().await
        .ok_or_else(|| "GPU compute actor not available".to_string())?;

    
    let anomaly_method = match method {
        "lof" | "local_outlier_factor" => AnomalyMethod::LocalOutlierFactor,
        "zscore" | "z_score" => AnomalyMethod::ZScore,
        _ => return Err(format!("Unsupported anomaly detection method: {}", method)),
    };

    
    let params = AnomalyParams {
        method: anomaly_method.clone(),
        k_neighbors: k_neighbors.unwrap_or(20),
        radius: radius.unwrap_or(1.0),
        feature_data: feature_data.clone(),
        threshold: threshold.unwrap_or(match anomaly_method {
            AnomalyMethod::LocalOutlierFactor => 1.5,
            AnomalyMethod::ZScore => 2.0,
        }),
    };

    
    validate_anomaly_params(&params)?;

    let msg = RunAnomalyDetection { params };

    match gpu_addr.send(msg).await {
        Ok(Ok(result)) => {
            info!("GPU anomaly detection completed: {} anomalies found", result.num_anomalies);
            Ok(convert_gpu_anomaly_result_to_anomalies(result, &anomaly_method))
        }
        Ok(Err(e)) => {
            error!("GPU anomaly detection failed: {}", e);
            Err(e)
        }
        Err(e) => {
            error!("GPU actor mailbox error: {}", e);
            Err(format!("Failed to communicate with GPU actor: {}", e))
        }
    }
}

pub async fn start_anomaly_detection() {
    warn!("start_anomaly_detection is deprecated - use run_gpu_anomaly_detection for real GPU processing");
}

async fn generate_anomaly(method: &str) -> Anomaly {
    let mut rng = rand::thread_rng();

    
    let node_id = format!("node_{}", rng.gen_range(1..=1000));

    
    let severity_weights = match method {
        "isolation_forest" => [0.1, 0.3, 0.4, 0.2], 
        "lof" => [0.05, 0.25, 0.5, 0.2],
        "autoencoder" => [0.15, 0.35, 0.35, 0.15],
        "statistical" => [0.2, 0.3, 0.3, 0.2],
        "temporal" => [0.25, 0.25, 0.3, 0.2],
        _ => [0.1, 0.3, 0.4, 0.2],
    };

    let random_val = rng.gen::<f32>();
    let severity = if random_val < severity_weights[0] {
        "critical"
    } else if random_val < severity_weights[0] + severity_weights[1] {
        "high"
    } else if random_val < severity_weights[0] + severity_weights[1] + severity_weights[2] {
        "medium"
    } else {
        "low"
    };

    
    let score = match severity {
        "critical" => 0.9 + rng.gen::<f32>() * 0.1,
        "high" => 0.7 + rng.gen::<f32>() * 0.2,
        "medium" => 0.4 + rng.gen::<f32>() * 0.3,
        "low" => rng.gen::<f32>() * 0.4,
        _ => 0.5,
    };

    
    let (anomaly_type, description) = generate_anomaly_details(method, severity);

    let mut metadata = HashMap::new();
    metadata.insert("detection_method".to_string(), serde_json::Value::String(method.to_string()));
    metadata.insert("confidence".to_string(), serde_json::Value::Number(safe_json_number(score as f64)));

    match method {
        "isolation_forest" => {
            metadata.insert("isolation_depth".to_string(),
                serde_json::Value::Number(serde_json::Number::from(rng.gen_range(2..=10))));
            metadata.insert("tree_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(rng.gen_range(50..=200))));
        },
        "lof" => {
            metadata.insert("local_density".to_string(),
                serde_json::Value::Number(safe_json_number(rng.gen::<f64>())));
            metadata.insert("neighbors_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(rng.gen_range(5..=30))));
        },
        "autoencoder" => {
            metadata.insert("reconstruction_error".to_string(),
                serde_json::Value::Number(safe_json_number(score as f64)));
            metadata.insert("latent_dimension".to_string(),
                serde_json::Value::Number(serde_json::Number::from(rng.gen_range(8..=128))));
        },
        "statistical" => {
            metadata.insert("z_score".to_string(),
                serde_json::Value::Number(safe_json_number((score * 6.0 - 3.0) as f64)));
            metadata.insert("iqr_position".to_string(),
                serde_json::Value::Number(safe_json_number(score as f64)));
        },
        "temporal" => {
            metadata.insert("time_window".to_string(),
                serde_json::Value::String(format!("{}s", rng.gen_range(30..=300))));
            metadata.insert("trend_deviation".to_string(),
                serde_json::Value::Number(safe_json_number(score as f64)));
        },
        _ => {}
    }

    Anomaly {
        id: Uuid::new_v4().to_string(),
        node_id,
        r#type: anomaly_type,
        severity: severity.to_string(),
        score,
        description,
        timestamp: Utc::now().timestamp() as u64,
        metadata: Some(serde_json::Value::Object(metadata.into_iter().collect())),
    }
}

fn generate_anomaly_details(method: &str, severity: &str) -> (String, String) {
    let mut rng = rand::thread_rng();

    match method {
        "isolation_forest" => {
            let types = ["structural_outlier", "connectivity_anomaly", "isolation_pattern"];
            let anomaly_type = types[rng.gen_range(0..types.len())].to_string();

            let description = match anomaly_type.as_str() {
                "structural_outlier" => format!("Node exhibits unusual structural properties with {} isolation depth", severity),
                "connectivity_anomaly" => format!("Abnormal connectivity pattern detected with {} confidence", severity),
                "isolation_pattern" => format!("Node isolated in feature space with {} significance", severity),
                _ => format!("Isolation forest detected {} anomaly", severity),
            };

            (anomaly_type, description)
        },
        "lof" => {
            let types = ["density_outlier", "local_anomaly", "neighborhood_deviation"];
            let anomaly_type = types[rng.gen_range(0..types.len())].to_string();

            let description = match anomaly_type.as_str() {
                "density_outlier" => format!("Node has {} local density compared to neighbors", severity),
                "local_anomaly" => format!("Local outlier factor indicates {} anomaly", severity),
                "neighborhood_deviation" => format!("Significant deviation from local neighborhood with {} severity", severity),
                _ => format!("Local outlier factor detected {} anomaly", severity),
            };

            (anomaly_type, description)
        },
        "autoencoder" => {
            let types = ["reconstruction_error", "latent_anomaly", "encoding_deviation"];
            let anomaly_type = types[rng.gen_range(0..types.len())].to_string();

            let description = match anomaly_type.as_str() {
                "reconstruction_error" => format!("High reconstruction error indicates {} anomaly", severity),
                "latent_anomaly" => format!("Anomalous pattern in latent space with {} confidence", severity),
                "encoding_deviation" => format!("Neural encoding shows {} deviation from normal patterns", severity),
                _ => format!("Autoencoder detected {} anomaly", severity),
            };

            (anomaly_type, description)
        },
        "statistical" => {
            let types = ["z_score_outlier", "iqr_outlier", "distribution_anomaly"];
            let anomaly_type = types[rng.gen_range(0..types.len())].to_string();

            let description = match anomaly_type.as_str() {
                "z_score_outlier" => format!("Z-score indicates {} statistical outlier", severity),
                "iqr_outlier" => format!("Value outside interquartile range with {} significance", severity),
                "distribution_anomaly" => format!("Statistical distribution shows {} anomaly", severity),
                _ => format!("Statistical analysis detected {} anomaly", severity),
            };

            (anomaly_type, description)
        },
        "temporal" => {
            let types = ["trend_anomaly", "seasonal_deviation", "temporal_outlier"];
            let anomaly_type = types[rng.gen_range(0..types.len())].to_string();

            let description = match anomaly_type.as_str() {
                "trend_anomaly" => format!("Temporal trend shows {} anomalous behavior", severity),
                "seasonal_deviation" => format!("Deviation from expected seasonal pattern with {} severity", severity),
                "temporal_outlier" => format!("Time-series analysis detected {} temporal outlier", severity),
                _ => format!("Temporal analysis detected {} anomaly", severity),
            };

            (anomaly_type, description)
        },
        _ => {
            let anomaly_type = "unknown_anomaly".to_string();
            let description = format!("Unknown detection method found {} anomaly", severity);
            (anomaly_type, description)
        }
    }
}

pub async fn cleanup_old_anomalies() {
    let mut state = ANOMALY_STATE.lock().await;
    let current_time = Utc::now().timestamp() as u64;
    let retention_period = 3600; 

    let initial_count = state.anomalies.len();
    state.anomalies.retain(|anomaly| current_time - anomaly.timestamp < retention_period);
    let removed_count = initial_count - state.anomalies.len();

    if removed_count > 0 {
        debug!("Cleaned up {} old anomalies", removed_count);

        
        let mut new_stats = AnomalyStats::default();
        for anomaly in &state.anomalies {
            match anomaly.severity.as_str() {
                "critical" => new_stats.critical += 1,
                "high" => new_stats.high += 1,
                "medium" => new_stats.medium += 1,
                "low" => new_stats.low += 1,
                _ => {}
            }
            new_stats.total += 1;
        }
        new_stats.last_updated = state.stats.last_updated;
        state.stats = new_stats;
    }
}

fn validate_anomaly_params(params: &AnomalyParams) -> Result<(), String> {
    match params.method {
        AnomalyMethod::LocalOutlierFactor => {
            if params.k_neighbors < 1 || params.k_neighbors > 1000 {
                return Err("k_neighbors must be between 1 and 1000".to_string());
            }
            if params.radius <= 0.0 || params.radius > 1000.0 {
                return Err("radius must be between 0.0 and 1000.0".to_string());
            }
            if params.threshold <= 0.0 || params.threshold > 10.0 {
                return Err("LOF threshold must be between 0.0 and 10.0".to_string());
            }
        }
        AnomalyMethod::ZScore => {
            if params.feature_data.is_none() {
                return Err("feature_data is required for Z-score anomaly detection".to_string());
            }
            if let Some(ref features) = params.feature_data {
                if features.is_empty() {
                    return Err("feature_data cannot be empty".to_string());
                }
            }
            if params.threshold <= 0.0 || params.threshold > 10.0 {
                return Err("Z-score threshold must be between 0.0 and 10.0".to_string());
            }
        }
    }
    Ok(())
}

fn convert_gpu_anomaly_result_to_anomalies(
    result: crate::actors::messages::AnomalyResult,
    method: &AnomalyMethod,
) -> Vec<Anomaly> {
    let mut anomalies = Vec::new();
    let current_time = Utc::now().timestamp() as u64;

    match method {
        AnomalyMethod::LocalOutlierFactor => {
            if let Some(lof_scores) = result.lof_scores {
                for (i, &score) in lof_scores.iter().enumerate() {
                    if score > result.anomaly_threshold {
                        let severity = determine_severity_from_lof_score(score);
                        let mut metadata = HashMap::new();
                        metadata.insert("lof_score".to_string(), serde_json::Value::Number(
                            safe_json_number(score as f64)
                        ));

                        if let Some(ref densities) = result.local_densities {
                            if i < densities.len() {
                                metadata.insert("local_density".to_string(),
                                    serde_json::Value::Number(
                                        safe_json_number(densities[i] as f64)
                                    )
                                );
                            }
                        }

                        anomalies.push(Anomaly {
                            id: Uuid::new_v4().to_string(),
                            node_id: format!("node_{}", i),
                            r#type: "local_outlier".to_string(),
                            severity: severity.to_string(),
                            score,
                            description: format!("Node has abnormally {} local density (LOF score: {:.3})",
                                                severity, score),
                            timestamp: current_time,
                            metadata: Some(serde_json::Value::Object(metadata.into_iter().collect())),
                        });
                    }
                }
            }
        }
        AnomalyMethod::ZScore => {
            if let Some(zscore_values) = result.zscore_values {
                for (i, &score) in zscore_values.iter().enumerate() {
                    let abs_score = score.abs();
                    if abs_score > result.anomaly_threshold {
                        let severity = determine_severity_from_zscore(abs_score);
                        let mut metadata = HashMap::new();
                        metadata.insert("zscore".to_string(), serde_json::Value::Number(
                            safe_json_number(score as f64)
                        ));
                        metadata.insert("abs_zscore".to_string(), serde_json::Value::Number(
                            safe_json_number(abs_score as f64)
                        ));

                        anomalies.push(Anomaly {
                            id: Uuid::new_v4().to_string(),
                            node_id: format!("node_{}", i),
                            r#type: "statistical_outlier".to_string(),
                            severity: severity.to_string(),
                            score: abs_score,
                            description: format!("Statistical outlier with {} significance (Z-score: {:.3})",
                                                severity, score),
                            timestamp: current_time,
                            metadata: Some(serde_json::Value::Object(metadata.into_iter().collect())),
                        });
                    }
                }
            }
        }
    }

    anomalies
}

fn determine_severity_from_lof_score(score: f32) -> &'static str {
    if score > 3.0 {
        "critical"
    } else if score > 2.5 {
        "high"
    } else if score > 2.0 {
        "medium"
    } else {
        "low"
    }
}

fn determine_severity_from_zscore(abs_score: f32) -> &'static str {
    if abs_score > 4.0 {
        "critical"
    } else if abs_score > 3.0 {
        "high"
    } else if abs_score > 2.5 {
        "medium"
    } else {
        "low"
    }
}