//! Mock data generation service
//! Centralises all mock data generation logic from handlers

use serde_json::json;

/// Generate mock clustering data for demonstration purposes
pub fn generate_clustering_mock() -> serde_json::Value {
    json!({
        "clusters": [
            {
                "id": 0,
                "label": "Core Infrastructure",
                "nodes": ["node_1", "node_3", "node_5"],
                "centroid": [0.5, 0.3, 0.1],
                "size": 3,
                "density": 0.85,
                "stability": 0.92
            },
            {
                "id": 1,
                "label": "User Interface",
                "nodes": ["node_2", "node_4", "node_6"],
                "centroid": [-0.3, 0.7, 0.2],
                "size": 3,
                "density": 0.78,
                "stability": 0.88
            }
        ],
        "metadata": {
            "algorithm": "spectral_clustering",
            "parameters": {
                "n_clusters": 2,
                "affinity": "nearest_neighbors",
                "n_neighbors": 10
            },
            "quality_metrics": {
                "silhouette_score": 0.73,
                "davies_bouldin_index": 0.45,
                "calinski_harabasz_score": 156.8
            }
        }
    })
}

/// Generate mock anomaly detection data
pub fn generate_anomaly_mock() -> serde_json::Value {
    json!({
        "anomalies": [
            {
                "node_id": "node_42",
                "score": 0.92,
                "type": "outlier",
                "detected_at": "2024-01-01T12:00:00Z",
                "features": {
                    "degree_centrality": 0.05,
                    "clustering_coefficient": 0.12,
                    "betweenness_centrality": 0.89
                }
            },
            {
                "node_id": "node_77",
                "score": 0.85,
                "type": "local_outlier",
                "detected_at": "2024-01-01T12:05:00Z",
                "features": {
                    "degree_centrality": 0.78,
                    "clustering_coefficient": 0.03,
                    "betweenness_centrality": 0.15
                }
            }
        ],
        "metadata": {
            "algorithm": "isolation_forest",
            "parameters": {
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": 256
            },
            "statistics": {
                "total_nodes_analyzed": 1000,
                "anomalies_detected": 2,
                "detection_rate": 0.002,
                "processing_time_ms": 45
            }
        }
    })
}

/// Generate mock stress majorization metrics
pub fn generate_stress_metrics_mock() -> serde_json::Value {
    json!({
        "stress": 0.0234,
        "iteration": 150,
        "converged": true,
        "improvement_rate": 0.0001,
        "computation_time_ms": 23,
        "node_displacements": {
            "max": 0.012,
            "mean": 0.005,
            "std": 0.003
        }
    })
}

/// Generate mock force physics statistics
pub fn generate_force_physics_mock() -> serde_json::Value {
    json!({
        "iteration_count": 1000,
        "average_velocity": 0.05,
        "kinetic_energy": 0.23,
        "total_forces": 15.7,
        "fps": 60.0,
        "nodes_count": 100,
        "edges_count": 150,
        "gpu_enabled": true,
        "compute_mode": "Advanced"
    })
}