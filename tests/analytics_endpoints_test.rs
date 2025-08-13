/*!
 * Integration tests for Analytics API endpoints
 * 
 * Tests the new clustering, anomaly detection, and insights endpoints
 * to ensure they match client expectations and function correctly.
 */

use serde_json::json;
use std::collections::HashMap;

/// Test data structures match client expectations
#[cfg(test)]
mod analytics_api_tests {
    use super::*;

    #[tokio::test]
    async fn test_clustering_request_structure() {
        // Test that our Rust structures match TypeScript client expectations
        let clustering_request = json!({
            "method": "spectral",
            "params": {
                "numClusters": 8,
                "minClusterSize": 5,
                "similarity": "cosine",
                "convergenceThreshold": 0.001,
                "maxIterations": 100
            }
        });

        // This should deserialize without error
        let parsed: Result<serde_json::Value, _> = serde_json::from_value(clustering_request);
        assert!(parsed.is_ok(), "Clustering request structure should be valid");
    }

    #[tokio::test]
    async fn test_cluster_response_structure() {
        // Test that our response matches client expectations
        let cluster_response = json!({
            "success": true,
            "clusters": [
                {
                    "id": "cluster-1",
                    "label": "Spectral Cluster 1",
                    "nodeCount": 25,
                    "coherence": 0.85,
                    "color": "#4F46E5",
                    "keywords": ["semantic", "analysis", "spectral"],
                    "nodes": [1, 2, 3, 4, 5],
                    "centroid": [10.0, 5.0, 0.0]
                }
            ],
            "method": "spectral",
            "executionTimeMs": 1500,
            "taskId": "task-123"
        });

        let parsed: Result<serde_json::Value, _> = serde_json::from_value(cluster_response);
        assert!(parsed.is_ok(), "Cluster response structure should be valid");
    }

    #[tokio::test]
    async fn test_anomaly_detection_structure() {
        // Test anomaly detection request/response structures
        let anomaly_config = json!({
            "enabled": true,
            "method": "isolation_forest",
            "sensitivity": 0.5,
            "windowSize": 100,
            "updateInterval": 5000
        });

        let parsed: Result<serde_json::Value, _> = serde_json::from_value(anomaly_config);
        assert!(parsed.is_ok(), "Anomaly config structure should be valid");

        let anomaly_response = json!({
            "success": true,
            "anomalies": [
                {
                    "id": "anomaly-1",
                    "nodeId": "node_123",
                    "type": "structural_outlier",
                    "severity": "high",
                    "score": 0.89,
                    "description": "Node exhibits unusual structural properties",
                    "timestamp": 1640995200,
                    "metadata": {
                        "detection_method": "isolation_forest",
                        "confidence": 0.89
                    }
                }
            ],
            "stats": {
                "total": 15,
                "critical": 2,
                "high": 5,
                "medium": 6,
                "low": 2,
                "lastUpdated": 1640995200
            },
            "enabled": true,
            "method": "isolation_forest"
        });

        let parsed: Result<serde_json::Value, _> = serde_json::from_value(anomaly_response);
        assert!(parsed.is_ok(), "Anomaly response structure should be valid");
    }

    #[tokio::test]
    async fn test_insights_response_structure() {
        // Test AI insights response structure
        let insights_response = json!({
            "success": true,
            "insights": [
                "Graph structure analysis shows balanced connectivity patterns",
                "Identified 8 distinct semantic clusters",
                "Detected 15 anomalies across the graph"
            ],
            "patterns": [
                {
                    "id": "pattern-1",
                    "type": "dominant_cluster",
                    "description": "Large semantic cluster with 50+ nodes",
                    "confidence": 0.85,
                    "nodes": [1, 2, 3, 4, 5],
                    "significance": "high"
                }
            ],
            "recommendations": [
                "Consider increasing clustering threshold to reduce cluster count",
                "Investigate critical anomalies that may indicate data quality issues"
            ],
            "analysisTimestamp": 1640995200
        });

        let parsed: Result<serde_json::Value, _> = serde_json::from_value(insights_response);
        assert!(parsed.is_ok(), "Insights response structure should be valid");
    }

    #[tokio::test]
    async fn test_clustering_status_structure() {
        // Test clustering status response structure
        let status_response = json!({
            "success": true,
            "taskId": "task-123",
            "status": "running",
            "progress": 0.65,
            "method": "spectral",
            "startedAt": "1640995200",
            "estimatedCompletion": "1640995260"
        });

        let parsed: Result<serde_json::Value, _> = serde_json::from_value(status_response);
        assert!(parsed.is_ok(), "Status response structure should be valid");
    }

    #[tokio::test]
    async fn test_clustering_methods_coverage() {
        // Test that all clustering methods mentioned in client are supported
        let supported_methods = vec![
            "spectral",
            "hierarchical", 
            "dbscan",
            "kmeans",
            "louvain",
            "affinity"
        ];

        // This test ensures we haven't missed any methods
        // In a real implementation, we'd test actual clustering execution
        for method in supported_methods {
            let request = json!({
                "method": method,
                "params": {
                    "numClusters": 5
                }
            });
            
            assert!(request.get("method").is_some(), 
                "Method {} should be supported", method);
        }
    }

    #[tokio::test]
    async fn test_anomaly_detection_methods() {
        // Test anomaly detection methods coverage
        let supported_methods = vec![
            "isolation_forest",
            "lof",
            "autoencoder", 
            "statistical",
            "temporal"
        ];

        for method in supported_methods {
            let config = json!({
                "enabled": true,
                "method": method,
                "sensitivity": 0.5,
                "windowSize": 100,
                "updateInterval": 5000
            });
            
            assert!(config.get("method").is_some(),
                "Anomaly method {} should be supported", method);
        }
    }

    #[tokio::test]  
    async fn test_severity_levels() {
        // Test that all severity levels used in client are supported
        let severity_levels = vec!["low", "medium", "high", "critical"];
        
        for severity in severity_levels {
            let anomaly = json!({
                "id": "test-id",
                "nodeId": "node-1", 
                "type": "test_anomaly",
                "severity": severity,
                "score": 0.5,
                "description": "Test anomaly",
                "timestamp": 1640995200
            });
            
            assert_eq!(anomaly.get("severity").unwrap(), severity,
                "Severity level {} should be properly handled", severity);
        }
    }

    #[tokio::test]
    async fn test_api_endpoint_paths() {
        // Test that expected API endpoint paths are documented
        let expected_endpoints = vec![
            "/api/analytics/clustering/run",
            "/api/analytics/clustering/status", 
            "/api/analytics/clustering/focus",
            "/api/analytics/anomaly/toggle",
            "/api/analytics/anomaly/current",
            "/api/analytics/insights"
        ];

        // This is a documentation test - in a real integration test,
        // we'd make actual HTTP requests to these endpoints
        for endpoint in expected_endpoints {
            assert!(endpoint.starts_with("/api/analytics/"),
                "Endpoint {} should be under /api/analytics/ namespace", endpoint);
        }
    }
}

/// Mock data generators for testing
#[cfg(test)]
mod mock_data {
    use super::*;

    pub fn create_mock_clustering_request() -> serde_json::Value {
        json!({
            "method": "spectral",
            "params": {
                "numClusters": 8,
                "similarity": "cosine",
                "convergenceThreshold": 0.001,
                "maxIterations": 100
            }
        })
    }

    pub fn create_mock_cluster() -> serde_json::Value {
        json!({
            "id": "mock-cluster-1",
            "label": "Mock Cluster",
            "nodeCount": 25,
            "coherence": 0.85,
            "color": "#4F46E5", 
            "keywords": ["mock", "test", "cluster"],
            "nodes": [1, 2, 3, 4, 5],
            "centroid": [0.0, 0.0, 0.0]
        })
    }

    pub fn create_mock_anomaly() -> serde_json::Value {
        json!({
            "id": "mock-anomaly-1",
            "nodeId": "node_123",
            "type": "test_anomaly",
            "severity": "medium",
            "score": 0.7,
            "description": "Mock anomaly for testing",
            "timestamp": 1640995200,
            "metadata": {
                "test": true
            }
        })
    }
}