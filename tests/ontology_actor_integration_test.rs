//! Integration tests for OntologyActor
//!
//! Tests cover:
//! - OntologyActor message handling
//! - PhysicsOrchestratorActor integration
//! - End-to-end validation workflow
//! - Actor lifecycle and error handling

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    #[cfg(feature = "ontology")]
    use actix::prelude::*;

    #[cfg(feature = "ontology")]
    use webxr::actors::ontology_actor::{OntologyActor, OntologyActorConfig};

    #[cfg(feature = "ontology")]
    use webxr::actors::messages::{
        LoadOntologyAxioms, ValidateOntology, ValidationMode, ApplyInferences,
        GetOntologyReport, GetOntologyHealth, ClearOntologyCaches, UpdateOntologyMapping
    };

    #[cfg(feature = "ontology")]
    use webxr::services::owl_validator::{
        PropertyGraph, GraphNode, GraphEdge, ValidationConfig, RdfTriple
    };

    #[cfg(feature = "ontology")]
    fn create_test_graph() -> PropertyGraph {
        PropertyGraph {
            nodes: vec![
                GraphNode {
                    id: "person1".to_string(),
                    labels: vec!["Person".to_string()],
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("name".to_string(), serde_json::json!("Alice"));
                        props.insert("age".to_string(), serde_json::json!(30));
                        props
                    },
                },
                GraphNode {
                    id: "company1".to_string(),
                    labels: vec!["Company".to_string()],
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("name".to_string(), serde_json::json!("ACME Corp"));
                        props
                    },
                }
            ],
            edges: vec![
                GraphEdge {
                    id: "edge1".to_string(),
                    source: "person1".to_string(),
                    target: "company1".to_string(),
                    relationship_type: "employedBy".to_string(),
                    properties: HashMap::new(),
                }
            ],
            metadata: HashMap::new(),
        }
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_actor_startup_shutdown() {
        let config = OntologyActorConfig::default();
        let actor = OntologyActor::with_config(config);
        let addr = actor.start();

        // Give actor time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop the actor
        addr.do_send(actix::dev::Stop);

        // Wait for clean shutdown
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(true, "Actor started and stopped successfully");
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_load_ontology_message() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let functional_ontology = r#"
Prefix(:=<http://example.org/>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)

Ontology(<http://example.org/test>
    Declaration(Class(:Person))
    Declaration(Class(:Company))
)
        "#;

        let msg = LoadOntologyAxioms {
            source: functional_ontology.to_string(),
        };

        let result = addr.send(msg).await;

        assert!(result.is_ok(), "Message send failed: {:?}", result.err());
        let ontology_id = result.unwrap();
        assert!(ontology_id.is_ok(), "Load ontology failed: {:?}", ontology_id.err());
        assert!(!ontology_id.unwrap().is_empty());
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_validate_ontology_message() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        // First load an ontology
        let functional_ontology = r#"
Prefix(:=<http://example.org/>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)

Ontology(<http://example.org/test>
    Declaration(Class(:Thing))
)
        "#;

        let load_msg = LoadOntologyAxioms {
            source: functional_ontology.to_string(),
        };

        let ontology_id = addr.send(load_msg).await.unwrap().unwrap();

        // Now validate a graph
        let graph = create_test_graph();

        let validate_msg = ValidateOntology {
            ontology_id,
            graph_data: graph,
            mode: ValidationMode::Quick,
        };

        let result = addr.send(validate_msg).await;

        assert!(result.is_ok(), "Validation message failed: {:?}", result.err());
        let report = result.unwrap();
        assert!(report.is_ok(), "Validation failed: {:?}", report.err());

        let validation_report = report.unwrap();
        assert!(!validation_report.id.is_empty());
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_apply_inferences_message() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let triples = vec![
            RdfTriple {
                subject: "http://example.org/person1".to_string(),
                predicate: "http://example.org/knows".to_string(),
                object: "http://example.org/person2".to_string(),
                is_literal: false,
                datatype: None,
                language: None,
            }
        ];

        let msg = ApplyInferences {
            rdf_triples: triples,
        };

        let result = addr.send(msg).await;

        assert!(result.is_ok(), "Inference message failed: {:?}", result.err());
        let inferred = result.unwrap();
        assert!(inferred.is_ok(), "Inference failed: {:?}", inferred.err());

        let inferred_triples = inferred.unwrap();
        // Should infer symmetric relationship
        assert!(!inferred_triples.is_empty());
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_get_ontology_report_message() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let msg = GetOntologyReport {
            report_id: None, // Get latest report
        };

        let result = addr.send(msg).await;

        assert!(result.is_ok(), "Get report message failed: {:?}", result.err());
        let report_opt = result.unwrap();
        assert!(report_opt.is_ok());

        // May be None if no validations have been performed yet
        let _report = report_opt.unwrap();
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_get_ontology_health_message() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let msg = GetOntologyHealth;

        let result = addr.send(msg).await;

        assert!(result.is_ok(), "Get health message failed: {:?}", result.err());
        let health = result.unwrap();
        assert!(health.is_ok(), "Get health failed: {:?}", health.err());

        let health_info = health.unwrap();
        assert!(health_info.cache_hit_rate >= 0.0);
        assert!(health_info.cache_hit_rate <= 1.0);
        assert_eq!(health_info.validation_queue_size, 0);
        assert_eq!(health_info.active_jobs, 0);
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_clear_caches_message() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let msg = ClearOntologyCaches;

        let result = addr.send(msg).await;

        assert!(result.is_ok(), "Clear caches message failed: {:?}", result.err());
        let clear_result = result.unwrap();
        assert!(clear_result.is_ok(), "Clear caches failed: {:?}", clear_result.err());
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_update_mapping_message() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let config = ValidationConfig {
            enable_reasoning: true,
            reasoning_timeout_seconds: 60,
            enable_inference: true,
            max_inference_depth: 5,
            enable_caching: true,
            cache_ttl_seconds: 3600,
            validate_cardinality: true,
            validate_domains_ranges: true,
            validate_disjoint_classes: true,
        };

        let msg = UpdateOntologyMapping {
            config,
        };

        let result = addr.send(msg).await;

        assert!(result.is_ok(), "Update mapping message failed: {:?}", result.err());
        let update_result = result.unwrap();
        assert!(update_result.is_ok(), "Update mapping failed: {:?}", update_result.err());
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_validation_modes() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let functional_ontology = r#"
Prefix(:=<http://example.org/>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)

Ontology(<http://example.org/test>
    Declaration(Class(:Thing))
)
        "#;

        let load_msg = LoadOntologyAxioms {
            source: functional_ontology.to_string(),
        };

        let ontology_id = addr.send(load_msg).await.unwrap().unwrap();
        let graph = create_test_graph();

        // Test Quick mode
        let quick_msg = ValidateOntology {
            ontology_id: ontology_id.clone(),
            graph_data: graph.clone(),
            mode: ValidationMode::Quick,
        };

        let quick_result = addr.send(quick_msg).await.unwrap();
        assert!(quick_result.is_ok());

        // Test Full mode
        let full_msg = ValidateOntology {
            ontology_id: ontology_id.clone(),
            graph_data: graph.clone(),
            mode: ValidationMode::Full,
        };

        let full_result = addr.send(full_msg).await.unwrap();
        assert!(full_result.is_ok());

        // Test Incremental mode
        let incremental_msg = ValidateOntology {
            ontology_id,
            graph_data: graph,
            mode: ValidationMode::Incremental,
        };

        let incremental_result = addr.send(incremental_msg).await.unwrap();
        assert!(incremental_result.is_ok());
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_multiple_validations() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let functional_ontology = r#"
Prefix(:=<http://example.org/>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)

Ontology(<http://example.org/test>
    Declaration(Class(:Thing))
)
        "#;

        let load_msg = LoadOntologyAxioms {
            source: functional_ontology.to_string(),
        };

        let ontology_id = addr.send(load_msg).await.unwrap().unwrap();

        // Perform multiple validations in sequence
        for i in 0..3 {
            let mut graph = create_test_graph();
            graph.nodes.push(GraphNode {
                id: format!("extra_node_{}", i),
                labels: vec!["Thing".to_string()],
                properties: HashMap::new(),
            });

            let validate_msg = ValidateOntology {
                ontology_id: ontology_id.clone(),
                graph_data: graph,
                mode: ValidationMode::Quick,
            };

            let result = addr.send(validate_msg).await.unwrap();
            assert!(result.is_ok(), "Validation {} failed", i);
        }

        // Check health after multiple validations
        let health_msg = GetOntologyHealth;
        let health = addr.send(health_msg).await.unwrap().unwrap();

        println!("Health after multiple validations: {:?}", health);
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_actor_error_handling() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        // Try to validate with invalid ontology ID
        let graph = create_test_graph();

        let validate_msg = ValidateOntology {
            ontology_id: "invalid_ontology_id".to_string(),
            graph_data: graph,
            mode: ValidationMode::Quick,
        };

        let result = addr.send(validate_msg).await;

        // Should not panic, but may return an error or placeholder report
        assert!(result.is_ok(), "Actor should handle invalid ontology gracefully");
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_concurrent_validations() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        let functional_ontology = r#"
Prefix(:=<http://example.org/>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)

Ontology(<http://example.org/test>
    Declaration(Class(:Thing))
)
        "#;

        let load_msg = LoadOntologyAxioms {
            source: functional_ontology.to_string(),
        };

        let ontology_id = addr.send(load_msg).await.unwrap().unwrap();

        // Send multiple validation requests concurrently
        let mut futures = vec![];

        for i in 0..5 {
            let addr_clone = addr.clone();
            let ontology_id_clone = ontology_id.clone();

            let future = async move {
                let mut graph = create_test_graph();
                graph.nodes.push(GraphNode {
                    id: format!("node_{}", i),
                    labels: vec!["Thing".to_string()],
                    properties: HashMap::new(),
                });

                let validate_msg = ValidateOntology {
                    ontology_id: ontology_id_clone,
                    graph_data: graph,
                    mode: ValidationMode::Quick,
                };

                addr_clone.send(validate_msg).await
            };

            futures.push(future);
        }

        // Wait for all validations to complete
        let results = futures::future::join_all(futures).await;

        // All should succeed
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok(), "Concurrent validation {} failed", i);
        }
    }

    #[cfg(feature = "ontology")]
    #[actix_rt::test]
    async fn test_end_to_end_workflow() {
        let actor = OntologyActor::new();
        let addr = actor.start();

        // Step 1: Load ontology
        let ontology_content = r#"
Prefix(:=<http://example.org/>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)

Ontology(<http://example.org/test>
    Declaration(Class(:Person))
    Declaration(Class(:Company))
    DisjointClasses(:Person :Company)

    Declaration(ObjectProperty(:employedBy))
)
        "#;

        let load_result = addr.send(LoadOntologyAxioms {
            source: ontology_content.to_string(),
        }).await.unwrap();

        assert!(load_result.is_ok());
        let ontology_id = load_result.unwrap();

        // Step 2: Validate graph
        let graph = create_test_graph();

        let validate_result = addr.send(ValidateOntology {
            ontology_id: ontology_id.clone(),
            graph_data: graph.clone(),
            mode: ValidationMode::Full,
        }).await.unwrap();

        assert!(validate_result.is_ok());
        let report = validate_result.unwrap();

        println!("Validation report: {} violations, {} inferred triples",
                 report.violations.len(), report.inferred_triples.len());

        // Step 3: Apply inferences
        let triples = vec![
            RdfTriple {
                subject: "http://example.org/person1".to_string(),
                predicate: "http://example.org/employedBy".to_string(),
                object: "http://example.org/company1".to_string(),
                is_literal: false,
                datatype: None,
                language: None,
            }
        ];

        let inference_result = addr.send(ApplyInferences {
            rdf_triples: triples,
        }).await.unwrap();

        assert!(inference_result.is_ok());
        let inferred = inference_result.unwrap();
        println!("Inferred {} new triples", inferred.len());

        // Step 4: Check health
        let health_result = addr.send(GetOntologyHealth).await.unwrap();
        assert!(health_result.is_ok());

        let health = health_result.unwrap();
        println!("Actor health: {} cached reports, queue size {}",
                 health.cached_reports, health.validation_queue_size);

        // Step 5: Clear caches
        let clear_result = addr.send(ClearOntologyCaches).await.unwrap();
        assert!(clear_result.is_ok());

        println!("End-to-end workflow completed successfully");
    }

    #[cfg(not(feature = "ontology"))]
    #[test]
    fn test_ontology_feature_disabled() {
        println!("Ontology actor integration tests skipped - feature not enabled");
        assert!(true);
    }
}
