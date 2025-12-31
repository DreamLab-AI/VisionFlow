/// End-to-End Pipeline Integration Test
///
/// Tests the complete flow: Upload OWL → Reasoning → Constraint Generation → GPU → Client
///
/// NOTE: These tests are disabled because the reasoning_actor module
/// does not exist in the reasoning module. The reasoning module only
/// exports custom_reasoner.

#[cfg(all(feature = "ontology", feature = "gpu"))]
mod pipeline_e2e {
    use actix_web::{test, web, App};
    // NOTE: reasoning_actor module does not exist - commenting out all tests
    // use webxr::reasoning::reasoning_actor::ReasoningActor;
    use webxr::gpu::semantic_forces::SemanticForceGenerator;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use std::time::Instant;
    use std::path::PathBuf;
    use std::fs;

    // All tests in this module are disabled because ReasoningActor does not exist
    // Re-enable when the module is available

    fn get_test_ontology() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ontologies/test_reasoning.owl");
        fs::read_to_string(&path).expect("Failed to load test ontology")
    }

    // NOTE: All tests below are disabled because ReasoningActor does not exist
    /*
    #[actix_web::test]
    async fn test_owl_upload_triggers_reasoning() {
        // Simulate OWL file upload via GitHub sync endpoint

        let owl_content = get_test_ontology();

        // Create reasoning actor
        let reasoning_actor = Arc::new(RwLock::new(ReasoningActor::new()));

        // Process OWL content
        let start = Instant::now();
        let result = reasoning_actor.write().await.process_owl_content(&owl_content).await;
        let reasoning_duration = start.elapsed();

        assert!(result.is_ok(), "OWL processing should succeed");

        let inferred_axioms = result.unwrap();
        assert!(!inferred_axioms.is_empty(), "Should produce inferred axioms");

        println!(
            "Reasoning completed in {}ms, produced {} axioms",
            reasoning_duration.as_millis(),
            inferred_axioms.len()
        );

        // Verify reasoning was triggered automatically
        assert!(
            reasoning_duration.as_millis() < 100,
            "Reasoning should complete quickly for test ontology"
        );
    }

    #[actix_web::test]
    async fn test_constraints_generated_with_correct_indices() {
        let owl_content = get_test_ontology();

        // Create reasoning actor and process
        let reasoning_actor = Arc::new(RwLock::new(ReasoningActor::new()));
        let inferred = reasoning_actor.write().await
            .process_owl_content(&owl_content).await
            .expect("Reasoning failed");

        // Create node mapping (simulating graph state)
        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Person".to_string(), 0);
        node_id_to_index.insert("Organization".to_string(), 1);
        node_id_to_index.insert("Company".to_string(), 2);
        node_id_to_index.insert("Employee".to_string(), 3);
        node_id_to_index.insert("Manager".to_string(), 4);

        // Generate constraints
        let start = Instant::now();
        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);
        let generation_duration = start.elapsed();

        println!(
            "Generated {} constraints in {}ms",
            constraints.len(),
            generation_duration.as_millis()
        );

        // Verify constraint indices are valid
        for constraint in &constraints {
            assert!(
                constraint.node_a < 5,
                "node_a index {} exceeds node count",
                constraint.node_a
            );
            assert!(
                constraint.node_b < 5,
                "node_b index {} exceeds node count",
                constraint.node_b
            );
        }

        // Verify constraint types
        let has_repulsion = constraints.iter().any(|c| c.constraint_type == "repulsion");
        let has_attraction = constraints.iter().any(|c| c.constraint_type == "attraction");

        assert!(has_repulsion, "Should have repulsion constraints");
        assert!(has_attraction, "Should have attraction constraints");
    }

    #[actix_web::test]
    async fn test_gpu_receives_constraints() {
        use webxr::gpu::types::SemanticConstraint;

        let owl_content = get_test_ontology();

        // Process reasoning
        let reasoning_actor = Arc::new(RwLock::new(ReasoningActor::new()));
        let inferred = reasoning_actor.write().await
            .process_owl_content(&owl_content).await
            .expect("Reasoning failed");

        // Generate constraints
        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Person".to_string(), 0);
        node_id_to_index.insert("Organization".to_string(), 1);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Simulate GPU constraint buffer preparation
        let gpu_buffer_size = constraints.len() * std::mem::size_of::<SemanticConstraint>();

        assert!(
            gpu_buffer_size > 0,
            "GPU buffer should have non-zero size"
        );

        println!(
            "Prepared GPU buffer: {} constraints, {} bytes",
            constraints.len(),
            gpu_buffer_size
        );

        // Verify constraints are in GPU-friendly format
        for constraint in &constraints {
            // Check that all fields are finite and valid
            assert!(constraint.strength.is_finite());
            assert!(constraint.distance_target.is_finite());
            assert!(constraint.strength >= 0.0);
        }
    }

    #[actix_web::test]
    async fn test_client_receives_updated_positions() {
        // This test simulates the full WebSocket flow
        // In practice, this would require a running server and WebSocket connection

        let owl_content = get_test_ontology();

        // Step 1: Upload and process OWL
        let reasoning_actor = Arc::new(RwLock::new(ReasoningActor::new()));
        let start_total = Instant::now();

        let inferred = reasoning_actor.write().await
            .process_owl_content(&owl_content).await
            .expect("Reasoning failed");

        // Step 2: Generate constraints
        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Person".to_string(), 0);
        node_id_to_index.insert("Organization".to_string(), 1);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Step 3: Simulate GPU physics update
        // (In real scenario, GPU kernel applies forces and updates positions)
        let mut node_positions = vec![
            [0.0f32, 0.0, 0.0], // Person
            [1.0f32, 0.0, 0.0], // Organization
        ];

        // Simulate repulsion between disjoint classes
        for constraint in &constraints {
            if constraint.constraint_type == "repulsion" {
                let idx_a = constraint.node_a as usize;
                let idx_b = constraint.node_b as usize;

                // Simple repulsion simulation
                let dx = node_positions[idx_b][0] - node_positions[idx_a][0];
                let force = constraint.strength * 0.1;

                if dx > 0.0 {
                    node_positions[idx_a][0] -= force;
                    node_positions[idx_b][0] += force;
                } else {
                    node_positions[idx_a][0] += force;
                    node_positions[idx_b][0] -= force;
                }
            }
        }

        // Step 4: Verify positions changed
        let distance = (node_positions[1][0] - node_positions[0][0]).abs();
        assert!(
            distance > 1.0,
            "Disjoint classes should be separated, distance: {}",
            distance
        );

        let total_duration = start_total.elapsed();
        println!(
            "Complete pipeline: Upload → Reasoning → GPU → Client in {}ms",
            total_duration.as_millis()
        );

        // Step 5: Verify latency is acceptable
        assert!(
            total_duration.as_millis() < 200,
            "Total pipeline latency should be <200ms for small ontology"
        );
    }

    #[actix_web::test]
    async fn test_cache_improves_second_upload() {
        let owl_content = get_test_ontology();

        let reasoning_actor = Arc::new(RwLock::new(ReasoningActor::new()));

        // First upload - cache miss
        let start1 = Instant::now();
        let result1 = reasoning_actor.write().await
            .process_owl_content(&owl_content).await;
        let duration1 = start1.elapsed();

        assert!(result1.is_ok());

        // Second upload - cache hit
        let start2 = Instant::now();
        let result2 = reasoning_actor.write().await
            .process_owl_content(&owl_content).await;
        let duration2 = start2.elapsed();

        assert!(result2.is_ok());

        println!(
            "First upload: {}ms, Second upload (cached): {}ms",
            duration1.as_millis(),
            duration2.as_millis()
        );

        // Cache hit should be significantly faster
        assert!(
            duration2 < duration1,
            "Cached processing should be faster"
        );
    }

    #[actix_web::test]
    async fn test_concurrent_reasoning_requests() {
        use tokio::task::JoinSet;

        let owl_content = get_test_ontology();
        let reasoning_actor = Arc::new(RwLock::new(ReasoningActor::new()));

        let mut tasks = JoinSet::new();

        // Spawn 5 concurrent reasoning tasks
        for i in 0..5 {
            let actor = reasoning_actor.clone();
            let owl = owl_content.clone();

            tasks.spawn(async move {
                let start = Instant::now();
                let result = actor.write().await.process_owl_content(&owl).await;
                (i, result, start.elapsed())
            });
        }

        // Collect results
        let mut durations = Vec::new();
        while let Some(result) = tasks.join_next().await {
            let (id, res, duration) = result.expect("Task failed");
            assert!(res.is_ok(), "Request {} failed", id);
            durations.push(duration);
            println!("Request {} completed in {}ms", id, duration.as_millis());
        }

        // Verify all completed
        assert_eq!(durations.len(), 5);

        // First request likely cache miss, rest should be hits
        let avg_duration: u128 = durations.iter()
            .map(|d| d.as_millis())
            .sum::<u128>() / durations.len() as u128;

        println!("Average duration: {}ms", avg_duration);
    }

    #[actix_web::test]
    async fn test_reasoning_error_handling() {
        let invalid_owl = "This is not valid OWL XML content";

        let reasoning_actor = Arc::new(RwLock::new(ReasoningActor::new()));
        let result = reasoning_actor.write().await
            .process_owl_content(invalid_owl).await;

        assert!(result.is_err(), "Invalid OWL should return error");

        // Verify system is still functional after error
        let valid_owl = get_test_ontology();
        let result2 = reasoning_actor.write().await
            .process_owl_content(&valid_owl).await;

        assert!(result2.is_ok(), "System should recover from errors");
    }
    */
}
