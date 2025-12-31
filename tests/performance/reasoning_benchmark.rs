/// Performance Benchmarks for Reasoning System
///
/// Benchmarks reasoning performance with ontologies of various sizes
///
/// NOTE: These tests are disabled because the inference_cache module
/// does not exist in the reasoning module. The reasoning module only
/// exports custom_reasoner.

#[cfg(feature = "ontology")]
mod reasoning_benchmarks {
    use webxr::reasoning::custom_reasoner::CustomReasoner;
    // NOTE: inference_cache module does not exist - commenting out related tests
    // use webxr::reasoning::inference_cache::InferenceCache;
    use std::time::Instant;
    use tempfile::TempDir;

    /// Generate a small test ontology (10 classes, 20 axioms)
    fn generate_small_ontology() -> String {
        r#"<?xml version="1.0"?>
<Ontology xmlns="http://www.w3.org/2002/07/owl#" ontologyIRI="http://bench.org/small">
    <Class rdf:about="http://bench.org/small#A"/>
    <Class rdf:about="http://bench.org/small#B">
        <rdfs:subClassOf rdf:resource="http://bench.org/small#A"/>
    </Class>
    <Class rdf:about="http://bench.org/small#C">
        <rdfs:subClassOf rdf:resource="http://bench.org/small#B"/>
    </Class>
    <Class rdf:about="http://bench.org/small#D">
        <rdfs:subClassOf rdf:resource="http://bench.org/small#C"/>
    </Class>
    <Class rdf:about="http://bench.org/small#E">
        <rdfs:subClassOf rdf:resource="http://bench.org/small#D"/>
    </Class>
    <Class rdf:about="http://bench.org/small#X">
        <disjointWith rdf:resource="http://bench.org/small#A"/>
    </Class>
    <Class rdf:about="http://bench.org/small#Y">
        <rdfs:subClassOf rdf:resource="http://bench.org/small#X"/>
    </Class>
    <Class rdf:about="http://bench.org/small#Z">
        <rdfs:subClassOf rdf:resource="http://bench.org/small#Y"/>
    </Class>
    <Class rdf:about="http://bench.org/small#P">
        <equivalentClass rdf:resource="http://bench.org/small#E"/>
    </Class>
    <Class rdf:about="http://bench.org/small#Q">
        <equivalentClass rdf:resource="http://bench.org/small#Z"/>
    </Class>
</Ontology>"#.to_string()
    }

    /// Generate a medium test ontology (100 classes, 200 axioms)
    fn generate_medium_ontology() -> String {
        let mut owl = String::from(r#"<?xml version="1.0"?>
<Ontology xmlns="http://www.w3.org/2002/07/owl#"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    ontologyIRI="http://bench.org/medium">
"#);

        // Generate 100 classes in a hierarchy
        for i in 0..100 {
            owl.push_str(&format!(
                r#"    <Class rdf:about="http://bench.org/medium#Class{}"/>"#,
                i
            ));
            owl.push('\n');

            if i > 0 && i % 10 != 0 {
                // Create subclass relationships
                owl.push_str(&format!(
                    r#"    <Class rdf:about="http://bench.org/medium#Class{}">
        <rdfs:subClassOf rdf:resource="http://bench.org/medium#Class{}"/>
    </Class>"#,
                    i,
                    i - 1
                ));
                owl.push('\n');
            }

            if i % 20 == 0 && i > 0 {
                // Create some disjoint relationships
                owl.push_str(&format!(
                    r#"    <Class rdf:about="http://bench.org/medium#Class{}">
        <disjointWith rdf:resource="http://bench.org/medium#Class{}"/>
    </Class>"#,
                    i,
                    i - 10
                ));
                owl.push('\n');
            }
        }

        owl.push_str("</Ontology>");
        owl
    }

    /// Generate a large test ontology (1000 classes, 5000 axioms)
    fn generate_large_ontology() -> String {
        let mut owl = String::from(r#"<?xml version="1.0"?>
<Ontology xmlns="http://www.w3.org/2002/07/owl#"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    ontologyIRI="http://bench.org/large">
"#);

        // Generate 1000 classes
        for i in 0..1000 {
            owl.push_str(&format!(
                r#"    <Class rdf:about="http://bench.org/large#Class{}"/>"#,
                i
            ));
            owl.push('\n');

            if i > 0 {
                // Create dense subclass hierarchy
                owl.push_str(&format!(
                    r#"    <Class rdf:about="http://bench.org/large#Class{}">
        <rdfs:subClassOf rdf:resource="http://bench.org/large#Class{}"/>
    </Class>"#,
                    i,
                    i - 1
                ));
                owl.push('\n');

                // Additional subclass to root every 100 classes
                if i % 100 == 0 {
                    owl.push_str(&format!(
                        r#"    <Class rdf:about="http://bench.org/large#Class{}">
        <rdfs:subClassOf rdf:resource="http://bench.org/large#Class0"/>
    </Class>"#,
                        i
                    ));
                    owl.push('\n');
                }
            }

            if i % 50 == 0 && i > 0 {
                // Create disjoint relationships
                owl.push_str(&format!(
                    r#"    <Class rdf:about="http://bench.org/large#Class{}">
        <disjointWith rdf:resource="http://bench.org/large#Class{}"/>
    </Class>"#,
                    i,
                    i - 25
                ));
                owl.push('\n');
            }

            if i % 100 == 0 && i > 0 && i < 500 {
                // Create equivalent classes
                owl.push_str(&format!(
                    r#"    <Class rdf:about="http://bench.org/large#Equiv{}">
        <equivalentClass rdf:resource="http://bench.org/large#Class{}"/>
    </Class>"#,
                    i,
                    i
                ));
                owl.push('\n');
            }
        }

        owl.push_str("</Ontology>");
        owl
    }

    #[test]
    fn benchmark_small_ontology_reasoning() {
        let owl = generate_small_ontology();
        let reasoner = CustomReasoner::new();

        let start = Instant::now();
        let result = reasoner.process_owl(&owl);
        let duration = start.elapsed();

        assert!(result.is_ok(), "Small ontology reasoning should succeed");
        let axioms = result.unwrap();

        println!("\n=== SMALL ONTOLOGY BENCHMARK ===");
        println!("Size: 10 classes, ~20 axioms");
        println!("Duration: {}ms", duration.as_millis());
        println!("Inferred axioms: {}", axioms.len());
        println!("Throughput: {:.2} axioms/ms", axioms.len() as f64 / duration.as_millis() as f64);

        assert!(
            duration.as_millis() < 50,
            "Small ontology should process in <50ms, took {}ms",
            duration.as_millis()
        );
    }

    #[test]
    fn benchmark_medium_ontology_reasoning() {
        let owl = generate_medium_ontology();
        let reasoner = CustomReasoner::new();

        let start = Instant::now();
        let result = reasoner.process_owl(&owl);
        let duration = start.elapsed();

        assert!(result.is_ok(), "Medium ontology reasoning should succeed");
        let axioms = result.unwrap();

        println!("\n=== MEDIUM ONTOLOGY BENCHMARK ===");
        println!("Size: 100 classes, ~200 axioms");
        println!("Duration: {}ms", duration.as_millis());
        println!("Inferred axioms: {}", axioms.len());
        println!("Throughput: {:.2} axioms/ms", axioms.len() as f64 / duration.as_millis() as f64);

        assert!(
            duration.as_millis() < 500,
            "Medium ontology should process in <500ms, took {}ms",
            duration.as_millis()
        );
    }

    #[test]
    fn benchmark_large_ontology_reasoning() {
        let owl = generate_large_ontology();
        let reasoner = CustomReasoner::new();

        let start = Instant::now();
        let result = reasoner.process_owl(&owl);
        let duration = start.elapsed();

        assert!(result.is_ok(), "Large ontology reasoning should succeed");
        let axioms = result.unwrap();

        println!("\n=== LARGE ONTOLOGY BENCHMARK ===");
        println!("Size: 1000 classes, ~5000 axioms");
        println!("Duration: {}ms", duration.as_millis());
        println!("Inferred axioms: {}", axioms.len());
        println!("Throughput: {:.2} axioms/ms", axioms.len() as f64 / duration.as_millis() as f64);

        assert!(
            duration.as_secs() < 5,
            "Large ontology should process in <5s, took {}s",
            duration.as_secs()
        );
    }

    #[test]
    fn benchmark_constraint_generation() {
        let owl = generate_medium_ontology();
        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl).expect("Reasoning failed");

        // Create node mapping for 100 classes
        let mut node_id_to_index = std::collections::HashMap::new();
        for i in 0..100 {
            node_id_to_index.insert(format!("Class{}", i), i);
        }

        use webxr::gpu::semantic_forces::SemanticForceGenerator;
        let force_gen = SemanticForceGenerator::new();

        let start = Instant::now();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);
        let duration = start.elapsed();

        println!("\n=== CONSTRAINT GENERATION BENCHMARK ===");
        println!("Node count: 100");
        println!("Inferred axioms: {}", inferred.len());
        println!("Generated constraints: {}", constraints.len());
        println!("Duration: {}ms", duration.as_millis());
        println!("Throughput: {:.2} constraints/ms", constraints.len() as f64 / duration.as_millis() as f64);

        assert!(
            duration.as_millis() < 100,
            "Constraint generation should be <100ms for 100 nodes"
        );
    }

    // NOTE: InferenceCache does not exist in the reasoning module
    // Commenting out this test until the module is available
    /*
    #[test]
    fn benchmark_cache_performance() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache = InferenceCache::new(temp_dir.path().to_path_buf());

        let owl = generate_medium_ontology();
        let reasoner = CustomReasoner::new();

        // First run - cache miss
        let start1 = Instant::now();
        let axioms1 = reasoner.process_owl(&owl).expect("Reasoning failed");
        let duration1 = start1.elapsed();
        cache.set(&owl, axioms1.clone());

        // Second run - cache hit
        let start2 = Instant::now();
        let cached = cache.get(&owl).expect("Cache miss unexpected");
        let duration2 = start2.elapsed();

        println!("\n=== CACHE PERFORMANCE BENCHMARK ===");
        println!("Ontology size: 100 classes");
        println!("First run (cache miss): {}ms", duration1.as_millis());
        println!("Second run (cache hit): {}μs", duration2.as_micros());
        println!("Speedup: {:.2}x", duration1.as_micros() as f64 / duration2.as_micros() as f64);

        assert_eq!(axioms1.len(), cached.len(), "Cache should return identical results");
        assert!(
            duration2.as_millis() < 10,
            "Cache hit should be <10ms"
        );
    }
    */

    #[test]
    fn benchmark_parallel_reasoning() {
        use rayon::prelude::*;

        let ontologies: Vec<String> = (0..10)
            .map(|_| generate_small_ontology())
            .collect();

        let start = Instant::now();
        let results: Vec<_> = ontologies.par_iter()
            .map(|owl| {
                let reasoner = CustomReasoner::new();
                reasoner.process_owl(owl)
            })
            .collect();
        let duration = start.elapsed();

        let success_count = results.iter().filter(|r| r.is_ok()).count();

        println!("\n=== PARALLEL REASONING BENCHMARK ===");
        println!("Ontology count: 10");
        println!("Successful: {}", success_count);
        println!("Total duration: {}ms", duration.as_millis());
        println!("Average per ontology: {}ms", duration.as_millis() / 10);

        assert_eq!(success_count, 10, "All parallel reasonings should succeed");
    }

    #[test]
    fn benchmark_gpu_constraint_application() {
        // Simulate GPU constraint application performance

        let owl = generate_medium_ontology();
        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl).expect("Reasoning failed");

        let mut node_id_to_index = std::collections::HashMap::new();
        for i in 0..100 {
            node_id_to_index.insert(format!("Class{}", i), i);
        }

        use webxr::gpu::semantic_forces::SemanticForceGenerator;
        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Simulate constraint application (CPU version for benchmark)
        let mut positions = vec![[0.0f32, 0.0, 0.0]; 100];
        let mut velocities = vec![[0.0f32, 0.0, 0.0]; 100];

        let start = Instant::now();
        for _ in 0..100 { // 100 physics iterations
            for constraint in &constraints {
                let idx_a = constraint.node_a as usize;
                let idx_b = constraint.node_b as usize;

                let dx = positions[idx_b][0] - positions[idx_a][0];
                let dy = positions[idx_b][1] - positions[idx_a][1];
                let dz = positions[idx_b][2] - positions[idx_a][2];

                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let force = constraint.strength * (dist - constraint.distance_target);

                if dist > 0.001 {
                    let fx = (dx / dist) * force;
                    let fy = (dy / dist) * force;
                    let fz = (dz / dist) * force;

                    velocities[idx_a][0] -= fx;
                    velocities[idx_a][1] -= fy;
                    velocities[idx_a][2] -= fz;

                    velocities[idx_b][0] += fx;
                    velocities[idx_b][1] += fy;
                    velocities[idx_b][2] += fz;
                }
            }

            // Update positions
            for i in 0..100 {
                positions[i][0] += velocities[i][0] * 0.01;
                positions[i][1] += velocities[i][1] * 0.01;
                positions[i][2] += velocities[i][2] * 0.01;
            }
        }
        let duration = start.elapsed();

        println!("\n=== GPU CONSTRAINT APPLICATION BENCHMARK (CPU simulation) ===");
        println!("Nodes: 100");
        println!("Constraints: {}", constraints.len());
        println!("Iterations: 100");
        println!("Duration: {}ms", duration.as_millis());
        println!("Per iteration: {}μs", duration.as_micros() / 100);

        // Note: Actual GPU version should be much faster
        println!("Note: GPU implementation expected to be 10-100x faster");
    }
}
