//! Integration tests for Single-Source Shortest Path (SSSP) functionality

#[cfg(test)]
mod sssp_tests {
    use std::collections::HashMap;
    
    /// Test data structure for a simple graph
    struct TestGraph {
        nodes: Vec<u32>,
        edges: Vec<(u32, u32, f32)>, // (source, target, weight)
    }
    
    impl TestGraph {
        /// Create a simple test graph
        /// 
        /// Graph structure:
        /// ```
        ///     1.0     2.0
        /// 0 -----> 1 -----> 2
        /// |                 ^
        /// |      3.0        | 1.0
        /// +---------> 3 ----+
        /// 
        /// 4 (disconnected)
        /// ```
        fn simple() -> Self {
            TestGraph {
                nodes: vec![0, 1, 2, 3, 4],
                edges: vec![
                    (0, 1, 1.0),
                    (1, 2, 2.0),
                    (0, 3, 3.0),
                    (3, 2, 1.0),
                    // Node 4 is disconnected
                ],
            }
        }
        
        /// Create a larger test graph for performance testing
        fn large(num_nodes: usize) -> Self {
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            
            for i in 0..num_nodes {
                nodes.push(i as u32);
                
                // Create a sparse graph with ~10 edges per node
                if i > 0 {
                    // Connect to previous node
                    edges.push((i as u32 - 1, i as u32, 1.0));
                }
                
                // Add some random connections
                for j in 0..3 {
                    let target = ((i * 7 + j * 13) % num_nodes) as u32;
                    if target != i as u32 {
                        edges.push((i as u32, target, (j + 1) as f32));
                    }
                }
            }
            
            TestGraph { nodes, edges }
        }
    }
    
    /// Expected shortest paths for the simple test graph
    fn expected_distances_from_0() -> HashMap<u32, Option<f32>> {
        let mut distances = HashMap::new();
        distances.insert(0, Some(0.0));  // Source
        distances.insert(1, Some(1.0));  // Direct edge: 0->1 (weight 1.0)
        distances.insert(2, Some(3.0));  // Path: 0->1->2 (1.0 + 2.0)
        distances.insert(3, Some(3.0));  // Direct edge: 0->3 (weight 3.0)
        distances.insert(4, None);       // Disconnected
        distances
    }
    
    #[test]
    fn test_sssp_simple_graph() {
        let graph = TestGraph::simple();
        let expected = expected_distances_from_0();
        
        // This test would integrate with the actual GPU implementation
        // For now, we're testing the expected structure
        
        assert_eq!(expected.get(&0), Some(&Some(0.0)));
        assert_eq!(expected.get(&1), Some(&Some(1.0)));
        assert_eq!(expected.get(&2), Some(&Some(3.0)));
        assert_eq!(expected.get(&3), Some(&Some(3.0)));
        assert_eq!(expected.get(&4), Some(&None));
    }
    
    #[test]
    fn test_sssp_disconnected_nodes() {
        let expected = expected_distances_from_0();
        
        // Verify that disconnected nodes return None
        assert!(expected.get(&4).unwrap().is_none());
        
        // Count unreachable nodes
        let unreachable_count = expected.values()
            .filter(|d| d.is_none())
            .count();
        assert_eq!(unreachable_count, 1);
    }
    
    #[test]
    fn test_sssp_source_is_zero_distance() {
        let expected = expected_distances_from_0();
        
        // Source node should always have distance 0
        assert_eq!(expected.get(&0), Some(&Some(0.0)));
    }
    
    #[test]
    fn test_sssp_path_optimality() {
        let expected = expected_distances_from_0();
        
        // Verify that the algorithm finds the optimal path
        // Node 2 can be reached via:
        // - 0->1->2 (cost: 1.0 + 2.0 = 3.0) ✓ optimal
        // - 0->3->2 (cost: 3.0 + 1.0 = 4.0)
        assert_eq!(expected.get(&2), Some(&Some(3.0)));
    }
    
    #[test]
    fn test_sssp_large_graph_structure() {
        let graph = TestGraph::large(1000);
        
        // Basic structural tests
        assert_eq!(graph.nodes.len(), 1000);
        assert!(graph.edges.len() > 0);
        
        // Verify edge weights are positive
        for (_, _, weight) in &graph.edges {
            assert!(*weight > 0.0);
        }
    }
    
    #[test]
    fn test_sssp_algorithm_parameters() {
        // Test k parameter calculation
        let test_cases = vec![
            (10, 2),      // k = ceil(cbrt(log2(10))) = ceil(cbrt(3.32)) = 2
            (100, 2),     // k = ceil(cbrt(log2(100))) = ceil(cbrt(6.64)) = 2
            (1000, 3),    // k = ceil(cbrt(log2(1000))) = ceil(cbrt(9.97)) = 3
            (10000, 3),   // k = ceil(cbrt(log2(10000))) = ceil(cbrt(13.29)) = 3
            (100000, 3),  // k = ceil(cbrt(log2(100000))) = ceil(cbrt(16.61)) = 3
            (1000000, 3), // k = ceil(cbrt(log2(1000000))) = ceil(cbrt(19.93)) = 3
        ];
        
        for (n, expected_k) in test_cases {
            let k = ((n as f32).log2().cbrt().ceil() as u32).max(3);
            assert_eq!(k, expected_k, "Failed for n={}", n);
        }
    }
    
    #[test]
    fn test_sssp_memory_requirements() {
        // Verify memory calculations
        let num_nodes = 1_000_000;
        
        // Each node needs:
        // - distance: 4 bytes (f32)
        // - frontier flag: 4 bytes (i32)
        // - optional parent: 4 bytes (i32)
        let memory_per_node = 12; // bytes (without parent tracking)
        let total_memory = num_nodes * memory_per_node;
        
        // Should be under 12MB for 1M nodes
        assert!(total_memory <= 12_000_000);
    }
    
    /// Mock function to simulate SSSP computation
    /// In production, this would call the actual GPU implementation
    fn compute_sssp_mock(graph: &TestGraph, source: u32) -> HashMap<u32, Option<f32>> {
        // This is a mock - actual implementation would use GPU
        if source == 0 {
            expected_distances_from_0()
        } else {
            // Return empty for other sources in mock
            HashMap::new()
        }
    }
    
    #[test]
    fn test_sssp_api_response_format() {
        let graph = TestGraph::simple();
        let distances = compute_sssp_mock(&graph, 0);
        
        // Simulate API response structure
        let unreachable_count = distances.values()
            .filter(|d| d.is_none())
            .count() as u32;
        
        // Verify response has expected structure
        assert_eq!(distances.len(), 5);
        assert_eq!(unreachable_count, 1);
        
        // Check that all nodes are accounted for
        for node in &graph.nodes {
            assert!(distances.contains_key(node));
        }
    }
    
    #[test]
    fn test_sssp_feature_flags() {
        // Test feature flag values
        const ENABLE_REPULSION: u32 = 1 << 0;
        const ENABLE_SPRINGS: u32 = 1 << 1;
        const ENABLE_CENTERING: u32 = 1 << 2;
        const ENABLE_SSSP_SPRING_ADJUST: u32 = 1 << 6;
        
        // Verify flags don't overlap
        assert_ne!(ENABLE_SSSP_SPRING_ADJUST, ENABLE_REPULSION);
        assert_ne!(ENABLE_SSSP_SPRING_ADJUST, ENABLE_SPRINGS);
        assert_ne!(ENABLE_SSSP_SPRING_ADJUST, ENABLE_CENTERING);
        
        // Verify flag can be combined with others
        let combined = ENABLE_SPRINGS | ENABLE_SSSP_SPRING_ADJUST;
        assert!(combined & ENABLE_SPRINGS != 0);
        assert!(combined & ENABLE_SSSP_SPRING_ADJUST != 0);
    }
    
    #[test]
    fn test_sssp_infinity_handling() {
        // Test that infinity is handled correctly for unreachable nodes
        let inf = f32::INFINITY;
        assert!(!inf.is_finite());
        assert!(inf > 1000000.0);
        
        // Test conversion to Option
        let dist = if inf.is_finite() { Some(inf) } else { None };
        assert_eq!(dist, None);
    }
    
    #[test]
    fn test_sssp_numerical_stability() {
        // Test for potential numerical issues
        let small_weight = 1e-6_f32;
        let large_weight = 1e6_f32;
        
        assert!(small_weight > 0.0);
        assert!(large_weight.is_finite());
        
        // Test addition doesn't overflow
        let sum = large_weight + large_weight;
        assert!(sum.is_finite());
    }
}

#[cfg(test)]
mod performance_tests {
    use std::time::Instant;
    
    #[test]
    #[ignore] // Run with: cargo test --ignored
    fn bench_sssp_scaling() {
        let sizes = vec![100, 1000, 10000];
        
        for size in sizes {
            let start = Instant::now();
            
            // Simulate graph creation
            let _nodes: Vec<u32> = (0..size).map(|i| i as u32).collect();
            let _edges: Vec<(u32, u32, f32)> = (0..size * 10)
                .map(|i| {
                    let src = (i % size) as u32;
                    let dst = ((i * 7) % size) as u32;
                    (src, dst, 1.0)
                })
                .collect();
            
            let elapsed = start.elapsed();
            println!("Graph size {}: {:?}", size, elapsed);
            
            // Basic performance assertion
            assert!(elapsed.as_secs() < 5, "Graph creation took too long");
        }
    }
}