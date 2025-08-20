//! Comprehensive GPU Safety Tests
//! 
//! Tests for all GPU safety mechanisms including bounds checking, memory validation,
//! error handling, and edge cases.

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::utils::gpu_safety::{
    GPUSafetyValidator, GPUSafetyConfig, GPUSafetyError, SafeKernelExecutor,
};
use crate::utils::memory_bounds::{
    ThreadSafeMemoryBoundsChecker, MemoryBounds, SafeArrayAccess, MemoryBoundsError,
};
use crate::gpu::safe_streaming_pipeline::{
    SafeStreamingPipeline, SafeSimplifiedNode, SafeCompressedEdge, SafeClientLOD, RenderData,
};
use crate::gpu::safe_visual_analytics::{
    SafeVec4, SafeTSNode, SafeTSEdge, SafeIsolationLayer, SafeVisualAnalyticsParams,
    SafeVisualAnalyticsGPU, SafeRenderData,
};

#[cfg(test)]
mod gpu_safety_validator_tests {
    use super::*;

    #[test]
    fn test_gpu_safety_config_validation() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Test valid buffer bounds
        assert!(validator.validate_buffer_bounds("test_nodes", 1000, 12).is_ok());
        assert!(validator.validate_buffer_bounds("test_edges", 5000, 16).is_ok());

        // Test exceeding node limits
        assert!(validator.validate_buffer_bounds("test_nodes", 2_000_000, 12).is_err());

        // Test exceeding edge limits
        assert!(validator.validate_buffer_bounds("test_edges", 10_000_000, 16).is_err());

        // Test memory overflow
        assert!(validator.validate_buffer_bounds("test_huge", usize::MAX / 2, 8).is_err());
    }

    #[test]
    fn test_kernel_parameter_validation() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Valid parameters
        assert!(validator.validate_kernel_params(1000, 2000, 10, 4, 256).is_ok());

        // Negative values
        assert!(validator.validate_kernel_params(-1, 2000, 10, 4, 256).is_err());
        assert!(validator.validate_kernel_params(1000, -1, 10, 4, 256).is_err());
        assert!(validator.validate_kernel_params(1000, 2000, -1, 4, 256).is_err());

        // Exceeding limits
        assert!(validator.validate_kernel_params(2_000_000, 2000, 10, 4, 256).is_err());
        assert!(validator.validate_kernel_params(1000, 10_000_000, 10, 4, 256).is_err());

        // Invalid grid/block sizes
        assert!(validator.validate_kernel_params(1000, 2000, 10, 0, 256).is_err());
        assert!(validator.validate_kernel_params(1000, 2000, 10, 4, 0).is_err());
        assert!(validator.validate_kernel_params(1000, 2000, 10, 4, 2048).is_err());
        assert!(validator.validate_kernel_params(1000, 2000, 10, 100000, 256).is_err());

        // Thread count overflow
        assert!(validator.validate_kernel_params(1000, 2000, 10, 65535, 1024).is_err());
    }

    #[test]
    fn test_memory_alignment_validation() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Valid aligned pointers
        let aligned_ptr = 0x1000 as *const u8; // 4KB aligned
        assert!(validator.validate_memory_alignment(aligned_ptr, 16).is_ok());
        assert!(validator.validate_memory_alignment(aligned_ptr, 32).is_ok());

        // Misaligned pointers
        let misaligned_ptr = 0x1001 as *const u8;
        assert!(validator.validate_memory_alignment(misaligned_ptr, 16).is_err());

        // Null pointer
        let null_ptr = std::ptr::null();
        assert!(validator.validate_memory_alignment(null_ptr, 16).is_err());
    }

    #[test]
    fn test_failure_tracking() {
        let config = GPUSafetyConfig {
            cpu_fallback_threshold: 3,
            ..Default::default()
        };
        let validator = GPUSafetyValidator::new(config);

        // Initially no fallback
        assert!(!validator.should_use_cpu_fallback());

        // Record failures
        validator.record_failure();
        assert!(!validator.should_use_cpu_fallback());

        validator.record_failure();
        assert!(!validator.should_use_cpu_fallback());

        validator.record_failure();
        assert!(validator.should_use_cpu_fallback());

        // Reset failures
        validator.reset_failures();
        assert!(!validator.should_use_cpu_fallback());
    }

    #[test]
    fn test_memory_tracking() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Track allocations
        assert!(validator.track_allocation("test1".to_string(), 1024).is_ok());
        assert!(validator.track_allocation("test2".to_string(), 2048).is_ok());

        let stats = validator.get_memory_stats().unwrap();
        assert_eq!(stats.current_allocated, 1024 + 2048);

        // Track deallocation
        validator.track_deallocation("test1");
        let stats = validator.get_memory_stats().unwrap();
        assert_eq!(stats.current_allocated, 2048);
    }

    #[test]
    fn test_pre_kernel_validation() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Valid data
        let nodes = vec![
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0),
        ];
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.5),
        ];

        assert!(validator.pre_kernel_validation(&nodes, &edges, 1, 256).is_ok());

        // Invalid edge references
        let invalid_edges = vec![
            (0, 5, 1.0), // Index 5 doesn't exist
        ];
        assert!(validator.pre_kernel_validation(&nodes, &invalid_edges, 1, 256).is_err());

        // Negative edge indices
        let negative_edges = vec![
            (-1, 1, 1.0),
        ];
        assert!(validator.pre_kernel_validation(&nodes, &negative_edges, 1, 256).is_err());

        // Invalid weights
        let nan_edges = vec![
            (0, 1, f32::NAN),
        ];
        assert!(validator.pre_kernel_validation(&nodes, &nan_edges, 1, 256).is_err());

        // Invalid positions
        let invalid_nodes = vec![
            (f32::INFINITY, 2.0, 3.0),
        ];
        assert!(validator.pre_kernel_validation(&invalid_nodes, &edges, 1, 256).is_err());
    }
}

#[cfg(test)]
mod memory_bounds_tests {
    use super::*;

    #[test]
    fn test_memory_bounds_creation() {
        let bounds = MemoryBounds::new("test_buffer".to_string(), 1000, 4, 4);
        
        assert_eq!(bounds.size, 1000);
        assert_eq!(bounds.element_size, 4);
        assert_eq!(bounds.element_count, 250); // 1000 / 4
        assert_eq!(bounds.alignment, 4);
    }

    #[test]
    fn test_memory_bounds_validation() {
        let bounds = MemoryBounds::new("test_buffer".to_string(), 1000, 4, 4);
        
        // Valid element access
        assert!(bounds.is_element_in_bounds(0));
        assert!(bounds.is_element_in_bounds(249)); // Last valid element
        assert!(!bounds.is_element_in_bounds(250)); // Out of bounds

        // Valid byte access
        assert!(bounds.is_byte_in_bounds(0));
        assert!(bounds.is_byte_in_bounds(999)); // Last valid byte
        assert!(!bounds.is_byte_in_bounds(1000)); // Out of bounds

        // Valid range access
        assert!(bounds.is_range_valid(0, 100));
        assert!(bounds.is_range_valid(200, 50));
        assert!(!bounds.is_range_valid(200, 100)); // Would exceed bounds
        assert!(!bounds.is_range_valid(250, 1)); // Start out of bounds
    }

    #[test]
    fn test_memory_bounds_registry() {
        let mut registry = crate::utils::memory_bounds::MemoryBoundsRegistry::new(10000);
        
        // Register allocation
        let bounds = MemoryBounds::new("test".to_string(), 1000, 4, 4);
        assert!(registry.register_allocation(bounds).is_ok());
        
        // Check access
        assert!(registry.check_element_access("test", 100, false).is_ok());
        assert!(registry.check_element_access("test", 300, false).is_err());
        
        // Check readonly
        let readonly_bounds = MemoryBounds::new("readonly".to_string(), 500, 4, 4)
            .with_readonly(true);
        assert!(registry.register_allocation(readonly_bounds).is_ok());
        
        assert!(registry.check_element_access("readonly", 50, false).is_ok());
        assert!(registry.check_element_access("readonly", 50, true).is_err());
        
        // Unregister
        assert!(registry.unregister_allocation("test").is_ok());
        assert!(registry.check_element_access("test", 100, false).is_err());
    }

    #[test]
    fn test_safe_array_access() {
        let data = vec![1, 2, 3, 4, 5];
        let mut safe_array = SafeArrayAccess::new(data, "test_array".to_string());
        
        // Valid access
        assert_eq!(*safe_array.get(0).unwrap(), 1);
        assert_eq!(*safe_array.get(4).unwrap(), 5);
        
        // Out of bounds
        assert!(safe_array.get(5).is_err());
        
        // Mutation
        *safe_array.get_mut(0).unwrap() = 10;
        assert_eq!(*safe_array.get(0).unwrap(), 10);
        
        // Slice access
        let slice = safe_array.slice(1, 3).unwrap();
        assert_eq!(slice, &[2, 3, 4]);
        
        // Invalid slice
        assert!(safe_array.slice(3, 5).is_err());
    }

    #[test]
    fn test_thread_safe_memory_bounds_checker() {
        let checker = Arc::new(ThreadSafeMemoryBoundsChecker::new(1024 * 1024));
        
        // Register allocation
        let bounds = MemoryBounds::new("test".to_string(), 1000, 4, 4);
        assert!(checker.register_allocation(bounds).is_ok());
        
        // Check access from multiple threads
        let checker_clone = checker.clone();
        let handle = std::thread::spawn(move || {
            checker_clone.check_element_access("test", 100, false)
        });
        
        assert!(handle.join().unwrap().is_ok());
        
        // Unregister
        assert!(checker.unregister_allocation("test").is_ok());
    }

    #[test]
    fn test_memory_bounds_overflow_protection() {
        let registry = crate::utils::memory_bounds::MemoryBoundsRegistry::new(1000);
        
        // This should fail due to size overflow
        let large_bounds = MemoryBounds::new("huge".to_string(), 2000, 1, 1);
        assert!(registry.register_allocation(large_bounds).is_err());
    }
}

#[cfg(test)]
mod safe_streaming_pipeline_tests {
    use super::*;
    use tokio::test;

    #[test]
    fn test_safe_simplified_node_validation() {
        // Valid node
        let valid_node = SafeSimplifiedNode::new(1.0, 2.0, 3.0, 10, 20, 30, 0);
        assert!(valid_node.is_ok());

        // Invalid coordinates
        assert!(SafeSimplifiedNode::new(f32::NAN, 2.0, 3.0, 10, 20, 30, 0).is_err());
        assert!(SafeSimplifiedNode::new(f32::INFINITY, 2.0, 3.0, 10, 20, 30, 0).is_err());
        assert!(SafeSimplifiedNode::new(1e7, 2.0, 3.0, 10, 20, 30, 0).is_err());
    }

    #[test]
    fn test_safe_compressed_edge_validation() {
        // Valid edge
        let edge = SafeCompressedEdge { source: 0, target: 1, weight: 128, bundling_id: 5 };
        assert!(edge.validate(10).is_ok());

        // Out of bounds
        assert!(edge.validate(1).is_err());

        // Self-loop
        let self_loop = SafeCompressedEdge { source: 5, target: 5, weight: 128, bundling_id: 5 };
        assert!(self_loop.validate(10).is_err());
    }

    #[test]
    fn test_safe_client_lod_validation() {
        // Valid LOD
        let valid_lod = SafeClientLOD::Mobile {
            max_nodes: 1000,
            max_edges: 2000,
            update_rate: 30,
            compression: true,
        };
        assert!(valid_lod.validate().is_ok());

        // Invalid update rate
        let invalid_lod = SafeClientLOD::Mobile {
            max_nodes: 1000,
            max_edges: 2000,
            update_rate: 0,
            compression: true,
        };
        assert!(invalid_lod.validate().is_err());

        // Excessive counts
        let excessive_lod = SafeClientLOD::Mobile {
            max_nodes: 20_000_000,
            max_edges: 2000,
            update_rate: 30,
            compression: true,
        };
        assert!(excessive_lod.validate().is_err());
    }

    #[test]
    async fn test_safe_frame_buffer() {
        let bounds_checker = Arc::new(ThreadSafeMemoryBoundsChecker::new(1024 * 1024 * 1024));
        let mut buffer = crate::gpu::safe_streaming_pipeline::SafeFrameBuffer::new(100, bounds_checker).unwrap();

        let positions = vec![1.0f32; 400]; // 100 nodes * 4 components
        let colors = vec![0.5f32; 400];
        let importance = vec![0.8f32; 100];

        // Valid update
        assert!(buffer.update_data(&positions, &colors, &importance, 1).is_ok());
        assert_eq!(buffer.get_current_frame(), 1);
        assert_eq!(buffer.get_node_count(), 100);

        // Invalid data sizes
        let invalid_positions = vec![1.0f32; 399]; // Not divisible by 4
        assert!(buffer.update_data(&invalid_positions, &colors, &importance, 2).is_err());

        let mismatched_importance = vec![0.8f32; 50]; // Wrong count
        assert!(buffer.update_data(&positions, &colors, &mismatched_importance, 2).is_err());

        // Invalid values
        let invalid_positions = vec![f32::NAN; 400];
        assert!(buffer.update_data(&invalid_positions, &colors, &importance, 2).is_err());

        let negative_importance = vec![-1.0f32; 100];
        assert!(buffer.update_data(&positions, &colors, &negative_importance, 2).is_err());

        // Access tests
        assert!(buffer.get_position(50, 0).is_ok());
        assert!(buffer.get_position(150, 0).is_err()); // Out of bounds
        assert!(buffer.get_position(50, 5).is_err()); // Invalid component

        assert!(buffer.get_importance(50).is_ok());
        assert!(buffer.get_importance(150).is_err()); // Out of bounds
    }

    #[test]
    fn test_render_data_validation() {
        // Valid render data
        let valid_data = RenderData {
            positions: vec![1.0f32; 40], // 10 nodes
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(valid_data.validate().is_ok());

        // Invalid positions length
        let invalid_data = RenderData {
            positions: vec![1.0f32; 39], // Not divisible by 4
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(invalid_data.validate().is_err());

        // Mismatched array sizes
        let mismatched_data = RenderData {
            positions: vec![1.0f32; 40], // 10 nodes
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 15], // Wrong count
            frame: 1,
        };
        assert!(mismatched_data.validate().is_err());

        // Invalid values
        let invalid_values = RenderData {
            positions: vec![f32::INFINITY; 40],
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(invalid_values.validate().is_err());
    }
}

#[cfg(test)]
mod safe_visual_analytics_tests {
    use super::*;

    #[test]
    fn test_safe_vec4_validation() {
        // Valid vector
        assert!(SafeVec4::new(1.0, 2.0, 3.0, 4.0).is_ok());

        // Invalid values
        assert!(SafeVec4::new(f32::NAN, 2.0, 3.0, 4.0).is_err());
        assert!(SafeVec4::new(f32::INFINITY, 2.0, 3.0, 4.0).is_err());
        assert!(SafeVec4::new(1e7, 2.0, 3.0, 4.0).is_err());

        // Normalization
        let vec = SafeVec4::new(3.0, 4.0, 0.0, 0.0).unwrap();
        let normalized = vec.normalize().unwrap();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-6);

        // Zero vector normalization should fail
        let zero_vec = SafeVec4::zero();
        assert!(zero_vec.normalize().is_err());
    }

    #[test]
    fn test_safe_ts_node_validation() {
        let mut node = SafeTSNode::new();
        assert!(node.validate().is_ok());

        // Invalid position
        node.position = SafeVec4 { x: f32::NAN, y: 0.0, z: 0.0, t: 0.0 };
        assert!(node.validate().is_err());

        // Reset and test invalid temporal coherence
        let mut node = SafeTSNode::new();
        node.temporal_coherence = -0.5;
        assert!(node.validate().is_err());

        // Reset and test invalid hierarchy level
        let mut node = SafeTSNode::new();
        node.hierarchy_level = -1;
        assert!(node.validate().is_err());

        // Invalid importance values
        let mut node = SafeTSNode::new();
        node.lod_importance = -1.0;
        assert!(node.validate().is_err());

        // Invalid clustering coefficient
        let mut node = SafeTSNode::new();
        node.clustering_coefficient = 1.5; // Should be <= 1.0
        assert!(node.validate().is_err());

        // Invalid damping
        let mut node = SafeTSNode::new();
        node.damping_local = 1.5; // Should be <= 1.0
        assert!(node.validate().is_err());
    }

    #[test]
    fn test_safe_ts_edge_validation() {
        // Valid edge
        assert!(SafeTSEdge::new(0, 1).is_ok());

        // Invalid indices
        assert!(SafeTSEdge::new(-1, 1).is_err());
        assert!(SafeTSEdge::new(0, -1).is_err());

        // Self-loop
        assert!(SafeTSEdge::new(5, 5).is_err());

        // Bounds checking
        let edge = SafeTSEdge::new(0, 1).unwrap();
        assert!(edge.validate(10).is_ok());
        assert!(edge.validate(1).is_err()); // target out of bounds

        // Invalid weights
        let mut edge = SafeTSEdge::new(0, 1).unwrap();
        edge.structural_weight = -1.0;
        assert!(edge.validate(10).is_err());

        let mut edge = SafeTSEdge::new(0, 1).unwrap();
        edge.formation_time = f32::INFINITY;
        assert!(edge.validate(10).is_err());
    }

    #[test]
    fn test_safe_isolation_layer_validation() {
        let layer = SafeIsolationLayer::new(0);
        assert!(layer.validate().is_ok());

        // Invalid layer ID
        let layer = SafeIsolationLayer::new(-1);
        assert!(layer.validate().is_err());

        // Invalid opacity
        let mut layer = SafeIsolationLayer::new(0);
        layer.opacity = 1.5;
        assert!(layer.validate().is_err());

        // Invalid focus radius
        let mut layer = SafeIsolationLayer::new(0);
        layer.focus_radius = -10.0;
        assert!(layer.validate().is_err());

        // Invalid temporal range
        let mut layer = SafeIsolationLayer::new(0);
        layer.temporal_range = [100.0, 50.0]; // start > end
        assert!(layer.validate().is_err());

        // Invalid force modulation
        let mut layer = SafeIsolationLayer::new(0);
        layer.force_modulation = 0.0; // Should be > 0
        assert!(layer.validate().is_err());
    }

    #[test]
    fn test_safe_visual_analytics_params_validation() {
        let mut params = SafeVisualAnalyticsParams {
            total_nodes: 1000,
            total_edges: 2000,
            active_layers: 1,
            hierarchy_depth: 3,
            current_frame: 0,
            time_step: 0.016,
            temporal_decay: 0.1,
            history_weight: 0.8,
            force_scale: [1.0, 0.5, 0.25, 0.125],
            damping: [0.9, 0.85, 0.8, 0.75],
            temperature: [1.0; 4],
            isolation_strength: 1.0,
            focus_gamma: 2.2,
            primary_focus_node: -1,
            context_alpha: 0.3,
            complexity_threshold: 0.5,
            saliency_boost: 1.5,
            information_bandwidth: 100.0,
            community_algorithm: 0,
            modularity_resolution: 1.0,
            topology_update_interval: 30,
            semantic_influence: 0.7,
            drift_threshold: 0.1,
            embedding_dims: 16,
            camera_position: SafeVec4::zero(),
            viewport_bounds: SafeVec4 { x: 2000.0, y: 2000.0, z: 1000.0, t: 100.0 },
            zoom_level: 1.0,
            time_window: 100.0,
        };

        assert!(params.validate().is_ok());

        // Negative counts
        params.total_nodes = -1;
        assert!(params.validate().is_err());

        // Excessive counts
        params.total_nodes = 20_000_000;
        assert!(params.validate().is_err());

        // Invalid time step
        params.total_nodes = 1000;
        params.time_step = -0.1;
        assert!(params.validate().is_err());

        params.time_step = 2.0; // Too large
        assert!(params.validate().is_err());

        // Invalid damping
        params.time_step = 0.016;
        params.damping[0] = 1.5; // > 1.0
        assert!(params.validate().is_err());

        // Invalid focus gamma
        params.damping[0] = 0.9;
        params.focus_gamma = 0.0; // Should be > 0
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_safe_render_data_validation() {
        // Valid data
        let valid_data = SafeRenderData {
            positions: vec![1.0f32; 40], // 10 nodes
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(valid_data.validate().is_ok());

        // Invalid positions length
        let invalid_data = SafeRenderData {
            positions: vec![1.0f32; 39], // Not divisible by 4
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(invalid_data.validate().is_err());

        // Invalid values
        let invalid_data = SafeRenderData {
            positions: vec![f32::NAN; 40],
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(invalid_data.validate().is_err());

        let invalid_data = SafeRenderData {
            positions: vec![1.0f32; 40],
            colors: vec![0.5f32; 40],
            importance: vec![-1.0f32; 10], // Negative importance
            frame: 1,
        };
        assert!(invalid_data.validate().is_err());
    }
}

#[cfg(test)]
mod cpu_fallback_tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_computation() {
        let mut positions = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let mut velocities = vec![(0.0, 0.0, 0.0); 3];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];

        let result = crate::utils::gpu_safety::cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1, 0.1, 0.9, 0.01
        );

        assert!(result.is_ok());
        
        // Positions should have changed
        assert!(positions[0] != (0.0, 0.0, 0.0) || 
                positions[1] != (1.0, 0.0, 0.0) || 
                positions[2] != (0.0, 1.0, 0.0));

        // Velocities should be updated
        assert!(velocities.iter().any(|&v| v != (0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_cpu_fallback_edge_cases() {
        // Mismatched array sizes
        let mut positions = vec![(0.0, 0.0, 0.0); 3];
        let mut velocities = vec![(0.0, 0.0, 0.0); 2]; // Wrong size
        let edges = vec![];

        let result = crate::utils::gpu_safety::cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1, 0.1, 0.9, 0.01
        );

        assert!(result.is_err());

        // Invalid edge references
        let mut positions = vec![(0.0, 0.0, 0.0); 3];
        let mut velocities = vec![(0.0, 0.0, 0.0); 3];
        let edges = vec![(0, 5, 1.0)]; // Node 5 doesn't exist

        let result = crate::utils::gpu_safety::cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1, 0.1, 0.9, 0.01
        );

        assert!(result.is_ok()); // Should skip invalid edges, not fail
    }

    #[test]
    fn test_cpu_fallback_stability() {
        // Test with coincident nodes
        let mut positions = vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)];
        let mut velocities = vec![(0.0, 0.0, 0.0); 2];
        let edges = vec![];

        let result = crate::utils::gpu_safety::cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1, 1.0, 0.9, 0.01
        );

        assert!(result.is_ok());
        
        // Nodes should be separated
        assert!(positions[0] != positions[1]);
    }

    #[test]
    fn test_cpu_fallback_velocity_clamping() {
        let mut positions = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)];
        let mut velocities = vec![(0.0, 0.0, 0.0); 2];
        let edges = vec![(0, 1, 1.0)];

        // Use very high forces to test clamping
        let result = crate::utils::gpu_safety::cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            100.0, 100.0, 0.9, 0.1
        );

        assert!(result.is_ok());
        
        // Velocities should be clamped
        for &(vx, vy, vz) in &velocities {
            let mag = (vx*vx + vy*vy + vz*vz).sqrt();
            assert!(mag <= 10.0 + 1e-6); // Allow for floating point error
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_safe_kernel_executor() {
        let config = GPUSafetyConfig {
            max_kernel_time_ms: 100,
            ..Default::default()
        };
        let validator = Arc::new(GPUSafetyValidator::new(config));
        let executor = SafeKernelExecutor::new(validator);

        // Test successful execution
        let result = executor.execute_with_timeout(|| {
            Ok("success")
        }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");

        // Test timeout
        let config = GPUSafetyConfig {
            max_kernel_time_ms: 10,
            ..Default::default()
        };
        let validator = Arc::new(GPUSafetyValidator::new(config));
        let executor = SafeKernelExecutor::new(validator);

        let result = executor.execute_with_timeout(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok("should timeout")
        }).await;
        assert!(result.is_err());
    }

    #[test]
    async fn test_complete_safety_pipeline() {
        // Create a complete safety pipeline and test it end-to-end
        let config = GPUSafetyConfig::default();
        let bounds_checker = Arc::new(ThreadSafeMemoryBoundsChecker::new(config.max_memory_bytes));
        
        // Test memory allocation
        let bounds = MemoryBounds::new("test_complete".to_string(), 1000, 4, 4);
        assert!(bounds_checker.register_allocation(bounds).is_ok());
        
        // Test access validation
        assert!(bounds_checker.check_element_access("test_complete", 100, false).is_ok());
        assert!(bounds_checker.check_element_access("test_complete", 300, false).is_err());
        
        // Test safe array with bounds checker
        let data = vec![1.0f32; 250]; // 250 elements * 4 bytes = 1000 bytes
        let safe_array = SafeArrayAccess::new(data, "test_complete".to_string())
            .with_bounds_checker(bounds_checker.clone());
        
        assert!(safe_array.get(100).is_ok());
        assert!(safe_array.get(300).is_err());
        
        // Cleanup
        assert!(bounds_checker.unregister_allocation("test_complete").is_ok());
    }

    #[test]
    fn test_error_propagation() {
        // Test that errors propagate correctly through the safety system
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Test buffer bounds error
        let result = validator.validate_buffer_bounds("test", usize::MAX, 8);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GPUSafetyError::InvalidBufferSize { .. }));

        // Test kernel params error
        let result = validator.validate_kernel_params(-1, 0, 0, 1, 256);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GPUSafetyError::InvalidKernelParams { .. }));

        // Test memory alignment error
        let misaligned_ptr = 0x1001 as *const u8;
        let result = validator.validate_memory_alignment(misaligned_ptr, 16);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GPUSafetyError::MisalignedAccess { .. }));
    }

    #[test]
    fn test_resource_exhaustion_protection() {
        // Test protection against resource exhaustion
        let config = GPUSafetyConfig {
            max_memory_bytes: 1000,
            ..Default::default()
        };
        let validator = GPUSafetyValidator::new(config);

        // Try to allocate more memory than allowed
        let result = validator.track_allocation("huge_allocation".to_string(), 2000);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GPUSafetyError::MemoryLimitExceeded { .. }));
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_validation_performance() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Test performance of bounds checking
        let start = Instant::now();
        for i in 0..10000 {
            let _ = validator.validate_buffer_bounds(&format!("test_{}", i), 1000, 12);
        }
        let elapsed = start.elapsed();
        
        // Should complete in reasonable time (< 100ms for 10k validations)
        assert!(elapsed.as_millis() < 100, "Validation too slow: {:?}", elapsed);
    }

    #[test]
    fn test_memory_bounds_performance() {
        let checker = Arc::new(ThreadSafeMemoryBoundsChecker::new(1024 * 1024 * 1024));
        
        // Register many allocations
        for i in 0..1000 {
            let bounds = MemoryBounds::new(format!("perf_test_{}", i), 1000, 4, 4);
            checker.register_allocation(bounds).unwrap();
        }

        // Test access checking performance
        let start = Instant::now();
        for i in 0..10000 {
            let name = format!("perf_test_{}", i % 1000);
            let _ = checker.check_element_access(&name, 100, false);
        }
        let elapsed = start.elapsed();
        
        // Should complete in reasonable time
        assert!(elapsed.as_millis() < 1000, "Access checking too slow: {:?}", elapsed);
        
        // Cleanup
        for i in 0..1000 {
            let name = format!("perf_test_{}", i);
            checker.unregister_allocation(&name).unwrap();
        }
    }

    #[test]
    fn test_cpu_fallback_performance() {
        // Test CPU fallback performance with larger graphs
        let num_nodes = 1000;
        let num_edges = 5000;
        
        let mut positions = vec![(0.0, 0.0, 0.0); num_nodes];
        let mut velocities = vec![(0.0, 0.0, 0.0); num_nodes];
        
        // Generate random positions
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for pos in &mut positions {
            pos.0 = rng.gen_range(-10.0..10.0);
            pos.1 = rng.gen_range(-10.0..10.0);
            pos.2 = rng.gen_range(-10.0..10.0);
        }
        
        // Generate random edges
        let mut edges = Vec::new();
        for _ in 0..num_edges {
            let src = rng.gen_range(0..num_nodes) as i32;
            let dst = rng.gen_range(0..num_nodes) as i32;
            if src != dst {
                edges.push((src, dst, rng.gen_range(0.1..2.0)));
            }
        }

        let start = Instant::now();
        let result = crate::utils::gpu_safety::cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1, 0.1, 0.9, 0.01
        );
        let elapsed = start.elapsed();
        
        assert!(result.is_ok());
        
        // Should complete in reasonable time (< 1s for 1000 nodes, 5000 edges)
        assert!(elapsed.as_secs() < 1, "CPU fallback too slow: {:?}", elapsed);
    }
}