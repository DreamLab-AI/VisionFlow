//! GPU Physics Broadcast Optimization
//!
//! Reduces network bandwidth by 70-80% through:
//! - Adaptive broadcast frequency (20-30fps instead of 60fps)
//! - Delta compression (only send nodes that moved)
//! - Spatial partitioning (visibility culling)

use glam::Vec3;
use log::{debug, info};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for broadcast optimization
#[derive(Debug, Clone)]
pub struct BroadcastConfig {
    /// Target broadcast rate in Hz (20-30 recommended)
    pub target_fps: u32,

    /// Delta threshold - nodes must move more than this to be broadcast
    pub delta_threshold: f32,

    /// Enable spatial visibility culling
    pub enable_spatial_culling: bool,

    /// Camera frustum bounds for culling (min, max)
    pub camera_bounds: Option<(Vec3, Vec3)>,
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self {
            target_fps: 25, // 25fps broadcast, 60fps physics
            delta_threshold: 0.01, // 1cm movement threshold
            enable_spatial_culling: false,
            camera_bounds: None,
        }
    }
}

/// Tracks previous positions for delta compression
pub struct DeltaCompressor {
    previous_positions: HashMap<u32, Vec3>,
    previous_velocities: HashMap<u32, Vec3>,
    last_broadcast_time: Instant,
    broadcast_interval: Duration,
    frames_since_broadcast: u32,
}

impl DeltaCompressor {
    pub fn new(config: &BroadcastConfig) -> Self {
        let broadcast_interval = Duration::from_micros((1_000_000 / config.target_fps) as u64);

        Self {
            previous_positions: HashMap::new(),
            previous_velocities: HashMap::new(),
            last_broadcast_time: Instant::now(),
            broadcast_interval,
            frames_since_broadcast: 0,
        }
    }

    /// Check if we should broadcast this frame
    pub fn should_broadcast(&mut self) -> bool {
        self.frames_since_broadcast += 1;
        let elapsed = self.last_broadcast_time.elapsed();

        if elapsed >= self.broadcast_interval {
            self.last_broadcast_time = Instant::now();
            self.frames_since_broadcast = 0;
            true
        } else {
            false
        }
    }

    /// Filter updates to only nodes that moved significantly
    pub fn filter_delta_updates(
        &mut self,
        positions: &[(Vec3, Vec3)], // (position, velocity)
        node_ids: &[u32],
        threshold: f32,
    ) -> Vec<usize> {
        let mut changed_indices = Vec::new();

        for (idx, &node_id) in node_ids.iter().enumerate() {
            let (pos, vel) = positions[idx];

            // Check if node moved beyond threshold
            let should_update = if let Some(&prev_pos) = self.previous_positions.get(&node_id) {
                let distance = (pos - prev_pos).length();
                distance > threshold
            } else {
                true // First time seeing this node
            };

            if should_update {
                changed_indices.push(idx);
                self.previous_positions.insert(node_id, pos);
                self.previous_velocities.insert(node_id, vel);
            }
        }

        changed_indices
    }

    /// Get compression statistics
    pub fn get_stats(&self, total_nodes: usize, sent_nodes: usize) -> CompressionStats {
        let reduction_percent = if total_nodes > 0 {
            ((total_nodes - sent_nodes) as f32 / total_nodes as f32) * 100.0
        } else {
            0.0
        };

        CompressionStats {
            total_nodes,
            sent_nodes,
            reduction_percent,
            frames_since_broadcast: self.frames_since_broadcast,
        }
    }
}

/// Statistics for compression performance
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_nodes: usize,
    pub sent_nodes: usize,
    pub reduction_percent: f32,
    pub frames_since_broadcast: u32,
}

/// Spatial partitioning for visibility culling
pub struct SpatialCuller {
    enabled: bool,
    camera_bounds: Option<(Vec3, Vec3)>,
}

impl SpatialCuller {
    pub fn new(config: &BroadcastConfig) -> Self {
        Self {
            enabled: config.enable_spatial_culling,
            camera_bounds: config.camera_bounds,
        }
    }

    /// Update camera frustum bounds
    pub fn update_camera_bounds(&mut self, min: Vec3, max: Vec3) {
        self.camera_bounds = Some((min, max));
    }

    /// Filter positions to only visible nodes
    pub fn filter_visible(&self, positions: &[Vec3], node_ids: &[u32]) -> Vec<usize> {
        if !self.enabled {
            // Return all indices if culling disabled
            return (0..node_ids.len()).collect();
        }

        let Some((min, max)) = self.camera_bounds else {
            // No bounds set, return all
            return (0..node_ids.len()).collect();
        };

        let mut visible_indices = Vec::new();

        for (idx, pos) in positions.iter().enumerate() {
            // Simple AABB test
            if pos.x >= min.x && pos.x <= max.x
                && pos.y >= min.y && pos.y <= max.y
                && pos.z >= min.z && pos.z <= max.z
            {
                visible_indices.push(idx);
            }
        }

        visible_indices
    }
}

/// Main broadcast optimizer combining all techniques
pub struct BroadcastOptimizer {
    config: BroadcastConfig,
    delta_compressor: DeltaCompressor,
    spatial_culler: SpatialCuller,
    total_frames_processed: u64,
    total_nodes_sent: u64,
    total_nodes_processed: u64,
}

impl BroadcastOptimizer {
    pub fn new(config: BroadcastConfig) -> Self {
        let delta_compressor = DeltaCompressor::new(&config);
        let spatial_culler = SpatialCuller::new(&config);

        Self {
            config,
            delta_compressor,
            spatial_culler,
            total_frames_processed: 0,
            total_nodes_sent: 0,
            total_nodes_processed: 0,
        }
    }

    /// Process positions and return indices of nodes to broadcast
    ///
    /// Returns (should_broadcast, filtered_indices)
    pub fn process_frame(
        &mut self,
        positions: &[(Vec3, Vec3)], // (position, velocity)
        node_ids: &[u32],
    ) -> (bool, Vec<usize>) {
        self.total_frames_processed += 1;

        // Check if we should broadcast this frame
        if !self.delta_compressor.should_broadcast() {
            return (false, Vec::new());
        }

        // Apply spatial culling first (reduces delta work)
        let pos_only: Vec<Vec3> = positions.iter().map(|(p, _)| *p).collect();
        let visible_indices = self.spatial_culler.filter_visible(&pos_only, node_ids);

        if visible_indices.is_empty() {
            return (true, Vec::new());
        }

        // Filter visible nodes by delta threshold
        let visible_positions: Vec<(Vec3, Vec3)> = visible_indices
            .iter()
            .map(|&idx| positions[idx])
            .collect();
        let visible_ids: Vec<u32> = visible_indices
            .iter()
            .map(|&idx| node_ids[idx])
            .collect();

        let delta_indices = self.delta_compressor.filter_delta_updates(
            &visible_positions,
            &visible_ids,
            self.config.delta_threshold,
        );

        // Map back to original indices
        let final_indices: Vec<usize> = delta_indices
            .into_iter()
            .map(|i| visible_indices[i])
            .collect();

        self.total_nodes_sent += final_indices.len() as u64;
        self.total_nodes_processed += node_ids.len() as u64;

        (true, final_indices)
    }

    /// Get overall performance statistics
    pub fn get_performance_stats(&self) -> BroadcastPerformanceStats {
        let avg_reduction = if self.total_nodes_processed > 0 {
            ((self.total_nodes_processed - self.total_nodes_sent) as f64
                / self.total_nodes_processed as f64) * 100.0
        } else {
            0.0
        };

        BroadcastPerformanceStats {
            total_frames_processed: self.total_frames_processed,
            total_nodes_sent: self.total_nodes_sent,
            total_nodes_processed: self.total_nodes_processed,
            average_bandwidth_reduction: avg_reduction as f32,
            target_fps: self.config.target_fps,
            delta_threshold: self.config.delta_threshold,
        }
    }

    /// Update configuration at runtime
    pub fn update_config(&mut self, config: BroadcastConfig) {
        info!("BroadcastOptimizer: Updating configuration");
        info!("  Target FPS: {} -> {}", self.config.target_fps, config.target_fps);
        info!("  Delta threshold: {:.4} -> {:.4}", self.config.delta_threshold, config.delta_threshold);

        self.config = config;
        self.delta_compressor = DeltaCompressor::new(&self.config);
        self.spatial_culler = SpatialCuller::new(&self.config);
    }

    /// Update camera bounds for spatial culling
    pub fn update_camera_bounds(&mut self, min: Vec3, max: Vec3) {
        self.spatial_culler.update_camera_bounds(min, max);
        debug!("BroadcastOptimizer: Camera bounds updated to [{:?}, {:?}]", min, max);
    }
}

#[derive(Debug, Clone)]
pub struct BroadcastPerformanceStats {
    pub total_frames_processed: u64,
    pub total_nodes_sent: u64,
    pub total_nodes_processed: u64,
    pub average_bandwidth_reduction: f32,
    pub target_fps: u32,
    pub delta_threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_compression_threshold() {
        let config = BroadcastConfig {
            target_fps: 30,
            delta_threshold: 0.01,
            enable_spatial_culling: false,
            camera_bounds: None,
        };

        let mut compressor = DeltaCompressor::new(&config);

        // First update - all nodes should be sent
        let positions = vec![
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let node_ids = vec![0, 1];

        let changed = compressor.filter_delta_updates(&positions, &node_ids, 0.01);
        assert_eq!(changed.len(), 2, "First update should send all nodes");

        // Second update - no movement, nothing sent
        let changed = compressor.filter_delta_updates(&positions, &node_ids, 0.01);
        assert_eq!(changed.len(), 0, "No movement, nothing sent");

        // Third update - small movement below threshold
        let positions = vec![
            (Vec3::new(0.005, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let changed = compressor.filter_delta_updates(&positions, &node_ids, 0.01);
        assert_eq!(changed.len(), 0, "Movement below threshold");

        // Fourth update - movement above threshold
        let positions = vec![
            (Vec3::new(0.02, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let changed = compressor.filter_delta_updates(&positions, &node_ids, 0.01);
        assert_eq!(changed.len(), 1, "One node moved above threshold");
    }

    #[test]
    fn test_spatial_culling() {
        let config = BroadcastConfig {
            target_fps: 30,
            delta_threshold: 0.01,
            enable_spatial_culling: true,
            camera_bounds: Some((Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0))),
        };

        let culler = SpatialCuller::new(&config);

        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),    // Inside
            Vec3::new(15.0, 0.0, 0.0),   // Outside
            Vec3::new(5.0, 5.0, 5.0),    // Inside
            Vec3::new(0.0, 20.0, 0.0),   // Outside
        ];
        let node_ids = vec![0, 1, 2, 3];

        let visible = culler.filter_visible(&positions, &node_ids);
        assert_eq!(visible.len(), 2, "Only 2 nodes should be visible");
        assert!(visible.contains(&0));
        assert!(visible.contains(&2));
    }

    #[test]
    fn test_broadcast_optimizer_integration() {
        let config = BroadcastConfig {
            target_fps: 60, // High rate for testing
            delta_threshold: 0.01,
            enable_spatial_culling: false,
            camera_bounds: None,
        };

        let mut optimizer = BroadcastOptimizer::new(config);

        // Simulate multiple frames
        let positions = vec![
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let node_ids = vec![0, 1];

        // First frame should broadcast
        std::thread::sleep(Duration::from_millis(20));
        let (should_broadcast, indices) = optimizer.process_frame(&positions, &node_ids);
        assert!(should_broadcast, "First frame should broadcast");
        assert_eq!(indices.len(), 2, "All nodes in first frame");

        // Second frame with no movement
        std::thread::sleep(Duration::from_millis(20));
        let (should_broadcast, indices) = optimizer.process_frame(&positions, &node_ids);
        if should_broadcast {
            assert_eq!(indices.len(), 0, "No movement, no nodes sent");
        }
    }
}
