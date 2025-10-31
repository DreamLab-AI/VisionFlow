// Constraint LOD - Level of Detail for Constraint Activation
// Week 3 Deliverable: Performance Optimization through LOD

use super::physics_constraint::*;

/// LOD (Level of Detail) level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LODLevel {
    /// Far zoom: Only highest priority constraints (60-80% reduction)
    Far = 0,

    /// Medium zoom: High + medium priority constraints
    Medium = 1,

    /// Near zoom: Most constraints active
    Near = 2,

    /// Close zoom: All constraints active
    Close = 3,
}

/// Configuration for LOD system
#[derive(Debug, Clone)]
pub struct LODConfig {
    /// Zoom thresholds for LOD levels
    /// [far_threshold, medium_threshold, near_threshold]
    pub zoom_thresholds: [f32; 3],

    /// Priority thresholds for each LOD level
    /// Only constraints with priority <= threshold are active
    pub priority_thresholds: [u8; 4],

    /// Whether to use adaptive LOD based on frame time
    pub adaptive: bool,

    /// Target frame time for adaptive LOD (milliseconds)
    pub target_frame_time: f32,

    /// Current frame time (updated dynamically)
    pub current_frame_time: f32,
}

impl Default for LODConfig {
    fn default() -> Self {
        Self {
            // Zoom thresholds: far > 1000, medium > 100, near > 10
            zoom_thresholds: [1000.0, 100.0, 10.0],

            // Priority thresholds: far=1-3, medium=1-5, near=1-7, close=all
            priority_thresholds: [3, 5, 7, 10],

            adaptive: true,
            target_frame_time: 16.67, // 60 FPS
            current_frame_time: 10.0,
        }
    }
}

/// Constraint LOD manager
pub struct ConstraintLOD {
    config: LODConfig,
    current_level: LODLevel,
    all_constraints: Vec<PhysicsConstraint>,
    active_constraints: Vec<PhysicsConstraint>,
}

impl ConstraintLOD {
    /// Create a new LOD manager
    pub fn new() -> Self {
        Self {
            config: LODConfig::default(),
            current_level: LODLevel::Close,
            all_constraints: Vec::new(),
            active_constraints: Vec::new(),
        }
    }

    /// Create a new LOD manager with custom configuration
    pub fn with_config(config: LODConfig) -> Self {
        Self {
            config,
            current_level: LODLevel::Close,
            all_constraints: Vec::new(),
            active_constraints: Vec::new(),
        }
    }

    /// Set all constraints (typically after axiom translation)
    pub fn set_constraints(&mut self, constraints: Vec<PhysicsConstraint>) {
        self.all_constraints = constraints;
        self.update_active_constraints();
    }

    /// Update zoom level and recalculate active constraints
    pub fn update_zoom(&mut self, zoom_distance: f32) {
        let new_level = self.calculate_lod_level(zoom_distance);

        if new_level != self.current_level {
            self.current_level = new_level;
            self.update_active_constraints();
        }
    }

    /// Update frame time for adaptive LOD
    pub fn update_frame_time(&mut self, frame_time_ms: f32) {
        if !self.config.adaptive {
            return;
        }

        self.config.current_frame_time = frame_time_ms;

        // If frame time exceeds target, reduce LOD level
        if frame_time_ms > self.config.target_frame_time * 1.2 {
            self.reduce_lod_level();
        }
        // If frame time is much lower, increase LOD level
        else if frame_time_ms < self.config.target_frame_time * 0.8 {
            self.increase_lod_level();
        }
    }

    /// Calculate LOD level from zoom distance
    fn calculate_lod_level(&self, zoom_distance: f32) -> LODLevel {
        if zoom_distance > self.config.zoom_thresholds[0] {
            LODLevel::Far
        } else if zoom_distance > self.config.zoom_thresholds[1] {
            LODLevel::Medium
        } else if zoom_distance > self.config.zoom_thresholds[2] {
            LODLevel::Near
        } else {
            LODLevel::Close
        }
    }

    /// Update active constraints based on current LOD level
    fn update_active_constraints(&mut self) {
        let priority_threshold = self.config.priority_thresholds[self.current_level as usize];

        self.active_constraints = self.all_constraints
            .iter()
            .filter(|c| self.should_activate_constraint(c, priority_threshold))
            .cloned()
            .collect();
    }

    /// Check if constraint should be activated at current LOD level
    fn should_activate_constraint(&self, constraint: &PhysicsConstraint, priority_threshold: u8) -> bool {
        // User-defined constraints always active
        if constraint.user_defined {
            return true;
        }

        // Check priority threshold
        if constraint.priority > priority_threshold {
            return false;
        }

        // Always activate hierarchical layer constraints (important for structure)
        if matches!(constraint.constraint_type, PhysicsConstraintType::HierarchicalLayer { .. }) {
            return true;
        }

        true
    }

    /// Reduce LOD level (show fewer constraints)
    fn reduce_lod_level(&mut self) {
        self.current_level = match self.current_level {
            LODLevel::Close => LODLevel::Near,
            LODLevel::Near => LODLevel::Medium,
            LODLevel::Medium => LODLevel::Far,
            LODLevel::Far => LODLevel::Far,
        };

        self.update_active_constraints();
    }

    /// Increase LOD level (show more constraints)
    fn increase_lod_level(&mut self) {
        self.current_level = match self.current_level {
            LODLevel::Far => LODLevel::Medium,
            LODLevel::Medium => LODLevel::Near,
            LODLevel::Near => LODLevel::Close,
            LODLevel::Close => LODLevel::Close,
        };

        self.update_active_constraints();
    }

    /// Get active constraints for current LOD level
    pub fn get_active_constraints(&self) -> &[PhysicsConstraint] {
        &self.active_constraints
    }

    /// Get all constraints
    pub fn get_all_constraints(&self) -> &[PhysicsConstraint] {
        &self.all_constraints
    }

    /// Get current LOD level
    pub fn get_current_level(&self) -> LODLevel {
        self.current_level
    }

    /// Get reduction percentage
    pub fn get_reduction_percentage(&self) -> f32 {
        if self.all_constraints.is_empty() {
            return 0.0;
        }

        let reduction = 1.0 - (self.active_constraints.len() as f32 / self.all_constraints.len() as f32);
        reduction * 100.0
    }

    /// Get LOD statistics
    pub fn get_stats(&self) -> LODStats {
        LODStats {
            lod_level: self.current_level,
            total_constraints: self.all_constraints.len(),
            active_constraints: self.active_constraints.len(),
            reduction_percentage: self.get_reduction_percentage(),
            frame_time_ms: self.config.current_frame_time,
            target_frame_time_ms: self.config.target_frame_time,
        }
    }

    /// Force set LOD level
    pub fn set_lod_level(&mut self, level: LODLevel) {
        self.current_level = level;
        self.update_active_constraints();
    }
}

impl Default for ConstraintLOD {
    fn default() -> Self {
        Self::new()
    }
}

/// LOD statistics
#[derive(Debug, Clone)]
pub struct LODStats {
    pub lod_level: LODLevel,
    pub total_constraints: usize,
    pub active_constraints: usize,
    pub reduction_percentage: f32,
    pub frame_time_ms: f32,
    pub target_frame_time_ms: f32,
}

impl std::fmt::Display for LODStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LOD {:?}: {}/{} constraints active ({:.1}% reduction) | Frame: {:.2}ms / {:.2}ms",
            self.lod_level,
            self.active_constraints,
            self.total_constraints,
            self.reduction_percentage,
            self.frame_time_ms,
            self.target_frame_time_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_constraints() -> Vec<PhysicsConstraint> {
        vec![
            PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 1), // Priority 1 (user-defined level)
            PhysicsConstraint::separation(vec![2, 3], 15.0, 0.6, 3), // Priority 3 (inferred)
            PhysicsConstraint::clustering(vec![3, 4], 20.0, 0.6, 5), // Priority 5 (asserted)
            PhysicsConstraint::clustering(vec![4, 5], 25.0, 0.7, 7), // Priority 7 (default)
            PhysicsConstraint::colocation(vec![5, 6], 2.0, 0.9, 10), // Priority 10 (lowest)
        ]
    }

    #[test]
    fn test_lod_level_calculation() {
        let lod = ConstraintLOD::new();

        assert_eq!(lod.calculate_lod_level(2000.0), LODLevel::Far);
        assert_eq!(lod.calculate_lod_level(500.0), LODLevel::Medium);
        assert_eq!(lod.calculate_lod_level(50.0), LODLevel::Near);
        assert_eq!(lod.calculate_lod_level(5.0), LODLevel::Close);
    }

    #[test]
    fn test_far_lod_reduction() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(2000.0); // Far zoom

        let active = lod.get_active_constraints();

        // At Far LOD, only priority <= 3 should be active
        assert!(active.len() <= 2);
        assert!(active.iter().all(|c| c.priority <= 3));
    }

    #[test]
    fn test_medium_lod() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(500.0); // Medium zoom

        let active = lod.get_active_constraints();

        // At Medium LOD, only priority <= 5 should be active
        assert!(active.len() <= 3);
        assert!(active.iter().all(|c| c.priority <= 5));
    }

    #[test]
    fn test_near_lod() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(50.0); // Near zoom

        let active = lod.get_active_constraints();

        // At Near LOD, only priority <= 7 should be active
        assert!(active.len() <= 4);
        assert!(active.iter().all(|c| c.priority <= 7));
    }

    #[test]
    fn test_close_lod_all_active() {
        let mut lod = ConstraintLOD::new();
        let constraints = create_test_constraints();
        lod.set_constraints(constraints.clone());

        lod.update_zoom(5.0); // Close zoom

        let active = lod.get_active_constraints();

        // At Close LOD, all constraints should be active
        assert_eq!(active.len(), constraints.len());
    }

    #[test]
    fn test_user_defined_always_active() {
        let mut lod = ConstraintLOD::new();

        let mut constraints = create_test_constraints();
        constraints.push(
            PhysicsConstraint::separation(vec![10, 11], 30.0, 0.9, 10)
                .mark_user_defined()
        );

        lod.set_constraints(constraints);
        lod.update_zoom(2000.0); // Far zoom

        let active = lod.get_active_constraints();

        // User-defined constraint should be active even at far zoom
        assert!(active.iter().any(|c| c.user_defined));
    }

    #[test]
    fn test_adaptive_lod_frame_time() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        // Start at close LOD
        lod.set_lod_level(LODLevel::Close);
        assert_eq!(lod.get_current_level(), LODLevel::Close);

        // Simulate high frame time (slow performance)
        lod.update_frame_time(25.0); // > target 16.67ms

        // LOD should reduce
        assert!(lod.get_current_level() < LODLevel::Close);
    }

    #[test]
    fn test_reduction_percentage() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(2000.0); // Far zoom

        let reduction = lod.get_reduction_percentage();

        // Should have significant reduction at far zoom
        assert!(reduction > 40.0);
        assert!(reduction <= 100.0);
    }

    #[test]
    fn test_lod_stats() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(500.0);
        lod.update_frame_time(15.0);

        let stats = lod.get_stats();

        assert_eq!(stats.lod_level, LODLevel::Medium);
        assert_eq!(stats.total_constraints, 5);
        assert!(stats.active_constraints <= 5);
        assert_eq!(stats.frame_time_ms, 15.0);
    }

    #[test]
    fn test_hierarchical_always_active() {
        let mut lod = ConstraintLOD::new();

        let mut constraints = vec![
            PhysicsConstraint::hierarchical_layer(vec![1, 2], 100.0, 0.7, 10),
            PhysicsConstraint::separation(vec![3, 4], 10.0, 0.5, 10),
        ];

        lod.set_constraints(constraints);
        lod.update_zoom(2000.0); // Far zoom

        let active = lod.get_active_constraints();

        // Hierarchical constraint should be active even with priority 10
        assert!(active.iter().any(|c| matches!(
            c.constraint_type,
            PhysicsConstraintType::HierarchicalLayer { .. }
        )));
    }

    #[test]
    fn test_custom_config() {
        let config = LODConfig {
            zoom_thresholds: [500.0, 100.0, 20.0],
            priority_thresholds: [2, 4, 6, 10],
            adaptive: false,
            target_frame_time: 33.33, // 30 FPS
            current_frame_time: 20.0,
        };

        let mut lod = ConstraintLOD::with_config(config);
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(600.0); // Should be Far with custom thresholds
        assert_eq!(lod.get_current_level(), LODLevel::Far);
    }

    #[test]
    fn test_empty_constraints() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(vec![]);

        lod.update_zoom(5.0);

        assert_eq!(lod.get_active_constraints().len(), 0);
        assert_eq!(lod.get_reduction_percentage(), 0.0);
    }
}
