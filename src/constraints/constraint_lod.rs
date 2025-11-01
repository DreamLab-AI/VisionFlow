// Constraint LOD - Level of Detail for Constraint Activation
// Week 3 Deliverable: Performance Optimization through LOD

use super::physics_constraint::*;

/
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LODLevel {
    
    Far = 0,

    
    Medium = 1,

    
    Near = 2,

    
    Close = 3,
}

/
#[derive(Debug, Clone)]
pub struct LODConfig {
    
    
    pub zoom_thresholds: [f32; 3],

    
    
    pub priority_thresholds: [u8; 4],

    
    pub adaptive: bool,

    
    pub target_frame_time: f32,

    
    pub current_frame_time: f32,
}

impl Default for LODConfig {
    fn default() -> Self {
        Self {
            
            zoom_thresholds: [1000.0, 100.0, 10.0],

            
            priority_thresholds: [3, 5, 7, 10],

            adaptive: true,
            target_frame_time: 16.67, 
            current_frame_time: 10.0,
        }
    }
}

/
pub struct ConstraintLOD {
    config: LODConfig,
    current_level: LODLevel,
    all_constraints: Vec<PhysicsConstraint>,
    active_constraints: Vec<PhysicsConstraint>,
}

impl ConstraintLOD {
    
    pub fn new() -> Self {
        Self {
            config: LODConfig::default(),
            current_level: LODLevel::Close,
            all_constraints: Vec::new(),
            active_constraints: Vec::new(),
        }
    }

    
    pub fn with_config(config: LODConfig) -> Self {
        Self {
            config,
            current_level: LODLevel::Close,
            all_constraints: Vec::new(),
            active_constraints: Vec::new(),
        }
    }

    
    pub fn set_constraints(&mut self, constraints: Vec<PhysicsConstraint>) {
        self.all_constraints = constraints;
        self.update_active_constraints();
    }

    
    pub fn update_zoom(&mut self, zoom_distance: f32) {
        let new_level = self.calculate_lod_level(zoom_distance);

        if new_level != self.current_level {
            self.current_level = new_level;
            self.update_active_constraints();
        }
    }

    
    pub fn update_frame_time(&mut self, frame_time_ms: f32) {
        if !self.config.adaptive {
            return;
        }

        self.config.current_frame_time = frame_time_ms;

        
        if frame_time_ms > self.config.target_frame_time * 1.2 {
            self.reduce_lod_level();
        }
        
        else if frame_time_ms < self.config.target_frame_time * 0.8 {
            self.increase_lod_level();
        }
    }

    
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

    
    fn update_active_constraints(&mut self) {
        let priority_threshold = self.config.priority_thresholds[self.current_level as usize];

        self.active_constraints = self.all_constraints
            .iter()
            .filter(|c| self.should_activate_constraint(c, priority_threshold))
            .cloned()
            .collect();
    }

    
    fn should_activate_constraint(&self, constraint: &PhysicsConstraint, priority_threshold: u8) -> bool {
        
        if constraint.user_defined {
            return true;
        }

        
        if constraint.priority > priority_threshold {
            return false;
        }

        
        if matches!(constraint.constraint_type, PhysicsConstraintType::HierarchicalLayer { .. }) {
            return true;
        }

        true
    }

    
    fn reduce_lod_level(&mut self) {
        self.current_level = match self.current_level {
            LODLevel::Close => LODLevel::Near,
            LODLevel::Near => LODLevel::Medium,
            LODLevel::Medium => LODLevel::Far,
            LODLevel::Far => LODLevel::Far,
        };

        self.update_active_constraints();
    }

    
    fn increase_lod_level(&mut self) {
        self.current_level = match self.current_level {
            LODLevel::Far => LODLevel::Medium,
            LODLevel::Medium => LODLevel::Near,
            LODLevel::Near => LODLevel::Close,
            LODLevel::Close => LODLevel::Close,
        };

        self.update_active_constraints();
    }

    
    pub fn get_active_constraints(&self) -> &[PhysicsConstraint] {
        &self.active_constraints
    }

    
    pub fn get_all_constraints(&self) -> &[PhysicsConstraint] {
        &self.all_constraints
    }

    
    pub fn get_current_level(&self) -> LODLevel {
        self.current_level
    }

    
    pub fn get_reduction_percentage(&self) -> f32 {
        if self.all_constraints.is_empty() {
            return 0.0;
        }

        let reduction = 1.0 - (self.active_constraints.len() as f32 / self.all_constraints.len() as f32);
        reduction * 100.0
    }

    
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

/
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
            PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 1), 
            PhysicsConstraint::separation(vec![2, 3], 15.0, 0.6, 3), 
            PhysicsConstraint::clustering(vec![3, 4], 20.0, 0.6, 5), 
            PhysicsConstraint::clustering(vec![4, 5], 25.0, 0.7, 7), 
            PhysicsConstraint::colocation(vec![5, 6], 2.0, 0.9, 10), 
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

        lod.update_zoom(2000.0); 

        let active = lod.get_active_constraints();

        
        assert!(active.len() <= 2);
        assert!(active.iter().all(|c| c.priority <= 3));
    }

    #[test]
    fn test_medium_lod() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(500.0); 

        let active = lod.get_active_constraints();

        
        assert!(active.len() <= 3);
        assert!(active.iter().all(|c| c.priority <= 5));
    }

    #[test]
    fn test_near_lod() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(50.0); 

        let active = lod.get_active_constraints();

        
        assert!(active.len() <= 4);
        assert!(active.iter().all(|c| c.priority <= 7));
    }

    #[test]
    fn test_close_lod_all_active() {
        let mut lod = ConstraintLOD::new();
        let constraints = create_test_constraints();
        lod.set_constraints(constraints.clone());

        lod.update_zoom(5.0); 

        let active = lod.get_active_constraints();

        
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
        lod.update_zoom(2000.0); 

        let active = lod.get_active_constraints();

        
        assert!(active.iter().any(|c| c.user_defined));
    }

    #[test]
    fn test_adaptive_lod_frame_time() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        
        lod.set_lod_level(LODLevel::Close);
        assert_eq!(lod.get_current_level(), LODLevel::Close);

        
        lod.update_frame_time(25.0); 

        
        assert!(lod.get_current_level() < LODLevel::Close);
    }

    #[test]
    fn test_reduction_percentage() {
        let mut lod = ConstraintLOD::new();
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(2000.0); 

        let reduction = lod.get_reduction_percentage();

        
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
        lod.update_zoom(2000.0); 

        let active = lod.get_active_constraints();

        
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
            target_frame_time: 33.33, 
            current_frame_time: 20.0,
        };

        let mut lod = ConstraintLOD::with_config(config);
        lod.set_constraints(create_test_constraints());

        lod.update_zoom(600.0); 
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
