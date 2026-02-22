use serde::{Deserialize, Serialize};
use specta::Type;
use validator::Validate;

use super::{CANONICAL_MAX_FORCE, CANONICAL_MAX_VELOCITY};

fn default_auto_balance_interval() -> u32 {
    500
}

fn default_constraint_ramp_frames() -> u32 {
    60
}

fn default_constraint_max_force_per_node() -> f32 {
    50.0
}

fn default_bounds_size() -> f32 {
    1000.0
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AutoPauseConfig {
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(alias = "equilibrium_velocity_threshold")]
    pub equilibrium_velocity_threshold: f32,
    #[validate(range(min = 1, max = 300))]
    #[serde(alias = "equilibrium_check_frames")]
    pub equilibrium_check_frames: u32,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(alias = "equilibrium_energy_threshold")]
    pub equilibrium_energy_threshold: f32,
    #[serde(alias = "pause_on_equilibrium")]
    pub pause_on_equilibrium: bool,
    #[serde(alias = "resume_on_interaction")]
    pub resume_on_interaction: bool,
}

impl AutoPauseConfig {
    pub fn default() -> Self {
        Self {
            enabled: true,
            equilibrium_velocity_threshold: 0.1,
            equilibrium_check_frames: 30,
            equilibrium_energy_threshold: 0.01,
            pause_on_equilibrium: true,
            resume_on_interaction: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AutoBalanceConfig {
    #[serde(alias = "stability_variance_threshold")]
    pub stability_variance_threshold: f32,
    #[serde(alias = "stability_frame_count")]
    pub stability_frame_count: u32,
    #[serde(alias = "clustering_distance_threshold")]
    pub clustering_distance_threshold: f32,
    #[serde(alias = "clustering_hysteresis_buffer")]
    pub clustering_hysteresis_buffer: f32,
    #[serde(alias = "bouncing_node_percentage")]
    pub bouncing_node_percentage: f32,
    #[serde(alias = "boundary_min_distance")]
    pub boundary_min_distance: f32,
    #[serde(alias = "boundary_max_distance")]
    pub boundary_max_distance: f32,
    #[serde(alias = "extreme_distance_threshold")]
    pub extreme_distance_threshold: f32,
    #[serde(alias = "explosion_distance_threshold")]
    pub explosion_distance_threshold: f32,
    #[serde(alias = "spreading_distance_threshold")]
    pub spreading_distance_threshold: f32,
    #[serde(alias = "spreading_hysteresis_buffer")]
    pub spreading_hysteresis_buffer: f32,
    #[serde(alias = "oscillation_detection_frames")]
    pub oscillation_detection_frames: usize,
    #[serde(alias = "oscillation_change_threshold")]
    pub oscillation_change_threshold: f32,
    #[serde(alias = "min_oscillation_changes")]
    pub min_oscillation_changes: usize,


    #[serde(alias = "parameter_adjustment_rate")]
    pub parameter_adjustment_rate: f32,
    #[serde(alias = "max_adjustment_factor")]
    pub max_adjustment_factor: f32,
    #[serde(alias = "min_adjustment_factor")]
    pub min_adjustment_factor: f32,
    #[serde(alias = "adjustment_cooldown_ms")]
    pub adjustment_cooldown_ms: u64,
    #[serde(alias = "state_change_cooldown_ms")]
    pub state_change_cooldown_ms: u64,
    #[serde(alias = "parameter_dampening_factor")]
    pub parameter_dampening_factor: f32,
    #[serde(alias = "hysteresis_delay_frames")]
    pub hysteresis_delay_frames: u32,


    #[serde(alias = "grid_cell_size_min")]
    pub grid_cell_size_min: f32,
    #[serde(alias = "grid_cell_size_max")]
    pub grid_cell_size_max: f32,
    #[serde(alias = "repulsion_cutoff_min")]
    pub repulsion_cutoff_min: f32,
    #[serde(alias = "repulsion_cutoff_max")]
    pub repulsion_cutoff_max: f32,
    #[serde(alias = "repulsion_softening_min")]
    pub repulsion_softening_min: f32,
    #[serde(alias = "repulsion_softening_max")]
    pub repulsion_softening_max: f32,
    #[serde(alias = "center_gravity_min")]
    pub center_gravity_min: f32,
    #[serde(alias = "center_gravity_max")]
    pub center_gravity_max: f32,


    #[serde(alias = "spatial_hash_efficiency_threshold")]
    pub spatial_hash_efficiency_threshold: f32,
    #[serde(alias = "cluster_density_threshold")]
    pub cluster_density_threshold: f32,
    #[serde(alias = "numerical_instability_threshold")]
    pub numerical_instability_threshold: f32,
}

impl AutoBalanceConfig {
    pub fn default() -> Self {
        Self {
            stability_variance_threshold: 100.0,
            stability_frame_count: 180,
            clustering_distance_threshold: 20.0,
            clustering_hysteresis_buffer: 5.0,
            bouncing_node_percentage: 0.33,
            boundary_min_distance: 90.0,
            boundary_max_distance: 110.0,
            extreme_distance_threshold: 1000.0,
            explosion_distance_threshold: 10000.0,
            spreading_distance_threshold: 500.0,
            spreading_hysteresis_buffer: 50.0,
            oscillation_detection_frames: 20,
            oscillation_change_threshold: 10.0,
            min_oscillation_changes: 8,


            parameter_adjustment_rate: 0.1,
            max_adjustment_factor: 0.2,
            min_adjustment_factor: -0.2,
            adjustment_cooldown_ms: 2000,
            state_change_cooldown_ms: 1000,
            parameter_dampening_factor: 0.05,
            hysteresis_delay_frames: 30,


            grid_cell_size_min: 1.0,
            grid_cell_size_max: 50.0,
            repulsion_cutoff_min: 5.0,
            repulsion_cutoff_max: 200.0,
            repulsion_softening_min: 1e-6,
            repulsion_softening_max: 1.0,
            center_gravity_min: 0.0,
            center_gravity_max: 0.1,


            spatial_hash_efficiency_threshold: 0.3,
            cluster_density_threshold: 50.0,
            numerical_instability_threshold: 1e-3,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsSettings {
    #[serde(default, alias = "auto_balance")]
    pub auto_balance: bool,
    #[serde(
        default = "default_auto_balance_interval",
        alias = "auto_balance_interval_ms"
    )]
    pub auto_balance_interval_ms: u32,
    #[serde(default, alias = "auto_balance_config")]
    #[validate(nested)]
    pub auto_balance_config: AutoBalanceConfig,
    #[serde(default, alias = "auto_pause")]
    #[validate(nested)]
    pub auto_pause: AutoPauseConfig,
    #[serde(default = "default_bounds_size", alias = "bounds_size")]
    pub bounds_size: f32,
    #[serde(alias = "separation_radius")]
    pub separation_radius: f32,
    #[serde(alias = "damping")]
    pub damping: f32,
    #[serde(alias = "enable_bounds")]
    pub enable_bounds: bool,
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[serde(alias = "iterations")]
    pub iterations: u32,
    #[serde(alias = "max_velocity")]
    pub max_velocity: f32,
    #[serde(alias = "max_force")]
    pub max_force: f32,
    #[serde(alias = "repel_k")]
    pub repel_k: f32,
    #[serde(alias = "spring_k")]
    pub spring_k: f32,
    #[deprecated(note = "Not wired to physics engine")]
    #[serde(alias = "mass_scale")]
    pub mass_scale: f32,
    #[serde(alias = "boundary_damping")]
    pub boundary_damping: f32,
    #[serde(alias = "update_threshold")]
    pub update_threshold: f32,
    #[serde(alias = "dt")]
    pub dt: f32,
    #[serde(alias = "temperature")]
    pub temperature: f32,
    #[serde(alias = "gravity")]
    pub gravity: f32,

    #[deprecated(note = "Not wired to physics engine")]
    #[serde(alias = "stress_weight")]
    pub stress_weight: f32,
    #[deprecated(note = "Not wired to physics engine")]
    #[serde(alias = "stress_alpha")]
    pub stress_alpha: f32,
    #[deprecated(note = "Not wired to physics engine")]
    #[serde(alias = "boundary_limit")]
    pub boundary_limit: f32,
    #[serde(alias = "alignment_strength")]
    pub alignment_strength: f32,
    #[serde(alias = "cluster_strength")]
    pub cluster_strength: f32,
    #[serde(alias = "compute_mode")]
    pub compute_mode: i32,


    #[serde(alias = "rest_length")]
    pub rest_length: f32,
    #[serde(alias = "repulsion_cutoff")]
    pub repulsion_cutoff: f32,
    #[serde(alias = "repulsion_softening_epsilon")]
    pub repulsion_softening_epsilon: f32,
    #[serde(alias = "center_gravity_k")]
    pub center_gravity_k: f32,
    #[serde(alias = "grid_cell_size")]
    pub grid_cell_size: f32,
    #[serde(alias = "warmup_iterations")]
    pub warmup_iterations: u32,
    #[serde(alias = "cooling_rate")]
    pub cooling_rate: f32,
    #[serde(alias = "boundary_extreme_multiplier")]
    pub boundary_extreme_multiplier: f32,
    #[serde(alias = "boundary_extreme_force_multiplier")]
    pub boundary_extreme_force_multiplier: f32,
    #[serde(alias = "boundary_velocity_damping")]
    pub boundary_velocity_damping: f32,

    #[serde(alias = "min_distance")]
    pub min_distance: f32,
    #[serde(alias = "max_repulsion_dist")]
    pub max_repulsion_dist: f32,
    #[serde(alias = "boundary_margin")]
    pub boundary_margin: f32,
    #[serde(alias = "boundary_force_strength")]
    pub boundary_force_strength: f32,
    #[deprecated(note = "Not wired to physics engine")]
    #[serde(alias = "warmup_curve")]
    pub warmup_curve: String,
    #[deprecated(note = "Not wired to physics engine")]
    #[serde(alias = "zero_velocity_iterations")]
    pub zero_velocity_iterations: u32,


    #[serde(
        alias = "constraint_ramp_frames",
        default = "default_constraint_ramp_frames"
    )]
    pub constraint_ramp_frames: u32,
    #[serde(
        alias = "constraint_max_force_per_node",
        default = "default_constraint_max_force_per_node"
    )]
    pub constraint_max_force_per_node: f32,


    #[serde(alias = "clustering_algorithm")]
    pub clustering_algorithm: String,
    #[serde(alias = "cluster_count")]
    pub cluster_count: u32,
    #[serde(alias = "clustering_resolution")]
    pub clustering_resolution: f32,
    #[serde(alias = "clustering_iterations")]
    pub clustering_iterations: u32,
}

#[allow(deprecated)]
impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            auto_balance: false,
            auto_balance_interval_ms: 500,
            auto_balance_config: AutoBalanceConfig::default(),
            auto_pause: AutoPauseConfig::default(),
            bounds_size: 500.0,
            separation_radius: 2.0,
            damping: 0.95,
            enable_bounds: true,
            enabled: true,
            iterations: 100,
            max_velocity: CANONICAL_MAX_VELOCITY,
            max_force: CANONICAL_MAX_FORCE,
            repel_k: 50.0,
            spring_k: 0.005,
            mass_scale: 1.0,
            boundary_damping: 0.95,
            update_threshold: 0.01,
            dt: 0.016,
            temperature: 0.01,
            gravity: 0.0001,
            stress_weight: 0.1,
            stress_alpha: 0.1,
            boundary_limit: 490.0,
            alignment_strength: 0.0,
            cluster_strength: 0.0,
            compute_mode: 0,

            rest_length: 50.0,
            repulsion_cutoff: 50.0,
            repulsion_softening_epsilon: 0.0001,
            center_gravity_k: 0.0,
            grid_cell_size: 50.0,
            warmup_iterations: 100,
            cooling_rate: 0.001,
            boundary_extreme_multiplier: 2.0,
            boundary_extreme_force_multiplier: 10.0,
            boundary_velocity_damping: 0.5,

            min_distance: 0.15,
            max_repulsion_dist: 50.0,
            boundary_margin: 0.85,
            boundary_force_strength: 2.0,
            warmup_curve: "quadratic".to_string(),
            zero_velocity_iterations: 5,

            constraint_ramp_frames: default_constraint_ramp_frames(),
            constraint_max_force_per_node: default_constraint_max_force_per_node(),

            clustering_algorithm: "none".to_string(),
            cluster_count: 5,
            clustering_resolution: 1.0,
            clustering_iterations: 30,
        }
    }
}

// Constraint system structures
// Note: ConstraintData has been moved to models/constraints.rs for GPU compatibility
// The old simple structure has been replaced with a GPU-optimized version

// Legacy constraint system for web API compatibility
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct LegacyConstraintData {
    #[serde(alias = "constraint_type")]
    pub constraint_type: i32,
    #[serde(alias = "strength")]
    pub strength: f32,
    #[serde(alias = "param1")]
    pub param1: f32,
    #[serde(alias = "param2")]
    pub param2: f32,
    #[serde(alias = "node_mask")]
    pub node_mask: i32,
    #[serde(alias = "enabled")]
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintSystem {
    #[serde(alias = "separation")]
    pub separation: LegacyConstraintData,
    #[serde(alias = "boundary")]
    pub boundary: LegacyConstraintData,
    #[serde(alias = "alignment")]
    pub alignment: LegacyConstraintData,
    #[serde(alias = "cluster")]
    pub cluster: LegacyConstraintData,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct ClusteringConfiguration {
    #[serde(alias = "algorithm")]
    pub algorithm: String,
    #[serde(alias = "num_clusters")]
    pub num_clusters: u32,
    #[serde(alias = "resolution")]
    pub resolution: f32,
    #[serde(alias = "iterations")]
    pub iterations: u32,
    #[serde(alias = "export_assignments")]
    pub export_assignments: bool,
    #[serde(alias = "auto_update")]
    pub auto_update: bool,
}

// Helper struct for physics updates
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsUpdate {
    #[serde(alias = "damping")]
    pub damping: Option<f32>,
    #[serde(alias = "spring_k")]
    pub spring_k: Option<f32>,
    #[serde(alias = "repel_k")]
    pub repel_k: Option<f32>,
    #[serde(alias = "iterations")]
    pub iterations: Option<u32>,
    #[serde(alias = "enabled")]
    pub enabled: Option<bool>,
    #[serde(alias = "bounds_size")]
    pub bounds_size: Option<f32>,
    #[serde(alias = "enable_bounds")]
    pub enable_bounds: Option<bool>,
    #[serde(alias = "max_velocity")]
    pub max_velocity: Option<f32>,
    #[serde(alias = "max_force")]
    pub max_force: Option<f32>,
    #[serde(alias = "separation_radius")]
    pub separation_radius: Option<f32>,
    #[serde(alias = "mass_scale")]
    pub mass_scale: Option<f32>,
    #[serde(alias = "boundary_damping")]
    pub boundary_damping: Option<f32>,
    #[serde(alias = "dt")]
    pub dt: Option<f32>,
    #[serde(alias = "temperature")]
    pub temperature: Option<f32>,
    #[serde(alias = "gravity")]
    pub gravity: Option<f32>,
    #[serde(alias = "update_threshold")]
    pub update_threshold: Option<f32>,

    #[serde(alias = "stress_weight")]
    pub stress_weight: Option<f32>,
    #[serde(alias = "stress_alpha")]
    pub stress_alpha: Option<f32>,
    #[serde(alias = "boundary_limit")]
    pub boundary_limit: Option<f32>,
    #[serde(alias = "alignment_strength")]
    pub alignment_strength: Option<f32>,
    #[serde(alias = "cluster_strength")]
    pub cluster_strength: Option<f32>,
    #[serde(alias = "compute_mode")]
    pub compute_mode: Option<i32>,

    #[serde(alias = "min_distance")]
    pub min_distance: Option<f32>,
    #[serde(alias = "max_repulsion_dist")]
    pub max_repulsion_dist: Option<f32>,
    #[serde(alias = "boundary_margin")]
    pub boundary_margin: Option<f32>,
    #[serde(alias = "boundary_force_strength")]
    pub boundary_force_strength: Option<f32>,
    #[serde(alias = "warmup_iterations")]
    pub warmup_iterations: Option<u32>,
    #[serde(alias = "warmup_curve")]
    pub warmup_curve: Option<String>,
    #[serde(alias = "zero_velocity_iterations")]
    pub zero_velocity_iterations: Option<u32>,
    #[serde(alias = "cooling_rate")]
    pub cooling_rate: Option<f32>,

    #[serde(alias = "clustering_algorithm")]
    pub clustering_algorithm: Option<String>,
    #[serde(alias = "cluster_count")]
    pub cluster_count: Option<u32>,
    #[serde(alias = "clustering_resolution")]
    pub clustering_resolution: Option<f32>,
    #[serde(alias = "clustering_iterations")]
    pub clustering_iterations: Option<u32>,

    #[serde(alias = "repulsion_softening_epsilon")]
    pub repulsion_softening_epsilon: Option<f32>,
    #[serde(alias = "center_gravity_k")]
    pub center_gravity_k: Option<f32>,
    #[serde(alias = "grid_cell_size")]
    pub grid_cell_size: Option<f32>,
    #[serde(alias = "rest_length")]
    pub rest_length: Option<f32>,
}
