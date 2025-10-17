// Internal Developer Configuration
// This file contains server-side only settings that are not exposed to clients
// These settings control internal behavior, performance tuning, and debug features

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock, OnceLock};
use log::{info, warn, error};

use crate::services::database_service::{DatabaseService, SettingValue};

static DEV_CONFIG: OnceLock<Arc<RwLock<DevConfig>>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevConfig {
    pub physics: PhysicsInternals,
    pub cuda: CudaInternals,
    pub network: NetworkInternals,
    pub rendering: RenderingInternals,
    pub performance: PerformanceInternals,
    pub debug: DebugInternals,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsInternals {
    // Force calculation parameters
    pub force_epsilon: f32,
    pub spring_length_multiplier: f32,
    pub spring_length_max: f32,
    pub spring_force_clamp_factor: f32,

    // CUDA kernel parameters
    pub rest_length: f32,
    pub repulsion_cutoff: f32,
    pub repulsion_softening_epsilon: f32,
    pub center_gravity_k: f32,
    pub grid_cell_size: f32,
    pub warmup_iterations: u32,
    pub cooling_rate: f32,

    // GPU kernel-specific physics parameters
    pub max_force: f32,
    pub max_velocity: f32,
    pub world_bounds_min: f32,
    pub world_bounds_max: f32,
    pub cell_size_lod: f32,
    pub k_neighbors_max: u32,
    pub anomaly_detection_radius: f32,
    pub learning_rate_default: f32,
    pub min_velocity_threshold: f32,
    pub stability_threshold: f32,

    // Additional kernel constants for fine-tuning
    pub norm_delta_cap: f32,
    pub position_constraint_attraction: f32,
    pub lof_score_min: f32,
    pub lof_score_max: f32,
    pub weight_precision_multiplier: f32,

    // Boundary behavior
    pub boundary_extreme_multiplier: f32,
    pub boundary_extreme_force_multiplier: f32,
    pub boundary_velocity_damping: f32,

    // Node distribution
    pub golden_ratio: f32,
    pub initial_radius_min: f32,
    pub initial_radius_range: f32,

    // Graph-based scaling
    pub cross_graph_repulsion_scale: f32,
    pub cross_graph_spring_scale: f32,

    // Clustering
    pub cluster_repulsion_scale: f32,
    pub importance_scale_factor: f32,

    // Distance thresholds
    pub repulsion_distance_squared_min: f32,
    pub stress_majorization_epsilon: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaInternals {
    // Warmup parameters
    pub warmup_iterations_default: u32,
    pub warmup_damping_start: f32,
    pub warmup_damping_end: f32,
    pub warmup_temperature_scale: f32,
    pub warmup_cooling_iterations: u32,

    // GPU safety
    pub max_kernel_time_ms: u32,
    pub max_gpu_failures: u32,
    pub debug_output_throttle: u32,
    pub debug_node_count: u32,

    // Memory limits
    pub max_nodes: u32,
    pub max_edges: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInternals {
    // Connection pooling
    pub pool_max_idle_per_host: usize,
    pub pool_idle_timeout_secs: u64,
    pub pool_connect_timeout_secs: u64,

    // Circuit breaker
    pub circuit_failure_threshold: u32,
    pub circuit_recovery_timeout_secs: u64,
    pub circuit_half_open_max_requests: u32,

    // Retry logic
    pub max_retry_attempts: u32,
    pub retry_base_delay_ms: u64,
    pub retry_max_delay_ms: u64,
    pub retry_exponential_base: f32,

    // WebSocket internals
    pub ws_ping_interval_secs: u64,
    pub ws_pong_timeout_secs: u64,
    pub ws_frame_size: usize,
    pub ws_max_pending_messages: usize,

    // Rate limiting internals
    pub rate_limit_burst_size: u32,
    pub rate_limit_refill_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingInternals {
    // Agent colors (hex strings)
    pub agent_colors: AgentColors,

    // Size calculations
    pub agent_base_size: f32,
    pub agent_size_per_task: f32,
    pub agent_max_size: f32,
    pub node_base_radius: f32,

    // Animation speeds
    pub pulse_speed: f32,
    pub rotate_speed: f32,
    pub glow_speed: f32,
    pub wave_speed: f32,

    // Quality thresholds
    pub lod_distance_high: f32,
    pub lod_distance_medium: f32,
    pub lod_distance_low: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentColors {
    pub coordinator: String,
    pub coder: String,
    pub architect: String,
    pub analyst: String,
    pub tester: String,
    pub researcher: String,
    pub reviewer: String,
    pub optimizer: String,
    pub documenter: String,
    pub default: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInternals {
    // Batching
    pub batch_size_nodes: usize,
    pub batch_size_edges: usize,
    pub batch_timeout_ms: u64,

    // Caching
    pub cache_ttl_secs: u64,
    pub cache_max_entries: usize,
    pub cache_eviction_percentage: f32,

    // Threading
    pub worker_threads: usize,
    pub blocking_threads: usize,
    pub stack_size_mb: usize,

    // Memory management
    pub gc_interval_secs: u64,
    pub memory_warning_threshold_mb: usize,
    pub memory_critical_threshold_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInternals {
    pub enable_cuda_debug: bool,
    pub enable_physics_debug: bool,
    pub enable_network_debug: bool,
    pub enable_memory_tracking: bool,
    pub enable_performance_tracking: bool,

    pub log_slow_operations_ms: u64,
    pub log_memory_usage_interval_secs: u64,
    pub profile_sample_rate: f32,
}

impl Default for DevConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsInternals {
                force_epsilon: 1e-8,
                spring_length_multiplier: 5.0,
                spring_length_max: 10.0,
                spring_force_clamp_factor: 0.5,

                // CUDA kernel parameters
                rest_length: 50.0,
                repulsion_cutoff: 50.0,
                repulsion_softening_epsilon: 0.0001,
                center_gravity_k: 0.005,
                grid_cell_size: 50.0,
                warmup_iterations: 100,
                cooling_rate: 0.001,

                // GPU kernel-specific physics parameters
                max_force: 15.0,
                max_velocity: 50.0,
                world_bounds_min: -1000.0,
                world_bounds_max: 1000.0,
                cell_size_lod: 100.0,
                k_neighbors_max: 32,
                anomaly_detection_radius: 150.0,
                learning_rate_default: 0.1,
                min_velocity_threshold: 0.01,
                stability_threshold: 1e-6,

                // Additional kernel constants for fine-tuning
                norm_delta_cap: 1000.0,
                position_constraint_attraction: 0.1,
                lof_score_min: 0.1,
                lof_score_max: 10.0,
                weight_precision_multiplier: 1000.0,

                boundary_extreme_multiplier: 2.0,
                boundary_extreme_force_multiplier: 10.0,
                boundary_velocity_damping: 0.5,
                golden_ratio: 1.618033988749895,
                initial_radius_min: 100.0,
                initial_radius_range: 300.0,
                cross_graph_repulsion_scale: 0.3,
                cross_graph_spring_scale: 0.5,
                cluster_repulsion_scale: 0.5,
                importance_scale_factor: 1.0,
                repulsion_distance_squared_min: 100.0,
                stress_majorization_epsilon: 0.001,
            },
            cuda: CudaInternals {
                warmup_iterations_default: 200,
                warmup_damping_start: 0.98,
                warmup_damping_end: 0.85,
                warmup_temperature_scale: 0.0001,
                warmup_cooling_iterations: 5,
                max_kernel_time_ms: 5000,
                max_gpu_failures: 5,
                debug_output_throttle: 60,
                debug_node_count: 3,
                max_nodes: 1_000_000,
                max_edges: 10_000_000,
            },
            network: NetworkInternals {
                pool_max_idle_per_host: 32,
                pool_idle_timeout_secs: 90,
                pool_connect_timeout_secs: 10,
                circuit_failure_threshold: 5,
                circuit_recovery_timeout_secs: 30,
                circuit_half_open_max_requests: 3,
                max_retry_attempts: 3,
                retry_base_delay_ms: 100,
                retry_max_delay_ms: 30000,
                retry_exponential_base: 2.0,
                ws_ping_interval_secs: 30,
                ws_pong_timeout_secs: 10,
                ws_frame_size: 65536,
                ws_max_pending_messages: 100,
                rate_limit_burst_size: 10,
                rate_limit_refill_rate: 1.0,
            },
            rendering: RenderingInternals {
                agent_colors: AgentColors {
                    coordinator: "#00FFFF".to_string(),
                    coder: "#00FF00".to_string(),
                    architect: "#FFA500".to_string(),
                    analyst: "#9370DB".to_string(),
                    tester: "#FF6347".to_string(),
                    researcher: "#FFD700".to_string(),
                    reviewer: "#4169E1".to_string(),
                    optimizer: "#7FFFD4".to_string(),
                    documenter: "#FF69B4".to_string(),
                    default: "#CCCCCC".to_string(),
                },
                agent_base_size: 1.0,
                agent_size_per_task: 0.2,
                agent_max_size: 2.0,
                node_base_radius: 15.0,
                pulse_speed: 2.0,
                rotate_speed: 1.0,
                glow_speed: 0.5,
                wave_speed: 0.5,
                lod_distance_high: 100.0,
                lod_distance_medium: 500.0,
                lod_distance_low: 1000.0,
            },
            performance: PerformanceInternals {
                batch_size_nodes: 1000,
                batch_size_edges: 5000,
                batch_timeout_ms: 100,
                cache_ttl_secs: 300,
                cache_max_entries: 10000,
                cache_eviction_percentage: 0.2,
                worker_threads: 4,
                blocking_threads: 512,
                stack_size_mb: 2,
                gc_interval_secs: 60,
                memory_warning_threshold_mb: 1024,
                memory_critical_threshold_mb: 2048,
            },
            debug: DebugInternals {
                enable_cuda_debug: false,
                enable_physics_debug: false,
                enable_network_debug: false,
                enable_memory_tracking: false,
                enable_performance_tracking: false,
                log_slow_operations_ms: 100,
                log_memory_usage_interval_secs: 60,
                profile_sample_rate: 0.01,
            },
        }
    }
}

impl DevConfig {
    /// Initialize DevConfig from database
    pub fn initialize(db_service: Arc<DatabaseService>) -> Result<(), Box<dyn std::error::Error>> {
        let config = Self::from_database(db_service)?;

        DEV_CONFIG.set(Arc::new(RwLock::new(config)))
            .map_err(|_| "DevConfig already initialized")?;

        info!("DevConfig initialized from database");
        Ok(())
    }

    /// Load configuration from database
    pub fn from_database(db_service: Arc<DatabaseService>) -> Result<Self, Box<dyn std::error::Error>> {
        // Check for deprecated file
        if std::path::Path::new("data/dev_config.toml").exists() {
            warn!("DEPRECATED: data/dev_config.toml file found. Configuration is now loaded from database.");
            warn!("Please remove or rename this file to data/dev_config.toml.deprecated");
        }

        let mut config = Self::default();

        // Load physics parameters
        config.physics.force_epsilon = Self::get_f32(&db_service, "dev.physics.force_epsilon")
            .unwrap_or(config.physics.force_epsilon);
        config.physics.spring_length_multiplier = Self::get_f32(&db_service, "dev.physics.spring_length_multiplier")
            .unwrap_or(config.physics.spring_length_multiplier);
        config.physics.spring_length_max = Self::get_f32(&db_service, "dev.physics.spring_length_max")
            .unwrap_or(config.physics.spring_length_max);
        config.physics.spring_force_clamp_factor = Self::get_f32(&db_service, "dev.physics.spring_force_clamp_factor")
            .unwrap_or(config.physics.spring_force_clamp_factor);
        config.physics.rest_length = Self::get_f32(&db_service, "dev.physics.rest_length")
            .unwrap_or(config.physics.rest_length);
        config.physics.repulsion_cutoff = Self::get_f32(&db_service, "dev.physics.repulsion_cutoff")
            .unwrap_or(config.physics.repulsion_cutoff);
        config.physics.repulsion_softening_epsilon = Self::get_f32(&db_service, "dev.physics.repulsion_softening_epsilon")
            .unwrap_or(config.physics.repulsion_softening_epsilon);
        config.physics.center_gravity_k = Self::get_f32(&db_service, "dev.physics.center_gravity_k")
            .unwrap_or(config.physics.center_gravity_k);
        config.physics.grid_cell_size = Self::get_f32(&db_service, "dev.physics.grid_cell_size")
            .unwrap_or(config.physics.grid_cell_size);
        config.physics.warmup_iterations = Self::get_u32(&db_service, "dev.physics.warmup_iterations")
            .unwrap_or(config.physics.warmup_iterations);
        config.physics.cooling_rate = Self::get_f32(&db_service, "dev.physics.cooling_rate")
            .unwrap_or(config.physics.cooling_rate);
        config.physics.max_force = Self::get_f32(&db_service, "dev.physics.max_force")
            .unwrap_or(config.physics.max_force);
        config.physics.max_velocity = Self::get_f32(&db_service, "dev.physics.max_velocity")
            .unwrap_or(config.physics.max_velocity);
        config.physics.world_bounds_min = Self::get_f32(&db_service, "dev.physics.world_bounds_min")
            .unwrap_or(config.physics.world_bounds_min);
        config.physics.world_bounds_max = Self::get_f32(&db_service, "dev.physics.world_bounds_max")
            .unwrap_or(config.physics.world_bounds_max);
        config.physics.cell_size_lod = Self::get_f32(&db_service, "dev.physics.cell_size_lod")
            .unwrap_or(config.physics.cell_size_lod);
        config.physics.k_neighbors_max = Self::get_u32(&db_service, "dev.physics.k_neighbors_max")
            .unwrap_or(config.physics.k_neighbors_max);
        config.physics.anomaly_detection_radius = Self::get_f32(&db_service, "dev.physics.anomaly_detection_radius")
            .unwrap_or(config.physics.anomaly_detection_radius);
        config.physics.learning_rate_default = Self::get_f32(&db_service, "dev.physics.learning_rate_default")
            .unwrap_or(config.physics.learning_rate_default);
        config.physics.min_velocity_threshold = Self::get_f32(&db_service, "dev.physics.min_velocity_threshold")
            .unwrap_or(config.physics.min_velocity_threshold);
        config.physics.stability_threshold = Self::get_f32(&db_service, "dev.physics.stability_threshold")
            .unwrap_or(config.physics.stability_threshold);
        config.physics.norm_delta_cap = Self::get_f32(&db_service, "dev.physics.norm_delta_cap")
            .unwrap_or(config.physics.norm_delta_cap);
        config.physics.position_constraint_attraction = Self::get_f32(&db_service, "dev.physics.position_constraint_attraction")
            .unwrap_or(config.physics.position_constraint_attraction);
        config.physics.lof_score_min = Self::get_f32(&db_service, "dev.physics.lof_score_min")
            .unwrap_or(config.physics.lof_score_min);
        config.physics.lof_score_max = Self::get_f32(&db_service, "dev.physics.lof_score_max")
            .unwrap_or(config.physics.lof_score_max);
        config.physics.weight_precision_multiplier = Self::get_f32(&db_service, "dev.physics.weight_precision_multiplier")
            .unwrap_or(config.physics.weight_precision_multiplier);
        config.physics.boundary_extreme_multiplier = Self::get_f32(&db_service, "dev.physics.boundary_extreme_multiplier")
            .unwrap_or(config.physics.boundary_extreme_multiplier);
        config.physics.boundary_extreme_force_multiplier = Self::get_f32(&db_service, "dev.physics.boundary_extreme_force_multiplier")
            .unwrap_or(config.physics.boundary_extreme_force_multiplier);
        config.physics.boundary_velocity_damping = Self::get_f32(&db_service, "dev.physics.boundary_velocity_damping")
            .unwrap_or(config.physics.boundary_velocity_damping);
        config.physics.golden_ratio = Self::get_f32(&db_service, "dev.physics.golden_ratio")
            .unwrap_or(config.physics.golden_ratio);
        config.physics.initial_radius_min = Self::get_f32(&db_service, "dev.physics.initial_radius_min")
            .unwrap_or(config.physics.initial_radius_min);
        config.physics.initial_radius_range = Self::get_f32(&db_service, "dev.physics.initial_radius_range")
            .unwrap_or(config.physics.initial_radius_range);
        config.physics.cross_graph_repulsion_scale = Self::get_f32(&db_service, "dev.physics.cross_graph_repulsion_scale")
            .unwrap_or(config.physics.cross_graph_repulsion_scale);
        config.physics.cross_graph_spring_scale = Self::get_f32(&db_service, "dev.physics.cross_graph_spring_scale")
            .unwrap_or(config.physics.cross_graph_spring_scale);
        config.physics.cluster_repulsion_scale = Self::get_f32(&db_service, "dev.physics.cluster_repulsion_scale")
            .unwrap_or(config.physics.cluster_repulsion_scale);
        config.physics.importance_scale_factor = Self::get_f32(&db_service, "dev.physics.importance_scale_factor")
            .unwrap_or(config.physics.importance_scale_factor);
        config.physics.repulsion_distance_squared_min = Self::get_f32(&db_service, "dev.physics.repulsion_distance_squared_min")
            .unwrap_or(config.physics.repulsion_distance_squared_min);
        config.physics.stress_majorization_epsilon = Self::get_f32(&db_service, "dev.physics.stress_majorization_epsilon")
            .unwrap_or(config.physics.stress_majorization_epsilon);

        // Load CUDA parameters
        config.cuda.warmup_iterations_default = Self::get_u32(&db_service, "dev.cuda.warmup_iterations_default")
            .unwrap_or(config.cuda.warmup_iterations_default);
        config.cuda.warmup_damping_start = Self::get_f32(&db_service, "dev.cuda.warmup_damping_start")
            .unwrap_or(config.cuda.warmup_damping_start);
        config.cuda.warmup_damping_end = Self::get_f32(&db_service, "dev.cuda.warmup_damping_end")
            .unwrap_or(config.cuda.warmup_damping_end);
        config.cuda.warmup_temperature_scale = Self::get_f32(&db_service, "dev.cuda.warmup_temperature_scale")
            .unwrap_or(config.cuda.warmup_temperature_scale);
        config.cuda.warmup_cooling_iterations = Self::get_u32(&db_service, "dev.cuda.warmup_cooling_iterations")
            .unwrap_or(config.cuda.warmup_cooling_iterations);
        config.cuda.max_kernel_time_ms = Self::get_u32(&db_service, "dev.cuda.max_kernel_time_ms")
            .unwrap_or(config.cuda.max_kernel_time_ms);
        config.cuda.max_gpu_failures = Self::get_u32(&db_service, "dev.cuda.max_gpu_failures")
            .unwrap_or(config.cuda.max_gpu_failures);
        config.cuda.debug_output_throttle = Self::get_u32(&db_service, "dev.cuda.debug_output_throttle")
            .unwrap_or(config.cuda.debug_output_throttle);
        config.cuda.debug_node_count = Self::get_u32(&db_service, "dev.cuda.debug_node_count")
            .unwrap_or(config.cuda.debug_node_count);
        config.cuda.max_nodes = Self::get_u32(&db_service, "dev.cuda.max_nodes")
            .unwrap_or(config.cuda.max_nodes);
        config.cuda.max_edges = Self::get_u32(&db_service, "dev.cuda.max_edges")
            .unwrap_or(config.cuda.max_edges);

        // Load network parameters
        config.network.pool_max_idle_per_host = Self::get_usize(&db_service, "dev.network.pool_max_idle_per_host")
            .unwrap_or(config.network.pool_max_idle_per_host);
        config.network.pool_idle_timeout_secs = Self::get_u64(&db_service, "dev.network.pool_idle_timeout_secs")
            .unwrap_or(config.network.pool_idle_timeout_secs);
        config.network.pool_connect_timeout_secs = Self::get_u64(&db_service, "dev.network.pool_connect_timeout_secs")
            .unwrap_or(config.network.pool_connect_timeout_secs);
        config.network.circuit_failure_threshold = Self::get_u32(&db_service, "dev.network.circuit_failure_threshold")
            .unwrap_or(config.network.circuit_failure_threshold);
        config.network.circuit_recovery_timeout_secs = Self::get_u64(&db_service, "dev.network.circuit_recovery_timeout_secs")
            .unwrap_or(config.network.circuit_recovery_timeout_secs);
        config.network.circuit_half_open_max_requests = Self::get_u32(&db_service, "dev.network.circuit_half_open_max_requests")
            .unwrap_or(config.network.circuit_half_open_max_requests);
        config.network.max_retry_attempts = Self::get_u32(&db_service, "dev.network.max_retry_attempts")
            .unwrap_or(config.network.max_retry_attempts);
        config.network.retry_base_delay_ms = Self::get_u64(&db_service, "dev.network.retry_base_delay_ms")
            .unwrap_or(config.network.retry_base_delay_ms);
        config.network.retry_max_delay_ms = Self::get_u64(&db_service, "dev.network.retry_max_delay_ms")
            .unwrap_or(config.network.retry_max_delay_ms);
        config.network.retry_exponential_base = Self::get_f32(&db_service, "dev.network.retry_exponential_base")
            .unwrap_or(config.network.retry_exponential_base);
        config.network.ws_ping_interval_secs = Self::get_u64(&db_service, "dev.network.ws_ping_interval_secs")
            .unwrap_or(config.network.ws_ping_interval_secs);
        config.network.ws_pong_timeout_secs = Self::get_u64(&db_service, "dev.network.ws_pong_timeout_secs")
            .unwrap_or(config.network.ws_pong_timeout_secs);
        config.network.ws_frame_size = Self::get_usize(&db_service, "dev.network.ws_frame_size")
            .unwrap_or(config.network.ws_frame_size);
        config.network.ws_max_pending_messages = Self::get_usize(&db_service, "dev.network.ws_max_pending_messages")
            .unwrap_or(config.network.ws_max_pending_messages);
        config.network.rate_limit_burst_size = Self::get_u32(&db_service, "dev.network.rate_limit_burst_size")
            .unwrap_or(config.network.rate_limit_burst_size);
        config.network.rate_limit_refill_rate = Self::get_f32(&db_service, "dev.network.rate_limit_refill_rate")
            .unwrap_or(config.network.rate_limit_refill_rate);

        // Load rendering parameters
        config.rendering.agent_base_size = Self::get_f32(&db_service, "dev.rendering.agent_base_size")
            .unwrap_or(config.rendering.agent_base_size);
        config.rendering.agent_size_per_task = Self::get_f32(&db_service, "dev.rendering.agent_size_per_task")
            .unwrap_or(config.rendering.agent_size_per_task);
        config.rendering.agent_max_size = Self::get_f32(&db_service, "dev.rendering.agent_max_size")
            .unwrap_or(config.rendering.agent_max_size);
        config.rendering.node_base_radius = Self::get_f32(&db_service, "dev.rendering.node_base_radius")
            .unwrap_or(config.rendering.node_base_radius);
        config.rendering.pulse_speed = Self::get_f32(&db_service, "dev.rendering.pulse_speed")
            .unwrap_or(config.rendering.pulse_speed);
        config.rendering.rotate_speed = Self::get_f32(&db_service, "dev.rendering.rotate_speed")
            .unwrap_or(config.rendering.rotate_speed);
        config.rendering.glow_speed = Self::get_f32(&db_service, "dev.rendering.glow_speed")
            .unwrap_or(config.rendering.glow_speed);
        config.rendering.wave_speed = Self::get_f32(&db_service, "dev.rendering.wave_speed")
            .unwrap_or(config.rendering.wave_speed);
        config.rendering.lod_distance_high = Self::get_f32(&db_service, "dev.rendering.lod_distance_high")
            .unwrap_or(config.rendering.lod_distance_high);
        config.rendering.lod_distance_medium = Self::get_f32(&db_service, "dev.rendering.lod_distance_medium")
            .unwrap_or(config.rendering.lod_distance_medium);
        config.rendering.lod_distance_low = Self::get_f32(&db_service, "dev.rendering.lod_distance_low")
            .unwrap_or(config.rendering.lod_distance_low);

        // Load agent colors
        config.rendering.agent_colors.coordinator = Self::get_string(&db_service, "dev.rendering.agent_colors.coordinator")
            .unwrap_or(config.rendering.agent_colors.coordinator);
        config.rendering.agent_colors.coder = Self::get_string(&db_service, "dev.rendering.agent_colors.coder")
            .unwrap_or(config.rendering.agent_colors.coder);
        config.rendering.agent_colors.architect = Self::get_string(&db_service, "dev.rendering.agent_colors.architect")
            .unwrap_or(config.rendering.agent_colors.architect);
        config.rendering.agent_colors.analyst = Self::get_string(&db_service, "dev.rendering.agent_colors.analyst")
            .unwrap_or(config.rendering.agent_colors.analyst);
        config.rendering.agent_colors.tester = Self::get_string(&db_service, "dev.rendering.agent_colors.tester")
            .unwrap_or(config.rendering.agent_colors.tester);
        config.rendering.agent_colors.researcher = Self::get_string(&db_service, "dev.rendering.agent_colors.researcher")
            .unwrap_or(config.rendering.agent_colors.researcher);
        config.rendering.agent_colors.reviewer = Self::get_string(&db_service, "dev.rendering.agent_colors.reviewer")
            .unwrap_or(config.rendering.agent_colors.reviewer);
        config.rendering.agent_colors.optimizer = Self::get_string(&db_service, "dev.rendering.agent_colors.optimizer")
            .unwrap_or(config.rendering.agent_colors.optimizer);
        config.rendering.agent_colors.documenter = Self::get_string(&db_service, "dev.rendering.agent_colors.documenter")
            .unwrap_or(config.rendering.agent_colors.documenter);
        config.rendering.agent_colors.default = Self::get_string(&db_service, "dev.rendering.agent_colors.default")
            .unwrap_or(config.rendering.agent_colors.default);

        // Load performance parameters
        config.performance.batch_size_nodes = Self::get_usize(&db_service, "dev.performance.batch_size_nodes")
            .unwrap_or(config.performance.batch_size_nodes);
        config.performance.batch_size_edges = Self::get_usize(&db_service, "dev.performance.batch_size_edges")
            .unwrap_or(config.performance.batch_size_edges);
        config.performance.batch_timeout_ms = Self::get_u64(&db_service, "dev.performance.batch_timeout_ms")
            .unwrap_or(config.performance.batch_timeout_ms);
        config.performance.cache_ttl_secs = Self::get_u64(&db_service, "dev.performance.cache_ttl_secs")
            .unwrap_or(config.performance.cache_ttl_secs);
        config.performance.cache_max_entries = Self::get_usize(&db_service, "dev.performance.cache_max_entries")
            .unwrap_or(config.performance.cache_max_entries);
        config.performance.cache_eviction_percentage = Self::get_f32(&db_service, "dev.performance.cache_eviction_percentage")
            .unwrap_or(config.performance.cache_eviction_percentage);
        config.performance.worker_threads = Self::get_usize(&db_service, "dev.performance.worker_threads")
            .unwrap_or(config.performance.worker_threads);
        config.performance.blocking_threads = Self::get_usize(&db_service, "dev.performance.blocking_threads")
            .unwrap_or(config.performance.blocking_threads);
        config.performance.stack_size_mb = Self::get_usize(&db_service, "dev.performance.stack_size_mb")
            .unwrap_or(config.performance.stack_size_mb);
        config.performance.gc_interval_secs = Self::get_u64(&db_service, "dev.performance.gc_interval_secs")
            .unwrap_or(config.performance.gc_interval_secs);
        config.performance.memory_warning_threshold_mb = Self::get_usize(&db_service, "dev.performance.memory_warning_threshold_mb")
            .unwrap_or(config.performance.memory_warning_threshold_mb);
        config.performance.memory_critical_threshold_mb = Self::get_usize(&db_service, "dev.performance.memory_critical_threshold_mb")
            .unwrap_or(config.performance.memory_critical_threshold_mb);

        // Load debug parameters
        config.debug.enable_cuda_debug = Self::get_bool(&db_service, "dev.debug.enable_cuda_debug")
            .unwrap_or(config.debug.enable_cuda_debug);
        config.debug.enable_physics_debug = Self::get_bool(&db_service, "dev.debug.enable_physics_debug")
            .unwrap_or(config.debug.enable_physics_debug);
        config.debug.enable_network_debug = Self::get_bool(&db_service, "dev.debug.enable_network_debug")
            .unwrap_or(config.debug.enable_network_debug);
        config.debug.enable_memory_tracking = Self::get_bool(&db_service, "dev.debug.enable_memory_tracking")
            .unwrap_or(config.debug.enable_memory_tracking);
        config.debug.enable_performance_tracking = Self::get_bool(&db_service, "dev.debug.enable_performance_tracking")
            .unwrap_or(config.debug.enable_performance_tracking);
        config.debug.log_slow_operations_ms = Self::get_u64(&db_service, "dev.debug.log_slow_operations_ms")
            .unwrap_or(config.debug.log_slow_operations_ms);
        config.debug.log_memory_usage_interval_secs = Self::get_u64(&db_service, "dev.debug.log_memory_usage_interval_secs")
            .unwrap_or(config.debug.log_memory_usage_interval_secs);
        config.debug.profile_sample_rate = Self::get_f32(&db_service, "dev.debug.profile_sample_rate")
            .unwrap_or(config.debug.profile_sample_rate);

        Ok(config)
    }

    /// Get the global developer configuration instance
    pub fn get() -> &'static Arc<RwLock<DevConfig>> {
        DEV_CONFIG.get().expect("DevConfig not initialized. Call DevConfig::initialize() first.")
    }

    // Helper methods for database access
    fn get_f32(db: &Arc<DatabaseService>, key: &str) -> Option<f32> {
        let value = db.get_setting(key).ok()??;
        match value {
            SettingValue::Float(f) => Some(f as f32),
            SettingValue::Integer(i) => Some(i as f32),
            _ => None,
        }
    }

    fn get_u32(db: &Arc<DatabaseService>, key: &str) -> Option<u32> {
        let value = db.get_setting(key).ok()??;
        match value {
            SettingValue::Integer(i) => Some(i as u32),
            _ => None,
        }
    }

    fn get_u64(db: &Arc<DatabaseService>, key: &str) -> Option<u64> {
        let value = db.get_setting(key).ok()??;
        match value {
            SettingValue::Integer(i) => Some(i as u64),
            _ => None,
        }
    }

    fn get_usize(db: &Arc<DatabaseService>, key: &str) -> Option<usize> {
        let value = db.get_setting(key).ok()??;
        match value {
            SettingValue::Integer(i) => Some(i as usize),
            _ => None,
        }
    }

    fn get_bool(db: &Arc<DatabaseService>, key: &str) -> Option<bool> {
        let value = db.get_setting(key).ok()??;
        match value {
            SettingValue::Boolean(b) => Some(b),
            _ => None,
        }
    }

    fn get_string(db: &Arc<DatabaseService>, key: &str) -> Option<String> {
        let value = db.get_setting(key).ok()??;
        match value {
            SettingValue::String(s) => Some(s),
            _ => None,
        }
    }
}

// Convenience functions for common access patterns
pub fn physics() -> PhysicsInternals {
    DevConfig::get().read().unwrap().physics.clone()
}

pub fn cuda() -> CudaInternals {
    DevConfig::get().read().unwrap().cuda.clone()
}

pub fn network() -> NetworkInternals {
    DevConfig::get().read().unwrap().network.clone()
}

pub fn rendering() -> RenderingInternals {
    DevConfig::get().read().unwrap().rendering.clone()
}

pub fn performance() -> PerformanceInternals {
    DevConfig::get().read().unwrap().performance.clone()
}

pub fn debug() -> DebugInternals {
    DevConfig::get().read().unwrap().debug.clone()
}
