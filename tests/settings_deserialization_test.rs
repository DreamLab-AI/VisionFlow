use serde_json::json;
use serde_yaml;

// Import the config module to test deserialization
use ext::config::AppFullSettings;

#[test]
fn test_bloom_to_glow_deserialization() {
    // Test YAML with 'bloom' field
    let yaml_with_bloom = r#"
visualisation:
  rendering:
    ambient_light_intensity: 1.0
    background_color: '#000000'
    directional_light_intensity: 1.0
    enable_ambient_occlusion: false
    enable_antialiasing: true
    enable_shadows: false
    environment_intensity: 1.0
  animations:
    enable_motion_blur: false
    enable_node_animations: true
    motion_blur_strength: 0.2
    selection_wave_enabled: true
    pulse_enabled: true
    pulse_speed: 1.0
    pulse_strength: 1.0
    wave_speed: 1.0
  bloom:
    enabled: true
    intensity: 0.5
    radius: 1.0
    threshold: 0.5
    diffuse_strength: 1.0
    atmospheric_density: 0.1
    volumetric_intensity: 0.1
    base_color: '#ffffff'
    emission_color: '#ffffff'
    opacity: 1.0
    pulse_speed: 1.0
    flow_speed: 1.0
    node_glow_strength: 1.0
    edge_glow_strength: 1.0
    environment_glow_strength: 1.0
  hologram:
    ring_count: 3
    ring_color: '#ffffff'
    ring_opacity: 1.0
    sphere_sizes: [1.0, 2.0]
    ring_rotation_speed: 1.0
    enable_buckminster: false
    buckminster_size: 1.0
    buckminster_opacity: 1.0
    enable_geodesic: false
    geodesic_size: 1.0
    geodesic_opacity: 1.0
    enable_triangle_sphere: false
    triangle_sphere_size: 1.0
    triangle_sphere_opacity: 1.0
    global_rotation_speed: 1.0
  graphs:
    logseq:
      nodes:
        base_color: '#ffffff'
        metalness: 0.0
        opacity: 1.0
        roughness: 0.0
        node_size: 1.0
        quality: 'medium'
        enable_instancing: false
        enable_hologram: false
        enable_metadata_shape: false
        enable_metadata_visualisation: false
      edges:
        arrow_size: 1.0
        base_width: 1.0
        color: '#ffffff'
        enable_arrows: false
        opacity: 1.0
        width_range: [1.0, 2.0]
        quality: 'medium'
      labels:
        desktop_font_size: 1.0
        enable_labels: false
        text_color: '#ffffff'
        text_outline_color: '#000000'
        text_outline_width: 1.0
        text_resolution: 32
        text_padding: 1.0
        billboard_mode: 'camera'
      physics:
        auto_balance: false
        auto_balance_interval_ms: 500
        auto_balance_config:
          stability_variance_threshold: 100.0
          stability_frame_count: 180
          clustering_distance_threshold: 20.0
          bouncing_node_percentage: 0.33
          boundary_min_distance: 90.0
          boundary_max_distance: 100.0
          extreme_distance_threshold: 1000.0
          explosion_distance_threshold: 10000.0
          spreading_distance_threshold: 500.0
          oscillation_detection_frames: 10
          oscillation_change_threshold: 5.0
          min_oscillation_changes: 5
          grid_cell_size_min: 1.0
          grid_cell_size_max: 50.0
          repulsion_cutoff_min: 5.0
          repulsion_cutoff_max: 200.0
          repulsion_softening_min: 0.000001
          repulsion_softening_max: 1.0
          center_gravity_min: 0.0
          center_gravity_max: 0.1
          spatial_hash_efficiency_threshold: 0.3
          cluster_density_threshold: 50.0
          numerical_instability_threshold: 0.001
        attraction_k: 0.0001
        bounds_size: 500.0
        separation_radius: 2.0
        damping: 0.95
        enable_bounds: true
        enabled: true
        iterations: 100
        max_velocity: 1.0
        max_force: 100.0
        repel_k: 50.0
        spring_k: 0.005
        mass_scale: 1.0
        boundary_damping: 0.95
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        stress_weight: 0.1
        stress_alpha: 0.1
        boundary_limit: 490.0
        alignment_strength: 0.0
        cluster_strength: 0.0
        compute_mode: 0
        rest_length: 50.0
        repulsion_cutoff: 50.0
        repulsion_softening_epsilon: 0.0001
        center_gravity_k: 0.0
        grid_cell_size: 50.0
        warmup_iterations: 100
        cooling_rate: 0.001
        boundary_extreme_multiplier: 2.0
        boundary_extreme_force_multiplier: 10.0
        boundary_velocity_damping: 0.5
        min_distance: 0.15
        max_repulsion_dist: 50.0
        boundary_margin: 0.85
        boundary_force_strength: 2.0
        warmup_curve: 'quadratic'
        zero_velocity_iterations: 5
        clustering_algorithm: 'none'
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 30
    visionflow:
      nodes:
        base_color: '#ffffff'
        metalness: 0.0
        opacity: 1.0
        roughness: 0.0
        node_size: 1.0
        quality: 'medium'
        enable_instancing: false
        enable_hologram: false
        enable_metadata_shape: false
        enable_metadata_visualisation: false
      edges:
        arrow_size: 1.0
        base_width: 1.0
        color: '#ffffff'
        enable_arrows: false
        opacity: 1.0
        width_range: [1.0, 2.0]
        quality: 'medium'
      labels:
        desktop_font_size: 1.0
        enable_labels: false
        text_color: '#ffffff'
        text_outline_color: '#000000'
        text_outline_width: 1.0
        text_resolution: 32
        text_padding: 1.0
        billboard_mode: 'camera'
      physics:
        auto_balance: false
        auto_balance_interval_ms: 500
        auto_balance_config:
          stability_variance_threshold: 100.0
          stability_frame_count: 180
          clustering_distance_threshold: 20.0
          bouncing_node_percentage: 0.33
          boundary_min_distance: 90.0
          boundary_max_distance: 100.0
          extreme_distance_threshold: 1000.0
          explosion_distance_threshold: 10000.0
          spreading_distance_threshold: 500.0
          oscillation_detection_frames: 10
          oscillation_change_threshold: 5.0
          min_oscillation_changes: 5
          grid_cell_size_min: 1.0
          grid_cell_size_max: 50.0
          repulsion_cutoff_min: 5.0
          repulsion_cutoff_max: 200.0
          repulsion_softening_min: 0.000001
          repulsion_softening_max: 1.0
          center_gravity_min: 0.0
          center_gravity_max: 0.1
          spatial_hash_efficiency_threshold: 0.3
          cluster_density_threshold: 50.0
          numerical_instability_threshold: 0.001
        attraction_k: 0.0001
        bounds_size: 500.0
        separation_radius: 2.0
        damping: 0.95
        enable_bounds: true
        enabled: true
        iterations: 100
        max_velocity: 1.0
        max_force: 100.0
        repel_k: 50.0
        spring_k: 0.005
        mass_scale: 1.0
        boundary_damping: 0.95
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        stress_weight: 0.1
        stress_alpha: 0.1
        boundary_limit: 490.0
        alignment_strength: 0.0
        cluster_strength: 0.0
        compute_mode: 0
        rest_length: 50.0
        repulsion_cutoff: 50.0
        repulsion_softening_epsilon: 0.0001
        center_gravity_k: 0.0
        grid_cell_size: 50.0
        warmup_iterations: 100
        cooling_rate: 0.001
        boundary_extreme_multiplier: 2.0
        boundary_extreme_force_multiplier: 10.0
        boundary_velocity_damping: 0.5
        min_distance: 0.15
        max_repulsion_dist: 50.0
        boundary_margin: 0.85
        boundary_force_strength: 2.0
        warmup_curve: 'quadratic'
        zero_velocity_iterations: 5
        clustering_algorithm: 'none'
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 30
system:
  network:
    bind_address: '0.0.0.0'
    domain: 'localhost'
    enable_http2: false
    enable_rate_limiting: false
    enable_tls: false
    max_request_size: 1048576
    min_tls_version: '1.2'
    port: 4000
    rate_limit_requests: 100
    rate_limit_window: 60
    tunnel_id: ''
    api_client_timeout: 30
    enable_metrics: false
    max_concurrent_requests: 10
    max_retries: 3
    metrics_port: 9090
    retry_delay: 1000
  websocket:
    binary_chunk_size: 2048
    binary_update_rate: 30
    min_update_rate: 5
    max_update_rate: 60
    motion_threshold: 0.05
    motion_damping: 0.9
    binary_message_version: 1
    compression_enabled: false
    compression_threshold: 512
    heartbeat_interval: 10000
    heartbeat_timeout: 60000
    max_connections: 100
    max_message_size: 1048576
    reconnect_attempts: 5
    reconnect_delay: 1000
    update_rate: 60
  security:
    allowed_origins: []
    audit_log_path: ''
    cookie_httponly: true
    cookie_samesite: 'Strict'
    cookie_secure: true
    csrf_token_timeout: 3600
    enable_audit_logging: false
    enable_request_validation: false
    session_timeout: 3600
  debug:
    enabled: false
  persist_settings: false
xr:
  room_scale: 1.0
  space_type: 'local-floor'
  quality: 'medium'
  interaction_distance: 1.5
  locomotion_method: 'teleport'
  teleport_ray_color: '#ffffff'
  controller_ray_color: '#ffffff'
  enable_hand_tracking: false
  hand_mesh_enabled: false
  hand_mesh_color: '#ffffff'
  hand_mesh_opacity: 1.0
  hand_point_size: 0.01
  hand_ray_enabled: false
  hand_ray_color: '#ffffff'
  hand_ray_width: 0.01
  gesture_smoothing: 0.5
  enable_haptics: false
  haptic_intensity: 0.5
  drag_threshold: 0.1
  pinch_threshold: 0.1
  rotation_threshold: 0.1
  interaction_radius: 0.1
  movement_speed: 1.0
  dead_zone: 0.1
  movement_axes:
    horizontal: 0
    vertical: 1
  enable_light_estimation: false
  enable_plane_detection: false
  enable_scene_understanding: false
  plane_color: '#ffffff'
  plane_opacity: 0.5
  plane_detection_distance: 5.0
  show_plane_overlay: false
  snap_to_floor: false
  enable_passthrough_portal: false
  passthrough_opacity: 0.5
  passthrough_brightness: 1.0
  passthrough_contrast: 1.0
  portal_size: 1.0
  portal_edge_color: '#ffffff'
  portal_edge_width: 0.01
auth:
  enabled: false
  provider: ''
  required: false
"#;

    // Test deserialization
    let settings: AppFullSettings = serde_yaml::from_str(yaml_with_bloom).expect("Should deserialize successfully");
    
    // Verify that glow settings were properly deserialized from bloom field
    assert_eq!(settings.visualisation.glow.enabled, true);
    assert_eq!(settings.visualisation.glow.intensity, 0.5);
    
    println!("Test passed: bloom field successfully deserialized into glow field");
}

#[test]
fn test_glow_field_deserialization() {
    // Test YAML with 'glow' field (for backward compatibility)
    let yaml_with_glow = r#"
visualisation:
  rendering:
    ambient_light_intensity: 1.0
    background_color: '#000000'
    directional_light_intensity: 1.0
    enable_ambient_occlusion: false
    enable_antialiasing: true
    enable_shadows: false
    environment_intensity: 1.0
  animations:
    enable_motion_blur: false
    enable_node_animations: true
    motion_blur_strength: 0.2
    selection_wave_enabled: true
    pulse_enabled: true
    pulse_speed: 1.0
    pulse_strength: 1.0
    wave_speed: 1.0
  glow:
    enabled: false
    intensity: 0.8
    radius: 1.5
    threshold: 0.3
    diffuse_strength: 1.0
    atmospheric_density: 0.1
    volumetric_intensity: 0.1
    base_color: '#ffffff'
    emission_color: '#ffffff'
    opacity: 1.0
    pulse_speed: 1.0
    flow_speed: 1.0
    node_glow_strength: 1.0
    edge_glow_strength: 1.0
    environment_glow_strength: 1.0
  hologram:
    ring_count: 3
    ring_color: '#ffffff'
    ring_opacity: 1.0
    sphere_sizes: [1.0, 2.0]
    ring_rotation_speed: 1.0
    enable_buckminster: false
    buckminster_size: 1.0
    buckminster_opacity: 1.0
    enable_geodesic: false
    geodesic_size: 1.0
    geodesic_opacity: 1.0
    enable_triangle_sphere: false
    triangle_sphere_size: 1.0
    triangle_sphere_opacity: 1.0
    global_rotation_speed: 1.0
  graphs:
    logseq:
      nodes:
        base_color: '#ffffff'
        metalness: 0.0
        opacity: 1.0
        roughness: 0.0
        node_size: 1.0
        quality: 'medium'
        enable_instancing: false
        enable_hologram: false
        enable_metadata_shape: false
        enable_metadata_visualisation: false
      edges:
        arrow_size: 1.0
        base_width: 1.0
        color: '#ffffff'
        enable_arrows: false
        opacity: 1.0
        width_range: [1.0, 2.0]
        quality: 'medium'
      labels:
        desktop_font_size: 1.0
        enable_labels: false
        text_color: '#ffffff'
        text_outline_color: '#000000'
        text_outline_width: 1.0
        text_resolution: 32
        text_padding: 1.0
        billboard_mode: 'camera'
      physics:
        auto_balance: false
        auto_balance_interval_ms: 500
        auto_balance_config:
          stability_variance_threshold: 100.0
          stability_frame_count: 180
          clustering_distance_threshold: 20.0
          bouncing_node_percentage: 0.33
          boundary_min_distance: 90.0
          boundary_max_distance: 100.0
          extreme_distance_threshold: 1000.0
          explosion_distance_threshold: 10000.0
          spreading_distance_threshold: 500.0
          oscillation_detection_frames: 10
          oscillation_change_threshold: 5.0
          min_oscillation_changes: 5
          grid_cell_size_min: 1.0
          grid_cell_size_max: 50.0
          repulsion_cutoff_min: 5.0
          repulsion_cutoff_max: 200.0
          repulsion_softening_min: 0.000001
          repulsion_softening_max: 1.0
          center_gravity_min: 0.0
          center_gravity_max: 0.1
          spatial_hash_efficiency_threshold: 0.3
          cluster_density_threshold: 50.0
          numerical_instability_threshold: 0.001
        attraction_k: 0.0001
        bounds_size: 500.0
        separation_radius: 2.0
        damping: 0.95
        enable_bounds: true
        enabled: true
        iterations: 100
        max_velocity: 1.0
        max_force: 100.0
        repel_k: 50.0
        spring_k: 0.005
        mass_scale: 1.0
        boundary_damping: 0.95
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        stress_weight: 0.1
        stress_alpha: 0.1
        boundary_limit: 490.0
        alignment_strength: 0.0
        cluster_strength: 0.0
        compute_mode: 0
        rest_length: 50.0
        repulsion_cutoff: 50.0
        repulsion_softening_epsilon: 0.0001
        center_gravity_k: 0.0
        grid_cell_size: 50.0
        warmup_iterations: 100
        cooling_rate: 0.001
        boundary_extreme_multiplier: 2.0
        boundary_extreme_force_multiplier: 10.0
        boundary_velocity_damping: 0.5
        min_distance: 0.15
        max_repulsion_dist: 50.0
        boundary_margin: 0.85
        boundary_force_strength: 2.0
        warmup_curve: 'quadratic'
        zero_velocity_iterations: 5
        clustering_algorithm: 'none'
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 30
    visionflow:
      nodes:
        base_color: '#ffffff'
        metalness: 0.0
        opacity: 1.0
        roughness: 0.0
        node_size: 1.0
        quality: 'medium'
        enable_instancing: false
        enable_hologram: false
        enable_metadata_shape: false
        enable_metadata_visualisation: false
      edges:
        arrow_size: 1.0
        base_width: 1.0
        color: '#ffffff'
        enable_arrows: false
        opacity: 1.0
        width_range: [1.0, 2.0]
        quality: 'medium'
      labels:
        desktop_font_size: 1.0
        enable_labels: false
        text_color: '#ffffff'
        text_outline_color: '#000000'
        text_outline_width: 1.0
        text_resolution: 32
        text_padding: 1.0
        billboard_mode: 'camera'
      physics:
        auto_balance: false
        auto_balance_interval_ms: 500
        auto_balance_config:
          stability_variance_threshold: 100.0
          stability_frame_count: 180
          clustering_distance_threshold: 20.0
          bouncing_node_percentage: 0.33
          boundary_min_distance: 90.0
          boundary_max_distance: 100.0
          extreme_distance_threshold: 1000.0
          explosion_distance_threshold: 10000.0
          spreading_distance_threshold: 500.0
          oscillation_detection_frames: 10
          oscillation_change_threshold: 5.0
          min_oscillation_changes: 5
          grid_cell_size_min: 1.0
          grid_cell_size_max: 50.0
          repulsion_cutoff_min: 5.0
          repulsion_cutoff_max: 200.0
          repulsion_softening_min: 0.000001
          repulsion_softening_max: 1.0
          center_gravity_min: 0.0
          center_gravity_max: 0.1
          spatial_hash_efficiency_threshold: 0.3
          cluster_density_threshold: 50.0
          numerical_instability_threshold: 0.001
        attraction_k: 0.0001
        bounds_size: 500.0
        separation_radius: 2.0
        damping: 0.95
        enable_bounds: true
        enabled: true
        iterations: 100
        max_velocity: 1.0
        max_force: 100.0
        repel_k: 50.0
        spring_k: 0.005
        mass_scale: 1.0
        boundary_damping: 0.95
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        stress_weight: 0.1
        stress_alpha: 0.1
        boundary_limit: 490.0
        alignment_strength: 0.0
        cluster_strength: 0.0
        compute_mode: 0
        rest_length: 50.0
        repulsion_cutoff: 50.0
        repulsion_softening_epsilon: 0.0001
        center_gravity_k: 0.0
        grid_cell_size: 50.0
        warmup_iterations: 100
        cooling_rate: 0.001
        boundary_extreme_multiplier: 2.0
        boundary_extreme_force_multiplier: 10.0
        boundary_velocity_damping: 0.5
        min_distance: 0.15
        max_repulsion_dist: 50.0
        boundary_margin: 0.85
        boundary_force_strength: 2.0
        warmup_curve: 'quadratic'
        zero_velocity_iterations: 5
        clustering_algorithm: 'none'
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 30
system:
  network:
    bind_address: '0.0.0.0'
    domain: 'localhost'
    enable_http2: false
    enable_rate_limiting: false
    enable_tls: false
    max_request_size: 1048576
    min_tls_version: '1.2'
    port: 4000
    rate_limit_requests: 100
    rate_limit_window: 60
    tunnel_id: ''
    api_client_timeout: 30
    enable_metrics: false
    max_concurrent_requests: 10
    max_retries: 3
    metrics_port: 9090
    retry_delay: 1000
  websocket:
    binary_chunk_size: 2048
    binary_update_rate: 30
    min_update_rate: 5
    max_update_rate: 60
    motion_threshold: 0.05
    motion_damping: 0.9
    binary_message_version: 1
    compression_enabled: false
    compression_threshold: 512
    heartbeat_interval: 10000
    heartbeat_timeout: 60000
    max_connections: 100
    max_message_size: 1048576
    reconnect_attempts: 5
    reconnect_delay: 1000
    update_rate: 60
  security:
    allowed_origins: []
    audit_log_path: ''
    cookie_httponly: true
    cookie_samesite: 'Strict'
    cookie_secure: true
    csrf_token_timeout: 3600
    enable_audit_logging: false
    enable_request_validation: false
    session_timeout: 3600
  debug:
    enabled: false
  persist_settings: false
xr:
  room_scale: 1.0
  space_type: 'local-floor'
  quality: 'medium'
  interaction_distance: 1.5
  locomotion_method: 'teleport'
  teleport_ray_color: '#ffffff'
  controller_ray_color: '#ffffff'
  enable_hand_tracking: false
  hand_mesh_enabled: false
  hand_mesh_color: '#ffffff'
  hand_mesh_opacity: 1.0
  hand_point_size: 0.01
  hand_ray_enabled: false
  hand_ray_color: '#ffffff'
  hand_ray_width: 0.01
  gesture_smoothing: 0.5
  enable_haptics: false
  haptic_intensity: 0.5
  drag_threshold: 0.1
  pinch_threshold: 0.1
  rotation_threshold: 0.1
  interaction_radius: 0.1
  movement_speed: 1.0
  dead_zone: 0.1
  movement_axes:
    horizontal: 0
    vertical: 1
  enable_light_estimation: false
  enable_plane_detection: false
  enable_scene_understanding: false
  plane_color: '#ffffff'
  plane_opacity: 0.5
  plane_detection_distance: 5.0
  show_plane_overlay: false
  snap_to_floor: false
  enable_passthrough_portal: false
  passthrough_opacity: 0.5
  passthrough_brightness: 1.0
  passthrough_contrast: 1.0
  portal_size: 1.0
  portal_edge_color: '#ffffff'
  portal_edge_width: 0.01
auth:
  enabled: false
  provider: ''
  required: false
"#;

    // Test deserialization
    let settings: AppFullSettings = serde_yaml::from_str(yaml_with_glow).expect("Should deserialize successfully with glow field");
    
    // Verify that glow settings were properly deserialized
    assert_eq!(settings.visualisation.glow.enabled, false);
    assert_eq!(settings.visualisation.glow.intensity, 0.8);
    
    println!("Test passed: glow field successfully deserialized");
}

#[test]
fn test_serialization_uses_bloom_name() {
    let settings = AppFullSettings::default();
    
    // Serialize to YAML
    let yaml = serde_yaml::to_string(&settings).expect("Should serialize successfully");
    
    // Check that the serialized YAML contains 'bloom' not 'glow'
    assert!(yaml.contains("bloom:"));
    assert!(!yaml.contains("glow:"));
    
    println!("Test passed: serialization uses 'bloom' field name");
}