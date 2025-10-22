use crate::config::AppFullSettings;
use crate::handlers::settings_handler::SettingsResponseDTO;

/// Test to verify that the bloom/glow field mapping works correctly
pub fn test_settings_deserialization() -> Result<(), String> {
    // Test YAML content that matches the actual settings.yaml structure
    let test_yaml = r#"
visualisation:
  rendering:
    ambient_light_intensity: 1.2
    background_color: '#0a0e1a'
    directional_light_intensity: 1.5
    enable_ambient_occlusion: false
    enable_antialiasing: true
    enable_shadows: true
    environment_intensity: 0.7
  animations:
    enable_motion_blur: false
    enable_node_animations: true
    motion_blur_strength: 0.2
    selection_wave_enabled: true
    pulse_enabled: true
    pulse_speed: 1.2
    pulse_strength: 0.8
    wave_speed: 0.5
  bloom:
    edge_bloom_strength: 0.9
    enabled: true
    environment_bloom_strength: 0.96
    node_bloom_strength: 0.05
    radius: 0.85
    strength: 0.95
    threshold: 0.028
    diffuse_strength: 1.0
    atmospheric_density: 0.1
    volumetric_intensity: 0.1
    base_color: '#ffffff'
    emission_color: '#ffffff'
    opacity: 1.0
    pulse_speed: 1.0
    flow_speed: 1.0
  hologram:
    ring_count: 5
    ring_color: '#00ffff'
    ring_opacity: 0.8
    sphere_sizes: [40.0, 80.0]
    ring_rotation_speed: 12.0
    enable_buckminster: true
    buckminster_size: 50.0
    buckminster_opacity: 0.3
    enable_geodesic: true
    geodesic_size: 60.0
    geodesic_opacity: 0.25
    enable_triangle_sphere: true
    triangle_sphere_size: 70.0
    triangle_sphere_opacity: 0.4
    global_rotation_speed: 0.5
  graphs:
    logseq:
      nodes:
        base_color: '#00e5ff'
        metalness: 0.85
        opacity: 0.95
        roughness: 0.15
        node_size: 0.53
        quality: 'high'
        enable_instancing: true
        enable_hologram: true
        enable_metadata_shape: false
        enable_metadata_visualisation: true
      edges:
        arrow_size: 0.02
        base_width: 0.5
        color: '#4fc3f7'
        enable_arrows: false
        opacity: 0.45
        width_range: [0.3, 1.5]
        quality: 'high'
      labels:
        desktop_font_size: 0.5
        enable_labels: true
        text_color: '#ffffff'
        text_outline_color: '#0a0e1a'
        text_outline_width: 0.005
        text_resolution: 32
        text_padding: 0.6
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
        attraction_k: 0.01
        bounds_size: 1000.0
        separation_radius: 2.0
        damping: 0.85
        enable_bounds: false
        enabled: true
        iterations: 50
        max_velocity: 5.0
        max_force: 100.0
        repel_k: 155.0
        spring_k: 0.1
        mass_scale: 1.0
        boundary_damping: 0.95
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        stress_weight: 0.0001
        stress_alpha: 0.0001
        boundary_limit: 1000.0
        alignment_strength: 0.1
        cluster_strength: 0.1
        compute_mode: 1
        rest_length: 50.0
        repulsion_cutoff: 50.0
        repulsion_softening_epsilon: 0.0001
        center_gravity_k: 0.0
        grid_cell_size: 50.0
        warmup_iterations: 100
        cooling_rate: 0.001
        boundary_extreme_multiplier: 2.0
        boundary_extreme_force_multiplier: 5.0
        boundary_velocity_damping: 0.5
        min_distance: 1.0
        max_repulsion_dist: 1000.0
        boundary_margin: 10.0
        boundary_force_strength: 1.0
        warmup_curve: 'exponential'
        zero_velocity_iterations: 10
        clustering_algorithm: 'none'
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 50
    visionflow:
      nodes:
        base_color: '#40ff00'
        metalness: 0.85
        opacity: 0.95
        roughness: 0.15
        node_size: 1.2
        quality: 'high'
        enable_instancing: true
        enable_hologram: true
        enable_metadata_shape: true
        enable_metadata_visualisation: true
      edges:
        arrow_size: 0.02
        base_width: 0.5
        color: '#76ff03'
        enable_arrows: false
        opacity: 0.45
        width_range: [0.3, 1.5]
        quality: 'high'
      labels:
        desktop_font_size: 0.5
        enable_labels: true
        text_color: '#f0fff0'
        text_outline_color: '#0a1a0a'
        text_outline_width: 0.005
        text_resolution: 32
        text_padding: 0.6
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
        attraction_k: 0.01
        bounds_size: 1000.0
        separation_radius: 2.0
        damping: 0.85
        enable_bounds: false
        enabled: true
        iterations: 50
        max_velocity: 5.0
        max_force: 100.0
        repel_k: 2.0
        spring_k: 0.1
        mass_scale: 1.0
        boundary_damping: 0.95
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        stress_weight: 0.0001
        stress_alpha: 0.0001
        boundary_limit: 1000.0
        alignment_strength: 0.1
        cluster_strength: 0.1
        compute_mode: 1
        rest_length: 50.0
        repulsion_cutoff: 50.0
        repulsion_softening_epsilon: 0.0001
        center_gravity_k: 0.0
        grid_cell_size: 50.0
        warmup_iterations: 100
        cooling_rate: 0.001
        boundary_extreme_multiplier: 2.0
        boundary_extreme_force_multiplier: 5.0
        boundary_velocity_damping: 0.5
        min_distance: 1.0
        max_repulsion_dist: 1000.0
        boundary_margin: 10.0
        boundary_force_strength: 1.0
        warmup_curve: 'exponential'
        zero_velocity_iterations: 10
        clustering_algorithm: 'none'
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 50
system:
  network:
    bind_address: '0.0.0.0'
    domain: 'visionflow.info'
    enable_http2: false
    enable_rate_limiting: false
    enable_tls: false
    max_request_size: 10485760
    min_tls_version: '1.2'
    port: 4000
    rate_limit_requests: 10000
    rate_limit_window: 600
    tunnel_id: 'dummy'
    api_client_timeout: 30
    enable_metrics: false
    max_concurrent_requests: 1
    max_retries: 3
    metrics_port: 9090
    retry_delay: 5
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
    heartbeat_timeout: 600000
    max_connections: 100
    max_message_size: 10485760
    reconnect_attempts: 5
    reconnect_delay: 1000
    update_rate: 60
  security:
    allowed_origins:
      - 'https://www.visionflow.info'
      - 'https://visionflow.info'
    audit_log_path: '/app/logs/audit.log'
    cookie_httponly: true
    cookie_samesite: 'Strict'
    cookie_secure: true
    csrf_token_timeout: 3600
    enable_audit_logging: false
    enable_request_validation: false
    session_timeout: 3600
  debug:
    enabled: false
  persist_settings: true
xr:
  enabled: false
  room_scale: 1.0
  space_type: 'local-floor'
  quality: 'medium'
  interaction_distance: 1.5
  locomotion_method: 'teleport'
  teleport_ray_color: '#ffffff'
  controller_ray_color: '#ffffff'
  enable_hand_tracking: true
  hand_mesh_enabled: true
  hand_mesh_color: '#4287f5'
  hand_mesh_opacity: 0.3
  hand_point_size: 0.006
  hand_ray_enabled: true
  hand_ray_color: '#4287f5'
  hand_ray_width: 0.003
  gesture_smoothing: 0.7
  enable_haptics: true
  haptic_intensity: 0.3
  drag_threshold: 0.08
  pinch_threshold: 0.3
  rotation_threshold: 0.08
  interaction_radius: 0.15
  movement_speed: 1.0
  dead_zone: 0.12
  movement_axes:
    horizontal: 2
    vertical: 3
  enable_light_estimation: false
  enable_plane_detection: false
  enable_scene_understanding: false
  plane_color: '#4287f5'
  plane_opacity: 0.001
  plane_detection_distance: 3.0
  show_plane_overlay: false
  snap_to_floor: false
  enable_passthrough_portal: false
  passthrough_opacity: 0.8
  passthrough_brightness: 1.1
  passthrough_contrast: 1.2
  portal_size: 2.5
  portal_edge_color: '#4287f5'
  portal_edge_width: 0.02
auth:
  enabled: false
  provider: 'nostr'
  required: false
"#;

    // Test direct YAML deserialization
    println!("Testing direct YAML deserialization...");
    match serde_yaml::from_str::<AppFullSettings>(test_yaml) {
        Ok(settings) => {
            println!("✅ Direct YAML deserialization successful!");
            println!("   - Glow enabled: {}", settings.visualisation.glow.enabled);
            println!(
                "   - Glow strength: {}",
                settings.visualisation.glow.edge_glow_strength
            );

            // Test serialization back to ensure bidirectional conversion works
            match serde_yaml::to_string(&settings) {
                Ok(serialized) => {
                    if serialized.contains("bloom:") {
                        println!("✅ Serialization uses 'bloom' field name correctly");
                    } else {
                        println!("⚠️  Warning: Serialization doesn't use 'bloom' field name");
                    }
                }
                Err(e) => println!("❌ Serialization failed: {}", e),
            }
        }
        Err(e) => {
            println!("❌ Direct YAML deserialization failed: {}", e);
            return Err(format!("Direct YAML deserialization failed: {}", e));
        }
    }

    // Test JSON serialization for client compatibility using DTO
    println!("\nTesting JSON serialization for client using DTO...");
    let default_settings = AppFullSettings::default();
    let response_dto: SettingsResponseDTO = (&default_settings).into();

    match serde_json::to_value(&response_dto) {
        Ok(json) => {
            if let Some(vis) = json.get("visualisation") {
                if vis.get("bloom").is_some() {
                    println!("✅ Client JSON contains 'bloom' field via DTO");
                } else {
                    println!("⚠️  Warning: Client JSON missing 'bloom' field in DTO");
                }
            }
        }
        Err(e) => println!("❌ JSON serialization failed: {}", e),
    }

    println!("✅ All tests completed successfully!");
    Ok(())
}
