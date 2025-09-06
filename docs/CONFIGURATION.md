# VisionFlow Configuration Guide

## Configuration Hierarchy

VisionFlow uses multiple configuration files with a clear precedence order. This document explains the role of each configuration file and how they interact.

## Configuration Files Overview

```
ext/
├── .env                     # Environment variables & secrets
├── .env_template            # Template for .env file
├── data/
│   └── settings.yaml        # Application settings (runtime configurable)
├── config.yml               # Cloudflare tunnel configuration
├── docker-compose.yml       # Docker service configuration with profiles
└── supervisord.*.conf       # Process management configuration
```

## File Roles and Precedence

### 1. Environment Variables (`.env`)
**Purpose:** Store secrets, API keys, and environment-specific configuration
**Precedence:** Highest - overrides all other configurations
**Scope:** Docker containers and application runtime

Key variables:
- `GITHUB_TOKEN` - Required for Logseq graph synchronisation
- `OPENAI_API_KEY` - AI service integration
- `ANTHROPIC_API_KEY` - Claude API integration
- `PERPLEXITY_API_KEY` - Search and query service
- `CLOUDFLARE_TUNNEL_TOKEN` - Production tunnel authentication
- `CUDA_ARCH` - GPU architecture for compilation
- `MCP_TCP_PORT` - Claude Flow MCP port (default: 9500)
- `MCP_TRANSPORT` - Transport protocol (tcp, websocket)
- `CLAUDE_FLOW_HOST` - Claude Flow MCP hostname (default: multi-agent-container)
- `RUST_LOG` - Logging level configuration

**Best Practice:** Never commit `.env` to version control. Use `.env_template` as reference.

**GPU Safety Configuration:**
- `GPU_MEMORY_LIMIT` - Maximum GPU memory (default: 8GB)
- `CPU_FALLBACK_THRESHOLD` - Failed GPU operations before CPU fallback (default: 3)
- `KERNEL_TIMEOUT` - GPU kernel timeout in seconds (default: 5)

### 2. Application Settings (`data/settings.yaml`)
**Purpose:** Runtime application configuration and defaults
**Precedence:** Medium - can be overridden by environment variables
**Scope:** Application behavior and feature configuration

Structure:
```yaml
system:
  network:
    bind_address: "0.0.0.0"
    port: 4000
    domain: "visionflow.info"
    enable_http2: false
    enable_rate_limiting: false
    enable_tls: false
    max_request_size: 10485760
    min_tls_version: "1.2"
    rate_limit_requests: 10000
    rate_limit_window: 600
    tunnel_id: "dummy"
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
      - "https://www.visionflow.info"
      - "https://visionflow.info"
    audit_log_path: "/app/logs/audit.log"
    cookie_httponly: true
    cookie_samesite: "Strict"
    cookie_secure: true
    csrf_token_timeout: 3600
    enable_audit_logging: false
    enable_request_validation: false
    session_timeout: 3600
  debug:
    enabled: false
  persist_settings: true

visualisation:
  rendering:
    ambient_light_intensity: 1.5544538
    background_color: '#0a0e1a'
    directional_light_intensity: 1.3690603
    enable_ambient_occlusion: true
    enable_antialiasing: false
    enable_shadows: true
    environment_intensity: 0.7
    shadow_map_size: '2048'
    shadow_bias: 0.0001
    context: desktop
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
    edge_bloom_strength: 0.3743524
    enabled: false
    environment_bloom_strength: 0.3672219
    node_bloom_strength: 0.30661243
    radius: 0.29591668
    strength: 0.3386998
    threshold: 0.41357028
  hologram:
    ring_count: 3
    ring_color: '#e83211'
    ring_opacity: 0.23174196
    sphere_sizes: [40.0, 80.0]
    ring_rotation_speed: 24.778564
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
    logseq:      # Knowledge graph (primary)
      nodes:
        base_color: '#a06522'
        metalness: 0.44209236
        opacity: 0.5098323
        roughness: 0.5561807
        node_size: 1.8685422
        quality: high
        enable_instancing: false
        enable_hologram: true
        enable_metadata_shape: false
        enable_metadata_visualisation: false
      edges:
        arrow_size: 0.34017882
        base_width: 1.9043301
        color: '#e8edee'
        enable_arrows: false
        opacity: 0.9519247
        width_range: [0.2, 1.0]
        quality: medium
      labels:
        desktop_font_size: 1.008701
        enable_labels: true
        text_color: '#5d5d09'
        text_outline_color: '#4561b5'
        text_outline_width: 0.0026382932
        text_resolution: 32
        text_padding: 0.6
        billboard_mode: camera
        show_metadata: true
        max_label_width: 5.0
      physics:
        # Auto-balance configuration
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
        # Core physics parameters
        attraction_k: 8.378364
        bounds_size: 5000.0
        separation_radius: 2.0
        damping: 0.85
        enable_bounds: true
        enabled: true
        iterations: 50
        max_velocity: 5.0
        repel_k: 2.0
        spring_k: 0.1
        mass_scale: 1.0
        boundary_damping: 0.23174196
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        # GPU-aligned fields
        stress_weight: 0.5847028
        stress_alpha: 0.55261546
        boundary_limit: 44.637775
        alignment_strength: 0.0
        cluster_strength: 0.0
        compute_mode: 0
        min_distance: 0.44966576
        max_repulsion_dist: 300.0
        boundary_margin: 0.85
        boundary_force_strength: 2.0
        warmup_iterations: 219
        warmup_curve: quadratic
        zero_velocity_iterations: 5
        cooling_rate: 0.0036785465
        # Clustering parameters
        clustering_algorithm: none
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 30
    visionflow:  # Agent graph (secondary)
      nodes:
        base_color: '#ff8800'
        metalness: 0.7
        opacity: 0.9
        roughness: 0.3
        node_size: 1.5
        quality: high
        enable_instancing: true
        enable_hologram: true
        enable_metadata_shape: true
        enable_metadata_visualisation: true
      edges:
        arrow_size: 0.7
        base_width: 0.15
        color: '#ffaa00'
        enable_arrows: true
        opacity: 0.6
        width_range: [0.15, 1.5]
        quality: high
      labels:
        desktop_font_size: 16.0
        enable_labels: true
        text_color: '#ffaa00'
        text_outline_color: '#000000'
        text_outline_width: 2.5
        text_resolution: 256
        text_padding: 5.0
        billboard_mode: screen
        show_metadata: true
        max_label_width: 8.0
      physics:
        # Auto-balance configuration
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
        # Core physics parameters  
        attraction_k: 0.0001
        bounds_size: 5000.0
        separation_radius: 2.0
        damping: 0.95
        enable_bounds: true
        enabled: true
        iterations: 100
        max_velocity: 1.0
        repel_k: 5.0
        spring_k: 0.5
        mass_scale: 1.0
        boundary_damping: 0.95
        update_threshold: 0.01
        dt: 0.016
        temperature: 0.01
        gravity: 0.0001
        # GPU-aligned fields
        stress_weight: 0.1
        stress_alpha: 0.1
        boundary_limit: 490.0
        alignment_strength: 0.0
        cluster_strength: 0.0
        compute_mode: 0
        min_distance: 0.15
        max_repulsion_dist: 300.0
        boundary_margin: 0.85
        boundary_force_strength: 2.0
        warmup_iterations: 200
        warmup_curve: quadratic
        zero_velocity_iterations: 5
        cooling_rate: 0.0001
        # Clustering parameters
        clustering_algorithm: none
        cluster_count: 5
        clustering_resolution: 1.0
        clustering_iterations: 30

xr:
  enabled: false
  client_side_enable_xr: false
  mode: "immersive-vr"
  room_scale: 1.0
  space_type: "local-floor"
  quality: medium
  render_scale: 0.92248344
  interaction_distance: 1.5
  locomotion_method: teleport
  teleport_ray_color: '#ffffff'
  controller_ray_color: '#ffffff'
  controller_model: default
  # Hand tracking
  enable_hand_tracking: true
  hand_mesh_enabled: true
  hand_mesh_color: '#4287f5'
  hand_mesh_opacity: 0.3
  hand_point_size: 0.006
  hand_ray_enabled: true
  hand_ray_color: '#4287f5'
  hand_ray_width: 0.003
  gesture_smoothing: 0.7
  # Haptics and interaction
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
  # Environment understanding
  enable_light_estimation: false
  enable_plane_detection: false
  enable_scene_understanding: false
  plane_color: '#4287f5'
  plane_opacity: 0.001
  plane_detection_distance: 3.0
  show_plane_overlay: false
  snap_to_floor: false
  # Passthrough portal
  enable_passthrough_portal: false
  passthrough_opacity: 0.8
  passthrough_brightness: 1.1
  passthrough_contrast: 1.2
  portal_size: 2.5
  portal_edge_color: '#4287f5'
  portal_edge_width: 0.02

auth:
  enabled: true
  provider: nostr
  required: true

# AI service settings (optional - configured via environment variables)
ragflow:
  agent_id: aa2e328812ef11f083dc0a0d6226f61b
  timeout: 30
  max_retries: 3

perplexity:
  model: "llama-3.1-sonar-small-128k-online"
  max_tokens: 4096
  temperature: 0.5
  top_p: 0.9
  presence_penalty: 0.0
  frequency_penalty: 0.0
  timeout: 30
  rate_limit: 100

openai:
  timeout: 30
  rate_limit: 100

kokoro:
  api_url: "http://recursing_bhaskara:8880"
  default_voice: af_heart
  default_format: mp3
  default_speed: 1.0
  timeout: 30
  stream: true
  return_timestamps: true
  sample_rate: 24000

whisper:
  api_url: "http://whisper-webui-backend:8000"
  default_model: base
  default_language: en
  timeout: 30
  temperature: 0.0
  return_timestamps: false
  vad_filter: false
  word_timestamps: false
```

**Note:** Legacy top-level keys (`nodes`, `edges`, `physics`) under `visualisation` are deprecated. Use the `graphs` namespace.

### 3. Docker Compose Configuration (`docker-compose.yml`)
**Purpose:** Container orchestration and service definitions
**Precedence:** Defines how environment variables are passed to containers
**Scope:** Container runtime, networking, volumes

Profiles:
- `dev` - Development environment with hot reloading
- `production` / `prod` - Optimised production deployment

Usage:
```bash
# Development
docker-compose --profile dev up

# Production
docker-compose --profile production up
```

### 4. Cloudflare Configuration (`config.yml`)
**Purpose:** Tunnel routing for production deployment
**Precedence:** Independent - only used by Cloudflare tunnel
**Scope:** External access and routing

Example:
```yaml
tunnel: your-tunnel-id
credentials-file: /etc/cloudflared/credentials.json
ingress:
  - hostname: your-domain.com
    service: http://webxr:4000
  - service: http_status:404
```

### 5. Supervisor Configuration (`supervisord.*.conf`)
**Purpose:** Process management within containers
**Precedence:** Independent - manages process lifecycle
**Scope:** Container internal processes

Files:
- `supervisord.dev.conf` - Development processes (Vite, Rust watch)
- `supervisord.conf` - Production processes (Nginx, Rust server)

## Configuration Loading Order

1. **Container Start:**
   - Docker Compose reads `.env` file
   - Environment variables set in container

2. **Application Start:**
   - Rust backend loads `settings.yaml`
   - Environment variables override settings values
   - Configuration merged and validated

3. **Runtime:**
   - Settings can be modified via API (`/api/settings`)
   - Changes persist to `settings.yaml`
   - Some settings require restart

## Environment Variable Overrides

Environment variables can override `settings.yaml` values using this pattern:

```bash
# Override system.network.port
SYSTEM_NETWORK_PORT=5000

# Override system.network.bind_address
SYSTEM_NETWORK_BIND_ADDRESS="127.0.0.1"

# Override system.debug.enabled
SYSTEM_DEBUG_ENABLED=true

# Override visualisation.graphs.logseq.physics.repel_k
VISUALISATION_GRAPHS_LOGSEQ_PHYSICS_REPEL_K=50.0

# Override visualisation.graphs.logseq.physics.spring_k
VISUALISATION_GRAPHS_LOGSEQ_PHYSICS_SPRING_K=0.005

# Override visualisation.graphs.visionflow.physics.damping
VISUALISATION_GRAPHS_VISIONFLOW_PHYSICS_DAMPING=0.98

# Override xr.enabled
XR_ENABLED=true

# Override auth.provider
AUTH_PROVIDER="custom"
```

## Configuration Best Practices

### Development
1. Copy `.env_template` to `.env`
2. Set required API keys
3. Use `docker-compose --profile dev up`
4. Modify `settings.yaml` for testing

### Production
1. Use environment variables for all secrets
2. Mount `settings.yaml` as read-only
3. Use `docker-compose --profile production up`
4. Configure Cloudflare tunnel for external access

### Testing Different Configurations
```bash
# Test with custom settings
docker run -v ./custom-settings.yaml:/app/settings.yaml ...

# Override specific values
SYSTEM_NETWORK_PORT=8080 docker-compose --profile dev up

# Use different GPU
CUDA_ARCH=89 NVIDIA_VISIBLE_DEVICES=1 docker-compose up
```

## Debugging Configuration Issues

### Check Loaded Configuration
```bash
# View current settings via API
curl http://localhost:4000/api/settings

# Check environment variables in container
docker exec visionflow_container env | sort

# Validate settings.yaml
docker exec visionflow_container cat /app/settings.yaml
```

### Common Issues

1. **Port conflicts:**
   - Check `SYSTEM_NETWORK_PORT` in environment
   - Verify `settings.yaml` port configuration
   - Ensure Docker port mapping matches

2. **GPU not detected:**
   - Verify `CUDA_ARCH` matches your GPU
   - Check `NVIDIA_VISIBLE_DEVICES` is correct
   - Ensure Docker has GPU runtime

3. **API keys not working:**
   - Confirm `.env` file is loaded
   - Check environment variables in container
   - Verify key format and validity

## Migration Notes

### From Legacy Configuration
If upgrading from an older version:

1. Move top-level physics settings to `graphs.logseq`:
   ```yaml
   # Old (deprecated)
   visualisation:
     physics:
       repulsion_strength: 1500

   # New
   visualisation:
     graphs:
       logseq:
         physics:
           repulsion_strength: 1500
   ```

2. Update environment variable names:
   ```bash
   # Old
   WEBSOCKET_PORT=3002

   # New
   MCP_TCP_PORT=9500
   MCP_TRANSPORT=tcp
   ```

## Security Considerations

1. **Never commit secrets:**
   - Use `.gitignore` for `.env`
   - Store production secrets in secure vault
   - Rotate API keys regularly

2. **Restrict file permissions:**
   ```bash
   chmod 600 .env
   chmod 644 settings.yaml
   ```

3. **Use read-only mounts in production:**
   ```yaml
   volumes:
     - ./settings.yaml:/app/settings.yaml:ro
   ```

## Support

For configuration issues:
1. Check this guide first
2. Review logs: `docker logs visionflow_container`
3. Consult `/docs/deployment/` for specific scenarios
4. Open an issue with configuration details (excluding secrets)