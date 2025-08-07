// Settings Migration - Converts old complex structure to new unified structure
// This ensures backward compatibility during the transition

use crate::config::AppFullSettings;
use crate::config::unified::{UnifiedSettings, GraphConfig};
use log::{info, warn};

impl From<AppFullSettings> for UnifiedSettings {
    fn from(old: AppFullSettings) -> Self {
        info!("Migrating from old settings structure to unified structure");
        
        let mut unified = UnifiedSettings::default();
        
        // Migrate visualisation settings
        let vis = &old.visualisation;
        
        // Create logseq graph config from old settings
        // Priority: Use multi-graph structure if available, fall back to flat structure
        // Check if any physics values are non-default to determine which to use
        let logseq_physics = if vis.graphs.logseq.physics.spring_strength != 0.0 
            || vis.graphs.logseq.physics.damping != 0.0 {
            info!("Using multi-graph physics for logseq");
            convert_physics(&vis.graphs.logseq.physics)
        } else {
            info!("Using legacy flat physics for logseq");
            convert_physics(&vis.physics)
        };
        
        unified.graphs.logseq = GraphConfig {
            nodes: convert_nodes(&vis.graphs.logseq.nodes, &vis.nodes),
            edges: convert_edges(&vis.graphs.logseq.edges, &vis.edges),
            labels: convert_labels(&vis.graphs.logseq.labels, &vis.labels),
            physics: logseq_physics,
        };
        
        // Create visionflow graph config
        unified.graphs.visionflow = GraphConfig {
            nodes: convert_nodes(&vis.graphs.visionflow.nodes, &vis.nodes),
            edges: convert_edges(&vis.graphs.visionflow.edges, &vis.edges),
            labels: convert_labels(&vis.graphs.visionflow.labels, &vis.labels),
            physics: convert_physics(&vis.graphs.visionflow.physics),
        };
        
        // Migrate global rendering settings
        unified.rendering = crate::config::unified::RenderingSettings {
            ambient_light_intensity: vis.rendering.ambient_light_intensity,
            background_color: vis.rendering.background_color.clone(),
            directional_light_intensity: vis.rendering.directional_light_intensity,
            enable_ambient_occlusion: vis.rendering.enable_ambient_occlusion,
            enable_antialiasing: vis.rendering.enable_antialiasing,
            enable_shadows: vis.rendering.enable_shadows,
            environment_intensity: vis.rendering.environment_intensity,
        };
        
        // Migrate animation settings
        unified.animations = crate::config::unified::AnimationSettings {
            enable_motion_blur: vis.animations.enable_motion_blur,
            enable_node_animations: vis.animations.enable_node_animations,
            motion_blur_strength: vis.animations.motion_blur_strength,
            selection_wave_enabled: vis.animations.selection_wave_enabled,
            pulse_enabled: vis.animations.pulse_enabled,
            pulse_speed: vis.animations.pulse_speed,
            pulse_strength: vis.animations.pulse_strength,
            wave_speed: vis.animations.wave_speed,
        };
        
        // Migrate bloom settings
        unified.bloom = crate::config::unified::BloomSettings {
            enabled: vis.bloom.enabled,
            strength: vis.bloom.strength,
            radius: vis.bloom.radius,
            node_bloom_strength: vis.bloom.node_bloom_strength,
            edge_bloom_strength: vis.bloom.edge_bloom_strength,
            environment_bloom_strength: vis.bloom.environment_bloom_strength,
        };
        
        // Migrate websocket settings
        unified.websocket = crate::config::unified::WebSocketSettings {
            update_rate: old.system.websocket.update_rate,
            compression_enabled: old.system.websocket.compression_enabled,
            compression_threshold: old.system.websocket.compression_threshold,
            reconnect_attempts: old.system.websocket.reconnect_attempts,
            reconnect_delay: old.system.websocket.reconnect_delay,
            binary_chunk_size: old.system.websocket.binary_chunk_size,
            heartbeat_interval: old.system.websocket.heartbeat_interval,
            max_message_size: old.system.websocket.max_message_size,
        };
        
        // Migrate debug settings
        unified.debug = crate::config::unified::DebugSettings {
            enabled: old.system.debug.enabled,
            enable_physics_debug: false, // New field, default to false
            enable_websocket_debug: old.system.debug.enable_websocket_debug,
            log_level: old.system.debug.log_level.clone(),
        };
        
        // Network settings (server-only)
        unified.network = Some(crate::config::unified::NetworkSettings {
            bind_address: old.system.network.bind_address.clone(),
            port: old.system.network.port,
            domain: old.system.network.domain.clone(),
        });
        
        info!("Settings migration completed successfully");
        unified
    }
}

// Helper functions for converting individual setting groups

fn convert_physics(old: &crate::config::PhysicsSettings) -> crate::config::unified::PhysicsSettings {
    crate::config::unified::PhysicsSettings {
        enabled: old.enabled,
        iterations: old.iterations,
        damping: old.damping,
        spring_strength: old.spring_strength,
        repulsion_strength: old.repulsion_strength,
        repulsion_distance: old.repulsion_distance,
        attraction_strength: old.attraction_strength,
        max_velocity: old.max_velocity,
        collision_radius: old.collision_radius,
        bounds_size: old.bounds_size,
        enable_bounds: old.enable_bounds,
        mass_scale: old.mass_scale,
        boundary_damping: old.boundary_damping,
    }
}

fn convert_nodes(
    graph_nodes: &crate::config::NodeSettings,
    fallback: &crate::config::NodeSettings
) -> crate::config::unified::NodeSettings {
    // Use graph-specific settings if they're not default, otherwise use fallback
    let nodes = if graph_nodes.base_color != String::default() {
        graph_nodes
    } else {
        fallback
    };
    
    crate::config::unified::NodeSettings {
        base_color: nodes.base_color.clone(),
        metalness: nodes.metalness,
        opacity: nodes.opacity,
        roughness: nodes.roughness,
        node_size: nodes.node_size,
        quality: nodes.quality.clone(),
        enable_instancing: nodes.enable_instancing,
        enable_hologram: nodes.enable_hologram,
        enable_metadata_shape: nodes.enable_metadata_shape,
        enable_metadata_visualisation: nodes.enable_metadata_visualisation,
    }
}

fn convert_edges(
    graph_edges: &crate::config::EdgeSettings,
    fallback: &crate::config::EdgeSettings
) -> crate::config::unified::EdgeSettings {
    let edges = if graph_edges.color != String::default() {
        graph_edges
    } else {
        fallback
    };
    
    crate::config::unified::EdgeSettings {
        arrow_size: edges.arrow_size,
        base_width: edges.base_width,
        color: edges.color.clone(),
        enable_arrows: edges.enable_arrows,
        opacity: edges.opacity,
        width_range: edges.width_range.clone(),
        quality: edges.quality.clone(),
    }
}

fn convert_labels(
    graph_labels: &crate::config::LabelSettings,
    fallback: &crate::config::LabelSettings
) -> crate::config::unified::LabelSettings {
    let labels = if graph_labels.text_color != String::default() {
        graph_labels
    } else {
        fallback
    };
    
    crate::config::unified::LabelSettings {
        desktop_font_size: labels.desktop_font_size,
        enable_labels: labels.enable_labels,
        text_color: labels.text_color.clone(),
        text_outline_color: labels.text_outline_color.clone(),
        text_outline_width: labels.text_outline_width,
        text_resolution: labels.text_resolution,
        text_padding: labels.text_padding,
        billboard_mode: labels.billboard_mode.clone(),
    }
}

// Backward compatibility: Convert UnifiedSettings to old format if needed
impl From<UnifiedSettings> for AppFullSettings {
    fn from(unified: UnifiedSettings) -> Self {
        warn!("Converting from unified settings to old format - this should be temporary!");
        
        let mut old = AppFullSettings::new().unwrap_or_else(|_| {
            panic!("Failed to create default AppFullSettings");
        });
        
        // Copy logseq settings to both flat and multi-graph structures for compatibility
        let logseq = &unified.graphs.logseq;
        
        // Update flat physics (legacy)
        old.visualisation.physics = crate::config::PhysicsSettings {
            attraction_strength: logseq.physics.attraction_strength,
            bounds_size: logseq.physics.bounds_size,
            collision_radius: logseq.physics.collision_radius,
            damping: logseq.physics.damping,
            enable_bounds: logseq.physics.enable_bounds,
            enabled: logseq.physics.enabled,
            iterations: logseq.physics.iterations,
            max_velocity: logseq.physics.max_velocity,
            repulsion_strength: logseq.physics.repulsion_strength,
            spring_strength: logseq.physics.spring_strength,
            repulsion_distance: logseq.physics.repulsion_distance,
            mass_scale: logseq.physics.mass_scale,
            boundary_damping: logseq.physics.boundary_damping,
        };
        
        // Update multi-graph physics
        old.visualisation.graphs.logseq.physics = old.visualisation.physics.clone();
        
        // Copy other settings...
        // (Similar conversion for nodes, edges, labels, etc.)
        
        old
    }
}