/**
 * Settings Configuration - Reorganized for Intuitive Navigation
 * Grouped by function and typical user workflow
 */

import type { SectionConfig } from './types';

export const SETTINGS_CONFIG: Record<string, SectionConfig> = {
  // 1. APPEARANCE - Visual styling of nodes, edges, and labels
  appearance: {
    title: 'Appearance',
    fields: [
      // Node Appearance
      { key: 'nodeColor', label: 'Node Color', type: 'color', path: 'visualisation.graphs.logseq.nodes.baseColor' },
      { key: 'nodeSize', label: 'Node Size', type: 'slider', min: 0.2, max: 2, path: 'visualisation.graphs.logseq.nodes.nodeSize' },
      { key: 'nodeOpacity', label: 'Node Opacity', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.nodes.opacity' },
      { key: 'nodeMetalness', label: 'Node Metalness', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.nodes.metalness' },
      { key: 'nodeRoughness', label: 'Node Roughness', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.nodes.roughness' },

      // Edge Appearance
      { key: 'edgeColor', label: 'Edge Color', type: 'color', path: 'visualisation.graphs.logseq.edges.color' },
      { key: 'edgeWidth', label: 'Edge Width', type: 'slider', min: 0.01, max: 2, path: 'visualisation.graphs.logseq.edges.baseWidth' },
      { key: 'edgeOpacity', label: 'Edge Opacity', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.edges.opacity' },
      { key: 'enableArrows', label: 'Show Arrows', type: 'toggle', path: 'visualisation.graphs.logseq.edges.enableArrows' },
      { key: 'arrowSize', label: 'Arrow Size', type: 'slider', min: 0.01, max: 0.5, path: 'visualisation.graphs.logseq.edges.arrowSize' },
      { key: 'useGradient', label: 'Edge Gradient', type: 'toggle', path: 'visualisation.graphs.logseq.edges.useGradient' },

      // Label Appearance
      { key: 'enableLabels', label: 'Show Labels', type: 'toggle', path: 'visualisation.graphs.logseq.labels.enableLabels' },
      { key: 'labelSize', label: 'Label Size', type: 'slider', min: 0.01, max: 1.5, path: 'visualisation.graphs.logseq.labels.desktopFontSize' },
      { key: 'labelColor', label: 'Label Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textColor' },
      { key: 'labelOutlineColor', label: 'Outline Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textOutlineColor' },
      { key: 'labelOutlineWidth', label: 'Outline Width', type: 'slider', min: 0, max: 0.01, path: 'visualisation.graphs.logseq.labels.textOutlineWidth' }
    ]
  },

  // 2. EFFECTS - Visual effects and animations
  effects: {
    title: 'Visual Effects',
    fields: [
      // Glow Effects
      { key: 'glow', label: 'Hologram Glow', type: 'toggle', path: 'visualisation.glow.enabled' },
      { key: 'glowIntensity', label: 'Glow Intensity', type: 'slider', min: 0, max: 5, step: 0.1, path: 'visualisation.glow.intensity' },
      { key: 'glowRadius', label: 'Glow Radius', type: 'slider', min: 0, max: 5, step: 0.05, path: 'visualisation.glow.radius' },
      { key: 'glowThreshold', label: 'Glow Threshold', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.glow.threshold' },
      { key: 'edgeGlowStrength', label: 'Edge Glow', type: 'slider', min: 0, max: 5, path: 'visualisation.graphs.logseq.edges.glowStrength' },

      // Hologram Effect
      { key: 'hologram', label: 'Hologram Effect', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableHologram' },
      { key: 'ringCount', label: 'Ring Count', type: 'slider', min: 0, max: 10, path: 'visualisation.hologram.ringCount' },
      { key: 'ringColor', label: 'Ring Color', type: 'color', path: 'visualisation.hologram.ringColor' },
      { key: 'ringOpacity', label: 'Ring Opacity', type: 'slider', min: 0, max: 1, path: 'visualisation.hologram.ringOpacity' },
      { key: 'ringRotationSpeed', label: 'Ring Speed', type: 'slider', min: 0, max: 5, path: 'visualisation.hologram.ringRotationSpeed' },

      // Flow Effect
      { key: 'flowEffect', label: 'Edge Flow', type: 'toggle', path: 'visualisation.graphs.logseq.edges.enableFlowEffect' },
      { key: 'flowSpeed', label: 'Flow Speed', type: 'slider', min: 0.1, max: 5, path: 'visualisation.graphs.logseq.edges.flowSpeed' },
      { key: 'flowIntensity', label: 'Flow Intensity', type: 'slider', min: 0, max: 10, path: 'visualisation.graphs.logseq.edges.flowIntensity' },
      { key: 'distanceIntensity', label: 'Distance Intensity', type: 'slider', min: 0, max: 10, path: 'visualisation.graphs.logseq.edges.distanceIntensity' },

      // Animations
      { key: 'nodeAnimations', label: 'Node Animations', type: 'toggle', path: 'visualisation.animations.enableNodeAnimations' },
      { key: 'pulseEnabled', label: 'Pulse Effect', type: 'toggle', path: 'visualisation.animations.pulseEnabled' },
      { key: 'pulseSpeed', label: 'Pulse Speed', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animations.pulseSpeed' },
      { key: 'pulseStrength', label: 'Pulse Strength', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animations.pulseStrength' },
      { key: 'selectionWave', label: 'Selection Wave', type: 'toggle', path: 'visualisation.animations.selectionWaveEnabled' },
      { key: 'waveSpeed', label: 'Wave Speed', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animations.waveSpeed' }
    ]
  },

  // 3. PHYSICS BASIC - Common physics parameters users typically adjust
  physics: {
    title: 'Physics - Basic',
    fields: [
      { key: 'enabled', label: 'Physics Enabled', type: 'toggle', path: 'visualisation.graphs.logseq.physics.enabled' },
      { key: 'autoBalance', label: '⚖️ Adaptive Balancing', type: 'toggle', path: 'visualisation.graphs.logseq.physics.autoBalance' },
      { key: 'damping', label: 'Damping', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.damping' },
      { key: 'springK', label: 'Spring Strength', type: 'slider', min: 0.0001, max: 10, path: 'visualisation.graphs.logseq.physics.springK' },
      { key: 'repelK', label: 'Repulsion Strength', type: 'slider', min: 0.1, max: 200, path: 'visualisation.graphs.logseq.physics.repelK' },
      { key: 'attractionK', label: 'Attraction Strength', type: 'slider', min: 0, max: 10, path: 'visualisation.graphs.logseq.physics.attractionK' },
      { key: 'maxVelocity', label: 'Max Velocity', type: 'slider', min: 0.1, max: 10, path: 'visualisation.graphs.logseq.physics.maxVelocity' },
      { key: 'separationRadius', label: 'Separation Radius', type: 'slider', min: 0.1, max: 10, path: 'visualisation.graphs.logseq.physics.separationRadius' },
      { key: 'restLength', label: 'Rest Length', type: 'slider', min: 10, max: 200, path: 'visualisation.graphs.logseq.physics.restLength' },
      { key: 'centerGravityK', label: 'Centre Gravity', type: 'slider', min: 0, max: 0.1, path: 'visualisation.graphs.logseq.physics.centerGravityK' },
      { key: 'enableBounds', label: 'Enable Bounds', type: 'toggle', path: 'visualisation.graphs.logseq.physics.enableBounds' },
      { key: 'boundsSize', label: 'Bounds Size', type: 'slider', min: 1, max: 10000, path: 'visualisation.graphs.logseq.physics.boundsSize' }
    ]
  },

  // 4. PHYSICS ADVANCED - Technical parameters for power users
  physicsAdvanced: {
    title: 'Physics - Advanced',
    fields: [
      { key: 'dt', label: 'Time Step (dt)', type: 'slider', min: 0.001, max: 0.1, path: 'visualisation.graphs.logseq.physics.dt' },
      { key: 'iterations', label: 'Iterations', type: 'slider', min: 1, max: 1000, path: 'visualisation.graphs.logseq.physics.iterations' },
      { key: 'warmupIterations', label: 'Warmup Iterations', type: 'slider', min: 0, max: 500, path: 'visualisation.graphs.logseq.physics.warmupIterations' },
      { key: 'coolingRate', label: 'Cooling Rate', type: 'slider', min: 0.00001, max: 0.01, path: 'visualisation.graphs.logseq.physics.coolingRate' },
      { key: 'updateThreshold', label: 'Update Threshold', type: 'slider', min: 0, max: 0.5, path: 'visualisation.graphs.logseq.physics.updateThreshold' },
      { key: 'stressWeight', label: 'Stress Weight', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.stressWeight' },
      { key: 'stressAlpha', label: 'Stress Alpha', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.stressAlpha' },
      { key: 'minDistance', label: 'Min Distance', type: 'slider', min: 0.05, max: 1, path: 'visualisation.graphs.logseq.physics.minDistance' },
      { key: 'maxRepulsionDist', label: 'Max Repulsion Distance', type: 'slider', min: 10, max: 200, path: 'visualisation.graphs.logseq.physics.maxRepulsionDist' },
      { key: 'repulsionCutoff', label: 'Repulsion Cutoff', type: 'slider', min: 10, max: 200, path: 'visualisation.graphs.logseq.physics.repulsionCutoff' },
      { key: 'repulsionSofteningEpsilon', label: 'Repulsion Epsilon', type: 'slider', min: 0.00001, max: 0.01, path: 'visualisation.graphs.logseq.physics.repulsionSofteningEpsilon' },
      { key: 'gridCellSize', label: 'Grid Cell Size', type: 'slider', min: 10, max: 100, path: 'visualisation.graphs.logseq.physics.gridCellSize' },
      { key: 'massScale', label: 'Mass Scale', type: 'slider', min: 0.1, max: 10, path: 'visualisation.graphs.logseq.physics.massScale' },
      { key: 'boundaryDamping', label: 'Boundary Damping', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.boundaryDamping' },
      { key: 'boundaryExtremeMultiplier', label: 'Boundary Extreme Multiplier', type: 'slider', min: 1, max: 5, path: 'visualisation.graphs.logseq.physics.boundaryExtremeMultiplier' },
      { key: 'boundaryExtremeForceMultiplier', label: 'Boundary Force Multiplier', type: 'slider', min: 1, max: 20, path: 'visualisation.graphs.logseq.physics.boundaryExtremeForceMultiplier' },
      { key: 'boundaryVelocityDamping', label: 'Boundary Velocity Damping', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.boundaryVelocityDamping' }
    ]
  },

  // 5. RENDERING - Lighting and rendering quality
  rendering: {
    title: 'Lighting & Rendering',
    fields: [
      // Lighting
      { key: 'ambientLight', label: 'Ambient Light', type: 'slider', min: 0, max: 2, path: 'visualisation.rendering.ambientLightIntensity' },
      { key: 'directionalLight', label: 'Directional Light', type: 'slider', min: 0, max: 2, path: 'visualisation.rendering.directionalLightIntensity' },

      // Rendering Quality
      { key: 'antialiasing', label: 'Antialiasing', type: 'toggle', path: 'visualisation.rendering.enableAntialiasing' },
      { key: 'shadows', label: 'Shadows', type: 'toggle', path: 'visualisation.rendering.enableShadows' },
      { key: 'ambientOcclusion', label: 'Ambient Occlusion', type: 'toggle', path: 'visualisation.rendering.enableAmbientOcclusion' },

      // Advanced Rendering
      { key: 'enableInstancing', label: 'GPU Instancing', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableInstancing' },
      { key: 'enableMetadataShape', label: 'Metadata Shape', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataShape' },
      { key: 'enableMetadataVis', label: 'Metadata Visualisation', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataVisualisation' },
      { key: 'nodeImportance', label: 'Node Importance', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableImportance' }
    ]
  },

  // 6. XR/IMMERSIVE - VR/AR and immersive mode settings
  xr: {
    title: 'XR / Immersive',
    fields: [
      { key: 'xrEnabled', label: 'XR Mode', type: 'toggle', path: 'xr.enabled' },
      { key: 'xrQuality', label: 'XR Quality', type: 'select', options: ['Low', 'Medium', 'High'], path: 'xr.quality' },
      { key: 'xrRenderScale', label: 'XR Render Scale', type: 'slider', min: 0.5, max: 2, path: 'xr.renderScale' },
      { key: 'xrAdaptiveQuality', label: 'Adaptive Quality', type: 'toggle', path: 'xr.enableAdaptiveQuality' },
      { key: 'xrPerformancePreset', label: 'Performance Preset', type: 'select', options: ['Battery Saver', 'Balanced', 'Performance'], path: 'xr.performance.preset' },
      { key: 'handTracking', label: 'Hand Tracking', type: 'toggle', path: 'xr.handTracking.enabled' },
      { key: 'enableHaptics', label: 'Haptic Feedback', type: 'toggle', path: 'xr.interactions.enableHaptics' },
      { key: 'xrComputeMode', label: 'Optimized Compute', type: 'toggle', path: 'xr.gpu.enableOptimizedCompute' }
    ]
  },

  // 7. SYSTEM - System and connection settings
  system: {
    title: 'System Settings',
    fields: [
      { key: 'persistSettings', label: 'Persist Settings on Server', type: 'toggle', path: 'system.persistSettingsOnServer' },
      { key: 'customBackendURL', label: 'Custom Backend URL', type: 'text', path: 'system.customBackendUrl' },
      { key: 'enableDebug', label: 'Debug Mode', type: 'toggle', path: 'system.debug.enabled' }
    ]
  },

  // 8. AUTHENTICATION - Login and authentication
  auth: {
    title: 'Authentication',
    fields: [
      { key: 'nostr', label: 'Nostr Login', type: 'nostr-button', path: 'auth.nostr' },
      { key: 'enabled', label: 'Enable Authentication', type: 'toggle', path: 'auth.enabled' },
      { key: 'required', label: 'Require Authentication', type: 'toggle', path: 'auth.required' },
      { key: 'provider', label: 'Auth Provider', type: 'text', path: 'auth.provider' }
    ]
  }
};
