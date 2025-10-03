/**
 * Settings Configuration for all tabs
 * Extracted from original IntegratedControlPanel
 */

import type { SectionConfig } from './types';

export const SETTINGS_CONFIG: Record<string, SectionConfig> = {
  dashboard: {
    title: 'Dashboard',
    fields: [
      { key: 'graphStatus', label: 'Show Graph Status', type: 'toggle', path: 'dashboard.showStatus' },
      { key: 'autoRefresh', label: 'Auto Refresh', type: 'toggle', path: 'dashboard.autoRefresh' },
      { key: 'refreshInterval', label: 'Refresh Interval (s)', type: 'slider', min: 1, max: 60, path: 'dashboard.refreshInterval' },
      { key: 'computeMode', label: 'Compute Mode', type: 'select', options: ['Basic Force-Directed', 'Dual Graph', 'Constraint-Enhanced', 'Visual Analytics'], path: 'dashboard.computeMode' },
      { key: 'iterationCount', label: 'Iteration Count', type: 'text', path: 'dashboard.iterationCount' },
      { key: 'convergenceIndicator', label: 'Show Convergence', type: 'toggle', path: 'dashboard.showConvergence' },
      { key: 'activeConstraints', label: 'Active Constraints', type: 'text', path: 'dashboard.activeConstraints' },
      { key: 'clusteringStatus', label: 'Clustering Active', type: 'toggle', path: 'dashboard.clusteringActive' }
    ]
  },

  visualization: {
    title: 'Visualization Settings',
    fields: [
      // Node Settings
      { key: 'nodeColor', label: 'Node Color', type: 'color', path: 'visualisation.graphs.logseq.nodes.baseColor' },
      { key: 'nodeSize', label: 'Node Size', type: 'slider', min: 0.2, max: 2, path: 'visualisation.graphs.logseq.nodes.nodeSize' },
      { key: 'nodeMetalness', label: 'Node Metalness', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.nodes.metalness' },
      { key: 'nodeOpacity', label: 'Node Opacity', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.nodes.opacity' },
      { key: 'nodeRoughness', label: 'Node Roughness', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.nodes.roughness' },
      { key: 'enableInstancing', label: 'Enable Instancing', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableInstancing' },
      { key: 'enableMetadataShape', label: 'Metadata Shape', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataShape' },
      { key: 'enableMetadataVis', label: 'Metadata Visual', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataVisualisation' },
      { key: 'nodeImportance', label: 'Node Importance', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableImportance' },

      // Edge Settings
      { key: 'edgeColor', label: 'Edge Color', type: 'color', path: 'visualisation.graphs.logseq.edges.color' },
      { key: 'edgeWidth', label: 'Edge Width', type: 'slider', min: 0.01, max: 2, path: 'visualisation.graphs.logseq.edges.baseWidth' },
      { key: 'edgeOpacity', label: 'Edge Opacity', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.edges.opacity' },
      { key: 'enableArrows', label: 'Show Arrows', type: 'toggle', path: 'visualisation.graphs.logseq.edges.enableArrows' },
      { key: 'arrowSize', label: 'Arrow Size', type: 'slider', min: 0.01, max: 0.5, path: 'visualisation.graphs.logseq.edges.arrowSize' },
      { key: 'glowStrength', label: 'Edge Glow', type: 'slider', min: 0, max: 5, path: 'visualisation.graphs.logseq.edges.glowStrength' },

      // Label Settings
      { key: 'enableLabels', label: 'Show Labels', type: 'toggle', path: 'visualisation.graphs.logseq.labels.enableLabels' },
      { key: 'labelSize', label: 'Label Size', type: 'slider', min: 0.01, max: 1.5, path: 'visualisation.graphs.logseq.labels.desktopFontSize' },
      { key: 'labelColor', label: 'Label Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textColor' },
      { key: 'labelOutlineColor', label: 'Outline Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textOutlineColor' },
      { key: 'labelOutlineWidth', label: 'Outline Width', type: 'slider', min: 0, max: 0.01, path: 'visualisation.graphs.logseq.labels.textOutlineWidth' },

      // GPU Visualization Features
      { key: 'temporalCoherence', label: 'Temporal Coherence', type: 'slider', min: 0, max: 1, path: 'visualisation.gpu.temporalCoherence' },
      { key: 'graphDifferentiation', label: 'Graph Differentiation', type: 'toggle', path: 'visualisation.gpu.enableGraphDifferentiation' },
      { key: 'clusterVisualization', label: 'Cluster Visualization', type: 'toggle', path: 'visualisation.gpu.enableClusterVisualization' },
      { key: 'stressOptimization', label: 'Stress Optimization', type: 'toggle', path: 'visualisation.gpu.enableStressOptimization' },

      // Lighting
      { key: 'ambientLight', label: 'Ambient Light', type: 'slider', min: 0, max: 2, path: 'visualisation.rendering.ambientLightIntensity' },
      { key: 'directionalLight', label: 'Direct Light', type: 'slider', min: 0, max: 2, path: 'visualisation.rendering.directionalLightIntensity' }
    ]
  },

  physics: {
    title: 'Physics Settings',
    fields: [
      { key: 'enabled', label: 'Physics Enabled', type: 'toggle', path: 'visualisation.graphs.logseq.physics.enabled' },
      { key: 'autoBalance', label: '⚖️ Adaptive Balancing', type: 'toggle', path: 'visualisation.graphs.logseq.physics.autoBalance' },
      { key: 'damping', label: 'Damping', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.damping' },
      { key: 'springK', label: 'Spring Strength (k)', type: 'slider', min: 0.0001, max: 10, path: 'visualisation.graphs.logseq.physics.springK' },
      { key: 'repelK', label: 'Repulsion Strength (k)', type: 'slider', min: 0.1, max: 200, path: 'visualisation.graphs.logseq.physics.repelK' },
      { key: 'attractionK', label: 'Attraction Strength (k)', type: 'slider', min: 0, max: 10, path: 'visualisation.graphs.logseq.physics.attractionK' },
      { key: 'dt', label: 'Time Step (dt)', type: 'slider', min: 0.001, max: 0.1, path: 'visualisation.graphs.logseq.physics.dt' },
      { key: 'maxVelocity', label: 'Max Velocity', type: 'slider', min: 0.1, max: 10, path: 'visualisation.graphs.logseq.physics.maxVelocity' },
      { key: 'separationRadius', label: 'Separation Radius', type: 'slider', min: 0.1, max: 10, path: 'visualisation.graphs.logseq.physics.separationRadius' },
      { key: 'enableBounds', label: 'Enable Bounds', type: 'toggle', path: 'visualisation.graphs.logseq.physics.enableBounds' },
      { key: 'boundsSize', label: 'Bounds Size', type: 'slider', min: 1, max: 10000, path: 'visualisation.graphs.logseq.physics.boundsSize' },
      { key: 'stressWeight', label: 'Stress Weight', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.stressWeight' },
      { key: 'stressAlpha', label: 'Stress Alpha', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.stressAlpha' },
      { key: 'minDistance', label: 'Min Distance', type: 'slider', min: 0.05, max: 1, path: 'visualisation.graphs.logseq.physics.minDistance' },
      { key: 'maxRepulsionDist', label: 'Max Repulsion Dist', type: 'slider', min: 10, max: 200, path: 'visualisation.graphs.logseq.physics.maxRepulsionDist' },
      { key: 'warmupIterations', label: 'Warmup Iterations', type: 'slider', min: 0, max: 500, path: 'visualisation.graphs.logseq.physics.warmupIterations' },
      { key: 'coolingRate', label: 'Cooling Rate', type: 'slider', min: 0.00001, max: 0.01, path: 'visualisation.graphs.logseq.physics.coolingRate' },
      { key: 'restLength', label: 'Rest Length', type: 'slider', min: 10, max: 200, path: 'visualisation.graphs.logseq.physics.restLength' },
      { key: 'repulsionCutoff', label: 'Repulsion Cutoff', type: 'slider', min: 10, max: 200, path: 'visualisation.graphs.logseq.physics.repulsionCutoff' },
      { key: 'repulsionSofteningEpsilon', label: 'Repulsion Epsilon', type: 'slider', min: 0.00001, max: 0.01, path: 'visualisation.graphs.logseq.physics.repulsionSofteningEpsilon' },
      { key: 'centerGravityK', label: 'Centre Gravity K', type: 'slider', min: 0, max: 0.1, path: 'visualisation.graphs.logseq.physics.centerGravityK' },
      { key: 'gridCellSize', label: 'Grid Cell Size', type: 'slider', min: 10, max: 100, path: 'visualisation.graphs.logseq.physics.gridCellSize' },
      { key: 'boundaryExtremeMultiplier', label: 'Boundary Extreme Mult', type: 'slider', min: 1, max: 5, path: 'visualisation.graphs.logseq.physics.boundaryExtremeMultiplier' },
      { key: 'boundaryExtremeForceMultiplier', label: 'Boundary Force Mult', type: 'slider', min: 1, max: 20, path: 'visualisation.graphs.logseq.physics.boundaryExtremeForceMultiplier' },
      { key: 'boundaryVelocityDamping', label: 'Boundary Vel Damping', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.boundaryVelocityDamping' },
      { key: 'iterations', label: 'Iterations', type: 'slider', min: 1, max: 1000, path: 'visualisation.graphs.logseq.physics.iterations' },
      { key: 'massScale', label: 'Mass Scale', type: 'slider', min: 0.1, max: 10, path: 'visualisation.graphs.logseq.physics.massScale' },
      { key: 'boundaryDamping', label: 'Boundary Damp', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.physics.boundaryDamping' },
      { key: 'updateThreshold', label: 'Update Threshold', type: 'slider', min: 0, max: 0.5, path: 'visualisation.graphs.logseq.physics.updateThreshold' }
    ]
  },

  analytics: {
    title: 'Analytics Settings',
    fields: [
      { key: 'enableMetrics', label: 'Enable Metrics', type: 'toggle', path: 'analytics.enableMetrics' },
      { key: 'updateInterval', label: 'Update Interval (s)', type: 'slider', min: 1, max: 60, path: 'analytics.updateInterval' },
      { key: 'showDegreeDistribution', label: 'Degree Distribution', type: 'toggle', path: 'analytics.showDegreeDistribution' },
      { key: 'showClustering', label: 'Clustering Coefficient', type: 'toggle', path: 'analytics.showClusteringCoefficient' },
      { key: 'showCentrality', label: 'Centrality Metrics', type: 'toggle', path: 'analytics.showCentrality' },
      { key: 'clusteringAlgorithm', label: 'Clustering Algorithm', type: 'select', options: ['none', 'kmeans', 'spectral', 'louvain'], path: 'analytics.clustering.algorithm' },
      { key: 'clusterCount', label: 'Cluster Count', type: 'slider', min: 2, max: 20, path: 'analytics.clustering.clusterCount' },
      { key: 'clusterResolution', label: 'Resolution', type: 'slider', min: 0.1, max: 2, path: 'analytics.clustering.resolution' },
      { key: 'clusterIterations', label: 'Cluster Iterations', type: 'slider', min: 10, max: 100, path: 'analytics.clustering.iterations' },
      { key: 'exportClusters', label: 'Export Clusters', type: 'toggle', path: 'analytics.clustering.exportEnabled' },
      { key: 'importDistances', label: 'Import Distances', type: 'toggle', path: 'analytics.clustering.importEnabled' }
    ]
  },

  performance: {
    title: 'Performance Settings',
    fields: [
      { key: 'showFPS', label: 'Show FPS', type: 'toggle', path: 'performance.showFPS' },
      { key: 'targetFPS', label: 'Target FPS', type: 'slider', min: 30, max: 144, path: 'performance.targetFPS' },
      { key: 'gpuMemoryLimit', label: 'GPU Memory (MB)', type: 'slider', min: 256, max: 8192, step: 256, path: 'performance.gpuMemoryLimit' },
      { key: 'levelOfDetail', label: 'Quality Preset', type: 'select', options: ['low', 'medium', 'high', 'ultra'], path: 'performance.levelOfDetail' },
      { key: 'adaptiveQuality', label: 'Adaptive Quality', type: 'toggle', path: 'performance.enableAdaptiveQuality' },
      { key: 'warmupDuration', label: 'Warmup Duration (s)', type: 'slider', min: 0, max: 10, path: 'performance.warmupDuration' },
      { key: 'convergenceThreshold', label: 'Convergence Threshold', type: 'slider', min: 0.001, max: 0.1, path: 'performance.convergenceThreshold' },
      { key: 'adaptiveCooling', label: 'Adaptive Cooling', type: 'toggle', path: 'performance.enableAdaptiveCooling' },
      { key: 'gpuBlockSize', label: 'GPU Block Size', type: 'select', options: ['64', '128', '256', '512'], path: 'performance.gpuBlockSize' },
      { key: 'memoryCoalescing', label: 'Memory Coalescing', type: 'toggle', path: 'performance.enableMemoryCoalescing' },
      { key: 'iterationLimit', label: 'Iteration Limit', type: 'slider', min: 100, max: 5000, path: 'performance.iterationLimit' }
    ]
  },

  integrations: {
    title: 'Visual Effects',
    fields: [
      { key: 'glow', label: 'Hologram Glow', type: 'toggle', path: 'visualisation.glow.enabled' },
      { key: 'glowIntensity', label: 'Glow Intensity', type: 'slider', min: 0, max: 5, step: 0.1, path: 'visualisation.glow.intensity' },
      { key: 'glowRadius', label: 'Glow Radius', type: 'slider', min: 0, max: 5, step: 0.05, path: 'visualisation.glow.radius' },
      { key: 'glowThreshold', label: 'Glow Threshold', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.glow.threshold' },
      { key: 'bloom', label: 'Graph Bloom', type: 'toggle', path: 'visualisation.bloom.enabled' },
      { key: 'bloomStrength', label: 'Bloom Strength', type: 'slider', min: 0, max: 5, step: 0.1, path: 'visualisation.bloom.strength' },
      { key: 'bloomRadius', label: 'Bloom Radius', type: 'slider', min: 0, max: 1, step: 0.05, path: 'visualisation.bloom.radius' },
      { key: 'bloomThreshold', label: 'Bloom Threshold', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.bloom.threshold' },
      { key: 'hologram', label: 'Hologram', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableHologram' },
      { key: 'ringCount', label: 'Ring Count', type: 'slider', min: 0, max: 10, path: 'visualisation.hologram.ringCount' },
      { key: 'ringColor', label: 'Ring Color', type: 'color', path: 'visualisation.hologram.ringColor' },
      { key: 'ringOpacity', label: 'Ring Opacity', type: 'slider', min: 0, max: 1, path: 'visualisation.hologram.ringOpacity' },
      { key: 'ringRotationSpeed', label: 'Ring Speed', type: 'slider', min: 0, max: 5, path: 'visualisation.hologram.ringRotationSpeed' },
      { key: 'flowEffect', label: 'Edge Flow', type: 'toggle', path: 'visualisation.graphs.logseq.edges.enableFlowEffect' },
      { key: 'flowSpeed', label: 'Flow Speed', type: 'slider', min: 0.1, max: 5, path: 'visualisation.graphs.logseq.edges.flowSpeed' },
      { key: 'flowIntensity', label: 'Flow Intensity', type: 'slider', min: 0, max: 10, path: 'visualisation.graphs.logseq.edges.flowIntensity' },
      { key: 'useGradient', label: 'Edge Gradient', type: 'toggle', path: 'visualisation.graphs.logseq.edges.useGradient' },
      { key: 'distanceIntensity', label: 'Distance Int', type: 'slider', min: 0, max: 10, path: 'visualisation.graphs.logseq.edges.distanceIntensity' },
      { key: 'nodeAnimations', label: 'Node Animations', type: 'toggle', path: 'visualisation.animation.enableNodeAnimations' },
      { key: 'pulseEnabled', label: 'Pulse Effect', type: 'toggle', path: 'visualisation.animation.pulseEnabled' },
      { key: 'pulseSpeed', label: 'Pulse Speed', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animation.pulseSpeed' },
      { key: 'pulseStrength', label: 'Pulse Strength', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animation.pulseStrength' },
      { key: 'selectionWave', label: 'Selection Wave', type: 'toggle', path: 'visualisation.animation.enableSelectionWave' },
      { key: 'waveSpeed', label: 'Wave Speed', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animation.waveSpeed' },
      { key: 'antialiasing', label: 'Antialiasing', type: 'toggle', path: 'visualisation.rendering.enableAntialiasing' },
      { key: 'shadows', label: 'Shadows', type: 'toggle', path: 'visualisation.rendering.enableShadows' },
      { key: 'ambientOcclusion', label: 'Ambient Occl', type: 'toggle', path: 'visualisation.rendering.enableAmbientOcclusion' }
    ]
  },

  developer: {
    title: 'Developer Tools',
    fields: [
      { key: 'consoleLogging', label: 'Console Logging', type: 'toggle', path: 'developer.consoleLogging' },
      { key: 'logLevel', label: 'Log Level', type: 'select', options: ['error', 'warn', 'info', 'debug'], path: 'developer.logLevel' },
      { key: 'showNodeIds', label: 'Show Node IDs', type: 'toggle', path: 'developer.showNodeIds' },
      { key: 'showEdgeWeights', label: 'Show Edge Weights', type: 'toggle', path: 'developer.showEdgeWeights' },
      { key: 'enableProfiler', label: 'Enable Profiler', type: 'toggle', path: 'developer.enableProfiler' },
      { key: 'apiDebugMode', label: 'API Debug Mode', type: 'toggle', path: 'developer.apiDebugMode' },
      { key: 'enableDebug', label: 'Debug Mode', type: 'toggle', path: 'system.debug.enabled' },
      { key: 'showMemory', label: 'Show Memory', type: 'toggle', path: 'system.debug.showMemory' },
      { key: 'perfDebug', label: 'Performance Debug', type: 'toggle', path: 'system.debug.enablePerformanceDebug' },
      { key: 'telemetry', label: 'Telemetry', type: 'toggle', path: 'system.debug.enableTelemetry' },
      { key: 'dataDebug', label: 'Data Debug', type: 'toggle', path: 'system.debug.enableDataDebug' },
      { key: 'wsDebug', label: 'WebSocket Debug', type: 'toggle', path: 'system.debug.enableWebSocketDebug' },
      { key: 'physicsDebug', label: 'Physics Debug', type: 'toggle', path: 'system.debug.enablePhysicsDebug' },
      { key: 'nodeDebug', label: 'Node Debug', type: 'toggle', path: 'system.debug.enableNodeDebug' },
      { key: 'shaderDebug', label: 'Shader Debug', type: 'toggle', path: 'system.debug.enableShaderDebug' },
      { key: 'matrixDebug', label: 'Matrix Debug', type: 'toggle', path: 'system.debug.enableMatrixDebug' },
      { key: 'forceVectors', label: 'Show Force Vectors', type: 'toggle', path: 'developer.gpu.showForceVectors' },
      { key: 'constraintVisualization', label: 'Constraint Visualization', type: 'toggle', path: 'developer.gpu.showConstraints' },
      { key: 'boundaryForceDisplay', label: 'Boundary Forces', type: 'toggle', path: 'developer.gpu.showBoundaryForces' },
      { key: 'convergenceGraph', label: 'Convergence Graph', type: 'toggle', path: 'developer.gpu.showConvergenceGraph' },
      { key: 'gpuTimingStats', label: 'GPU Timing Stats', type: 'toggle', path: 'developer.gpu.showTimingStats' }
    ]
  },

  auth: {
    title: 'Authentication Settings',
    fields: [
      { key: 'nostr', label: 'Nostr Login', type: 'nostr-button', path: 'auth.nostr' },
      { key: 'enabled', label: 'Auth Required', type: 'toggle', path: 'auth.enabled' },
      { key: 'required', label: 'Require Auth', type: 'toggle', path: 'auth.required' },
      { key: 'provider', label: 'Auth Provider', type: 'text', path: 'auth.provider' }
    ]
  },

  xr: {
    title: 'XR/AR Settings',
    fields: [
      { key: 'persistSettings', label: 'Persist Settings', type: 'toggle', path: 'system.persistSettingsOnServer' },
      { key: 'customBackendURL', label: 'Custom Backend URL', type: 'text', path: 'system.customBackendUrl' },
      { key: 'xrEnabled', label: 'XR Mode', type: 'toggle', path: 'xr.enabled' },
      { key: 'xrQuality', label: 'XR Quality', type: 'select', options: ['Low', 'Medium', 'High'], path: 'xr.quality' },
      { key: 'xrRenderScale', label: 'XR Render Scale', type: 'slider', min: 0.5, max: 2, path: 'xr.renderScale' },
      { key: 'handTracking', label: 'Hand Tracking', type: 'toggle', path: 'xr.handTracking.enabled' },
      { key: 'enableHaptics', label: 'Haptics', type: 'toggle', path: 'xr.interactions.enableHaptics' },
      { key: 'xrComputeMode', label: 'XR Compute Mode', type: 'toggle', path: 'xr.gpu.enableOptimizedCompute' },
      { key: 'xrPerformancePreset', label: 'XR Performance', type: 'select', options: ['Battery Saver', 'Balanced', 'Performance'], path: 'xr.performance.preset' },
      { key: 'xrAdaptiveQuality', label: 'XR Adaptive Quality', type: 'toggle', path: 'xr.enableAdaptiveQuality' }
    ]
  }
};
