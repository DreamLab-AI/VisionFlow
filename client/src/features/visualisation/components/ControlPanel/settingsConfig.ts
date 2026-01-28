

import type { SectionConfig } from './types';

export const SETTINGS_CONFIG: Record<string, SectionConfig> = {
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  visualization: {
    title: 'Visualization Settings',
    fields: [
      
      { key: 'nodeColor', label: 'Node Color', type: 'color', path: 'visualisation.graphs.logseq.nodes.baseColor' },
      { key: 'nodeSize', label: 'Node Size', type: 'slider', min: 0.2, max: 2, path: 'visualisation.graphs.logseq.nodes.nodeSize' },
      { key: 'nodeMetalness', label: 'Node Metalness', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.nodes.metalness' },
      { key: 'nodeOpacity', label: 'Node Opacity', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.nodes.opacity' },
      { key: 'nodeRoughness', label: 'Node Roughness', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.nodes.roughness' },
      { key: 'enableInstancing', label: 'Enable Instancing', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableInstancing' },
      { key: 'enableMetadataShape', label: 'Metadata Shape', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataShape' },
      { key: 'enableMetadataVis', label: 'Metadata Visual', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataVisualisation' },
      { key: 'nodeImportance', label: 'Node Importance', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableImportance' },

      
      { key: 'edgeColor', label: 'Edge Color', type: 'color', path: 'visualisation.graphs.logseq.edges.color' },
      { key: 'edgeWidth', label: 'Edge Width', type: 'slider', min: 0.01, max: 2, path: 'visualisation.graphs.logseq.edges.baseWidth' },
      { key: 'edgeOpacity', label: 'Edge Opacity', type: 'slider', min: 0, max: 1, path: 'visualisation.graphs.logseq.edges.opacity' },
      { key: 'enableArrows', label: 'Show Arrows', type: 'toggle', path: 'visualisation.graphs.logseq.edges.enableArrows' },
      { key: 'arrowSize', label: 'Arrow Size', type: 'slider', min: 0.01, max: 0.5, path: 'visualisation.graphs.logseq.edges.arrowSize' },
      { key: 'glowStrength', label: 'Edge Glow', type: 'slider', min: 0, max: 5, path: 'visualisation.graphs.logseq.edges.glowStrength' },

      
      { key: 'enableLabels', label: 'Show Labels', type: 'toggle', path: 'visualisation.graphs.logseq.labels.enableLabels' },
      { key: 'labelSize', label: 'Label Size', type: 'slider', min: 0.01, max: 1.5, path: 'visualisation.graphs.logseq.labels.desktopFontSize' },
      { key: 'labelColor', label: 'Label Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textColor' },
      { key: 'labelOutlineColor', label: 'Outline Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textOutlineColor' },
      { key: 'labelOutlineWidth', label: 'Outline Width', type: 'slider', min: 0, max: 0.01, path: 'visualisation.graphs.logseq.labels.textOutlineWidth' },
      { key: 'labelDistanceThreshold', label: 'Label Distance', type: 'slider', min: 50, max: 2000, path: 'visualisation.graphs.logseq.labels.labelDistanceThreshold' },


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

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  integrations: {
    title: 'Visual Effects',
    fields: [
      { key: 'glow', label: 'Hologram Glow', type: 'toggle', path: 'visualisation.glow.enabled' },
      { key: 'glowIntensity', label: 'Glow Intensity', type: 'slider', min: 0, max: 5, step: 0.1, path: 'visualisation.glow.intensity' },
      { key: 'glowRadius', label: 'Glow Radius', type: 'slider', min: 0, max: 5, step: 0.05, path: 'visualisation.glow.radius' },
      { key: 'glowThreshold', label: 'Glow Threshold', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.glow.threshold' },
      
      
      
      
      
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
      { key: 'nodeAnimations', label: 'Node Animations', type: 'toggle', path: 'visualisation.animations.enableNodeAnimations' },
      { key: 'pulseEnabled', label: 'Pulse Effect', type: 'toggle', path: 'visualisation.animations.pulseEnabled' },
      { key: 'pulseSpeed', label: 'Pulse Speed', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animations.pulseSpeed' },
      { key: 'pulseStrength', label: 'Pulse Strength', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animations.pulseStrength' },
      { key: 'selectionWave', label: 'Selection Wave', type: 'toggle', path: 'visualisation.animations.selectionWaveEnabled' },
      { key: 'waveSpeed', label: 'Wave Speed', type: 'slider', min: 0.1, max: 2, path: 'visualisation.animations.waveSpeed' },
      { key: 'antialiasing', label: 'Antialiasing', type: 'toggle', path: 'visualisation.rendering.enableAntialiasing' },
      { key: 'shadows', label: 'Shadows', type: 'toggle', path: 'visualisation.rendering.enableShadows' },
      { key: 'ambientOcclusion', label: 'Ambient Occl', type: 'toggle', path: 'visualisation.rendering.enableAmbientOcclusion' }
    ]
  },

  developer: {
    title: 'Developer Tools',
    fields: [
      
      
      
      
      
      
      
      { key: 'enableDebug', label: 'Debug Mode', type: 'toggle', path: 'system.debug.enabled' },
      
      
      
      
      
      
      
      
      
      
      
      
      
      
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

  'quality-gates': {
    title: 'Quality Gates',
    fields: [
      // Compute Mode
      { key: 'gpuAcceleration', label: 'GPU Acceleration', type: 'toggle', path: 'qualityGates.gpuAcceleration' },
      { key: 'autoAdjust', label: 'Auto-Adjust Quality', type: 'toggle', path: 'qualityGates.autoAdjust' },
      { key: 'minFpsThreshold', label: 'Min FPS Threshold', type: 'slider', min: 15, max: 60, step: 5, path: 'qualityGates.minFpsThreshold' },
      { key: 'maxNodeCount', label: 'Max Node Count', type: 'slider', min: 1000, max: 500000, step: 5000, path: 'qualityGates.maxNodeCount' },
      // Physics Features
      { key: 'ontologyPhysics', label: 'Ontology Physics', type: 'toggle', path: 'qualityGates.ontologyPhysics' },
      { key: 'semanticForces', label: 'Semantic Forces', type: 'toggle', path: 'qualityGates.semanticForces' },
      { key: 'layoutMode', label: 'Layout Mode', type: 'select', options: ['force-directed', 'dag-topdown', 'dag-radial', 'dag-leftright', 'type-clustering'], path: 'qualityGates.layoutMode' },
      { key: 'gnnPhysics', label: 'GNN-Enhanced Physics', type: 'toggle', path: 'qualityGates.gnnPhysics' },
      // Analytics Visualization
      { key: 'showClusters', label: 'Show Clusters', type: 'toggle', path: 'qualityGates.showClusters' },
      { key: 'showAnomalies', label: 'Show Anomalies', type: 'toggle', path: 'qualityGates.showAnomalies' },
      { key: 'showCommunities', label: 'Show Communities', type: 'toggle', path: 'qualityGates.showCommunities' },
      // Advanced Features
      { key: 'ruvectorEnabled', label: 'RuVector Integration', type: 'toggle', path: 'qualityGates.ruvectorEnabled' }
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
