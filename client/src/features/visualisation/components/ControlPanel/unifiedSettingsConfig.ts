/**
 * Unified Settings Configuration
 *
 * Reorganized tab structure with proper basic/advanced mode gating.
 * All settings are categorized by feature area with isAdvanced flags.
 */

import type { SectionConfig } from './types';
import {
  Eye, Sparkles, BarChart3, Gauge, Settings,
  Bot, Code, Network, Shield
} from 'lucide-react';
// @ts-ignore - These icons exist in lucide-react but types may be outdated
import { Atom, Glasses } from 'lucide-react';

export interface UnifiedTabConfig {
  id: string;
  label: string;
  icon: typeof Eye;
  description: string;
  buttonKey?: string;
  isAdvanced?: boolean;
  isPowerUserOnly?: boolean;
}

// ============================================================================
// TAB DEFINITIONS - New unified structure
// ============================================================================

export const UNIFIED_TABS: UnifiedTabConfig[] = [
  // Basic Tabs (always visible)
  {
    id: 'graph',
    label: 'Graph',
    icon: Network,
    description: 'Node, edge, and label visualization settings',
    buttonKey: '1'
  },
  {
    id: 'physics',
    label: 'Physics',
    icon: Atom,
    description: 'Simulation and force-directed layout controls',
    buttonKey: '2'
  },
  {
    id: 'effects',
    label: 'Effects',
    icon: Sparkles,
    description: 'Visual effects, glow, and animations',
    buttonKey: '3'
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: BarChart3,
    description: 'Metrics, filtering, and clustering',
    buttonKey: '4'
  },
  {
    id: 'quality',
    label: 'Quality',
    icon: Gauge,
    description: 'GPU, performance, and quality gates',
    buttonKey: '5'
  },
  {
    id: 'system',
    label: 'System',
    icon: Settings,
    description: 'Network, authentication, and system settings',
    buttonKey: '6'
  },
  // Advanced Tabs
  {
    id: 'xr',
    label: 'XR',
    icon: Glasses,
    description: 'VR/AR immersive settings',
    buttonKey: '7',
    isAdvanced: true
  },
  {
    id: 'ai',
    label: 'AI',
    icon: Bot,
    description: 'RAGFlow, Perplexity, and AI integrations',
    buttonKey: '8',
    isAdvanced: true,
    isPowerUserOnly: true
  },
  {
    id: 'developer',
    label: 'Dev',
    icon: Code,
    description: 'Debug tools and developer options',
    buttonKey: '9',
    isAdvanced: true,
    isPowerUserOnly: true
  }
];

// ============================================================================
// SETTINGS DEFINITIONS - With proper isAdvanced flags
// ============================================================================

export const UNIFIED_SETTINGS_CONFIG: Record<string, SectionConfig> = {
  // -------------------------------------------------------------------------
  // GRAPH TAB - Basic visualization
  // -------------------------------------------------------------------------
  graph: {
    title: 'Graph Visualization',
    fields: [
      // Nodes - Basic
      { key: 'nodeColor', label: 'Node Color', type: 'color', path: 'visualisation.graphs.logseq.nodes.baseColor', description: 'Base color for nodes' },
      { key: 'nodeSize', label: 'Node Size', type: 'slider', min: 0.2, max: 2, step: 0.1, path: 'visualisation.graphs.logseq.nodes.nodeSize', description: 'Size multiplier for nodes' },
      { key: 'nodeOpacity', label: 'Node Opacity', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.nodes.opacity', description: 'Transparency of nodes' },
      { key: 'enableInstancing', label: 'GPU Instancing', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableInstancing', description: 'Use GPU instancing for performance' },
      // Nodes - Advanced
      { key: 'nodeMetalness', label: 'Metalness', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.nodes.metalness', description: 'Metallic appearance', isAdvanced: true },
      { key: 'nodeRoughness', label: 'Roughness', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.nodes.roughness', description: 'Surface roughness', isAdvanced: true },
      { key: 'enableMetadataShape', label: 'Metadata Shape', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataShape', description: 'Shape based on metadata', isAdvanced: true },
      { key: 'enableMetadataVis', label: 'Metadata Visual', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableMetadataVisualisation', description: 'Visual encoding of metadata', isAdvanced: true },
      { key: 'nodeImportance', label: 'Show Importance', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableImportance', description: 'Size by importance score', isAdvanced: true },

      // Edges - Basic
      { key: 'edgeColor', label: 'Edge Color', type: 'color', path: 'visualisation.graphs.logseq.edges.color', description: 'Base color for edges' },
      { key: 'edgeWidth', label: 'Edge Width', type: 'slider', min: 0.01, max: 2, step: 0.01, path: 'visualisation.graphs.logseq.edges.baseWidth', description: 'Width of edges' },
      { key: 'edgeOpacity', label: 'Edge Opacity', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.edges.opacity', description: 'Transparency of edges' },
      { key: 'enableArrows', label: 'Show Arrows', type: 'toggle', path: 'visualisation.graphs.logseq.edges.enableArrows', description: 'Display direction arrows' },
      // Edges - Advanced
      { key: 'arrowSize', label: 'Arrow Size', type: 'slider', min: 0.01, max: 0.5, step: 0.01, path: 'visualisation.graphs.logseq.edges.arrowSize', description: 'Size of direction arrows', isAdvanced: true },
      { key: 'useGradient', label: 'Edge Gradient', type: 'toggle', path: 'visualisation.graphs.logseq.edges.useGradient', description: 'Use gradient coloring', isAdvanced: true },
      { key: 'distanceIntensity', label: 'Distance Intensity', type: 'slider', min: 0, max: 10, step: 0.1, path: 'visualisation.graphs.logseq.edges.distanceIntensity', description: 'Intensity based on distance', isAdvanced: true },

      // Labels - Basic
      { key: 'enableLabels', label: 'Show Labels', type: 'toggle', path: 'visualisation.graphs.logseq.labels.enableLabels', description: 'Display node labels' },
      { key: 'labelSize', label: 'Label Size', type: 'slider', min: 0.01, max: 1.5, step: 0.01, path: 'visualisation.graphs.logseq.labels.desktopFontSize', description: 'Font size for labels' },
      { key: 'labelColor', label: 'Label Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textColor', description: 'Color of label text' },
      // Labels - Advanced
      { key: 'labelOutlineColor', label: 'Outline Color', type: 'color', path: 'visualisation.graphs.logseq.labels.textOutlineColor', description: 'Label outline color', isAdvanced: true },
      { key: 'labelOutlineWidth', label: 'Outline Width', type: 'slider', min: 0, max: 0.01, step: 0.001, path: 'visualisation.graphs.logseq.labels.textOutlineWidth', description: 'Label outline width', isAdvanced: true },

      // Rendering - Basic
      { key: 'ambientLight', label: 'Ambient Light', type: 'slider', min: 0, max: 2, step: 0.1, path: 'visualisation.rendering.ambientLightIntensity', description: 'Overall scene brightness' },
      { key: 'directionalLight', label: 'Direct Light', type: 'slider', min: 0, max: 2, step: 0.1, path: 'visualisation.rendering.directionalLightIntensity', description: 'Directional light intensity' },
      // Rendering - Advanced
      { key: 'antialiasing', label: 'Antialiasing', type: 'toggle', path: 'visualisation.rendering.enableAntialiasing', description: 'Smooth edges', isAdvanced: true },
      { key: 'shadows', label: 'Shadows', type: 'toggle', path: 'visualisation.rendering.enableShadows', description: 'Enable shadows', isAdvanced: true },
      { key: 'ambientOcclusion', label: 'Ambient Occlusion', type: 'toggle', path: 'visualisation.rendering.enableAmbientOcclusion', description: 'SSAO effect', isAdvanced: true }
    ]
  },

  // -------------------------------------------------------------------------
  // PHYSICS TAB - Simulation controls
  // -------------------------------------------------------------------------
  physics: {
    title: 'Physics Simulation',
    fields: [
      // Core - Basic
      { key: 'enabled', label: 'Physics Enabled', type: 'toggle', path: 'visualisation.graphs.logseq.physics.enabled', description: 'Enable physics simulation' },
      { key: 'autoBalance', label: 'Auto Balance', type: 'toggle', path: 'visualisation.graphs.logseq.physics.autoBalance', description: 'Adaptive force balancing' },
      { key: 'damping', label: 'Damping', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.physics.damping', description: 'Velocity damping' },
      { key: 'springK', label: 'Spring Strength', type: 'slider', min: 0.0001, max: 10, step: 0.01, path: 'visualisation.graphs.logseq.physics.springK', description: 'Edge spring constant' },
      { key: 'repelK', label: 'Repulsion', type: 'slider', min: 0.1, max: 200, step: 1, path: 'visualisation.graphs.logseq.physics.repelK', description: 'Node repulsion strength' },
      { key: 'attractionK', label: 'Attraction', type: 'slider', min: 0, max: 10, step: 0.1, path: 'visualisation.graphs.logseq.physics.attractionK', description: 'Edge attraction strength' },

      // Dynamics - Basic
      { key: 'maxVelocity', label: 'Max Velocity', type: 'slider', min: 0.1, max: 10, step: 0.1, path: 'visualisation.graphs.logseq.physics.maxVelocity', description: 'Maximum node speed' },
      { key: 'enableBounds', label: 'Enable Bounds', type: 'toggle', path: 'visualisation.graphs.logseq.physics.enableBounds', description: 'Constrain to bounds' },
      { key: 'boundsSize', label: 'Bounds Size', type: 'slider', min: 1, max: 10000, step: 100, path: 'visualisation.graphs.logseq.physics.boundsSize', description: 'Size of bounding box' },

      // Advanced dynamics
      { key: 'dt', label: 'Time Step', type: 'slider', min: 0.001, max: 0.1, step: 0.001, path: 'visualisation.graphs.logseq.physics.dt', description: 'Simulation time step', isAdvanced: true },
      { key: 'separationRadius', label: 'Separation Radius', type: 'slider', min: 0.1, max: 10, step: 0.1, path: 'visualisation.graphs.logseq.physics.separationRadius', description: 'Minimum node separation', isAdvanced: true },
      { key: 'iterations', label: 'Iterations', type: 'slider', min: 1, max: 1000, step: 10, path: 'visualisation.graphs.logseq.physics.iterations', description: 'Solver iterations per frame', isAdvanced: true },
      { key: 'warmupIterations', label: 'Warmup Iterations', type: 'slider', min: 0, max: 500, step: 10, path: 'visualisation.graphs.logseq.physics.warmupIterations', description: 'Initial stabilization iterations', isAdvanced: true },
      { key: 'coolingRate', label: 'Cooling Rate', type: 'slider', min: 0.00001, max: 0.01, step: 0.0001, path: 'visualisation.graphs.logseq.physics.coolingRate', description: 'Simulated annealing rate', isAdvanced: true },

      // Fine-tuning - Advanced
      { key: 'stressWeight', label: 'Stress Weight', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.physics.stressWeight', description: 'Stress majorization weight', isAdvanced: true },
      { key: 'stressAlpha', label: 'Stress Alpha', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.physics.stressAlpha', description: 'Stress majorization alpha', isAdvanced: true },
      { key: 'minDistance', label: 'Min Distance', type: 'slider', min: 0.05, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.physics.minDistance', description: 'Minimum repulsion distance', isAdvanced: true },
      { key: 'maxRepulsionDist', label: 'Max Repulsion Dist', type: 'slider', min: 10, max: 200, step: 5, path: 'visualisation.graphs.logseq.physics.maxRepulsionDist', description: 'Maximum repulsion range', isAdvanced: true },
      { key: 'restLength', label: 'Rest Length', type: 'slider', min: 10, max: 200, step: 5, path: 'visualisation.graphs.logseq.physics.restLength', description: 'Spring rest length', isAdvanced: true },
      { key: 'repulsionCutoff', label: 'Repulsion Cutoff', type: 'slider', min: 10, max: 200, step: 5, path: 'visualisation.graphs.logseq.physics.repulsionCutoff', description: 'Cutoff for repulsion forces', isAdvanced: true },
      { key: 'centerGravityK', label: 'Center Gravity', type: 'slider', min: 0, max: 0.1, step: 0.001, path: 'visualisation.graphs.logseq.physics.centerGravityK', description: 'Pull towards center', isAdvanced: true },
      { key: 'gridCellSize', label: 'Grid Cell Size', type: 'slider', min: 10, max: 100, step: 5, path: 'visualisation.graphs.logseq.physics.gridCellSize', description: 'Spatial grid cell size', isAdvanced: true },
      { key: 'massScale', label: 'Mass Scale', type: 'slider', min: 0.1, max: 10, step: 0.1, path: 'visualisation.graphs.logseq.physics.massScale', description: 'Node mass multiplier', isAdvanced: true },
      { key: 'repulsionSofteningEpsilon', label: 'Repulsion Epsilon', type: 'slider', min: 0.00001, max: 0.01, step: 0.0001, path: 'visualisation.graphs.logseq.physics.repulsionSofteningEpsilon', description: 'Softening for close nodes', isAdvanced: true },
      { key: 'boundaryExtremeMultiplier', label: 'Boundary Extreme Mult', type: 'slider', min: 1, max: 5, step: 0.1, path: 'visualisation.graphs.logseq.physics.boundaryExtremeMultiplier', description: 'Boundary force multiplier', isAdvanced: true },
      { key: 'boundaryExtremeForceMultiplier', label: 'Boundary Force Mult', type: 'slider', min: 1, max: 20, step: 1, path: 'visualisation.graphs.logseq.physics.boundaryExtremeForceMultiplier', description: 'Extreme boundary force', isAdvanced: true },
      { key: 'boundaryVelocityDamping', label: 'Boundary Vel Damping', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.physics.boundaryVelocityDamping', description: 'Damping at boundaries', isAdvanced: true },
      { key: 'boundaryDamping', label: 'Boundary Damping', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.graphs.logseq.physics.boundaryDamping', description: 'General boundary damping', isAdvanced: true },
      { key: 'updateThreshold', label: 'Update Threshold', type: 'slider', min: 0, max: 0.5, step: 0.01, path: 'visualisation.graphs.logseq.physics.updateThreshold', description: 'Movement threshold for updates', isAdvanced: true }
    ]
  },

  // -------------------------------------------------------------------------
  // EFFECTS TAB - Visual effects
  // -------------------------------------------------------------------------
  effects: {
    title: 'Visual Effects',
    fields: [
      // Glow - Basic
      { key: 'glow', label: 'Hologram Glow', type: 'toggle', path: 'visualisation.glow.enabled', description: 'Enable glow effect' },
      { key: 'glowIntensity', label: 'Glow Intensity', type: 'slider', min: 0, max: 5, step: 0.1, path: 'visualisation.glow.intensity', description: 'Brightness of glow' },
      { key: 'glowRadius', label: 'Glow Radius', type: 'slider', min: 0, max: 5, step: 0.05, path: 'visualisation.glow.radius', description: 'Size of glow effect' },
      { key: 'glowThreshold', label: 'Glow Threshold', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.glow.threshold', description: 'Minimum brightness for glow', isAdvanced: true },

      // Hologram - Basic
      { key: 'hologram', label: 'Hologram Effect', type: 'toggle', path: 'visualisation.graphs.logseq.nodes.enableHologram', description: 'Enable hologram rings' },
      { key: 'ringCount', label: 'Ring Count', type: 'slider', min: 0, max: 10, step: 1, path: 'visualisation.hologram.ringCount', description: 'Number of hologram rings' },
      { key: 'ringColor', label: 'Ring Color', type: 'color', path: 'visualisation.hologram.ringColor', description: 'Color of rings' },
      { key: 'ringOpacity', label: 'Ring Opacity', type: 'slider', min: 0, max: 1, step: 0.01, path: 'visualisation.hologram.ringOpacity', description: 'Transparency of rings' },
      { key: 'ringRotationSpeed', label: 'Ring Speed', type: 'slider', min: 0, max: 5, step: 0.1, path: 'visualisation.hologram.ringRotationSpeed', description: 'Rotation speed', isAdvanced: true },

      // Edge Effects - Basic
      { key: 'flowEffect', label: 'Edge Flow', type: 'toggle', path: 'visualisation.graphs.logseq.edges.enableFlowEffect', description: 'Animated flow on edges' },
      { key: 'flowSpeed', label: 'Flow Speed', type: 'slider', min: 0.1, max: 5, step: 0.1, path: 'visualisation.graphs.logseq.edges.flowSpeed', description: 'Speed of flow animation', isAdvanced: true },
      { key: 'flowIntensity', label: 'Flow Intensity', type: 'slider', min: 0, max: 10, step: 0.1, path: 'visualisation.graphs.logseq.edges.flowIntensity', description: 'Intensity of flow effect', isAdvanced: true },
      { key: 'glowStrength', label: 'Edge Glow', type: 'slider', min: 0, max: 5, step: 0.1, path: 'visualisation.graphs.logseq.edges.glowStrength', description: 'Glow intensity on edges' },

      // Animation - Basic
      { key: 'nodeAnimations', label: 'Node Animations', type: 'toggle', path: 'visualisation.animations.enableNodeAnimations', description: 'Enable node animations' },
      { key: 'pulseEnabled', label: 'Pulse Effect', type: 'toggle', path: 'visualisation.animations.pulseEnabled', description: 'Pulsing effect on nodes' },
      { key: 'pulseSpeed', label: 'Pulse Speed', type: 'slider', min: 0.1, max: 2, step: 0.1, path: 'visualisation.animations.pulseSpeed', description: 'Speed of pulse', isAdvanced: true },
      { key: 'pulseStrength', label: 'Pulse Strength', type: 'slider', min: 0.1, max: 2, step: 0.1, path: 'visualisation.animations.pulseStrength', description: 'Intensity of pulse', isAdvanced: true },
      { key: 'selectionWave', label: 'Selection Wave', type: 'toggle', path: 'visualisation.animations.selectionWaveEnabled', description: 'Wave effect on selection', isAdvanced: true },
      { key: 'waveSpeed', label: 'Wave Speed', type: 'slider', min: 0.1, max: 2, step: 0.1, path: 'visualisation.animations.waveSpeed', description: 'Speed of selection wave', isAdvanced: true }
    ]
  },

  // -------------------------------------------------------------------------
  // ANALYTICS TAB - Metrics and clustering (filtering moved to Quality tab)
  // -------------------------------------------------------------------------
  analytics: {
    title: 'Analytics & Metrics',
    fields: [
      // Metrics - Basic
      { key: 'enableMetrics', label: 'Enable Metrics', type: 'toggle', path: 'analytics.enableMetrics', description: 'Show graph metrics' },
      { key: 'updateInterval', label: 'Update Interval', type: 'slider', min: 1, max: 60, step: 1, path: 'analytics.updateInterval', description: 'Seconds between updates' },
      { key: 'showDegreeDistribution', label: 'Degree Distribution', type: 'toggle', path: 'analytics.showDegreeDistribution', description: 'Show degree statistics', isAdvanced: true },
      { key: 'showClustering', label: 'Clustering Coefficient', type: 'toggle', path: 'analytics.showClusteringCoefficient', description: 'Show clustering metrics', isAdvanced: true },
      { key: 'showCentrality', label: 'Centrality Metrics', type: 'toggle', path: 'analytics.showCentrality', description: 'Show centrality scores', isAdvanced: true },

      // Clustering - Advanced
      { key: 'clusteringAlgorithm', label: 'Clustering Algorithm', type: 'select', options: ['none', 'kmeans', 'spectral', 'louvain'], path: 'analytics.clustering.algorithm', description: 'Algorithm for clustering', isAdvanced: true },
      { key: 'clusterCount', label: 'Cluster Count', type: 'slider', min: 2, max: 20, step: 1, path: 'analytics.clustering.clusterCount', description: 'Number of clusters', isAdvanced: true },
      { key: 'clusterResolution', label: 'Resolution', type: 'slider', min: 0.1, max: 2, step: 0.1, path: 'analytics.clustering.resolution', description: 'Clustering resolution', isAdvanced: true },
      { key: 'clusterIterations', label: 'Cluster Iterations', type: 'slider', min: 10, max: 100, step: 5, path: 'analytics.clustering.iterations', description: 'Algorithm iterations', isAdvanced: true },
      { key: 'exportClusters', label: 'Export Clusters', type: 'toggle', path: 'analytics.clustering.exportEnabled', description: 'Enable cluster export', isAdvanced: true, isPowerUserOnly: true },
      { key: 'importDistances', label: 'Import Distances', type: 'toggle', path: 'analytics.clustering.importEnabled', description: 'Enable distance import', isAdvanced: true, isPowerUserOnly: true }
    ]
  },

  // -------------------------------------------------------------------------
  // QUALITY TAB - Node filtering, GPU, and performance
  // -------------------------------------------------------------------------
  quality: {
    title: 'Quality & Filtering',
    fields: [
      // Node Filtering - Basic (moved from Analytics)
      { key: 'filterEnabled', label: 'Enable Filtering', type: 'toggle', path: 'nodeFilter.enabled', description: 'Enable node filtering' },
      { key: 'filterByQuality', label: 'Filter by Quality', type: 'toggle', path: 'nodeFilter.filterByQuality', description: 'Use quality score for filtering' },
      { key: 'qualityThreshold', label: 'Quality Threshold', type: 'slider', min: 0, max: 1, step: 0.05, path: 'nodeFilter.qualityThreshold', description: 'Minimum quality score (0-1)' },
      { key: 'filterByAuthority', label: 'Filter by Authority', type: 'toggle', path: 'nodeFilter.filterByAuthority', description: 'Use authority score for filtering' },
      { key: 'authorityThreshold', label: 'Authority Threshold', type: 'slider', min: 0, max: 1, step: 0.05, path: 'nodeFilter.authorityThreshold', description: 'Minimum authority score (0-1)' },
      { key: 'filterMode', label: 'Filter Mode', type: 'select', options: ['or', 'and'], path: 'nodeFilter.filterMode', description: 'How to combine filters (and = both, or = either)', isAdvanced: true },
      // Refresh Graph Button - applies current filter settings
      { key: 'refreshGraph', label: 'Refresh Graph', type: 'action-button', action: 'refresh_graph', description: 'Apply filter changes and reload graph' },

      // GPU Settings - Basic
      { key: 'gpuAcceleration', label: 'GPU Acceleration', type: 'toggle', path: 'qualityGates.gpuAcceleration', description: 'Enable GPU compute' },
      { key: 'autoAdjust', label: 'Auto-Adjust Quality', type: 'toggle', path: 'qualityGates.autoAdjust', description: 'Automatic quality scaling' },
      { key: 'minFpsThreshold', label: 'Min FPS Threshold', type: 'slider', min: 15, max: 60, step: 5, path: 'qualityGates.minFpsThreshold', description: 'Minimum acceptable FPS' },
      { key: 'maxNodeCount', label: 'Max Node Count', type: 'slider', min: 1000, max: 50000, step: 1000, path: 'qualityGates.maxNodeCount', description: 'Maximum nodes to render' },

      // Physics Features - Basic
      { key: 'ontologyPhysics', label: 'Ontology Physics', type: 'toggle', path: 'qualityGates.ontologyPhysics', description: 'Ontology-based forces' },
      { key: 'semanticForces', label: 'Semantic Forces', type: 'toggle', path: 'qualityGates.semanticForces', description: 'DAG and semantic layout' },
      { key: 'layoutMode', label: 'Layout Mode', type: 'select', options: ['force-directed', 'dag-topdown', 'dag-radial', 'dag-leftright', 'type-clustering'], path: 'qualityGates.layoutMode', description: 'Graph layout algorithm' },

      // Visualization - Basic
      { key: 'showClusters', label: 'Show Clusters', type: 'toggle', path: 'qualityGates.showClusters', description: 'Color-coded node groups' },
      { key: 'showAnomalies', label: 'Show Anomalies', type: 'toggle', path: 'qualityGates.showAnomalies', description: 'Highlight outliers' },
      { key: 'showCommunities', label: 'Show Communities', type: 'toggle', path: 'qualityGates.showCommunities', description: 'Louvain communities', isAdvanced: true },

      // Advanced Features
      { key: 'gnnPhysics', label: 'GNN-Enhanced Physics', type: 'toggle', path: 'qualityGates.gnnPhysics', description: 'Graph Neural Network weights', isAdvanced: true, isPowerUserOnly: true },
      { key: 'ruvectorEnabled', label: 'RuVector Integration', type: 'toggle', path: 'qualityGates.ruvectorEnabled', description: 'HNSW similarity search', isAdvanced: true, isPowerUserOnly: true },

      // LOD Settings - Advanced
      { key: 'lodEnabled', label: 'LOD Enabled', type: 'toggle', path: 'constraints.lodEnabled', description: 'Level-of-detail system', isAdvanced: true },
      { key: 'farThreshold', label: 'Far Threshold', type: 'slider', min: 100, max: 2000, step: 50, path: 'constraints.farThreshold', description: 'Far LOD distance', isAdvanced: true },
      { key: 'mediumThreshold', label: 'Medium Threshold', type: 'slider', min: 50, max: 500, step: 25, path: 'constraints.mediumThreshold', description: 'Medium LOD distance', isAdvanced: true },
      { key: 'nearThreshold', label: 'Near Threshold', type: 'slider', min: 5, max: 100, step: 5, path: 'constraints.nearThreshold', description: 'Near LOD distance', isAdvanced: true },
      { key: 'progressiveActivation', label: 'Progressive Activation', type: 'toggle', path: 'constraints.progressiveActivation', description: 'Gradual LOD transitions', isAdvanced: true },
      { key: 'activationFrames', label: 'Activation Frames', type: 'slider', min: 1, max: 600, step: 10, path: 'constraints.activationFrames', description: 'Transition duration', isAdvanced: true }
    ]
  },

  // -------------------------------------------------------------------------
  // SYSTEM TAB - Network and auth
  // -------------------------------------------------------------------------
  system: {
    title: 'System Settings',
    fields: [
      // Auth - Basic
      { key: 'nostr', label: 'Nostr Login', type: 'nostr-button', path: 'auth.nostr', description: 'Connect with Nostr' },
      { key: 'authEnabled', label: 'Auth Enabled', type: 'toggle', path: 'auth.enabled', description: 'Enable authentication', isAdvanced: true },
      { key: 'authRequired', label: 'Auth Required', type: 'toggle', path: 'auth.required', description: 'Require authentication', isAdvanced: true, isPowerUserOnly: true },

      // System - Basic
      { key: 'persistSettings', label: 'Persist Settings', type: 'toggle', path: 'system.persistSettingsOnServer', description: 'Save to server' },
      { key: 'customBackendURL', label: 'Custom Backend URL', type: 'text', path: 'system.customBackendUrl', description: 'Override backend URL', isAdvanced: true, isPowerUserOnly: true },

      // Debug - Advanced
      { key: 'enableDebug', label: 'Debug Mode', type: 'toggle', path: 'system.debug.enabled', description: 'Enable debug logging', isAdvanced: true }
    ]
  },

  // -------------------------------------------------------------------------
  // XR TAB - VR/AR settings (advanced)
  // -------------------------------------------------------------------------
  xr: {
    title: 'XR/AR Settings',
    isAdvanced: true,
    fields: [
      // Core XR - Basic within tab
      { key: 'xrEnabled', label: 'XR Mode', type: 'toggle', path: 'xr.enabled', description: 'Enable XR features' },
      { key: 'xrQuality', label: 'XR Quality', type: 'select', options: ['Low', 'Medium', 'High'], path: 'xr.quality', description: 'Rendering quality' },
      { key: 'xrRenderScale', label: 'XR Render Scale', type: 'slider', min: 0.5, max: 2, step: 0.1, path: 'xr.renderScale', description: 'Resolution scale' },

      // Hand Tracking
      { key: 'handTracking', label: 'Hand Tracking', type: 'toggle', path: 'xr.handTracking.enabled', description: 'Enable hand input' },
      { key: 'enableHaptics', label: 'Haptics', type: 'toggle', path: 'xr.interactions.enableHaptics', description: 'Haptic feedback' },

      // Performance
      { key: 'xrComputeMode', label: 'XR Compute Mode', type: 'toggle', path: 'xr.gpu.enableOptimizedCompute', description: 'Optimized GPU compute', isAdvanced: true },
      { key: 'xrPerformancePreset', label: 'XR Performance', type: 'select', options: ['Battery Saver', 'Balanced', 'Performance'], path: 'xr.performance.preset', description: 'Performance preset', isAdvanced: true },
      { key: 'xrAdaptiveQuality', label: 'Adaptive Quality', type: 'toggle', path: 'xr.enableAdaptiveQuality', description: 'Auto-adjust quality', isAdvanced: true }
    ]
  },

  // -------------------------------------------------------------------------
  // AI TAB - AI integrations (advanced, power user)
  // -------------------------------------------------------------------------
  ai: {
    title: 'AI Integrations',
    isAdvanced: true,
    isPowerUserOnly: true,
    fields: [
      // RAGFlow
      { key: 'ragflowApiUrl', label: 'RAGFlow API URL', type: 'text', path: 'ragflow.apiBaseUrl', description: 'RAGFlow endpoint' },
      { key: 'ragflowAgentId', label: 'Agent ID', type: 'text', path: 'ragflow.agentId', description: 'RAGFlow agent ID' },
      { key: 'ragflowTimeout', label: 'Timeout (ms)', type: 'slider', min: 5000, max: 120000, step: 1000, path: 'ragflow.timeout', description: 'Request timeout' },

      // Perplexity
      { key: 'perplexityModel', label: 'Perplexity Model', type: 'text', path: 'perplexity.model', description: 'Model selection' },
      { key: 'perplexityMaxTokens', label: 'Max Tokens', type: 'slider', min: 100, max: 4096, step: 100, path: 'perplexity.maxTokens', description: 'Maximum response tokens' },
      { key: 'perplexityTemperature', label: 'Temperature', type: 'slider', min: 0, max: 2, step: 0.1, path: 'perplexity.temperature', description: 'Response randomness' },

      // OpenAI
      { key: 'openaiBaseUrl', label: 'OpenAI Base URL', type: 'text', path: 'openai.baseUrl', description: 'API endpoint' },
      { key: 'openaiTimeout', label: 'Timeout (ms)', type: 'slider', min: 5000, max: 120000, step: 1000, path: 'openai.timeout', description: 'Request timeout' },

      // Kokoro TTS
      { key: 'kokoroApiUrl', label: 'Kokoro API URL', type: 'text', path: 'kokoro.apiUrl', description: 'TTS endpoint' },
      { key: 'kokoroVoice', label: 'Default Voice', type: 'text', path: 'kokoro.defaultVoice', description: 'Voice selection' },
      { key: 'kokoroSpeed', label: 'Speech Speed', type: 'slider', min: 0.5, max: 2, step: 0.1, path: 'kokoro.defaultSpeed', description: 'Playback speed' },

      // Whisper
      { key: 'whisperApiUrl', label: 'Whisper API URL', type: 'text', path: 'whisper.apiUrl', description: 'STT endpoint' },
      { key: 'whisperModel', label: 'Whisper Model', type: 'text', path: 'whisper.defaultModel', description: 'Model selection' },
      { key: 'whisperLanguage', label: 'Language', type: 'text', path: 'whisper.defaultLanguage', description: 'Input language' }
    ]
  },

  // -------------------------------------------------------------------------
  // DEVELOPER TAB - Debug tools (advanced, power user)
  // -------------------------------------------------------------------------
  developer: {
    title: 'Developer Tools',
    isAdvanced: true,
    isPowerUserOnly: true,
    fields: [
      // Logging
      { key: 'enableDebug', label: 'Debug Mode', type: 'toggle', path: 'system.debug.enabled', description: 'Enable debug mode' },
      { key: 'enableDataDebug', label: 'Data Debug', type: 'toggle', path: 'system.debug.enableDataDebug', description: 'Log data operations' },
      { key: 'enableWebsocketDebug', label: 'WebSocket Debug', type: 'toggle', path: 'system.debug.enableWebsocketDebug', description: 'Log WebSocket traffic' },
      { key: 'logBinaryHeaders', label: 'Log Binary Headers', type: 'toggle', path: 'system.debug.logBinaryHeaders', description: 'Log binary message headers' },
      { key: 'logFullJson', label: 'Log Full JSON', type: 'toggle', path: 'system.debug.logFullJson', description: 'Log complete JSON payloads' },
      { key: 'enablePhysicsDebug', label: 'Physics Debug', type: 'toggle', path: 'system.debug.enablePhysicsDebug', description: 'Physics visualization' },
      { key: 'enableNodeDebug', label: 'Node Debug', type: 'toggle', path: 'system.debug.enableNodeDebug', description: 'Node state logging' },
      { key: 'enableShaderDebug', label: 'Shader Debug', type: 'toggle', path: 'system.debug.enableShaderDebug', description: 'Shader debugging' },
      { key: 'enableMatrixDebug', label: 'Matrix Debug', type: 'toggle', path: 'system.debug.enableMatrixDebug', description: 'Matrix transformations' },
      { key: 'enablePerformanceDebug', label: 'Performance Debug', type: 'toggle', path: 'system.debug.enablePerformanceDebug', description: 'Performance metrics' },

      // GPU Debug
      { key: 'showForceVectors', label: 'Show Force Vectors', type: 'toggle', path: 'developer.gpu.showForceVectors', description: 'Visualize physics forces' },
      { key: 'showConstraints', label: 'Show Constraints', type: 'toggle', path: 'developer.gpu.showConstraints', description: 'Visualize constraints' },
      { key: 'showBoundaryForces', label: 'Show Boundary Forces', type: 'toggle', path: 'developer.gpu.showBoundaryForces', description: 'Boundary force visualization' },
      { key: 'showConvergenceGraph', label: 'Convergence Graph', type: 'toggle', path: 'developer.gpu.showConvergenceGraph', description: 'Physics convergence plot' }
    ]
  }
};

// Helper to filter fields based on mode and permissions
export function filterSettingsFields(
  fields: SectionConfig['fields'],
  advancedMode: boolean,
  isPowerUser: boolean
): SectionConfig['fields'] {
  return fields.filter(field => {
    // Hide advanced fields in basic mode
    if (field.isAdvanced && !advancedMode) return false;
    // Hide power user fields from non-power users
    if (field.isPowerUserOnly && !isPowerUser) return false;
    return true;
  });
}

// Helper to filter tabs based on mode and permissions
export function filterTabs(
  tabs: UnifiedTabConfig[],
  advancedMode: boolean,
  isPowerUser: boolean
): UnifiedTabConfig[] {
  return tabs.filter(tab => {
    if (tab.isAdvanced && !advancedMode) return false;
    if (tab.isPowerUserOnly && !isPowerUser) return false;
    return true;
  });
}
