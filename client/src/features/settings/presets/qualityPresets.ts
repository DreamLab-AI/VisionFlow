/**
 * Quality Presets System
 *
 * Provides one-click quality configurations for 571 settings.
 * Each preset optimizes different aspects: battery life, performance, quality, etc.
 */

export interface QualityPreset {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: 'performance' | 'balanced' | 'quality' | 'ultra';
  settings: Record<string, any>;
  systemRequirements?: {
    minRAM?: number;
    minVRAM?: number;
    recommendedGPU?: string;
  };
}

export const QUALITY_PRESETS: QualityPreset[] = [
  {
    id: 'low',
    name: 'Low (Battery Saver)',
    description: 'Optimized for battery life and older hardware',
    icon: 'Battery',
    category: 'performance',
    systemRequirements: {
      minRAM: 4,
      minVRAM: 1,
      recommendedGPU: 'Integrated Graphics'
    },
    settings: {
      // Physics Settings (Reduced for performance)
      'visualisation.graphs.logseq.physics.iterations': 100,
      'visualisation.graphs.logseq.physics.warmupIterations': 50,
      'visualisation.graphs.logseq.physics.deltaTime': 0.02,
      'visualisation.graphs.logseq.physics.gravity': -0.5,
      'visualisation.graphs.logseq.physics.springStiffness': 0.05,
      'visualisation.graphs.logseq.physics.springDamping': 0.5,
      'visualisation.graphs.logseq.physics.repulsionStrength': 50,
      'visualisation.graphs.logseq.physics.attractionStrength': 0.01,

      // Performance Settings (Maximum efficiency)
      'performance.targetFPS': 30,
      'performance.enableAdaptiveQuality': true,
      'performance.gpuMemoryLimit': 1024,
      'performance.maxConcurrentTasks': 2,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 2,
      'performance.cullingDistance': 50,

      // Visualization Settings (Minimal quality)
      'visualisation.graphs.logseq.nodes.nodeSize': 0.7,
      'visualisation.graphs.logseq.nodes.labelSize': 10,
      'visualisation.graphs.logseq.nodes.maxLabels': 20,
      'visualisation.graphs.logseq.edges.edgeThickness': 1,
      'visualisation.graphs.logseq.edges.maxEdges': 500,
      'visualisation.graphs.logseq.edges.enableCurves': false,

      // Rendering Settings (Disabled effects)
      'visualisation.rendering.enableAntialiasing': false,
      'visualisation.rendering.enableShadows': false,
      'visualisation.rendering.enableAmbientOcclusion': false,
      'visualisation.rendering.enableBloom': false,
      'visualisation.rendering.shadowQuality': 'low',
      'visualisation.rendering.textureQuality': 'low',
      'visualisation.rendering.meshQuality': 'low',

      // Glow Effects (Disabled)
      'visualisation.glow.enabled': false,
      'visualisation.glow.intensity': 0,
      'visualisation.glow.threshold': 1.0,

      // XR Settings (Optimized for performance)
      'xr.renderScale': 0.7,
      'xr.enableAdaptiveQuality': true,
      'xr.targetFrameRate': 60,
      'xr.enableFoveatedRendering': true,
      'xr.foveationLevel': 'high',

      // Animation Settings (Reduced)
      'visualisation.animations.duration': 200,
      'visualisation.animations.enableSpring': false,
      'visualisation.animations.particleCount': 10,

      // Camera Settings (Basic)
      'visualisation.camera.fov': 60,
      'visualisation.camera.smoothing': 0.05,
      'visualisation.camera.enableAutoRotate': false,

      // Memory Settings (Conservative)
      'performance.texturePoolSize': 256,
      'performance.geometryPoolSize': 128,
      'performance.enableGarbageCollection': true,
      'performance.gcInterval': 30000,
    }
  },
  {
    id: 'medium',
    name: 'Medium (Balanced)',
    description: 'Balanced performance and visual quality',
    icon: 'Cpu',
    category: 'balanced',
    systemRequirements: {
      minRAM: 8,
      minVRAM: 2,
      recommendedGPU: 'GTX 1060 / RX 580'
    },
    settings: {
      // Physics Settings (Balanced)
      'visualisation.graphs.logseq.physics.iterations': 300,
      'visualisation.graphs.logseq.physics.warmupIterations': 100,
      'visualisation.graphs.logseq.physics.deltaTime': 0.016,
      'visualisation.graphs.logseq.physics.gravity': -1.0,
      'visualisation.graphs.logseq.physics.springStiffness': 0.1,
      'visualisation.graphs.logseq.physics.springDamping': 0.3,
      'visualisation.graphs.logseq.physics.repulsionStrength': 100,
      'visualisation.graphs.logseq.physics.attractionStrength': 0.02,

      // Performance Settings (Balanced)
      'performance.targetFPS': 60,
      'performance.enableAdaptiveQuality': true,
      'performance.gpuMemoryLimit': 2048,
      'performance.maxConcurrentTasks': 4,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 3,
      'performance.cullingDistance': 100,

      // Visualization Settings (Medium quality)
      'visualisation.graphs.logseq.nodes.nodeSize': 1.0,
      'visualisation.graphs.logseq.nodes.labelSize': 12,
      'visualisation.graphs.logseq.nodes.maxLabels': 50,
      'visualisation.graphs.logseq.edges.edgeThickness': 2,
      'visualisation.graphs.logseq.edges.maxEdges': 1000,
      'visualisation.graphs.logseq.edges.enableCurves': true,

      // Rendering Settings (Selective effects)
      'visualisation.rendering.enableAntialiasing': true,
      'visualisation.rendering.enableShadows': false,
      'visualisation.rendering.enableAmbientOcclusion': false,
      'visualisation.rendering.enableBloom': true,
      'visualisation.rendering.shadowQuality': 'medium',
      'visualisation.rendering.textureQuality': 'medium',
      'visualisation.rendering.meshQuality': 'medium',

      // Glow Effects (Moderate)
      'visualisation.glow.enabled': true,
      'visualisation.glow.intensity': 0.5,
      'visualisation.glow.threshold': 0.8,
      'visualisation.glow.radius': 5,

      // XR Settings (Balanced)
      'xr.renderScale': 1.0,
      'xr.enableAdaptiveQuality': true,
      'xr.targetFrameRate': 72,
      'xr.enableFoveatedRendering': true,
      'xr.foveationLevel': 'medium',

      // Animation Settings (Smooth)
      'visualisation.animations.duration': 300,
      'visualisation.animations.enableSpring': true,
      'visualisation.animations.particleCount': 50,

      // Camera Settings (Enhanced)
      'visualisation.camera.fov': 70,
      'visualisation.camera.smoothing': 0.1,
      'visualisation.camera.enableAutoRotate': false,

      // Memory Settings (Balanced)
      'performance.texturePoolSize': 512,
      'performance.geometryPoolSize': 256,
      'performance.enableGarbageCollection': true,
      'performance.gcInterval': 60000,
    }
  },
  {
    id: 'high',
    name: 'High (Recommended)',
    description: 'High quality for modern hardware',
    icon: 'Zap',
    category: 'quality',
    systemRequirements: {
      minRAM: 16,
      minVRAM: 4,
      recommendedGPU: 'RTX 2060 / RX 5700'
    },
    settings: {
      // Physics Settings (High accuracy)
      'visualisation.graphs.logseq.physics.iterations': 500,
      'visualisation.graphs.logseq.physics.warmupIterations': 200,
      'visualisation.graphs.logseq.physics.deltaTime': 0.01,
      'visualisation.graphs.logseq.physics.gravity': -1.5,
      'visualisation.graphs.logseq.physics.springStiffness': 0.15,
      'visualisation.graphs.logseq.physics.springDamping': 0.2,
      'visualisation.graphs.logseq.physics.repulsionStrength': 150,
      'visualisation.graphs.logseq.physics.attractionStrength': 0.03,

      // Performance Settings (High quality)
      'performance.targetFPS': 60,
      'performance.enableAdaptiveQuality': false,
      'performance.gpuMemoryLimit': 4096,
      'performance.maxConcurrentTasks': 8,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 4,
      'performance.cullingDistance': 150,

      // Visualization Settings (High quality)
      'visualisation.graphs.logseq.nodes.nodeSize': 1.2,
      'visualisation.graphs.logseq.nodes.labelSize': 14,
      'visualisation.graphs.logseq.nodes.maxLabels': 100,
      'visualisation.graphs.logseq.edges.edgeThickness': 3,
      'visualisation.graphs.logseq.edges.maxEdges': 2000,
      'visualisation.graphs.logseq.edges.enableCurves': true,

      // Rendering Settings (Most effects enabled)
      'visualisation.rendering.enableAntialiasing': true,
      'visualisation.rendering.enableShadows': true,
      'visualisation.rendering.enableAmbientOcclusion': true,
      'visualisation.rendering.enableBloom': true,
      'visualisation.rendering.shadowQuality': 'high',
      'visualisation.rendering.textureQuality': 'high',
      'visualisation.rendering.meshQuality': 'high',

      // Glow Effects (Enhanced)
      'visualisation.glow.enabled': true,
      'visualisation.glow.intensity': 0.8,
      'visualisation.glow.threshold': 0.6,
      'visualisation.glow.radius': 8,
      'visualisation.glow.samples': 32,

      // XR Settings (High quality)
      'xr.renderScale': 1.2,
      'xr.enableAdaptiveQuality': false,
      'xr.targetFrameRate': 90,
      'xr.enableFoveatedRendering': true,
      'xr.foveationLevel': 'low',

      // Animation Settings (Fluid)
      'visualisation.animations.duration': 400,
      'visualisation.animations.enableSpring': true,
      'visualisation.animations.particleCount': 100,

      // Camera Settings (Professional)
      'visualisation.camera.fov': 75,
      'visualisation.camera.smoothing': 0.15,
      'visualisation.camera.enableAutoRotate': true,

      // Memory Settings (Generous)
      'performance.texturePoolSize': 1024,
      'performance.geometryPoolSize': 512,
      'performance.enableGarbageCollection': true,
      'performance.gcInterval': 90000,
    }
  },
  {
    id: 'ultra',
    name: 'Ultra (High-End)',
    description: 'Maximum quality for high-end systems',
    icon: 'Rocket',
    category: 'ultra',
    systemRequirements: {
      minRAM: 32,
      minVRAM: 8,
      recommendedGPU: 'RTX 3080 / RX 6800 XT or better'
    },
    settings: {
      // Physics Settings (Maximum accuracy)
      'visualisation.graphs.logseq.physics.iterations': 1000,
      'visualisation.graphs.logseq.physics.warmupIterations': 500,
      'visualisation.graphs.logseq.physics.deltaTime': 0.008,
      'visualisation.graphs.logseq.physics.gravity': -2.0,
      'visualisation.graphs.logseq.physics.springStiffness': 0.2,
      'visualisation.graphs.logseq.physics.springDamping': 0.1,
      'visualisation.graphs.logseq.physics.repulsionStrength': 200,
      'visualisation.graphs.logseq.physics.attractionStrength': 0.05,

      // Performance Settings (Maximum quality)
      'performance.targetFPS': 120,
      'performance.enableAdaptiveQuality': false,
      'performance.gpuMemoryLimit': 8192,
      'performance.maxConcurrentTasks': 16,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 5,
      'performance.cullingDistance': 200,

      // Visualization Settings (Maximum quality)
      'visualisation.graphs.logseq.nodes.nodeSize': 1.5,
      'visualisation.graphs.logseq.nodes.labelSize': 16,
      'visualisation.graphs.logseq.nodes.maxLabels': 200,
      'visualisation.graphs.logseq.edges.edgeThickness': 4,
      'visualisation.graphs.logseq.edges.maxEdges': 5000,
      'visualisation.graphs.logseq.edges.enableCurves': true,

      // Rendering Settings (All effects enabled)
      'visualisation.rendering.enableAntialiasing': true,
      'visualisation.rendering.enableShadows': true,
      'visualisation.rendering.enableAmbientOcclusion': true,
      'visualisation.rendering.enableBloom': true,
      'visualisation.rendering.enableSSR': true,
      'visualisation.rendering.enableVolumetricLighting': true,
      'visualisation.rendering.shadowQuality': 'ultra',
      'visualisation.rendering.textureQuality': 'ultra',
      'visualisation.rendering.meshQuality': 'ultra',

      // Glow Effects (Maximum)
      'visualisation.glow.enabled': true,
      'visualisation.glow.intensity': 1.0,
      'visualisation.glow.threshold': 0.4,
      'visualisation.glow.radius': 12,
      'visualisation.glow.samples': 64,
      'visualisation.glow.enableHDR': true,

      // XR Settings (Maximum quality)
      'xr.renderScale': 1.5,
      'xr.enableAdaptiveQuality': false,
      'xr.targetFrameRate': 120,
      'xr.enableFoveatedRendering': false,
      'xr.enableSupersampling': true,

      // Animation Settings (Cinematic)
      'visualisation.animations.duration': 500,
      'visualisation.animations.enableSpring': true,
      'visualisation.animations.particleCount': 200,
      'visualisation.animations.enableMotionBlur': true,

      // Camera Settings (Cinematic)
      'visualisation.camera.fov': 80,
      'visualisation.camera.smoothing': 0.2,
      'visualisation.camera.enableAutoRotate': true,
      'visualisation.camera.enableDepthOfField': true,

      // Memory Settings (Maximum)
      'performance.texturePoolSize': 2048,
      'performance.geometryPoolSize': 1024,
      'performance.enableGarbageCollection': false,
      'performance.cacheSize': 4096,
    }
  }
];

/**
 * Get preset by ID
 */
export const getPresetById = (id: string): QualityPreset | undefined => {
  return QUALITY_PRESETS.find(preset => preset.id === id);
};

/**
 * Get recommended preset based on system capabilities
 */
export const getRecommendedPreset = (systemInfo: {
  ram: number;
  vram: number;
  gpu: string;
}): QualityPreset => {
  // Simple heuristic for recommendation
  if (systemInfo.vram >= 8 && systemInfo.ram >= 32) {
    return QUALITY_PRESETS[3]; // Ultra
  } else if (systemInfo.vram >= 4 && systemInfo.ram >= 16) {
    return QUALITY_PRESETS[2]; // High
  } else if (systemInfo.vram >= 2 && systemInfo.ram >= 8) {
    return QUALITY_PRESETS[1]; // Medium
  } else {
    return QUALITY_PRESETS[0]; // Low
  }
};

/**
 * Validate preset settings against schema
 */
export const validatePresetSettings = (settings: Record<string, any>): boolean => {
  // Add validation logic here
  // For now, just check if settings object exists
  return settings !== null && typeof settings === 'object';
};
