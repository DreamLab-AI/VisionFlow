

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
      
      'visualisation.graphs.logseq.physics.iterations': 100,
      'visualisation.graphs.logseq.physics.warmupIterations': 50,
      'visualisation.graphs.logseq.physics.dt': 0.02,
      'visualisation.graphs.logseq.physics.gravity': -0.5,
      'visualisation.graphs.logseq.physics.springK': 0.05,
      'visualisation.graphs.logseq.physics.damping': 0.5,
      'visualisation.graphs.logseq.physics.repelK': 50,
      'visualisation.graphs.logseq.physics.centerGravityK': 0.01,

      
      'performance.targetFPS': 30,
      'performance.enableAdaptiveQuality': true,
      'performance.gpuMemoryLimit': 1024,
      'performance.maxConcurrentTasks': 2,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 2,
      'performance.cullingDistance': 50,

      
      'visualisation.graphs.logseq.nodes.nodeSize': 0.7,
      'visualisation.graphs.logseq.nodes.labelSize': 1.0,
      'visualisation.graphs.logseq.nodes.maxLabels': 20,
      'visualisation.graphs.logseq.edges.edgeThickness': 1,
      'visualisation.graphs.logseq.edges.maxEdges': 500,
      'visualisation.graphs.logseq.edges.enableCurves': false,

      
      'visualisation.rendering.enableAntialiasing': false,
      'visualisation.rendering.enableShadows': false,
      'visualisation.rendering.enableAmbientOcclusion': false,
      'visualisation.rendering.enableBloom': false,
      'visualisation.rendering.shadowQuality': 'low',
      'visualisation.rendering.textureQuality': 'low',
      'visualisation.rendering.meshQuality': 'low',

      
      'visualisation.glow.enabled': false,
      'visualisation.glow.intensity': 0,
      'visualisation.glow.threshold': 1.0,

      
      'xr.renderScale': 0.7,
      'xr.enableAdaptiveQuality': true,
      'xr.targetFrameRate': 60,
      'xr.enableFoveatedRendering': true,
      'xr.foveationLevel': 'high',

      
      'visualisation.animations.duration': 200,
      'visualisation.animations.enableSpring': false,
      'visualisation.animations.particleCount': 10,

      
      'visualisation.camera.fov': 60,
      'visualisation.camera.smoothing': 0.05,
      'visualisation.camera.enableAutoRotate': false,

      
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
      
      'visualisation.graphs.logseq.physics.iterations': 300,
      'visualisation.graphs.logseq.physics.warmupIterations': 100,
      'visualisation.graphs.logseq.physics.dt': 0.016,
      'visualisation.graphs.logseq.physics.gravity': -1.0,
      'visualisation.graphs.logseq.physics.springK': 0.1,
      'visualisation.graphs.logseq.physics.damping': 0.3,
      'visualisation.graphs.logseq.physics.repelK': 100,
      'visualisation.graphs.logseq.physics.centerGravityK': 0.02,

      
      'performance.targetFPS': 60,
      'performance.enableAdaptiveQuality': true,
      'performance.gpuMemoryLimit': 2048,
      'performance.maxConcurrentTasks': 4,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 3,
      'performance.cullingDistance': 100,

      
      'visualisation.graphs.logseq.nodes.nodeSize': 1.0,
      'visualisation.graphs.logseq.nodes.labelSize': 1.2,
      'visualisation.graphs.logseq.nodes.maxLabels': 50,
      'visualisation.graphs.logseq.edges.edgeThickness': 2,
      'visualisation.graphs.logseq.edges.maxEdges': 1000,
      'visualisation.graphs.logseq.edges.enableCurves': true,

      
      'visualisation.rendering.enableAntialiasing': true,
      'visualisation.rendering.enableShadows': false,
      'visualisation.rendering.enableAmbientOcclusion': false,
      'visualisation.rendering.enableBloom': true,
      'visualisation.rendering.shadowQuality': 'medium',
      'visualisation.rendering.textureQuality': 'medium',
      'visualisation.rendering.meshQuality': 'medium',

      
      'visualisation.glow.enabled': true,
      'visualisation.glow.intensity': 0.5,
      'visualisation.glow.threshold': 0.8,
      'visualisation.glow.radius': 5,

      
      'xr.renderScale': 1.0,
      'xr.enableAdaptiveQuality': true,
      'xr.targetFrameRate': 72,
      'xr.enableFoveatedRendering': true,
      'xr.foveationLevel': 'medium',

      
      'visualisation.animations.duration': 300,
      'visualisation.animations.enableSpring': true,
      'visualisation.animations.particleCount': 50,

      
      'visualisation.camera.fov': 70,
      'visualisation.camera.smoothing': 0.1,
      'visualisation.camera.enableAutoRotate': false,

      
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
      
      'visualisation.graphs.logseq.physics.iterations': 500,
      'visualisation.graphs.logseq.physics.warmupIterations': 200,
      'visualisation.graphs.logseq.physics.dt': 0.01,
      'visualisation.graphs.logseq.physics.gravity': -1.5,
      'visualisation.graphs.logseq.physics.springK': 0.15,
      'visualisation.graphs.logseq.physics.damping': 0.2,
      'visualisation.graphs.logseq.physics.repelK': 150,
      'visualisation.graphs.logseq.physics.centerGravityK': 0.03,

      
      'performance.targetFPS': 60,
      'performance.enableAdaptiveQuality': false,
      'performance.gpuMemoryLimit': 4096,
      'performance.maxConcurrentTasks': 8,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 4,
      'performance.cullingDistance': 150,

      
      'visualisation.graphs.logseq.nodes.nodeSize': 1.2,
      'visualisation.graphs.logseq.nodes.labelSize': 1.4,
      'visualisation.graphs.logseq.nodes.maxLabels': 100,
      'visualisation.graphs.logseq.edges.edgeThickness': 3,
      'visualisation.graphs.logseq.edges.maxEdges': 2000,
      'visualisation.graphs.logseq.edges.enableCurves': true,

      
      'visualisation.rendering.enableAntialiasing': true,
      'visualisation.rendering.enableShadows': true,
      'visualisation.rendering.enableAmbientOcclusion': true,
      'visualisation.rendering.enableBloom': true,
      'visualisation.rendering.shadowQuality': 'high',
      'visualisation.rendering.textureQuality': 'high',
      'visualisation.rendering.meshQuality': 'high',

      
      'visualisation.glow.enabled': true,
      'visualisation.glow.intensity': 0.8,
      'visualisation.glow.threshold': 0.6,
      'visualisation.glow.radius': 8,
      'visualisation.glow.samples': 32,

      
      'xr.renderScale': 1.2,
      'xr.enableAdaptiveQuality': false,
      'xr.targetFrameRate': 90,
      'xr.enableFoveatedRendering': true,
      'xr.foveationLevel': 'low',

      
      'visualisation.animations.duration': 400,
      'visualisation.animations.enableSpring': true,
      'visualisation.animations.particleCount': 100,

      
      'visualisation.camera.fov': 75,
      'visualisation.camera.smoothing': 0.15,
      'visualisation.camera.enableAutoRotate': true,

      
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
      
      'visualisation.graphs.logseq.physics.iterations': 1000,
      'visualisation.graphs.logseq.physics.warmupIterations': 500,
      'visualisation.graphs.logseq.physics.dt': 0.008,
      'visualisation.graphs.logseq.physics.gravity': -2.0,
      'visualisation.graphs.logseq.physics.springK': 0.2,
      'visualisation.graphs.logseq.physics.damping': 0.1,
      'visualisation.graphs.logseq.physics.repelK': 200,
      'visualisation.graphs.logseq.physics.centerGravityK': 0.05,

      
      'performance.targetFPS': 120,
      'performance.enableAdaptiveQuality': false,
      'performance.gpuMemoryLimit': 8192,
      'performance.maxConcurrentTasks': 16,
      'performance.enableOcclusion': true,
      'performance.lodLevels': 5,
      'performance.cullingDistance': 200,

      
      'visualisation.graphs.logseq.nodes.nodeSize': 1.5,
      'visualisation.graphs.logseq.nodes.labelSize': 1.6,
      'visualisation.graphs.logseq.nodes.maxLabels': 200,
      'visualisation.graphs.logseq.edges.edgeThickness': 4,
      'visualisation.graphs.logseq.edges.maxEdges': 5000,
      'visualisation.graphs.logseq.edges.enableCurves': true,

      
      'visualisation.rendering.enableAntialiasing': true,
      'visualisation.rendering.enableShadows': true,
      'visualisation.rendering.enableAmbientOcclusion': true,
      'visualisation.rendering.enableBloom': true,
      'visualisation.rendering.enableSSR': true,
      'visualisation.rendering.enableVolumetricLighting': true,
      'visualisation.rendering.shadowQuality': 'ultra',
      'visualisation.rendering.textureQuality': 'ultra',
      'visualisation.rendering.meshQuality': 'ultra',

      
      'visualisation.glow.enabled': true,
      'visualisation.glow.intensity': 1.0,
      'visualisation.glow.threshold': 0.4,
      'visualisation.glow.radius': 12,
      'visualisation.glow.samples': 64,
      'visualisation.glow.enableHDR': true,

      
      'xr.renderScale': 1.5,
      'xr.enableAdaptiveQuality': false,
      'xr.targetFrameRate': 120,
      'xr.enableFoveatedRendering': false,
      'xr.enableSupersampling': true,

      
      'visualisation.animations.duration': 500,
      'visualisation.animations.enableSpring': true,
      'visualisation.animations.particleCount': 200,
      'visualisation.animations.enableMotionBlur': true,

      
      'visualisation.camera.fov': 80,
      'visualisation.camera.smoothing': 0.2,
      'visualisation.camera.enableAutoRotate': true,
      'visualisation.camera.enableDepthOfField': true,

      
      'performance.texturePoolSize': 2048,
      'performance.geometryPoolSize': 1024,
      'performance.enableGarbageCollection': false,
      'performance.cacheSize': 4096,
    }
  }
];


export const getPresetById = (id: string): QualityPreset | undefined => {
  return QUALITY_PRESETS.find(preset => preset.id === id);
};


export const getRecommendedPreset = (systemInfo: {
  ram: number;
  vram: number;
  gpu: string;
}): QualityPreset => {
  
  if (systemInfo.vram >= 8 && systemInfo.ram >= 32) {
    return QUALITY_PRESETS[3]; 
  } else if (systemInfo.vram >= 4 && systemInfo.ram >= 16) {
    return QUALITY_PRESETS[2]; 
  } else if (systemInfo.vram >= 2 && systemInfo.ram >= 8) {
    return QUALITY_PRESETS[1]; 
  } else {
    return QUALITY_PRESETS[0]; 
  }
};


export const validatePresetSettings = (settings: Record<string, any>): boolean => {
  
  
  return settings !== null && typeof settings === 'object';
};
