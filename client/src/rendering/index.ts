// Main rendering system exports
export {
  CustomEffectsRenderer,
  type CustomEffectsConfig,
  type CustomEffectsRendererProps,
  defaultCustomEffectsConfig
} from './CustomEffectsRenderer';

export {
  DiffuseWireframeMaterial,
  DiffuseHologramRingMaterial,
  DiffuseMoteMaterial
} from './DiffuseWireframeMaterial';

export {
  DiffuseEffectsIntegration,
  DiffuseEffectsControls,
  useDiffuseEffectsControls,
  type DiffuseEffectsIntegrationProps
} from './DiffuseEffectsIntegration';

// Note: HologramManager components should be imported directly from their location
// to avoid circular dependencies

// Utility function to determine if diffuse effects should be used
export const shouldUseDiffuseEffects = (settings: any): boolean => {
  return settings?.visualisation?.diffuseEffects?.enabled !== false;
};

// Utility function to get optimal diffuse settings based on performance
export const getOptimalDiffuseSettings = (isXRMode: boolean, isLowPowerMode: boolean) => {
  const baseIntensity = isXRMode ? 1.0 : 0.8;
  const baseRadius = isXRMode ? 1.5 : 2.0;
  const baseOpacity = isXRMode ? 0.8 : 0.7;
  
  if (isLowPowerMode) {
    return {
      intensity: baseIntensity * 0.6,
      radius: baseRadius * 0.7,
      opacity: baseOpacity * 0.8,
      distanceFieldScale: 0.8,
      wireframeThickness: 0.003
    };
  }
  
  return {
    intensity: baseIntensity,
    radius: baseRadius,
    opacity: baseOpacity,
    distanceFieldScale: 1.0,
    wireframeThickness: 0.002
  };
};