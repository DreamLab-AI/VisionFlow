import React, { useMemo } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { CustomEffectsRenderer, CustomEffectsConfig } from './CustomEffectsRenderer';
import { PostProcessingEffects } from '@/features/graph/components/PostProcessingEffects';
import * as THREE from 'three';
import { createLogger } from '@/utils/logger';

const logger = createLogger('DiffuseEffectsIntegration');

export interface DiffuseEffectsIntegrationProps {
  enableDiffuseEffects?: boolean;
  enableBloomForGraphs?: boolean;
  children?: React.ReactNode;
}

/**
 * Integration component that manages both diffuse effects for background elements
 * and selective bloom effects for force-directed graphs
 */
export const DiffuseEffectsIntegration: React.FC<DiffuseEffectsIntegrationProps> = ({
  enableDiffuseEffects = true,
  enableBloomForGraphs = true,
  children
}) => {
  const settings = useSettingsStore(state => state.settings?.visualisation);
  
  // Create custom effects configuration from settings
  const diffuseConfig = useMemo((): Partial<CustomEffectsConfig> => {
    const hologramSettings = settings?.hologram;
    const baseColor = new THREE.Color(hologramSettings?.ringColor || '#00ffff');
    
    return {
      diffuse: {
        enabled: enableDiffuseEffects && (settings?.diffuseEffects?.enabled !== false),
        intensity: settings?.diffuseEffects?.intensity || 0.8,
        radius: settings?.diffuseEffects?.radius || 2.0,
        color: baseColor,
        opacity: hologramSettings?.ringOpacity || 0.7,
        distanceFieldScale: settings?.diffuseEffects?.distanceFieldScale || 1.0,
        wireframeThickness: settings?.diffuseEffects?.wireframeThickness || 0.002
      },
      motes: {
        enabled: enableDiffuseEffects && (settings?.motes?.enabled !== false),
        intensity: settings?.motes?.intensity || 0.6,
        size: settings?.motes?.size || 1.0,
        color: baseColor,
        opacity: settings?.motes?.opacity || 0.8
      },
      rings: {
        enabled: enableDiffuseEffects && (hologramSettings?.enableTriangleSphere !== false),
        glow: settings?.rings?.glow || 0.9,
        width: settings?.rings?.width || 0.05,
        color: baseColor,
        opacity: hologramSettings?.ringOpacity || 0.7
      }
    };
  }, [settings, enableDiffuseEffects]);
  
  // Log configuration for debugging
  React.useEffect(() => {
    logger.debug('DiffuseEffectsIntegration configuration', {
      enableDiffuseEffects,
      enableBloomForGraphs,
      diffuseConfig,
      hologramSettings: settings?.hologram
    });
  }, [enableDiffuseEffects, enableBloomForGraphs, diffuseConfig, settings?.hologram]);
  
  return (
    <>
      {/* Custom diffuse effects for background elements (rings, wireframes, motes) */}
      {enableDiffuseEffects && (
        <CustomEffectsRenderer
          config={diffuseConfig}
          backgroundElementsOnly={true}
        >
          {children}
        </CustomEffectsRenderer>
      )}
      
      {/* Selective bloom effects for force-directed graphs only */}
      {enableBloomForGraphs && (
        <PostProcessingEffects graphElementsOnly={true} />
      )}
      
      {/* Render children without effects wrapper if diffuse effects are disabled */}
      {!enableDiffuseEffects && children}
    </>
  );
};

/**
 * Hook to provide diffuse effects controls to components
 */
export const useDiffuseEffectsControls = () => {
  const { settings, updateSettings } = useSettingsStore();
  
  const updateDiffuseConfig = React.useCallback((
    updates: Partial<CustomEffectsConfig>
  ) => {
    const newSettings = {
      ...settings,
      visualisation: {
        ...settings?.visualisation,
        diffuseEffects: {
          ...settings?.visualisation?.diffuseEffects,
          ...updates.diffuse
        },
        motes: {
          ...settings?.visualisation?.motes,
          ...updates.motes
        },
        rings: {
          ...settings?.visualisation?.rings,
          ...updates.rings
        }
      }
    };
    
    updateSettings(newSettings);
    logger.debug('Updated diffuse effects configuration', updates);
  }, [settings, updateSettings]);
  
  const toggleDiffuseEffects = React.useCallback((enabled: boolean) => {
    updateDiffuseConfig({
      diffuse: { enabled },
      motes: { enabled },
      rings: { enabled }
    });
  }, [updateDiffuseConfig]);
  
  const updateIntensity = React.useCallback((intensity: number) => {
    updateDiffuseConfig({
      diffuse: { intensity },
      motes: { intensity: intensity * 0.75 },
      rings: { glow: intensity * 1.1 }
    });
  }, [updateDiffuseConfig]);
  
  const updateColor = React.useCallback((color: THREE.Color) => {
    updateDiffuseConfig({
      diffuse: { color },
      motes: { color },
      rings: { color }
    });
  }, [updateDiffuseConfig]);
  
  const updateOpacity = React.useCallback((opacity: number) => {
    updateDiffuseConfig({
      diffuse: { opacity },
      motes: { opacity },
      rings: { opacity }
    });
  }, [updateDiffuseConfig]);
  
  return {
    config: {
      enabled: settings?.visualisation?.diffuseEffects?.enabled !== false,
      intensity: settings?.visualisation?.diffuseEffects?.intensity || 0.8,
      radius: settings?.visualisation?.diffuseEffects?.radius || 2.0,
      opacity: settings?.visualisation?.diffuseEffects?.opacity || 0.7,
      color: new THREE.Color(settings?.visualisation?.hologram?.ringColor || '#00ffff')
    },
    controls: {
      toggleDiffuseEffects,
      updateIntensity,
      updateColor,
      updateOpacity,
      updateDiffuseConfig
    }
  };
};

/**
 * Component for rendering diffuse effect controls in settings panels
 */
export const DiffuseEffectsControls: React.FC<{
  className?: string;
}> = ({ className }) => {
  const { config, controls } = useDiffuseEffectsControls();
  
  return (
    <div className={className}>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Diffuse Effects</label>
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={(e) => controls.toggleDiffuseEffects(e.target.checked)}
            className="rounded"
          />
        </div>
        
        {config.enabled && (
          <>
            <div>
              <label className="text-sm font-medium block mb-1">
                Intensity: {config.intensity.toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={2}
                step={0.1}
                value={config.intensity}
                onChange={(e) => controls.updateIntensity(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="text-sm font-medium block mb-1">
                Opacity: {config.opacity.toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={config.opacity}
                onChange={(e) => controls.updateOpacity(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="text-sm font-medium block mb-1">Color</label>
              <input
                type="color"
                value={`#${config.color.getHexString()}`}
                onChange={(e) => controls.updateColor(new THREE.Color(e.target.value))}
                className="w-full h-8 rounded border"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default DiffuseEffectsIntegration;