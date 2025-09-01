import { useSelectiveSetting, useSettingSetter } from './useSelectiveSettingsStore';
import { SettingsPath } from '../types/generated/settings';
import { useCallback } from 'react';

/**
 * Hook for visualization-specific settings with selective access
 * Provides optimized access to visualization settings
 */
export function useVisualizationSettings() {
  const { set, batchSet } = useSettingSetter();
  
  // Core visualization settings
  const visualizationEnabled = useSelectiveSetting<boolean>('visualisation.enabled');
  const theme = useSelectiveSetting<string>('visualisation.theme');
  const quality = useSelectiveSetting<'low' | 'medium' | 'high' | 'ultra'>('visualisation.quality');
  const antialiasing = useSelectiveSetting<boolean>('visualisation.antialiasing');
  const shadows = useSelectiveSetting<boolean>('visualisation.shadows');
  
  // Camera settings
  const cameraFov = useSelectiveSetting<number>('visualisation.camera.fov');
  const cameraNear = useSelectiveSetting<number>('visualisation.camera.near');
  const cameraFar = useSelectiveSetting<number>('visualisation.camera.far');
  const cameraPosition = useSelectiveSetting<[number, number, number]>('visualisation.camera.position');
  
  // Lighting settings
  const ambientLightIntensity = useSelectiveSetting<number>('visualisation.lighting.ambient.intensity');
  const directionalLightIntensity = useSelectiveSetting<number>('visualisation.lighting.directional.intensity');
  const directionalLightPosition = useSelectiveSetting<[number, number, number]>('visualisation.lighting.directional.position');
  
  // Post-processing settings
  const bloomEnabled = useSelectiveSetting<boolean>('visualisation.postProcessing.bloom.enabled');
  const bloomIntensity = useSelectiveSetting<number>('visualisation.postProcessing.bloom.intensity');
  const ssaoEnabled = useSelectiveSetting<boolean>('visualisation.postProcessing.ssao.enabled');
  const ssaoIntensity = useSelectiveSetting<number>('visualisation.postProcessing.ssao.intensity');
  
  // Helper functions for common operations
  const updateVisualizationSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `visualisation.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateCameraSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `visualisation.camera.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateLightingSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `visualisation.lighting.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updatePostProcessingSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `visualisation.postProcessing.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const batchUpdateVisualization = useCallback(async (
    updates: Record<string, any>
  ) => {
    const batchUpdates = Object.entries(updates).map(([path, value]) => ({
      path: `visualisation.${path}` as SettingsPath,
      value
    }));
    
    await batchSet(batchUpdates);
  }, [batchSet]);
  
  return {
    // Core settings
    visualizationEnabled,
    theme,
    quality,
    antialiasing,
    shadows,
    
    // Camera settings
    camera: {
      fov: cameraFov,
      near: cameraNear,
      far: cameraFar,
      position: cameraPosition
    },
    
    // Lighting settings
    lighting: {
      ambient: { intensity: ambientLightIntensity },
      directional: { 
        intensity: directionalLightIntensity,
        position: directionalLightPosition
      }
    },
    
    // Post-processing settings
    postProcessing: {
      bloom: {
        enabled: bloomEnabled,
        intensity: bloomIntensity
      },
      ssao: {
        enabled: ssaoEnabled,
        intensity: ssaoIntensity
      }
    },
    
    // Update functions
    updateVisualizationSetting,
    updateCameraSetting,
    updateLightingSetting,
    updatePostProcessingSetting,
    batchUpdateVisualization
  };
}

/**
 * Hook for XR-specific visualization settings
 */
export function useXRVisualizationSettings() {
  const { set } = useSettingSetter();
  
  const xrEnabled = useSelectiveSetting<boolean>('visualisation.xr.enabled');
  const xrQuality = useSelectiveSetting<'low' | 'medium' | 'high'>('visualisation.xr.quality');
  const handTracking = useSelectiveSetting<boolean>('visualisation.xr.handTracking');
  const eyeTracking = useSelectiveSetting<boolean>('visualisation.xr.eyeTracking');
  const roomScale = useSelectiveSetting<boolean>('visualisation.xr.roomScale');
  
  const updateXRSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `visualisation.xr.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  return {
    xrEnabled,
    xrQuality,
    handTracking,
    eyeTracking,
    roomScale,
    updateXRSetting
  };
}

/**
 * Hook for performance-related visualization settings
 */
export function useVisualizationPerformanceSettings() {
  const { set, batchSet } = useSettingSetter();
  
  const frameRateLimit = useSelectiveSetting<number>('visualisation.performance.frameRateLimit');
  const adaptiveQuality = useSelectiveSetting<boolean>('visualisation.performance.adaptiveQuality');
  const cullingEnabled = useSelectiveSetting<boolean>('visualisation.performance.culling');
  const lodEnabled = useSelectiveSetting<boolean>('visualisation.performance.lod');
  const instancedRendering = useSelectiveSetting<boolean>('visualisation.performance.instancing');
  
  const updatePerformanceSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `visualisation.performance.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const optimizeForPerformance = useCallback(async () => {
    await batchSet([
      { path: 'visualisation.quality', value: 'medium' },
      { path: 'visualisation.shadows', value: false },
      { path: 'visualisation.postProcessing.bloom.enabled', value: false },
      { path: 'visualisation.postProcessing.ssao.enabled', value: false },
      { path: 'visualisation.performance.frameRateLimit', value: 30 },
      { path: 'visualisation.performance.adaptiveQuality', value: true },
      { path: 'visualisation.performance.culling', value: true },
      { path: 'visualisation.performance.lod', value: true }
    ]);
  }, [batchSet]);
  
  const optimizeForQuality = useCallback(async () => {
    await batchSet([
      { path: 'visualisation.quality', value: 'ultra' },
      { path: 'visualisation.shadows', value: true },
      { path: 'visualisation.postProcessing.bloom.enabled', value: true },
      { path: 'visualisation.postProcessing.ssao.enabled', value: true },
      { path: 'visualisation.performance.frameRateLimit', value: 60 },
      { path: 'visualisation.performance.adaptiveQuality', value: false }
    ]);
  }, [batchSet]);
  
  return {
    frameRateLimit,
    adaptiveQuality,
    cullingEnabled,
    lodEnabled,
    instancedRendering,
    updatePerformanceSetting,
    optimizeForPerformance,
    optimizeForQuality
  };
}