import { useSelectiveSetting, useSettingSetter } from './useSelectiveSettingsStore';
import { SettingsPath } from '../types/generated/settings';
import { useCallback } from 'react';

/**
 * Hook for performance-related settings with selective access
 * Provides optimized access to system performance settings
 */
export function usePerformanceSettings() {
  const { set, batchSet } = useSettingSetter();
  
  // Core performance settings
  const debugMode = useSelectiveSetting<boolean>('system.debug');
  const performanceMonitoring = useSelectiveSetting<boolean>('system.performance.monitoring');
  const memoryOptimization = useSelectiveSetting<boolean>('system.performance.memoryOptimization');
  const cacheEnabled = useSelectiveSetting<boolean>('system.performance.cache');
  const cacheTTL = useSelectiveSetting<number>('system.performance.cacheTTL');
  
  // WebSocket performance settings
  const wsReconnectAttempts = useSelectiveSetting<number>('system.websocket.reconnectAttempts');
  const wsReconnectInterval = useSelectiveSetting<number>('system.websocket.reconnectInterval');
  const wsHeartbeatInterval = useSelectiveSetting<number>('system.websocket.heartbeatInterval');
  const wsBufferSize = useSelectiveSetting<number>('system.websocket.bufferSize');
  
  // Rendering performance settings
  const renderFrameRate = useSelectiveSetting<number>('system.performance.rendering.frameRate');
  const renderBatchSize = useSelectiveSetting<number>('system.performance.rendering.batchSize');
  const renderCullingEnabled = useSelectiveSetting<boolean>('system.performance.rendering.culling');
  const renderLODEnabled = useSelectiveSetting<boolean>('system.performance.rendering.lod');
  
  // Memory management settings
  const gcThreshold = useSelectiveSetting<number>('system.performance.memory.gcThreshold');
  const maxCacheSize = useSelectiveSetting<number>('system.performance.memory.maxCacheSize');
  const preloadEnabled = useSelectiveSetting<boolean>('system.performance.memory.preload');
  const lazyLoadingEnabled = useSelectiveSetting<boolean>('system.performance.memory.lazyLoading');
  
  // Helper functions for performance tuning
  const updatePerformanceSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.performance.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateWebSocketSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.websocket.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateRenderingSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.performance.rendering.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateMemorySetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.performance.memory.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  // Performance preset functions
  const setPerformancePreset = useCallback(async (preset: 'low' | 'medium' | 'high' | 'ultra') => {
    const presets = {
      low: [
        { path: 'system.performance.rendering.frameRate', value: 30 },
        { path: 'system.performance.rendering.batchSize', value: 50 },
        { path: 'system.performance.rendering.culling', value: true },
        { path: 'system.performance.rendering.lod', value: true },
        { path: 'system.performance.memoryOptimization', value: true },
        { path: 'system.performance.cache', value: true },
        { path: 'system.performance.memory.lazyLoading', value: true },
        { path: 'system.performance.memory.preload', value: false }
      ],
      medium: [
        { path: 'system.performance.rendering.frameRate', value: 45 },
        { path: 'system.performance.rendering.batchSize', value: 100 },
        { path: 'system.performance.rendering.culling', value: true },
        { path: 'system.performance.rendering.lod', value: true },
        { path: 'system.performance.memoryOptimization', value: true },
        { path: 'system.performance.cache', value: true },
        { path: 'system.performance.memory.lazyLoading', value: true },
        { path: 'system.performance.memory.preload', value: false }
      ],
      high: [
        { path: 'system.performance.rendering.frameRate', value: 60 },
        { path: 'system.performance.rendering.batchSize', value: 200 },
        { path: 'system.performance.rendering.culling', value: false },
        { path: 'system.performance.rendering.lod', value: false },
        { path: 'system.performance.memoryOptimization', value: false },
        { path: 'system.performance.cache', value: true },
        { path: 'system.performance.memory.lazyLoading', value: false },
        { path: 'system.performance.memory.preload', value: true }
      ],
      ultra: [
        { path: 'system.performance.rendering.frameRate', value: 120 },
        { path: 'system.performance.rendering.batchSize', value: 500 },
        { path: 'system.performance.rendering.culling', value: false },
        { path: 'system.performance.rendering.lod', value: false },
        { path: 'system.performance.memoryOptimization', value: false },
        { path: 'system.performance.cache', value: true },
        { path: 'system.performance.memory.lazyLoading', value: false },
        { path: 'system.performance.memory.preload', value: true }
      ]
    };
    
    await batchSet(presets[preset]);
  }, [batchSet]);
  
  const optimizeForBattery = useCallback(async () => {
    await batchSet([
      { path: 'system.performance.rendering.frameRate', value: 30 },
      { path: 'system.performance.memoryOptimization', value: true },
      { path: 'system.performance.rendering.culling', value: true },
      { path: 'system.performance.rendering.lod', value: true },
      { path: 'system.performance.memory.lazyLoading', value: true },
      { path: 'system.websocket.heartbeatInterval', value: 30000 },
      { path: 'system.performance.monitoring', value: false }
    ]);
  }, [batchSet]);
  
  const optimizeForDesktop = useCallback(async () => {
    await batchSet([
      { path: 'system.performance.rendering.frameRate', value: 60 },
      { path: 'system.performance.memoryOptimization', value: false },
      { path: 'system.performance.rendering.culling', value: false },
      { path: 'system.performance.rendering.lod', value: false },
      { path: 'system.performance.memory.preload', value: true },
      { path: 'system.websocket.heartbeatInterval', value: 10000 },
      { path: 'system.performance.monitoring', value: true }
    ]);
  }, [batchSet]);
  
  return {
    // Core performance settings
    debugMode,
    performanceMonitoring,
    memoryOptimization,
    cacheEnabled,
    cacheTTL,
    
    // WebSocket settings
    webSocket: {
      reconnectAttempts: wsReconnectAttempts,
      reconnectInterval: wsReconnectInterval,
      heartbeatInterval: wsHeartbeatInterval,
      bufferSize: wsBufferSize
    },
    
    // Rendering settings
    rendering: {
      frameRate: renderFrameRate,
      batchSize: renderBatchSize,
      cullingEnabled: renderCullingEnabled,
      lodEnabled: renderLODEnabled
    },
    
    // Memory settings
    memory: {
      gcThreshold: gcThreshold,
      maxCacheSize: maxCacheSize,
      preloadEnabled: preloadEnabled,
      lazyLoadingEnabled: lazyLoadingEnabled
    },
    
    // Update functions
    updatePerformanceSetting,
    updateWebSocketSetting,
    updateRenderingSetting,
    updateMemorySetting,
    
    // Preset functions
    setPerformancePreset,
    optimizeForBattery,
    optimizeForDesktop
  };
}

/**
 * Hook for monitoring performance metrics
 */
export function usePerformanceMonitoring() {
  const performanceEnabled = useSelectiveSetting<boolean>('system.performance.monitoring');
  const metricsInterval = useSelectiveSetting<number>('system.performance.metricsInterval');
  const fpsCounter = useSelectiveSetting<boolean>('system.performance.showFPS');
  const memoryUsage = useSelectiveSetting<boolean>('system.performance.showMemory');
  const networkLatency = useSelectiveSetting<boolean>('system.performance.showLatency');
  
  const { set } = useSettingSetter();
  
  const updateMonitoringSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.performance.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  return {
    performanceEnabled,
    metricsInterval,
    fpsCounter,
    memoryUsage,
    networkLatency,
    updateMonitoringSetting
  };
}

/**
 * Hook for adaptive performance management
 * Automatically adjusts settings based on system capabilities
 */
export function useAdaptivePerformance() {
  const adaptiveEnabled = useSelectiveSetting<boolean>('system.performance.adaptive.enabled');
  const autoQualityAdjust = useSelectiveSetting<boolean>('system.performance.adaptive.autoQuality');
  const targetFPS = useSelectiveSetting<number>('system.performance.adaptive.targetFPS');
  const fpsThreshold = useSelectiveSetting<number>('system.performance.adaptive.fpsThreshold');
  
  const { set, batchSet } = useSettingSetter();
  
  const updateAdaptiveSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.performance.adaptive.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const enableAdaptivePerformance = useCallback(async () => {
    await batchSet([
      { path: 'system.performance.adaptive.enabled', value: true },
      { path: 'system.performance.adaptive.autoQuality', value: true },
      { path: 'system.performance.adaptive.targetFPS', value: 60 },
      { path: 'system.performance.adaptive.fpsThreshold', value: 55 }
    ]);
  }, [batchSet]);
  
  const disableAdaptivePerformance = useCallback(async () => {
    await set('system.performance.adaptive.enabled', false);
  }, [set]);
  
  return {
    adaptiveEnabled,
    autoQualityAdjust,
    targetFPS,
    fpsThreshold,
    updateAdaptiveSetting,
    enableAdaptivePerformance,
    disableAdaptivePerformance
  };
}