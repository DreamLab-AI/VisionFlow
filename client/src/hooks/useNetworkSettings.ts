import { useSelectiveSetting, useSettingSetter } from './useSelectiveSettingsStore';
import { SettingsPath } from '../types/generated/settings';
import { useCallback } from 'react';

/**
 * Hook for network-related settings with selective access
 * Provides optimized access to network configuration settings
 */
export function useNetworkSettings() {
  const { set, batchSet } = useSettingSetter();
  
  // WebSocket settings
  const wsUrl = useSelectiveSetting<string>('system.websocket.url');
  const wsAutoReconnect = useSelectiveSetting<boolean>('system.websocket.autoReconnect');
  const wsReconnectAttempts = useSelectiveSetting<number>('system.websocket.reconnectAttempts');
  const wsReconnectInterval = useSelectiveSetting<number>('system.websocket.reconnectInterval');
  const wsTimeout = useSelectiveSetting<number>('system.websocket.timeout');
  const wsHeartbeat = useSelectiveSetting<boolean>('system.websocket.heartbeat');
  const wsHeartbeatInterval = useSelectiveSetting<number>('system.websocket.heartbeatInterval');
  const wsBufferSize = useSelectiveSetting<number>('system.websocket.bufferSize');
  const wsCompression = useSelectiveSetting<boolean>('system.websocket.compression');
  
  // HTTP settings
  const httpTimeout = useSelectiveSetting<number>('system.http.timeout');
  const httpRetries = useSelectiveSetting<number>('system.http.retries');
  const httpRetryDelay = useSelectiveSetting<number>('system.http.retryDelay');
  const httpCaching = useSelectiveSetting<boolean>('system.http.caching');
  const httpCacheTTL = useSelectiveSetting<number>('system.http.cacheTTL');
  
  // Backend settings
  const customBackendUrl = useSelectiveSetting<string>('system.customBackendUrl');
  const apiVersion = useSelectiveSetting<string>('system.apiVersion');
  const apiKey = useSelectiveSetting<string>('system.apiKey');
  
  // Voice/Audio network settings
  const voiceLatencyMode = useSelectiveSetting<'low' | 'normal' | 'high'>('system.voice.latencyMode');
  const voiceBufferSize = useSelectiveSetting<number>('system.voice.bufferSize');
  const voiceCompressionLevel = useSelectiveSetting<number>('system.voice.compressionLevel');
  const voiceAutoGain = useSelectiveSetting<boolean>('system.voice.autoGain');
  
  // Network quality settings
  const networkQuality = useSelectiveSetting<'auto' | 'low' | 'medium' | 'high'>('system.network.quality');
  const bandwidthLimit = useSelectiveSetting<number>('system.network.bandwidthLimit');
  const adaptiveBitrate = useSelectiveSetting<boolean>('system.network.adaptiveBitrate');
  const networkMonitoring = useSelectiveSetting<boolean>('system.network.monitoring');
  
  // Helper functions
  const updateWebSocketSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.websocket.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateHttpSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.http.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateVoiceSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.voice.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateNetworkSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `system.network.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  // Network preset configurations
  const setNetworkPreset = useCallback(async (preset: 'mobile' | 'wifi' | 'ethernet' | 'enterprise') => {
    const presets = {
      mobile: [
        // Conservative settings for mobile networks
        { path: 'system.websocket.reconnectInterval', value: 5000 },
        { path: 'system.websocket.heartbeatInterval', value: 30000 },
        { path: 'system.websocket.bufferSize', value: 1024 },
        { path: 'system.websocket.compression', value: true },
        { path: 'system.http.timeout', value: 10000 },
        { path: 'system.http.retries', value: 3 },
        { path: 'system.network.quality', value: 'low' },
        { path: 'system.network.adaptiveBitrate', value: true },
        { path: 'system.voice.latencyMode', value: 'high' },
        { path: 'system.voice.compressionLevel', value: 8 }
      ],
      wifi: [
        // Balanced settings for WiFi
        { path: 'system.websocket.reconnectInterval', value: 3000 },
        { path: 'system.websocket.heartbeatInterval', value: 15000 },
        { path: 'system.websocket.bufferSize', value: 4096 },
        { path: 'system.websocket.compression', value: true },
        { path: 'system.http.timeout', value: 8000 },
        { path: 'system.http.retries', value: 2 },
        { path: 'system.network.quality', value: 'medium' },
        { path: 'system.network.adaptiveBitrate', value: true },
        { path: 'system.voice.latencyMode', value: 'normal' },
        { path: 'system.voice.compressionLevel', value: 5 }
      ],
      ethernet: [
        // High performance settings for wired connections
        { path: 'system.websocket.reconnectInterval', value: 1000 },
        { path: 'system.websocket.heartbeatInterval', value: 10000 },
        { path: 'system.websocket.bufferSize', value: 8192 },
        { path: 'system.websocket.compression', value: false },
        { path: 'system.http.timeout', value: 5000 },
        { path: 'system.http.retries', value: 1 },
        { path: 'system.network.quality', value: 'high' },
        { path: 'system.network.adaptiveBitrate', value: false },
        { path: 'system.voice.latencyMode', value: 'low' },
        { path: 'system.voice.compressionLevel', value: 2 }
      ],
      enterprise: [
        // Enterprise/datacenter settings
        { path: 'system.websocket.reconnectInterval', value: 500 },
        { path: 'system.websocket.heartbeatInterval', value: 5000 },
        { path: 'system.websocket.bufferSize', value: 16384 },
        { path: 'system.websocket.compression', value: false },
        { path: 'system.http.timeout', value: 3000 },
        { path: 'system.http.retries', value: 1 },
        { path: 'system.network.quality', value: 'high' },
        { path: 'system.network.adaptiveBitrate', value: false },
        { path: 'system.voice.latencyMode', value: 'low' },
        { path: 'system.voice.compressionLevel', value: 0 }
      ]
    };
    
    await batchSet(presets[preset]);
  }, [batchSet]);
  
  // Optimize for different scenarios
  const optimizeForLatency = useCallback(async () => {
    await batchSet([
      { path: 'system.websocket.bufferSize', value: 512 },
      { path: 'system.websocket.compression', value: false },
      { path: 'system.voice.latencyMode', value: 'low' },
      { path: 'system.voice.bufferSize', value: 256 },
      { path: 'system.http.timeout', value: 3000 },
      { path: 'system.network.adaptiveBitrate', value: false }
    ]);
  }, [batchSet]);
  
  const optimizeForBandwidth = useCallback(async () => {
    await batchSet([
      { path: 'system.websocket.compression', value: true },
      { path: 'system.voice.compressionLevel', value: 9 },
      { path: 'system.http.caching', value: true },
      { path: 'system.network.adaptiveBitrate', value: true },
      { path: 'system.network.quality', value: 'low' }
    ]);
  }, [batchSet]);
  
  const optimizeForReliability = useCallback(async () => {
    await batchSet([
      { path: 'system.websocket.autoReconnect', value: true },
      { path: 'system.websocket.reconnectAttempts', value: 10 },
      { path: 'system.websocket.heartbeat', value: true },
      { path: 'system.http.retries', value: 5 },
      { path: 'system.http.retryDelay', value: 2000 },
      { path: 'system.network.monitoring', value: true }
    ]);
  }, [batchSet]);
  
  return {
    // WebSocket settings
    webSocket: {
      url: wsUrl,
      autoReconnect: wsAutoReconnect,
      reconnectAttempts: wsReconnectAttempts,
      reconnectInterval: wsReconnectInterval,
      timeout: wsTimeout,
      heartbeat: wsHeartbeat,
      heartbeatInterval: wsHeartbeatInterval,
      bufferSize: wsBufferSize,
      compression: wsCompression
    },
    
    // HTTP settings
    http: {
      timeout: httpTimeout,
      retries: httpRetries,
      retryDelay: httpRetryDelay,
      caching: httpCaching,
      cacheTTL: httpCacheTTL
    },
    
    // Backend settings
    backend: {
      customUrl: customBackendUrl,
      apiVersion: apiVersion,
      apiKey: apiKey
    },
    
    // Voice settings
    voice: {
      latencyMode: voiceLatencyMode,
      bufferSize: voiceBufferSize,
      compressionLevel: voiceCompressionLevel,
      autoGain: voiceAutoGain
    },
    
    // Network quality settings
    network: {
      quality: networkQuality,
      bandwidthLimit: bandwidthLimit,
      adaptiveBitrate: adaptiveBitrate,
      monitoring: networkMonitoring
    },
    
    // Update functions
    updateWebSocketSetting,
    updateHttpSetting,
    updateVoiceSetting,
    updateNetworkSetting,
    
    // Preset functions
    setNetworkPreset,
    optimizeForLatency,
    optimizeForBandwidth,
    optimizeForReliability
  };
}

/**
 * Hook for connection status and monitoring
 */
export function useNetworkStatus() {
  const networkMonitoring = useSelectiveSetting<boolean>('system.network.monitoring');
  const connectionQuality = useSelectiveSetting<'poor' | 'fair' | 'good' | 'excellent'>('system.network.connectionQuality');
  const latency = useSelectiveSetting<number>('system.network.latency');
  const bandwidth = useSelectiveSetting<number>('system.network.bandwidth');
  const packetLoss = useSelectiveSetting<number>('system.network.packetLoss');
  
  const { set } = useSettingSetter();
  
  const updateNetworkStatus = useCallback(async (status: {
    quality?: 'poor' | 'fair' | 'good' | 'excellent';
    latency?: number;
    bandwidth?: number;
    packetLoss?: number;
  }) => {
    const updates = [];
    if (status.quality !== undefined) {
      updates.push({ path: 'system.network.connectionQuality', value: status.quality });
    }
    if (status.latency !== undefined) {
      updates.push({ path: 'system.network.latency', value: status.latency });
    }
    if (status.bandwidth !== undefined) {
      updates.push({ path: 'system.network.bandwidth', value: status.bandwidth });
    }
    if (status.packetLoss !== undefined) {
      updates.push({ path: 'system.network.packetLoss', value: status.packetLoss });
    }
    
    if (updates.length > 0) {
      const { batchSet } = useSettingSetter();
      await batchSet(updates);
    }
  }, []);
  
  return {
    networkMonitoring,
    connectionQuality,
    latency,
    bandwidth,
    packetLoss,
    updateNetworkStatus
  };
}