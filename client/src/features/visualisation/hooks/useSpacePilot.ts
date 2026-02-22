import { useEffect, useState, useRef, useCallback } from 'react';
import { useThree } from '@react-three/fiber';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { SpacePilotController, SpacePilotConfig, defaultSpacePilotConfig } from '../controls/SpacePilotController';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('useSpacePilot');

// Type alias for OrbitControls
type OrbitControls = OrbitControlsImpl;

import { SpaceDriver } from '../../../services/SpaceDriverService';

export interface SpacePilotOptions {
  enabled?: boolean;
  config?: Partial<SpacePilotConfig>;
  orbitControlsRef?: React.RefObject<OrbitControls | null>;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onModeChange?: (mode: 'camera' | 'object' | 'navigation') => void;
}

export interface SpacePilotHookReturn {
  isConnected: boolean;
  isSupported: boolean;
  currentMode: 'camera' | 'object' | 'navigation';
  setMode: (mode: 'camera' | 'object' | 'navigation') => void;
  connect: () => Promise<void>;
  disconnect: () => void;
  calibrate: () => void;
  resetView: () => void;
  levelHorizon: () => void;
  updateConfig: (config: Partial<SpacePilotConfig>) => void;
  config: SpacePilotConfig;
}


export function useSpacePilot(options: SpacePilotOptions = {}): SpacePilotHookReturn {
  const {
    enabled = true,
    config: userConfig = {},
    orbitControlsRef,
    onConnect,
    onDisconnect,
    onModeChange
  } = options;

  const { camera, scene, gl } = useThree();
  const [isConnected, setIsConnected] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [currentMode, setCurrentMode] = useState<'camera' | 'object' | 'navigation'>('camera');
  
  const controllerRef = useRef<SpacePilotController | null>(null);
  const configRef = useRef<SpacePilotConfig>({ ...defaultSpacePilotConfig, ...userConfig });
  
  
  const settings = useSettingsStore(state => state.settings);
  const spacePilotSettings = settings?.visualisation?.spacePilot;

  
  useEffect(() => {
    setIsSupported('hid' in navigator);
  }, []);

  
  useEffect(() => {
    if (!enabled || !camera || !isSupported) return;

    
    const mergedConfig = {
      ...defaultSpacePilotConfig,
      ...spacePilotSettings,
      ...userConfig
    };
    configRef.current = mergedConfig;

    
    const orbitControls = orbitControlsRef?.current || undefined;

    
    const controller = new SpacePilotController(
      camera,
      mergedConfig,
      orbitControls as OrbitControls | undefined
    );
    controllerRef.current = controller;

    
    setCurrentMode(mergedConfig.mode);

    return () => {
      controller.stop();
      controllerRef.current = null;
    };
  }, [enabled, camera, orbitControlsRef, isSupported, spacePilotSettings, userConfig]);

  
  useEffect(() => {
    if (!enabled || !isSupported || !controllerRef.current) return;

    const handleTranslate = (event: CustomEvent) => {
      controllerRef.current?.handleTranslation(event.detail);
    };

    const handleRotate = (event: CustomEvent) => {
      controllerRef.current?.handleRotation(event.detail);
    };

    const handleButtons = (event: CustomEvent) => {
      controllerRef.current?.handleButtons(event.detail);
    };

    const handleConnect = (event: CustomEvent) => {
      setIsConnected(true);
      controllerRef.current?.start();
      onConnect?.();
    };

    const handleDisconnect = () => {
      setIsConnected(false);
      controllerRef.current?.stop();
      onDisconnect?.();
    };

    
    SpaceDriver.addEventListener('translate', handleTranslate as EventListener);
    SpaceDriver.addEventListener('rotate', handleRotate as EventListener);
    SpaceDriver.addEventListener('buttons', handleButtons as EventListener);
    SpaceDriver.addEventListener('connect', handleConnect as EventListener);
    SpaceDriver.addEventListener('disconnect', handleDisconnect as EventListener);

    
    
    

    return () => {
      SpaceDriver.removeEventListener('translate', handleTranslate as EventListener);
      SpaceDriver.removeEventListener('rotate', handleRotate as EventListener);
      SpaceDriver.removeEventListener('buttons', handleButtons as EventListener);
      SpaceDriver.removeEventListener('connect', handleConnect as EventListener);
      SpaceDriver.removeEventListener('disconnect', handleDisconnect as EventListener);
    };
  }, [enabled, isSupported, onConnect, onDisconnect]);

  
  const connect = useCallback(async () => {
    if (!isSupported) {
      logger.error('WebHID is not supported in this browser');
      return;
    }

    try {
      await SpaceDriver.scan();
    } catch (error) {
      logger.error('Failed to connect to SpacePilot:', error);
    }
  }, [isSupported]);

  
  const disconnect = useCallback(() => {
    SpaceDriver.disconnect();
  }, []);

  
  const calibrate = useCallback(() => {
    controllerRef.current?.stop();
    controllerRef.current?.start();
  }, []);

  
  const resetView = useCallback(() => {
    const orbitControls = orbitControlsRef?.current;
    if (orbitControls && 'reset' in orbitControls) {
      orbitControls.reset();
    } else if (camera) {
      camera.position.set(0, 10, 20);
      camera.lookAt(0, 0, 0);
    }
  }, [camera, orbitControlsRef]);

  // Level the horizon without resetting position or look direction
  const levelHorizon = useCallback(() => {
    controllerRef.current?.levelHorizon();
  }, []);

  
  const setMode = useCallback((mode: 'camera' | 'object' | 'navigation') => {
    setCurrentMode(mode);
    controllerRef.current?.setMode(mode);
    onModeChange?.(mode);
    
    
    const updateSettingsFn = useSettingsStore.getState().updateSettings;
    updateSettingsFn((draft) => {
      const d = draft as unknown as Record<string, unknown>;
      if (!d.visualisation) d.visualisation = {};
      const vis = d.visualisation as Record<string, unknown>;
      if (!vis.spacePilot) vis.spacePilot = {};
      (vis.spacePilot as Record<string, unknown>).mode = mode;
    });
  }, [onModeChange]);

  
  const updateConfig = useCallback((newConfig: Partial<SpacePilotConfig>) => {
    const mergedConfig = { ...configRef.current, ...newConfig };
    configRef.current = mergedConfig;
    controllerRef.current?.updateConfig(mergedConfig);
    
    
    const updateSettingsFn = useSettingsStore.getState().updateSettings;
    updateSettingsFn((draft) => {
      const d = draft as unknown as Record<string, unknown>;
      if (!d.visualisation) d.visualisation = {};
      (d.visualisation as Record<string, unknown>).spacePilot = mergedConfig;
    });
  }, []);

  return {
    isConnected,
    isSupported,
    currentMode,
    setMode,
    connect,
    disconnect,
    calibrate,
    resetView,
    levelHorizon,
    updateConfig,
    config: configRef.current
  };
}