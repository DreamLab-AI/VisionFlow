import { useEffect, useState, useRef, useCallback } from 'react';
import { useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { SpacePilotController, SpacePilotConfig, defaultSpacePilotConfig } from '../controls/SpacePilotController';
import { useSettingsStore } from '../../../store/settingsStore';

import { SpaceDriver } from '../../../services/SpaceDriverService';

export interface SpacePilotOptions {
  enabled?: boolean;
  config?: Partial<SpacePilotConfig>;
  orbitControlsRef?: React.RefObject<any>;
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

    
    SpaceDriver.addEventListener('translate', handleTranslate);
    SpaceDriver.addEventListener('rotate', handleRotate);
    SpaceDriver.addEventListener('buttons', handleButtons);
    SpaceDriver.addEventListener('connect', handleConnect);
    SpaceDriver.addEventListener('disconnect', handleDisconnect);

    
    
    

    return () => {
      SpaceDriver.removeEventListener('translate', handleTranslate);
      SpaceDriver.removeEventListener('rotate', handleRotate);
      SpaceDriver.removeEventListener('buttons', handleButtons);
      SpaceDriver.removeEventListener('connect', handleConnect);
      SpaceDriver.removeEventListener('disconnect', handleDisconnect);
    };
  }, [enabled, isSupported, onConnect, onDisconnect]);

  
  const connect = useCallback(async () => {
    if (!isSupported) {
      console.error('WebHID is not supported in this browser');
      return;
    }

    try {
      await SpaceDriver.scan();
    } catch (error) {
      console.error('Failed to connect to SpacePilot:', error);
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

  
  const setMode = useCallback((mode: 'camera' | 'object' | 'navigation') => {
    setCurrentMode(mode);
    controllerRef.current?.setMode(mode);
    onModeChange?.(mode);
    
    
    const updateSettings = useSettingsStore.getState().updateSettings;
    updateSettings({
      visualisation: {
        spacePilot: {
          mode
        }
      }
    });
  }, [onModeChange]);

  
  const updateConfig = useCallback((newConfig: Partial<SpacePilotConfig>) => {
    const mergedConfig = { ...configRef.current, ...newConfig };
    configRef.current = mergedConfig;
    controllerRef.current?.updateConfig(mergedConfig);
    
    
    const updateSettings = useSettingsStore.getState().updateSettings;
    updateSettings({
      visualisation: {
        spacePilot: mergedConfig
      }
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
    updateConfig,
    config: configRef.current
  };
}