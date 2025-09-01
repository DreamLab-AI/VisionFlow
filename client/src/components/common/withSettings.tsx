import React, { ComponentType, forwardRef } from 'react';
import { useSelectiveSetting, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { SettingsPath } from '@/types/generated/settings';

/**
 * Settings injection interface for HOC
 */
export interface WithSettingsProps {
  settings: SettingsInjectedProps;
}

export interface SettingsInjectedProps {
  // Direct access methods
  get: <T>(path: SettingsPath) => T | undefined;
  set: (path: SettingsPath, value: any) => Promise<void>;
  batchSet: (updates: { path: SettingsPath; value: any }[]) => Promise<void>;
  
  // Common settings shortcuts
  theme: string | undefined;
  fontSize: string | undefined;
  debugMode: boolean | undefined;
  
  // Utility functions
  updateSetting: <T>(path: SettingsPath, value: T) => Promise<void>;
  updateMultiple: (updates: Record<SettingsPath, any>) => Promise<void>;
  
  // Boolean toggles
  toggleBoolean: (path: SettingsPath) => Promise<void>;
}

/**
 * Configuration for withSettings HOC
 */
export interface WithSettingsConfig {
  // Specific settings to inject
  inject?: SettingsPath[];
  
  // Common settings shortcuts to include
  includeTheme?: boolean;
  includeFontSize?: boolean;
  includeDebugMode?: boolean;
  
  // Custom settings mapping
  customMappings?: Record<string, SettingsPath>;
  
  // Display name for debugging
  displayName?: string;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: WithSettingsConfig = {
  includeTheme: true,
  includeFontSize: true,
  includeDebugMode: false
};

/**
 * Higher-Order Component that injects settings into wrapped component
 * @param Component - Component to wrap
 * @param config - Configuration for settings injection
 * @returns Enhanced component with settings props
 */
export function withSettings<P extends WithSettingsProps>(
  Component: ComponentType<P>,
  config: WithSettingsConfig = {}
) {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  
  const WithSettingsComponent = forwardRef<any, Omit<P, 'settings'>>((props, ref) => {
    const { set, batchSet } = useSettingSetter();
    
    // Inject common settings based on config
    const theme = finalConfig.includeTheme ? useSelectiveSetting<string>('ui.theme') : undefined;
    const fontSize = finalConfig.includeFontSize ? useSelectiveSetting<string>('ui.fontSize') : undefined;
    const debugMode = finalConfig.includeDebugMode ? useSelectiveSetting<boolean>('system.debug') : undefined;
    
    // Inject specific settings
    const injectedSettings: Record<string, any> = {};
    if (finalConfig.inject) {
      finalConfig.inject.forEach(path => {
        // Create a key from the path (e.g., 'ui.theme' -> 'uiTheme')
        const key = path.replace(/\./g, '').replace(/[A-Z]/g, letter => letter.toLowerCase());
        // eslint-disable-next-line react-hooks/rules-of-hooks\n        injectedSettings[key] = useSelectiveSetting(path);\n      });\n    }\n    \n    // Inject custom mappings\n    if (finalConfig.customMappings) {\n      Object.entries(finalConfig.customMappings).forEach(([key, path]) => {\n        // eslint-disable-next-line react-hooks/rules-of-hooks\n        injectedSettings[key] = useSelectiveSetting(path);\n      });\n    }\n    \n    // Direct access getter (limited functionality in HOC)\n    const get = React.useCallback(<T>(path: SettingsPath): T | undefined => {\n      // This is a limitation of the HOC approach - we can't dynamically use hooks\n      // In practice, you'd need to pre-define all paths you might need\n      console.warn('withSettings: get() has limitations. Consider using hooks directly.');\n      return undefined;\n    }, []);\n    \n    // Update functions\n    const updateSetting = React.useCallback(async <T>(path: SettingsPath, value: T) => {\n      await set(path, value);\n    }, [set]);\n    \n    const updateMultiple = React.useCallback(async (updates: Record<SettingsPath, any>) => {\n      const batchUpdates = Object.entries(updates).map(([path, value]) => ({\n        path: path as SettingsPath,\n        value\n      }));\n      \n      await batchSet(batchUpdates);\n    }, [batchSet]);\n    \n    const toggleBoolean = React.useCallback(async (path: SettingsPath) => {\n      // This requires getting the current value, which is problematic in HOC\n      // A real implementation would need access to the store\n      console.warn('withSettings: toggleBoolean() requires store access. Consider using hooks directly.');\n    }, []);\n    \n    // Create settings object to inject\n    const settings: SettingsInjectedProps = {\n      get,\n      set,\n      batchSet,\n      theme,\n      fontSize,\n      debugMode,\n      updateSetting,\n      updateMultiple,\n      toggleBoolean,\n      ...injectedSettings\n    };\n    \n    return <Component {...(props as P)} settings={settings} ref={ref} />;\n  });\n  \n  // Set display name for debugging\n  const componentName = Component.displayName || Component.name || 'Component';\n  WithSettingsComponent.displayName = finalConfig.displayName || `withSettings(${componentName})`;\n  \n  return WithSettingsComponent;\n}\n\n/**\n * Specialized HOC for theme-related settings\n */\nexport function withThemeSettings<P extends WithSettingsProps>(Component: ComponentType<P>) {\n  return withSettings(Component, {\n    includeTheme: true,\n    includeFontSize: true,\n    inject: [\n      'ui.accentColor',\n      'ui.borderRadius',\n      'ui.effects.animations',\n      'ui.effects.shadows'\n    ],\n    displayName: `withThemeSettings(${Component.displayName || Component.name})`\n  });\n}\n\n/**\n * Specialized HOC for performance-related settings\n */\nexport function withPerformanceSettings<P extends WithSettingsProps>(Component: ComponentType<P>) {\n  return withSettings(Component, {\n    includeDebugMode: true,\n    inject: [\n      'system.performance.memoryOptimization',\n      'system.performance.rendering.frameRate',\n      'system.performance.monitoring'\n    ],\n    customMappings: {\n      memoryOptimization: 'system.performance.memoryOptimization',\n      frameRate: 'system.performance.rendering.frameRate',\n      monitoring: 'system.performance.monitoring'\n    },\n    displayName: `withPerformanceSettings(${Component.displayName || Component.name})`\n  });\n}\n\n/**\n * Specialized HOC for visualization settings\n */\nexport function withVisualizationSettings<P extends WithSettingsProps>(Component: ComponentType<P>) {\n  return withSettings(Component, {\n    inject: [\n      'visualisation.quality',\n      'visualisation.antialiasing',\n      'visualisation.shadows',\n      'visualisation.camera.fov'\n    ],\n    customMappings: {\n      visualQuality: 'visualisation.quality',\n      antialiasing: 'visualisation.antialiasing',\n      shadows: 'visualisation.shadows',\n      cameraFov: 'visualisation.camera.fov'\n    },\n    displayName: `withVisualizationSettings(${Component.displayName || Component.name})`\n  });\n}\n\n/**\n * Specialized HOC for network/connectivity settings\n */\nexport function withNetworkSettings<P extends WithSettingsProps>(Component: ComponentType<P>) {\n  return withSettings(Component, {\n    inject: [\n      'system.websocket.autoReconnect',\n      'system.websocket.reconnectAttempts',\n      'system.http.timeout'\n    ],\n    customMappings: {\n      autoReconnect: 'system.websocket.autoReconnect',\n      reconnectAttempts: 'system.websocket.reconnectAttempts',\n      httpTimeout: 'system.http.timeout'\n    },\n    displayName: `withNetworkSettings(${Component.displayName || Component.name})`\n  });\n}\n\n/**\n * Utility type for extracting settings props from a component\n */\nexport type ExtractSettingsProps<P> = P extends WithSettingsProps ? P['settings'] : never;\n\n/**\n * Utility function to create a settings-aware component\n * Alternative to HOC that uses render props pattern\n */\nexport function SettingsWrapper<T = any>({\n  children,\n  inject = [],\n  customMappings = {}\n}: {\n  children: (settings: SettingsInjectedProps & T) => React.ReactNode;\n  inject?: SettingsPath[];\n  customMappings?: Record<keyof T, SettingsPath>;\n}) {\n  const { set, batchSet } = useSettingSetter();\n  \n  // Common settings\n  const theme = useSelectiveSetting<string>('ui.theme');\n  const fontSize = useSelectiveSetting<string>('ui.fontSize');\n  const debugMode = useSelectiveSetting<boolean>('system.debug');\n  \n  // Injected settings\n  const injectedSettings: any = {};\n  inject.forEach(path => {\n    const key = path.replace(/\\./g, '');\n    // eslint-disable-next-line react-hooks/rules-of-hooks\n    injectedSettings[key] = useSelectiveSetting(path);\n  });\n  \n  // Custom mappings\n  Object.entries(customMappings).forEach(([key, path]) => {\n    // eslint-disable-next-line react-hooks/rules-of-hooks\n    injectedSettings[key] = useSelectiveSetting(path);\n  });\n  \n  // Update functions\n  const updateSetting = React.useCallback(async (path: SettingsPath, value: any) => {\n    await set(path, value);\n  }, [set]);\n  \n  const updateMultiple = React.useCallback(async (updates: Record<SettingsPath, any>) => {\n    const batchUpdates = Object.entries(updates).map(([path, value]) => ({\n      path: path as SettingsPath,\n      value\n    }));\n    \n    await batchSet(batchUpdates);\n  }, [batchSet]);\n  \n  const settings: SettingsInjectedProps & T = {\n    get: () => undefined, // Limited in render prop pattern\n    set,\n    batchSet,\n    theme,\n    fontSize,\n    debugMode,\n    updateSetting,\n    updateMultiple,\n    toggleBoolean: async () => {},\n    ...injectedSettings\n  };\n  \n  return <>{children(settings)}</>;\n}\n\n/**\n * Hook version of settings injection for functional components\n * This is generally preferred over the HOC approach\n */\nexport function useSettingsInjection(config: WithSettingsConfig = {}) {\n  const finalConfig = { ...DEFAULT_CONFIG, ...config };\n  const { set, batchSet } = useSettingSetter();\n  \n  // Common settings\n  const theme = finalConfig.includeTheme ? useSelectiveSetting<string>('ui.theme') : undefined;\n  const fontSize = finalConfig.includeFontSize ? useSelectiveSetting<string>('ui.fontSize') : undefined;\n  const debugMode = finalConfig.includeDebugMode ? useSelectiveSetting<boolean>('system.debug') : undefined;\n  \n  // Injected settings\n  const injectedSettings: Record<string, any> = {};\n  if (finalConfig.inject) {\n    finalConfig.inject.forEach(path => {\n      const key = path.replace(/\\./g, '').replace(/[A-Z]/g, letter => letter.toLowerCase());\n      // eslint-disable-next-line react-hooks/rules-of-hooks\n      injectedSettings[key] = useSelectiveSetting(path);\n    });\n  }\n  \n  // Custom mappings\n  if (finalConfig.customMappings) {\n    Object.entries(finalConfig.customMappings).forEach(([key, path]) => {\n      // eslint-disable-next-line react-hooks/rules-of-hooks\n      injectedSettings[key] = useSelectiveSetting(path);\n    });\n  }\n  \n  // Update functions\n  const updateSetting = React.useCallback(async (path: SettingsPath, value: any) => {\n    await set(path, value);\n  }, [set]);\n  \n  const updateMultiple = React.useCallback(async (updates: Record<SettingsPath, any>) => {\n    const batchUpdates = Object.entries(updates).map(([path, value]) => ({\n      path: path as SettingsPath,\n      value\n    }));\n    \n    await batchSet(batchUpdates);\n  }, [batchSet]);\n  \n  return {\n    set,\n    batchSet,\n    theme,\n    fontSize,\n    debugMode,\n    updateSetting,\n    updateMultiple,\n    ...injectedSettings\n  };\n}