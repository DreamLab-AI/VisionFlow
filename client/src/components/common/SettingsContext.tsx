import React, { createContext, useContext, ReactNode, useCallback, useState, useEffect } from 'react';
import { useSelectiveSetting, useSettingSetter } from '../hooks/useSelectiveSettingsStore';

/**
 * Settings Context Interface
 * Provides a simplified interface for accessing settings without needing hooks everywhere
 */
interface SettingsContextValue {
  // Direct getters/setters
  get: <T>(path: string) => T | undefined;
  set: (path: string, value: any) => Promise<void>;
  batchSet: (updates: { path: string; value: any }[]) => Promise<void>;
  
  // Bulk operations  
  getMultiple: <T extends Record<string, any>>(paths: Record<keyof T, string>) => T;
  updateMultiple: (updates: Record<string, any>) => Promise<void>;
  
  // Common settings shortcuts
  theme: string | undefined;
  fontSize: string | undefined;
  debugMode: boolean | undefined;
  performanceMode: string | undefined;
  
  // Utility functions
  isLoading: boolean;
  hasChanges: boolean;
  
  // Event handlers
  onSettingChange: (path: string, handler: (value: any) => void) => () => void;
}

/**
 * Settings Context
 */
const SettingsContext = createContext<SettingsContextValue | null>(null);

/**
 * Settings Context Props
 */
interface SettingsContextProps {
  children: ReactNode;
}

/**
 * Settings Context Provider
 * Provides easy access to settings throughout the component tree
 */
export function SettingsContextProvider({ children }: SettingsContextProps) {
  const { set, batchSet } = useSettingSetter();
  
  // Common settings that are frequently accessed
  const theme = useSelectiveSetting<string>('ui.theme');
  const fontSize = useSelectiveSetting<string>('ui.fontSize');
  const debugMode = useSelectiveSetting<boolean>('system.debug');
  const performanceMode = useSelectiveSetting<string>('system.performance.mode');
  
  // Loading and change states (simplified)
  const isLoading = false;
  const hasChanges = false;
  
  // Direct getter function (simplified implementation)
  const get = useCallback(<T>(path: string): T | undefined => {
    try {
      const stored = localStorage.getItem(`settings.${path}`);
      return stored ? JSON.parse(stored) : undefined;
    } catch {
      return undefined;
    }
  }, []);
  
  // Bulk getter function
  const getMultiple = useCallback(<T extends Record<string, any>>(
    paths: Record<keyof T, string>
  ): T => {
    const result = {} as T;
    Object.entries(paths).forEach(([key, path]) => {
      (result as any)[key] = get(path);
    });
    return result;
  }, [get]);
  
  // Bulk setter function
  const updateMultiple = useCallback(async (updates: Record<string, any>) => {
    const batchUpdates = Object.entries(updates).map(([path, value]) => ({
      path,
      value
    }));
    
    await batchSet(batchUpdates);
  }, [batchSet]);
  
  // Event handler registration (simplified)
  const onSettingChange = useCallback((
    path: string,
    handler: (value: any) => void
  ): (() => void) => {
    // Return no-op unsubscribe function
    return () => {};
  }, []);
  
  const contextValue: SettingsContextValue = {
    get,
    set,
    batchSet,
    getMultiple,
    updateMultiple,
    theme,
    fontSize,
    debugMode,
    performanceMode,
    isLoading,
    hasChanges,
    onSettingChange
  };
  
  return (
    <SettingsContext.Provider value={contextValue}>
      {children}
    </SettingsContext.Provider>
  );
}

/**
 * Hook to use Settings Context
 */
export function useSettingsContext(): SettingsContextValue {
  const context = useContext(SettingsContext);
  
  if (!context) {
    throw new Error('useSettingsContext must be used within a SettingsContextProvider');
  }
  
  return context;
}

/**
 * Hook for getting a single setting with context
 */
export function useContextSetting<T>(path: string): [T | undefined, (value: T) => Promise<void>] {
  const { get, set } = useSettingsContext();
  const [value, setValue] = useState<T | undefined>(get<T>(path));
  
  useEffect(() => {
    setValue(get<T>(path));
  }, [get, path]);
  
  const setValueAndUpdate = useCallback(async (newValue: T) => {
    await set(path, newValue);
    setValue(newValue);
  }, [path, set]);
  
  return [value, setValueAndUpdate];
}

/**
 * Hook for boolean settings with toggle functionality
 */
export function useContextBooleanSetting(
  path: string
): [boolean | undefined, () => Promise<void>, (value: boolean) => Promise<void>] {
  const [value, setValue] = useContextSetting<boolean>(path);
  
  const toggle = useCallback(async () => {
    await setValue(!value);
  }, [value, setValue]);
  
  return [value, toggle, setValue];
}

/**
 * Hook for theme-related settings
 */
export function useThemeContext() {
  const { theme, fontSize, set } = useSettingsContext();
  
  const setTheme = useCallback(async (newTheme: string) => {
    await set('ui.theme', newTheme);
  }, [set]);
  
  const setFontSize = useCallback(async (newSize: string) => {
    await set('ui.fontSize', newSize);
  }, [set]);
  
  const toggleTheme = useCallback(async () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    await setTheme(newTheme);
  }, [theme, setTheme]);
  
  return {
    theme,
    fontSize,
    setTheme,
    setFontSize,
    toggleTheme
  };
}
