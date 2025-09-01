import { useSelectiveSetting, useSettingSetter, useSelectiveSettings } from './useSelectiveSettingsStore';
import { SettingsPath } from '../types/generated/settings';
import { useCallback } from 'react';

/**
 * General-purpose settings hook with selective access
 * Provides a unified interface for common settings operations
 */
export function useSettings() {
  const { set, batchSet } = useSettingSetter();
  
  return {
    // Direct access to selective setting functions
    get: useSelectiveSetting,
    set,
    batchSet,
    
    // Utility functions
    update: useCallback(async <T>(path: SettingsPath, value: T) => {
      await set(path, value);
    }, [set]),
    
    updateMultiple: useCallback(async (updates: Record<SettingsPath, any>) => {
      const batchUpdates = Object.entries(updates).map(([path, value]) => ({
        path: path as SettingsPath,
        value
      }));
      await batchSet(batchUpdates);
    }, [batchSet])
  };
}

/**
 * Hook for getting multiple settings at once
 * @param paths - Record mapping keys to settings paths
 * @returns Object with current values for all requested settings
 */
export function useMultipleSettings<T extends Record<string, any>>(
  paths: Record<keyof T, SettingsPath>
): T {
  return useSelectiveSettings(paths);
}

/**
 * Hook for a single setting with get/set operations
 * @param path - Settings path
 * @returns Tuple of [value, setter function]
 */
export function useSetting<T>(path: SettingsPath): [T | undefined, (value: T) => Promise<void>] {
  const value = useSelectiveSetting<T>(path);
  const { set } = useSettingSetter();
  
  const setValue = useCallback(async (newValue: T) => {
    await set(path, newValue);
  }, [path, set]);
  
  return [value, setValue];
}

/**
 * Hook for toggling boolean settings
 * @param path - Settings path for boolean value
 * @returns Tuple of [value, toggle function]
 */
export function useBooleanSetting(path: SettingsPath): [boolean | undefined, () => Promise<void>] {
  const value = useSelectiveSetting<boolean>(path);
  const { set } = useSettingSetter();
  
  const toggle = useCallback(async () => {
    await set(path, !value);
  }, [path, value, set]);
  
  return [value, toggle];
}

/**
 * Hook for settings with validation
 * @param path - Settings path
 * @param validator - Validation function
 * @returns Tuple of [value, validated setter function]
 */
export function useValidatedSetting<T>(
  path: SettingsPath, 
  validator: (value: T) => boolean
): [T | undefined, (value: T) => Promise<boolean>] {
  const value = useSelectiveSetting<T>(path);
  const { set } = useSettingSetter();
  
  const setValidated = useCallback(async (newValue: T): Promise<boolean> => {
    if (validator(newValue)) {
      await set(path, newValue);
      return true;
    }
    return false;
  }, [path, validator, set]);
  
  return [value, setValidated];
}

/**
 * Hook for settings with default values
 * @param path - Settings path
 * @param defaultValue - Default value if setting is undefined
 * @returns The setting value or default
 */
export function useSettingWithDefault<T>(path: SettingsPath, defaultValue: T): T {
  const value = useSelectiveSetting<T>(path);
  return value !== undefined ? value : defaultValue;
}

/**
 * Hook for derived settings (computed from other settings)
 * @param paths - Array of settings paths to depend on
 * @param compute - Function to compute derived value
 * @returns The computed value
 */
export function useDerivedSetting<T, K extends SettingsPath[]>(
  paths: K,
  compute: (values: (any | undefined)[]) => T
): T {
  const values = paths.map(path => useSelectiveSetting(path));
  return compute(values);
}

/**
 * Hook for temporary setting changes (with revert capability)
 * @param path - Settings path
 * @returns Object with current value and temporary change functions
 */
export function useTemporarySetting<T>(path: SettingsPath) {
  const currentValue = useSelectiveSetting<T>(path);
  const { set } = useSettingSetter();
  
  const setTemporary = useCallback(async (tempValue: T) => {
    // Store original value for potential revert
    const originalValue = currentValue;
    await set(path, tempValue);
    
    // Return function to revert to original value
    return async () => {
      if (originalValue !== undefined) {
        await set(path, originalValue);
      }
    };
  }, [path, currentValue, set]);
  
  return {
    currentValue,
    setTemporary
  };
}

/**
 * Hook for settings history/undo functionality
 * @param path - Settings path
 * @param historyLimit - Maximum number of history entries to keep
 * @returns Object with value, setter, and history functions
 */
export function useSettingWithHistory<T>(path: SettingsPath, historyLimit = 10) {
  const currentValue = useSelectiveSetting<T>(path);
  const { set } = useSettingSetter();
  
  // Note: In a real implementation, you'd want to store history in a more persistent way
  // This is a simplified version for demonstration
  
  const setWithHistory = useCallback(async (newValue: T) => {
    // In practice, you'd want to store the previous value in history before setting new value
    await set(path, newValue);
  }, [path, set]);
  
  return {
    value: currentValue,
    setValue: setWithHistory,
    // Additional history methods would be implemented here
    canUndo: false, // Placeholder
    undo: async () => {}, // Placeholder
    canRedo: false, // Placeholder
    redo: async () => {} // Placeholder
  };
}

/**
 * Hook for settings synchronization across components
 * Useful when multiple components need to stay in sync with the same setting
 * @param path - Settings path
 * @param syncKey - Unique key for this sync group
 * @returns Tuple of [value, synchronized setter]
 */
export function useSynchronizedSetting<T>(
  path: SettingsPath,
  syncKey: string
): [T | undefined, (value: T) => Promise<void>] {
  const value = useSelectiveSetting<T>(path);
  const { set } = useSettingSetter();
  
  const setSynchronized = useCallback(async (newValue: T) => {
    // Set the value - the selective subscription system will handle propagation
    await set(path, newValue);
    
    // Could emit custom events here for additional sync if needed
    window.dispatchEvent(new CustomEvent(`setting-sync-${syncKey}`, {
      detail: { path, value: newValue }
    }));
  }, [path, syncKey, set]);
  
  return [value, setSynchronized];
}