/**
 * Hook to easily access and update settings values
 * Provides a simple interface similar to useState for settings values
 */

import { useCallback } from 'react';
import { useVisualizationConfig } from '../providers/VisualizationConfigProvider';

/**
 * Hook to get and set a specific settings value
 * @param path - Dot-notation path to the settings value (e.g., 'visualisation.rendering.backgroundColor')
 * @param defaultValue - Optional default value if the settings value doesn't exist
 * @returns [value, setValue] tuple similar to useState
 */
export function useVisualizationValue<T = any>(
  path: string,
  defaultValue?: T
): [T, (value: T) => void] {
  const { getSetting, setSetting } = useVisualizationConfig();
  
  const value = getSetting<T>(path) ?? defaultValue;
  
  const setValue = useCallback((newValue: T) => {
    setSetting(path, newValue);
  }, [path, setSetting]);
  
  return [value, setValue];
}

/**
 * Hook to get multiple settings values at once
 * @param paths - Array of dot-notation paths
 * @returns Object with path as key and value as value
 */
export function useVisualizationValues<T extends Record<string, any>>(
  paths: string[]
): T {
  const { getSetting } = useVisualizationConfig();
  
  const values = {} as T;
  paths.forEach(path => {
    values[path as keyof T] = getSetting(path);
  });
  
  return values;
}

/**
 * Hook to get a whole section of the settings
 * @param section - Top-level section name (e.g., 'visualisation', 'system')
 * @returns The settings section object
 */
export function useVisualizationSection<T = any>(section: string): T {
  const { getSetting } = useVisualizationConfig();
  return getSetting<T>(section);
}

// Re-export the main hook for convenience
export { useVisualizationConfig } from '../providers/VisualizationConfigProvider';