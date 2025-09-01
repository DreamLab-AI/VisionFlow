import { useCallback, useEffect, useRef } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { SettingsPath } from '@/types/generated/settings';
import { createLogger } from '@/utils/logger';

const logger = createLogger('useSelectiveSettingsStore');

/**
 * Custom hook for selective subscriptions to settings store
 * This hook optimizes re-renders by only subscribing to specific paths
 */
export function useSelectiveSetting<T>(path: SettingsPath): T {
  // Use zustand's built-in selector for reactive subscriptions
  return useSettingsStore(state => state.get<T>(path));
}

/**
 * Hook for subscribing to multiple settings paths
 * Returns an object with the current values
 */
export function useSelectiveSettings<T extends Record<string, any>>(
  paths: Record<keyof T, SettingsPath>
): T {
  // Create a selector that returns all requested values
  return useSettingsStore(
    useCallback(
      (state) => {
        const result = {} as T;
        for (const key in paths) {
          result[key] = state.get(paths[key]);
        }
        return result;
      },
      [paths]
    )
  );
}

/**
 * Hook for setting values with optimized updates
 * Returns a setter function that batches updates
 */
export function useSettingSetter() {
  const set = useSettingsStore(state => state.set);
  const batchSet = useSettingsStore(state => state.batchSet);
  
  const batchedSet = useCallback(async (updates: Record<SettingsPath, any>) => {
    const pathValuePairs = Object.entries(updates).map(([path, value]) => ({
      path: path as SettingsPath,
      value
    }));
    
    await batchSet(pathValuePairs);
  }, [batchSet]);
  
  return {
    set,
    batchedSet,
    batchSet // Expose batchSet directly for single updates
  };
}

/**
 * Hook for subscribing to settings changes with a callback
 * Useful for side effects when settings change
 */
export function useSettingsSubscription(
  path: SettingsPath,
  callback: (value: any) => void,
  dependencies: React.DependencyList = []
) {
  const callbackRef = useRef(callback);
  
  // Update callback ref when it changes
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  // Get the current value and subscribe to changes
  const value = useSelectiveSetting(path);
  
  // Effect for calling callback when value changes
  useEffect(() => {
    callbackRef.current(value);
  }, [value, ...dependencies]);
}

/**
 * Hook for getting settings with a selector function
 * This allows for derived state from settings
 */
export function useSettingsSelector<T>(
  selector: (partialSettings: any) => T,
  equalityFn?: (prev: T, next: T) => boolean
): T {
  return useSettingsStore(state => selector(state.partialSettings), equalityFn);
}