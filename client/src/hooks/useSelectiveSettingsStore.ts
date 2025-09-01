import { useCallback, useEffect, useRef, useMemo } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { SettingsPath } from '@/types/generated/settings';
import { createLogger } from '@/utils/logger';

const logger = createLogger('useSelectiveSettingsStore');

/**
 * Custom hook for selective subscriptions to settings store
 * This hook optimizes re-renders by only subscribing to specific paths
 */
export function useSelectiveSetting<T>(path: SettingsPath): T {
  // Ensure the path is loaded before accessing it
  const ensureLoaded = useSettingsStore(state => state.ensureLoaded);
  const loadedPaths = useSettingsStore(state => state.loadedPaths);
  const partialSettings = useSettingsStore(state => state.partialSettings);
  
  // Check if path is loaded and trigger load if needed (side effect in useEffect)
  useEffect(() => {
    const isPathLoaded = loadedPaths.has(path) || 
      [...loadedPaths].some(loadedPath => 
        path.startsWith(loadedPath + '.') || loadedPath.startsWith(path + '.')
      );
    
    if (!isPathLoaded) {
      logger.debug(`Path ${path} not loaded, triggering load`);
      ensureLoaded([path]).catch(error => {
        logger.error(`Failed to load path ${path}:`, error);
      });
    }
  }, [path, loadedPaths, ensureLoaded]);
  
  // Pure selector that just reads the value without side effects
  return useSettingsStore(
    useCallback(
      (state) => {
        if (!path?.trim()) {
          return state.partialSettings as unknown as T;
        }
        
        // Navigate the partial settings using the path
        const pathParts = path.split('.');
        let current: any = state.partialSettings;
        
        for (const part of pathParts) {
          if (current && typeof current === 'object' && part in current) {
            current = current[part];
          } else {
            return undefined as unknown as T;
          }
        }
        
        return current as T;
      },
      [path]
    )
  );
}

/**
 * Hook for subscribing to multiple settings paths
 * Returns an object with the current values
 */
export function useSelectiveSettings<T extends Record<string, any>>(
  paths: Record<keyof T, SettingsPath>
): T {
  const ensureLoaded = useSettingsStore(state => state.ensureLoaded);
  const loadedPaths = useSettingsStore(state => state.loadedPaths);
  
  // Track paths in a ref to avoid re-running effect unnecessarily
  const pathsRef = useRef(paths);
  pathsRef.current = paths;
  
  // Load paths if needed
  useEffect(() => {
    const pathsToLoad: SettingsPath[] = [];
    for (const key in pathsRef.current) {
      const path = pathsRef.current[key];
      const isPathLoaded = loadedPaths.has(path) || 
        [...loadedPaths].some(loadedPath => 
          path.startsWith(loadedPath + '.') || loadedPath.startsWith(path + '.')
        );
      
      if (!isPathLoaded) {
        pathsToLoad.push(path);
      }
    }
    
    if (pathsToLoad.length > 0) {
      logger.debug(`Loading paths: ${pathsToLoad.join(', ')}`);
      ensureLoaded(pathsToLoad).catch(error => {
        logger.error(`Failed to load paths:`, error);
      });
    }
  }, [loadedPaths, ensureLoaded]);
  
  return useSettingsStore((state) => {
    const result = {} as T;
    for (const key in paths) {
      const path = paths[key];
      if (!path?.trim()) {
        result[key] = state.partialSettings as any;
        continue;
      }
      
      // Navigate the partial settings using the path
      const pathParts = path.split('.');
      let current: any = state.partialSettings;
      
      for (const part of pathParts) {
        if (current && typeof current === 'object' && part in current) {
          current = current[part];
        } else {
          current = undefined;
          break;
        }
      }
      
      result[key] = current;
    }
    return result;
  });
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