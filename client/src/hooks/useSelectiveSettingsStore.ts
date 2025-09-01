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
 * 
 * IMPORTANT: The paths object MUST be stable (memoized) to avoid infinite loops
 */
export function useSelectiveSettings<T extends Record<string, any>>(
  paths: Record<keyof T, SettingsPath>
): T {
  const ensureLoaded = useSettingsStore(state => state.ensureLoaded);
  const loadedPaths = useSettingsStore(state => state.loadedPaths);
  
  // Create a stable key from paths for memoization
  const pathsKey = useMemo(() => {
    const keys = Object.keys(paths).sort();
    return keys.map(k => `${k}:${paths[k as keyof T]}`).join('|');
  }, [paths]);
  
  // Load paths if needed
  useEffect(() => {
    const pathsToLoad: SettingsPath[] = [];
    for (const key in paths) {
      const path = paths[key];
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
  }, [pathsKey, loadedPaths, ensureLoaded]); // Use stable key
  
  // Create a cached selector to avoid the getSnapshot warning
  const selectorRef = useRef<{
    lastState: any;
    lastResult: T;
    selector: (state: any) => T;
  } | null>(null);
  
  // Initialize or update the selector ref when paths change
  if (!selectorRef.current || selectorRef.current.selector.toString() !== pathsKey) {
    selectorRef.current = {
      lastState: null,
      lastResult: {} as T,
      selector: (state: any) => {
        // Return cached result if state hasn't changed
        if (selectorRef.current && selectorRef.current.lastState === state) {
          return selectorRef.current.lastResult;
        }
        
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
        
        // Cache the result
        if (selectorRef.current) {
          selectorRef.current.lastState = state;
          selectorRef.current.lastResult = result;
        }
        
        return result;
      }
    };
  }
  
  const selector = selectorRef.current.selector;
  
  // Use shallow equality to prevent unnecessary re-renders
  return useSettingsStore(selector, (prev, next) => {
    // Shallow compare the result objects
    const prevKeys = Object.keys(prev);
    const nextKeys = Object.keys(next);
    if (prevKeys.length !== nextKeys.length) return false;
    for (const key of prevKeys) {
      if (prev[key] !== next[key]) return false;
    }
    return true;
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