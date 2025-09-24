import { useCallback, useEffect, useRef, useMemo, useState } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { SettingsPath } from '@/features/settings/config/settings';
import { settingsApi } from '@/api/settingsApi';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useSelectiveSettingsStore');

// Request deduplication tracking
const requestMap = new Map<string, Promise<any>>();

// Response caching with TTL
interface CacheEntry<T> {
  value: T;
  timestamp: number;
  ttl: number;
}

const responseCache = new Map<string, CacheEntry<any>>();
const CACHE_TTL = 5000; // 5 seconds

// Debounce utilities
const debounceMap = new Map<string, ReturnType<typeof setTimeout>>();
const DEBOUNCE_DELAY = 50; // 50ms for optimal responsiveness

/**
 * Shallow equality check for preventing unnecessary re-renders
 */
function shallowEqual<T>(a: T, b: T): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return a === b;
  
  const keysA = Object.keys(a as any);
  const keysB = Object.keys(b as any);
  
  if (keysA.length !== keysB.length) return false;
  
  for (const key of keysA) {
    if (!(key in (b as any)) || (a as any)[key] !== (b as any)[key]) {
      return false;
    }
  }
  
  return true;
}

/**
 * Get cached response or return undefined if expired
 */
function getCachedResponse<T>(key: string): T | undefined {
  const entry = responseCache.get(key);
  if (!entry) return undefined;
  
  const now = Date.now();
  if (now - entry.timestamp > entry.ttl) {
    responseCache.delete(key);
    return undefined;
  }
  
  return entry.value;
}

/**
 * Cache a response with TTL
 */
function setCachedResponse<T>(key: string, value: T, ttl: number = CACHE_TTL): void {
  responseCache.set(key, {
    value,
    timestamp: Date.now(),
    ttl
  });
}

/**
 * Deduplicated API request - prevents duplicate requests for the same path
 */
async function getDedicatedSetting<T>(path: SettingsPath): Promise<T> {
  // Check cache first
  const cached = getCachedResponse<T>(path);
  if (cached !== undefined) {
    return cached;
  }
  
  // Check if request is already in progress
  if (requestMap.has(path)) {
    return requestMap.get(path) as Promise<T>;
  }
  
  // Create new request
  const request = settingsApi.getSettingByPath(path)
    .then(value => {
      // Cache the response
      setCachedResponse(path, value);
      return value as T;
    })
    .finally(() => {
      // Remove from request map when done
      requestMap.delete(path);
    });
  
  // Store request to prevent duplicates
  requestMap.set(path, request);
  return request;
}

/**
 * Debounced batch loader for multiple settings paths
 */
function debouncedBatchLoad(paths: string[], callback: (results: Record<string, any>) => void): void {
  const key = paths.sort().join('|');
  
  // Clear existing timeout
  if (debounceMap.has(key)) {
    clearTimeout(debounceMap.get(key)!);
  }
  
  // Set new timeout
  const timeout = setTimeout(async () => {
    try {
      // Check cache for all paths first
      const cachedResults: Record<string, any> = {};
      const uncachedPaths: string[] = [];
      
      for (const path of paths) {
        const cached = getCachedResponse(path);
        if (cached !== undefined) {
          cachedResults[path] = cached;
        } else {
          uncachedPaths.push(path);
        }
      }
      
      // Fetch uncached paths
      let apiResults: Record<string, any> = {};
      if (uncachedPaths.length > 0) {
        apiResults = await settingsApi.getSettingsByPaths(uncachedPaths);
        
        // Cache the results
        for (const [path, value] of Object.entries(apiResults)) {
          setCachedResponse(path, value);
        }
      }
      
      // Combine cached and API results
      const allResults = { ...cachedResults, ...apiResults };
      callback(allResults);
    } catch (error) {
      logger.error('Batch load failed:', error);
      callback({});
    } finally {
      debounceMap.delete(key);
    }
  }, DEBOUNCE_DELAY);
  
  debounceMap.set(key, timeout);
}

/**
 * Enhanced hook for selective single setting subscription with caching and deduplication
 */
export function useSelectiveSetting<T>(
  path: SettingsPath,
  options: {
    enableCache?: boolean;
    enableDeduplication?: boolean;
    fallbackToStore?: boolean;
  } = {}
): T {
  const {
    enableCache = true,
    enableDeduplication = true,
    fallbackToStore = true
  } = options;

  // Memoized selector to prevent unnecessary re-renders
  const selector = useCallback((state: any) => state.get<T>(path), [path]);
  const storeValue = useSettingsStore(selector, shallowEqual);
  
  const [apiValue, setApiValue] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  
  // Memoized path for stable reference
  const stablePath = useMemo(() => path, [path]);
  
  useEffect(() => {
    let mounted = true;
    
    // Subscribe to store changes for this specific path
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
    }
    
    unsubscribeRef.current = useSettingsStore.getState().subscribe(
      stablePath,
      () => {
        if (mounted) {
          // Force re-render is handled by zustand
        }
      },
      true
    );
    
    // Optionally fetch from API with caching/deduplication
    if (enableCache || enableDeduplication) {
      setLoading(true);
      
      const fetchValue = enableDeduplication 
        ? getDedicatedSetting<T>(stablePath)
        : settingsApi.getSettingByPath(stablePath);
      
      fetchValue
        .then(value => {
          if (mounted) {
            setApiValue(value);
          }
        })
        .catch(error => {
          logger.error(`Failed to fetch setting ${stablePath}:`, error);
        })
        .finally(() => {
          if (mounted) {
            setLoading(false);
          }
        });
    }
    
    return () => {
      mounted = false;
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
    };
  }, [stablePath, enableCache, enableDeduplication]);
  
  // Return API value if available, otherwise fall back to store
  return useMemo(() => {
    if (apiValue !== null && (enableCache || enableDeduplication)) {
      return apiValue;
    }
    return fallbackToStore ? storeValue : (null as unknown as T);
  }, [apiValue, storeValue, enableCache, enableDeduplication, fallbackToStore]);
}

/**
 * Enhanced hook for selective multiple settings subscription with batched loading
 */
export function useSelectiveSettings<T extends Record<string, any>>(
  paths: Record<keyof T, SettingsPath>,
  options: {
    enableBatchLoading?: boolean;
    enableCache?: boolean;
    fallbackToStore?: boolean;
  } = {}
): T {
  const {
    enableBatchLoading = true,
    enableCache = true,
    fallbackToStore = true
  } = options;

  // Memoized paths for stable references
  const stablePaths = useMemo(() => paths, [JSON.stringify(paths)]);
  
  // Memoized selectors for each path
  const selectors = useMemo(() => {
    const sels: Record<string, (state: any) => any> = {};
    for (const [key, path] of Object.entries(stablePaths)) {
      sels[key] = (state: any) => state.get(path);
    }
    return sels;
  }, [stablePaths]);
  
  // Get store values with shallow equality
  const storeValues = useSettingsStore(
    state => {
      const values = {} as T;
      for (const [key, selector] of Object.entries(selectors)) {
        values[key as keyof T] = selector(state);
      }
      return values;
    },
    shallowEqual
  );
  
  const [apiValues, setApiValues] = useState<Partial<T>>({});
  const [loading, setLoading] = useState(false);
  const unsubscribesRef = useRef<(() => void)[]>([]);
  
  useEffect(() => {
    let mounted = true;
    
    // Clean up previous subscriptions
    unsubscribesRef.current.forEach(unsub => unsub());
    unsubscribesRef.current = [];
    
    // Subscribe to each path individually
    for (const [key, path] of Object.entries(stablePaths)) {
      const unsubscribe = useSettingsStore.getState().subscribe(
        path,
        () => {
          if (mounted) {
            // Re-render handled by zustand
          }
        },
        true
      );
      unsubscribesRef.current.push(unsubscribe);
    }
    
    // Batch load from API if enabled
    if (enableBatchLoading) {
      setLoading(true);
      const pathsList = Object.values(stablePaths);
      
      debouncedBatchLoad(pathsList, (results) => {
        if (mounted) {
          const mappedResults = {} as Partial<T>;
          for (const [key, path] of Object.entries(stablePaths)) {
            if (results[path] !== undefined) {
              mappedResults[key as keyof T] = results[path];
            }
          }
          setApiValues(mappedResults);
          setLoading(false);
        }
      });
    }
    
    return () => {
      mounted = false;
      unsubscribesRef.current.forEach(unsub => unsub());
      unsubscribesRef.current = [];
    };
  }, [stablePaths, enableBatchLoading]);
  
  // Merge API values with store values
  return useMemo(() => {
    const result = {} as T;
    for (const key of Object.keys(stablePaths) as (keyof T)[]) {
      if (apiValues[key] !== undefined && (enableBatchLoading || enableCache)) {
        result[key] = apiValues[key]!;
      } else if (fallbackToStore) {
        result[key] = storeValues[key];
      } else {
        result[key] = null as any;
      }
    }
    return result;
  }, [apiValues, storeValues, stablePaths, enableBatchLoading, enableCache, fallbackToStore]);
}

/**
 * Enhanced setter hook with intelligent batching and debouncing
 */
export function useSettingSetter() {
  const updateSettings = useSettingsStore(state => state.updateSettings);
  const setByPath = useSettingsStore(state => state.setByPath);
  const batchUpdate = useSettingsStore(state => state.batchUpdate);
  
  // Memoized single setter with debouncing
  const debouncedSet = useCallback((path: SettingsPath, value: any) => {
    const key = `single_${path}`;
    
    // Clear existing timeout
    if (debounceMap.has(key)) {
      clearTimeout(debounceMap.get(key)!);
    }
    
    // Set new debounced timeout
    const timeout = setTimeout(() => {
      setByPath(path, value);
      debounceMap.delete(key);
    }, DEBOUNCE_DELAY);
    
    debounceMap.set(key, timeout);
  }, [setByPath]);
  
  // Memoized batch setter with intelligent batching
  const batchedSet = useCallback((updates: Record<SettingsPath, any>) => {
    const updateArray = Object.entries(updates).map(([path, value]) => ({
      path,
      value
    }));
    
    const key = `batch_${Object.keys(updates).sort().join('|')}`;
    
    // Clear existing timeout
    if (debounceMap.has(key)) {
      clearTimeout(debounceMap.get(key)!);
    }
    
    // Set new debounced timeout
    const timeout = setTimeout(() => {
      batchUpdate(updateArray);
      debounceMap.delete(key);
    }, DEBOUNCE_DELAY);
    
    debounceMap.set(key, timeout);
  }, [batchUpdate]);
  
  // Memoized immediate update (uses updateSettings directly)
  const immediateSet = useCallback((updater: (draft: any) => void) => {
    updateSettings(updater);
  }, [updateSettings]);
  
  return useMemo(() => ({
    set: debouncedSet,           // Single path setter with debouncing
    batchedSet,                  // Multiple paths setter with batching
    immediateSet,                // Immediate update for complex changes
    updateSettings               // Direct access to updateSettings
  }), [debouncedSet, batchedSet, immediateSet, updateSettings]);
}

/**
 * Enhanced subscription hook for side effects with memoization
 */
export function useSettingsSubscription(
  path: SettingsPath,
  callback: (value: any) => void,
  options: {
    immediate?: boolean;
    enableCache?: boolean;
    dependencies?: React.DependencyList;
  } = {}
) {
  const {
    immediate = true,
    enableCache = false,
    dependencies = []
  } = options;

  const callbackRef = useRef(callback);
  const stablePath = useMemo(() => path, [path]);
  
  // Update callback ref when it changes
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  useEffect(() => {
    let mounted = true;
    
    const handleChange = async () => {
      try {
        let value: any;
        
        if (enableCache) {
          // Use cached/deduplicated fetch
          value = await getDedicatedSetting(stablePath);
        } else {
          // Use store value
          value = useSettingsStore.getState().get(stablePath);
        }
        
        if (mounted) {
          callbackRef.current(value);
        }
      } catch (error) {
        logger.error(`Subscription callback failed for path ${stablePath}:`, error);
      }
    };
    
    // Call immediately if requested
    if (immediate) {
      handleChange();
    }
    
    // Subscribe to changes
    const unsubscribe = useSettingsStore.getState().subscribe(
      stablePath,
      handleChange,
      false
    );
    
    return () => {
      mounted = false;
      unsubscribe();
    };
  }, [stablePath, enableCache, immediate, ...dependencies]);
}

/**
 * Enhanced selector hook with memoization and shallow equality
 */
export function useSettingsSelector<T>(
  selector: (settings: any) => T,
  options: {
    equalityFn?: (prev: T, next: T) => boolean;
    enableCache?: boolean;
    cacheTTL?: number;
  } = {}
): T {
  const {
    equalityFn = shallowEqual,
    enableCache = false,
    cacheTTL = CACHE_TTL
  } = options;

  // Memoized selector for stable reference
  const memoizedSelector = useCallback(selector, [selector.toString()]);
  
  // Use zustand selector with custom equality function
  const value = useSettingsStore(
    state => memoizedSelector(state.settings),
    equalityFn
  );
  
  // Optionally cache the computed value
  const cacheKey = useMemo(() => {
    if (!enableCache) return null;
    return `selector_${memoizedSelector.toString()}_${JSON.stringify(value)}`;
  }, [enableCache, memoizedSelector, value]);
  
  useEffect(() => {
    if (cacheKey && enableCache) {
      setCachedResponse(cacheKey, value, cacheTTL);
    }
  }, [cacheKey, enableCache, value, cacheTTL]);
  
  return value;
}

/**
 * Utility hook for clearing caches (useful for testing or manual cache invalidation)
 */
export function useCacheManager() {
  const clearCache = useCallback(() => {
    responseCache.clear();
    requestMap.clear();
    debounceMap.forEach(timeout => clearTimeout(timeout));
    debounceMap.clear();
    logger.info('Settings cache cleared');
  }, []);
  
  const getCacheStats = useCallback(() => {
    return {
      responseCacheSize: responseCache.size,
      activeRequests: requestMap.size,
      pendingDebounces: debounceMap.size
    };
  }, []);
  
  return useMemo(() => ({
    clearCache,
    getCacheStats
  }), [clearCache, getCacheStats]);
}

// Export cache utilities for testing
export const settingsHookUtils = {
  getCachedResponse,
  setCachedResponse,
  shallowEqual,
  responseCache,
  requestMap,
  debounceMap,
  CACHE_TTL,
  DEBOUNCE_DELAY
};