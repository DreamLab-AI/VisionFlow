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
const CACHE_TTL = 5000;
const MAX_CACHE_SIZE = 500;

// Debounce utilities
const debounceMap = new Map<string, ReturnType<typeof setTimeout>>();
const DEBOUNCE_DELAY = 50; 


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


function setCachedResponse<T>(key: string, value: T, ttl: number = CACHE_TTL): void {
  // Evict oldest entry when cache exceeds max size (simple LRU approximation)
  if (responseCache.size >= MAX_CACHE_SIZE && !responseCache.has(key)) {
    const oldestKey = responseCache.keys().next().value;
    if (oldestKey !== undefined) {
      responseCache.delete(oldestKey);
    }
  }
  responseCache.set(key, {
    value,
    timestamp: Date.now(),
    ttl
  });
}


async function getDedicatedSetting<T>(path: SettingsPath): Promise<T> {
  
  const cached = getCachedResponse<T>(path);
  if (cached !== undefined) {
    return cached;
  }
  
  
  if (requestMap.has(path)) {
    return requestMap.get(path) as Promise<T>;
  }
  
  
  const request = settingsApi.getSettingByPath(path)
    .then(value => {
      
      setCachedResponse(path, value);
      return value as T;
    })
    .finally(() => {
      
      requestMap.delete(path);
    });
  
  
  requestMap.set(path, request);
  return request;
}


function debouncedBatchLoad(paths: string[], callback: (results: Record<string, any>) => void): void {
  const key = paths.sort().join('|');
  
  
  if (debounceMap.has(key)) {
    clearTimeout(debounceMap.get(key)!);
  }
  
  
  const timeout = setTimeout(async () => {
    try {
      
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
      
      
      let apiResults: Record<string, any> = {};
      if (uncachedPaths.length > 0) {
        apiResults = await settingsApi.getSettingsByPaths(uncachedPaths);
        
        
        for (const [path, value] of Object.entries(apiResults)) {
          setCachedResponse(path, value);
        }
      }
      
      
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


  const selector = useCallback((state: any) => state.get(path) as T, [path]);
  const storeValue = useSettingsStore(selector) as T;
  
  const [apiValue, setApiValue] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  
  
  const stablePath = useMemo(() => path, [path]);
  
  useEffect(() => {
    let mounted = true;
    
    
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
    }
    
    unsubscribeRef.current = useSettingsStore.getState().subscribe(
      stablePath,
      () => {
        if (mounted) {
          
        }
      },
      true
    );
    
    
    if (enableCache || enableDeduplication) {
      setLoading(true);
      
      const fetchValue = enableDeduplication 
        ? getDedicatedSetting<T>(stablePath)
        : settingsApi.getSettingByPath(stablePath);
      
      fetchValue
        .then(value => {
          if (mounted) {
            setApiValue(value as T);
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
  
  
  return useMemo(() => {
    if (apiValue !== null && (enableCache || enableDeduplication)) {
      return apiValue;
    }
    return fallbackToStore ? storeValue : (null as unknown as T);
  }, [apiValue, storeValue, enableCache, enableDeduplication, fallbackToStore]);
}


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

  
  const stablePaths = useMemo(() => paths, [JSON.stringify(paths)]);
  
  
  const selectors = useMemo(() => {
    const sels: Record<string, (state: any) => any> = {};
    for (const [key, path] of Object.entries(stablePaths)) {
      sels[key] = (state: any) => state.get(path);
    }
    return sels;
  }, [stablePaths]);
  

  const storeValues = useSettingsStore(
    state => {
      const values = {} as T;
      for (const [key, selector] of Object.entries(selectors)) {
        values[key as keyof T] = selector(state);
      }
      return values;
    }
  );
  
  const [apiValues, setApiValues] = useState<Partial<T>>({});
  const [loading, setLoading] = useState(false);
  const unsubscribesRef = useRef<(() => void)[]>([]);
  
  useEffect(() => {
    let mounted = true;
    
    
    unsubscribesRef.current.forEach(unsub => unsub());
    unsubscribesRef.current = [];
    
    
    for (const [key, path] of Object.entries(stablePaths)) {
      const unsubscribe = useSettingsStore.getState().subscribe(
        path,
        () => {
          if (mounted) {
            
          }
        },
        true
      );
      unsubscribesRef.current.push(unsubscribe);
    }
    
    
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


export function useSettingSetter() {
  const updateSettings = useSettingsStore(state => state.updateSettings);
  const setByPath = useSettingsStore(state => state.setByPath);
  const batchUpdate = useSettingsStore(state => state.batchUpdate);
  
  
  const debouncedSet = useCallback((path: SettingsPath, value: any) => {
    const key = `single_${path}`;
    
    
    if (debounceMap.has(key)) {
      clearTimeout(debounceMap.get(key)!);
    }
    
    
    const timeout = setTimeout(() => {
      setByPath(path, value);
      debounceMap.delete(key);
    }, DEBOUNCE_DELAY);
    
    debounceMap.set(key, timeout);
  }, [setByPath]);
  
  
  const batchedSet = useCallback((updates: Record<SettingsPath, any>) => {
    const updateArray = Object.entries(updates).map(([path, value]) => ({
      path,
      value
    }));
    
    const key = `batch_${Object.keys(updates).sort().join('|')}`;
    
    
    if (debounceMap.has(key)) {
      clearTimeout(debounceMap.get(key)!);
    }
    
    
    const timeout = setTimeout(() => {
      batchUpdate(updateArray);
      debounceMap.delete(key);
    }, DEBOUNCE_DELAY);
    
    debounceMap.set(key, timeout);
  }, [batchUpdate]);
  
  
  const immediateSet = useCallback((updater: (draft: any) => void) => {
    updateSettings(updater);
  }, [updateSettings]);
  
  return useMemo(() => ({
    set: debouncedSet,           
    batchedSet,                  
    immediateSet,                
    updateSettings               
  }), [debouncedSet, batchedSet, immediateSet, updateSettings]);
}


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
  // Stabilize spread dependencies into a single JSON key
  const depsKey = JSON.stringify(dependencies);


  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    let mounted = true;

    const handleChange = async () => {
      try {
        let value: any;

        if (enableCache) {

          value = await getDedicatedSetting(stablePath);
        } else {

          value = useSettingsStore.getState().get(stablePath);
        }

        if (mounted) {
          callbackRef.current(value);
        }
      } catch (error) {
        logger.error(`Subscription callback failed for path ${stablePath}:`, error);
      }
    };


    if (immediate) {
      handleChange();
    }


    const unsubscribe = useSettingsStore.getState().subscribe(
      stablePath,
      handleChange,
      false
    );

    return () => {
      mounted = false;
      unsubscribe();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stablePath, enableCache, immediate, depsKey]);
}


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


  const memoizedSelector = useCallback(selector, [selector.toString()]);


  const value = useSettingsStore(
    state => memoizedSelector(state.settings)
  ) as T;


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