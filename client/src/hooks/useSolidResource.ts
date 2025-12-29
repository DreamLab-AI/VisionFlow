/**
 * useSolidResource Hook
 *
 * Hook for LDP resource operations:
 * - Fetches JSON-LD resources
 * - Provides put/post/delete mutations
 * - Auto-refreshes on WebSocket notifications
 * - Implements caching with React Query-like patterns
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import solidPodService, {
  JsonLdDocument,
  SolidNotification,
} from '../services/SolidPodService';
import { useSolidPod } from './useSolidPod';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('useSolidResource');

// --- Types ---

export interface ResourceState<T extends JsonLdDocument = JsonLdDocument> {
  /** Fetched resource data */
  data: T | null;
  /** Whether initial fetch or mutation is in progress */
  isLoading: boolean;
  /** Whether a refresh is in progress */
  isRefreshing: boolean;
  /** Error message if operation failed */
  error: string | null;
  /** Last successful fetch timestamp */
  lastFetched: Date | null;
  /** Whether resource exists */
  exists: boolean;
}

export interface UseSolidResourceOptions {
  /** Automatically fetch on mount */
  autoFetch?: boolean;
  /** Subscribe to WebSocket notifications for auto-refresh */
  autoRefresh?: boolean;
  /** Refetch interval in milliseconds (0 to disable) */
  refetchInterval?: number;
  /** Content type for mutations */
  contentType?: 'application/ld+json' | 'text/turtle';
}

export interface UseSolidResourceReturn<T extends JsonLdDocument = JsonLdDocument> extends ResourceState<T> {
  /** Update or create the resource (PUT) */
  put: (data: T) => Promise<boolean>;
  /** Create resource in container (POST) */
  post: (data: T, slug?: string) => Promise<string | null>;
  /** Delete the resource */
  remove: () => Promise<boolean>;
  /** Manually refresh the resource */
  refresh: () => Promise<void>;
  /** Check if resource exists */
  checkExists: () => Promise<boolean>;
}

// --- Simple Cache ---

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  path: string;
}

const resourceCache = new Map<string, CacheEntry<JsonLdDocument>>();
const CACHE_TTL = 30000; // 30 seconds

function getCached<T extends JsonLdDocument>(path: string): T | null {
  const entry = resourceCache.get(path);
  if (!entry) return null;

  const isStale = Date.now() - entry.timestamp > CACHE_TTL;
  if (isStale) {
    resourceCache.delete(path);
    return null;
  }

  return entry.data as T;
}

function setCache<T extends JsonLdDocument>(path: string, data: T): void {
  resourceCache.set(path, {
    data,
    timestamp: Date.now(),
    path,
  });
}

function invalidateCache(path: string): void {
  resourceCache.delete(path);
  // Also invalidate parent container
  const parentPath = path.substring(0, path.lastIndexOf('/') + 1);
  if (parentPath !== path) {
    resourceCache.delete(parentPath);
  }
}

// --- Hook Implementation ---

export function useSolidResource<T extends JsonLdDocument = JsonLdDocument>(
  path: string,
  options: UseSolidResourceOptions = {}
): UseSolidResourceReturn<T> {
  const {
    autoFetch = true,
    autoRefresh = true,
    refetchInterval = 0,
    contentType = 'application/ld+json',
  } = options;

  const { hasPod, subscribe, isConnected } = useSolidPod();

  const [state, setState] = useState<ResourceState<T>>({
    data: null,
    isLoading: false,
    isRefreshing: false,
    error: null,
    lastFetched: null,
    exists: false,
  });

  const mountedRef = useRef(true);
  const fetchInProgressRef = useRef(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // --- Fetch Resource ---

  const fetchResource = useCallback(async (isRefresh = false): Promise<T | null> => {
    if (!path || fetchInProgressRef.current) return null;

    fetchInProgressRef.current = true;

    // Check cache first (not for refresh)
    if (!isRefresh) {
      const cached = getCached<T>(path);
      if (cached) {
        setState((prev) => ({
          ...prev,
          data: cached,
          exists: true,
          lastFetched: new Date(),
        }));
        fetchInProgressRef.current = false;
        return cached;
      }
    }

    setState((prev) => ({
      ...prev,
      isLoading: !isRefresh && !prev.data,
      isRefreshing: isRefresh,
      error: null,
    }));

    try {
      const data = await solidPodService.fetchJsonLd(path) as T;

      if (mountedRef.current) {
        setCache(path, data);
        setState((prev) => ({
          ...prev,
          data,
          isLoading: false,
          isRefreshing: false,
          exists: true,
          lastFetched: new Date(),
        }));
      }

      logger.debug('Resource fetched', { path });
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch resource';
      const is404 = errorMessage.includes('404');

      if (mountedRef.current) {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          isRefreshing: false,
          error: is404 ? null : errorMessage,
          exists: false,
          data: is404 ? null : prev.data,
        }));
      }

      if (!is404) {
        logger.error('Resource fetch failed', { path, error: err });
      }

      return null;
    } finally {
      fetchInProgressRef.current = false;
    }
  }, [path]);

  // --- Refresh ---

  const refresh = useCallback(async (): Promise<void> => {
    await fetchResource(true);
  }, [fetchResource]);

  // --- Check Exists ---

  const checkExists = useCallback(async (): Promise<boolean> => {
    try {
      const exists = await solidPodService.resourceExists(path);
      setState((prev) => ({ ...prev, exists }));
      return exists;
    } catch {
      return false;
    }
  }, [path]);

  // --- PUT ---

  const put = useCallback(async (data: T): Promise<boolean> => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const success = await solidPodService.putResource(path, data, contentType);

      if (success) {
        invalidateCache(path);

        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            data,
            isLoading: false,
            exists: true,
            lastFetched: new Date(),
          }));
        }

        logger.debug('Resource updated', { path });
      } else {
        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: 'Failed to update resource',
          }));
        }
      }

      return success;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Update failed';
      logger.error('PUT failed', { path, error: err });

      if (mountedRef.current) {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
      }

      return false;
    }
  }, [path, contentType]);

  // --- POST ---

  const post = useCallback(async (data: T, slug?: string): Promise<string | null> => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const location = await solidPodService.postResource(path, data, slug);

      if (location) {
        invalidateCache(path);

        if (mountedRef.current) {
          setState((prev) => ({ ...prev, isLoading: false }));
        }

        logger.debug('Resource created', { container: path, location });
      } else {
        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: 'Failed to create resource',
          }));
        }
      }

      return location;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Create failed';
      logger.error('POST failed', { path, error: err });

      if (mountedRef.current) {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
      }

      return null;
    }
  }, [path]);

  // --- DELETE ---

  const remove = useCallback(async (): Promise<boolean> => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const success = await solidPodService.deleteResource(path);

      if (success) {
        invalidateCache(path);

        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            data: null,
            isLoading: false,
            exists: false,
          }));
        }

        logger.debug('Resource deleted', { path });
      } else {
        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: 'Failed to delete resource',
          }));
        }
      }

      return success;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Delete failed';
      logger.error('DELETE failed', { path, error: err });

      if (mountedRef.current) {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
      }

      return false;
    }
  }, [path]);

  // --- WebSocket Notification Handler ---

  const handleNotification = useCallback((notification: SolidNotification) => {
    if (notification.type === 'pub') {
      logger.debug('Resource change notification', { url: notification.url, path });
      invalidateCache(path);
      refresh();
    }
  }, [path, refresh]);

  // --- Effects ---

  // Track mount state
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Auto-fetch on mount/path change
  useEffect(() => {
    if (autoFetch && hasPod && path) {
      fetchResource();
    }
  }, [autoFetch, hasPod, path, fetchResource]);

  // WebSocket subscription for auto-refresh
  useEffect(() => {
    if (!autoRefresh || !isConnected || !path) return;

    const unsubscribe = subscribe(path, handleNotification);

    return () => {
      unsubscribe();
    };
  }, [autoRefresh, isConnected, path, subscribe, handleNotification]);

  // Refetch interval
  useEffect(() => {
    if (refetchInterval <= 0 || !hasPod || !path) return;

    intervalRef.current = setInterval(() => {
      refresh();
    }, refetchInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [refetchInterval, hasPod, path, refresh]);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return {
    ...state,
    put,
    post,
    remove,
    refresh,
    checkExists,
  };
}

export default useSolidResource;
