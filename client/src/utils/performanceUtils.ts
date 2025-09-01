import React from 'react';
import { createLogger } from './logger';

const logger = createLogger('PerformanceUtils');

/**
 * Enhanced React.memo with deep comparison for complex props
 */
export function memoizeComponent<T extends React.ComponentType<any>>(
  Component: T,
  propsAreEqual?: (prevProps: React.ComponentProps<T>, nextProps: React.ComponentProps<T>) => boolean
): T {
  const MemoizedComponent = React.memo(Component, propsAreEqual) as T;
  MemoizedComponent.displayName = `Memoized(${Component.displayName || Component.name})`;
  return MemoizedComponent;
}

/**
 * Performance monitoring HOC
 */
export function withPerformanceMonitoring<P extends object>(
  Component: React.ComponentType<P>,
  componentName?: string
) {
  const name = componentName || Component.displayName || Component.name || 'Unknown';
  
  return React.forwardRef<any, P>((props, ref) => {
    const renderStartTime = React.useRef(performance.now());
    const renderCount = React.useRef(0);
    
    React.useEffect(() => {
      const renderTime = performance.now() - renderStartTime.current;
      renderCount.current += 1;
      
      if (renderTime > 16) { // Longer than one frame at 60fps
        logger.warn(`Slow render detected in ${name}`, {
          renderTime: `${renderTime.toFixed(2)}ms`,
          renderCount: renderCount.current,
          props: Object.keys(props)
        });
      }
    });
    
    renderStartTime.current = performance.now();
    
    return React.createElement(Component, { ...props, ref });
  });
}

/**
 * Debounce hook for expensive operations
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = React.useState<T>(value);

  React.useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Throttle hook for high-frequency updates
 */
export function useThrottle<T>(value: T, limit: number): T {
  const [throttledValue, setThrottledValue] = React.useState<T>(value);
  const lastRan = React.useRef(Date.now());

  React.useEffect(() => {
    const handler = setTimeout(() => {
      if (Date.now() - lastRan.current >= limit) {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }
    }, limit - (Date.now() - lastRan.current));

    return () => {
      clearTimeout(handler);
    };
  }, [value, limit]);

  return throttledValue;
}

/**
 * Request deduplication utility
 */
class RequestDeduplicator {
  private pendingRequests = new Map<string, Promise<any>>();
  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>();

  async dedupe<T>(
    key: string,
    requestFn: () => Promise<T>,
    cacheTTL: number = 5000
  ): Promise<T> {
    // Check cache first
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < cached.ttl) {
      return cached.data;
    }

    // Check if request is already pending
    if (this.pendingRequests.has(key)) {
      return this.pendingRequests.get(key)!;
    }

    // Make new request
    const promise = requestFn()
      .then((data) => {
        this.cache.set(key, { data, timestamp: Date.now(), ttl: cacheTTL });
        return data;
      })
      .finally(() => {
        this.pendingRequests.delete(key);
      });

    this.pendingRequests.set(key, promise);
    return promise;
  }

  clearCache(keyPrefix?: string) {
    if (keyPrefix) {
      for (const key of this.cache.keys()) {
        if (key.startsWith(keyPrefix)) {
          this.cache.delete(key);
        }
      }
    } else {
      this.cache.clear();
    }
  }
}

export const requestDeduplicator = new RequestDeduplicator();

/**
 * Performance-optimized settings comparison
 */
export function settingsEqual(prev: any, next: any): boolean {
  if (prev === next) return true;
  if (!prev || !next) return false;
  
  const prevKeys = Object.keys(prev);
  const nextKeys = Object.keys(next);
  
  if (prevKeys.length !== nextKeys.length) return false;
  
  for (const key of prevKeys) {
    if (prev[key] !== next[key]) return false;
  }
  
  return true;
}

/**
 * Stable object reference hook
 */
export function useStableObject<T extends Record<string, any>>(obj: T): T {
  const ref = React.useRef<T>(obj);
  
  if (!settingsEqual(ref.current, obj)) {
    ref.current = obj;
  }
  
  return ref.current;
}