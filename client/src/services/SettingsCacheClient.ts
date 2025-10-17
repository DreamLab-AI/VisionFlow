import { createLogger } from '../utils/loggerConfig';
import { webSocketService } from './WebSocketService';

interface CachedSetting {
  value: any;
  path: string;
  timestamp: number;
  version: number;
  hash: string;
  ttl: number;
}

interface CacheMetrics {
  hits: number;
  misses: number;
  invalidations: number;
  bandwidthSaved: number;
  storageUsed: number;
  lastSync: number;
}

interface DeltaUpdate {
  path: string;
  value: any;
  oldValue?: any;
  timestamp: number;
  operation: 'set' | 'delete' | 'batch';
}

interface PerformanceMetrics {
  cacheHitRate: number;
  averageResponseTime: number;
  bandwidthSavings: number;
  totalRequests: number;
}

type SettingsChangeCallback = (path: string, value: any) => void;

const logger = createLogger('SettingsCacheClient');

export class SettingsCacheClient {
  private cache = new Map<string, CachedSetting>();
  private metrics: CacheMetrics;
  private subscribers = new Map<string, Set<SettingsChangeCallback>>();
  private readonly CACHE_PREFIX = 'hive_settings_';
  private readonly CACHE_VERSION = '2.0';
  private readonly DEFAULT_TTL = 300000; // 5 minutes
  private readonly MAX_CACHE_SIZE = 1000;
  private readonly STORAGE_QUOTA = 5 * 1024 * 1024; // 5MB
  private wsUnsubscribe: (() => void) | null = null;

  constructor() {
    this.metrics = {
      hits: 0,
      misses: 0,
      invalidations: 0,
      bandwidthSaved: 0,
      storageUsed: 0,
      lastSync: Date.now()
    };

    this.initializeCache();
    this.setupWebSocketListener();
    this.startCacheMaintenanceTimer();
  }

  private initializeCache(): void {
    try {
      const stored = localStorage.getItem(`${this.CACHE_PREFIX}cache`);
      if (stored) {
        const parsedCache = JSON.parse(stored);

        if (parsedCache.version === this.CACHE_VERSION) {
          Object.entries(parsedCache.data).forEach(([key, value]) => {
            const cached = value as CachedSetting;

            if (Date.now() - cached.timestamp < cached.ttl) {
              this.cache.set(key, cached);
            }
          });

          logger.info(`Loaded ${this.cache.size} cached settings from localStorage`);
        } else {
          logger.info('Cache version mismatch, clearing localStorage cache');
          this.clearLocalStorage();
        }
      }

      const storedMetrics = localStorage.getItem(`${this.CACHE_PREFIX}metrics`);
      if (storedMetrics) {
        this.metrics = { ...this.metrics, ...JSON.parse(storedMetrics) };
      }

    } catch (error) {
      logger.warn('Failed to initialize cache from localStorage:', error);
      this.clearLocalStorage();
    }
  }

  private setupWebSocketListener(): void {
    this.wsUnsubscribe = webSocketService.onMessage((message) => {
      switch (message.type) {
        case 'settings_changed':
          this.handleSettingChanged(message.data);
          break;

        case 'settings_batch_changed':
          this.handleBatchSettingsChanged(message.data);
          break;

        case 'cache_invalidate':
          this.handleCacheInvalidation(message.data);
          break;
      }
    });
  }

  private handleSettingChanged(data: any): void {
    const { path, value, timestamp } = data;

    this.setCachedValue(path, value, {
      timestamp,
      fromWebSocket: true
    });

    this.notifySubscribers(path, value);
  }

  private handleBatchSettingsChanged(data: any): void {
    const { updates, timestamp } = data;

    updates.forEach((update: any) => {
      this.setCachedValue(update.path, update.value, {
        timestamp,
        fromWebSocket: true
      });
    });

    this.notifyBatchSubscribers(updates);
  }

  private handleCacheInvalidation(data: any): void {
    const { paths, reason } = data;

    paths.forEach((path: string) => {
      this.cache.delete(path);
      this.invalidateLocalStorage(path);
      this.metrics.invalidations++;
    });

    logger.info(`Cache invalidated for ${paths.length} paths. Reason: ${reason}`);
  }

  public async get(path: string, options: { useCache?: boolean, ttl?: number } = {}): Promise<any> {
    const startTime = performance.now();
    const { useCache = true, ttl = this.DEFAULT_TTL } = options;

    if (useCache) {
      const cached = this.getCachedValue(path);
      if (cached && this.isCacheValid(cached)) {
        this.metrics.hits++;
        this.metrics.bandwidthSaved += this.estimatePayloadSize(cached.value);

        const responseTime = performance.now() - startTime;
        logger.debug(`Cache hit for ${path} in ${responseTime.toFixed(2)}ms`);

        return cached.value;
      }
    }

    this.metrics.misses++;

    try {
      const response = await fetch(`/api/settings/path?path=${encodeURIComponent(path)}`, {
        headers: {
          'Cache-Control': 'no-cache',
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success && result.value !== undefined) {
        this.setCachedValue(path, result.value, { ttl });

        const responseTime = performance.now() - startTime;
        logger.debug(`Fetched ${path} from server in ${responseTime.toFixed(2)}ms`);

        return result.value;
      } else {
        throw new Error(result.error || 'Failed to get setting value');
      }

    } catch (error) {
      logger.error(`Failed to fetch setting ${path}:`, error);
      throw error;
    }
  }

  public async getBatch(paths: string[], options: { useCache?: boolean } = {}): Promise<Record<string, any>> {
    const startTime = performance.now();
    const { useCache = true } = options;

    const results: Record<string, any> = {};
    const uncachedPaths: string[] = [];

    if (useCache) {
      paths.forEach(path => {
        const cached = this.getCachedValue(path);
        if (cached && this.isCacheValid(cached)) {
          results[path] = cached.value;
          this.metrics.hits++;
        } else {
          uncachedPaths.push(path);
          this.metrics.misses++;
        }
      });
    } else {
      uncachedPaths.push(...paths);
      this.metrics.misses += paths.length;
    }

    if (uncachedPaths.length > 0) {
      try {
        const response = await fetch('/api/settings/batch', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache'
          },
          body: JSON.stringify({
            paths: uncachedPaths
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const batchResult = await response.json();

        Object.entries(batchResult).forEach(([path, value]) => {
          results[path] = value;
          this.setCachedValue(path, value);
        });

      } catch (error) {
        logger.error('Failed to fetch batch settings:', error);
        throw error;
      }
    }

    const responseTime = performance.now() - startTime;
    const cacheHitRate = (this.metrics.hits / (this.metrics.hits + this.metrics.misses)) * 100;

    logger.debug(`Batch fetch completed in ${responseTime.toFixed(2)}ms. Cache hit rate: ${cacheHitRate.toFixed(1)}%`);

    return results;
  }

  public subscribe(path: string, callback: SettingsChangeCallback): () => void {
    if (!this.subscribers.has(path)) {
      this.subscribers.set(path, new Set());
    }

    this.subscribers.get(path)!.add(callback);

    return () => {
      const callbacks = this.subscribers.get(path);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscribers.delete(path);
        }
      }
    };
  }

  public getPerformanceMetrics(): PerformanceMetrics {
    const totalRequests = this.metrics.hits + this.metrics.misses;

    return {
      cacheHitRate: totalRequests > 0 ? (this.metrics.hits / totalRequests) * 100 : 0,
      averageResponseTime: 0,
      bandwidthSavings: this.metrics.bandwidthSaved,
      totalRequests
    };
  }

  public clearCache(): void {
    this.cache.clear();
    this.clearLocalStorage();
    this.metrics = {
      hits: 0,
      misses: 0,
      invalidations: 0,
      bandwidthSaved: 0,
      storageUsed: 0,
      lastSync: Date.now()
    };

    logger.info('All caches cleared and metrics reset');
  }

  public destroy(): void {
    if (this.wsUnsubscribe) {
      this.wsUnsubscribe();
      this.wsUnsubscribe = null;
    }
    this.subscribers.clear();
    this.cache.clear();
  }

  private getCachedValue(path: string): CachedSetting | null {
    return this.cache.get(path) || null;
  }

  private setCachedValue(path: string, value: any, options: {
    timestamp?: number,
    ttl?: number,
    fromWebSocket?: boolean
  } = {}): void {
    const {
      timestamp = Date.now(),
      ttl = this.DEFAULT_TTL,
      fromWebSocket = false
    } = options;

    const hash = this.calculateHash(value);
    const version = this.getNextVersion();

    const cached: CachedSetting = {
      value,
      path,
      timestamp,
      version,
      hash,
      ttl
    };

    this.cache.set(path, cached);
    this.persistToLocalStorage();
    this.updateStorageMetrics();
    this.enforCacheSizeLimit();
  }

  private isCacheValid(cached: CachedSetting): boolean {
    return (Date.now() - cached.timestamp) < cached.ttl;
  }

  private calculateHash(value: any): string {
    return btoa(JSON.stringify(value)).slice(0, 16);
  }

  private getNextVersion(): number {
    return Date.now();
  }

  private persistToLocalStorage(): void {
    try {
      const cacheData = {
        version: this.CACHE_VERSION,
        timestamp: Date.now(),
        data: Object.fromEntries(this.cache)
      };

      const serialized = JSON.stringify(cacheData);

      if (serialized.length > this.STORAGE_QUOTA) {
        logger.warn('Cache size exceeds storage quota, performing cleanup');
        this.cleanupOldestEntries();
        return;
      }

      localStorage.setItem(`${this.CACHE_PREFIX}cache`, serialized);
      this.persistMetrics();

    } catch (error) {
      logger.warn('Failed to persist cache to localStorage:', error);

      if (error instanceof Error && error.name === 'QuotaExceededError') {
        this.cleanupOldestEntries();
      }
    }
  }

  private persistMetrics(): void {
    try {
      localStorage.setItem(`${this.CACHE_PREFIX}metrics`, JSON.stringify(this.metrics));
    } catch (error) {
      logger.warn('Failed to persist metrics:', error);
    }
  }

  private invalidateLocalStorage(path: string): void {
    try {
      const stored = localStorage.getItem(`${this.CACHE_PREFIX}cache`);
      if (stored) {
        const parsed = JSON.parse(stored);
        delete parsed.data[path];
        localStorage.setItem(`${this.CACHE_PREFIX}cache`, JSON.stringify(parsed));
      }
    } catch (error) {
      logger.warn('Failed to invalidate localStorage cache:', error);
    }
  }

  private clearLocalStorage(): void {
    try {
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith(this.CACHE_PREFIX)) {
          localStorage.removeItem(key);
        }
      });
    } catch (error) {
      logger.warn('Failed to clear localStorage:', error);
    }
  }

  private updateStorageMetrics(): void {
    this.metrics.storageUsed = this.cache.size;
  }

  private enforCacheSizeLimit(): void {
    if (this.cache.size > this.MAX_CACHE_SIZE) {
      const entries = Array.from(this.cache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

      const toRemove = entries.slice(0, Math.floor(this.MAX_CACHE_SIZE * 0.2));
      toRemove.forEach(([key]) => {
        this.cache.delete(key);
      });

      logger.info(`Removed ${toRemove.length} old cache entries`);
    }
  }

  private cleanupOldestEntries(): void {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

    const toRemove = entries.slice(0, Math.floor(entries.length * 0.3));
    toRemove.forEach(([key]) => {
      this.cache.delete(key);
    });

    this.persistToLocalStorage();
  }

  private estimatePayloadSize(value: any): number {
    try {
      return JSON.stringify(value).length;
    } catch {
      return 0;
    }
  }

  private startCacheMaintenanceTimer(): void {
    setInterval(() => {
      const now = Date.now();
      const expiredKeys: string[] = [];

      this.cache.forEach((cached, key) => {
        if (now - cached.timestamp > cached.ttl) {
          expiredKeys.push(key);
        }
      });

      expiredKeys.forEach(key => {
        this.cache.delete(key);
      });

      if (expiredKeys.length > 0) {
        logger.debug(`Cleaned up ${expiredKeys.length} expired cache entries`);
        this.persistToLocalStorage();
      }

    }, 5 * 60 * 1000);
  }

  private notifySubscribers(path: string, value: any): void {
    const callbacks = this.subscribers.get(path);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(path, value);
        } catch (error) {
          logger.error(`Error in settings subscriber for ${path}:`, error);
        }
      });
    }

    window.dispatchEvent(new CustomEvent('settingChanged', {
      detail: { path, value }
    }));
  }

  private notifyBatchSubscribers(updates: any[]): void {
    updates.forEach(({ path, value }) => {
      this.notifySubscribers(path, value);
    });

    window.dispatchEvent(new CustomEvent('settingsBatchChanged', {
      detail: { updates }
    }));
  }
}

export const settingsCacheClient = new SettingsCacheClient();
