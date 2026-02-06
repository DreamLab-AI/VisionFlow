// High-Performance Client-Side Settings Cache with Smart Invalidation
// Implements localStorage caching, versioning, and WebSocket delta synchronization

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

export class SettingsCacheClient {
  private cache = new Map<string, CachedSetting>();
  private metrics: CacheMetrics;
  private websocket: WebSocket | null = null;
  private compressionWorker: Worker | null = null;
  private readonly CACHE_PREFIX = 'hive_settings_';
  private readonly CACHE_VERSION = '1.0';
  private readonly DEFAULT_TTL = 300000; // 5 minutes
  private readonly MAX_CACHE_SIZE = 1000;
  private readonly STORAGE_QUOTA = 5 * 1024 * 1024; // 5MB
  
  constructor(private wsUrl: string = 'ws://localhost:3000/ws') {
    this.metrics = {
      hits: 0,
      misses: 0,
      invalidations: 0,
      bandwidthSaved: 0,
      storageUsed: 0,
      lastSync: Date.now()
    };
    
    this.initializeCache();
    this.setupWebSocket();
    this.initializeCompressionWorker();
    this.startCacheMaintenanceTimer();
  }
  
  private initializeCache(): void {
    try {
      // Load cache from localStorage
      const stored = localStorage.getItem(`${this.CACHE_PREFIX}cache`);
      if (stored) {
        const parsedCache = JSON.parse(stored);
        
        // Validate cache version
        if (parsedCache.version === this.CACHE_VERSION) {
          Object.entries(parsedCache.data).forEach(([key, value]) => {
            const cached = value as CachedSetting;
            
            // Check if cache entry is still valid
            if (Date.now() - cached.timestamp < cached.ttl) {
              this.cache.set(key, cached);
            }
          });
          
          console.log(`Loaded ${this.cache.size} cached settings from localStorage`);
        } else {
          console.log('Cache version mismatch, clearing localStorage cache');
          this.clearLocalStorage();
        }
      }
      
      // Load metrics
      const storedMetrics = localStorage.getItem(`${this.CACHE_PREFIX}metrics`);
      if (storedMetrics) {
        const parsedMetrics = JSON.parse(storedMetrics);
        if (parsedMetrics && typeof parsedMetrics === 'object' && !Array.isArray(parsedMetrics)) {
          this.metrics = { ...this.metrics, ...parsedMetrics };
        }
      }
      
    } catch (error) {
      console.warn('Failed to initialize cache from localStorage:', error);
      this.clearLocalStorage();
    }
  }
  
  private setupWebSocket(): void {
    try {
      this.websocket = new WebSocket(this.wsUrl);
      
      this.websocket.onopen = () => {
        console.log('WebSocket connected for settings synchronization');
        this.requestDeltaSync();
      };
      
      this.websocket.onmessage = (event) => {
        this.handleWebSocketMessage(event);
      };
      
      this.websocket.onclose = () => {
        console.log('WebSocket disconnected, attempting reconnection...');
        setTimeout(() => this.setupWebSocket(), 5000);
      };
      
      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
    } catch (error) {
      console.warn('Failed to setup WebSocket:', error);
    }
  }
  
  private initializeCompressionWorker(): void {
    if (typeof Worker !== 'undefined') {
      try {
        // Create compression worker for large payloads
        const workerBlob = new Blob([`
          // Simple LZ-string compression worker
          self.onmessage = function(e) {
            const { action, data, id } = e.data;
            
            try {
              if (action === 'compress') {
                // Simple compression simulation (in real implementation, use LZ-string or similar)
                const compressed = btoa(JSON.stringify(data));
                self.postMessage({ id, result: compressed, originalSize: JSON.stringify(data).length, compressedSize: compressed.length });
              } else if (action === 'decompress') {
                const decompressed = JSON.parse(atob(data));
                self.postMessage({ id, result: decompressed });
              }
            } catch (error) {
              self.postMessage({ id, error: error.message });
            }
          };
        `], { type: 'application/javascript' });
        
        this.compressionWorker = new Worker(URL.createObjectURL(workerBlob));
        
        this.compressionWorker.onmessage = (e) => {
          this.handleCompressionWorkerMessage(e);
        };
        
      } catch (error) {
        console.warn('Failed to initialize compression worker:', error);
      }
    }
  }
  
  private handleWebSocketMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'settingsChanged':
          this.handleSettingChanged(message);
          break;
          
        case 'settingsBatchChanged':
          this.handleBatchSettingsChanged(message);
          break;
          
        case 'deltaSync':
          this.handleDeltaSync(message);
          break;
          
        case 'cacheInvalidate':
          this.handleCacheInvalidation(message);
          break;
          
        default:
          console.log('Unknown WebSocket message type:', message.type);
      }
      
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }
  
  private handleSettingChanged(message: any): void {
    const { path, value, timestamp } = message;
    
    // Update local cache
    this.setCachedValue(path, value, {
      timestamp,
      fromWebSocket: true
    });
    
    // Notify subscribers
    this.notifySubscribers(path, value);
  }
  
  private handleBatchSettingsChanged(message: any): void {
    const { updates, timestamp } = message;
    
    updates.forEach((update: any) => {
      this.setCachedValue(update.path, update.value, {
        timestamp,
        fromWebSocket: true
      });
    });
    
    // Batch notify subscribers
    this.notifyBatchSubscribers(updates);
  }
  
  private handleDeltaSync(message: any): void {
    const { deltas } = message;
    
    deltas.forEach((delta: DeltaUpdate) => {
      switch (delta.operation) {
        case 'set':
          this.setCachedValue(delta.path, delta.value, {
            timestamp: delta.timestamp,
            fromWebSocket: true
          });
          break;
          
        case 'delete':
          this.cache.delete(delta.path);
          this.invalidateLocalStorage(delta.path);
          break;
          
        case 'batch':
          // Handle batch operation
          this.handleBatchDelta(delta);
          break;
      }
    });
    
    this.metrics.lastSync = Date.now();
    this.persistMetrics();
  }
  
  private handleCacheInvalidation(message: any): void {
    const { paths, reason } = message;
    
    paths.forEach((path: string) => {
      this.cache.delete(path);
      this.invalidateLocalStorage(path);
      this.metrics.invalidations++;
    });
    
    console.log(`Cache invalidated for ${paths.length} paths. Reason: ${reason}`);
  }
  
  /**
   * Get a setting value with intelligent caching
   */
  public async get(path: string, options: { useCache?: boolean, ttl?: number } = {}): Promise<any> {
    const startTime = performance.now();
    const { useCache = true, ttl = this.DEFAULT_TTL } = options;
    
    // Check local cache first
    if (useCache) {
      const cached = this.getCachedValue(path);
      if (cached && this.isCacheValid(cached)) {
        this.metrics.hits++;
        this.metrics.bandwidthSaved += this.estimatePayloadSize(cached.value);
        
        const responseTime = performance.now() - startTime;
        console.log(`Cache hit for ${path} in ${responseTime.toFixed(2)}ms`);
        
        return cached.value;
      }
    }
    
    // Cache miss - fetch from server
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
        // Cache the result
        this.setCachedValue(path, result.value, { ttl });
        
        const responseTime = performance.now() - startTime;
        console.log(`Fetched ${path} from server in ${responseTime.toFixed(2)}ms`);
        
        return result.value;
      } else {
        throw new Error(result.error || 'Failed to get setting value');
      }
      
    } catch (error) {
      console.error(`Failed to fetch setting ${path}:`, error);
      throw error;
    }
  }
  
  /**
   * Get multiple settings in a single optimized request
   */
  public async getBatch(paths: string[], options: { useCache?: boolean } = {}): Promise<Record<string, any>> {
    const startTime = performance.now();
    const { useCache = true } = options;
    
    const results: Record<string, any> = {};
    const uncachedPaths: string[] = [];
    
    // Check cache for each path
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
    
    // Fetch uncached paths from server
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
        
        if (batchResult.success) {
          Object.entries(batchResult.results).forEach(([path, value]) => {
            results[path] = value;
            this.setCachedValue(path, value);
          });
        }
        
      } catch (error) {
        console.error('Failed to fetch batch settings:', error);
        throw error;
      }
    }
    
    const responseTime = performance.now() - startTime;
    const cacheHitRate = (this.metrics.hits / (this.metrics.hits + this.metrics.misses)) * 100;
    
    console.log(`Batch fetch completed in ${responseTime.toFixed(2)}ms. Cache hit rate: ${cacheHitRate.toFixed(1)}%`);
    
    return results;
  }
  
  /**
   * Set a setting value with cache update
   */
  public async set(path: string, value: any, options: { broadcast?: boolean } = {}): Promise<void> {
    const startTime = performance.now();
    const { broadcast = true } = options;
    
    try {
      const response = await fetch('/api/settings/path', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ path, value })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Update local cache immediately
        this.setCachedValue(path, value);
        
        // Broadcast via WebSocket if enabled
        if (broadcast && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
          this.websocket.send(JSON.stringify({
            type: 'settingChanged',
            path,
            value,
            timestamp: Date.now()
          }));
        }
        
        const responseTime = performance.now() - startTime;
        console.log(`Updated ${path} in ${responseTime.toFixed(2)}ms`);
        
      } else {
        throw new Error(result.error || 'Failed to update setting');
      }
      
    } catch (error) {
      console.error(`Failed to set setting ${path}:`, error);
      throw error;
    }
  }
  
  /**
   * Set multiple settings in an optimized batch operation
   */
  public async setBatch(updates: Array<{path: string, value: any}>, options: { broadcast?: boolean } = {}): Promise<void> {
    const startTime = performance.now();
    const { broadcast = true } = options;
    
    try {
      const response = await fetch('/api/settings/batch', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ updates })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Update local cache for all successful updates
        updates.forEach(({path, value}) => {
          this.setCachedValue(path, value);
        });
        
        // Broadcast batch update via WebSocket
        if (broadcast && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
          this.websocket.send(JSON.stringify({
            type: 'settingsBatchChanged',
            updates,
            timestamp: Date.now()
          }));
        }
        
        const responseTime = performance.now() - startTime;
        console.log(`Batch updated ${updates.length} settings in ${responseTime.toFixed(2)}ms`);
        
      } else {
        throw new Error(result.error || 'Failed to update settings batch');
      }
      
    } catch (error) {
      console.error('Failed to set batch settings:', error);
      throw error;
    }
  }
  
  /**
   * Get comprehensive performance metrics
   */
  public getPerformanceMetrics(): PerformanceMetrics {
    const totalRequests = this.metrics.hits + this.metrics.misses;
    
    return {
      cacheHitRate: totalRequests > 0 ? (this.metrics.hits / totalRequests) * 100 : 0,
      averageResponseTime: 0, // Would need to track this
      bandwidthSavings: this.metrics.bandwidthSaved,
      totalRequests
    };
  }
  
  /**
   * Clear all caches and reset metrics
   */
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
    
    console.log('All caches cleared and metrics reset');
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
    
    // Add to memory cache
    this.cache.set(path, cached);
    
    // Persist to localStorage (async)
    this.persistToLocalStorage();
    
    // Update storage metrics
    this.updateStorageMetrics();
    
    // Cleanup if cache is too large
    this.enforCacheSizeLimit();
  }
  
  private isCacheValid(cached: CachedSetting): boolean {
    return (Date.now() - cached.timestamp) < cached.ttl;
  }
  
  private calculateHash(value: any): string {
    // Simple hash function for cache validation
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
      
      // Check storage quota
      if (serialized.length > this.STORAGE_QUOTA) {
        console.warn('Cache size exceeds storage quota, performing cleanup');
        this.cleanupOldestEntries();
        return;
      }
      
      localStorage.setItem(`${this.CACHE_PREFIX}cache`, serialized);
      this.persistMetrics();
      
    } catch (error) {
      console.warn('Failed to persist cache to localStorage:', error);
      
      // If storage is full, try cleanup
      if (error.name === 'QuotaExceededError') {
        this.cleanupOldestEntries();
      }
    }
  }
  
  private persistMetrics(): void {
    try {
      localStorage.setItem(`${this.CACHE_PREFIX}metrics`, JSON.stringify(this.metrics));
    } catch (error) {
      console.warn('Failed to persist metrics:', error);
    }
  }
  
  private invalidateLocalStorage(path: string): void {
    // Remove specific path from localStorage cache
    try {
      const stored = localStorage.getItem(`${this.CACHE_PREFIX}cache`);
      if (stored) {
        const parsed = JSON.parse(stored);
        delete parsed.data[path];
        localStorage.setItem(`${this.CACHE_PREFIX}cache`, JSON.stringify(parsed));
      }
    } catch (error) {
      console.warn('Failed to invalidate localStorage cache:', error);
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
      console.warn('Failed to clear localStorage:', error);
    }
  }
  
  private updateStorageMetrics(): void {
    this.metrics.storageUsed = this.cache.size;
  }
  
  private enforCacheSizeLimit(): void {
    if (this.cache.size > this.MAX_CACHE_SIZE) {
      // Remove oldest entries
      const entries = Array.from(this.cache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      const toRemove = entries.slice(0, Math.floor(this.MAX_CACHE_SIZE * 0.2));
      toRemove.forEach(([key]) => {
        this.cache.delete(key);
      });
      
      console.log(`Removed ${toRemove.length} old cache entries`);
    }
  }
  
  private cleanupOldestEntries(): void {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
    
    // Remove oldest 30%
    const toRemove = entries.slice(0, Math.floor(entries.length * 0.3));
    toRemove.forEach(([key]) => {
      this.cache.delete(key);
    });
    
    // Try to persist again
    this.persistToLocalStorage();
  }
  
  private estimatePayloadSize(value: any): number {
    try {
      return JSON.stringify(value).length;
    } catch {
      return 0;
    }
  }
  
  private requestDeltaSync(): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'requestDeltaSync',
        lastSync: this.metrics.lastSync
      }));
    }
  }
  
  private handleBatchDelta(delta: DeltaUpdate): void {
    // Handle batch delta operations
    if (Array.isArray(delta.value)) {
      delta.value.forEach((update: any) => {
        this.setCachedValue(update.path, update.value, {
          timestamp: delta.timestamp,
          fromWebSocket: true
        });
      });
    }
  }
  
  private handleCompressionWorkerMessage(e: MessageEvent): void {
    // Handle compression worker responses
    const { id, result, error, originalSize, compressedSize } = e.data;
    
    if (error) {
      console.warn('Compression worker error:', error);
      return;
    }
    
    if (originalSize && compressedSize) {
      const ratio = (originalSize - compressedSize) / originalSize;
      console.log(`Compression achieved ${(ratio * 100).toFixed(1)}% size reduction`);
    }
  }
  
  private startCacheMaintenanceTimer(): void {
    // Cleanup expired entries every 5 minutes
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
        console.log(`Cleaned up ${expiredKeys.length} expired cache entries`);
        this.persistToLocalStorage();
      }
      
    }, 5 * 60 * 1000);
  }
  
  private notifySubscribers(path: string, value: any): void {
    // Emit custom event for subscribers
    window.dispatchEvent(new CustomEvent('settingChanged', {
      detail: { path, value }
    }));
  }
  
  private notifyBatchSubscribers(updates: any[]): void {
    // Emit custom event for batch updates
    window.dispatchEvent(new CustomEvent('settingsBatchChanged', {
      detail: { updates }
    }));
  }
}