// Settings API Client - Path-based interface for granular settings operations with concurrent update handling
import { Settings } from '../features/settings/config/settings';
import { createLogger } from '../utils/loggerConfig';
import { unifiedApiClient, isApiError } from '../services/api/UnifiedApiClient';

const logger = createLogger('SettingsApi');
const API_BASE = '/api/settings';

// Debouncing and batching configuration
const DEBOUNCE_DELAY_MS = 50; // 50ms debounce for UI responsiveness
const BATCH_SIZE_LIMIT = 25;  // Maximum batch size to prevent server overload
const CRITICAL_UPDATE_IMMEDIATE = true; // Process physics updates immediately

// Batch operation interface for multiple path updates
interface BatchOperation {
  path: string;
  value: any;
}

// Priority levels for update batching (matches server-side enum)
enum UpdatePriority {
  Critical = 1,  // Physics parameters that affect GPU simulation
  High = 2,      // Visual settings that impact rendering
  Normal = 3,    // General configuration changes
  Low = 4,       // Non-critical settings like UI preferences
}

// Debounced update queue entry
interface DebouncedUpdate {
  path: string;
  value: any;
  priority: UpdatePriority;
  timestamp: number;
}

// Debouncing and batching manager
class SettingsUpdateManager {
  private updateQueue: Map<string, DebouncedUpdate> = new Map();
  private debounceTimer: NodeJS.Timeout | null = null;
  private processingBatch = false;

  private determinePriority(path: string): UpdatePriority {
    if (path.includes('.physics.')) {
      // Physics parameters are critical for GPU simulation
      return UpdatePriority.Critical;
    } else if (path.includes('.bloom.') || path.includes('.glow.') || path.includes('.visual')) {
      // Visual settings are high priority for user experience
      return UpdatePriority.High;
    } else if (path.includes('.system.') || path.includes('.security.')) {
      // System settings have normal priority
      return UpdatePriority.Normal;
    } else {
      // UI preferences and other settings are low priority
      return UpdatePriority.Low;
    }
  }

  async queueUpdate(path: string, value: any): Promise<void> {
    const priority = this.determinePriority(path);
    const update: DebouncedUpdate = {
      path,
      value,
      priority,
      timestamp: Date.now()
    };

    // Update the queue (overwrite previous value for same path)
    this.updateQueue.set(path, update);

    // For critical updates (physics), process immediately
    if (priority === UpdatePriority.Critical && CRITICAL_UPDATE_IMMEDIATE) {
      logger.info(`[DEBOUNCE] Critical update detected, processing immediately: ${path}`);
      return this.processCriticalUpdate(update);
    }

    // For non-critical updates, use debouncing
    this.scheduleDebounce();
  }

  private scheduleDebounce(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.processBatch();
    }, DEBOUNCE_DELAY_MS);
  }

  private async processCriticalUpdate(update: DebouncedUpdate): Promise<void> {
    try {
      const encodedPath = encodeURIComponent(update.path);
      // NEW: Path is now in URL (database-backed API)
      await unifiedApiClient.putData(`${API_BASE}/path/${encodedPath}`, update.value);

      logger.info(`[DEBOUNCE] Critical update processed immediately: ${update.path}`);

      // Remove from queue since it's been processed
      this.updateQueue.delete(update.path);
    } catch (error) {
      const errorMessage = isApiError(error)
        ? error.message
        : `Failed to update critical setting: ${update.path}`;
      logger.error(`[DEBOUNCE] Critical update failed for ${update.path}:`, error);
      throw new Error(errorMessage);
    }
  }

  private async processBatch(): Promise<void> {
    if (this.processingBatch || this.updateQueue.size === 0) {
      return;
    }

    this.processingBatch = true;
    this.debounceTimer = null;

    try {
      const updates = Array.from(this.updateQueue.values());
      const batchOperations: BatchOperation[] = updates.map(update => ({
        path: update.path,
        value: update.value
      }));

      // Sort by priority (Critical first, Low last)
      batchOperations.sort((a, b) => {
        const aPriority = this.determinePriority(a.path);
        const bPriority = this.determinePriority(b.path);
        return aPriority - bPriority;
      });

      logger.info(`[DEBOUNCE] Processing batch of ${batchOperations.length} updates, priority breakdown:`, {
        critical: batchOperations.filter(op => this.determinePriority(op.path) === UpdatePriority.Critical).length,
        high: batchOperations.filter(op => this.determinePriority(op.path) === UpdatePriority.High).length,
        normal: batchOperations.filter(op => this.determinePriority(op.path) === UpdatePriority.Normal).length,
        low: batchOperations.filter(op => this.determinePriority(op.path) === UpdatePriority.Low).length,
      });

      // Process in chunks to prevent server overload
      const chunks = this.chunkArray(batchOperations, BATCH_SIZE_LIMIT);
      
      for (const chunk of chunks) {
        await this.processBatchChunk(chunk);
      }

      // Clear the queue after successful processing
      this.updateQueue.clear();
      
    } catch (error) {
      logger.error('[DEBOUNCE] Batch processing failed:', error);
      // Don't clear queue on error, allow retry
      throw error;
    } finally {
      this.processingBatch = false;
    }
  }

  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  private async processBatchChunk(chunk: BatchOperation[]): Promise<void> {
    // NEW: Database backend doesn't have batch update endpoint yet
    // Use individual PUT calls for each path (database handles them efficiently)
    logger.info(`[DEBOUNCE] Processing chunk of ${chunk.length} updates via individual database writes`);

    const results = await Promise.allSettled(
      chunk.map(async ({ path, value }) => {
        try {
          const encodedPath = encodeURIComponent(path);
          // NEW: Path in URL, value as body (database-backed API)
          await unifiedApiClient.putData(`${API_BASE}/path/${encodedPath}`, value);
          return { path, success: true };
        } catch (err) {
          const errorMessage = isApiError(err) ? err.message : `Failed to update: ${path}`;
          logger.error(`[DEBOUNCE] Individual update failed for ${path}:`, err);
          throw new Error(errorMessage);
        }
      })
    );

    const failures = results.filter(result => result.status === 'rejected');
    if (failures.length > 0) {
      throw new Error(`${failures.length} out of ${chunk.length} individual updates failed`);
    }

    logger.info(`[DEBOUNCE] Successfully processed chunk of ${chunk.length} updates`);
  }

  // Force process all pending updates immediately
  async flush(): Promise<void> {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    await this.processBatch();
  }
}

// Global update manager instance
const updateManager = new SettingsUpdateManager();

export const settingsApi = {
  /**
   * Get a single setting by its dot-notation path
   * @param path - Dot notation path (e.g., "visualisation.nodes.baseColor")
   * @returns The setting value
   */
  async getSettingByPath(path: string): Promise<any> {
    try {
      const encodedPath = encodeURIComponent(path);
      // NEW: Path is now in URL, not query param (database-backed API)
      const result = await unifiedApiClient.getData(`${API_BASE}/path/${encodedPath}`);
      return result.value; // Backend returns { path, value }
    } catch (error) {
      const errorMessage = isApiError(error)
        ? error.message
        : `Failed to get setting at path: ${path}`;
      throw new Error(errorMessage);
    }
  },
  
  /**
   * Update a single setting by its dot-notation path with debouncing and priority handling
   * @param path - Dot notation path (e.g., "visualisation.nodes.baseColor")
   * @param value - New value for the setting
   */
  async updateSettingByPath(path: string, value: any): Promise<void> {
    // Route through debouncing manager for concurrent update handling
    return updateManager.queueUpdate(path, value);
  },
  
  /**
   * Get multiple settings by their paths in a single request using optimized batch endpoint
   * @param paths - Array of dot notation paths
   * @returns Object mapping paths to their values
   */
  async getSettingsByPaths(paths: string[]): Promise<Record<string, any>> {
    if (!paths || paths.length === 0) {
      return {};
    }

    try {
      // Use the optimized batch POST endpoint
      const result = await unifiedApiClient.postData(`${API_BASE}/batch`, { paths });
      logger.info(`Successfully fetched ${paths.length} settings using batch endpoint`);
      return result; // Server returns { path: value } mapping
    } catch (error) {
      logger.warn('Batch endpoint failed, falling back to individual requests:', error);
      
      // Fallback to individual path requests
      const result: Record<string, any> = {};
      const results = await Promise.allSettled(
        paths.map(async (path) => {
          try {
            const value = await this.getSettingByPath(path);
            return { path, value };
          } catch (err) {
            logger.error(`Failed to fetch path ${path}:`, err);
            return { path, value: undefined };
          }
        })
      );
      
      // Process results
      for (const [index, promiseResult] of results.entries()) {
        if (promiseResult.status === 'fulfilled') {
          const { path, value } = promiseResult.value;
          result[path] = value;
        } else {
          const path = paths[index];
          logger.error(`Failed to process path ${path}:`, promiseResult.reason);
          result[path] = undefined;
        }
      }
      
      logger.info(`Fallback completed: fetched ${Object.keys(result).length}/${paths.length} settings`);
      return result;
    }
  },
  
  /**
   * Update multiple settings by their paths
   * NEW: Database backend handles individual updates efficiently
   * @param updates - Array of path-value updates
   */
  async updateSettingsByPaths(updates: BatchOperation[]): Promise<void> {
    if (!updates || updates.length === 0) {
      return;
    }

    logger.info(`Updating ${updates.length} settings via database-backed API`);

    // Use individual path updates (database backend is optimized for this)
    const results = await Promise.allSettled(
      updates.map(async ({ path, value }) => {
        try {
          await this.updateSettingByPath(path, value);
          return { path, success: true };
        } catch (err) {
          logger.error(`Failed to update path ${path}:`, err);
          return { path, success: false, error: err };
        }
      })
    );

    // Check if any individual updates failed
    const failures = results
      .map((result, index) => ({
        result,
        update: updates[index]
      }))
      .filter(({ result }) => result.status === 'rejected' || (result.status === 'fulfilled' && !result.value.success));

    if (failures.length > 0) {
      logger.error(`${failures.length} out of ${updates.length} settings updates failed`);
      throw new Error(`Failed to update settings: ${failures.length} out of ${updates.length} individual path updates failed`);
    } else {
      logger.info(`Successfully updated ${updates.length} settings`);
    }
  },
  
  // REMOVED: Legacy bulk settings methods have been completely removed.
  // All settings operations now use granular path-based methods for better performance.
  
  /**
   * Reset settings to defaults
   */
  async resetSettings(): Promise<Settings> {
    try {
      const settings = await unifiedApiClient.postData<Settings>(`${API_BASE}/reset`);
      return settings;
    } catch (error) {
      const errorMessage = isApiError(error) ? error.message : 'Failed to reset settings';
      throw new Error(errorMessage);
    }
  },
  
  
  /**
   * Export settings to JSON string
   */
  exportSettings(settings: Settings): string {
    return JSON.stringify(settings, null, 2);
  },
  
  /**
   * Import settings from JSON string
   */
  importSettings(jsonString: string): Settings {
    try {
      const settings = JSON.parse(jsonString);
      
      // Basic validation
      if (!settings?.visualisation || !settings?.system) {
        throw new Error('Invalid settings format');
      }
      
      return settings as Settings;
    } catch (error) {
      logger.error('Failed to parse imported settings:', error);
      throw new Error('Invalid settings file format');
    }
  },
  
  /**
   * Force flush all pending debounced updates immediately
   * Useful before page navigation or critical operations
   */
  async flushPendingUpdates(): Promise<void> {
    return updateManager.flush();
  },
  
  /**
   * Update setting immediately without debouncing (use sparingly)
   * @param path - Dot notation path
   * @param value - New value
   */
  async updateSettingImmediately(path: string, value: any): Promise<void> {
    try {
      const encodedPath = encodeURIComponent(path);
      // NEW: Path in URL (database-backed API)
      await unifiedApiClient.putData(`${API_BASE}/path/${encodedPath}`, value);
    } catch (error) {
      const errorMessage = isApiError(error)
        ? error.message
        : `Failed to update setting immediately: ${path}`;
      throw new Error(errorMessage);
    }
  },

  /**
   * Get physics settings for a specific graph
   * CRITICAL: Maintains separation between logseq and visionflow graphs
   * @param graphName - 'logseq', 'visionflow', or 'default'
   * @returns Physics settings for the specified graph
   */
  async getPhysicsSettings(graphName: 'logseq' | 'visionflow' | 'default'): Promise<any> {
    try {
      const result = await unifiedApiClient.getData(`${API_BASE}/physics/${graphName}`);
      logger.info(`[PHYSICS] Loaded physics settings for graph: ${graphName}`);
      return result;
    } catch (error) {
      const errorMessage = isApiError(error)
        ? error.message
        : `Failed to get physics settings for graph: ${graphName}`;
      throw new Error(errorMessage);
    }
  },

  /**
   * Update physics settings for a specific graph
   * CRITICAL: Validates graph name to prevent conflation bug
   * @param graphName - 'logseq', 'visionflow', or 'default'
   * @param physicsSettings - Physics configuration object
   */
  async updatePhysicsSettings(
    graphName: 'logseq' | 'visionflow' | 'default',
    physicsSettings: any
  ): Promise<void> {
    // Validate graph name to prevent conflation
    if (graphName !== 'logseq' && graphName !== 'visionflow' && graphName !== 'default') {
      const error = `Invalid graph name: ${graphName}. Must be 'logseq', 'visionflow', or 'default'`;
      logger.error(`[PHYSICS] ${error}`);
      throw new Error(error);
    }

    try {
      await unifiedApiClient.putData(`${API_BASE}/physics/${graphName}`, physicsSettings);
      logger.info(`[PHYSICS] Updated physics settings for graph: ${graphName}`);
    } catch (error) {
      const errorMessage = isApiError(error)
        ? error.message
        : `Failed to update physics settings for graph: ${graphName}`;
      logger.error(`[PHYSICS] ${errorMessage}`, error);
      throw new Error(errorMessage);
    }
  },

  /**
   * Validate that logseq and visionflow graphs have separate physics settings
   * CRITICAL: Prevents the graph conflation bug mentioned by user
   * @returns true if graphs are properly separated, false if conflated
   */
  async validateGraphSeparation(): Promise<{ valid: boolean; message?: string }> {
    try {
      const logseqPhysics = await this.getPhysicsSettings('logseq');
      const visionflowPhysics = await this.getPhysicsSettings('visionflow');

      // Check if physics settings are identical (sign of conflation)
      const logseqStr = JSON.stringify(logseqPhysics);
      const visionflowStr = JSON.stringify(visionflowPhysics);

      if (logseqStr === visionflowStr) {
        const message = 'WARNING: Logseq and Visionflow graphs have identical physics settings - possible conflation!';
        logger.warn(`[PHYSICS] ${message}`);
        return { valid: false, message };
      }

      logger.info('[PHYSICS] Graph separation validated - logseq and visionflow have distinct settings');
      return { valid: true };
    } catch (error) {
      logger.error('[PHYSICS] Failed to validate graph separation:', error);
      return { valid: false, message: 'Failed to validate graph separation' };
    }
  },

  /**
   * Health check for settings service
   * @returns Health status and cache information
   */
  async getHealth(): Promise<any> {
    try {
      const result = await unifiedApiClient.getData(`${API_BASE}/health`);
      logger.info('[SETTINGS] Health check successful', result);
      return result;
    } catch (error) {
      logger.error('[SETTINGS] Health check failed:', error);
      throw error;
    }
  },

  /**
   * Clear settings cache on backend
   */
  async clearCache(): Promise<void> {
    try {
      await unifiedApiClient.postData(`${API_BASE}/cache/clear`, {});
      logger.info('[SETTINGS] Cache cleared successfully');
    } catch (error) {
      logger.error('[SETTINGS] Failed to clear cache:', error);
      throw error;
    }
  },

};

// Export types for use by other modules
export type { BatchOperation };