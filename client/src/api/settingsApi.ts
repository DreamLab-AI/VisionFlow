// Settings API Client - Path-based interface for granular settings operations
import { Settings } from '../features/settings/config/settings';
import { logger } from '../utils/logger';

const API_BASE = '/api/settings';

// Batch operation interface for multiple path updates
interface BatchOperation {
  path: string;
  value: any;
}

export const settingsApi = {
  /**
   * Get a single setting by its dot-notation path
   * @param path - Dot notation path (e.g., "visualisation.nodes.baseColor")
   * @returns The setting value
   */
  async getSettingByPath(path: string): Promise<any> {
    const encodedPath = encodeURIComponent(path);
    const response = await fetch(`${API_BASE}/path?path=${encodedPath}`);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: `Failed to get setting at path: ${path}` }));
      throw new Error(error.error || `Failed to get setting at path ${path}: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result.value; // Backend returns { value: actualValue }
  },
  
  /**
   * Update a single setting by its dot-notation path
   * @param path - Dot notation path (e.g., "visualisation.nodes.baseColor")
   * @param value - New value for the setting
   */
  async updateSettingByPath(path: string, value: any): Promise<void> {
    const response = await fetch(`${API_BASE}/path`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ path, value }),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: `Failed to update setting at path: ${path}` }));
      throw new Error(error.error || `Failed to update setting at path ${path}: ${response.statusText}`);
    }
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
      const response = await fetch(`${API_BASE}/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ paths }),
      });
      
      if (!response.ok) {
        throw new Error(`Batch read failed: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
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
   * Update multiple settings by their paths in a single transaction
   * @param updates - Array of path-value updates
   */
  async updateSettingsByPaths(updates: BatchOperation[]): Promise<void> {
    if (!updates || updates.length === 0) {
      return;
    }

    try {
      // Try the server's batch endpoint first
      const response = await fetch(`${API_BASE}/batch`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          updates
        }),
      });
      
      if (response.ok) {
        logger.info(`Successfully updated ${updates.length} settings using batch endpoint`);
        return;
      }
      
      // If batch endpoint fails, fall back to individual updates
      logger.warn(`Batch endpoint failed (${response.status}), falling back to individual updates`);
      throw new Error(`Batch endpoint returned ${response.status}`);
      
    } catch (error) {
      logger.warn('Error with batch endpoint, attempting individual updates fallback:', error);
      
      // Fallback: Use individual path updates for better reliability
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
        
        // If individual updates fail, we don't have a fallback since we're removing legacy bulk updates
        throw new Error(`Failed to update settings: ${failures.length} out of ${updates.length} individual path updates failed`);
      } else {
        logger.info(`Successfully updated ${updates.length} settings using individual path updates`);
      }
    }
  },
  
  // REMOVED: Legacy bulk settings methods have been completely removed.
  // All settings operations now use granular path-based methods for better performance.
  
  /**
   * Reset settings to defaults
   */
  async resetSettings(): Promise<Settings> {
    const response = await fetch(`${API_BASE}/reset`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to reset settings' }));
      throw new Error(error.error || `Failed to reset settings: ${response.statusText}`);
    }
    
    const settings = await response.json();
    return settings;
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
  
};

// Export types for use by other modules
export type { BatchOperation };