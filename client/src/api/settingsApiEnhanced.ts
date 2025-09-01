// Enhanced Settings API Client - Support for granular operations
import { Settings } from '../types/generated/settings';
import { createLogger } from '../utils/logger';

const logger = createLogger('SettingsAPI');
const API_BASE = '/api/settings';

export interface PathValue {
  path: string;
  value: any;
}

export interface GranularUpdateResponse {
  updatedPaths: string[];
  newValues: Record<string, any>;
}

export interface SettingsApiError {
  error: string;
  invalidPaths?: string[];
  validationErrors?: Array<{
    path: string;
    message: string;
  }>;
}

export const settingsApi = {
  /**
   * Fetch all settings from the server
   */
  async fetchSettings(): Promise<Settings> {
    const response = await fetch(API_BASE);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch settings' }));
      throw new Error(error.error || `Failed to fetch settings: ${response.statusText}`);
    }
    
    const settings = await response.json();
    return settings;
  },

  /**
   * Fetch specific settings by their dot-notation paths
   * @param paths Array of dot-notation paths like ['visualisation.glow.intensity', 'system.network.port']
   * @returns Partial settings object containing only requested paths
   */
  async getSettingsByPaths(paths: string[]): Promise<Partial<Settings>> {
    if (!paths.length) {
      throw new Error('At least one path must be specified');
    }

    const pathsQuery = paths.join(',');
    const response = await fetch(`${API_BASE}/get?paths=${encodeURIComponent(pathsQuery)}`);
    
    if (!response.ok) {
      const error: SettingsApiError = await response.json().catch(() => ({ error: 'Failed to fetch settings by paths' }));
      
      if (error.invalidPaths?.length) {
        throw new Error(`Invalid paths: ${error.invalidPaths.join(', ')}`);
      }
      
      throw new Error(error.error || `Failed to fetch settings by paths: ${response.statusText}`);
    }
    
    const partialSettings = await response.json();
    logger.debug(`Fetched ${paths.length} settings paths`, { paths, result: partialSettings });
    return partialSettings;
  },

  /**
   * Update specific settings using path-value pairs
   * @param updates Array of path-value objects
   * @returns Information about what was updated
   */
  async updateSettingsByPath(updates: PathValue[]): Promise<GranularUpdateResponse> {
    if (!updates.length) {
      throw new Error('At least one update must be specified');
    }

    // Validate update structure
    for (const update of updates) {
      if (!update.path || update.value === undefined) {
        throw new Error('Each update must have both path and value');
      }
    }

    const response = await fetch(`${API_BASE}/set`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    });
    
    if (!response.ok) {
      const error: SettingsApiError = await response.json().catch(() => ({ error: 'Failed to update settings by path' }));
      
      if (error.validationErrors?.length) {
        const errorMessages = error.validationErrors.map(e => `${e.path}: ${e.message}`);
        throw new Error(`Validation errors: ${errorMessages.join(', ')}`);
      }
      
      if (error.invalidPaths?.length) {
        throw new Error(`Invalid paths: ${error.invalidPaths.join(', ')}`);
      }
      
      throw new Error(error.error || `Failed to update settings by path: ${response.statusText}`);
    }
    
    const updateResponse = await response.json();
    logger.debug(`Updated ${updates.length} settings paths`, { updates, result: updateResponse });
    return updateResponse;
  },

  /**
   * Update a single setting by path
   * @param path Dot-notation path to the setting
   * @param value New value for the setting
   */
  async updateSettingByPath(path: string, value: any): Promise<GranularUpdateResponse> {
    return this.updateSettingsByPath([{ path, value }]);
  },

  /**
   * Get a single setting value by path
   * @param path Dot-notation path to the setting
   * @returns The setting value or undefined if not found
   */
  async getSettingByPath(path: string): Promise<any> {
    const result = await this.getSettingsByPaths([path]);
    return this.extractValueByPath(result, path);
  },

  /**
   * Batch multiple setting updates efficiently
   * Automatically splits large batches to avoid request size limits
   */
  async batchUpdateSettings(updates: PathValue[], batchSize: number = 50): Promise<GranularUpdateResponse> {
    if (!updates.length) {
      return { updatedPaths: [], newValues: {} };
    }

    if (updates.length <= batchSize) {
      return this.updateSettingsByPath(updates);
    }

    // Split into smaller batches
    const batches: PathValue[][] = [];
    for (let i = 0; i < updates.length; i += batchSize) {
      batches.push(updates.slice(i, i + batchSize));
    }

    const results = await Promise.all(
      batches.map(batch => this.updateSettingsByPath(batch))
    );

    // Combine results
    const combinedResult: GranularUpdateResponse = {
      updatedPaths: [],
      newValues: {}
    };

    for (const result of results) {
      combinedResult.updatedPaths.push(...result.updatedPaths);
      Object.assign(combinedResult.newValues, result.newValues);
    }

    return combinedResult;
  },

  /**
   * Subscribe to setting changes (mock implementation for testing)
   * In a real implementation, this would use WebSocket or SSE
   */
  subscribeToSettings(paths: string[], callback: (path: string, value: any) => void): () => void {
    // Mock subscription - in real implementation would establish WebSocket connection
    logger.debug('Subscribing to settings paths', { paths });
    
    // Return unsubscribe function
    return () => {
      logger.debug('Unsubscribing from settings paths', { paths });
    };
  },

  /**
   * Update settings with a partial update (legacy method for backwards compatibility)
   */
  async updateSettings(update: Partial<Settings>): Promise<Settings> {
    const response = await fetch(API_BASE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(update),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to update settings' }));
      throw new Error(error.error || `Failed to update settings: ${response.statusText}`);
    }
    
    const updatedSettings = await response.json();
    return updatedSettings;
  },

  /**
   * Save full settings object (legacy method)
   */
  async saveSettings(settings: Settings): Promise<Settings> {
    const response = await fetch(API_BASE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to save settings' }));
      throw new Error(error.error || `Failed to save settings: ${response.statusText}`);
    }
    
    const savedSettings = await response.json();
    return savedSettings;
  },

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
   * Reset specific settings paths to their default values
   */
  async resetSettingsPaths(paths: string[]): Promise<GranularUpdateResponse> {
    const response = await fetch(`${API_BASE}/reset-paths`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ paths }),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to reset settings paths' }));
      throw new Error(error.error || `Failed to reset settings paths: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * Get validation schema for settings paths
   */
  async getValidationSchema(paths?: string[]): Promise<Record<string, any>> {
    const url = new URL(`${API_BASE}/schema`, window.location.origin);
    if (paths?.length) {
      url.searchParams.set('paths', paths.join(','));
    }

    const response = await fetch(url.toString());
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to get validation schema' }));
      throw new Error(error.error || `Failed to get validation schema: ${response.statusText}`);
    }
    
    return response.json();
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
      if (!settings.visualisation || !settings.system) {
        throw new Error('Invalid settings format');
      }
      
      return settings as Settings;
    } catch (error) {
      logger.error('Failed to parse imported settings:', error);
      throw new Error('Invalid settings file format');
    }
  },

  /**
   * Helper to extract value from nested object by dot-notation path
   */
  extractValueByPath(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => {
      return current && current[key] !== undefined ? current[key] : undefined;
    }, obj);
  },

  /**
   * Helper to set value in nested object by dot-notation path
   */
  setValueByPath(obj: any, path: string, value: any): void {
    const keys = path.split('.');
    const lastKey = keys.pop()!;
    
    const target = keys.reduce((current, key) => {
      if (!current[key] || typeof current[key] !== 'object') {
        current[key] = {};
      }
      return current[key];
    }, obj);
    
    target[lastKey] = value;
  },

  /**
   * Validate path format
   */
  isValidPath(path: string): boolean {
    return /^[a-zA-Z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)*$/.test(path);
  },

  /**
   * Get available setting paths (for debugging/development)
   */
  async getAvailablePaths(): Promise<string[]> {
    const response = await fetch(`${API_BASE}/paths`);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to get available paths' }));
      throw new Error(error.error || `Failed to get available paths: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result.paths || [];
  }
};