// Settings API Client - Path-based interface for granular settings operations
import { Settings, SettingsUpdate } from '../features/settings/config/settings';
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
    const response = await fetch(`${API_BASE}/${encodedPath}`);
    
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
    const encodedPath = encodeURIComponent(path);
    const response = await fetch(`${API_BASE}/${encodedPath}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ value }),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: `Failed to update setting at path: ${path}` }));
      throw new Error(error.error || `Failed to update setting at path ${path}: ${response.statusText}`);
    }
  },
  
  /**
   * Get multiple settings by their paths in a single request
   * @param paths - Array of dot notation paths
   * @returns Object mapping paths to their values
   */
  async getSettingsByPaths(paths: string[]): Promise<Record<string, any>> {
    const response = await fetch(`${API_BASE}/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'get',
        paths
      }),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to get settings by paths' }));
      throw new Error(error.error || `Failed to get settings by paths: ${response.statusText}`);
    }
    
    return await response.json();
  },
  
  /**
   * Update multiple settings by their paths in a single transaction
   * @param updates - Array of path-value updates
   */
  async updateSettingsByPaths(updates: BatchOperation[]): Promise<void> {
    const response = await fetch(`${API_BASE}/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        operation: 'update',
        updates
      }),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to update settings by paths' }));
      throw new Error(error.error || `Failed to update settings by paths: ${response.statusText}`);
    }
  },
  
  // Legacy methods kept for backward compatibility during transition
  /**
   * Fetch all settings from the server (DEPRECATED - use path-based methods)
   * @deprecated Use getSettingsByPaths for better performance
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
   * Update settings with a partial update (DEPRECATED - use path-based methods)
   * @deprecated Use updateSettingsByPaths for better performance
   */
  async updateSettings(update: SettingsUpdate): Promise<Settings> {
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
   * Save full settings object (DEPRECATED - use path-based methods)
   * @deprecated Use updateSettingsByPaths for better performance
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
  
  // Export batch operation type for external use
  BatchOperation,
};

// Export types for use by other modules
export type { BatchOperation };