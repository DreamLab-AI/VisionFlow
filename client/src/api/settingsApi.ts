// Settings API Client - Unified interface for all settings operations
import { Settings, SettingsUpdate } from '../features/settings/config/settings';
import { logger } from '../utils/logger';

const API_BASE = '/api/settings';

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
   * Update settings with a partial update
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
   * Save full settings object
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
};