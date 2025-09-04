// Settings API Client - Granular path-based operations only
// This file implements ONLY the new granular API methods - old full-object methods have been removed
import { SettingsPath } from '../types/generated/settings';
import { logger } from '../utils/logger';

const API_BASE = '/api/settings';

// Types for granular API operations
export interface PathValuePair {
  path: string;
  value: any;
}

export interface SettingsByPathsResponse {
  [path: string]: any;
}

/**
 * Granular Settings API - Path-based operations for optimal network efficiency
 * All methods work with dot-notation paths (e.g., 'visualisation.nodes.opacity')
 */
export const settingsApi = {
  /**
   * Fetch specific settings by their dot-notation paths
   * This reduces network traffic by only requesting needed values
   * @param paths Array of dot-notation paths to fetch
   * @returns Object with path as key and setting value as value
   */
  async getSettingsByPaths(paths: string[]): Promise<SettingsByPathsResponse> {
    if (!paths.length) {
      throw new Error('No paths provided');
    }

    const queryParams = new URLSearchParams();
    queryParams.append('paths', paths.join(','));
    
    const response = await fetch(`${API_BASE}/get?${queryParams.toString()}`);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Failed to fetch settings by paths' }));
      throw new Error(error.error || `Failed to fetch settings by paths: ${response.statusText}`);
    }
    
    const result = await response.json();
    logger.debug('Fetched settings by paths:', { paths, result });
    return result;
  },

  /**
   * Update specific settings by their paths
   * This reduces network traffic by only sending changed values
   * @param updates Array of path-value pairs to update
   * @returns Updated settings object (partial or full based on backend implementation)
   */
  async updateSettingsByPath(updates: PathValuePair[]): Promise<any> {
    if (!updates.length) {
      throw new Error('No updates provided');
    }

    // Log the request for debugging
    logger.debug('Sending settings update request:', { updates });

    const response = await fetch(`${API_BASE}/set`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ updates }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      logger.error('Settings update failed:', { 
        status: response.status, 
        statusText: response.statusText,
        updates,
        errorResponse: errorText 
      });
      const error = (() => {
        try {
          return JSON.parse(errorText);
        } catch {
          return { error: errorText || 'Failed to update settings by path' };
        }
      })();
      throw new Error(error.error || `Failed to update settings by path: ${response.statusText}`);
    }
    
    const result = await response.json();
    logger.debug('Updated settings by path:', { updates, result });
    return result;
  },

  /**
   * Get a single setting value by its path
   * @param path Dot-notation path to the setting
   * @returns The setting value
   */
  async getSetting(path: string): Promise<any> {
    if (!path.trim()) {
      throw new Error('Path cannot be empty');
    }

    const result = await this.getSettingsByPaths([path]);
    return result[path];
  },

  /**
   * Set a single setting value by its path
   * @param path Dot-notation path to the setting
   * @param value New value for the setting
   * @returns Updated settings (partial or full based on backend)
   */
  async setSetting(path: string, value: any): Promise<any> {
    if (!path.trim()) {
      throw new Error('Path cannot be empty');
    }

    return this.updateSettingsByPath([{ path, value }]);
  },

  /**
   * Reset settings by fetching defaults for specific sections
   * This replaces the old fetchSettings method with a more granular approach
   * @param sections Array of section names to reset (defaults to all essential sections)
   * @returns Settings object with reset values
   */
  async resetSettingsSections(sections: string[] = ['essential']): Promise<any> {
    try {
      // For now, we'll use the existing GET endpoint for reset operations
      // In the future, this could be replaced with a dedicated reset endpoint
      const response = await fetch(`${API_BASE}/reset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sections }),
      });

      if (!response.ok) {
        // Fallback to full settings endpoint if reset endpoint doesn't exist
        const fallbackResponse = await fetch(`${API_BASE}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!fallbackResponse.ok) {
          throw new Error(`HTTP ${fallbackResponse.status}: ${fallbackResponse.statusText}`);
        }

        const settings = await fallbackResponse.json();
        logger.debug('Reset settings using fallback method');
        return settings;
      }

      const settings = await response.json();
      logger.debug('Reset settings for sections:', sections);
      return settings;
    } catch (error) {
      logger.error('Failed to reset settings:', error);
      throw error;
    }
  },
};