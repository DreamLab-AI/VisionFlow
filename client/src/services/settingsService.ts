import { Settings } from '../features/settings/config/settings';
import { apiService } from './apiService';
import { createLogger, createErrorMetadata } from '../utils/logger';
import { convertSnakeToCamelCase, convertCamelToSnakeCase } from '../utils/caseConversion';
import { debugState } from '../utils/debugState';

const logger = createLogger('SettingsService');

/**
 * Service for managing settings API interactions
 */
class SettingsService {
  private static instance: SettingsService;

  private constructor() {}

  public static getInstance(): SettingsService {
    if (!SettingsService.instance) {
      SettingsService.instance = new SettingsService();
    }
    return SettingsService.instance;
  }

  /**
   * Fetch settings from the server
   * @returns The settings from the server, converted to camelCase
   */
  public async fetchSettings(): Promise<Settings | null> {
    try {
      // Use new unified settings endpoint
      const rawSettings = await apiService.get<Record<string, any>>('/settings');
      
      // Server now sends camelCase directly from the new endpoint
      const settings = rawSettings as Settings;
      
      if (debugState.isEnabled()) {
        logger.info('Fetched settings from server:', { settings });
      }
      
      return settings;
    } catch (error) {
      logger.error('Failed to fetch settings:', createErrorMetadata(error));
      return null;
    }
  }

  /**
   * Save settings to the server
   * @param settings The settings to save, in camelCase
   * @param authHeaders Optional authentication headers
   * @returns The updated settings from the server
   */
  public async saveSettings(
    settings: Settings,
    authHeaders: Record<string, string> = {}
  ): Promise<Settings | null> {
    try {
      // New endpoint accepts camelCase directly
      if (debugState.isEnabled()) {
        logger.info('Saving settings to server:', { settings });
      }
      
      // Send settings to the server
      const rawUpdatedSettings = await apiService.post<Record<string, any>>(
        '/settings',
        settings,
        authHeaders
      );
      
      // Server returns camelCase from the new endpoint
      const updatedSettings = rawUpdatedSettings as Settings;
      
      if (debugState.isEnabled()) {
        logger.info('Settings saved successfully:', { updatedSettings });
      }
      
      return updatedSettings;
    } catch (error) {
      logger.error('Failed to save settings:', createErrorMetadata(error));
      return null;
    }
  }

  /**
   * Clear the settings cache on the server
   * @param authHeaders Authentication headers
   * @returns Whether the operation was successful
   */
  public async clearSettingsCache(authHeaders: Record<string, string>): Promise<boolean> {
    try {
      await apiService.post('/user-settings/clear-cache', {}, authHeaders);
      logger.info('Settings cache cleared successfully');
      return true;
    } catch (error) {
      logger.error('Failed to clear settings cache:', createErrorMetadata(error));
      return false;
    }
  }
}

export const settingsService = SettingsService.getInstance();
