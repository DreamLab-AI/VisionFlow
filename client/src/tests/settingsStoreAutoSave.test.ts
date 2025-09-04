import { describe, it, expect, beforeEach, vi } from 'vitest';
import { settingsStoreUtils } from '../store/settingsStore';

// Test to verify AutoSaveManager integration in settingsStore
describe('SettingsStore AutoSave Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('AutoSaveManager availability', () => {
    it('should expose AutoSaveManager through settingsStoreUtils', () => {
      expect(settingsStoreUtils.autoSaveManager).toBeDefined();
      expect(settingsStoreUtils.autoSaveManager.queueChange).toBeTypeOf('function');
      expect(settingsStoreUtils.autoSaveManager.queueChanges).toBeTypeOf('function');
      expect(settingsStoreUtils.autoSaveManager.forceFlush).toBeTypeOf('function');
      expect(settingsStoreUtils.autoSaveManager.hasPendingChanges).toBeTypeOf('function');
      expect(settingsStoreUtils.autoSaveManager.getPendingCount).toBeTypeOf('function');
      expect(settingsStoreUtils.autoSaveManager.setInitialized).toBeTypeOf('function');
    });

    it('should provide access to utility functions', () => {
      expect(settingsStoreUtils.getSectionPaths).toBeTypeOf('function');
      expect(settingsStoreUtils.setNestedValue).toBeTypeOf('function');
      expect(settingsStoreUtils.getAllSettingsPaths).toBeTypeOf('function');
    });
  });

  describe('AutoSaveManager basic functionality', () => {
    it('should start with no pending changes', () => {
      expect(settingsStoreUtils.autoSaveManager.getPendingCount()).toBe(0);
      expect(settingsStoreUtils.autoSaveManager.hasPendingChanges()).toBe(false);
    });

    it('should handle initialization state', () => {
      // Test that we can set initialized state
      settingsStoreUtils.autoSaveManager.setInitialized(true);
      settingsStoreUtils.autoSaveManager.queueChange('test.setting', 'testValue');
      expect(settingsStoreUtils.autoSaveManager.getPendingCount()).toBe(1);
      
      // Reset for other tests
      settingsStoreUtils.autoSaveManager.setInitialized(false);
    });
  });
});