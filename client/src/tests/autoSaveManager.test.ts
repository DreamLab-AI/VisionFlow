import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { AutoSaveManager } from '../store/autoSaveManager';
import { settingsApi } from '../api/settingsApi';

// Mock the settingsApi
vi.mock('../api/settingsApi');
const mockSettingsApi = vi.mocked(settingsApi);

// Mock toast
vi.mock('../features/design-system/components/Toast', () => ({
  toast: {
    error: vi.fn()
  }
}));

// Mock logger
vi.mock('../utils/logger', () => ({
  createLogger: () => ({
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn()
  })
}));

describe('AutoSaveManager', () => {
  let autoSaveManager: AutoSaveManager;

  beforeEach(() => {
    autoSaveManager = new AutoSaveManager();
    autoSaveManager.setInitialized(true);
    vi.clearAllMocks();
    vi.clearAllTimers();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('queueChange', () => {
    it('should queue changes and debounce batch updates', async () => {
      mockSettingsApi.updateSettingsByPaths.mockResolvedValue();

      // Queue multiple changes
      autoSaveManager.queueChange('setting1', 'value1');
      autoSaveManager.queueChange('setting2', 'value2');
      autoSaveManager.queueChange('setting1', 'updatedValue1'); // Should overwrite

      expect(autoSaveManager.getPendingCount()).toBe(2);
      expect(autoSaveManager.hasPendingChanges()).toBe(true);

      // Fast-forward past debounce delay
      vi.advanceTimersByTime(600);

      // Wait for async flush
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledWith([
        { path: 'setting1', value: 'updatedValue1' },
        { path: 'setting2', value: 'value2' }
      ]);

      expect(autoSaveManager.getPendingCount()).toBe(0);
      expect(autoSaveManager.hasPendingChanges()).toBe(false);
    });

    it('should not queue changes when not initialized', () => {
      autoSaveManager.setInitialized(false);
      
      autoSaveManager.queueChange('setting1', 'value1');
      
      expect(autoSaveManager.getPendingCount()).toBe(0);
      expect(autoSaveManager.hasPendingChanges()).toBe(false);
    });
  });

  describe('queueChanges', () => {
    it('should queue multiple changes in batch', async () => {
      mockSettingsApi.updateSettingsByPaths.mockResolvedValue();

      const changes = new Map([
        ['setting1', 'value1'],
        ['setting2', 'value2'],
        ['setting3', 'value3']
      ]);

      autoSaveManager.queueChanges(changes);

      expect(autoSaveManager.getPendingCount()).toBe(3);

      // Fast-forward past debounce delay
      vi.advanceTimersByTime(600);
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledWith([
        { path: 'setting1', value: 'value1' },
        { path: 'setting2', value: 'value2' },
        { path: 'setting3', value: 'value3' }
      ]);
    });
  });

  describe('forceFlush', () => {
    it('should immediately flush pending changes', async () => {
      mockSettingsApi.updateSettingsByPaths.mockResolvedValue();

      autoSaveManager.queueChange('setting1', 'value1');
      
      // Force flush should bypass debounce
      await autoSaveManager.forceFlush();

      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledWith([
        { path: 'setting1', value: 'value1' }
      ]);
      expect(autoSaveManager.getPendingCount()).toBe(0);
    });
  });

  describe('retry logic', () => {
    it('should retry failed saves with exponential backoff', async () => {
      const error = new Error('Network error');
      mockSettingsApi.updateSettingsByPaths
        .mockRejectedValueOnce(error)
        .mockRejectedValueOnce(error)
        .mockResolvedValueOnce();

      autoSaveManager.queueChange('setting1', 'value1');

      // Trigger first attempt
      vi.advanceTimersByTime(600);
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledTimes(1);
      expect(autoSaveManager.hasPendingChanges()).toBe(true); // Still pending due to error

      // First retry (after 1s)
      vi.advanceTimersByTime(1000);
      await new Promise(resolve => setTimeout(resolve, 0));
      vi.advanceTimersByTime(600); // Additional debounce
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledTimes(2);

      // Second retry (after 2s)
      vi.advanceTimersByTime(2000);
      await new Promise(resolve => setTimeout(resolve, 0));
      vi.advanceTimersByTime(600); // Additional debounce
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledTimes(3);
      expect(autoSaveManager.hasPendingChanges()).toBe(false); // Should be cleared on success
    });

    it('should show error toast after max retries exceeded', async () => {
      const { toast } = require('../client/src/features/design-system/components/Toast');
      const error = new Error('Persistent network error');
      mockSettingsApi.updateSettingsByPaths.mockRejectedValue(error);

      autoSaveManager.queueChange('setting1', 'value1');

      // Initial attempt + 3 retries
      for (let i = 0; i < 4; i++) {
        vi.advanceTimersByTime(600);
        await new Promise(resolve => setTimeout(resolve, 0));
        if (i < 3) {
          vi.advanceTimersByTime(1000 * Math.pow(2, i)); // Exponential backoff
        }
      }

      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledTimes(4);
      expect(toast.error).toHaveBeenCalledWith('Failed to save setting: setting1');
      expect(autoSaveManager.hasPendingChanges()).toBe(true); // Still pending for user to retry later
    });
  });

  describe('debouncing behavior', () => {
    it('should reset debounce timer on new changes', async () => {
      mockSettingsApi.updateSettingsByPaths.mockResolvedValue();

      autoSaveManager.queueChange('setting1', 'value1');
      
      // Advance part way through debounce
      vi.advanceTimersByTime(300);
      
      // Add another change - should reset timer
      autoSaveManager.queueChange('setting2', 'value2');
      
      // Advance the remainder of original delay
      vi.advanceTimersByTime(300);
      
      // Should not have flushed yet
      expect(mockSettingsApi.updateSettingsByPaths).not.toHaveBeenCalled();
      
      // Advance the full delay from the second change
      vi.advanceTimersByTime(300);
      await new Promise(resolve => setTimeout(resolve, 0));
      
      // Now should have flushed
      expect(mockSettingsApi.updateSettingsByPaths).toHaveBeenCalledWith([
        { path: 'setting1', value: 'value1' },
        { path: 'setting2', value: 'value2' }
      ]);
    });
  });
});