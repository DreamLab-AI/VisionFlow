import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AutoSaveManager } from '@/store/autoSaveManager';
import { createMockSettings, createMockFetchResponse, waitFor, measurePerformance } from '../utils/testFactories';

// Mock the API module
vi.mock('@/api/settings', () => ({
  updateSettings: vi.fn(),
}));

describe('AutoSaveManager - Advanced Tests', () => {
  let autoSaveManager: AutoSaveManager;
  let mockUpdateSettings: any;
  
  beforeEach(() => {
    vi.useFakeTimers();
    mockUpdateSettings = vi.fn();
    require('@/api/settings').updateSettings = mockUpdateSettings;
    autoSaveManager = new AutoSaveManager();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    autoSaveManager.destroy();
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle null and undefined values gracefully', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });

      // Test null values
      autoSaveManager.scheduleUpdate('visualisation.glow.baseColor', null);
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();

      expect(mockUpdateSettings).toHaveBeenCalledWith([
        { path: 'visualisation.glow.baseColor', value: null }
      ]);
    });

    it('should handle extremely large objects', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const largeObject = {
        data: new Array(10000).fill(0).map((_, i) => ({ 
          id: i, 
          value: `item_${i}`,
          nested: { deep: { value: Math.random() } }
        }))
      };

      autoSaveManager.scheduleUpdate('visualisation.largeData', largeObject);
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();

      expect(mockUpdateSettings).toHaveBeenCalledWith([
        { path: 'visualisation.largeData', value: largeObject }
      ]);
    });

    it('should handle circular references by throwing meaningful error', () => {
      const circularObj: any = { name: 'test' };
      circularObj.self = circularObj;

      expect(() => {
        autoSaveManager.scheduleUpdate('test.circular', circularObj);
      }).toThrow(/circular/i);
    });

    it('should handle API failures with exponential backoff', async () => {
      const attempts: number[] = [];
      
      mockUpdateSettings.mockImplementation(() => {
        attempts.push(Date.now());
        return Promise.reject(new Error('Network error'));
      });

      autoSaveManager.scheduleUpdate('test.path', 'value');
      
      // Initial attempt
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      expect(attempts).toHaveLength(1);

      // First retry (2s delay)
      vi.advanceTimersByTime(2000);
      await vi.runAllTimersAsync();
      expect(attempts).toHaveLength(2);

      // Second retry (4s delay)
      vi.advanceTimersByTime(4000);
      await vi.runAllTimersAsync();
      expect(attempts).toHaveLength(3);

      // Third retry (8s delay)
      vi.advanceTimersByTime(8000);
      await vi.runAllTimersAsync();
      expect(attempts).toHaveLength(4);

      // After max retries, should stop
      vi.advanceTimersByTime(16000);
      await vi.runAllTimersAsync();
      expect(attempts).toHaveLength(4);
    });

    it('should handle mixed success/failure batches correctly', async () => {
      let callCount = 0;
      mockUpdateSettings.mockImplementation((updates) => {
        callCount++;
        if (callCount === 1) {
          // First call fails
          return Promise.reject(new Error('Temporary failure'));
        }
        // Subsequent calls succeed
        return Promise.resolve({ success: true });
      });

      autoSaveManager.scheduleUpdate('path1', 'value1');
      autoSaveManager.scheduleUpdate('path2', 'value2');
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();

      expect(mockUpdateSettings).toHaveBeenCalledTimes(1);
      
      // Wait for retry
      vi.advanceTimersByTime(2000);
      await vi.runAllTimersAsync();

      expect(mockUpdateSettings).toHaveBeenCalledTimes(2);
      expect(mockUpdateSettings).toHaveBeenLastCalledWith([
        { path: 'path1', value: 'value1' },
        { path: 'path2', value: 'value2' }
      ]);
    });
  });

  describe('Performance and Memory Management', () => {
    it('should handle high-frequency updates efficiently', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const updateCount = 1000;
      const startMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      // Schedule many rapid updates
      for (let i = 0; i < updateCount; i++) {
        autoSaveManager.scheduleUpdate(`test.path.${i % 10}`, `value_${i}`);
        
        // Some updates to same path (should be debounced)
        if (i % 3 === 0) {
          autoSaveManager.scheduleUpdate('test.common', `common_${i}`);
        }
      }

      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();

      // Should batch efficiently - much fewer calls than updates
      expect(mockUpdateSettings.mock.calls.length).toBeLessThan(updateCount / 5);
      
      // Memory should not leak significantly
      if ((performance as any).memory) {
        const endMemory = (performance as any).memory.usedJSHeapSize;
        const memoryIncrease = endMemory - startMemory;
        expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024); // < 10MB
      }
    });

    it('should maintain performance with large batch sizes', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const batchSize = 500;
      
      for (let i = 0; i < batchSize; i++) {
        autoSaveManager.scheduleUpdate(`test.batch.${i}`, { 
          data: `value_${i}`,
          timestamp: Date.now(),
          metadata: { index: i, type: 'test' }
        });
      }

      const performanceResult = await measurePerformance(async () => {
        vi.advanceTimersByTime(1000);
        await vi.runAllTimersAsync();
      });

      expect(performanceResult.average).toBeLessThan(100); // < 100ms
      expect(mockUpdateSettings).toHaveBeenCalled();
    });

    it('should clean up resources properly on destroy', () => {
      const manager = new AutoSaveManager();
      
      // Schedule some updates
      manager.scheduleUpdate('test1', 'value1');
      manager.scheduleUpdate('test2', 'value2');
      
      // Verify internal state exists
      expect((manager as any).pendingUpdates.size).toBeGreaterThan(0);
      
      // Destroy should clean up
      manager.destroy();
      
      expect((manager as any).pendingUpdates.size).toBe(0);
      expect((manager as any).saveTimeout).toBeNull();
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle concurrent updates to same path correctly', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const promises = [];
      
      // Simulate concurrent updates
      for (let i = 0; i < 10; i++) {
        promises.push(
          new Promise<void>((resolve) => {
            setTimeout(() => {
              autoSaveManager.scheduleUpdate('test.concurrent', `value_${i}`);
              resolve();
            }, i * 10);
          })
        );
      }
      
      await Promise.all(promises);
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalledTimes(1);
      // Should have the last value
      expect(mockUpdateSettings).toHaveBeenCalledWith([
        { path: 'test.concurrent', value: 'value_9' }
      ]);
    });

    it('should handle rapid successive calls without race conditions', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      // Rapid fire updates
      for (let i = 0; i < 100; i++) {
        autoSaveManager.scheduleUpdate(`path${i % 5}`, `value${i}`);
      }
      
      vi.advanceTimersByTime(500); // Half the debounce time
      
      // More updates while first batch is pending
      for (let i = 100; i < 200; i++) {
        autoSaveManager.scheduleUpdate(`path${i % 5}`, `value${i}`);
      }
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      // Should have handled all updates correctly
      expect(mockUpdateSettings).toHaveBeenCalled();
      const calls = mockUpdateSettings.mock.calls;
      const totalUpdates = calls.reduce((sum, call) => sum + call[0].length, 0);
      expect(totalUpdates).toBe(5); // 5 unique paths
    });
  });

  describe('Complex Data Structure Handling', () => {
    it('should handle deeply nested objects', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const deepObject = {
        level1: {
          level2: {
            level3: {
              level4: {
                level5: {
                  value: 'deep',
                  array: [1, 2, { nested: true }],
                  null_value: null,
                  undefined_value: undefined
                }
              }
            }
          }
        }
      };
      
      autoSaveManager.scheduleUpdate('test.deep', deepObject);
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalledWith([
        { path: 'test.deep', value: deepObject }
      ]);
    });

    it('should handle arrays with mixed data types', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const mixedArray = [
        'string',
        42,
        true,
        null,
        { object: 'in array' },
        [1, 2, 3],
        undefined
      ];
      
      autoSaveManager.scheduleUpdate('test.mixed', mixedArray);
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalledWith([
        { path: 'test.mixed', value: mixedArray }
      ]);
    });

    it('should handle special JavaScript values', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const specialValues = {
        infinity: Infinity,
        negInfinity: -Infinity,
        nan: NaN,
        date: new Date(),
        regex: /test/gi,
        bigNumber: Number.MAX_SAFE_INTEGER
      };
      
      autoSaveManager.scheduleUpdate('test.special', specialValues);
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalled();
    });
  });

  describe('Resilience and Recovery', () => {
    it('should recover from temporary network issues', async () => {
      let failCount = 0;
      mockUpdateSettings.mockImplementation(() => {
        failCount++;
        if (failCount <= 2) {
          return Promise.reject(new Error('Network timeout'));
        }
        return Promise.resolve({ success: true });
      });
      
      autoSaveManager.scheduleUpdate('test.recovery', 'test_value');
      
      // Initial attempt fails
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      // First retry fails
      vi.advanceTimersByTime(2000);
      await vi.runAllTimersAsync();
      
      // Second retry succeeds
      vi.advanceTimersByTime(4000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalledTimes(3);
      expect(failCount).toBe(3);
    });

    it('should handle partial batch failures correctly', async () => {
      mockUpdateSettings
        .mockResolvedValueOnce({ success: false, error: 'Validation failed' })
        .mockResolvedValueOnce({ success: true });
      
      autoSaveManager.scheduleUpdate('valid.path', 'good_value');
      autoSaveManager.scheduleUpdate('invalid.path', 'bad_value');
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      // Should retry failed batch
      vi.advanceTimersByTime(2000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalledTimes(2);
    });
  });

  describe('State Management and Consistency', () => {
    it('should maintain update order for same path', async () => {
      const updateOrder: string[] = [];
      
      mockUpdateSettings.mockImplementation((updates) => {
        updates.forEach((update: any) => {
          updateOrder.push(update.value);
        });
        return Promise.resolve({ success: true });
      });
      
      // Schedule updates in specific order
      autoSaveManager.scheduleUpdate('test.order', 'first');
      
      setTimeout(() => {
        autoSaveManager.scheduleUpdate('test.order', 'second');
      }, 100);
      
      setTimeout(() => {
        autoSaveManager.scheduleUpdate('test.order', 'third');
      }, 200);
      
      vi.advanceTimersByTime(1500);
      await vi.runAllTimersAsync();
      
      // Should only have the final value
      expect(updateOrder).toEqual(['third']);
    });

    it('should provide accurate pending state information', () => {
      autoSaveManager.scheduleUpdate('path1', 'value1');
      autoSaveManager.scheduleUpdate('path2', 'value2');
      
      expect(autoSaveManager.hasPendingUpdates()).toBe(true);
      expect(autoSaveManager.getPendingPaths()).toContain('path1');
      expect(autoSaveManager.getPendingPaths()).toContain('path2');
      expect(autoSaveManager.getPendingUpdateCount()).toBe(2);
    });

    it('should clear pending state after successful save', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      autoSaveManager.scheduleUpdate('path1', 'value1');
      expect(autoSaveManager.hasPendingUpdates()).toBe(true);
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      expect(autoSaveManager.hasPendingUpdates()).toBe(false);
      expect(autoSaveManager.getPendingUpdateCount()).toBe(0);
    });
  });

  describe('Configuration and Customization', () => {
    it('should respect custom debounce delays', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const customManager = new AutoSaveManager(2000); // 2 second debounce
      
      customManager.scheduleUpdate('test', 'value');
      
      // Should not trigger after normal debounce time
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      expect(mockUpdateSettings).not.toHaveBeenCalled();
      
      // Should trigger after custom debounce time
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      expect(mockUpdateSettings).toHaveBeenCalled();
      
      customManager.destroy();
    });

    it('should handle zero debounce delay for immediate updates', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const immediateManager = new AutoSaveManager(0);
      
      immediateManager.scheduleUpdate('test', 'value');
      
      // Should trigger immediately
      await vi.runAllTimersAsync();
      expect(mockUpdateSettings).toHaveBeenCalled();
      
      immediateManager.destroy();
    });
  });

  describe('Integration Scenarios', () => {
    it('should work correctly with rapid user interactions', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      // Simulate rapid slider movements
      for (let i = 0; i <= 100; i += 5) {
        autoSaveManager.scheduleUpdate('visualisation.glow.nodeGlowStrength', i / 100);
        vi.advanceTimersByTime(50); // 50ms between updates
      }
      
      // Wait for debounce
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalledTimes(1);
      expect(mockUpdateSettings).toHaveBeenCalledWith([
        { path: 'visualisation.glow.nodeGlowStrength', value: 1.0 }
      ]);
    });

    it('should handle form field updates correctly', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      // Simulate typing in text fields
      const textFields = [
        'visualisation.glow.baseColor',
        'system.audit.auditLogPath',
        'xr.spaceType'
      ];
      
      textFields.forEach((field, index) => {
        // Simulate incremental typing
        for (let i = 1; i <= 10; i++) {
          autoSaveManager.scheduleUpdate(field, `value${index}_${'x'.repeat(i)}`);
          vi.advanceTimersByTime(100);
        }
      });
      
      vi.advanceTimersByTime(1000);
      await vi.runAllTimersAsync();
      
      expect(mockUpdateSettings).toHaveBeenCalledTimes(1);
      const call = mockUpdateSettings.mock.calls[0][0];
      expect(call).toHaveLength(3);
      expect(call.find((u: any) => u.path === textFields[0]).value).toBe('value0_xxxxxxxxxx');
    });
  });
});