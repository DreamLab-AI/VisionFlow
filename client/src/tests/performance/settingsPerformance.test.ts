import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AutoSaveManager } from '@/store/autoSaveManager';
import { measurePerformance, runConcurrently, createMockSettings } from '../utils/testFactories';

// Mock the API module
vi.mock('@/api/settings', () => ({
  updateSettings: vi.fn(),
  getSettings: vi.fn(),
}));

describe('Settings Performance Tests', () => {
  let mockUpdateSettings: any;
  let mockGetSettings: any;
  
  beforeEach(() => {
    vi.useFakeTimers();
    mockUpdateSettings = vi.fn();
    mockGetSettings = vi.fn();
    require('@/api/settings').updateSettings = mockUpdateSettings;
    require('@/api/settings').getSettings = mockGetSettings;
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe('AutoSaveManager Performance', () => {
    it('should handle rapid updates efficiently', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      const manager = new AutoSaveManager(500); // 500ms debounce
      
      const performance = await measurePerformance(async () => {
        // Simulate rapid user interactions (1000 updates)
        for (let i = 0; i < 1000; i++) {
          manager.scheduleUpdate(`test.path.${i % 10}`, `value_${i}`);
        }
        
        vi.advanceTimersByTime(1000);
        await vi.runAllTimersAsync();
      });
      
      expect(performance.average).toBeLessThan(100); // < 100ms
      expect(mockUpdateSettings.mock.calls.length).toBeLessThan(20); // Efficient batching
      
      manager.destroy();
    });

    it('should maintain performance with large batch sizes', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      const manager = new AutoSaveManager(100);
      
      const performance = await measurePerformance(async () => {
        // Create large batch update
        for (let i = 0; i < 500; i++) {
          manager.scheduleUpdate(`batch.item.${i}`, {
            id: i,
            data: `data_${i}`,
            timestamp: Date.now(),
            metadata: { index: i, type: 'batch_test' }
          });
        }
        
        vi.advanceTimersByTime(500);
        await vi.runAllTimersAsync();
      });
      
      expect(performance.average).toBeLessThan(200); // < 200ms for large batch
      expect(mockUpdateSettings).toHaveBeenCalled();
      
      manager.destroy();
    });

    it('should handle concurrent managers efficiently', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const managers = Array.from({ length: 10 }, () => new AutoSaveManager(100));
      
      const performance = await measurePerformance(async () => {
        // Each manager schedules updates concurrently
        managers.forEach((manager, index) => {
          for (let i = 0; i < 100; i++) {
            manager.scheduleUpdate(`manager${index}.item${i}`, `value_${i}`);
          }
        });
        
        vi.advanceTimersByTime(500);
        await vi.runAllTimersAsync();
      });
      
      expect(performance.average).toBeLessThan(500); // < 500ms for 10 concurrent managers
      
      managers.forEach(manager => manager.destroy());
    });

    it('should have minimal memory footprint', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      const manager = new AutoSaveManager(100);
      
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      // Create many updates and let them process
      for (let round = 0; round < 10; round++) {
        for (let i = 0; i < 100; i++) {
          manager.scheduleUpdate(`memory.test.${i}`, `round_${round}_value_${i}`);
        }
        
        vi.advanceTimersByTime(200);
        await vi.runAllTimersAsync();
        
        // Force garbage collection between rounds if available
        if (global.gc) {
          global.gc();
        }
      }
      
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be reasonable (< 5MB)
      if (initialMemory && finalMemory) {
        expect(memoryIncrease).toBeLessThan(5 * 1024 * 1024);
      }
      
      manager.destroy();
    });
  });

  describe('API Call Performance', () => {
    it('should optimize single field requests', async () => {
      const mockResponse = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5
          }
        }
      };
      
      mockGetSettings.mockResolvedValue(mockResponse);
      
      const performance = await measurePerformance(async () => {
        const { getSettings } = await import('@/api/settings');
        await getSettings(['visualisation.glow.nodeGlowStrength']);
      });
      
      expect(performance.average).toBeLessThan(50); // < 50ms for single field
      expect(mockGetSettings).toHaveBeenCalledTimes(1);
    });

    it('should handle large path lists efficiently', async () => {
      const largePaths = Array.from({ length: 100 }, (_, i) => `test.path${i}`);
      const mockResponse = createMockSettings();
      
      mockGetSettings.mockResolvedValue(mockResponse);
      
      const performance = await measurePerformance(async () => {
        const { getSettings } = await import('@/api/settings');
        await getSettings(largePaths);
      });
      
      expect(performance.average).toBeLessThan(200); // < 200ms for 100 paths
      expect(mockGetSettings).toHaveBeenCalledTimes(1);
    });

    it('should batch update requests efficiently', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true, updated: 50 });
      
      const largeUpdates = Array.from({ length: 50 }, (_, i) => ({
        path: `batch.update${i}`,
        value: {
          index: i,
          data: `data_${i}`,
          timestamp: Date.now()
        }
      }));
      
      const performance = await measurePerformance(async () => {
        const { updateSettings } = await import('@/api/settings');
        await updateSettings(largeUpdates);
      });
      
      expect(performance.average).toBeLessThan(300); // < 300ms for 50 updates
      expect(mockUpdateSettings).toHaveBeenCalledTimes(1);
      expect(mockUpdateSettings).toHaveBeenCalledWith(largeUpdates);
    });

    it('should handle concurrent API calls efficiently', async () => {
      mockGetSettings.mockResolvedValue(createMockSettings());
      mockUpdateSettings.mockResolvedValue({ success: true });
      
      const tasks = Array.from({ length: 20 }, (_, i) => async () => {
        if (i % 2 === 0) {
          // GET requests
          const { getSettings } = await import('@/api/settings');
          return await getSettings([`concurrent.get${i}`]);
        } else {
          // UPDATE requests
          const { updateSettings } = await import('@/api/settings');
          return await updateSettings([{
            path: `concurrent.update${i}`,
            value: `value_${i}`
          }]);
        }
      });
      
      const performance = await measurePerformance(async () => {
        await runConcurrently(tasks);
      });
      
      expect(performance.average).toBeLessThan(1000); // < 1s for 20 concurrent calls
      expect(mockGetSettings.mock.calls.length + mockUpdateSettings.mock.calls.length).toBe(20);
    });
  });

  describe('Data Processing Performance', () => {
    it('should serialize large settings objects quickly', async () => {
      const largeSettings = {
        ...createMockSettings(),
        visualisation: {
          ...createMockSettings().visualisation,
          colorSchemes: Array.from({ length: 1000 }, (_, i) => `scheme_${i}`),
          customNodes: Array.from({ length: 500 }, (_, i) => ({
            id: i,
            name: `node_${i}`,
            properties: {
              color: `#${Math.random().toString(16).substr(2, 6)}`,
              size: Math.random() * 100,
              data: `data_${i}`.repeat(10)
            }
          }))
        }
      };
      
      const performance = await measurePerformance(() => {
        const serialized = JSON.stringify(largeSettings);
        const parsed = JSON.parse(serialized);
        
        // Verify data integrity
        expect(parsed.visualisation.colorSchemes).toHaveLength(1000);
        expect(parsed.visualisation.customNodes).toHaveLength(500);
      }, 10);
      
      expect(performance.average).toBeLessThan(100); // < 100ms for large object serialization
    });

    it('should handle deep nested object updates efficiently', async () => {
      const createDeepObject = (depth: number): any => {
        if (depth === 0) return { value: 'leaf', id: Math.random() };
        return {
          level: depth,
          data: `level_${depth}_data`,
          nested: createDeepObject(depth - 1),
          siblings: Array.from({ length: 3 }, (_, i) => ({ id: i, value: `sibling_${i}` }))
        };
      };
      
      const deepObject = createDeepObject(20);
      
      const performance = await measurePerformance(() => {
        // Simulate deep object processing
        const serialized = JSON.stringify(deepObject);
        const parsed = JSON.parse(serialized);
        
        // Traverse and validate structure
        let current = parsed;
        let depth = 0;
        while (current.nested && depth < 25) {
          expect(current.level).toBeGreaterThan(0);
          expect(current.siblings).toHaveLength(3);
          current = current.nested;
          depth++;
        }
        
        expect(depth).toBe(20);
      }, 5);
      
      expect(performance.average).toBeLessThan(50); // < 50ms for deep object processing
    });

    it('should validate input data efficiently', async () => {
      const testCases = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 2.5, valid: true },
        { path: 'visualisation.glow.nodeGlowStrength', value: -1.0, valid: false },
        { path: 'visualisation.glow.baseColor', value: '#ff0000', valid: true },
        { path: 'visualisation.glow.baseColor', value: 'invalid', valid: false },
        { path: 'system.debugMode', value: true, valid: true },
        { path: 'system.debugMode', value: 'true', valid: false },
        { path: 'system.maxConnections', value: 100, valid: true },
        { path: 'system.maxConnections', value: -1, valid: false },
      ];
      
      const performance = await measurePerformance(() => {
        testCases.forEach(({ path, value, valid }) => {
          // Simulate validation logic (would be actual validation in real code)
          const isValid = validateTestValue(path, value);
          expect(isValid).toBe(valid);
        });
      }, 100);
      
      expect(performance.average).toBeLessThan(10); // < 10ms for validation batch
    });
  });

  describe('Memory Leak Detection', () => {
    it('should not leak memory with repeated manager creation/destruction', async () => {
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      // Create and destroy managers repeatedly
      for (let i = 0; i < 100; i++) {
        const manager = new AutoSaveManager(50);
        
        // Schedule some updates
        for (let j = 0; j < 10; j++) {
          manager.scheduleUpdate(`leak.test.${j}`, `value_${i}_${j}`);
        }
        
        // Process updates
        vi.advanceTimersByTime(100);
        await vi.runAllTimersAsync();
        
        // Destroy manager
        manager.destroy();
        
        // Periodic garbage collection
        if (i % 10 === 0 && global.gc) {
          global.gc();
        }
      }
      
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be minimal (< 2MB)
      if (initialMemory && finalMemory) {
        expect(memoryIncrease).toBeLessThan(2 * 1024 * 1024);
      }
    });

    it('should clean up event listeners and timers', () => {
      const managers: AutoSaveManager[] = [];
      
      // Create multiple managers
      for (let i = 0; i < 50; i++) {
        const manager = new AutoSaveManager(100);
        managers.push(manager);
        
        // Schedule updates to create internal state
        manager.scheduleUpdate(`cleanup.test.${i}`, `value_${i}`);
      }
      
      // Verify internal state exists
      managers.forEach(manager => {
        expect(manager.hasPendingUpdates()).toBe(true);
      });
      
      // Destroy all managers
      managers.forEach(manager => manager.destroy());
      
      // Verify cleanup
      managers.forEach(manager => {
        expect(manager.hasPendingUpdates()).toBe(false);
        expect(manager.getPendingUpdateCount()).toBe(0);
      });
    });
  });

  describe('Stress Testing', () => {
    it('should handle extreme load without degradation', async () => {
      mockUpdateSettings.mockResolvedValue({ success: true });
      const manager = new AutoSaveManager(100);
      
      const extremeLoad = async () => {
        // Create extreme load: 10,000 rapid updates
        for (let i = 0; i < 10000; i++) {
          manager.scheduleUpdate(
            `stress.test.${i % 100}`, // 100 unique paths
            {
              iteration: i,
              timestamp: Date.now(),
              data: `data_${i}`,
              metadata: {
                batch: Math.floor(i / 100),
                index: i % 100
              }
            }
          );
          
          // Occasional micro-delay to simulate real usage
          if (i % 1000 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }
        
        vi.advanceTimersByTime(500);
        await vi.runAllTimersAsync();
      };
      
      const performance = await measurePerformance(extremeLoad);
      
      expect(performance.average).toBeLessThan(2000); // < 2s for extreme load
      expect(mockUpdateSettings).toHaveBeenCalled();
      
      // Verify the system is still responsive
      manager.scheduleUpdate('post.stress.test', 'responsive');
      vi.advanceTimersByTime(200);
      await vi.runAllTimersAsync();
      
      manager.destroy();
    });

    it('should maintain accuracy under concurrent stress', async () => {
      mockUpdateSettings.mockImplementation((updates) => {
        // Simulate processing time
        return new Promise(resolve => {
          setTimeout(() => resolve({ success: true, updated: updates.length }), 10);
        });
      });
      
      const stressTasks = Array.from({ length: 50 }, (_, i) => async () => {
        const manager = new AutoSaveManager(50);
        
        // Each task creates its own load
        for (let j = 0; j < 100; j++) {
          manager.scheduleUpdate(`stress.task${i}.item${j}`, `value_${i}_${j}`);
        }
        
        vi.advanceTimersByTime(200);
        await vi.runAllTimersAsync();
        
        const pendingCount = manager.getPendingUpdateCount();
        manager.destroy();
        
        return { taskId: i, pendingCount };
      });
      
      const performance = await measurePerformance(async () => {
        const results = await runConcurrently(stressTasks, 10); // Max 10 concurrent
        
        // Verify all tasks completed successfully
        expect(results).toHaveLength(50);
        results.forEach(result => {
          expect(result.taskId).toBeGreaterThanOrEqual(0);
          expect(result.taskId).toBeLessThan(50);
          expect(result.pendingCount).toBe(0); // All updates should be processed
        });
      });
      
      expect(performance.average).toBeLessThan(5000); // < 5s for concurrent stress test
    });
  });

  // Helper function for validation simulation
  function validateTestValue(path: string, value: any): boolean {
    if (path.includes('nodeGlowStrength')) {
      return typeof value === 'number' && value >= 0 && value <= 10;
    }
    if (path.includes('baseColor')) {
      return typeof value === 'string' && /^#[0-9a-fA-F]{6}$/.test(value);
    }
    if (path.includes('debugMode')) {
      return typeof value === 'boolean';
    }
    if (path.includes('maxConnections')) {
      return typeof value === 'number' && value > 0 && value <= 10000;
    }
    return true;
  }
});