import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createLogger, createAgentLogger, loggerManager } from './dynamicLogger';
import { clientDebugState } from './clientDebugState';

// Mock dependencies
vi.mock('./clientDebugState', () => ({
  clientDebugState: {
    get: vi.fn(),
    set: vi.fn(),
    subscribe: vi.fn(),
    isEnabled: vi.fn(),
    getAll: vi.fn(),
  }
}));

vi.mock('./loggerConfig', () => ({
  createDynamicLogger: vi.fn(),
  createDynamicAgentLogger: vi.fn(),
  updateAllLoggers: vi.fn(),
}));

vi.mock('./logger', () => ({
  createLogger: vi.fn(),
  createAgentLogger: vi.fn(),
}));

describe('Runtime Logger Integration Tests', () => {
  let subscriptionCallbacks: Record<string, Function>;
  let mockLogger: any;
  let mockAgentLogger: any;

  beforeEach(() => {
    subscriptionCallbacks = {};

    // Create mock loggers with tracking
    mockLogger = {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      getLogs: vi.fn(() => [
        { timestamp: '2023-01-01T00:00:00.000Z', level: 'info', namespace: 'test', message: 'test message', args: [] }
      ]),
      updateSettings: vi.fn(),
      destroy: vi.fn(),
    };

    mockAgentLogger = {
      ...mockLogger,
      logAgentAction: vi.fn(),
      logWebSocketMessage: vi.fn(),
      logThreeJSAction: vi.fn(),
      logPerformance: vi.fn(),
      getAgentTelemetry: vi.fn(() => []),
      getWebSocketTelemetry: vi.fn(() => []),
      getThreeJSTelemetry: vi.fn(() => []),
      clearTelemetry: vi.fn(),
    };

    // Mock loggerConfig to return our mocks
    const { createDynamicLogger, createDynamicAgentLogger } = require('./loggerConfig');
    createDynamicLogger.mockReturnValue(mockLogger);
    createDynamicAgentLogger.mockReturnValue(mockAgentLogger);

    // Setup subscription capture
    (clientDebugState.subscribe as any).mockImplementation((key: string, callback: Function) => {
      subscriptionCallbacks[key] = callback;
      return () => delete subscriptionCallbacks[key];
    });

    // Setup default values
    (clientDebugState.get as any).mockImplementation((key: string) => {
      switch (key) {
        case 'logLevel': return 'info';
        case 'consoleLogging': return true;
        case 'enabled': return false;
        default: return false;
      }
    });

    (clientDebugState.isEnabled as any).mockReturnValue(false);

    // Clear logger manager
    loggerManager.cleanup();
    vi.clearAllMocks();
  });

  afterEach(() => {
    loggerManager.cleanup();
  });

  describe('Control Center Integration', () => {
    it('should update loggers when Control Center settings change', () => {
      // Create some loggers
      const logger1 = createLogger('logger1');
      const logger2 = createLogger('logger2');
      const agentLogger = createAgentLogger('agent1');

      // Verify subscriptions were set up
      expect(clientDebugState.subscribe).toHaveBeenCalledWith('enabled', expect.any(Function));
      expect(clientDebugState.subscribe).toHaveBeenCalledWith('logLevel', expect.any(Function));
      expect(clientDebugState.subscribe).toHaveBeenCalledWith('consoleLogging', expect.any(Function));

      // Simulate Control Center changing log level
      (clientDebugState.get as any).mockImplementation((key: string) => {
        switch (key) {
          case 'logLevel': return 'debug';
          case 'consoleLogging': return true;
          case 'enabled': return true;
          default: return false;
        }
      });

      // Trigger the log level change callback
      if (subscriptionCallbacks['logLevel']) {
        subscriptionCallbacks['logLevel']('debug');
      }

      // Verify that the underlying system was notified
      const { updateAllLoggers } = require('./loggerConfig');
      expect(updateAllLoggers).toHaveBeenCalled();
    });

    it('should handle master debug switch changes', () => {
      const logger = createLogger('test');

      // Initially disabled
      expect(clientDebugState.isEnabled).toHaveBeenCalled();

      // Enable master debug
      (clientDebugState.isEnabled as any).mockReturnValue(true);
      (clientDebugState.get as any).mockImplementation((key: string) => {
        switch (key) {
          case 'enabled': return true;
          case 'logLevel': return 'debug';
          case 'consoleLogging': return true;
          default: return false;
        }
      });

      // Trigger the enabled change callback
      if (subscriptionCallbacks['enabled']) {
        subscriptionCallbacks['enabled'](true);
      }

      // Verify that settings were updated
      const { updateAllLoggers } = require('./loggerConfig');
      expect(updateAllLoggers).toHaveBeenCalled();
    });

    it('should handle console logging toggle', () => {
      const logger = createLogger('test');

      // Initially enabled
      expect(logger).toBeDefined();

      // Disable console logging
      (clientDebugState.get as any).mockImplementation((key: string) => {
        switch (key) {
          case 'logLevel': return 'info';
          case 'consoleLogging': return false;
          case 'enabled': return true;
          default: return false;
        }
      });

      // Trigger the console logging change callback
      if (subscriptionCallbacks['consoleLogging']) {
        subscriptionCallbacks['consoleLogging'](false);
      }

      // Verify that settings were updated
      const { updateAllLoggers } = require('./loggerConfig');
      expect(updateAllLoggers).toHaveBeenCalled();
    });
  });

  describe('Multiple Logger Synchronization', () => {
    it('should update all logger instances simultaneously', () => {
      // Create multiple loggers
      const loggers = [];
      for (let i = 0; i < 5; i++) {
        loggers.push(createLogger(`logger-${i}`));
      }

      const agentLoggers = [];
      for (let i = 0; i < 3; i++) {
        agentLoggers.push(createAgentLogger(`agent-${i}`));
      }

      expect(loggerManager.getRegisteredLoggers().length).toBe(8);

      // Simulate settings change
      loggerManager.updateAllLoggers();

      // All loggers should be notified of the update
      const { updateAllLoggers } = require('./loggerConfig');
      expect(updateAllLoggers).toHaveBeenCalled();
    });

    it('should handle rapid setting changes without errors', () => {
      const logger = createLogger('test');

      // Simulate rapid changes
      const changes = [
        { key: 'logLevel', value: 'debug' },
        { key: 'logLevel', value: 'info' },
        { key: 'consoleLogging', value: false },
        { key: 'consoleLogging', value: true },
        { key: 'enabled', value: true },
        { key: 'enabled', value: false },
      ];

      changes.forEach(({ key, value }) => {
        if (subscriptionCallbacks[key]) {
          subscriptionCallbacks[key](value);
        }
      });

      // Should not throw errors
      expect(logger).toBeDefined();
    });
  });

  describe('Memory Management', () => {
    it('should clean up subscriptions when loggers are destroyed', () => {
      const unsubscribeMocks = [vi.fn(), vi.fn(), vi.fn()];
      let mockIndex = 0;

      (clientDebugState.subscribe as any).mockImplementation(() => {
        return unsubscribeMocks[mockIndex++];
      });

      // Create a logger (this should set up 3 subscriptions)
      createLogger('test');

      // Cleanup should call all unsubscribe functions
      loggerManager.cleanup();

      unsubscribeMocks.forEach(unsubscribe => {
        expect(unsubscribe).toHaveBeenCalled();
      });
    });

    it('should handle many logger instances efficiently', () => {
      const loggerCount = 100;
      const startTime = performance.now();

      // Create many loggers
      for (let i = 0; i < loggerCount; i++) {
        createLogger(`logger-${i}`);
      }

      const creationTime = performance.now() - startTime;

      // Should create loggers reasonably quickly (less than 100ms for 100 loggers)
      expect(creationTime).toBeLessThan(100);
      expect(loggerManager.getRegisteredLoggers().length).toBe(loggerCount);

      // Cleanup should also be fast
      const cleanupStartTime = performance.now();
      loggerManager.cleanup();
      const cleanupTime = performance.now() - cleanupStartTime;

      expect(cleanupTime).toBeLessThan(50);
      expect(loggerManager.getRegisteredLoggers().length).toBe(0);
    });

    it('should handle cleanup errors gracefully', () => {
      // Create a logger that throws during destroy
      const faultyLogger = {
        ...mockLogger,
        destroy: vi.fn(() => { throw new Error('Destroy failed'); }),
      };

      const { createDynamicLogger } = require('./loggerConfig');
      createDynamicLogger.mockReturnValueOnce(faultyLogger);

      createLogger('faulty');

      // Cleanup should not throw even if individual logger cleanup fails
      expect(() => loggerManager.cleanup()).not.toThrow();
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle initial state with no settings', () => {
      // Mock debug state to return undefined
      (clientDebugState.get as any).mockReturnValue(undefined);
      (clientDebugState.isEnabled as any).mockReturnValue(false);

      // Should not throw error
      expect(() => createLogger('test')).not.toThrow();

      const logger = createLogger('test');
      expect(logger).toBeDefined();
    });

    it('should handle invalid log levels gracefully', () => {
      (clientDebugState.get as any).mockImplementation((key: string) => {
        if (key === 'logLevel') return 'invalid-level';
        return true;
      });

      // Should not throw error
      expect(() => createLogger('test')).not.toThrow();
    });

    it('should handle subscription failures gracefully', () => {
      (clientDebugState.subscribe as any).mockImplementation(() => {
        throw new Error('Subscription failed');
      });

      // Should not throw error and should still create logger
      expect(() => createLogger('test')).not.toThrow();

      const logger = createLogger('test');
      expect(logger).toBeDefined();
    });

    it('should handle logger creation during setting updates', () => {
      let createdDuringCallback = false;

      (clientDebugState.subscribe as any).mockImplementation((key: string, callback: Function) => {
        subscriptionCallbacks[key] = () => {
          // Create a logger during the callback
          if (!createdDuringCallback) {
            createdDuringCallback = true;
            createLogger('created-during-callback');
          }
          callback();
        };
        return () => {};
      });

      createLogger('initial');

      // Trigger a settings change
      if (subscriptionCallbacks['logLevel']) {
        subscriptionCallbacks['logLevel']('debug');
      }

      // Should handle this gracefully
      expect(loggerManager.getRegisteredLoggers().length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Performance Monitoring', () => {
    it('should track logger metrics accurately', () => {
      const logger = createLogger('test', { category: 'performance' });

      // Simulate some logging activity by calling the wrapped logger methods
      logger.info('test message 1');
      logger.warn('test message 2');
      logger.error('test message 3');

      const metrics = loggerManager.getLoggerMetrics('test');
      expect(metrics.length).toBe(1);

      const metric = metrics[0];
      expect(metric.namespace).toBe('test');
      expect(metric.category).toBe('performance');
      expect(metric).toHaveProperty('createdAt');
      expect(metric).toHaveProperty('lastUsed');
      expect(metric).toHaveProperty('messageCount');
      expect(metric).toHaveProperty('errorCount');
      expect(metric).toHaveProperty('isActive');
    });

    it('should provide system-wide settings and metrics', () => {
      createLogger('ui-logger', { category: 'ui' });
      createLogger('api-logger', { category: 'api' });
      createAgentLogger('physics-agent', { category: 'physics' });

      const settings = loggerManager.getCurrentSettings();

      expect(settings).toHaveProperty('enabled');
      expect(settings).toHaveProperty('consoleLogging');
      expect(settings).toHaveProperty('logLevel');
      expect(settings).toHaveProperty('registeredCount', 3);
      expect(settings).toHaveProperty('categoryCounts');

      expect(settings.categoryCounts.ui).toBe(1);
      expect(settings.categoryCounts.api).toBe(1);
      expect(settings.categoryCounts.physics).toBe(1);
    });

    it('should filter loggers by category', () => {
      createLogger('ui1', { category: 'ui' });
      createLogger('ui2', { category: 'ui' });
      createLogger('api1', { category: 'api' });

      const uiLoggers = loggerManager.getLoggersByCategory('ui');
      expect(uiLoggers.length).toBe(2);

      const apiLoggers = loggerManager.getLoggersByCategory('api');
      expect(apiLoggers.length).toBe(1);

      const nonExistentLoggers = loggerManager.getLoggersByCategory('nonexistent');
      expect(nonExistentLoggers.length).toBe(0);
    });
  });

  describe('Environment Variable Fallback', () => {
    it('should fall back to environment variables when debug state fails', () => {
      // Mock debug state to throw errors
      (clientDebugState.get as any).mockImplementation(() => {
        throw new Error('Debug state unavailable');
      });
      (clientDebugState.isEnabled as any).mockImplementation(() => {
        throw new Error('Debug state unavailable');
      });

      // Mock environment variables
      vi.stubGlobal('import.meta', {
        env: {
          VITE_LOG_LEVEL: 'warn',
          DEV: false,
        }
      });

      // Should still create logger without throwing
      expect(() => createLogger('test')).not.toThrow();

      const logger = createLogger('test');
      expect(logger).toBeDefined();
    });

    it('should prefer debug state over environment variables when available', () => {
      // Set up environment variables
      vi.stubGlobal('import.meta', {
        env: {
          VITE_LOG_LEVEL: 'error',
          DEV: false,
        }
      });

      // Set up debug state with different value
      (clientDebugState.get as any).mockImplementation((key: string) => {
        if (key === 'logLevel') return 'debug';
        if (key === 'consoleLogging') return true;
        return false;
      });

      const logger = createLogger('test');
      expect(logger).toBeDefined();

      // Should respect debug state over environment
      const settings = loggerManager.getCurrentSettings();
      expect(settings.logLevel).toBe('debug');
    });

    it('should handle missing environment variables gracefully', () => {
      vi.stubGlobal('import.meta', {
        env: {}
      });

      (clientDebugState.get as any).mockReturnValue(undefined);

      // Should not throw error
      expect(() => createLogger('test')).not.toThrow();
    });
  });
});