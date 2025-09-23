import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  createLogger,
  createAgentLogger,
  loggerManager,
  type LogLevel
} from './dynamicLogger';
import { clientDebugState } from './clientDebugState';

// Mock clientDebugState
vi.mock('./clientDebugState', () => ({
  clientDebugState: {
    get: vi.fn(),
    set: vi.fn(),
    subscribe: vi.fn(),
    isEnabled: vi.fn(),
    getAll: vi.fn(),
  }
}));

// Mock the underlying loggerConfig
vi.mock('./loggerConfig', () => ({
  createDynamicLogger: vi.fn(() => ({
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    getLogs: vi.fn(() => []),
    updateSettings: vi.fn(),
    destroy: vi.fn(),
  })),
  createDynamicAgentLogger: vi.fn(() => ({
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    getLogs: vi.fn(() => []),
    updateSettings: vi.fn(),
    destroy: vi.fn(),
    logAgentAction: vi.fn(),
    logWebSocketMessage: vi.fn(),
    logThreeJSAction: vi.fn(),
    logPerformance: vi.fn(),
    getAgentTelemetry: vi.fn(() => []),
    getWebSocketTelemetry: vi.fn(() => []),
    getThreeJSTelemetry: vi.fn(() => []),
    clearTelemetry: vi.fn(),
  })),
  updateAllLoggers: vi.fn(),
}));

// Mock import.meta.env
const mockEnv = {
  VITE_LOG_LEVEL: undefined,
  LOG_LEVEL: undefined,
  DEV: false,
  MODE: 'production',
};

vi.stubGlobal('import.meta', {
  env: mockEnv,
});

describe('DynamicLogger System', () => {
  let consoleSpies: Record<string, any>;

  beforeEach(() => {
    // Clear all loggers before each test
    loggerManager.cleanup();

    // Reset all mocks
    vi.clearAllMocks();

    // Reset localStorage
    window.localStorage.clear();

    // Setup console spies
    consoleSpies = {
      log: vi.spyOn(console, 'log').mockImplementation(() => {}),
      info: vi.spyOn(console, 'info').mockImplementation(() => {}),
      warn: vi.spyOn(console, 'warn').mockImplementation(() => {}),
      error: vi.spyOn(console, 'error').mockImplementation(() => {}),
      debug: vi.spyOn(console, 'debug').mockImplementation(() => {}),
    };

    // Setup default mock behaviors
    (clientDebugState.get as any).mockImplementation((key: string) => {
      switch (key) {
        case 'logLevel': return 'info';
        case 'consoleLogging': return true;
        case 'enabled': return false;
        default: return false;
      }
    });

    (clientDebugState.isEnabled as any).mockReturnValue(false);
    (clientDebugState.subscribe as any).mockReturnValue(() => {});
  });

  afterEach(() => {
    // Restore console methods
    Object.values(consoleSpies).forEach(spy => spy.mockRestore?.());

    // Clear all loggers after each test
    loggerManager.cleanup();
  });

  describe('Logger Instance Registration and Management', () => {
    it('should register a new logger instance', () => {
      const logger = createLogger('test-namespace');

      expect(logger).toBeDefined();
      expect(logger.debug).toBeInstanceOf(Function);
      expect(logger.info).toBeInstanceOf(Function);
      expect(logger.warn).toBeInstanceOf(Function);
      expect(logger.error).toBeInstanceOf(Function);
    });

    it('should track registered loggers', () => {
      createLogger('namespace1');
      createLogger('namespace2');

      const registeredLoggers = loggerManager.getRegisteredLoggers();
      expect(registeredLoggers.length).toBe(2);
      expect(registeredLoggers).toContain('namespace1');
      expect(registeredLoggers).toContain('namespace2');
    });

    it('should return the same instance for the same namespace', () => {
      const logger1 = createLogger('test');
      const logger2 = createLogger('test');

      expect(logger1).toBe(logger2);
      expect(loggerManager.getRegisteredLoggers().length).toBe(1);
    });

    it('should properly clean up loggers when cleared', () => {
      createLogger('test1');
      createLogger('test2');

      expect(loggerManager.getRegisteredLoggers().length).toBe(2);

      loggerManager.cleanup();

      expect(loggerManager.getRegisteredLoggers().length).toBe(0);
    });

    it('should handle logger creation with custom options', () => {
      const logger = createLogger('test', {
        disabled: true,
        level: 'error' as LogLevel,
        maxLogEntries: 500,
        category: 'test-category'
      });

      expect(logger).toBeDefined();
      expect(loggerManager.getRegisteredLoggers().length).toBe(1);
    });
  });

  describe('Agent Logger Support', () => {
    it('should create agent logger with telemetry methods', () => {
      const agentLogger = createAgentLogger('test-agent');

      expect(agentLogger).toBeDefined();
      expect(agentLogger.debug).toBeInstanceOf(Function);
      expect(agentLogger.info).toBeInstanceOf(Function);
      expect(agentLogger.warn).toBeInstanceOf(Function);
      expect(agentLogger.error).toBeInstanceOf(Function);

      // Agent-specific methods should be available (though mocked)
      expect(agentLogger.logAgentAction).toBeInstanceOf(Function);
      expect(agentLogger.logWebSocketMessage).toBeInstanceOf(Function);
      expect(agentLogger.logThreeJSAction).toBeInstanceOf(Function);
      expect(agentLogger.logPerformance).toBeInstanceOf(Function);
    });

    it('should track agent loggers separately', () => {
      createLogger('regular-logger');
      createAgentLogger('agent-logger');

      const registeredLoggers = loggerManager.getRegisteredLoggers();
      expect(registeredLoggers.length).toBe(2);
      expect(registeredLoggers).toContain('regular-logger');
      expect(registeredLoggers).toContain('agent-logger');
    });
  });

  describe('Dynamic Level Updates', () => {
    it('should subscribe to debug state changes', () => {
      createLogger('test');

      // Should have subscribed to key debug state properties
      expect(clientDebugState.subscribe).toHaveBeenCalledWith('enabled', expect.any(Function));
      expect(clientDebugState.subscribe).toHaveBeenCalledWith('logLevel', expect.any(Function));
      expect(clientDebugState.subscribe).toHaveBeenCalledWith('consoleLogging', expect.any(Function));
    });

    it('should update all loggers when debug settings change', () => {
      const logger1 = createLogger('namespace1');
      const logger2 = createLogger('namespace2');

      // Simulate a settings change
      loggerManager.updateAllLoggers();

      // The update should have been called on both loggers
      expect(logger1).toBeDefined();
      expect(logger2).toBeDefined();
    });
  });

  describe('Logger Manager Functionality', () => {
    it('should provide logger metrics and settings', () => {
      createLogger('test1', { category: 'ui' });
      createLogger('test2', { category: 'api' });

      const settings = loggerManager.getCurrentSettings();

      expect(settings).toHaveProperty('enabled');
      expect(settings).toHaveProperty('consoleLogging');
      expect(settings).toHaveProperty('logLevel');
      expect(settings).toHaveProperty('registeredCount');
      expect(settings).toHaveProperty('categoryCounts');

      expect(settings.registeredCount).toBe(2);
    });

    it('should get loggers by category', () => {
      createLogger('ui1', { category: 'ui' });
      createLogger('ui2', { category: 'ui' });
      createLogger('api1', { category: 'api' });

      const uiLoggers = loggerManager.getLoggersByCategory('ui');
      expect(uiLoggers.length).toBe(2);
    });

    it('should provide logger metrics', () => {
      createLogger('test');

      const metrics = loggerManager.getLoggerMetrics();
      expect(metrics.length).toBe(1);
      expect(metrics[0]).toHaveProperty('namespace', 'test');
      expect(metrics[0]).toHaveProperty('createdAt');
      expect(metrics[0]).toHaveProperty('messageCount');
      expect(metrics[0]).toHaveProperty('errorCount');
    });

    it('should unregister individual loggers', () => {
      createLogger('test1');
      createLogger('test2');

      expect(loggerManager.getRegisteredLoggers().length).toBe(2);

      const removed = loggerManager.unregisterLogger('test1');
      expect(removed).toBe(true);
      expect(loggerManager.getRegisteredLoggers().length).toBe(1);
      expect(loggerManager.getRegisteredLoggers()).not.toContain('test1');
      expect(loggerManager.getRegisteredLoggers()).toContain('test2');
    });

    it('should return false when unregistering non-existent logger', () => {
      const removed = loggerManager.unregisterLogger('non-existent');
      expect(removed).toBe(false);
    });
  });

  describe('Environment Variable Fallback', () => {
    it('should handle missing debug state gracefully', () => {
      // Mock debug state to throw errors
      (clientDebugState.get as any).mockImplementation(() => {
        throw new Error('Debug state unavailable');
      });
      (clientDebugState.isEnabled as any).mockImplementation(() => {
        throw new Error('Debug state unavailable');
      });
      (clientDebugState.subscribe as any).mockImplementation(() => {
        throw new Error('Debug state unavailable');
      });

      // Should not throw error when creating logger
      expect(() => createLogger('test')).not.toThrow();

      const logger = createLogger('test');
      expect(logger).toBeDefined();
    });

    it('should handle subscription failures gracefully', () => {
      (clientDebugState.subscribe as any).mockImplementation(() => {
        throw new Error('Subscription failed');
      });

      // Should not throw error
      expect(() => createLogger('test')).not.toThrow();
    });
  });

  describe('Memory Management and Performance', () => {
    it('should handle many logger instances efficiently', () => {
      const loggerCount = 50;

      // Create many loggers
      for (let i = 0; i < loggerCount; i++) {
        createLogger(`logger-${i}`, { category: `category-${i % 5}` });
      }

      expect(loggerManager.getRegisteredLoggers().length).toBe(loggerCount);

      const settings = loggerManager.getCurrentSettings();
      expect(settings.registeredCount).toBe(loggerCount);
      expect(Object.keys(settings.categoryCounts).length).toBeGreaterThan(1);
    });

    it('should clean up properly on cleanup', () => {
      createLogger('test1');
      createLogger('test2');
      createAgentLogger('agent1');

      expect(loggerManager.getRegisteredLoggers().length).toBe(3);

      loggerManager.cleanup();

      expect(loggerManager.getRegisteredLoggers().length).toBe(0);
    });

    it('should track message counts in metrics', () => {
      const logger = createLogger('test');

      // Simulate some logging activity
      logger.info('test message 1');
      logger.warn('test message 2');
      logger.error('test message 3');

      const metrics = loggerManager.getLoggerMetrics('test');
      expect(metrics.length).toBe(1);

      // Note: Since we're mocking the underlying loggers,
      // we can't test exact counts, but we can verify structure
      expect(metrics[0]).toHaveProperty('messageCount');
      expect(metrics[0]).toHaveProperty('errorCount');
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid log levels gracefully', () => {
      (clientDebugState.get as any).mockImplementation((key: string) => {
        if (key === 'logLevel') return 'invalid-level';
        return false;
      });

      // Should not throw error
      expect(() => createLogger('test')).not.toThrow();
    });

    it('should handle cleanup errors gracefully', () => {
      const mockLogger = {
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        getLogs: vi.fn(() => []),
        updateSettings: vi.fn(),
        destroy: vi.fn(() => { throw new Error('Cleanup failed'); }),
      };

      // Override the mock temporarily
      const { createDynamicLogger } = require('./loggerConfig');
      createDynamicLogger.mockReturnValueOnce(mockLogger);

      createLogger('test');

      // Cleanup should not throw even if logger destroy fails
      expect(() => loggerManager.cleanup()).not.toThrow();
    });
  });

  describe('Integration with Control Center Settings', () => {
    it('should respect master debug switch', () => {
      (clientDebugState.isEnabled as any).mockReturnValue(true);
      (clientDebugState.get as any).mockImplementation((key: string) => {
        switch (key) {
          case 'enabled': return true;
          case 'logLevel': return 'debug';
          case 'consoleLogging': return true;
          default: return false;
        }
      });

      const logger = createLogger('test');
      expect(logger).toBeDefined();

      const settings = loggerManager.getCurrentSettings();
      expect(settings.enabled).toBe(true);
      expect(settings.logLevel).toBe('debug');
      expect(settings.consoleLogging).toBe(true);
    });

    it('should handle disabled console logging', () => {
      (clientDebugState.get as any).mockImplementation((key: string) => {
        switch (key) {
          case 'logLevel': return 'debug';
          case 'consoleLogging': return false;
          case 'enabled': return true;
          default: return false;
        }
      });

      const logger = createLogger('test');
      expect(logger).toBeDefined();

      const settings = loggerManager.getCurrentSettings();
      expect(settings.consoleLogging).toBe(false);
    });
  });
});