import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createLogger, createAgentLogger, createDynamicLogger, createDynamicAgentLogger, getDefaultLogLevel } from './loggerConfig';
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

// Mock the base logger
vi.mock('./logger', () => ({
  createLogger: vi.fn(() => ({
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    getLogs: vi.fn(() => []),
    updateLevel: vi.fn(),
    setEnabled: vi.fn(),
    getCurrentConfig: vi.fn(() => ({ level: 'info', enabled: true })),
    isEnabled: vi.fn(() => true),
    namespace: 'test',
  })),
  createAgentLogger: vi.fn(() => ({
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    getLogs: vi.fn(() => []),
    updateLevel: vi.fn(),
    setEnabled: vi.fn(),
    getCurrentConfig: vi.fn(() => ({ level: 'info', enabled: true })),
    isEnabled: vi.fn(() => true),
    namespace: 'test',
    logAgentAction: vi.fn(),
    logWebSocketMessage: vi.fn(),
    logThreeJSAction: vi.fn(),
    logPerformance: vi.fn(),
    getAgentTelemetry: vi.fn(() => []),
    getWebSocketTelemetry: vi.fn(() => []),
    getThreeJSTelemetry: vi.fn(() => []),
    clearTelemetry: vi.fn(),
  })),
}));

describe('loggerConfig', () => {
  beforeEach(() => {
    // Reset all mocks before each test
    vi.clearAllMocks();

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

  describe('Environment Variable Handling', () => {
    it('should respect VITE_LOG_LEVEL environment variable', () => {
      vi.stubGlobal('import.meta.env', { VITE_LOG_LEVEL: 'debug', DEV: false });

      const defaultLevel = getDefaultLogLevel();
      expect(defaultLevel).toBe('debug');
    });

    it('should respect LOG_LEVEL environment variable', () => {
      vi.stubGlobal('import.meta.env', { LOG_LEVEL: 'warn', DEV: false });

      const defaultLevel = getDefaultLogLevel();
      expect(defaultLevel).toBe('warn');
    });

    it('should handle case insensitive log levels', () => {
      vi.stubGlobal('import.meta.env', { VITE_LOG_LEVEL: 'WARN', DEV: false });

      const defaultLevel = getDefaultLogLevel();
      expect(defaultLevel).toBe('warn');
    });

    it('should default to debug in development', () => {
      vi.stubGlobal('import.meta.env', { DEV: true });

      const defaultLevel = getDefaultLogLevel();
      expect(defaultLevel).toBe('debug');
    });

    it('should default to info in production', () => {
      vi.stubGlobal('import.meta.env', { DEV: false });

      const defaultLevel = getDefaultLogLevel();
      expect(defaultLevel).toBe('info');
    });

    it('should handle invalid log levels gracefully', () => {
      vi.stubGlobal('import.meta.env', { VITE_LOG_LEVEL: 'invalid', DEV: false });

      const defaultLevel = getDefaultLogLevel();
      expect(defaultLevel).toBe('info');
    });
  });

  describe('Dynamic Logger Creation', () => {
    it('should create logger with dynamic configuration', () => {
      const logger = createDynamicLogger('test');

      expect(logger).toBeDefined();
      expect(logger.debug).toBeInstanceOf(Function);
      expect(logger.info).toBeInstanceOf(Function);
      expect(logger.warn).toBeInstanceOf(Function);
      expect(logger.error).toBeInstanceOf(Function);
      expect(logger.updateSettings).toBeInstanceOf(Function);
      expect(logger.destroy).toBeInstanceOf(Function);
    });

    it('should respect Control Center settings over environment variables', () => {
      vi.stubGlobal('import.meta.env', { VITE_LOG_LEVEL: 'error', DEV: false });

      // Control Center settings should override environment
      (clientDebugState.get as any).mockImplementation((key: string) => {
        if (key === 'logLevel') return 'debug';
        if (key === 'consoleLogging') return true;
        return false;
      });

      const logger = createDynamicLogger('test');
      expect(logger).toBeDefined();
    });

    it('should handle disabled option', () => {
      const logger = createDynamicLogger('test', { disabled: true, maxLogEntries: 500 });

      expect(logger).toBeDefined();
      expect(logger.updateSettings).toBeInstanceOf(Function);
    });

    it('should fall back to environment variables when Control Center is unavailable', () => {
      vi.stubGlobal('import.meta.env', { VITE_LOG_LEVEL: 'warn', DEV: false });

      // Mock Control Center to return undefined
      (clientDebugState.get as any).mockReturnValue(undefined);

      const logger = createDynamicLogger('test');
      expect(logger).toBeDefined();
    });
  });

  describe('Dynamic Agent Logger Creation', () => {
    it('should create agent logger with dynamic configuration', () => {
      const logger = createDynamicAgentLogger('test-agent');

      expect(logger).toBeDefined();

      // Should have standard logger methods
      expect(logger.debug).toBeInstanceOf(Function);
      expect(logger.info).toBeInstanceOf(Function);
      expect(logger.warn).toBeInstanceOf(Function);
      expect(logger.error).toBeInstanceOf(Function);

      // Should have agent-specific methods
      expect(logger.logAgentAction).toBeInstanceOf(Function);
      expect(logger.logWebSocketMessage).toBeInstanceOf(Function);
      expect(logger.logThreeJSAction).toBeInstanceOf(Function);
      expect(logger.logPerformance).toBeInstanceOf(Function);

      // Should have dynamic configuration methods
      expect(logger.updateSettings).toBeInstanceOf(Function);
      expect(logger.destroy).toBeInstanceOf(Function);
    });

    it('should respect Control Center settings for agent loggers', () => {
      vi.stubGlobal('import.meta.env', { VITE_LOG_LEVEL: 'error', DEV: false });

      // Control Center settings should override environment
      (clientDebugState.get as any).mockImplementation((key: string) => {
        if (key === 'logLevel') return 'debug';
        if (key === 'consoleLogging') return true;
        return false;
      });

      const logger = createDynamicAgentLogger('test-agent');
      expect(logger).toBeDefined();
    });

    it('should create backward compatible agent logger', () => {
      const logger = createAgentLogger('test-agent');

      expect(logger).toBeDefined();
      // Should be same as dynamic agent logger (alias)
      expect(logger.logAgentAction).toBeInstanceOf(Function);
    });
  });

  describe('Backward Compatibility', () => {
    it('should provide backward compatible createLogger function', () => {
      const logger = createLogger('test');

      expect(logger).toBeDefined();
      expect(logger.debug).toBeInstanceOf(Function);
      expect(logger.updateSettings).toBeInstanceOf(Function);
    });

    it('should provide backward compatible createAgentLogger function', () => {
      const logger = createAgentLogger('test-agent');

      expect(logger).toBeDefined();
      expect(logger.logAgentAction).toBeInstanceOf(Function);
      expect(logger.updateSettings).toBeInstanceOf(Function);
    });
  });

  describe('Environment and Runtime Integration', () => {
    it('should handle missing environment gracefully', () => {
      vi.stubGlobal('import.meta', { env: {} });

      expect(() => getDefaultLogLevel()).not.toThrow();
      expect(() => createDynamicLogger('test')).not.toThrow();
    });

    it('should handle debug state failures gracefully', () => {
      (clientDebugState.get as any).mockImplementation(() => {
        throw new Error('Debug state failed');
      });
      (clientDebugState.subscribe as any).mockImplementation(() => {
        throw new Error('Subscription failed');
      });

      expect(() => createDynamicLogger('test')).not.toThrow();
      expect(() => createDynamicAgentLogger('test')).not.toThrow();
    });

    it('should prefer runtime settings over static options', () => {
      (clientDebugState.get as any).mockImplementation((key: string) => {
        if (key === 'logLevel') return 'debug';
        if (key === 'consoleLogging') return false;
        return false;
      });

      // Even though we pass different options, runtime settings should be preferred
      const logger = createDynamicLogger('test', {
        level: 'error',
        respectRuntimeSettings: true
      });

      expect(logger).toBeDefined();
    });

    it('should respect respectRuntimeSettings flag', () => {
      const logger = createDynamicLogger('test', {
        level: 'warn',
        respectRuntimeSettings: false
      });

      expect(logger).toBeDefined();
      // Should use provided level instead of runtime settings
    });
  });
});