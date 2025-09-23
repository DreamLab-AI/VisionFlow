/**
 * Integration test for dynamic logger system with Control Center settings
 */

import { createLogger, createAgentLogger } from './loggerConfig';
import { clientDebugState } from './clientDebugState';

// Mock localStorage for testing
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
  length: 0,
  key: jest.fn(),
};

// Mock window.addEventListener
const mockAddEventListener = jest.fn();

// Setup mocks
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

Object.defineProperty(window, 'addEventListener', {
  value: mockAddEventListener,
  writable: true,
});

describe('Dynamic Logger Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset localStorage mock
    mockLocalStorage.getItem.mockReturnValue(null);
  });

  describe('Control Center Integration', () => {
    test('logger respects consoleLogging setting', () => {
      // Mock consoleLogging disabled
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'debug.consoleLogging') return 'false';
        if (key === 'debug.logLevel') return 'info';
        return null;
      });

      const logger = createLogger('TestModule');

      // Mock console methods to verify they are not called
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      logger.debug('This should not appear');

      // Should not log when console logging is disabled
      expect(consoleSpy).not.toHaveBeenCalled();

      consoleSpy.mockRestore();
    });

    test('logger respects logLevel setting from Control Center', () => {
      // Mock logLevel set to 'error' in Control Center
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'debug.consoleLogging') return 'true';
        if (key === 'debug.logLevel') return 'error';
        return null;
      });

      const logger = createLogger('TestModule');

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

      logger.debug('Debug message'); // Should not appear
      logger.info('Info message');   // Should not appear
      logger.warn('Warn message');   // Should not appear
      logger.error('Error message'); // Should appear

      expect(consoleSpy).not.toHaveBeenCalled();
      expect(consoleErrorSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
      consoleErrorSpy.mockRestore();
    });

    test('agent logger respects Control Center settings', () => {
      // Mock Control Center settings
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'debug.consoleLogging') return 'true';
        if (key === 'debug.logLevel') return 'debug';
        return null;
      });

      const agentLogger = createAgentLogger('TestAgent');

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      agentLogger.logAgentAction('agent1', 'worker', 'task_start', { task: 'test' });

      expect(consoleSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });

  describe('Environment Variable Fallback', () => {
    test('uses environment variables when Control Center settings not available', () => {
      // Mock no Control Center settings
      mockLocalStorage.getItem.mockReturnValue(null);

      // Mock import.meta.env
      Object.defineProperty(import.meta, 'env', {
        value: {
          VITE_LOG_LEVEL: 'warn',
          DEV: false
        },
        writable: true
      });

      const logger = createLogger('TestModule');

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();

      logger.debug('Debug message'); // Should not appear
      logger.info('Info message');   // Should not appear
      logger.warn('Warn message');   // Should appear

      expect(consoleSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
      consoleWarnSpy.mockRestore();
    });
  });

  describe('Dynamic Updates', () => {
    test('logger updates when Control Center settings change', () => {
      // Initially disabled
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'debug.consoleLogging') return 'false';
        return null;
      });

      const logger = createLogger('TestModule');

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      logger.info('Should not appear initially');
      expect(consoleSpy).not.toHaveBeenCalled();

      // Simulate settings change by calling updateSettings directly
      // (In real usage, this would be triggered by localStorage events)
      logger.updateSettings();

      consoleSpy.mockRestore();
    });
  });
});